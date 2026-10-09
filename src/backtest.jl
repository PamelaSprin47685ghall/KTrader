import Serialization

# Elapsed monotonic time, not process CPU time. Worker buckets are sums of task
# elapsed times and can overlap each other and the end-to-end wall clock.
const BACKTEST_EXECUTION_BUCKETS = (:prepare, :initialize, :checkpoint,
    :coordinator_advance, :block_advance, :solve, :consume)
mutable struct BacktestExecutionTiming
    seconds::Vector{Float64}
    wall_seconds::Float64
    windows::Int
    blocks::Int
    coordinator_advances::Int
    block_advances::Int
end
BacktestExecutionTiming() = BacktestExecutionTiming(zeros(length(BACKTEST_EXECUTION_BUCKETS)),0.0,0,0,0,0)
function reset_backtest_timing!(trace::BacktestExecutionTiming)
    fill!(trace.seconds,0.0)
    trace.wall_seconds=0.0
    trace.windows=trace.blocks=trace.coordinator_advances=trace.block_advances=0
    trace
end
function backtest_timed(f,trace,bucket)
    trace === nothing && return f()
    started=time_ns()
    try
        f()
    finally
        trace.seconds[findfirst(==(bucket),BACKTEST_EXECUTION_BUCKETS)]+=(time_ns()-started)*1e-9
    end
end

# Balanced contiguous ranges, at most date_tasks blocks over all decision days.
function backtest_timeblocks(n_items::Integer,date_tasks::Integer)
    n_items>0 && date_tasks>0 || throw(ArgumentError("invalid timeblock topology"))
    n_blocks=min(n_items,date_tasks)
    [(div((i-1)*n_items,n_blocks)+1):div(i*n_items,n_blocks) for i in 1:n_blocks]
end

# Per-block slots of the bounded in-memory handoff, used only by adaptive mode.
# chunk_size keeps its role as a cap on outstanding (produced, unconsumed)
# results there: blocks*buffer <= chunk_size+blocks-1, exactly chunk_size when
# blocks divides chunk_size. Every live block keeps at least one slot so the
# strictly date-ordered consumer can always receive the next day it needs:
# bounded progress without deadlock. This cannot give full parallelism, though:
# with a date-ordered consumer, any in-memory buffer of size c lets later blocks
# run at most c days ahead before blocking on put!, so the steady state has one
# active solver. The default (non-adaptive) path therefore spills results to a
# per-date disk spool instead, where producers never block on consumption.
backtest_block_buffer(chunk_size::Integer,blocks::Integer)=max(1,cld(chunk_size,blocks))

# A closed, drained Channel raises InvalidStateException on the deployed
# Julia runtime, not EOFError. Producers publish their failure BEFORE closing
# the channel; drain valid buffered results, then preserve that same failure.
# A normally closed but unexpectedly short stream is a protocol error.
function _backtest_take(channel,failures,block,k)
    try
        take!(channel)
    catch failure
        closed = !isopen(channel) && (failure isa InvalidStateException || failure isa EOFError)
        closed || rethrow()
        cause=failures[block]
        cause===nothing && error("block $block ended before decision $k")
        throw(cause)
    end
end

# Function barrier at the deserialization/channel boundary: the scheduler
# receives `Any`, but the numerical consumer should specialize on the actual
# decision and holdings types rather than the scheduler's captured boxes.
# Same caller-owned timing, seed, scenarios and original Kelly objective.
function _backtest_target(decision,held,adaptive,scenario_seed,S,quadrature_tol,max_scenarios)
    if adaptive
        result=timed(decision.timing,:scenario) do
            adaptive_scenario_weights(decision.model,decision.free,held;
                rng=MersenneTwister(scenario_seed),tol=quadrature_tol,max_scenarios)
        end
        return result.weights,result.S
    end
    weights=timed(decision.timing,:Kelly) do
        scenario_weights(decision.X,decision.active,decision.free,held)
    end
    weights,S
end

"""One continuous schedule of large time blocks, consumed strictly in date order.

The whole decision range is partitioned once into at most date_tasks contiguous
blocks. Each block task owns exactly one initial exact checkpoint
(engine=:incremental: the coordinator advances once to each block head and never
solves) or one warm-started workspace (engine=:batch), then advances its own
dates sequentially across the entire backtest: no per-chunk coordinator/core
cloning and no warm-start resets at chunk boundaries. seed+t, signal prices,
tradability and marking returns are unchanged, so no future bar can reach an
earlier decision.

Default (adaptive=false) results - scenario matrix, active set, tradability,
decision timing - flow through a per-date disk spool owned by this backtest: at
most K files (one per decision day, exactly one writer each), each about S*N*8
bytes, written once and deleted by the consumer on read-back; the whole spool
directory is removed in the finally after every worker has joined, including on
failure paths. Memory holds at most the one result being consumed - far below
the chunk_size bound - and block tasks never block on consumption progress, so
every block solves in parallel across the whole run. chunk_size is retained for
compatibility and no longer bounds this path.

adaptive=true keeps the bounded in-memory channels (chunk_size caps
outstanding results: blocks*buffer <= chunk_size+blocks-1) because serializing
whole adaptive models per day would cost more than it returns. That mode's
parallelism is explicitly limited: with a strictly date-ordered consumer, later
blocks run at most backtest_block_buffer days ahead before blocking, so the
steady state is close to a single active block.

engine=:batch remains the prefix-fit reference. Optional execution_timing
records elapsed time (worker buckets are task-elapsed sums; wall_seconds is the
end-to-end span) without changing the return fields.
"""
function backtest_v1(b::Bars; from::Date,S=300,seed=1,ridge_alpha=nothing,F_folds=3,
                     date_tasks=Threads.nthreads(),blas_threads=1,chunk_size=min(24,4date_tasks),
                     adaptive=false,quadrature_tol=1e-5,max_scenarios=512,
                     engine=:batch,execution_timing=nothing,
                     _spool_parent=nothing)
    all(x -> x isa Integer && x>0,(date_tasks,blas_threads,chunk_size)) ||
        throw(ArgumentError("invalid execution topology"))
    F_folds isa Integer && F_folds>=2 || throw(ArgumentError("F_folds must be at least two"))
    _spool_parent === nothing || _spool_parent isa AbstractString ||
        throw(ArgumentError("_spool_parent must be a directory path or nothing"))
    engine in (:batch,:incremental) || throw(ArgumentError("engine must be :batch or :incremental"))
    if engine === :incremental
        all(name -> isdefined(@__MODULE__,name),
            (:initialize_inference,:advance_exact!,:solve_current!,:inference_checkpoint)) ||
            throw(ArgumentError("incremental inference API is not installed; no silent batch fallback"))
    end
    execution_timing === nothing || reset_backtest_timing!(execution_timing)
    wall_started=time_ns()
    T,N=size(b.adj)
    i0=findfirst(>=(from),b.dates)
    (i0 === nothing || i0==T) && error("no decision day on/after $from")
    ds=i0:T-1; K=length(ds)
    W=zeros(K,N); We=zeros(K,N); ret=zeros(K); ew=zeros(K)
    timings=zeros(K,length(TIMING_BUCKETS)); scenario_counts=zeros(Int,K)
    h=nothing; he=nothing; locked_days=0; locked_days_ew=0
    original=BLAS.get_num_threads()
    # Hoisted for the finally: spool cleanup must run whether the try body
    # succeeded or failed.
    spool_dir=nothing
    try
        BLAS.set_num_threads(blas_threads)
        signal,cache=backtest_timed(execution_timing,:prepare) do
            signal=signal_prices(b)
            # The public cache consumer verifies the ENTIRE ruler prefix on
            # every fit, so building/retaining T×N×|TAUS| entries saves no
            # scan here. Use the existing no-ruler-cache reference instead.
            # Incremental state owns logs/statistics already; its optional
            # initialization history cache would be checked once, then unused.
            cache=engine===:batch ? PriceHistoryCache(signal) : nothing
            signal,cache
        end
        # One fixed partition for the entire backtest: blocks, checkpoints and
        # warm-start chains scale with date_tasks, never with chunk windows.
        ranges=backtest_timeblocks(K,date_tasks)
        blocks=length(ranges)
        block_of=Vector{Int}(undef,K)
        for (block,range) in enumerate(ranges)
            for k in range
                block_of[k]=block
            end
        end
        execution_timing === nothing || begin
            execution_timing.windows=1
            execution_timing.blocks=blocks
        end
        if engine === :incremental
            # This coordinator only advances statistics. It never solves and
            # never sees holdings. Each block receives its single checkpoint at
            # its own head, before any block task exists.
            coordinator=backtest_timed(execution_timing,:initialize) do
                initialize_inference(view(signal,1:i0,:);ridge_alpha,F_folds)
            end
            states=Vector{Any}(undef,blocks)
            coordinator_t=i0
            for block in 1:blocks
                first_t=ds[first(ranges[block])]
                while coordinator_t<first_t
                    coordinator_t+=1
                    backtest_timed(execution_timing,:coordinator_advance) do
                        advance_exact!(coordinator,view(signal,coordinator_t,:))
                    end
                    execution_timing === nothing || (execution_timing.coordinator_advances+=1)
                end
                states[block]=backtest_timed(execution_timing,:checkpoint) do
                    inference_checkpoint(coordinator)
                end
                states[block] === coordinator && error("checkpoint must return an independent state")
            end
    end
        cancel=Threads.Atomic{Bool}(false)
        failures=Vector{Any}(nothing,blocks)
        block_traces=[execution_timing === nothing ? nothing : BacktestExecutionTiming() for _ in 1:blocks]
        # Sole-consumer handoff. adaptive keeps bounded in-memory channels;
        # the default spills each day's result to a one-writer spool file and
        # posts only a signal token, whose channel is sized to the whole block,
        # so producers never block on consumption progress.
        if adaptive
            handoff=[Channel{Any}(backtest_block_buffer(chunk_size,blocks)) for _ in 1:blocks]
        else
            spool_dir=mktempdir(_spool_parent === nothing ? tempdir() : String(_spool_parent);
                                prefix="ktrader_backtest_")
            spool_path(k)=joinpath(spool_dir,"decision_$k.bin")
            handoff=[Channel{Int}(length(ranges[block])) for block in 1:blocks]
        end
        # Each task owns its checkpoint or workspace, warm starts and timing. No
        # mutable buffer is assigned via threadid(); tasks may migrate safely.
        @sync begin
            for block in 1:blocks
                Threads.@spawn begin
                    trace=block_traces[block]
                    try
                        workspace=engine === :batch ? FitWorkspace() : nothing
                        warm=engine === :batch ? fill((1.0,1.0),F_folds+1) : nothing
                        state=engine === :incremental ? states[block] : nothing
                        head=first(ranges[block])
                        for k in ranges[block]
                            cancel[] && break
                            t=ds[k]
                            timing=DecisionTiming()
                            if engine === :incremental && k != head
                                backtest_timed(trace,:block_advance) do
                                    advance_exact!(state,view(signal,t,:))
                                end
                                trace === nothing || (trace.block_advances+=1)
                            end
                            # Both APIs must return a model detached from mutable fit buffers.
                            model=backtest_timed(trace,:solve) do
                                if engine === :incremental
                                    solve_current!(state;timing)
                                else
                                    fit_v1(view(signal,1:t,:);ridge_alpha,F_folds,
                                           history_cache=cache,alpha_initial=warm,timing,workspace)
                                end
                            end
                            active=copy(model.active_indices)
                            free=falses(N)
                            for j in active
                                free[j]=b.bar[t,j]
                            end
                            if adaptive
                                put!(handoff[block],(;model,X=nothing,active,free,timing))
                            else
                                X=timed(timing,:scenario) do
                                    generate_scenarios_v1(model;S,rng=MersenneTwister(seed+t))
                                end
                                # Single-writer per-date file; the token below is
                                # posted only after the write completed, so the
                                # consumer never sees a partial file.
                                open(spool_path(k),"w") do io
                                    Serialization.serialize(io,(;model=nothing,X,active,free,timing))
                                end
                                put!(handoff[block],k)
                            end
                        end
                        close(handoff[block])
                    catch failure
                        # A cancelled run closes channels under running feet:
                        # a blocked put! then wakes with a closed-channel error,
                        # which is the shutdown path, not a reportable failure.
                        cancel[] || (failures[block]=failure)
                        close(handoff[block])
                    end
                end
            end
            # Sole consumer: only this task mutates holdings and the ledger,
            # strictly in date order. seed+t and all decision math are unchanged.
            try
                for k in 1:K
                    block=block_of[k]
                    if adaptive
                        decision=_backtest_take(handoff[block],failures,block,k)
                    else
                        token=_backtest_take(handoff[block],failures,block,k)
                        token==k || error("out-of-order spool signal from block $block: got $token, expected $k")
                        decision=open(spool_path(k),"r") do io
                            Serialization.deserialize(io)
                        end
                    end
                    t=ds[k]
                    backtest_timed(execution_timing,:consume) do
                        free=decision.free; timing=decision.timing
                        has_locked=h !== nothing && any(j -> !free[j] && h[j]>0,1:N)
                        has_locked && (locked_days+=1)
                        he !== nothing && any(j -> !free[j] && he[j]>0,1:N) && (locked_days_ew+=1)
                        w,used_scenarios=_backtest_target(decision,h,adaptive,seed+t,
                            S,quadrature_tol,max_scenarios)
                        scenario_counts[k]=used_scenarios
                        sum(w)>0 || error("no modeled tradable asset for initial portfolio on $(b.dates[t])")
                        we=equal_weights_v1(free,he)
                        gross=b.adj[t+1,:]./b.adj[t,:]
                        gross_clean=ifelse.(isfinite.(gross),gross,1.0)
                        ret[k]=dot(w,gross_clean)-1
                        ew[k]=dot(we,gross_clean)-1
                        W[k,:].=w; We[k,:].=we
                        h=(w.*gross_clean)./(1+ret[k])
                        he=(we.*gross_clean)./(1+ew[k])
                        timings[k,:].=timing.seconds
                    end
                    adaptive || rm(spool_path(k);force=true)
                end
            catch failure
                # Unblocks every producer: closed channels wake blocked put!
                # and the cancel flag stops further work; @sync joins all tasks
                # before this failure propagates. Spool files of unconsumed
                # days are left for the finally's directory removal.
                cancel[]=true
                for channel in handoff
                    close(channel)
                end
                rethrow()
            end
        end
        if execution_timing !== nothing
            for block_trace in block_traces
                block_trace === nothing && continue
                execution_timing.seconds .+= block_trace.seconds
                execution_timing.block_advances+=block_trace.block_advances
            end
        end
    finally
        BLAS.set_num_threads(original)
        spool_dir === nothing || rm(spool_dir;recursive=true,force=true)
        execution_timing === nothing || (execution_timing.wall_seconds=(time_ns()-wall_started)*1e-9)
    end
    wealth(r)=[1.0;cumprod(1.0.+r)]
    (; dates=b.dates[ds],symbols=b.symbols,weights=W,weights_ew=We,ret,ew,locked_days,locked_days_ew,
       wealth=wealth(ret),wealth_ew=wealth(ew),timing_buckets=TIMING_BUCKETS,timings,scenario_counts)
end

function equal_weights_v1(free::AbstractVector{Bool},held)
    N=length(free)
    indices=findall(free)
    current=held === nothing ? zeros(N) : held
    locked=current.*.!free
    budget=1-sum(locked)
    (isempty(indices) || budget<=1e-12) && return copy(current)
    out=copy(locked)
    out[indices].=budget/length(indices)
    out
end

function summarize(r::AbstractVector; periods=252)
    w=[1.0;cumprod(1.0.+r)]
    (; cagr=expm1(mean(log1p.(r))*periods),vol=std(r)*sqrt(periods),
       sharpe=mean(r)/std(r)*sqrt(periods),maxdd=maximum(1.0.-w./accumulate(max,w)),final=w[end])
end
