"""Bounded parallel decisions, consumed in date order before the next chunk.
Signal prices and activity metadata are independent of marking prices and scenarios.
"""
function backtest_v1(b::Bars; from::Date,S=300,seed=1,ridge_alpha=nothing,F_folds=3,
                     date_tasks=Threads.nthreads(),blas_threads=1,chunk_size=4date_tasks,
                     adaptive=false,quadrature_tol=1e-5,max_scenarios=512)
    T,N=size(b.adj)
    i0=findfirst(>=(from),b.dates)
    (i0 === nothing || i0==T) && error("no decision day on/after $from")
    date_tasks>0 && blas_threads>0 && chunk_size>0 || throw(ArgumentError("invalid execution topology"))
    ds=i0:T-1; K=length(ds)
    signal=signal_prices(b)
    cache=PriceHistoryCache(signal)
    ruler_stats=build_prefix_ruler_stats(cache.log_prices,cache.first_price)
    W=zeros(K,N); We=zeros(K,N); ret=zeros(K); ew=zeros(K)
    timings=zeros(K,length(TIMING_BUCKETS)); scenario_counts=zeros(Int,K)
    h=nothing; he=nothing; locked_days=0; locked_days_ew=0
    warm=[fill((1.0,1.0),F_folds+1) for _ in 1:date_tasks]
    # Each OS thread gets its own FitWorkspace to prevent shared memory race conditions
    n_threads = Threads.maxthreadid()
    workspaces=[FitWorkspace() for _ in 1:n_threads]
    original=BLAS.get_num_threads()
    BLAS.set_num_threads(blas_threads)
    try
        for start in 1:chunk_size:K
            stop=min(K,start+chunk_size-1)
            n_items = stop - start + 1
            results = Vector{Any}(undef, n_items)
            Threads.@threads :greedy for local_index in 1:n_items
                tid = Threads.threadid()
                worker = (tid - 1) % date_tasks + 1
                k = start + local_index - 1; t = ds[k]
                    timing=DecisionTiming()
                    model=fit_v1(view(signal,1:t,:); ridge_alpha,F_folds,ruler_stats,
                                 history_cache=cache,alpha_initial=warm[worker],timing,workspace=workspaces[tid])
                    free=falses(N)
                    for j in model.active_indices
                        free[j]=b.bar[t,j]
                end
                    rng=MersenneTwister(seed+t)
                    if adaptive
                        results[local_index]=(; model,X=nothing,w=nothing,active=model.active_indices,free,timing)
                    else
                        X=timed(timing,:scenario) do
                            generate_scenarios_v1(model; S,rng)
                    end
                        w=timed(timing,:Kelly) do
                            scenario_weights(X,model.active_indices,free)
                    end
                        results[local_index]=(; model=nothing,X,w,active=model.active_indices,free,timing)
                end
            end
            for local_index in eachindex(results)
                k=start+local_index-1; t=ds[k]
                decision=results[local_index]
                free=decision.free; timing=decision.timing
                has_locked=h !== nothing && any(j -> !free[j] && h[j]>0,1:N)
                has_locked && (locked_days+=1)
                he !== nothing && any(j -> !free[j] && he[j]>0,1:N) && (locked_days_ew+=1)
                if adaptive
                    result=timed(timing,:scenario) do
                        adaptive_scenario_weights(decision.model,free,h;
                            rng=MersenneTwister(seed+t),tol=quadrature_tol,max_scenarios)
                    end
                    w=result.weights
                    scenario_counts[k]=result.S
                else
                    w=has_locked ? timed(timing,:Kelly) do
                        scenario_weights(decision.X,decision.active,free,h)
                    end : decision.w
                    scenario_counts[k]=S
                end
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
                results[local_index]=nothing
            end
        end
    finally
        BLAS.set_num_threads(original)
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
