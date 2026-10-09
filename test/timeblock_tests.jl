# Standalone until the suite owner includes this file in test/runtests.jl.
using Test, Random, LinearAlgebra, Statistics, Dates
using KTrader

function timeblock_equivalent(reference,result)
    @test keys(result)==keys(reference)
    @test result.dates==reference.dates && result.symbols==reference.symbols
    for name in (:ret,:ew,:wealth,:wealth_ew)
        @test isapprox(getproperty(result,name),getproperty(reference,name);atol=1e-8,rtol=1e-8)
    end
    for name in (:weights,:weights_ew)
        @test isapprox(getproperty(result,name),getproperty(reference,name);atol=1e-6,rtol=1e-6)
    end
    for name in (:locked_days,:locked_days_ew,:scenario_counts,:timing_buckets)
        @test getproperty(result,name)==getproperty(reference,name)
    end
    @test all(isfinite,result.timings) && all(result.timings .>= 0)
end

function timeblock_fixture(;flat=false)
    T=344; first_decision=T-9
    P=flat ? ones(T,4) : exp.(cumsum(0.01randn(MersenneTwister(804),T,4);dims=1))
    P[90:93,1].=NaN
    P[first_decision,1]=NaN # Only B is free initially: its holding is certainly nonzero.
    P[first_decision+1:first_decision+3,2].=NaN
    P[1:first_decision+2,3].=NaN # C activates after two consecutive printed prices.
    P[:,4].=NaN
    dates=collect(Date(2020,1,1).+Day.(0:T-1))
    Bars(dates,["A","B","C","Z"],P,P),first_decision
end

@testset "Bounded continuous timeblocks" begin
    @testset "Contiguous partitions and bounded result buffers" begin
        for n in (1,2,7,11), tasks in (1,2,5,20)
            ranges=KTrader.backtest_timeblocks(n,tasks)
            @test length(ranges)==min(n,tasks)
            @test reduce(vcat,collect.(ranges))==collect(1:n)
            @test all(!isempty,ranges)
        end
        @test_throws ArgumentError KTrader.backtest_timeblocks(0,2)
        @test_throws ArgumentError KTrader.backtest_timeblocks(3,0)
        # chunk_size still bounds outstanding results in the adaptive in-memory
        # path: blocks*buffer <= chunk_size+blocks-1 (exactly chunk_size when
        # blocks divides chunk), and every live block keeps at least one slot so
        # the strictly date-ordered consumer can never be starved into deadlock.
        for chunk in (1,2,5,24,30), blocks in (1,2,3,5,8,20)
            buffer=KTrader.backtest_block_buffer(chunk,blocks)
            @test buffer>=1
            @test blocks*buffer<=chunk+blocks-1
        end
        @test KTrader.backtest_block_buffer(24,8)==3
        @test KTrader.backtest_block_buffer(24,3)==8
        @test KTrader.backtest_block_buffer(1,8)==1
    end

    # Missing production APIs are an explicit failure, not a skipped incremental suite.
    @testset "Production engine installed" begin
        for name in (:initialize_inference,:advance_exact!,:solve_current!,:inference_checkpoint)
            @test isdefined(KTrader,name)
        end
    end

    # The default (non-adaptive) path runs through a per-date disk spool, so
    # these equivalence and topology regressions exercise spool write/read-back
    # round-trips on both engines.
    @testset "Same mathematical model across engines, tasks, chunks and tails" begin
        bars,i0=timeblock_fixture()
        from=bars.dates[i0]
        options=(;from,S=64,seed=19,ridge_alpha=2.0,F_folds=2,blas_threads=1)
        reference=backtest_v1(bars;options...,engine=:batch,date_tasks=1,chunk_size=1)
        @test keys(reference)==(:dates,:symbols,:weights,:weights_ew,:ret,:ew,:locked_days,:locked_days_ew,
            :wealth,:wealth_ew,:timing_buckets,:timings,:scenario_counts)
        @test reference.locked_days>=3 && reference.locked_days_ew>=3
        @test reference.weights[1,2] ≈ 1.0 atol=1e-8
        @test all(iszero,reference.weights[:,4])
        @test all(iszero,reference.weights[1:4,3])
        K=length(reference.ret)
        # (2,1) is the tightest backpressure topology; on the spool path the
        # signal channels are sized to whole blocks, so completing without
        # deadlock while never blocking producers is the regression.
        for engine in (:batch,:incremental), (tasks,chunk) in ((1,1),(1,7),(2,3),(3,5),(20,2),(2,30),(2,1))
            trace=KTrader.BacktestExecutionTiming()
            result=backtest_v1(bars;options...,engine,date_tasks=tasks,chunk_size=chunk,execution_timing=trace)
            timeblock_equivalent(reference,result)
            # One continuous schedule: a single ordered consumption stream over
            # at most date_tasks blocks; advances scale with blocks, not chunks.
            blocks=min(K,tasks)
            @test trace.windows==1
            @test trace.blocks==blocks
            ranges=KTrader.backtest_timeblocks(K,tasks)
            # Real checkpoint semantics (line-derived, verified against the
            # observed Evaluated values): initialize consumes 1:i0, then the
            # coordinator advances MONOTONICALLY to the last block's first
            # decision day ds[first(last(ranges))] = i0 + relative - 1, so the
            # exact advance_exact! count is first(last(ranges)) - 1. The old
            # expectation (bare first(last(ranges)), no -1) was the deleted
            # collector's semantics, lost in migration; all 7 incremental cases
            # failed by exactly +1 while all batch cases (expect 0) passed.
            # Boundary pins: single block -> 0 advances; K=10/tasks=2 -> 5.
            @test trace.coordinator_advances==(engine===:incremental ? first(last(ranges))-1 : 0)
            if engine===:incremental && length(ranges)==1
                @test trace.coordinator_advances==0
            end
            @test trace.block_advances==(engine===:incremental ? K-blocks : 0)
            @test all(trace.seconds .>= 0) && trace.wall_seconds>0
            if engine===:batch
                @test trace.coordinator_advances==0
                @test all(iszero,trace.seconds[2:5])
            end
        end
        # Nondefault folds, and EB rather than caller-fixed ridge.
        short=bars.dates[end-2]
        for alpha in (nothing,3.0)
            batch=backtest_v1(bars;from=short,S=32,ridge_alpha=alpha,F_folds=4,engine=:batch,chunk_size=2,date_tasks=1)
            incremental=backtest_v1(bars;from=short,S=32,ridge_alpha=alpha,F_folds=4,engine=:incremental,chunk_size=2,date_tasks=2)
            timeblock_equivalent(batch,incremental)
        end
    end

    @testset "Spool directory is cleaned up on success and on failure" begin
        bars,i0=timeblock_fixture()
        from=bars.dates[i0]
        parent=mktempdir()
        options=(;from,S=32,seed=19,ridge_alpha=2.0,F_folds=2,blas_threads=1)
        for engine in (:batch,:incremental)
            backtest_v1(bars;options...,engine,date_tasks=2,chunk_size=3,_spool_parent=parent)
            @test isempty(readdir(parent))
        end
        original=BLAS.get_num_threads()
        for engine in (:batch,:incremental)
            # Worker failure mid-run: unconsumed spool files must not survive;
            # the finally removes the whole directory after every worker joins.
            @test_throws Exception backtest_v1(bars;from,engine,F_folds=1000,
                blas_threads=original==1 ? 2 : 1,date_tasks=2,chunk_size=3,_spool_parent=parent)
            @test isempty(readdir(parent))
            @test BLAS.get_num_threads()==original
        end
        @test_throws ArgumentError backtest_v1(bars;from,engine=:batch,_spool_parent=17)
        rm(parent;recursive=true,force=true)
    end

    @testset "Checkpoints scale with blocks, never with chunk_size" begin
        bars,i0=timeblock_fixture()
        from=bars.dates[i0]
        K=length(bars.dates)-i0
        for tasks in (1,2,4), chunk in (1,5,30)
            trace=KTrader.BacktestExecutionTiming()
            backtest_v1(bars;from,S=32,seed=23,ridge_alpha=2.0,F_folds=2,
                engine=:incremental,date_tasks=tasks,chunk_size=chunk,
                execution_timing=trace)
            blocks=min(K,tasks)
            # The production trace is the stable non-diagnostic check: the
            # block count depends on blocks only, not on chunk_size.
            @test trace.blocks==blocks
            @test trace.windows==1
            @test trace.block_advances==K-blocks
        end
    end

    @testset "Partial failures still fail loudly without probe counters" begin
        # First decision day fully untradable: the consumer fails on day one.
        # With the probe plumbing removed, the invariant that stays is the
        # failure itself (fail-loud); BLAS restoration is asserted in the
        # invalid-parameters testset below.
        T=344; first_decision=T-9
        P=exp.(cumsum(0.01randn(MersenneTwister(805),T,4);dims=1))
        P[90:93,1].=NaN
        P[1:first_decision+2,3].=NaN
        P[:,4].=NaN
        P[first_decision,:].=NaN
        dates=collect(Date(2020,1,1).+Day.(0:T-1))
        bars=Bars(dates,["A","B","C","Z"],P,P)
        failed=try
            backtest_v1(bars;from=dates[first_decision],S=32,seed=31,ridge_alpha=2.0,F_folds=2,
                engine=:incremental,date_tasks=2,chunk_size=3)
            nothing
        catch e
            e
        end
        @test failed !== nothing
    end

    @testset "Adaptive consumes real locked holdings, not speculative weights" begin
        bars,i0=timeblock_fixture(flat=true)
        options=(;from=bars.dates[i0],adaptive=true,ridge_alpha=2.0,F_folds=2,seed=41)
        reference=backtest_v1(bars;options...,engine=:batch,date_tasks=1,chunk_size=1)
        @test reference.locked_days>=3
        @test all(==(128),reference.scenario_counts)
        # (2,1) keeps the bounded-channel deadlock regression alive for the
        # adaptive in-memory path, which the spool path no longer exercises.
        for engine in (:batch,:incremental), (tasks,chunk) in ((1,5),(2,3),(4,7),(2,1))
            result=backtest_v1(bars;options...,engine,date_tasks=tasks,chunk_size=chunk)
            timeblock_equivalent(reference,result)
        end
    end

    @testset "Production checkpoints and old models own their mutable data" begin
        P=exp.(cumsum(0.01randn(MersenneTwister(88),350,3);dims=1))
        t=338; options=(;ridge_alpha=2.0,F_folds=2)
        coordinator=KTrader.initialize_inference(view(P,1:t,:);options...)
        left=KTrader.inference_checkpoint(coordinator)
        right=KTrader.inference_checkpoint(coordinator)
        @test left !== coordinator && right !== coordinator && left !== right
        # Coordinator advances without solving before checkpoints are handed out.
        for s in t+1:t+3
            KTrader.advance_exact!(coordinator,view(P,s,:))
        end
        KTrader.solve_current!(coordinator)
        past=KTrader.solve_current!(left)
        direct=fit_v1(view(P,1:t,:);options...)
        @test past.mu_pred ≈ direct.mu_pred atol=1e-10 rtol=1e-8
        @test past.res_history ≈ direct.res_history atol=1e-8 rtol=1e-8
        old=deepcopy(past)
        draws=generate_scenarios_v1(past;S=64,rng=MersenneTwister(14))
        states=(left,right); ends=(t+2,t+5); models=Vector{Any}(undef,2)
        # Two workers advancing independent checkpoints of the same prefix in
        # parallel must not disturb each other's models.
        @sync for block in 1:2
            Threads.@spawn begin
                for s in t+1:ends[block]
                    KTrader.advance_exact!(states[block],view(P,s,:))
                end
                models[block]=KTrader.solve_current!(states[block])
            end
        end
        @test isequal(past.mu_pred,old.mu_pred)
        @test isequal(past.res_history,old.res_history)
        @test isequal(past.pred_moments,old.pred_moments)
        @test isequal(generate_scenarios_v1(past;S=64,rng=MersenneTwister(14)),draws)
        for block in 1:2
            expected=fit_v1(view(P,1:ends[block],:);options...)
            @test models[block].mu_pred ≈ expected.mu_pred atol=1e-10 rtol=1e-8
            @test models[block].res_history ≈ expected.res_history atol=1e-8 rtol=1e-8
        end
        # The batch workspace is also reused only within its task; returned models detach.
        workspace=KTrader.FitWorkspace()
        batch_old=fit_v1(view(P,1:t,:);options...,workspace)
        batch_copy=deepcopy(batch_old)
        batch_draws=generate_scenarios_v1(batch_old;S=64,rng=MersenneTwister(15))
        fit_v1(P;options...,workspace)
        @test isequal(batch_old.mu_pred,batch_copy.mu_pred)
        @test isequal(batch_old.res_history,batch_copy.res_history)
        @test isequal(batch_old.pred_moments,batch_copy.pred_moments)
        @test isequal(generate_scenarios_v1(batch_old;S=64,rng=MersenneTwister(15)),batch_draws)
    end

    @testset "Dense ragged history retains the batch mathematical model" begin
        T=WARMUP+320
        P=ones(T,2); P[3:5:T,2].=NaN
        dates=collect(Date(2021,1,1).+Day.(0:T-1))
        bars=Bars(dates,["A","B"],P,P)
        options=(;from=dates[end-3],ridge_alpha=2.0,F_folds=2,S=64,seed=16)
        batch=backtest_v1(bars;options...,engine=:batch,date_tasks=1,chunk_size=1)
        incremental=backtest_v1(bars;options...,engine=:incremental,date_tasks=2,chunk_size=3)
        timeblock_equivalent(batch,incremental)
    end

    @testset "Future bars cannot affect earlier weights" begin
        T=344; P=exp.(cumsum(0.01randn(MersenneTwister(90),T,2);dims=1))
        dates=collect(Date(2020,1,1).+Day.(0:T-1))
        modified=copy(P); modified[T-2:end,:].*=exp.([0.2,-0.3]')
        a=Bars(dates,["A","B"],P,P); b=Bars(dates,["A","B"],modified,modified)
        model=fit_v1(view(P,1:T-3,:);ridge_alpha=2.0,F_folds=2)
        X=generate_scenarios_v1(model;S=64,rng=MersenneTwister(1+T-3))
        expected=KTrader.scenario_weights(X,model.active_indices,trues(2))
        for engine in (:batch,:incremental)
            first=backtest_v1(a;from=dates[T-3],S=64,ridge_alpha=2.0,F_folds=2,engine,date_tasks=2,chunk_size=3)
            changed=backtest_v1(b;from=dates[T-3],S=64,ridge_alpha=2.0,F_folds=2,engine,date_tasks=2,chunk_size=3)
            @test first.weights[1,:] ≈ changed.weights[1,:] atol=1e-6 rtol=1e-6
            @test first.weights[1,:] ≈ expected atol=1e-6 rtol=1e-6
        end
    end

    @testset "Invalid parameters and worker failures restore BLAS" begin
        bars,i0=timeblock_fixture()
        original=BLAS.get_num_threads()
        from=bars.dates[i0]
        @test_throws ArgumentError backtest_v1(bars;from,engine=:unknown)
        @test_throws ArgumentError backtest_v1(bars;from,chunk_size=0)
        @test_throws ArgumentError backtest_v1(bars;from,date_tasks=0)
        @test_throws ArgumentError backtest_v1(bars;from,F_folds=1)
        @test_throws ErrorException backtest_v1(bars;from=bars.dates[end])
        for engine in (:batch,:incremental)
            # Worker failures propagate through the handoff: the failing block
            # task stores the failure and closes its channel, the ordered
            # consumer rethrows it, cancellation unblocks every other producer,
            # @sync joins all tasks, and the finally clause still restores BLAS.
            @test_throws Exception backtest_v1(bars;from,engine,F_folds=1000,
                blas_threads=original==1 ? 2 : 1,date_tasks=2,chunk_size=3)
            @test BLAS.get_num_threads()==original
            recovered=backtest_v1(bars;from=bars.dates[end-1],engine,ridge_alpha=2.0,F_folds=2,S=32)
            @test length(recovered.ret)==1 && all(isfinite,recovered.weights)
            @test BLAS.get_num_threads()==original
        end
    end
end
