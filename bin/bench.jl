#!/usr/bin/env julia
# Compare engines on the same full-history panel. No broker or external requests.
#
# 历史 release 工具（2.0-RC legacy line，2026-10-10 入口切换后标注）：
# 本脚本对比旧线 `backtest_v1` 的 batch / incremental 引擎，属于 2.0.0
# 历史线的性能工具。Gate-0 当前唯一入口是 bin/backtest.jl（KTraderGate0
# slow reference，无引擎对比面；其运行成本见该脚本头注释）。保留不删除
# （D-002/§46），不再作为当前入口维护；旧参数面（ENGINE / SCENARIOS /
# F_FOLDS / ADAPTIVE_SCENARIOS 等）仅在本历史线内有效。
using KTrader, Dates, Statistics, Random, LinearAlgebra, Printf

function report_execution(trace)
    @printf("  inference/consumer wall span %.6fs; windows=%d blocks=%d coordinator_advances=%d block_advances=%d\n",
        trace.wall_seconds,trace.windows,trace.blocks,trace.coordinator_advances,trace.block_advances)
    for (i,bucket) in enumerate(KTrader.BACKTEST_EXECUTION_BUCKETS)
        @printf("  %-22s %.6fs\n",bucket,trace.seconds[i])
    end
    println("  block_advance/solve are sums of task elapsed times, not process CPU or total wall time")
end

function check_equivalence(reference,result)
    reference.dates==result.dates && reference.symbols==result.symbols || error("decision coordinates changed")
    for name in (:ret,:ew,:wealth,:wealth_ew)
        isapprox(getproperty(reference,name),getproperty(result,name);atol=1e-8,rtol=1e-8) ||
            error("numerical equivalence failed for $name")
    end
    for name in (:weights,:weights_ew)
        isapprox(getproperty(reference,name),getproperty(result,name);atol=1e-6,rtol=1e-6) ||
            error("numerical equivalence failed for $name")
    end
    for name in (:locked_days,:locked_days_ew,:scenario_counts,:timing_buckets)
        getproperty(reference,name)==getproperty(result,name) || error("semantics changed for $name")
    end
end

function main()
    data=joinpath(@__DIR__,"..","data")
    b=load_bars(data)
    T,N=size(b.adj)
    days=parse(Int,get(ENV,"BENCH_DAYS","500"))
    repeats=parse(Int,get(ENV,"BENCH_REPEATS","3"))
    days>0 && repeats>0 || throw(ArgumentError("BENCH_DAYS and BENCH_REPEATS must be positive"))
    date_tasks=parse(Int,get(ENV,"DATE_TASKS",string(Threads.nthreads())))
    blas_threads=parse(Int,get(ENV,"BLAS_THREADS","1"))
    chunk_size=parse(Int,get(ENV,"CHUNK_SIZE",string(min(24,4date_tasks))))
    S=parse(Int,get(ENV,"SCENARIOS","300"))
    seed=parse(Int,get(ENV,"SEED","1"))
    F_folds=parse(Int,get(ENV,"F_FOLDS","3"))
    alpha=get(ENV,"RIDGE_ALPHA","auto")
    ridge_alpha=alpha=="auto" ? nothing : parse(Float64,alpha)
    adaptive=get(ENV,"ADAPTIVE_SCENARIOS","false")=="true"
    quadrature_tol=parse(Float64,get(ENV,"QUADRATURE_TOL","1e-5"))
    max_scenarios=parse(Int,get(ENV,"MAX_SCENARIOS","512"))
    selection=Symbol(get(ENV,"ENGINE","batch"))
    selection in (:batch,:incremental,:compare) || throw(ArgumentError("ENGINE must be batch, incremental or compare"))
    engines=selection===:compare ? (:batch,:incremental) : (selection,)
    options=(;S,seed,ridge_alpha,F_folds,date_tasks,blas_threads,chunk_size,adaptive,quadrature_tol,max_scenarios)
    from=b.dates[max(1,T-days)]
    println("Julia=$(VERSION) threads=$(Threads.nthreads()) data=$data panel=($T,$N) input=$(first(b.dates)):$(last(b.dates))")
    println("ENGINE=$selection BENCH_DAYS=$days BENCH_REPEATS=$repeats from=$from options=$options")
    # Warm the block/checkpoint path too. Measured runs always retain b's full history.
    P=adaptive ? ones(600,4) : exp.(cumsum(0.01randn(MersenneTwister(8),600,4);dims=1))
    dates=collect(Date(2020,1,1).+Day.(0:599))
    warmbars=Bars(dates,["A","B","C","D"],P,P)
    for engine in engines
        warmup=@elapsed backtest_v1(warmbars;from=dates[end-2],options...,engine)
        @printf("Warmup engine=%s %.6fs (excluded from measured trials)\n",string(engine),warmup)
    end
    walls=Dict(engine=>Float64[] for engine in engines)
    reference=nothing
    for trial in 1:repeats
        for engine in (isodd(trial) ? engines : reverse(engines))
            trace=KTrader.BacktestExecutionTiming()
            measured=@timed backtest_v1(b;from,options...,engine,execution_timing=trace)
            result=measured.value
            equivalence=reference === nothing ? "reference" : "checked"
            if reference === nothing
                reference=result
            else
                check_equivalence(reference,result)
            end
            push!(walls[engine],measured.time)
            @printf("Trial %d engine=%s decisions=%d wall=%.6fs days/s=%.3f allocated=%.3fMiB gc=%.6fs\n",
                trial,string(engine),length(result.ret),measured.time,length(result.ret)/measured.time,
                measured.bytes/2.0^20,measured.gctime)
            report_execution(trace)
            for rows in (1:length(result.ret),max(1,length(result.ret)-499):length(result.ret))
                println("Decision-bucket elapsed sums $(first(rows)):$(last(rows)); nested in execution buckets")
                for (i,bucket) in enumerate(result.timing_buckets)
                    @printf("  %-10s %.6fs\n",bucket,sum(view(result.timings,rows,i)))
                end
            end
            @printf("  CAGR=%.6f final_wealth=%.6f scenario_counts=%d:%d equivalence=%s\n",
                summarize(result.ret).cagr,result.wealth[end],minimum(result.scenario_counts),maximum(result.scenario_counts),equivalence)
        end
    end
    for engine in engines
        @printf("Summary engine=%s wall median=%.6fs min=%.6fs max=%.6fs trials=%d\n",
            string(engine),median(walls[engine]),minimum(walls[engine]),maximum(walls[engine]),repeats)
    end
end
main()
