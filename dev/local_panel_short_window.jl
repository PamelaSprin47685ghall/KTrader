# Two real local-panel decision days; full history, default EB/F3/S300.
# One explicit same-API warm-up followed by ONE measured replay. This is
# bounded end-to-end evidence, not a long-horizon throughput benchmark.
# Admission history: the BLAS1 warm-up hit its 45s scoped deadline; BLAS6
# was NOT attempted. Preserve that log and isolate startup costs before any
# new invocation; this script is not an accepted throughput benchmark.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test

const SHORT_EVIDENCE=joinpath(@__DIR__,"evidence","release_qualification_20261009")

function short_window_once(bars,blas)
    trace=KTrader.BacktestExecutionTiming()
    obs=@timed KTrader.backtest_v1(bars;from=bars.dates[end-2],S=300,
        seed=1,F_folds=3,date_tasks=1,blas_threads=blas,engine=:batch,
        execution_timing=trace)
    metrics=(;seconds=obs.time,bytes=obs.bytes,gc_seconds=obs.gctime,
        compile_time=obs.compile_time,recompile_time=obs.recompile_time,
        trace_seconds=copy(trace.seconds),wall_seconds=trace.wall_seconds,
        blocks=trace.blocks,windows=trace.windows)
    (;result=obs.value,metrics)
end

function short_window_main()
    length(ARGS)==1 && only(ARGS) in ("--blas=1","--blas=6") ||
        error("expected --blas=1 or --blas=6; date_tasks=1 and two days are fixed")
    blas=parse(Int,last(split(only(ARGS),'=')))
    Threads.nthreads()==1 || error("this comparison requires JULIA_NUM_THREADS=1")
    stem="short_blas$(blas)"
    path=joinpath(SHORT_EVIDENCE,stem*".jls")
    metapath=joinpath(SHORT_EVIDENCE,stem*".toml")
    !ispath(path) && !ispath(metapath) || error("refusing to overwrite short-window evidence")
    sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed: $p")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    @assert size(bars.adj)==(14310,65)
    @assert issorted(bars.dates) && allunique(bars.dates)
    @assert isequal(isfinite.(KTrader.signal_prices(bars)),bars.bar)
    BLAS.set_num_threads(blas)
    println("source=",sources["src/response.jl"]," incremental_source=",sources["src/incremental.jl"])
    println("Julia=",VERSION," CPU=",Sys.CPU_NAME," BLAS=",BLAS.get_config(),
        " Julia_threads=",Threads.nthreads()," date_tasks=1 BLAS_threads=",blas)
    println("local CSV N65 T14310; decisions=",bars.dates[end-2:end-1],
        "; signal prefixes 1:14308 and 1:14309; S300/F3/default independent EB")
    println("BEGIN explicit same-API two-day warm-up; measured replay resets holdings and alpha chains"); flush(stdout)
    warm=short_window_once(bars,blas)
    println("warmup ",warm.metrics); flush(stdout)
    GC.gc() # outside the measured call; in-call GC remains included
    println("BEGIN measured two-day replay"); flush(stdout)
    measured=short_window_once(bars,blas)
    result=measured.result
    println("measured ",measured.metrics)
    println("window_days_per_second=",2/measured.metrics.seconds,
        " (includes setup, scheduling, spill I/O and Kelly; two-day window only)")
    for (i,bucket) in enumerate(KTrader.TIMING_BUCKETS)
        println("daily_bucket ",bucket,"=",result.timings[:,i])
    end
    @testset "Two-day full-history backtest: deterministic replay and portfolio accounting" begin
        @test result.dates==bars.dates[end-2:end-1]
        @test size(result.weights)==(2,65)
        @test result.scenario_counts==[300,300]
        @test all(isfinite,result.weights) && minimum(result.weights)>=0
        @test all(abs.(vec(sum(result.weights;dims=2)).-1).<=1e-8)
        @test isapprox(result.weights,warm.result.weights;atol=1e-9,rtol=1e-9)
        @test isapprox(result.ret,warm.result.ret;atol=1e-12,rtol=1e-10)
        @test isapprox(result.wealth,warm.result.wealth;atol=1e-12,rtol=1e-10)
        @test result.locked_days==warm.result.locked_days
        @test result.weights_ew==warm.result.weights_ew
    end
    panel_sources()==sources || error("runtime source changed during short window")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during short window")
    end
    # Preserve the measured observation BEFORE cross-thread assertions, so
    # a real discrepancy is inspectable without rerunning either workload.
    # Same-config replay passed; cross-config acceptance is ONLY in the log.
    !ispath(path) && !ispath(metapath) || error("output appeared during measurement")
    serialize(path,(;sources,input_hashes=PANEL_INPUT_HASHES,result,
        metrics=measured.metrics,warmup=warm.metrics,blas))
    digest=panel_digest(path)
    open(metapath,"w") do io
        TOML.print(io,Dict("file"=>basename(path),"sha256"=>digest,
            "bytes"=>filesize(path),"source_hashes"=>sources,
            "input_hashes"=>PANEL_INPUT_HASHES,"blas_threads"=>blas,
            "status"=>"same-config replay passed; cross-config verdict in log"))
    end
    println("artifact=",basename(path)," sha256=",digest); flush(stdout)
    otherstem="short_blas$(blas==1 ? 6 : 1)"
    othermeta=joinpath(SHORT_EVIDENCE,otherstem*".toml")
    if isfile(othermeta)
        meta=TOML.parsefile(othermeta)
        meta["file"]==otherstem*".jls" || error("comparison filename mismatch")
        meta["source_hashes"]==sources && meta["input_hashes"]==PANEL_INPUT_HASHES ||
            error("cannot compare different sources or inputs")
        data=read(joinpath(SHORT_EVIDENCE,meta["file"]))
        length(data)==meta["bytes"] && bytes2hex(sha256(data))==meta["sha256"] || error("comparison bytes changed")
        other=deserialize(IOBuffer(data))
        other.sources==sources || error("comparison record source mismatch")
        @testset "Same dates and model under BLAS1 and BLAS6" begin
            @test result.dates==other.result.dates
            @test result.scenario_counts==other.result.scenario_counts
            # Existing I0 weight-comparison budget, applied per day to the
            # whole portfolio (not N times a per-cell absolute tolerance).
            @test maximum(sum(abs.(result.weights.-other.result.weights);dims=2))<=1e-7
            @test isapprox(result.ret,other.result.ret;atol=1e-9,rtol=1e-9)
        end
        println("cross_thread_weight_max_abs=",maximum(abs.(result.weights.-other.result.weights)),
            " return_max_abs=",maximum(abs.(result.ret.-other.result.ret)))
    else
        println("cross-thread comparison NOT YET RUN; no other configuration record")
    end
    panel_sources()==sources || error("runtime source changed during short window")
    println("source_unchanged=true; not a 2.0 sign-off"); flush(stdout)
end
short_window_main()
