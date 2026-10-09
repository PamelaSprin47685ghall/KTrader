# Complete histories, production backtest, default GC, bounded window.
# Four-day results compare against the preceding authenticated source;
# eight-day evidence extends the window, not a statistical throughput claim.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test
const BCW_OUT=joinpath(@__DIR__,"evidence","batch_memory_20261009")
function bcw_run(bars,days,tasks,blas)
    trace=KTrader.BacktestExecutionTiming()
    o=@timed KTrader.backtest_v1(bars;from=bars.dates[end-days],S=300,seed=1,
        F_folds=3,engine=:batch,date_tasks=tasks,blas_threads=blas,execution_timing=trace)
    (;result=o.value,seconds=o.time,bytes=o.bytes,gc=o.gctime,compile=o.compile_time,
      recompile=o.recompile_time,stages=copy(trace.seconds),wall=trace.wall_seconds)
end
function bcw_baseline(sources)
    dir=joinpath(@__DIR__,"evidence","oof_memory_20261009")
    meta=TOML.parsefile(joinpath(dir,"after_b6.toml"))
    meta["sha256"]=="3b7968ec5aec533e985f48dc88fb12cbac9085d3abcd591c19bddf3f0fc43f92" || error("wrong four-day baseline")
    data=read(joinpath(dir,"after_b6.jls"))
    length(data)==meta["bytes"] && bytes2hex(sha256(data))==meta["sha256"] || error("baseline bytes changed")
    baseline=deserialize(IOBuffer(data))
    baseline.sources==meta["sources"] && meta["inputs"]==PANEL_INPUT_HASHES || error("baseline metadata mismatch")
    Set(keys(baseline.sources))==Set(keys(sources)) || error("runtime source set changed")
    all(k=="src/backtest.jl" || baseline.sources[k]==sources[k] for k in keys(sources)) || error("unreviewed production change")
    baseline.measured
end
function bcw_main()
    length(ARGS)==1 && only(ARGS) in ("4-b6","4-p2","8-b6","8-p2") || error("mode: 4-b6 | 4-p2 | 8-b6 | 8-p2")
    mode=only(ARGS); days=parse(Int,first(split(mode,'-')))
    tasks=endswith(mode,"p2") ? 2 : 1; blas=tasks==2 ? 1 : 6
    Threads.nthreads()==tasks || error("Julia thread count must match tasks")
    haskey(ENV,"JULIA_HEAP_SIZE_HINT") && error("this experiment requires default GC; do not silently add a hint")
    path=joinpath(BCW_OUT,"window_"*mode*".jls"); meta=path*".toml"
    !ispath(path) && !ispath(meta) || error("refusing to overwrite evidence")
    sources=panel_sources()
    baseline=bcw_baseline(sources)
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    @assert size(bars.adj)==(14310,65) && issorted(bars.dates) && allunique(bars.dates)
    println("Julia=",VERSION," CPU=",Sys.CPU_NAME," date_tasks=",tasks," BLAS=",blas," default GC",
        " source=",sources["src/backtest.jl"])
    println("full N65 history, dates=",bars.dates[end-days:end-1]," F3/S300/default EB; no reduced model")
    prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
    small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
    println("BEGIN explicit synthetic API startup"); flush(stdout)
    startup=bcw_run(small,2,tasks,blas)
    println("startup seconds=",startup.seconds," compile=",startup.compile); flush(stdout)
    # Two real dates are a specialization warm-up, never masqueraded as an
    # eight-day before/after or a repeated statistically independent sample.
    println("BEGIN real two-day warmup"); flush(stdout)
    warm=bcw_run(bars,2,tasks,blas)
    println("warm seconds=",warm.seconds," bytes=",warm.bytes," compile=",warm.compile); flush(stdout)
    GC.gc() # outside measured span; in-call GC remains counted
    println("BEGIN measured ",days,"-day full production call"); flush(stdout)
    m=bcw_run(bars,days,tasks,blas); r=m.result
    println("measured seconds=",m.seconds," bytes=",m.bytes," gc=",m.gc," compile=",m.compile,
        " recompile=",m.recompile," window_rate=",days/m.seconds); flush(stdout)
    @testset "Full-history production window: dates and accounting" begin
        @test r.dates==bars.dates[end-days:end-1]
        @test r.scenario_counts==fill(300,days)
        @test size(r.weights)==(days,65)
        @test all(isfinite,r.weights) && minimum(r.weights)>=0
        @test maximum(abs.(sum(r.weights;dims=2).-1))<=1e-8
        @test all(isfinite,r.ret) && all(isfinite,r.wealth)
        @test r.wealth≈vcat(1.0,cumprod(1 .+ r.ret))
    end
    for (i,b) in enumerate(KTrader.TIMING_BUCKETS)
        println("daily ",b," ",r.timings[:,i])
    end
    sources==panel_sources() || error("source changed during window")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed during window")
    end
    !ispath(path) && !ispath(meta) || error("output appeared")
    serialize(path,(;sources,inputs=PANEL_INPUT_HASHES,measured=m,mode,warmup=(;seconds=warm.seconds,compile=warm.compile)))
    digest=panel_digest(path)
    open(meta,"w") do io
        TOML.print(io,Dict("sha256"=>digest,"bytes"=>filesize(path),"sources"=>sources,
            "inputs"=>PANEL_INPUT_HASHES,"status"=>"accounting passed; regression verdict in log"))
    end
    println("saved=",basename(path)," sha256=",digest); flush(stdout)
    if days==4
        @testset "Four-day old-cache/new-no-cache regression" begin
            @test r.dates==baseline.result.dates
            @test maximum(sum(abs.(r.weights-baseline.result.weights);dims=2))<=1e-7
            @test isapprox(r.ret,baseline.result.ret;atol=tasks==1 ? 1e-12 : 1e-9,rtol=tasks==1 ? 1e-10 : 1e-9)
            @test r.weights_ew==baseline.result.weights_ew
            @test r.locked_days==baseline.result.locked_days
        end
        println("vs_previous_weight_L1=",maximum(sum(abs.(r.weights-baseline.result.weights);dims=2)),
            " return_max=",maximum(abs.(r.ret-baseline.result.ret))," same_topology=",tasks==1)
    end
    other=joinpath(BCW_OUT,"window_"*string(days)* (tasks==1 ? "-p2" : "-b6")*".jls")
    if isfile(other*".toml")
        md=TOML.parsefile(other*".toml"); data=read(other)
        md["sources"]==sources && md["inputs"]==PANEL_INPUT_HASHES || error("cross topology mismatch")
        bytes2hex(sha256(data))==md["sha256"] && length(data)==md["bytes"] || error("cross topology bytes changed")
        o=deserialize(IOBuffer(data)).measured.result
        @testset "Current-source equal window across task/BLAS topology" begin
            @test r.dates==o.dates
            @test maximum(sum(abs.(r.weights-o.weights);dims=2))<=1e-7
            @test isapprox(r.ret,o.ret;atol=1e-9,rtol=1e-9)
        end
        println("cross_topology_weight_L1=",maximum(sum(abs.(r.weights-o.weights);dims=2)))
    end
    println("all runtime/input hashes unchanged; only a bounded window, no release sign-off")
end
bcw_main()
