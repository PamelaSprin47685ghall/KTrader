# Full-history, two-day production backtest. Small synthetic startup work is
# explicit and outside measurements; no truncation of the measured history.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test
const BW_OUT=joinpath(@__DIR__,"evidence","release_batch_20261009")

function bw_run(bars,blas,tasks)
    trace=KTrader.BacktestExecutionTiming()
    obs=@timed KTrader.backtest_v1(bars;from=bars.dates[end-2],S=300,seed=1,
        F_folds=3,engine=:batch,date_tasks=tasks,blas_threads=blas,execution_timing=trace)
    metrics=(;seconds=obs.time,bytes=obs.bytes,gc=obs.gctime,
        compile=obs.compile_time,recompile=obs.recompile_time,
        stages=copy(trace.seconds),wall=trace.wall_seconds)
    (;result=obs.value,metrics)
end
function bw_main()
    length(ARGS)==1 && only(ARGS) in ("b1","b6","p2") || error("mode: b1 | b6 | p2")
    mode=only(ARGS); tasks=mode=="p2" ? 2 : 1; blas=mode=="b6" ? 6 : 1
    Threads.nthreads()==tasks || error("Julia thread count must equal date_tasks")
    path=joinpath(BW_OUT,"window_"*mode*".jls"); meta=path*".toml"
    !ispath(path) && !ispath(meta) || error("refusing to overwrite window evidence")
    sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    @assert size(bars.adj)==(14310,65) && issorted(bars.dates) && allunique(bars.dates)
    println("Julia=",VERSION," CPU=",Sys.CPU_NAME," BLAS=",BLAS.get_config(),
        " date_tasks=",tasks," BLAS_threads=",blas," source=",sources["src/backtest.jl"])
    println("EXPLICIT synthetic startup: T340/N3/F3/S300/default EB, two days; excluded from measured real history")
    prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
    small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
    startup=bw_run(small,blas,tasks)
    println("synthetic_startup ",startup.metrics); flush(stdout)
    GC.gc()
    println("BEGIN real warm-up; N65 histories 1:14308 and 1:14309, dates=",bars.dates[end-2:end-1]); flush(stdout)
    warm=bw_run(bars,blas,tasks)
    println("real_warmup ",warm.metrics); flush(stdout)
    GC.gc()
    println("BEGIN measured replay; resets holdings and EB initials"); flush(stdout)
    measured=bw_run(bars,blas,tasks); result=measured.result
    println("measured ",measured.metrics," two_day_rate=",2/measured.metrics.seconds)
    @testset "Real two-day replay, complete price prefixes and accounting" begin
        @test result.dates==bars.dates[end-2:end-1]
        @test size(result.weights)==(2,65)
        @test result.scenario_counts==[300,300]
        @test all(isfinite,result.weights) && minimum(result.weights)>=0
        @test maximum(abs.(sum(result.weights;dims=2).-1))<=1e-8
        @test maximum(sum(abs.(result.weights-warm.result.weights);dims=2))<=1e-7
        @test isapprox(result.ret,warm.result.ret;atol=1e-12,rtol=1e-10)
        @test result.locked_days==warm.result.locked_days
        @test result.weights_ew==warm.result.weights_ew
    end
    for (i,bucket) in enumerate(KTrader.TIMING_BUCKETS)
        println("daily ",bucket," ",result.timings[:,i])
    end
    sources==panel_sources() || error("source changed during benchmark")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed during benchmark")
    end
    !ispath(path) && !ispath(meta) || error("output appeared during work")
    serialize(path,(;sources,result,metrics=measured.metrics,warmup=warm.metrics,startup=startup.metrics,mode))
    digest=panel_digest(path)
    open(meta,"w") do io
        TOML.print(io,Dict("sha256"=>digest,"sources"=>sources,"input_hashes"=>PANEL_INPUT_HASHES,"bytes"=>filesize(path)))
    end
    println("persisted=",basename(path)," sha256=",digest,"; one measured window, not long-run throughput")
    # Other configurations must have exactly the same source and input bytes.
    for other in ("b1","b6","p2")
        other==mode && continue
        p=joinpath(BW_OUT,"window_"*other*".jls")
        isfile(p*".toml") || continue
        m=TOML.parsefile(p*".toml")
        m["sources"]==sources && m["input_hashes"]==PANEL_INPUT_HASHES || error("cross-config source/input mismatch")
        data=read(p)
        length(data)==m["bytes"] && bytes2hex(sha256(data))==m["sha256"] || error("cross-config artifact changed")
        expected=deserialize(IOBuffer(data))
        @testset "Thread topology comparison $other vs $mode" begin
            @test result.dates==expected.result.dates
            @test maximum(sum(abs.(result.weights-expected.result.weights);dims=2))<=1e-7
            @test isapprox(result.ret,expected.result.ret;atol=1e-9,rtol=1e-9)
        end
        println("cross_config=",other," weight_L1_max=",maximum(sum(abs.(result.weights-expected.result.weights);dims=2)))
    end
    println("all source hashes unchanged; no production-default topology changed")
end
bw_main()
