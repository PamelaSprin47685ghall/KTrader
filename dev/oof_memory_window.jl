# Before/after benchmark of identical full-history four-day production work.
# Reference data remains fixed; this file never edits runtime code or limits.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test
const OM_OUT=joinpath(@__DIR__,"evidence","oof_memory_20261009")

function om_run(bars,days,tasks,blas)
    trace=KTrader.BacktestExecutionTiming()
    obs=@timed KTrader.backtest_v1(bars;from=bars.dates[end-days],S=300,seed=1,
        F_folds=3,engine=:batch,date_tasks=tasks,blas_threads=blas,execution_timing=trace)
    (;result=obs.value,seconds=obs.time,bytes=obs.bytes,gc=obs.gctime,
      compile=obs.compile_time,recompile=obs.recompile_time,
      trace=copy(trace.seconds),wall=trace.wall_seconds)
end
function om_load(stem)
    meta=TOML.parsefile(joinpath(OM_OUT,stem*".toml"))
    data=read(joinpath(OM_OUT,stem*".jls"))
    length(data)==meta["bytes"] && bytes2hex(sha256(data))==meta["sha256"] || error("comparison bytes changed")
    record=deserialize(IOBuffer(data))
    record.sources==meta["sources"] && meta["inputs"]==PANEL_INPUT_HASHES || error("comparison metadata changed")
    record
end
function om_main()
    length(ARGS)==2 || error("usage: before|after b6|p2")
    phase,mode=ARGS
    phase in ("before","after") && mode in ("b6","p2") || error("unsupported phase/topology")
    tasks=mode=="p2" ? 2 : 1; blas=mode=="p2" ? 1 : 6; days=4
    Threads.nthreads()==tasks || error("Julia threads must match tasks")
    stem=phase*"_"*mode
    all(!ispath(joinpath(OM_OUT,stem*s)) for s in (".jls",".toml")) || error("refusing to overwrite result")
    sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("local CSV changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    @assert size(bars.adj)==(14310,65) && issorted(bars.dates) && allunique(bars.dates)
    println("phase=",phase," tasks=",tasks," blas=",blas," Julia=",VERSION," CPU=",Sys.CPU_NAME,
        " heap_hint=",get(ENV,"JULIA_HEAP_SIZE_HINT","default"))
    println("source=",sources["src/response.jl"]," full prefix dates=",bars.dates[end-days:end-1]," S300/F3/default EB")
    flush(stdout)
    prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
    small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
    println("BEGIN explicit synthetic two-day API startup"); flush(stdout)
    startup=om_run(small,2,tasks,blas)
    println("startup seconds=",startup.seconds," compile=",startup.compile); flush(stdout)
    println("BEGIN real four-day warmup; full histories, fresh account/alpha chain"); flush(stdout)
    warm=om_run(bars,days,tasks,blas)
    println("warmup seconds=",warm.seconds," bytes=",warm.bytes," gc=",warm.gc," compile=",warm.compile); flush(stdout)
    GC.gc() # outside measured span; in-call GC remains included
    println("BEGIN measured real four-day replay"); flush(stdout)
    m=om_run(bars,days,tasks,blas); r=m.result
    println("measured seconds=",m.seconds," bytes=",m.bytes," gc=",m.gc,
        " compile=",m.compile," recompile=",m.recompile," window_days_per_second=",days/m.seconds); flush(stdout)
    @testset "Four-day full-history replay and accounting" begin
        @test r.dates==bars.dates[end-days:end-1]
        @test r.scenario_counts==fill(300,days)
        @test all(isfinite,r.weights) && minimum(r.weights)>=0
        @test maximum(abs.(sum(r.weights;dims=2).-1))<=1e-8
        @test maximum(sum(abs.(r.weights-warm.result.weights);dims=2))<=1e-7
        @test isapprox(r.ret,warm.result.ret;atol=1e-12,rtol=1e-10)
        @test r.weights_ew==warm.result.weights_ew
        @test r.locked_days==warm.result.locked_days
    end
    for (i,bucket) in enumerate(KTrader.TIMING_BUCKETS)
        println("daily ",bucket," ",r.timings[:,i])
    end
    sources==panel_sources() || error("source changed during work")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during work")
    end
    # Persist the same-config-verified observation before any cross-run
    # check. A failed comparison must not destroy the measured outputs.
    record=(;sources,measured=m,warmup=(;seconds=warm.seconds,compile=warm.compile),phase,mode,
        heap_hint=get(ENV,"JULIA_HEAP_SIZE_HINT","default"))
    path=joinpath(OM_OUT,stem*".jls")
    all(!ispath(joinpath(OM_OUT,stem*s)) for s in (".jls",".toml")) || error("output appeared")
    serialize(path,record)
    digest=panel_digest(path)
    open(joinpath(OM_OUT,stem*".toml"),"w") do io
        TOML.print(io,Dict("sources"=>sources,"inputs"=>PANEL_INPUT_HASHES,"sha256"=>digest,"bytes"=>filesize(path),
            "status"=>"same-config replay passed; comparison verdict in log"))
    end
    println("saved=",stem," sha256=",digest); flush(stdout)
    if phase=="after"
        baseline=isfile(joinpath(OM_OUT,"before_"*mode*".toml")) ? "before_"*mode : "before_b6"
        prior=om_load(baseline)
        println("comparison_baseline=",baseline," same_topology=",prior.mode==mode,
            "; a different topology verifies outputs, NOT a paired speedup")
        Set(keys(prior.sources))==Set(keys(sources)) || error("runtime source set changed")
        all(p=="src/response.jl" || prior.sources[p]==sources[p] for p in keys(sources)) || error("unreviewed runtime change")
        @testset "Same histories: same-topology regression or cross-topology comparison" begin
            @test r.dates==prior.measured.result.dates
            @test maximum(sum(abs.(r.weights-prior.measured.result.weights);dims=2))<=1e-7
            # Same-topology before/after retains its strict gate. Different
            # task/BLAS topology uses the pre-existing batch_window.jl gate,
            # NOT the same-config threshold accidentally used on first try.
            if prior.mode==mode
                @test isapprox(r.ret,prior.measured.result.ret;atol=1e-12,rtol=1e-10)
            else
                @test isapprox(r.ret,prior.measured.result.ret;atol=1e-9,rtol=1e-9)
            end
            @test r.weights_ew==prior.measured.result.weights_ew
            @test r.locked_days==prior.measured.result.locked_days
        end
        println("before_after_weight_L1_max=",maximum(sum(abs.(r.weights-prior.measured.result.weights);dims=2)),
            " return_max_abs=",maximum(abs.(r.ret-prior.measured.result.ret)))
    end
    sources==panel_sources() || error("source changed during work")
    println("runtime hashes unchanged; one measured window, no throughput ceiling claim")
end
om_main()
