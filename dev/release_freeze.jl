# One explicit closure action per bounded command; no production modification.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test, Statistics
const RF_OUT=joinpath(@__DIR__,"evidence","release_freeze_20261009")
const RF_QUAL=joinpath(@__DIR__,"evidence","batch_memory_20261009","qualification")
const RF_OLD_HASH="680f1494d2d435847af2b4972d8be440e9f4f519fb2af0889789e302dbabeeba"
const RF_RUNS=("latest1","latest2","earlier1","earlier2")

function rf_write(path,record)
    ispath(path) && error("refusing to overwrite $path")
    open(path,"w") do io
        TOML.print(io,record;sorted=true)
    end
end
function rf_qualification()
    path=joinpath(RF_QUAL,"qualification_snapshot.toml")
    panel_digest(path)==RF_OLD_HASH || error("qualification snapshot changed")
    q=TOML.parsefile(path)
    for (p,h) in q["sources"]
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("qualified file changed: $p")
    end
    # New runtime/test files cannot hide outside the old qualification inventory.
    for dir in ("src","test"), (root,_,files) in walkdir(joinpath(PANEL_ROOT,dir)), f in files
        endswith(f,".jl") || endswith(f,".jls") || continue
        relpath(joinpath(root,f),PANEL_ROOT) in keys(q["sources"]) || error("unqualified new file: $f")
    end
    receipts=Dict{String,String}()
    for (group,entries) in q["groups"]
        p=joinpath(RF_QUAL,"pass_"*group*".toml")
        r=TOML.parsefile(p)
        r["passed"] && r["snapshot"]==RF_OLD_HASH && r["entries"]==entries || error("invalid receipt $group")
        receipts[group]=panel_digest(p)
    end
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed")
    end
    (;q,receipts)
end
function rf_frozen()
    current=rf_qualification()
    s=TOML.parsefile(joinpath(RF_OUT,"freeze.toml"))
    s["driver_sha256"]==panel_digest(@__FILE__) || error("closure driver changed")
    s["qualified_sources"]==current.q["sources"] && s["receipts"]==current.receipts || error("frozen state changed")
    s["inputs"]==PANEL_INPUT_HASHES || error("input declaration changed")
    s
end
function rf_load(path,digest)
    bytes=read(path)
    bytes2hex(sha256(bytes))==digest || error("artifact bytes changed: $path")
    deserialize(IOBuffer(bytes))
end
function rf_run(bars,days)
    trace=KTrader.BacktestExecutionTiming()
    o=@timed KTrader.backtest_v1(bars;from=bars.dates[end-days],S=300,seed=1,
        F_folds=3,engine=:batch,date_tasks=1,blas_threads=6,execution_timing=trace)
    (;result=o.value,seconds=o.time,bytes=o.bytes,gc=o.gctime,compile=o.compile_time,
      recompile=o.recompile_time,stages=copy(trace.seconds),wall=trace.wall_seconds)
end
function rf_status(s)
    completed=String[]
    for label in RF_RUNS
        path=joinpath(RF_OUT,label*".toml")
        if !isfile(path)
            println("MISSING ",label); continue
        end
        r=TOML.parsefile(path)
        r["passed"] && r["freeze_sha256"]==panel_digest(joinpath(RF_OUT,"freeze.toml")) || error("wrong run receipt")
        artifact=rf_load(joinpath(RF_OUT,label*".jls"),r["sha256"])
        artifact.sources==panel_sources() || error("run sources differ")
        println("PASS ",label," seconds=",r["seconds"]," bytes=",r["bytes"]," gc=",r["gc"],
            " compile=",r["compile"]," dates=",r["dates"])
        push!(completed,label)
    end
    println("QUALIFIED groups=",length(s["receipts"])," standard_files=",s["standard_file_count"],
        " frozen_replays=",length(completed),"/",length(RF_RUNS),"; not a 2.0 performance sign-off")
    length(completed)==length(RF_RUNS) || error("repeat verification incomplete")
end
function rf_main()
    length(ARGS)==1 || error("one action: snapshot | status | latest1 | latest2 | earlier1 | earlier2")
    action=only(ARGS)
    if action=="snapshot"
        state=rf_qualification()
        rf_write(joinpath(RF_OUT,"freeze.toml"),Dict(
            "qualification_sha256"=>RF_OLD_HASH,"qualified_sources"=>state.q["sources"],
            "receipts"=>state.receipts,"standard_file_count"=>length(state.q["standard_files"]),
            "driver_sha256"=>panel_digest(@__FILE__),"inputs"=>PANEL_INPUT_HASHES))
        println("FROZEN groups=",length(state.receipts)," standard_files=",length(state.q["standard_files"]),
            " snapshot=",panel_digest(joinpath(RF_OUT,"freeze.toml"))); return
    end
    s=rf_frozen()
    action=="status" && return rf_status(s)
    action in RF_RUNS || error("unknown action")
    Threads.nthreads()==1 || error("fixed single-task protocol")
    haskey(ENV,"JULIA_HEAP_SIZE_HINT") && error("default GC required")
    path=joinpath(RF_OUT,action*".jls"); meta=joinpath(RF_OUT,action*".toml")
    !ispath(path) && !ispath(meta) || error("output already exists")
    sources=panel_sources(); BLAS.set_num_threads(6)
    allbars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    @assert size(allbars.adj)==(14310,65)
    earlier=startswith(action,"earlier")
    t=size(allbars.adj,1)-(earlier ? 8 : 0)
    # Only the end date changes: every decision still consumes its FULL prefix.
    bars=KTrader.Bars(allbars.dates[1:t],copy(allbars.symbols),allbars.close[1:t,:],
        allbars.adj[1:t,:],allbars.bar[1:t,:])
    println("RUN ",action," Julia=",VERSION," CPU=",Sys.CPU_NAME," BLAS=",BLAS.get_config(),
        " threads=",BLAS.get_num_threads()," full_rows=",t," dates=",bars.dates[end-8:end-1]); flush(stdout)
    prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
    small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
    startup=rf_run(small,2)
    println("synthetic_startup seconds=",startup.seconds," compile=",startup.compile); flush(stdout)
    warm=rf_run(bars,2)
    println("real_two_day_warmup seconds=",warm.seconds," compile=",warm.compile); flush(stdout)
    GC.gc()
    println("BEGIN measured8 full-history days; fresh holdings/alpha chain"); flush(stdout)
    measured=rf_run(bars,8); r=measured.result
    println("measured seconds=",measured.seconds," allocated_bytes=",measured.bytes," gc=",measured.gc,
        " compile=",measured.compile," recompile=",measured.recompile); flush(stdout)
    @testset "Frozen eight-day execution, complete inputs and accounting" begin
        @test r.dates==bars.dates[end-8:end-1]
        @test size(r.weights)==(8,65) && r.scenario_counts==fill(300,8)
        @test all(isfinite,r.weights) && minimum(r.weights)>=0
        @test maximum(abs.(sum(r.weights;dims=2).-1))<=1e-8
        @test all(isfinite,r.ret) && all(isfinite,r.wealth)
        @test r.wealth≈vcat(1.0,cumprod(1 .+ r.ret))
        @test measured.compile==0.0 && measured.recompile==0.0
    end
    rf_frozen()
    # Persist raw measurement first; only a passing receipt admits this run.
    serialize(path,(;sources,measured,action,warmup=(;seconds=warm.seconds,compile=warm.compile)))
    digest=panel_digest(path)
    expected=nothing
    if !earlier
        old=rf_load(joinpath(@__DIR__,"evidence","batch_memory_20261009","window_8-b6.jls"),
            "f315928940e9d4fde8ad2b9e0a5ec720e7e69f23ea2017b95725f260fb095704")
        old.sources==sources || error("reference sources differ")
        expected=old.measured.result
    elseif action=="earlier2"
        m=TOML.parsefile(joinpath(RF_OUT,"earlier1.toml"))
        m["passed"] || error("earlier first observation not accepted")
        old=rf_load(joinpath(RF_OUT,"earlier1.jls"),m["sha256"])
        old.sources==sources || error("earlier-window sources differ")
        expected=old.measured.result
    end
    if expected!==nothing
        @testset "Independent process replay at original same-topology gates" begin
            @test r.dates==expected.dates && r.symbols==expected.symbols
            @test maximum(sum(abs.(r.weights-expected.weights);dims=2))<=1e-7
            @test isapprox(r.ret,expected.ret;atol=1e-12,rtol=1e-10)
            @test isapprox(r.wealth,expected.wealth;atol=1e-12,rtol=1e-10)
            @test r.weights_ew==expected.weights_ew && r.locked_days==expected.locked_days
        end
        println("weight_L1_max=",maximum(sum(abs.(r.weights-expected.weights);dims=2)),
            " return_max=",maximum(abs.(r.ret-expected.ret)))
    else
        println("first earlier-window observation; independent replay still pending")
    end
    rf_frozen()
    rf_write(meta,Dict("passed"=>true,"freeze_sha256"=>panel_digest(joinpath(RF_OUT,"freeze.toml")),
        "sha256"=>digest,"seconds"=>measured.seconds,"bytes"=>measured.bytes,"gc"=>measured.gc,
        "compile"=>measured.compile,"dates"=>string.(r.dates)))
    println("SAVED ",action," sha256=",digest," qualified files and inputs unchanged"); flush(stdout)
end
rf_main()
