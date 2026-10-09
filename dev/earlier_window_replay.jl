# Full earlier eight-day production backtest; explicit single-day real warmup
# avoids spending the scope twice on the unrelated final two-day alpha chain.
# The measured eight days, numerical budgets and process guard are unchanged.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test
const EWR_OUT=joinpath(@__DIR__,"evidence","earlier_closure_20261009")
function ewr_fit(index,sources)
    p=joinpath(EWR_OUT,"fit_day_"*string(index)*".jls")
    meta=TOML.parsefile(p*".toml"); bytes=read(p)
    meta["sources"]==sources && meta["inputs"]==PANEL_INPUT_HASHES || error("fit source/input mismatch")
    length(bytes)==meta["bytes"] && bytes2hex(sha256(bytes))==meta["sha256"] || error("fit bytes changed")
    r=deserialize(IOBuffer(bytes))
    r.sources==sources && r.index==index || error("fit identity mismatch")
    r
end
function ewr_run(bars,days)
    trace=KTrader.BacktestExecutionTiming()
    o=@timed KTrader.backtest_v1(bars;from=bars.dates[end-days],S=300,seed=1,
        F_folds=3,engine=:batch,date_tasks=1,blas_threads=6,execution_timing=trace)
    (;result=o.value,seconds=o.time,bytes=o.bytes,gc=o.gctime,
        compile=o.compile_time,recompile=o.recompile_time,stages=trace.seconds)
end
function ewr_main()
    length(ARGS)==1 && only(ARGS) in ("1","2","verify1","verify2") || error("action: 1 | 2 | verify1 | verify2")
    startswith(only(ARGS),"verify") && return ewr_verify(last(only(ARGS)))
    runid=only(ARGS); path=joinpath(EWR_OUT,"window_"*runid*".jls"); meta=path*".toml"
    !ispath(path) && !ispath(meta) || error("refusing to overwrite replay")
    Threads.nthreads()==1 || error("single-task protocol")
    haskey(ENV,"JULIA_HEAP_SIZE_HINT") && error("default GC required")
    BLAS.set_num_threads(6); sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    b=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    t=14302
    @assert size(b.adj)==(14310,65)
    bars=KTrader.Bars(b.dates[1:t],copy(b.symbols),b.close[1:t,:],b.adj[1:t,:],b.bar[1:t,:])
    # Require the complete diagnostic chain BEFORE starting another window.
    for i in 1:8
        m=TOML.parsefile(joinpath(EWR_OUT,"fit_day_"*string(i)*".jls.toml"))
        m["sources"]==sources || error("diagnostic chain incomplete or stale")
    end
    println("run=",runid," full N65 prefixes1:14294..14301 dates=",bars.dates[end-8:end-1],
        " default EB/F3/S300 task1 BLAS6 Julia=",VERSION," CPU=",Sys.CPU_NAME); flush(stdout)
    prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
    small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
    startup=ewr_run(small,2)
    println("synthetic_startup seconds=",startup.seconds," compile=",startup.compile); flush(stdout)
    firstend=t-7
    firstbars=KTrader.Bars(b.dates[1:firstend],copy(b.symbols),b.close[1:firstend,:],b.adj[1:firstend,:],b.bar[1:firstend,:])
    warm=ewr_run(firstbars,1)
    println("real_first_day_warmup seconds=",warm.seconds," compile=",warm.compile,
        " decision=",only(warm.result.dates)); flush(stdout)
    GC.gc()
    println("BEGIN full production eight-day call, fresh holdings and initial alpha pairs"); flush(stdout)
    m=ewr_run(bars,8); r=m.result
    println("MEASURED seconds=",m.seconds," bytes=",m.bytes," gc=",m.gc,
        " compile=",m.compile," recompile=",m.recompile," rate=",8/m.seconds); flush(stdout)
    @testset "Complete earlier window and portfolio accounting" begin
        @test r.dates==bars.dates[end-8:end-1]
        @test size(r.weights)==(8,65) && r.scenario_counts==fill(300,8)
        @test all(isfinite,r.weights) && minimum(r.weights)>=0
        @test maximum(abs.(sum(r.weights;dims=2).-1))<=1e-8
        @test all(isfinite,r.ret) && all(isfinite,r.wealth)
        @test r.wealth≈vcat(1.0,cumprod(1 .+ r.ret))
    end
    sources==panel_sources() || error("runtime changed")
    serialize(path,(;sources,inputs=PANEL_INPUT_HASHES,measured=m,runid))
    digest=panel_digest(path)
    # A failed comparison preserves raw measurements but never a pass receipt.
    println("raw_artifact=",basename(path)," sha256=",digest); flush(stdout)
    sources==panel_sources() || error("runtime changed after comparisons")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed after comparisons")
    end
    open(meta,"w") do io
        TOML.print(io,Dict("accounting_passed"=>true,"comparison_pending"=>true,"sources"=>sources,"sha256"=>digest,
            "seconds"=>m.seconds,"compile"=>m.compile,"dates"=>string.(r.dates)))
    end
    println("MEASUREMENT COMPLETE; independent artifact comparison still required")
end

function ewr_verify(id)
    runid=string(id); sources=panel_sources(); BLAS.set_num_threads(6)
    path=joinpath(EWR_OUT,"window_"*runid*".jls")
    receipt=joinpath(EWR_OUT,"verified_"*runid*".toml")
    ispath(receipt) && error("refusing to overwrite verification")
    # Run1 completed the numeric window and printed this hash before timing
    # out in postprocessing. Authenticate exactly those surviving bytes;
    # its producer command remains RC124 even after a separate verification.
    digest=runid=="1" ? "99cf04c814ebfbfe888edb32be15216f5b32fb6738fab442f114902a1c05e8cc" :
        TOML.parsefile(path*".toml")["sha256"]
    bytes=read(path)
    bytes2hex(sha256(bytes))==digest || error("measured artifact changed")
    record=deserialize(IOBuffer(bytes)); r=record.measured.result
    record.sources==sources && record.inputs==PANEL_INPUT_HASHES || error("measured identity changed")
    bytes=nothing; GC.gc()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data")); held=nothing
    println("VERIFY run=",runid," sha256=",digest," numeric_window_seconds=",record.measured.seconds,
        "; no posterior fit, only authenticated model scenario/Kelly comparisons"); flush(stdout)
    @testset "Production schedule vs separately authenticated chronological fits" begin
        for i in 1:8
            p=ewr_fit(i,sources)
            @test p.t==14293+i && p.date==r.dates[i]
            X=KTrader.generate_scenarios_v1(p.model;S=300,rng=MersenneTwister(1+p.t))
            free=falses(65)
            for j in p.model.active_indices
                free[j]=bars.bar[p.t,j]
            end
            w=KTrader.scenario_weights(X,p.model.active_indices,free,held)
            @test norm(w-vec(r.weights[i,:]),1)<=1e-7
            gross=bars.adj[p.t+1,:]./bars.adj[p.t,:]
            clean=ifelse.(isfinite.(gross),gross,1.0)
            ret=dot(w,clean)-1
            @test isapprox(ret,r.ret[i];atol=1e-12,rtol=1e-10)
            held=(w.*clean)./(1+ret)
            println("verified date=",p.date," weight_L1=",norm(w-vec(r.weights[i,:]),1)); flush(stdout)
            p=nothing; X=nothing; GC.gc() # verifier only, not measured solver span
        end
    end
    if runid=="2"
        m=TOML.parsefile(joinpath(EWR_OUT,"verified_1.toml"))
        m["passed"] && m["sources"]==sources || error("first verification missing")
        bytes=read(joinpath(EWR_OUT,"window_1.jls"))
        bytes2hex(sha256(bytes))==m["sha256"] || error("first output changed")
        old=deserialize(IOBuffer(bytes)).measured.result
        @testset "Independent complete earlier-window process replay" begin
            @test r.dates==old.dates
            @test maximum(sum(abs.(r.weights-old.weights);dims=2))<=1e-7
            @test isapprox(r.ret,old.ret;atol=1e-12,rtol=1e-10)
            @test r.weights_ew==old.weights_ew && r.locked_days==old.locked_days
        end
        println("replay_weight_L1_max=",maximum(sum(abs.(r.weights-old.weights);dims=2)),
            " return_max=",maximum(abs.(r.ret-old.ret)))
    end
    sources==panel_sources() || error("source changed during verification")
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during verification")
    end
    open(receipt,"w") do io
        TOML.print(io,Dict("passed"=>true,"sources"=>sources,"sha256"=>digest,
            "producer_command_rc"=>(runid=="1" ? 124 : 0),"seconds"=>record.measured.seconds))
    end
    println("VERIFIED; producer status retained separately")
end
ewr_main()
