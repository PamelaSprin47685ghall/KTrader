# Locate the frozen earlier-window failure using the existing production fit.
# No changed initial values, tolerance, alternate solver, or exception swallow.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Test
const RFP_OUT=joinpath(@__DIR__,"evidence","release_freeze_20261009")
function rfp_main()
    length(ARGS)==1 && only(ARGS) in ("handoff","direct","latest","earlier") || error("phase: handoff | direct | latest | earlier")
    phase=only(ARGS)
    BLAS.set_num_threads(6); sources=panel_sources()
    for (p,h) in PANEL_INPUT_HASHES
        panel_digest(joinpath(PANEL_ROOT,p))==h || error("input changed")
    end
    bars=KTrader.load_bars(joinpath(PANEL_ROOT,"data"))
    t=size(bars.adj,1)-(phase=="latest" ? 0 : 8)
    earlier=KTrader.Bars(bars.dates[1:t],copy(bars.symbols),bars.close[1:t,:],bars.adj[1:t,:],bars.bar[1:t,:])
    println("phase=",phase," source=",sources["src/backtest.jl"]," response_source=",sources["src/response.jl"],
        " first_decision=",earlier.dates[end-2]," last_decision=",earlier.dates[end-1]); flush(stdout)
    try
        if phase=="handoff"
            KTrader.backtest_v1(earlier;from=earlier.dates[end-2],S=300,seed=1,
                F_folds=3,engine=:batch,date_tasks=1,blas_threads=6)
        elseif phase=="direct"
            signal=KTrader.signal_prices(earlier)
            history=KTrader.PriceHistoryCache(signal)
            workspace=KTrader.FitWorkspace()
            initials=fill((1.0,1.0),4)
            for day in (t-2):(t-1)
                println("BEGIN date=",earlier.dates[day]," prefix=1:",day); flush(stdout)
                KTrader.fit_v1(view(signal,1:day,:);F_folds=3,history_cache=history,
                    alpha_initial=initials,workspace)
                println("END date=",earlier.dates[day]); flush(stdout)
            end
        else
            # Explicit post-repair comparison, never relabel old freeze passes.
            q=TOML.parsefile(joinpath(RFP_OUT,"freeze.toml"))["qualified_sources"]
            for (p,h) in sources
                p in ("src/backtest.jl","src/response.jl") || q[p]==h || error("unreviewed runtime change: $p")
            end
            output=joinpath(RFP_OUT,"repaired_"*phase*".jls")
            ispath(output) && error("refusing to overwrite repaired observation")
            prices=exp.(cumsum(0.01randn(MersenneTwister(199),340,3);dims=1))
            small=KTrader.Bars(collect(Date(2020,1,1).+Day.(0:339)),["A","B","C"],prices,prices)
            KTrader.backtest_v1(small;from=small.dates[end-2],S=300,seed=1,F_folds=3,
                engine=:batch,date_tasks=1,blas_threads=6)
            println("synthetic API startup completed; BEGIN real two-day warmup"); flush(stdout)
            KTrader.backtest_v1(earlier;from=earlier.dates[end-2],S=300,seed=1,F_folds=3,
                engine=:batch,date_tasks=1,blas_threads=6)
            GC.gc()
            println("BEGIN repaired eight-day window ",earlier.dates[end-8:end-1]); flush(stdout)
            observation=@timed KTrader.backtest_v1(earlier;from=earlier.dates[end-8],S=300,
                seed=1,F_folds=3,engine=:batch,date_tasks=1,blas_threads=6)
            r=observation.value
            println("seconds=",observation.time," bytes=",observation.bytes," gc=",observation.gctime,
                " compile=",observation.compile_time," recompile=",observation.recompile_time); flush(stdout)
            @testset "Post-repair production window at original accounting gates" begin
                @test r.dates==earlier.dates[end-8:end-1]
                @test size(r.weights)==(8,65) && r.scenario_counts==fill(300,8)
                @test all(isfinite,r.weights) && minimum(r.weights)>=0
                @test maximum(abs.(sum(r.weights;dims=2).-1))<=1e-8
                @test all(isfinite,r.ret) && all(isfinite,r.wealth)
                @test r.wealth≈vcat(1.0,cumprod(1 .+ r.ret))
            end
            if phase=="latest"
                path=joinpath(@__DIR__,"evidence","batch_memory_20261009","window_8-b6.jls")
                bytes=read(path)
                bytes2hex(sha256(bytes))=="f315928940e9d4fde8ad2b9e0a5ec720e7e69f23ea2017b95725f260fb095704" || error("golden bytes changed")
                old=deserialize(IOBuffer(bytes)).measured.result
                @testset "Previously successful window retains its golden output" begin
                    @test r.dates==old.dates && r.symbols==old.symbols
                    @test maximum(sum(abs.(r.weights-old.weights);dims=2))<=1e-7
                    @test isapprox(r.ret,old.ret;atol=1e-12,rtol=1e-10)
                    @test r.weights_ew==old.weights_ew && r.locked_days==old.locked_days
                end
                println("golden_weight_L1=",maximum(sum(abs.(r.weights-old.weights);dims=2)),
                    " return_max=",maximum(abs.(r.ret-old.ret)))
            end
            sources==panel_sources() || error("source changed")
            for (p,h) in PANEL_INPUT_HASHES
                panel_digest(joinpath(PANEL_ROOT,p))==h || error("CSV changed during reproduction")
            end
            serialize(output,(;sources,inputs=PANEL_INPUT_HASHES,result=r,phase,
                seconds=observation.time,bytes=observation.bytes,gc=observation.gctime,compile=observation.compile_time))
            println("saved=",basename(output)," sha256=",panel_digest(output)); flush(stdout)
        end
        println("completed without failure")
    finally
        sources==panel_sources() || error("runtime changed during reproduction")
        println("runtime source unchanged DURING command; no input/tolerance/budget overrides in this probe"); flush(stdout)
    end
end
rfp_main()
