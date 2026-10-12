# Regression tests for the lock-free (independent decision) contracts:
# decision-path forecast identity, shard partition, durable checkpoints,
# replay refusals. Synthetic but full-model (no mocks of the posterior/Kelly).
using Test,Dates,Random,LinearAlgebra,Serialization
module LockFreeContractEnv
    using Dates,LinearAlgebra,Random
    include(joinpath(@__DIR__,"../../src/gate0/market.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/oof.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/predictive.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/quadrature.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/driver.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/backtest.jl"))
end
const L=LockFreeContractEnv

function synth(T=290;seed=19)
    rng=MersenneTwister(seed)
    p=exp.(cumsum(0.008 .* randn(rng,T,2);dims=1))
    dates=[Date(2020,1,1)+Day(t-1) for t in 1:T]
    mf=L.MarketFacts(dates,["A","B"],p,p,trues(T,2))
    mf,L.Eligibility(mf),p,dates
end

@testset "shard days form an exact partition of the decision window" begin
    for (a,b) in ((13808,14310),(5,6),(10,17)), n in 1:8
        all=reduce(vcat,[L._lockfree_shard_days(a,b,s,n) for s in 1:n])
        @test sort(all)==collect(a:b-1)
        @test length(unique(all))==length(all)
    end
    @test_throws ArgumentError L._lockfree_shard_days(10,10,1,1)
    @test_throws ArgumentError L._lockfree_shard_days(1,9,5,4)
end

@testset "durable checkpoint: atomic replace, no temp residue, fsync path" begin
    dir=mktempdir()
    path=joinpath(dir,"sub","ck.jls")
    L._write_backtest_checkpoint(path,(;v=1,x=[1.0,2.0]))
    L._write_backtest_checkpoint(path,(;v=2,x=[3.0]))
    @test deserialize(path)==(;v=2,x=[3.0])
    @test readdir(dirname(path))==["ck.jls"]
end

@testset "decision-path forecast is bitwise the forecast the driver caches" begin
    mf,el,p,dates=synth()
    t=280
    cache=L._ForecastCache(mf,el,(1e-6,2048,5.0))
    cache.selective=true
    dec=L.single_day_decision(mf,el,zeros(2);t=t,forecast_cache=cache,
                              min_scenarios=64,max_scenarios=65536)
    @test dec.mode==:adaptive
    f=L._decision_forecast(mf,el,t)
    @test cache.entries[t].active==f.active
    @test cache.entries[t].mean_log_return==f.mean_log_return   # bitwise
    # causality: prices after t cannot change the decision-path forecast
    q=copy(p); q[t+1:end,1].*=3.0
    mf2=L.MarketFacts(dates,["A","B"],q,q,trues(size(q)...))
    @test L._decision_forecast(mf2,L.Eligibility(mf2),t).mean_log_return==f.mean_log_return
    # same quantity as the historical (mean_at) path up to roundoff only
    h=L._historical_forecast(mf,el,t)
    @test maximum(abs,h.mean_log_return.-f.mean_log_return)<=1e-12*max(1.0,maximum(abs,f.mean_log_return))
end

@testset "replay refuses a mislabelled (out-of-order) decision and locked risk" begin
    mf,el,_,_=synth()
    c=(;A=0.0,B=0.0,B_audit=0.0,
        C=(;feasibility=0.0,kkt_residual=0.0,objective_gap=0.0,objective=0.0,dual=1.0),
        converged=true,kelly_objective=0.0)
    mk(t)=L.Gate0DayDecision(:adaptive,t,[1,2],[1,2],[0.3,0.3],[0.3,0.3],0.4,0.0,1.0,c,
           1024,L._no_varsplit(),L._no_concentration(2),Int[],
           (;reason=:computed,error_text="",dropped_free_universe=Int[],certificate_path=:weight_stable))
    decs=[mk(t) for t in 270:273]
    @test L.replay_independent_gate0(mf,el,decs;t_start=270,t_end=274).equity[end]>0
    swapped=[decs[2],decs[1],decs[3],decs[4]]
    @test_throws ErrorException L.replay_independent_gate0(mf,el,swapped;t_start=270,t_end=274)
    red=(;c...,converged=false)
    bad=[decs[1:3];L.Gate0DayDecision(:adaptive,273,[1,2],[1,2],[0.3,0.3],[0.3,0.3],0.4,0.0,1.0,red,
           1024,L._no_varsplit(),L._no_concentration(2),Int[],
           (;reason=:computed,error_text="",dropped_free_universe=Int[],certificate_path=:weight_stable))]
    @test_throws ErrorException L.replay_independent_gate0(mf,el,bad;t_start=270,t_end=274)
    T=size(mf.observed,1)
    locked=L.Eligibility(el.model_admitted,el.trade_eligible,
        BitMatrix([i==272 && j==2 ? false : el.executable[i,j] for i in 1:T,j in 1:2]))
    @test_throws ErrorException L._assert_lockfree_independent_window(mf,locked,270,274,zeros(2))
end
