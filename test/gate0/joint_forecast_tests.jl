using Test, Dates, Random
module JointForecastEnv
    using Dates
    include(joinpath(@__DIR__,"../../src/gate0/market.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/driver.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/backtest.jl"))
end
using .JointForecastEnv

@testset "joint-only forecasts match every posterior residual used by the risk domain" begin
    rng=MersenneTwister(20261012)
    T=306
    p=exp.(cumsum(0.007 .* randn(rng,T,2);dims=1))
    obs=trues(T,2)
    obs[1:279,2] .= false
    p[1:279,2] .= NaN
    obs[295,1] = false
    p[295,1] = p[294,1]
    dates=[Date(2020,1,1)+Day(t-1) for t in 1:T]
    mf=MarketFacts(dates,["OLD","IPO"],p,p,obs)
    el=Eligibility(mf)
    t=305
    prep=JointForecastEnv._response_prefix(mf,el,t)
    ids=[270,278,280,281,282,283,287,292,295,296,298,305]
    reference=JointForecastEnv._market_prequential_residuals(
        mf,el,ids,prep.act,prep.s1)
    cache=JointForecastEnv._ForecastCache(mf,el,(1e-6,2048,5.0))
    selected=JointForecastEnv._market_prequential_residuals(
        mf,el,ids,prep.act,prep.s1;
        focus_risk=[1,2],forecast_cache=cache)
    for (k,u) in enumerate(ids)
        eligible=all(j -> mf.observed[u-1,j] && mf.observed[u,j] &&
                          el.model_admitted[u-1,j],1:2)
        if eligible
            @test isequal(selected[k,:],reference[k,:])
        else
            @test all(isnan,selected[k,:])
            @test !haskey(cache.entries,u-1)
        end
    end
    raw_ids=collect(257:305)
    raw_masks=[BitVector([obs[u-1,j] && obs[u,j] for j in 1:2])
               for u in raw_ids]
    floor,peeled=JointForecastEnv._forecast_risk_floor(
        raw_ids,raw_masks,[1,2],Int[],[1,2],282,el,[1,2])
    @test floor == [1]
    @test peeled == [2]
end
