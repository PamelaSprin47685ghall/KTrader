using Test, Dates, Random, LinearAlgebra

module CausalForecastEnv
    using Dates
    include(joinpath(@__DIR__, "../../src/gate0/market.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/response.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/driver.jl"))
end
using .CausalForecastEnv

@testset "historical preprocessing cannot see future prices or listings" begin
    rng = MersenneTwister(20261011)
    T, origin = 310, 280
    prices = exp.(cumsum(0.01 .* randn(rng, T, 2); dims=1))
    observed = trues(T, 2)
    observed[1:290, 2] .= false
    prices[1:290, 2] .= NaN
    dates = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 1)+Day(T-1))
    mf = MarketFacts(dates, ["OLD", "IPO"], prices, prices, observed)
    el = Eligibility(mf)
    prefix = CausalForecastEnv._response_prefix(mf, el, origin)
    compact = CausalForecastEnv._response_prefix(mf, el, origin; compact=true)
    cache = CausalForecastEnv._ForecastCache(mf, el, (1e-6, 2048, 5.0))
    prepared = CausalForecastEnv._response_prefix(mf,el,origin;
                     compact=true,prepared=cache)
    @test prepared.act == compact.act
    @test prepared.s1 ≈ compact.s1 atol=1e-12
    @test prepared.Xtr ≈ compact.Xtr atol=1e-12
    @test prepared.Ytr ≈ compact.Ytr atol=1e-12
    @test prepared.xnow ≈ compact.xnow atol=1e-12
    @test compact.act == prefix.act
    @test compact.s1 == prefix.s1
    @test compact.E == prefix.E
    @test compact.Xtr == prefix.mp.X[prefix.rows, :]
    @test compact.Ytr == prefix.mp.Y[prefix.rows, :]
    @test compact.xnow == vec(prefix.mp.X_full[origin, :])
    @test prefix.act == [1]
    @test size(prefix.mp.X_full, 1) == origin
    @test all(<(origin), prefix.rows)

    changed = copy(prices)
    changed[origin+1:end, 1] .*= exp.(range(1.0, 20.0; length=T-origin))
    mf2 = MarketFacts(dates, ["OLD", "IPO"], changed, changed, observed)
    earlier = CausalForecastEnv._response_prefix(mf2, Eligibility(mf2), origin)
    @test earlier.s1 == prefix.s1
    @test earlier.mp.X_full == prefix.mp.X_full
    @test earlier.mp.Y == prefix.mp.Y
    @test CausalForecastEnv._response_prefix(mf2, Eligibility(mf2), T).s1[1] != prefix.s1[1]

    # Full posterior forecast, not a mock estimator or fixed-alpha shortcut.
    f1 = CausalForecastEnv._historical_forecast(mf, el, origin)
    f2 = CausalForecastEnv._historical_forecast(mf2, Eligibility(mf2), origin)
    @test f1 !== nothing
    @test f1.active == [1]
    @test f1.mean_log_return == f2.mean_log_return
    @test all(isfinite, f1.mean_log_return)
    f_cached = CausalForecastEnv._historical_forecast(mf,el,origin;
                                                      preprocessed=cache)
    @test f_cached.active == f1.active
    @test f_cached.mean_log_return ≈ f1.mean_log_return atol=1e-11

    today = CausalForecastEnv._response_prefix(mf, el, T)
    cold = CausalForecastEnv._market_prequential_residuals(mf, el, [origin+1],
                                                         today.act, today.s1)
    cached = CausalForecastEnv._market_prequential_residuals(mf, el, [origin+1],
        today.act, today.s1; forecast_cache=cache)
    replay = CausalForecastEnv._market_prequential_residuals(mf, el, [origin+1],
        today.act, today.s1; forecast_cache=cache)
    @test isequal(cold, cached)
    @test isequal(cached, replay)
    @test isnan(cached[1, 2]) # future IPO must not erase the old asset's surprise
    @test cached[1, 1] ≈ (log(prices[origin+1, 1]/prices[origin, 1]) -
                           f1.mean_log_return[1]) / today.s1[1] atol=1e-12
    @test cache.hits == 1
    @test cache.additions == 1
    CausalForecastEnv._cache_forecast!(cache, origin, f1)
    @test cache.additions == 1 # a resumed current-day fit is not a new origin
    @test_throws ArgumentError single_day_decision(mf2, Eligibility(mf2), zeros(2);
                                                   t=origin, forecast_cache=cache)
end
