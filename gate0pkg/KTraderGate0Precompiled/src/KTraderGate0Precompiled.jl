# Precompiled (pkgimage-cached) carrier of the UNCHANGED KTraderGate0 source.
# It includes src/gate0/KTraderGate0.jl and market_io.jl verbatim, so the
# numerical code is byte-identical; only per-process JIT is removed. Julia
# records every included file as a precompile dependency: any source edit
# invalidates the cache. A synthetic workload exercises the hot methods.
module KTraderGate0Precompiled
using PrecompileTools, Dates, LinearAlgebra, Random
import Convex, Clarabel  # pkgimage-cache the conic backend used by canonical tie-break
const GATE0_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "src", "gate0"))
include(joinpath(GATE0_SRC, "KTraderGate0.jl"))
Base.include(KTraderGate0, joinpath(GATE0_SRC, "market_io.jl"))
const G = KTraderGate0
@setup_workload begin
    rng = MersenneTwister(19)
    T = 290
    p = exp.(cumsum(0.008 .* randn(rng, T, 2); dims=1))
    dates = [Date(2020,1,1) + Day(t-1) for t in 1:T]
    @compile_workload begin
        mf = G.MarketFacts(dates, ["A","B"], p, p, trues(T, 2))
        el = G.Eligibility(mf)
        cache = G._ForecastCache(mf, el, (1e-6, 2048, 5.0))
        cache.selective = true
        G._historical_forecast(mf, el, 270; preprocessed=cache)
        G._decision_forecast(mf, el, 270)
        G.single_day_decision(mf, el, zeros(2); t=280, forecast_cache=cache,
                              mode=:adaptive, min_scenarios=64, max_scenarios=65536,
                              mu_qmc=true, mu_chisq_qmc=true)
        # Exercise the conic canonical tie-break (duplicate columns => tie).
        g = exp.(0.01 .* randn(rng, 256, 1)) .* 1.0005
        G.cash_kelly(hcat(g, g, exp.(0.01 .* randn(rng, 256, 1))))
    end
end
end
