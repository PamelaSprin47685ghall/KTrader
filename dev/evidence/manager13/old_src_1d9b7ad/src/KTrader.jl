"""
KTrader — Path Kelly: causal price-history model -> posterior-predictive scenarios
-> exact Bayesian Kelly weights -> rebalance executor. See README.md.
"""
module KTrader

using LinearAlgebra, Statistics, Random, Dates
using TimeZones
using PrecompileTools
using DSP
using Convex, Clarabel
using HTTP, JSON3, CSV

include("data.jl")
include("model.jl")
include("modecov.jl")
include("backtest.jl")
include("broker.jl")
include("live.jl")

# Compile the whole decision path at package-precompile time, so a run starts hot.
@compile_workload begin
    rng = MersenneTwister(0)
    P = exp.(cumsum(0.01 .* randn(rng, 900, 4), dims = 1))
    path_kelly(P; S = 20, rng)
    path_kelly(P; S = 20, rng, response = false)
    theta_posterior(fit_response(P); draws = 50)
    path_kelly(P; S = 20, rng, tradable = [true, true, false, true], held = [0.1, 0.2, 0.3, 0.4])
    P[1:100, 4] .= NaN                                   # ragged panel: late-listed asset
    path_kelly(P; S = 20, rng, volmodel = false)
end

export Bars, download_bars, save_bars, load_bars, load_universe,
       path_kelly, fit_response, theta_posterior, conditional_mean, predict,
       predictive_log_returns, kelly_weights, eligible, firstrows, scenarios, allocate, frac_weights, DGRID, is_final, settle_due,
       mode_vol_model, draw, cond_cov,
       backtest, equal_weights, summarize,
       Broker, tradier, rebalance!, tradable, held_weights, target_shares, LiveState, live_step!, settle!, preview_history

end
