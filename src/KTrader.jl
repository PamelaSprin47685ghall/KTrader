"""
KTrader — Path Kelly: causal price-history model -> posterior-predictive scenarios
-> exact Bayesian Kelly weights -> rebalance executor. See README.md.
"""
module KTrader

using LinearAlgebra, Statistics, Random, Dates
using TimeZones
using PrecompileTools
using Convex, Clarabel
using HTTP, JSON3, CSV

include("data.jl")
include("geometry.jl")
include("response.jl")
include("predict.jl")
include("kelly.jl")
include("backtest.jl")
include("broker.jl")
include("live.jl")

# Precompile workload
@compile_workload begin
    rng = MersenneTwister(0)
    P = exp.(cumsum(0.01 .* randn(rng, 600, 4), dims = 1))
    path_kelly_v1(P; S = 20, rng)
    fit_v1(P)
end

export Bars, download_bars, save_bars, load_bars, load_universe, signal_prices,
       # Geometry Layer (V0.95: Helmert Fixed Gauge)
       TAUS, BANDS, BANDCOL, WARMUP, ruler, center_of_mass, center_of_mass_decomposition,
       helmert_basis, project_helmert_gauge, pairwise_covariance,
       # Response Layer (V0.95: Full Multivariate G with Trace Neutrality & Matrix-Free Matrix-Normal)
       path_basis_1d, build_path_basis, fit_response_operator, predict_modes, ResponseOperator,
       optimize_evidence_alpha,
       # Prediction Layer (V0.95: Out-of-fold residuals & Causal Fractional Mixture)
       V1Model, fit_v1, generate_scenarios_v1, causal_fractional_posterior, DGRID_V1, frac_weights,
       # Kelly Layer
       kelly_weights_v1, path_kelly_v1, path_kelly,
       # Backtest Layer
       backtest_v1, equal_weights_v1, summarize,
       # Execution Layer
       Broker, tradier, rebalance!, tradable, held_weights, target_shares, LiveState, live_step!, settle!, preview_history,
       is_final, settle_due

end
