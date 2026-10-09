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
using FFTW, Distributions

include("data.jl")
include("geometry.jl")
include("numerics.jl")
include("response.jl")
include("residual_oracle.jl")
include("prepare.jl")
include("predict.jl")
include("incremental.jl")
include("kelly.jl")
include("backtest.jl")
include("broker.jl")
include("live.jl")

function __init__()
    # Native FFTW plan pointers cannot survive Julia's precompile serialization.
    empty!(FRACTIONAL_FFT_POOL)
end

# Precompile workload
@compile_workload begin
    rng = MersenneTwister(0)
    P = exp.(cumsum(0.01 .* randn(rng, 600, 4), dims = 1))
    path_kelly_v1(P; S = 20, rng)
    fit_v1(P)
end

export Bars, download_bars, save_bars, load_bars, load_universe, signal_prices,
       # Geometry Layer (V0.95: Helmert Fixed Gauge)
       TAUS, BANDS, BANDCOL, WARMUP, ruler, center_of_mass, center_of_mass_decomposition, pairwise_covariance,
       PrefixRulerStats, build_prefix_ruler_stats, ruler_from_stats,
       # Response Layer (V1.0: Exact Gaussian Conditioning & Matrix-Normal EB)
       path_basis_1d, compute_s_perp, build_X_rel_stacked, compute_B_rel_at_t,
       fit_response_operator, ResponseOperator,
       predictive_moments, condition_trace_neutrality, optimize_matrix_normal_eb,
       # Prediction Layer (V0.95: Out-of-fold residuals & Causal Fractional Mixture)
       V1Model, fit_v1, generate_scenarios_v1, causal_fractional_posterior, DGRID_V1, frac_weights,
       active_universe_indices,
       # M1 PreparedProblem boundary: typed prepare + single solve owner
       PreparedProblem, FoldStatistics, MacroStatistics, MacroFoldBlock,
       prepare_reference, prepare_incremental, solve,
       # Lazy OOF Residual Oracle (production path) & dense test reference
       ResidualOracle, dense_oof_residuals,
       ExactInferenceState, initialize_inference, advance_exact!, solve_current!,
       inference_checkpoint, inference_diagnostics,
       # Kelly Layer
       kelly_weights_v1, fast_kelly_solver, clarabel_kelly_solver, path_kelly_v1, path_kelly,
       # Backtest Layer
       backtest_v1, equal_weights_v1, summarize,
       # Execution Layer
       Broker, tradier, rebalance!, tradable, held_weights, target_shares, LiveState, live_step!, settle!, preview_history,
       is_final, settle_due

end
