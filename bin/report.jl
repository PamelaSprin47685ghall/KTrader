# Usage: julia --project=. bin/report.jl     (reads data/, writes nothing)
# What the model believes at the last decision date, and the weights it implies.
# The V1.0 response posterior is a full Matrix-Normal operator, not per-(band, mode)
# scalars; the legacy (rho, theta) table is not its statistic. This reports the
# actual V1.0 quantities: EB alphas, trace diagnostics, predictive moments, and
# the Kelly weights they imply.
using KTrader, Dates, Random, Statistics, Printf
b = load_bars(joinpath(@__DIR__, "..", "data"))
S = parse(Int, get(ENV, "SCENARIOS", "1000"))
adaptive = get(ENV, "SCENARIOS", "") == "adaptive"

signal = signal_prices(b)
model = fit_v1(signal)
free = b.bar[end, :] .& (trues(length(b.symbols)))
tradable = b.bar[end, :]

println("decision date ", b.dates[end], ": ", length(model.active_indices), " active of ",
        length(b.symbols), " assets, ", count(tradable), " tradable")

resp = model.resp
@printf("EB alpha (macro)  = %.4g\n", resp.alpha_macro)
@printf("EB alpha (relative, trace-neutral support) = %.4g\n", resp.alpha_rel)
@printf("trace(A) = %.3g   trace(B) = %.3g   (both ≡ 0 on the parameter support)\n",
        resp.trace_real, resp.trace_imag)

moments = model.pred_moments
@printf("\npredictive moments at decision time:\n")
@printf("  macro mean mu_m = %.4g, macro sd = %.4g\n", moments.mu_m, sqrt(moments.var_m))
@printf("  relative covariance rank = %d (zero-sum subspace), Frobenius = %.3g\n",
        size(moments.L_rel, 1), norm(moments.L_rel))
@printf("  fractional d-posterior: ")
for (g, d) in enumerate(DGRID_V1)
    @printf("d=%.2f:%.2f  ", d, model.d_posterior[g])
end
println()
@printf("  v_forecasts (per d): min %.3g, max %.3g; v_bootstrap = %.3g\n",
        minimum(model.v_forecasts), maximum(model.v_forecasts), model.v_bootstrap)

if adaptive
    result = KTrader.adaptive_scenario_weights(model, tradable; rng = MersenneTwister(1))
    w = result.weights
    @printf("\nadaptive quadrature converged at S = %d (gap %.3g)\n", result.S, result.objective_gap)
else
    X = generate_scenarios_v1(model; S, rng = MersenneTwister(1))
    w = KTrader.scenario_weights(X, model.active_indices, tradable)
    cert = KTrader.kelly_certificate(view(X, :, [j for j in model.active_indices if tradable[j]]),
        w[[j for j in model.active_indices if tradable[j]]])
    @printf("\nKelly certificate: feasibility %.3g, KKT residual %.3g, objective gap %.3g\n",
        cert.feasibility, cert.kkt_residual, cert.objective_gap)
end

println("\nKelly weights (non-zero):")
for i in sortperm(w; rev = true)
    w[i] > 1e-4 && @printf("  %-8s %6.2f%%\n", b.symbols[i], 100w[i])
end
@printf("effective number of holdings 1/Σw² = %.2f   (Σw = %.6f)\n", 1 / sum(abs2, w), sum(w))
