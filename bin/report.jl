# Usage: julia --project=. bin/report.jl     (reads data/, writes nothing)
# What the model believes at the last decision date: the posterior over (ρ, θ) of every (band, mode)
# sorted by evidence (α ascending: most evidence first; α → 1e7 means ρ → 0, no evidence), then the Kelly
# weights those beliefs imply and the effective number of holdings 1/Σw².
using KTrader, Dates, Random, Statistics, Printf
b = load_bars(joinpath(@__DIR__, "..", "data"))
S = parse(Int, get(ENV, "SCENARIOS", "1000"))
mask, r0 = eligible(b.adj)
any(mask) || error("no asset has enough history")
free = mask .& b.bar[end, :]
fit = fit_response(b.adj[r0:end, mask])

println("decision date ", b.dates[end], ": ", count(mask), " eligible of ", length(b.symbols), " assets, ", count(free),
        " tradable, window ", b.dates[r0], " … ", b.dates[end], " (", fit.n, " fitted rows), S = ", S)
@printf("\n%4s %2s %5s %7s %7s %7s %7s %5s %5s %8s %8s %10s\n",
        "τ", "k", "macro", "ρ5", "ρ50", "ρ95", "θmean", "R1", "R2", "p_revert", "p_right", "alpha")
for m in sort(theta_posterior(fit); by = m -> m.alpha)
    @printf("%4d %2d %5s %7.3f %7.3f %7.3f %7.2f %5.2f %5.2f %8.2f %8.2f %10.3g\n",
            m.τ, m.k, m.macro_ ? "yes" : "", m.ρ5, m.ρ50, m.ρ95, m.θmean, m.R1, m.R2, m.p_revert, m.p_right, m.alpha)
end

w = allocate(exp.(predict(fit; S, rng = MersenneTwister(1))), mask, free)
println("\nKelly weights (non-zero):")
for i in sortperm(w; rev = true)
    w[i] > 1e-4 && @printf("  %-8s %6.2f%%\n", b.symbols[i], 100w[i])
end
@printf("effective number of holdings 1/Σw² = %.2f   (Σw = %.6f)\n", 1 / sum(abs2, w), sum(w))
