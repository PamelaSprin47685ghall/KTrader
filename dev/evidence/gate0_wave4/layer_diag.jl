# 诊断（Wave 4）：testset 6 的 history 实际形态——A/B 逐轮值与符号。
module DQ
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "quadrature.jl"))
end
using .DQ: SobolOwenRule, rule_points, RQMCScenarioSource,
    adaptive_scenario_kelly
using Random, Distributions, LinearAlgebra

MU = [0.0015, 0.0008]
SIG = [0.08, 0.06]
function gen(M::Int, rule::SobolOwenRule, mu_rng::Random.AbstractRNG)
    pts = rule_points(rule, M)
    g = Matrix{Float64}(undef, M, 2)
    for s in 1:M, j in 1:2
        g[s, j] = exp(MU[j] + SIG[j] * quantile(Normal(), pts[s, j]))
    end
    g
end
src = RQMCScenarioSource(gen, 2, 2)
r = adaptive_scenario_kelly(src; rule_seed = UInt64(0x0000BEEF),
                            min_scenarios = 8, max_scenarios = 4096,
                            weight_tol = 1e-2, utility_tol = 1e-7)
println("converged M = ", r.M, ", iterations = ", r.iterations)
println("w_risky = ", r.w_risky, "  w_cash = ", r.w_cash)
for (i, h) in enumerate(r.history)
    println("round $i: M=$(h.M)→$(h.M2)  A=$(h.A)  B=$(h.B)  (B>0: $(h.B > 0))")
end
# 逐层 w 值（手动复刻 solve_layer）
using .DQ: locked_wealth_gate0, cash_kelly
opt_rule = SobolOwenRule(2, UInt64(0x0000BEEF))
for M in (8, 16, 32)
    X = gen(M, opt_rule, MersenneTwister(0))
    w, wc, _ = cash_kelly(X; base = zeros(M), budget = 1.0, tol = 1e-8)
    println("M=$M: w = ", w, "  wc = ", wc,
            "  E[gross col1] = ", round(mean(X[:, 1]), digits = 6),
            "  E[gross col2] = ", round(mean(X[:, 2]), digits = 6))
end
