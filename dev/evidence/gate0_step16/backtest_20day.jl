# 20-day 合成运行（D-089 阶梯——裁决 8 后置满足）
module D20
    using Dates
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "market.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "predictive.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "quadrature.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "driver.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "backtest.jl"))
end
using .D20: MarketFacts, Eligibility, run_gate0_backtest
using Random, Dates

# 合成 fixture：N=4、T=300、drift 信号（同 backtest_tests 形态）
rng = MersenneTwister(2026)
logp = cumsum(0.003 .+ 0.01 .* randn(rng, 300, 4); dims = 1)
close = exp.(logp)
mf = MarketFacts(
    collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(299))),
    ["A$j" for j in 1:4], close, copy(close), trues(300, 4))
el = Eligibility(mf)

# 20 决策日：t ∈ 277..296（marking 至 297）
res = run_gate0_backtest(mf, el; t_start = 277, t_end = 297,
                         seed = 0x0000000000005678,
                         mode = :reference, S_reference = 32, diag_S = 16)
println("20-day backtest OK: days = ", length(res.days))
println("equity[end] = ", res.equity[end])
println("equity finite & positive: ",
        all(x -> isfinite(x) && x > 0, res.equity))
println("benchmark curves finite: ",
        all(isfinite, res.equity_ew_daily) && all(isfinite, res.equity_ew_buyhold))
println("wealth conservation per day: ",
        all(d -> abs(sum(d.decision.w_universe) + d.decision.w_cash - 1.0) < 1e-9,
            res.days))
