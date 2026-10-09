# 诊断（最后一轮）：段 6（t=327..336）逐日定位 Clarabel SLOW_PROGRESS
# 的具体决策日与该日 Kelly 问题形态（M、N）。
module D6
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
end
using .D6: MarketFacts, Eligibility, single_day_decision
using Random, Dates, LinearAlgebra

rng = MersenneTwister(2026)
logp = cumsum(0.003 .+ 0.01 .* randn(rng, 337, 4); dims = 1)
close = exp.(logp)
mf = MarketFacts(collect(Date(2024,1,1):Day(1):(Date(2024,1,1)+Day(336))),
                 ["A$j" for j in 1:4], close, copy(close), trues(337, 4))
el = Eligibility(mf)
function run_days()
    held = parse.(Float64, split(readline(joinpath(@__DIR__, "held60_5.txt")), ','))
    println("held0 = ", held)
    for t in 327:336
        try
            dec = single_day_decision(mf, el, held; t = t,
                                      seed = 0x0000000000005678,
                                      mode = :reference, S_reference = 32,
                                      diag_S = 16, kelly_tol = 5e-3)
            println("t=", t, "  OK  mode=", dec.mode,
                    "  M=", dec.M, "  N_R=", length(dec.R_t))
            g = [mf.adj[min(t+1,337), j] / mf.adj[t, j] for j in 1:4]
            wf = dot(dec.w_universe, g) + dec.w_cash
            held = [dec.w_universe[j] * g[j] / wf for j in 1:4]
        catch e
            println("t=", t, "  RED: ", sprint(showerror, e)[1:min(end,120)])
            break
        end
    end
end
run_days()
