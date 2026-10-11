# S4_finish_diag.jl — finish 组成诊断（t=338）：分步计时
# 用法: julia S4_finish_diag.jl <step>   step ∈ raw|scaled|tb
using Serialization, LinearAlgebra
module Gate0BatchCaptureA
    using Dates
    include(joinpath(@__DIR__, "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "predictive.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "src", "gate0", "quadrature.jl"))
end
using .Gate0BatchCaptureA: cash_kelly_certificate, cash_kelly_certified,
    _cash_kelly_newton_polish, _cash_kelly_solve, _cash_kelly_canonical_tiebreak

function main()
    step = ARGS[1]
    D = joinpath(@__DIR__, "..", "archive", "evidence",
                 "gate0_multiday_run_20261010", "batch_out")
    xp = deserialize(joinpath(D, "batch_t338_X2.bin"))
    prep = deserialize(joinpath(D, "batch_t338_prep.bin"))
    pr = deserialize(joinpath(D, "batch_t338_pre2.bin"))
    Xf = xp.X[:, pr.free_pos]
    b = pr.b
    budget = pr.budget
    tol = pr.tol
    cs = pr.cs
    w = pr.res.w_val
    println("t=338  X_free=", size(Xf), "  budget=", budget, "  cs_len=", length(cs),
            "  res.reason=", pr.res.reason, "  res.status=", pr.res.status)
    flush(stdout)
    if step == "raw"
        t0 = time()
        wp = _cash_kelly_newton_polish(Xf, b, w, budget)
        dt = time() - t0
        println("raw_polish = ", round(dt; digits = 1), "s  (200 iters)")
        if wp !== nothing
            cp = cash_kelly_certificate(Xf, wp[1:1], wp[2]; base = b, budget = budget)
            println("  polished cert: kkt=", cp.kkt_residual, " gap=", cp.objective_gap)
        end
    elseif step == "scaled"
        t0 = time()
        res2 = _cash_kelly_solve(Xf ./ reshape(cs, :, 1), b ./ cs, 1.0 ./ cs,
                                 budget, tol)
        dt = time() - t0
        println("scaled_solve = ", round(dt; digits = 1), "s  status_ok=", res2.status_ok,
                " reason=", res2.reason)
    elseif step == "tb"
        cert0 = cash_kelly_certificate(Xf, w[1:1], w[2]; base = b, budget = budget)
        t0 = time()
        tb = _cash_kelly_canonical_tiebreak(Xf, b, budget, cert0.objective, w,
                                            pr.tie_eps, tol)
        dt = time() - t0
        println("tie_break = ", round(dt; digits = 1), "s  tb=",
                tb === nothing ? "nothing" : "ok")
    end
    flush(stdout)
end

main()
