# FB_diag.jl — b2b 三分计时诊断（T_main / T_tb；T_total 由 b2b 实测引用）
# 用法: julia FB_diag.jl <t> [main|tb]
#   main: 仅 _cash_kelly_solve 计时（t=332 关键判定）；tb: 追加 tie-break 计时（t=330）
using Serialization, LinearAlgebra, Random
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
using .Gate0BatchCaptureA: locked_wealth_gate0, cash_kelly, cash_kelly_certificate,
    cash_kelly_certified, _cash_kelly_solve, _cash_kelly_canonical_tiebreak

function main()
    t = parse(Int, ARGS[1])
    mode = length(ARGS) > 1 ? ARGS[2] : "tb"
    D = joinpath(@__DIR__, "..", "archive", "evidence",
                 "gate0_multiday_run_20261010", "batch_out")
    d = deserialize(joinpath(D, "batch_t$(t)_prep.bin"))
    xp = deserialize(joinpath(D, "batch_t$(t)_X2.bin"))
    X = xp.X
    S, n = size(X)
    locked = d.locked
    free_pos = [k for k in 1:length(locked) if locked[k] == 0]
    Xf = X[:, free_pos]
    base = locked_wealth_gate0(X, locked)
    budget = 1.0 - sum(locked)
    tol = 1e-8
    println("t=", t, "  S=", S, "  n_free=", length(free_pos),
            "  budget=", budget, "  tol=", tol, "  mode=", mode)
    flush(stdout)
    t0 = time()
    res = try
        _cash_kelly_solve(Xf, base, ones(S), budget, tol)
    catch e
        println("T_main THREW: ", first(sprint(showerror, e), 120))
        nothing
    end
    t_main = time() - t0
    res === nothing && return
    println("T_main = ", round(t_main; digits = 1), "s   status_ok=", res.status_ok,
            "  reason=", res.reason, "  status=", res.status)
    flush(stdout)
    res.status_ok || return
    w = res.w_val
    cert = cash_kelly_certificate(Xf, w[1:n], w[n+1]; base = base, budget = budget)
    println("raw cert: kkt=", cert.kkt_residual, "  gap=", cert.objective_gap)
    flush(stdout)
    mode == "main" && return
    all_cash = all(<(1e-12), view(w, 1:n))
    if all_cash
        println("T_tb = 0s (all-cash skip; tie-break not invoked)")
    else
        t0 = time()
        tb = _cash_kelly_canonical_tiebreak(Xf, base, budget, cert.objective, w,
                                            tol, tol)
        println("T_tb = ", round(time() - t0; digits = 1), "s   tb=",
                tb === nothing ? "nothing" : "ok")
    end
    flush(stdout)
end

main()
