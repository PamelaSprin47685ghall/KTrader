# 诊断（Step 16）：backtest_tests testset (4) 的 locked fixture（mf_l，
# seed=34）fold posterior 的 tail 证书失败形态。复刻 driver 链到
# fit_fold_posteriors，打印 fold 1 的粗扫描 mode、域边界 8 点 logf、
# 内部 16 点 m_ref 与判据要求值。
module DBt
    using Dates
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "market.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "oof.jl"))
end
using .DBt: MarketFacts, Eligibility, signal_prices, domain_mode_basis,
    build_mode_problem, WARMUP, sufficient_stats, lambda_diag, log_evidence,
    log_prior_d035a, fold_grid, fold_stats, train_stats, ruler
using Random, LinearAlgebra, Dates, Statistics

# 粗扫描（供下方循环使用）
function coarse_scan(logf)
    grid = range(-12.0, 12.0; length = 21)
    m0, mp2, mval = 0.0, 0.0, -Inf
    for u0 in grid, up in grid
        v = logf(u0, up)
        if v > mval
            mval = v; m0 = u0; mp2 = up
        end
    end
    (m0, mp2, mval)
end

# 复刻 _mk_mf（drift=0.003——当前测试形态）
function mk_mf(N, T, seed; drift = 0.003)
    rng = MersenneTwister(seed)
    logp = cumsum(drift .+ 0.01 .* randn(rng, T, N); dims = 1)
    MarketFacts(collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T - 1))),
                ["A$j" for j in 1:N], exp.(logp), exp.(logp), trues(T, N))
end

# 复刻 driver 步骤 1-4（mf_l: N=3, T=280, seed=34）——多日：t=271/272
mf = mk_mf(3, 280, 34)
E = domain_mode_basis(3)
for t in (271, 272)
    signal = signal_prices(mf)[:, [1, 2, 3]]
    x_log = log.(signal)
    f_first = [findfirst(view(mf.observed, 1:t, j)) for j in 1:3]
    r = Matrix{Float64}(undef, t, 3)
    r[1, :] .= NaN
    r[2:t, :] .= x_log[2:t, :] .- x_log[1:t-1, :]
    obs_ret = isfinite.(r)
    s1 = vec(ruler(x_log, Int[i for i in f_first])[:, 1])
    mp = build_mode_problem(r, Matrix{Bool}(obs_ret), s1, E; dc = true)
    rows = [s for s in WARMUP:(t-1) if any(view(obs_ret, s + 1, :))]
    X_tr = mp.X[rows, :]
    Y_tr = mp.Y[rows, :]
    println("=== t = ", t, "  rows = ", length(rows))
    folds = fold_grid(length(rows); F_folds = 3)
    fs = fold_stats(X_tr, Y_tr, folds)
    for f in 1:3
        st_train = train_stats(fs.full, fs.folds[f])
        logf(u0, up) = log_evidence(st_train, lambda_diag(43, u0, up, true)) +
                       log_prior_d035a(u0, up)
        local m0, mp2, mval
        m0, mp2, mval = coarse_scan(logf)
        local u_span, u0lo, u0hi, uplo, uphi
        u_span = 10.0
        u0lo, u0hi = m0 - u_span, m0 + u_span
        uplo, uphi = mp2 - u_span, mp2 + u_span
        inner = [(u0lo + i * (u0hi - u0lo) / 5, uplo + j * (uphi - uplo) / 5)
                 for i in 1:4, j in 1:4]
        m_ref = maximum(logf(p[1], p[2]) for p in inner)
        area = (u0hi - u0lo) * (uphi - uplo)
        bpts = [(u0lo, uplo), (u0hi, uplo), (u0lo, uphi), (u0hi, uphi),
                ((u0lo+u0hi)/2, uplo), ((u0lo+u0hi)/2, uphi),
                (u0lo, (uplo+uphi)/2), (u0hi, (uplo+uphi)/2)]
        plateau = maximum(logf(p[1], p[2]) for p in bpts)
        gap = m_ref - plateau
        tail_rel = 10 * exp(-u_span)
        old_ok = all(logf(p[1], p[2]) <= m_ref + log(tail_rel) + log(area)
                     for p in bpts)
        println("  fold $f: gap = ", round(gap, digits=2),
                "  old-crit ", old_ok ? "[OK]" : "[RED]",
                "  mode = ($m0, $mp2)")
        if !old_ok
            for p in bpts
                v = logf(p[1], p[2])
                println("    b ", p, " -> ", round(v, digits=2))
            end
        end
    end
end
