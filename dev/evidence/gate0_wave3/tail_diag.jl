# 诊断脚本（Wave 3 收尾轮）：DC 形态（P=1+14N）+ 纯噪声 Y 的
# fit_fold_posteriors 二维 quadrature tail 证书失败形态——打印粗扫描
# mode、域边界 8 点 logf、内部 16 点 m_ref、左尾衰减斜率。
module DiagEnv4
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "oof.jl"))
end
using .DiagEnv4: sufficient_stats, lambda_diag, log_evidence, log_prior_d035a,
    fold_grid, fold_stats, train_stats
using Random

SEED = 20261013
N_ct = 3
P_ct = 1 + 14 * N_ct
n_ct = 66
X_ct = hcat(fill(1.0, n_ct),
            randn(MersenneTwister(SEED + 20), n_ct, P_ct - 1))
Y_ct = 0.05 .* randn(MersenneTwister(SEED + 21), n_ct, N_ct)
folds = fold_grid(n_ct; F_folds = 3)

fs = fold_stats(X_ct, Y_ct, folds)
st_train = train_stats(fs.full, fs.folds[1])     # fold 1 的 train 统计
has_a0 = true
logf(u0, up) = log_evidence(st_train, lambda_diag(P_ct, u0, up, has_a0)) +
               log_prior_d035a(u0, up)

# 粗扫描（fit_fold_posteriors 同款；包函数避免 soft-scope）
function coarse_scan(logf)
    grid = range(-12.0, 12.0; length = 21)
    m0, mp, mval = 0.0, 0.0, -Inf
    for u0 in grid, up in grid
        v = logf(u0, up)
        if v > mval
            mval = v; m0 = u0; mp = up
        end
    end
    (m0, mp, mval)
end
m0, mp, mval = coarse_scan(logf)
println("coarse mode = ($m0, $mp), mval = $mval")

u_span = 10.0
u0lo, u0hi = m0 - u_span, m0 + u_span
uplo, uphi = mp - u_span, mp + u_span
init_grid = 4
inner = [(u0lo + i * (u0hi - u0lo) / (init_grid + 1),
          uplo + j * (uphi - uplo) / (init_grid + 1))
         for i in 1:init_grid, j in 1:init_grid]
m_ref = maximum(logf(p[1], p[2]) for p in inner)
println("inner 16-pt m_ref = $m_ref")
println("required: boundary logf <= m_ref + log(exp(-10)) = ", m_ref - 10.0)

bpts = [("corner lo-lo", (u0lo, uplo)), ("corner hi-lo", (u0hi, uplo)),
        ("corner lo-hi", (u0lo, uphi)), ("corner hi-hi", (u0hi, uphi)),
        ("edge u0mid-lo", ((u0lo + u0hi) / 2, uplo)),
        ("edge u0mid-hi", ((u0lo + u0hi) / 2, uphi)),
        ("edge lo-upmid", (u0lo, (uplo + uphi) / 2)),
        ("edge hi-upmid", (u0hi, (uplo + uphi) / 2))]
for (nm, p) in bpts
    println("boundary $nm ($p): logf = ", logf(p[1], p[2]))
end
# 左尾斜率：u₀ → -∞ 的 logf 衰减（理论：(N/2)·u₀ 线性）
for u0v in (m0 - 20.0, m0 - 30.0, m0 - 40.0)
    println("logf(u0=$u0v, up=$mp) = ", logf(u0v, mp))
end
# u₀ 剖面（固定 up=mp）：峰在哪
println("--- u0 profile (up = $mp) ---")
for u0v in range(u0lo, u0hi; length = 11)
    println("  u0 = ", round(u0v, digits=1), " -> logf = ",
            round(logf(u0v, mp), digits=2))
end
