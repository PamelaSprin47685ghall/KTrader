# 诊断脚本（Wave 3）：隔离 fit_full_posterior 的 quadrature tail 证书失败形态。
# 构造 posterior_tests.jl testset 7 的同款数据（SEED+6），打印 logf 的
# 峰值参照（粗扫描 mval / adaptive 内部 16 点 m_ref）与域边界 8 点值。
module DiagEnv
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
end
using .DiagEnv: sufficient_stats, log_evidence, log_prior_d035a, lambda_diag
using Random

rng = MersenneTwister(20261009 + 6)
N, P, n = 2, 3, 30
X = randn(rng, n, P)
G = 0.5 .* randn(rng, N, P)
Y = X * G' .+ 0.3 .* randn(rng, n, N)
st = sufficient_stats(X, Y)
has_a0 = (P == 1 + 14 * N)
logf(u0, up) = log_evidence(st, lambda_diag(P, u0, up, has_a0)) + log_prior_d035a(u0, up)

# fit_full_posterior 的粗扫描（同款：21×21 on [-12,12]；包进函数避免
# 脚本顶层 soft-scope 赋值歧义）
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
println("coarse mode = ($m0, $mp), mval = ", mval)

u_span = 10.0
u0lo, u0hi = m0 - u_span, m0 + u_span
uplo, uphi = mp - u_span, mp + u_span
println("domain = ($u0lo, $u0hi) x ($uplo, $uphi)")

# adaptive_quadrature_2d 的内部 m_ref（init_grid=4 的 16 点）
init_grid = 4
inner = [(u0lo + i * (u0hi - u0lo) / (init_grid + 1),
          uplo + j * (uphi - uplo) / (init_grid + 1))
         for i in 1:init_grid, j in 1:init_grid]
m_ref = maximum(logf(p[1], p[2]) for p in inner)
println("inner 16-pt m_ref = ", m_ref)

# 边界 8 点
bpts = [(u0lo, uplo), (u0hi, uplo), (u0lo, uphi), (u0hi, uphi),
        ((u0lo + u0hi) / 2, uplo), ((u0lo + u0hi) / 2, uphi),
        (u0lo, (uplo + uphi) / 2), (u0hi, (uplo + uphi) / 2)]
for p in bpts
    println("boundary ($p): logf = ", logf(p[1], p[2]))
end
# 域中心（mode）值与真峰细扫
println("logf(mode) = ", logf(m0, mp))
fine = maximum(logf(u0, up) for u0 in range(m0 - 2, m0 + 2; length = 41),
                                     up in range(mp - 2, mp + 2; length = 41))
println("fine-grid peak near mode = ", fine)
