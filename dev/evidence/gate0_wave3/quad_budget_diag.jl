# 区分实验（Wave 3）：一维 quadrature 预算耗尽是 (a) max_cells 不够还是
# (b) 顶点极差估计器结构性过保守。同款数据直接调 _adaptive_quadrature_1d，
# 扫 max_cells；并对照「中点-梯形差」估计器（Simpson 型，对光滑函数紧）。
module DiagEnv2
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
end
using .DiagEnv2: sufficient_stats, log_evidence, log_prior_d035a, lambda_diag,
    _adaptive_quadrature_1d
using Random

rng = MersenneTwister(20261009 + 6)
N, P, n = 2, 3, 30
X = randn(rng, n, P)
G = 0.5 .* randn(rng, N, P)
Y = X * G' .+ 0.3 .* randn(rng, n, N)
st = sufficient_stats(X, Y)
logf1(up) = log_evidence(st, lambda_diag(P, 0.0, up, false)) +
            log_prior_d035a(0.0, up)
# 粗扫描 mode（一维：up 网格；包函数避免 soft-scope）
function coarse1d(logf1)
    mp, mval = 0.0, -Inf
    for up in range(-12.0, 12.0; length = 21)
        v = logf1(up)
        if v > mval
            mval = v; mp = up
        end
    end
    (mp, mval)
end
mp, mval = coarse1d(logf1)
println("1d coarse mode up = $mp, mval = $mval")

for mc in (2048, 8192, 32768)
    r = try
        q = _adaptive_quadrature_1d(logf1, (mp - 10.0, mp + 10.0);
                                    tol = 1e-4, max_cells = mc,
                                    tail_rel = exp(-10.0), m_ref = mval)
        "OK: n_cells=$(q.n_cells), rel=$(q.rel_err)"
    catch e
        "FAIL: $e"
    end
    println("max_cells=$mc -> ", r)
end
