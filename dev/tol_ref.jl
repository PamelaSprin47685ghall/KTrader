# =============================================================================
# dev/tol_ref.jl —— D-066 定稿实验的解析基准辅助（由 dev/tol_trace.jl include）
# =============================================================================
# 性质：纯函数辅助（不运行任何命令、不接触 src/test、无 world 表依赖——μ/σ 或
# 行矩阵由调用者传入）。本文件由 dev/tol_trace.jl 启动时 include；不独立运行。
#
# 用途（设计 docs/NUMERICAL_TOLERANCE_FINALIZATION.md §3.2）：
# - A1–A3（known Gaussian 2/3-asset）：Gauss-Hermite 高精度积分上的目标函数 +
#   KKT 方程 Newton 求根 → w*；2D 自洽检查 = 节点 80→120 的 w* L1 差 < 1e-6；
#   3D 只做局部最优性验证（一阶条件残差 < 1e-6）。
# - A4/A5a/A5b（离散行世界）：解析锚（A4 对称角点；A5 平局集描述）——不引入
#   数值求根。
#
# 与 test/gate0/quadrature_tests.jl 的关系（诚实登记）：
# - gh_nodes 与 quadrature_tests.jl 的 gauss_hermite_nodes 数学同式
#   （Golub-Welsch，physicists' Hermite，e^{-x²} 权重；off-diagonal sqrt(i/2)
#   是该文件已修复后的正确形式）。本文件是 dev 侧独立实现：test 文件是 module
#   （含 Test 依赖与全部 @testset），不可被 dev 工具 include；且设计 §3.2 要求
#   80/120 节点参数化 + KKT 求根的高精度基准，与既有 40 节点 + 网格的轻量形态
#   是不同精度的两件工具。二者不可互相冒充：既有基准保留在 test 内不动。
#
# 数值约定：
# - 目标 I(w) = E[log(wc + Σ_j w_j·g_j)]，g_j = exp(μ_j + √2·σ_j·x_j)，x ~ N(0,1) iid；
#   约束参数化 wc = 1 − Σw（cash 为 numeraire，D-068），故
#   wealth = 1 + Σ_j w_j·(g_j − 1)，梯度 ∂I/∂w_j = E[(g_j−1)/wealth]。
# - GH 归一：2D 乘 1/π、3D 乘 1/π^{3/2}（Σww = √π）。
# - 内部解保证：A1/A2/A3 的预注册参数下连续近似 w_j ≈ μ_j/σ_j² 之和 < 1
#   （A1 约 0.234+0.222；A2 约 0.117+0.111；A3 约 0.148+0.184+0.240）；
#   起点取该近似，Newton 步长保护在内部区。
# - 全部函数 fail loudly（不收敛/离开内部/非有限 → throw）；调用者
#   （tol_trace.jl 的 ref/join）负责翻译退出码。
#
# 执行批次用法（示例；本文件交付时未运行）：
#   julia --startup-file=no --project=. dev/tol_trace.jl ref A1 --out <dir>
#   julia --startup-file=no --project=. dev/tol_trace.jl ref A3 --out <dir>
# 自洽失败（2D diff ≥ 1e-6 / 3D 残差 ≥ 1e-6）由 ref 子命令以 rc=40 fail loudly。
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Gauss-Hermite 节点（Golub-Welsch）
# ---------------------------------------------------------------------------

"""gh_nodes(n) -> (x, w)：物理学家' Hermite（∫e^{-x²}f dx）的 n 节点/权重。

节点 = Jacobi 矩阵（off-diagonal sqrt(i/2)）特征值；权重 = √π·(v₁)²。
与 quadrature_tests.jl 的 gauss_hermite_nodes 数学同式（见文件头登记）。"""
function gh_nodes(n::Int)
    n >= 2 || throw(ArgumentError("gh_nodes: n ≥ 2 需要（got $n）"))
    T = zeros(n, n)
    for i in 1:(n - 1)
        T[i, i + 1] = T[i + 1, i] = sqrt(i / 2)
    end
    F = eigen(Symmetric(T))
    (F.values, sqrt(pi) .* (F.vectors[1, :] .^ 2))
end

# ---------------------------------------------------------------------------
# 2. GH 目标 / 梯度 / Hessian（2D、3D）
# ---------------------------------------------------------------------------

function _gh_axis_table(mu::Float64, sigma::Float64, x::Vector{Float64})
    [exp(mu + sqrt(2) * sigma * xi) for xi in x]
end

"""gh_2d_eval(mu, sigma, w; n_gh) -> (; I, g, H)

2D 目标、梯度、Hessian（约束参数化，见文件头）。w 长度 2。"""
function gh_2d_eval(mu::Vector{Float64}, sigma::Vector{Float64},
                    w::Vector{Float64}; n_gh::Int)
    (length(mu) == 2 && length(sigma) == 2 && length(w) == 2) ||
        throw(DimensionMismatch("gh_2d_eval: 需要 2D 参数"))
    x, ww = gh_nodes(n_gh)
    e1 = _gh_axis_table(mu[1], sigma[1], x)
    e2 = _gh_axis_table(mu[2], sigma[2], x)
    I = 0.0
    g1 = 0.0
    g2 = 0.0
    h11 = 0.0
    h12 = 0.0
    h22 = 0.0
    @inbounds for i in 1:n_gh
        a1 = e1[i] - 1.0
        wi = ww[i]
        for j in 1:n_gh
            W = wi * ww[j]
            a2 = e2[j] - 1.0
            wealth = 1.0 + w[1] * a1 + w[2] * a2
            wealth > 0.0 || return (; I = -Inf, g = [NaN, NaN], H = fill(NaN, 2, 2))
            I += W * log(wealth)
            iw = W / wealth
            g1 += iw * a1
            g2 += iw * a2
            iw2 = iw / wealth
            h11 -= iw2 * a1 * a1
            h12 -= iw2 * a1 * a2
            h22 -= iw2 * a2 * a2
        end
    end
    c = 1 / pi
    (; I = c * I, g = c * [g1, g2], H = c * [h11 h12; h12 h22])
end

"""gh_3d_eval(mu, sigma, w; n_gh) -> (; I, g, H)：3D（张量积，设计 §3.2 用 40）。"""
function gh_3d_eval(mu::Vector{Float64}, sigma::Vector{Float64},
                    w::Vector{Float64}; n_gh::Int)
    (length(mu) == 3 && length(sigma) == 3 && length(w) == 3) ||
        throw(DimensionMismatch("gh_3d_eval: 需要 3D 参数"))
    x, ww = gh_nodes(n_gh)
    e1 = _gh_axis_table(mu[1], sigma[1], x)
    e2 = _gh_axis_table(mu[2], sigma[2], x)
    e3 = _gh_axis_table(mu[3], sigma[3], x)
    I = 0.0
    g = zeros(3)
    H = zeros(3, 3)
    @inbounds for i in 1:n_gh
        a1 = e1[i] - 1.0
        wi = ww[i]
        for j in 1:n_gh
            a2 = e2[j] - 1.0
            wij = wi * ww[j]
            for k in 1:n_gh
                W = wij * ww[k]
                a3 = e3[k] - 1.0
                wealth = 1.0 + w[1] * a1 + w[2] * a2 + w[3] * a3
                wealth > 0.0 ||
                    return (; I = -Inf, g = fill(NaN, 3), H = fill(NaN, 3, 3))
                I += W * log(wealth)
                iw = W / wealth
                g[1] += iw * a1
                g[2] += iw * a2
                g[3] += iw * a3
                iw2 = iw / wealth
                # 3×3 外积按分量累加（避免每点分配）
                H[1, 1] -= iw2 * a1 * a1
                H[1, 2] -= iw2 * a1 * a2
                H[1, 3] -= iw2 * a1 * a3
                H[2, 2] -= iw2 * a2 * a2
                H[2, 3] -= iw2 * a2 * a3
                H[3, 3] -= iw2 * a3 * a3
            end
        end
    end
    H[2, 1] = H[1, 2]
    H[3, 1] = H[1, 3]
    H[3, 2] = H[2, 3]
    c = 1 / pi^(3 / 2)
    (; I = c * I, g = c * g, H = c * H)
end

# ---------------------------------------------------------------------------
# 3. KKT 求根（Newton，内部解）
# ---------------------------------------------------------------------------

"""gh_kelly_kkt_root(mu, sigma; n_gh, tol, maxit) -> (; w, wc, I, grad_res, iters)

在 GH 目标上解内部一阶条件 ∂I/∂w = 0（Newton + 内部区步长保护）。
A1–A3 预注册参数保证内部解存在（见文件头）。不收敛 → throw（fail loudly）。"""
function gh_kelly_kkt_root(mu::Vector{Float64}, sigma::Vector{Float64};
                           n_gh::Int = 80, tol::Float64 = 1e-12,
                           maxit::Int = 100)
    N = length(mu)
    N in (2, 3) || throw(ArgumentError("gh_kelly_kkt_root 支持 N=2,3（got $N）"))
    length(sigma) == N || throw(DimensionMismatch("sigma 长度 ≠ μ 长度"))
    w = mu ./ (sigma .^ 2)                    # 连续近似起点
    if !(all(x -> x > 0, w) && sum(w) < 1)    # 防御：起点压回内部区
        w = 0.5 .* w ./ sum(w)
        (all(x -> x > 0, w) && sum(w) < 1) ||
            throw(ErrorException("gh_kelly_kkt_root: 起点无法置入内部区（μ/σ 非法？）"))
    end
    iters = 0
    while iters < maxit
        iters += 1
        r = N == 2 ? gh_2d_eval(mu, sigma, w; n_gh = n_gh) :
                     gh_3d_eval(mu, sigma, w; n_gh = n_gh)
        (isfinite(r.I) && all(isfinite, r.g)) ||
            throw(ErrorException("gh_kelly_kkt_root: 目标求值非有限（wealth 非正）"))
        grad_res = norm(r.g, Inf)
        grad_res <= tol &&
            return (; w = copy(w), wc = 1.0 - sum(w), I = r.I,
                    grad_res = grad_res, iters = iters)
        Δ = -(r.H \ r.g)
        all(isfinite, Δ) ||
            throw(ErrorException("gh_kelly_kkt_root: Newton 步非有限"))
        α = 1.0
        moved = false
        for _ in 1:40
            wtry = w .+ α .* Δ
            if all(x -> x > 0, wtry) && sum(wtry) < 1
                w = wtry
                moved = true
                break
            end
            α *= 0.5
        end
        moved ||
            throw(ErrorException("gh_kelly_kkt_root: Newton 步离开内部区（α 减半 40 次仍失败）"))
    end
    throw(ErrorException("gh_kelly_kkt_root: $maxit 次迭代未达 tol=$tol"))
end

# ---------------------------------------------------------------------------
# 4. 自洽检查入口（A1/A2 2D；A3 3D）
# ---------------------------------------------------------------------------

"""gh_refcheck_2d(mu, sigma; n_a=80, n_b=120, tol=1e-6)

2D 自洽：节点 80→120 的 w* L1 差 < tol（设计 §3.2「差值 < 1e-6 方可使用」）。"""
function gh_refcheck_2d(mu::Vector{Float64}, sigma::Vector{Float64};
                        n_a::Int = 80, n_b::Int = 120, tol::Float64 = 1e-6)
    ra = gh_kelly_kkt_root(mu, sigma; n_gh = n_a)
    rb = gh_kelly_kkt_root(mu, sigma; n_gh = n_b)
    diff = norm(vcat(ra.w, ra.wc) - vcat(rb.w, rb.wc), 1)
    (; ra = ra, rb = rb, diff = diff, consistent = (diff < tol))
end

"""gh_refcheck_3d(mu, sigma; n_gh=40, tol=1e-6)

3D 局部最优性验证（一阶条件残差 < tol；设计 §3.2 对 3D 只要求此）。"""
function gh_refcheck_3d(mu::Vector{Float64}, sigma::Vector{Float64};
                        n_gh::Int = 40, tol::Float64 = 1e-6)
    ra = gh_kelly_kkt_root(mu, sigma; n_gh = n_gh)
    (; ra = ra, consistent = (ra.grad_res < tol))
end

# ---------------------------------------------------------------------------
# 5. 解析锚（A4/A5a/A5b）
# ---------------------------------------------------------------------------

"""analytic_world_ref(tag) -> (; has_wref, w_ref, wc_ref, objective, note)

离散行世界的解析锚（引用 kelly_cash_tests.jl 的既有断言；本函数只复述结构，
不重算数值）：
- A4：D-090(a) 对称角点 w*=(1/3,1/3,1/3)、cash=0；objective = log(61/60)
  （w* 处 wealth = (1.25+0.9+0.9)/3 = 61/60）。
- A5a：D-090(b) 平局射线 {(u,u,u), cash=1−3u}——wealth ≡ 1、objective ≡ 0；
  无唯一 w*（has_wref=false）。
- A5b：D-091 无信号世界——objective ≡ 0、无集中判据 max(w) ≤ 1/3+0.01；
  无唯一 w*（has_wref=false）。"""
function analytic_world_ref(tag::String)
    if tag == "A4"
        return (; has_wref = true, w_ref = [1 / 3, 1 / 3, 1 / 3], wc_ref = 0.0,
                objective = log(61 / 60),
                note = "D-090(a) 对称角点（kelly_cash_tests.jl 的解析断言）；" *
                       "cash 边界是置换对称构造下的数学必然")
    elseif tag == "A5a"
        return (; has_wref = false, w_ref = Float64[], wc_ref = NaN,
                objective = 0.0,
                note = "D-090(b) 平局射线 {(u,u,u), cash=1−3u}：wealth ≡ 1、" *
                       "objective ≡ 0（kelly_cash_tests.jl）；无唯一 w*")
    elseif tag == "A5b"
        return (; has_wref = false, w_ref = Float64[], wc_ref = NaN,
                objective = 0.0,
                note = "D-091 无信号：objective ≡ 0、无集中判据 max(w) ≤ 1/3+0.01" *
                       "（kelly_cash_tests.jl）；无唯一 w*")
    end
    throw(ArgumentError("analytic_world_ref: 仅支持 A4/A5a/A5b（got $tag）"))
end
