# =============================================================================
# KTraderGate0 OOF full-mode folds — 施工图 Step 9
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付。include 顺序：geometry.jl → modes.jl → response.jl →
# posterior.jl → 本文件（本文件消费 posterior.jl 的同作用域件——
# SufficientStats/sufficient_stats/lambda_diag/log_evidence/log_prior_d035a/
# adaptive_quadrature_2d/_fit_node/ResponsePosterior/ProprietyCertificate；
# `_fit_node` 未 export 但同 module include 链下可见）。
#
# 规范来源：
# - SPEC §28-§29（fold 规范：contiguous folds、train = full − fold 的充分
#   统计差分、每 fold 独立超参）；
# - RP §9（D-043：fold 独立 quadrature、held-out 行不进 fold 证据——
#   差分统计的构造性推论；D-041 期望语义：OOF 预测用 fold posterior 的
#   完整后验均值含 α 积分；D-044：OOF 不是模型选择工具——不得据 OOF
#   Sharpe 选 bands/priors/d/universe）；
# - 施工图 Step 9：dense residual reference 先实现（lazy oracle 归后续
#   性能轮）。
#
# 结构说明（如实声明）：本文件的 posterior 装配段（粗扫描 + quadrature +
# 节点装配）与 posterior.jl 的 `fit_full_posterior` 生产路径段**同构**——
# 单 owner 归属 posterior.jl 的数学；oof.jl 持有 fold 差分编排。集成轮
# 可重构为共享内部函数（届时对 posterior.jl 的修改需另行授权——本步
# 硬约束不改它）。
#
# 无观察行语义（钉死）：mode 层的 X/Y 是 mode 坐标（mode_field 的零嵌入
# 语义：|O_s|=0 时 y=0——不携带 mask）；「无观察行不进统计」的行集判定
# 归 prepare/consumer 层，经 `fold_grid(rows)` 的行集入口表达（调用方
# 从 rows 剔除）。oof 层不自行判定观察性。
#
# warm-start 语义（D-043，钉死）：昨日**同 fold** 的 quadrature 网格可作
# 初始域收窄（严格过去信息）；本 reference **不使用** warm start（D-083
# reference 级每 fold 独立全跑可接受），集成/增量轮可按此语义接入。

export fold_grid, fold_stats, train_stats, fit_fold_posteriors,
       oof_residual_rows_full

# ---------------------------------------------------------------------------
# contiguous fold 划分（SPEC §28）
# ---------------------------------------------------------------------------

"""
    fold_grid(rows; F_folds = 3) -> Vector{Vector{Int}}
    fold_grid(n::Int; F_folds = 3)

contiguous fold 划分（SPEC §28）：`rows`（行索引向量，默认 `1:n`）按
顺序切成 `F_folds` 个连续段——`q, r = divrem(|rows|, F)`，前 `r` 个
fold 各 `q+1` 行、其余 `q` 行；拼接恰覆盖 `rows`（无遗漏/重复）。

**行身份语义（docstring 钉死）**：行索引 s 对应全局日 u_s（ResidualOracle
的 ts_total 契约——行 s ↔ feature 时刻 s、目标 s+1；本层持有的是
`build_mode_problem` 配对后的行集）。

无观察行：由调用方从 `rows` 剔除（行集入口——见文件头「无观察行语义」）。

fail-loudly：`F_folds < 2` 或 `F_folds > length(rows)` 抛 `ArgumentError`。
"""
function fold_grid(rows::AbstractVector{Int}; F_folds::Int = 3)
    n = length(rows)
    (F_folds >= 2 && F_folds <= n) ||
        throw(ArgumentError("fold_grid: need 2 ≤ F_folds ≤ length(rows) (got F_folds=$F_folds, rows=$n — SPEC §28 contiguous folds)"))
    q, r = divrem(n, F_folds)
    folds = Vector{Vector{Int}}()
    pos = 1
    for f in 1:F_folds
        len = q + (f <= r ? 1 : 0)
        push!(folds, collect(rows[pos:(pos + len - 1)]))
        pos += len
    end
    folds
end
fold_grid(n::Int; F_folds::Int = 3) = fold_grid(collect(1:n); F_folds = F_folds)

# ---------------------------------------------------------------------------
# fold 充分统计与差分（SPEC §28 恒等式）
# ---------------------------------------------------------------------------

"""
    fold_stats(X, Y, folds) -> (; full, folds)

full 统计（`sufficient_stats(X, Y)`）与每 fold 的行子集统计
（`sufficient_stats(X[fold_rows, :], Y[fold_rows, :])`）。
"""
function fold_stats(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64},
                    folds::Vector{Vector{Int}})
    full = sufficient_stats(X, Y)
    per = [sufficient_stats(X[f, :], Y[f, :]) for f in folds]
    (; full = full, folds = per)
end

"""
    train_stats(full_st, fold_st) -> SufficientStats

**train 差分恒等式（SPEC §28）**：`train^{(−f)} = full − fold`——
`(n, Sxx, Sxy, Syy)` 逐元素差。**这是精确代数恒等式，非近似**（充分
统计的线性性）；浮点下差分路径相对直接重算存在 eps 级舍入微扰（求和
顺序不同）——语义隔离不受影响（held-out 行的贡献在 full 与 fold 中
同现、差分中相消），数值精度由测试的差分-vs-直接对照断言钉死
（atol 1e-10）。
"""
function train_stats(full_st::SufficientStats, fold_st::SufficientStats)
    SufficientStats(full_st.n - fold_st.n,
                    full_st.Sxx - fold_st.Sxx,
                    full_st.Sxy - fold_st.Sxy,
                    full_st.Syy - fold_st.Syy)
end

# ---------------------------------------------------------------------------
# 每 fold 完整 posterior（D-043：独立 quadrature）
# ---------------------------------------------------------------------------

"""
    fit_fold_posteriors(X, Y, folds; tol = 1e-6, max_cells = 2048,
                        u_span = 10.0)
        -> (; folds, posteriors::Vector{ResponsePosterior})

每 fold 一个**完整 posterior**（D-043）：

1. `fold_stats`（full + 每 fold 统计）；
2. train 统计 = 差分（`train_stats`——held-out 行不进 fold 证据是差分的
   **构造性**推论，非实现巧合：行 s 的贡献只出现在 full 与其所属 fold
   的统计中，差分后相消）；
3. **fold 独立 quadrature**：每 fold 用自己的 train 统计走
   `adaptive_quadrature_2d`（粗扫描 mode 仅作该 fold 的初始域中心）——
   **不复用** full-data posterior 的节点/权重（D-043 原文）；
4. **fold 独立 propriety gate**：`n^{(−f)} ≥ N` 且 `rank(Syy^{(−f)}) = N`
   ——违反即 error，文本含 **"posterior improper"** 与 fold 编号；
5. 每 fold 的 `ResponsePosterior` 持有 train 行子矩阵（X/Y 字段与差分
   统计一致——供 `oof_residual_rows` 类消费）。

**D-044 纪律（钉死）**：OOF 只用于 innovation 校准（残差行的构造），
不是模型选择工具——不得据 OOF Sharpe 选 bands/priors/d/universe。
"""
function fit_fold_posteriors(X::AbstractMatrix{Float64},
                             Y::AbstractMatrix{Float64},
                             folds::Vector{Vector{Int}};
                             tol::Float64 = 1e-6,
                             max_cells::Int = 2048,
                             u_span::Float64 = 10.0)
    n, P = size(X)
    N = size(Y, 2)
    fs = fold_stats(X, Y, folds)
    posteriors = Vector{ResponsePosterior}()
    for (f, rows) in enumerate(folds)
        st_train = train_stats(fs.full, fs.folds[f])
        # fold 独立 A2 gate（含 fold 编号；D-036 文本）
        rank_syy = rank(Symmetric(st_train.Syy))
        (st_train.n >= N && rank_syy == N) ||
            error("fit_fold_posteriors: posterior improper (fold f=$f) — train-layer condition failed (n⁽⁻ᶠ⁾=$(st_train.n), N=$N, rank(Syy⁽⁻ᶠ⁾)=$rank_syy; D-043/D-036)")
        has_a0 = (P == 1 + 14 * N)
        logf(u0, up) = log_evidence(st_train, lambda_diag(P, u0, up, has_a0)) +
                       log_prior_d035a(u0, up)
        # 粗扫描 mode：该 fold 的初始域中心（D-038 bracket 语义）
        grid = range(-12.0, 12.0; length = 21)
        m0, mp, mval = 0.0, 0.0, -Inf
        for u0 in grid, up in grid
            v = logf(u0, up)
            if v > mval
                mval = v; m0 = u0; mp = up
            end
        end
        # tail_rel 标定（Step 16 短窗口轮，两分支同步）：10·exp(−u_span)——
        # Wave 4 predictive 轮的 area 因子（posterior.jl 判据内）对部分
        # fold 的 α₀ 平台-峰差（随机游走无信号 fixture 的 fold 间波动）
        # 仍不够（backtest_tests seed=31 实测红）；一个数量级余量后仍为
        # 强约束（域外质量 ≤ ~4.5e-4·参照·面积比）。
        if has_a0
            quad = adaptive_quadrature_2d(logf, (m0 - u_span, m0 + u_span),
                                          (mp - u_span, mp + u_span);
                                          tol = tol, max_cells = max_cells,
                                          tail_rel = 10 * exp(-u_span))
            nodes = quad.nodes
            weights = quad.weights
            logZ = quad.logZ
        else
            logf1(up) = log_evidence(st_train, lambda_diag(P, 0.0, up, false)) +
                        log_prior_d035a(0.0, up)
            quad1 = _adaptive_quadrature_1d(logf1, (mp - u_span, mp + u_span);
                                            tol = tol,
                                            max_cells = 4 * max_cells,
                                            tail_rel = 10 * exp(-u_span),
                                            m_ref = mval)
            nodes = [(0.0, up) for up in quad1.nodes]
            weights = quad1.weights
            logZ = quad1.logZ
        end
        conds = [_fit_node(st_train, lambda_diag(P, u0, up, has_a0), weights[k])
                 for (k, (u0, up)) in enumerate(nodes)]
        levs = [log_evidence(st_train, lambda_diag(P, u0, up, has_a0))
                for (u0, up) in nodes]
        push!(posteriors, ResponsePosterior(st_train.n, N, P, has_a0, :d035a,
            nodes, weights, levs, logZ,
            ProprietyCertificate(true, true, true, true), conds,
            st_train.n + 1 - N, st_train,
            Matrix(X[rows, :]), Matrix(Y[rows, :])))
    end
    (; folds = folds, posteriors = posteriors)
end

# ---------------------------------------------------------------------------
# 完整 fold 网格的 OOF 残差行（RP §7.3(a) 输出契约）
# ---------------------------------------------------------------------------

"""
    oof_residual_rows_full(fp, X, Y, E_active) -> Matrix{Float64}   # n × N

完整 fold 网格的 OOF 残差行（RP §7.3(a) 输出契约——innovation 层的规范
输入）：

```
ε̃_s = E_active·[y_s − ŷ_s^{(−fold(s))}] ,
ŷ_s^{(−f)} = B̂mix^{(−f)}·x_s ,  B̂mix^{(−f)} = Σ_k p_k^{(−f)}·B̂_k^{(−f)}
```

- **D-041 期望语义**：ŷ 是该行所属 fold 的 train posterior 的**完整后验
  均值**（含 α 积分——节点混合），非 plug-in 点估计；
- **含 b₀ 扣除**（DC 列在 B̂·x 内自动参与）；
- **held-out 隔离（构造性）**：行 s 的预测只用 `train^{(−fold(s))}` 的
  差分统计——行 s 的贡献在 full 与 fold(s) 中同现、差分相消；
- 返回 **asset 空间**行（E_active·r_mode，E 为 active 域 mode→asset 基
  `domain_mode_basis(N)`）；行 mask 契约归 VI 层（O_s 判定）。

`fp` 为 `fit_fold_posteriors` 的返回值（含 folds 行归属与 posteriors）。
"""
function oof_residual_rows_full(fp::NamedTuple,
                                X::AbstractMatrix{Float64},
                                Y::AbstractMatrix{Float64},
                                E::AbstractMatrix{Float64})
    folds = fp.folds
    posteriors = fp.posteriors
    N = size(Y, 2)
    (size(E, 1) == N && size(E, 2) == N) ||
        throw(DimensionMismatch("oof_residual_rows_full: E must be N×N domain mode basis"))
    # 行归属表（fold → 行集；拼接应覆盖全部行——构造保证）
    Bmix = [sum(c.weight .* c.B_hat for c in post.conditional) for post in posteriors]
    out = Matrix{Float64}(undef, size(Y, 1), N)
    filled = falses(size(Y, 1))
    for (f, rows) in enumerate(folds)
        for s in rows
            r_mode = Y[s, :] .- Bmix[f] * view(X, s, :)
            out[s, :] .= E * r_mode
            filled[s] = true
        end
    end
    all(filled) || error("oof_residual_rows_full: fold grid does not cover all rows (uncovered rows: $(findall(!, filled)))")
    out
end
