# =============================================================================
# KTraderGate0 OOF full-mode folds（legacy/reference diagnostic）+ prequential
# residual history（production 残差源，P0-1 裁决）
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付。include 顺序：geometry.jl → modes.jl → response.jl →
# posterior.jl → 本文件（本文件消费 posterior.jl 的同作用域件——
# SufficientStats/sufficient_stats/lambda_diag/log_evidence/log_prior_d035a/
# adaptive_quadrature_2d/_fit_node/ResponsePosterior/ProprietyCertificate；
# `_fit_node` 未 export 但同 module include 链下可见）。
#
# 本文件两部分（P0-1 裁决后）：
#
# 1. **legacy OOF full-mode folds（保留，reference/diagnostic）**：原 Step 9
#    的 fold_grid/fold_stats/train_stats/fit_fold_posteriors/
#    oof_residual_rows_full。规范来源：SPEC §28-§29、RP §9（D-043/D-044）。
#    **P0-1 裁决后 production 决策链不再消费 fold residual**——3-fold OOF
#    对历史日期 s 的预测模型可以使用 s+1..t 的数据，对今天的 backtest 无
#    future leakage，但不是「历史当时真正发生的 forecast surprise」，对
#    non-Markov innovation law 是本质区别。本组函数保留为 legacy/reference
#    diagnostic（测试与对照仍可调用），**不得再被 production 路径调用**；
#    docstring 与 export 名不变（测试兼容面），仅文件头与各 docstring 加
#    显式 legacy 标注。
#
# 2. **strict prequential residual history（production 残差源，P0-1）**：
#    `prequential_residual_rows`（不 export，同 module include 链可见）——
#    对每个历史日期 s，用截至 s 的信息拟合模型、预测 s+1、观察 r_{s+1}、
#    得到 ε_{s+1}、追加到 residual history（裁决：H_t → predict r_{t+1} →
#    observe r_{t+1} → ε_{t+1} → append）。输出形态与原 OOF residual rows
#    一致（行 = 历史日期、列 = 资产，供 innovation 层消费）；行身份 =
#    目标日 u_s（与 row_ids 契约同源）。
#
# 无观察行语义（钉死）：mode 层的 X/Y 是 mode 坐标（mode_field 的零嵌入
# 语义：|O_s|=0 时 y=0——不携带 mask）；「无观察行不进统计」的行集判定
# 归 prepare/consumer 层（调用方从 rows 剔除）。prequential 的输入行集
# 同样由调用方剔除空观察行；**行集内每行都参与拟合与残差**（prequential
# 没有 held-out 概念——每行 s 的预测只用 ≤s−1 的信息，行 s 自身是目标）。
#
# warm-start 语义（钉死）：本 reference **不使用** warm start（D-083
# reference 级每步独立全跑可接受），集成/增量轮可按此语义接入（届时
# prequential 的充分统计可递推——见 prequential_residual_rows docstring
# 的复杂度注记）。

export fold_grid, fold_stats, train_stats, fit_fold_posteriors,
       oof_residual_rows_full
# 注意：prequential_residual_rows **不 export**——同 module include 链下
# 可见（driver 接线直接调用）；保持既有导出面 5 名不变（market_tests 的
# 精确导出面断言 87 名依赖此面，P0-1 任务约束不改 test）。

# ---------------------------------------------------------------------------
# 【legacy】contiguous fold 划分（SPEC §28）——P0-1 后仅作 reference/
# diagnostic；production 残差源已切换为 prequential_residual_rows
# ---------------------------------------------------------------------------

"""
    fold_grid(rows; F_folds = 3) -> Vector{Vector{Int}}
    fold_grid(n::Int; F_folds = 3)

**【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费】**
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
# 【legacy】fold 充分统计与差分（SPEC §28 恒等式）——P0-1 后仅作
# reference/diagnostic
# ---------------------------------------------------------------------------

"""
    fold_stats(X, Y, folds) -> (; full, folds)

**【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费】**
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

**【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费】**
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
# 【legacy】每 fold 完整 posterior（D-043：独立 quadrature）——P0-1 后仅作
# reference/diagnostic
# ---------------------------------------------------------------------------

"""
    fit_fold_posteriors(X, Y, folds; tol = 1e-6, max_cells = 2048,
                        u_span = 5.0)
        -> (; folds, posteriors::Vector{ResponsePosterior})

**【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费；production
残差源已切换为 prequential_residual_rows（strict prequential）】**

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
                             u_span::Float64 = 5.0)
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
        # tail_rel 标定（Step 16 短窗口轮，两分支同步）：10·exp(−u_span)。
        # u_span 默认已与 posterior.jl 对齐为 5（2026-10-10 静态推导：tail
        # certificate 仍保护尾部、非放宽 tolerance——见 fit_full_posterior
        # docstring 的 u_span 选择说明）。原始动机（Wave 4：posterior.jl
        # 判据内 area 因子对部分 fold 的 α₀ 平台-峰差仍不够，backtest_tests
        # seed=31 实测红）与 m_ref 低估峰值同源——eng-posterior 已修复
        # adaptive_quadrature_2d 的 m_ref 参数（本调用点补传 mval，与一维
        # 分支对齐），误报源消除。保留 tail_rel=10·exp(−u_span) 一个数量级
        # 余量作为保守选择（域外质量 ≤ ~10·exp(−u_span)·参照·面积比，仍为
        # 强约束；u_span=5 时 ≈ 0.067·参照·面积比）；若 m_ref 修复后实测
        # 证明不再需要该余量，可后续收紧回 exp(−u_span)——由 DevOps 运行
        # 验证决定，不得由回测选择。
        if has_a0
            # DC 形态（P=1+14N）：(u₀, u_p) 二维 quadrature（D-033 双 group）
            # max_cells 不显式传（默认 8192——Manager 裁决 2026-10-10 与
            # posterior.jl fit_full_posterior DC 分支对称；本函数签名
            # max_cells 参数仍透传给一维路径的 4*max_cells，二维路径用
            # adaptive_quadrature_2d 默认值，避免调用方误传旧 2048）。
            quad = adaptive_quadrature_2d(logf, (m0 - u_span, m0 + u_span),
                                          (mp - u_span, mp + u_span);
                                          tol = tol,
                                          tail_rel = 10 * exp(-u_span),
                                          m_ref = mval)
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
# 【legacy】完整 fold 网格的 OOF 残差行（RP §7.3(a) 输出契约）——P0-1 后仅
# 作 reference/diagnostic
# ---------------------------------------------------------------------------

"""
    oof_residual_rows_full(fp, X, Y, E_active) -> Matrix{Float64}   # n × N

**【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费；production
残差源已切换为 prequential_residual_rows（strict prequential）。本函数
保留供 legacy 对照/测试/诊断使用，不得再被 production 决策链调用】**

完整 fold 网格的 OOF 残差行（RP §7.3(a) 输出契约——原 innovation 层
的规范输入）：

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

# =============================================================================
# 【production】strict prequential residual history（P0-1 裁决）
# =============================================================================
#
# 裁决（P0-1）：innovation 残差必须是 prequential——历史当时真正发生的
# forecast surprise：
#
#     H_t → predict r_{t+1} → observe r_{t+1} → ε_{t+1} → append
#
# 而不是 3-fold OOF（历史日期 s 的 OOF 模型可以使用 s+1..t 的数据——对
# 今天的 backtest 无 future leakage，但不是历史当时真正发生的 surprise，
# 对 non-Markov innovation law 是本质区别）。F_folds 退出 production
# theory；本函数是 production 残差源的唯一实现。
#
# 实现选择（D-082/D-083 reference 级：单线程、无缓存、无优化）：
#
# 对每个行 s（按行序递增 = 时间顺序）：
#   1. train = 行 1..s−1（严格过去信息——**构造性因果**：行 s 的预测
#      只用 ≤s−1 的信息，行 s 自身是目标，绝不进入自己的训练集）；
#   2. 用 `fit_full_posterior` 的完整后验（D-034/D-035a quadrature）求
#      ŷ_s = B̂mix·x_s（完整后验均值，D-041 期望语义——与 legacy OOF 的
#      B̂mix 同构）；
#   3. ε_s = E_active·(y_s − ŷ_s)（asset 空间残差行，含 b₀ 扣除——DC 列
#      在 B̂·x 内自动参与）；
#   4. 追加到输出行 s。
#
# 早期行（train 不足）或单行 posterior 不可定义（A2 propriety 不满足，
# 或 posterior quadrature 未收敛——D-067 尾证书/预算耗尽/质量为零；
# 修法 A：早期行 HalfCauchy-τ 后验极宽时尾证书失败属预期行级不可定义，
# 与 A2 未满足同语义）：该行残差不可定义 → 输出 **NaN**（与 OOF 输出
# 形态一致——innovation 层用 row_masks 表达缺失，NaN 合法，VI §1.2
# 契约 4/5）。prequential 从第一个可定义行开始；若**没有任何**行可定义
# （全部行 s ≤ n 的 train 都退化/求积失败），fail loudly（文本含
# "prequential residual"）——不得静默返回全 NaN 冒装有信息的残差历史。
#
# 复杂度（reference 级，D-083 单日慢可接受）：
# - 时间：每行 s 一次 sufficient_stats（O(s·P·N)）+ 一次完整 posterior
#   quadrature（O(cells·(P³ + N³)) 量级）；总 O(n²·P·N) + n·quadrature。
# - 空间：O(n·N)（输出矩阵）+ O(P² + N²)（单步工作集）。
# - 递推潜力（如实声明，未实现）：充分统计 (n, Sxx, Sxy, Syy) 对行追加
#   是精确 rank-one 更新——集成/增量轮可把每步 sufficient_stats 从
#   O(s·P·N) 降到 O(P·N)（届时对 posterior.jl 的修改需另行授权；本步
#   硬约束不改它，保持 reference 级每步独立重算）。
#
# 输出契约（与 legacy oof_residual_rows_full 同形态，供 innovation_state
# 消费）：n × N（asset 空间），行序 = 输入行序（时间顺序），行身份 =
# 目标日 u_s（调用方 row_ids 与之一一对应）。

"""
    prequential_residual_rows(X, Y, E_active; tol = 1e-6, max_cells = 2048,
                              u_span = 5.0) -> Matrix{Float64}   # n × N

strict prequential residual history（P0-1 裁决——production 残差源）：

```
ε_s = E_active·[y_s − ŷ_s] ,
ŷ_s = B̂mix^{(≤s−1)}·x_s ,  B̂mix^{(≤s−1)} = Σ_k p_k^{(≤s−1)}·B̂_k^{(≤s−1)}
```

- **strictly causal（构造性）**：行 s 的预测只用行 1..s−1 的充分统计拟合
  的 posterior——行 s 自身绝不进入自己的训练集（与 OOF 的「train =
  full − held-out fold」不同：prequential 没有 held-out 概念，每行都是
  历史当时真正发生的 forecast surprise）；
- **D-041 期望语义**：ŷ 是 train posterior 的**完整后验均值**（含 α
  积分——节点混合），非 plug-in 点估计；
- **含 b₀ 扣除**（DC 列在 B̂·x 内自动参与）；
- 返回 **asset 空间**行（E_active·r_mode，E 为 active 域 mode→asset 基
  `domain_mode_basis(N)`）；行 mask 契约归 VI 层（O_s 判定）；
- **早期行或求积失败的行**（train 不足 A2 propriety，或 posterior
  quadrature 未收敛——D-067 尾证书/预算耗尽/质量为零；修法 A：单行
  求积失败与该行 posterior 不可定义同语义）：该行残差不可定义 →
  输出 NaN（innovation 层以 row_masks 表达缺失）；**若所有行都不可
  定义** → fail loudly（文本含 "prequential residual"）。

`X`/`Y` 为 mode 坐标配对行（build_mode_problem 输出；调用方已剔除空观察
行）；行序 = 时间顺序（严格递增）。
"""
function prequential_residual_rows(X::AbstractMatrix{Float64},
                                   Y::AbstractMatrix{Float64},
                                   E::AbstractMatrix{Float64};
                                   tol::Float64 = 1e-6,
                                   max_cells::Int = 2048,
                                   u_span::Float64 = 5.0)
    n, P = size(X)
    n2, N = size(Y)
    n2 == n || throw(DimensionMismatch("prequential_residual_rows: X/Y row counts differ ($n vs $n2)"))
    (size(E, 1) == N && size(E, 2) == N) ||
        throw(DimensionMismatch("prequential_residual_rows: E must be N×N domain mode basis"))
    n >= 1 || throw(ArgumentError("prequential_residual_rows: need ≥1 training row"))
    out = Matrix{Float64}(undef, n, N)
    # （ii）行级任务并行（Manager 裁决 2026-10-10；测量：67 行串行 ≈200s/日）。
    # 每行只读共享输入（X/Y/E）、只写 out[s, :]，无跨行归约 ⇒ 完成顺序
    # 不影响结果、与串行路径逐位一致（单线程会话自动退化为顺序执行）。
    # 非行级错误记入槽位、循环后按行序在主线程抛出（保持原错误语义）。
    filled_flags = fill(false, n)
    errs_slot = Vector{Any}(undef, n)
    fill!(errs_slot, nothing)
    Threads.@threads for s in 1:n
        # train = 行 1..s−1（strictly causal；s=1 时空 train → 跳过）
        if s - 1 < 1
            out[s, :] .= NaN
            continue
        end
        X_tr = X[1:(s-1), :]
        Y_tr = Y[1:(s-1), :]
        # A2 propriety gate 由 fit_full_posterior 内部执行（D-036）——
        # 早期行不满足时该行残差不可定义，输出 NaN（与 OOF 输出形态一致）。
        local post::ResponsePosterior
        try
            post = fit_full_posterior(X_tr, Y_tr; tol = tol,
                                      max_cells = max_cells, u_span = u_span)
        catch err
            # 单行 posterior 不可定义 → 该行残差不可定义 → NaN（修法 A：
            # 与 A2 propriety 未满足同语义）。判定覆盖：
            # - "posterior improper"：A2 数据层 gate（fit_full_posterior 内部）
            # - "Numerical integration did not converge"：D-067 求积/收敛
            #   失败（尾质量证书、预算耗尽、被积质量为零——posterior.jl
            #   三处同文；早期行 HalfCauchy-τ 后验极宽时尾证书失败属预期
            #   行级不可定义，不是实现 bug）
            # 其余错误（调用契约 ArgumentError、PosDefException 等真正的
            # 实现/编程错误）仍 rethrow——不得把实现 bug 掩盖成 NaN。
            if _row_undefined_error(err)
                out[s, :] .= NaN
                continue
            end
            # 非行级错误：记槽位、循环后按行序重抛（原语义）
            errs_slot[s] = err
            continue
        end
        B_mix = sum(c.weight .* c.B_hat for c in post.conditional)
        r_mode = Y[s, :] .- B_mix * view(X, s, :)
        out[s, :] .= E * r_mode
        filled_flags[s] = true
    end
    for s in 1:n
        errs_slot[s] === nothing || throw(errs_slot[s])
    end
    any(filled_flags) ||
        error("prequential_residual_rows: no row has a proper train posterior (n=$n, N=$N — A2 propriety/quadrature never met for any row; prequential residual history undefined)")
    out
end

# ---------------------------------------------------------------------------
# 【helper】单行 posterior 不可定义的错误判定（修法 A，P0-1）
# ---------------------------------------------------------------------------

# 单行 posterior 不可定义 → 该行残差不可定义 → NaN。判定覆盖两类：
#
# - "posterior improper"：A2 数据层 propriety gate（n < N 或
#   rank(Syy) < N——fit_full_posterior 内部，D-036）；
# - "Numerical integration did not converge"：D-067 求积/收敛失败
#   （尾质量证书、预算耗尽、被积质量为零——posterior.jl 三处同文；
#   早期行 HalfCauchy-τ 后验极宽时尾证书失败属预期行级不可定义，
#   不是实现 bug）。
#
# 其余错误（调用契约 ArgumentError、数值实现 PosDefException 等真正的
# 编程/实现错误）不是行级不可定义，不得吞——rethrow 由调用方负责。
function _row_undefined_error(err)
    err isa ErrorException || return false
    occursin("posterior improper", err.msg) && return true
    occursin("Numerical integration did not converge", err.msg) && return true
    false
end
