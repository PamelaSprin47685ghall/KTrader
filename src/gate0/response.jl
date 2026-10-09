# =============================================================================
# KTraderGate0 full block response reference（固定 ridge）— 施工图 Step 4
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付（同 geometry.jl / modes.jl），测试经临时 module include。
# include 顺序：geometry.jl → modes.jl → 本文件。
#
# 职责边界（施工图 Step 4 + Step 5 / §2.4）：**只做**统一 mode-space
# response 的固定 ridge reference——验证 design/target algebra、四块
# cross-response 的可恢复性（D-092）与 DC/intercept 通道（D-093）。
# 本步明确不做：
# - full posterior / quadrature / propriety gate（Step 8 的 ResponsePosterior）；
# - trace conditioning（D-028 已降级为 hypothesis，永不进默认 core）；
# - EB / 任何超参推断（alpha 是调用方给定的固定数值，非策略参数）。
#
# 规范来源：
# - 模型：docs/GATE0_RESPONSE_POSTERIOR.md §1（y_{t+1} = b₀ + G·x_t + ε；
#   本步无 b₀，G ∈ R^{N×14N}，四块全开）。
# - D-025/D-026：macro↔relative cross-response 必须允许——G 的四个 block
#   全部由数据决定，不施加 block-diagonal、不施加 trace 约束。
# - D-082/D-083：slow single-threaded reference——正规方程直接求解，无
#   workspace、无缓存、无 primal/dual 切换、无优化。
#
# 块语义（钉死；行块/列块切分，与 GATE0_RESPONSE_POSTERIOR §1.3 的矩阵
# 分块记号一致）：
#
#   B ∈ R^{N×P}，P = 14N；行块 = **输出** mode（行 1 = m、行 2:N = q），
#   列块 = **输入** mode（列 1:14 = B_m、列 15:P = B_⊥）：
#
#   B_mm       = B[1:1, 1:14]      m 输出 ← macro path 输入
#   B_mperp    = B[1:1, 15:P]      m 输出 ← relative path 输入
#                                    （relative path 预测整体市场）
#   B_perpm    = B[2:N, 1:14]      q 输出 ← macro path 输入
#                                    （macro path 预测横截面 rotation）
#   B_perpperp = B[2:N, 15:P]      q 输出 ← relative path 输入
#
#   记号勘误（如实记录）：GATE0_RESPONSE_POSTERIOR §1.3 正文句「relative
#   path 可以预测整体市场（G_{⊥m} 方向）」与其**自身矩阵分块定义相反**——
#   按矩阵记号（行 = 输出），该方向是 G_{m⊥}（右上块）。本文件按矩阵分块
#   语义执行；D-092 测试对两个 cross 块均构造非零并断言恢复，覆盖不受
#   该正文笔误影响。

export build_mode_problem, fit_full_block_ridge, predict_mode, block_views

# 文件自含依赖声明（与 geometry.jl / kelly.jl 的模式一致）：本文件使用
# Symmetric / I / cholesky（LinearAlgebra 导出名）。本文件以「裸文件 +
# include」形态被两种加载路径消费（顶层 module KTraderGate0 与测试的
# 临时 module），include 时此语句在加载者作用域执行——两条路径同时获得
# 所需名字。
using LinearAlgebra

# ---------------------------------------------------------------------------
# build_mode_problem：design/target 组装（消费 Step 3 producer 接口）
# ---------------------------------------------------------------------------

"""
    build_mode_problem(returns, observed, s1, E; s_m = nothing, s_perp = nothing)
        -> (; X, Y, X_full, y_modes, B_m, B_perp, X_m, X_rel,
            m_series, e_embedded, E, Q, N, T, s_m, s_perp)

组装统一 mode-space design/target（施工图 Step 4；消费 Step 3 的
`modes.jl` producer 接口：`mode_field` → `cumulative_path_coordinates` →
`mode_basis_design`）。

**裸输入形态**（MarketFacts / Eligibility → 裸输入的提取由集成轮接线，
本层不绑数据结构）：

- `returns ∈ R^{T×N}`：日 log returns（缺失坐标 NaN 占位，D-022 语义）；
- `observed::T×N`：观察 mask（D-022：永久保留，缺失 ≠ 零收益）；
- `s1 ∈ R^N`：per-asset 一日 ruler（`ruler(x,f)[:,1]` 语义）；
- `E ∈ R^{N×N}`：域级 mode 基（`domain_mode_basis(N)`，裁决 E1）；
- `s_m`/`s_perp`：默认自动计算（`fast_s_m(X_m)` / `compute_s_perp(X_rel)`），
  可显式传入（测试注入用）；`s_m` 接受 |BANDS| 或 |TAUS| 维（`fast_s_m`
  返回 9 维，`path_basis_1d` 经 BANDCOL 分派——modes.jl 契约）。

**时序对齐（钉死）**：

- `X_full` 行 `t` = `t` 时刻 feature（含 ≤t 信息，`path_basis_1d` 的
  窗口语义；前缀不足 `t < 2τ+1` 的行为 0）；
- 配对：`X = X_full[1:T−1, :]`、`Y = y_modes[2:T, :]`——即
  **Y 行 s 是 X 行 s 的下一日 mode 输出**（`y_{s+1}` 对 `x_s`；
  GATE0_RESPONSE_POSTERIOR §1.1 的 ts_total 语义：feature 在 s、目标在
  s+1）；
- 本函数返回**全量**配对行（1:T−1）；前缀不足行是否剔除（旧线
  `ts_total = WARMUP:T−2` 语义）由调用方决定，不在本层硬编码。

**列布局（钉死，与 `mode_basis_design` 输出一致）**：
- `dc = false`（默认，Step 4 形态）：`X_full = [B_m | B_⊥]`——列 1:14 =
  B_m（macro path）、列 15:14N = B_⊥（relative path gauge 坐标），
  共 14N 列；
- `dc = true`（Step 5，D-031）：`X_full = [1 | B_m | B_⊥]`——列 1 = DC
  常数列 `x_{0,t} = 1`、列 2:15 = B_m、列 16:14N+1 = B_⊥，共 1+14N 列
  （与 GATE0_RESPONSE_POSTERIOR.md §1 维度表一致；modes.jl 已预留该
  列位说明）。

**DC 语义（D-031/D-032，钉死）**：DC 列对应系数 `b₀ = B[:,1] ∈ R^N`
在**统一 mode 输出空间**（行 1 = m：可含 macro 无条件漂移；行 2:N = q：
可含 zero-sum relative 长期漂移）。**零中心纪律（D-032，Step 8 义务）**：
本步固定 ridge 无先验项，但任何未来 prior 版本的 DC 先验均值**必须为 0**；
禁止人工正收益 prior、禁止按历史赢家指定正先验——b₀ 的非零 posterior
只能来自价格证据。

fail-loudly：维度不一致抛 `DimensionMismatch`；observed 坐标的 return
非 finite 抛 `DomainError`（`mode_field` 契约）；T < 2（无配对行）抛
`ArgumentError`。
"""
function build_mode_problem(returns::AbstractMatrix{Float64},
                            observed::AbstractMatrix{Bool},
                            s1::AbstractVector{Float64},
                            E::AbstractMatrix{Float64};
                            s_m::Union{Nothing,AbstractVector{Float64}} = nothing,
                            s_perp::Union{Nothing,AbstractVector{Float64}} = nothing,
                            dc::Bool = false)
    T, N = size(returns)
    (size(observed, 1) == T && size(observed, 2) == N) ||
        throw(DimensionMismatch("build_mode_problem: observed must be T×N matching returns"))
    (length(s1) == N && size(E, 1) == N && size(E, 2) == N) ||
        throw(DimensionMismatch("build_mode_problem: s1/E must have dimension N"))
    T >= 2 ||
        throw(ArgumentError("build_mode_problem: need T ≥ 2 rows to form (x_s, y_{s+1}) pairs"))
    # 逐行 mode_field（契约校验在 mode_field 内：s1 有限正、observed 坐标
    # return 有限——fail loudly）
    m_series = Vector{Float64}(undef, T)
    e_embedded = Matrix{Float64}(undef, T, N)
    y_modes = Matrix{Float64}(undef, T, N)
    for t in 1:T
        mf = mode_field(view(returns, t, :), view(observed, t, :), s1, E)
        m_series[t] = mf.y[1]
        e_embedded[t, :] .= mf.e
        y_modes[t, :] .= mf.y
    end
    X_m, X_rel = cumulative_path_coordinates(m_series, e_embedded)
    s_m_eff = s_m === nothing ? fast_s_m(X_m) : s_m
    s_perp_eff = s_perp === nothing ? compute_s_perp(X_rel) : s_perp
    Q = view(E, :, 2:N)                       # 域基 relative 块（Helmert）
    des = mode_basis_design(X_m, X_rel, Q, s_m_eff, s_perp_eff)
    X_core = hcat(des.B_m, des.B_perp)       # T × 14N（动态列）
    X_full = dc ? hcat(fill(1.0, T), X_core) : X_core   # dc=true：[1 | B_m | B_⊥]（D-031）
    X = X_full[1:(T - 1), :]                  # 行 s = s 时刻 feature
    Y = y_modes[2:T, :]                       # 行 s = s+1 时刻 mode 输出
    (; X = X, Y = Y, X_full = X_full, y_modes = y_modes,
       B_m = des.B_m, B_perp = des.B_perp, X_q = des.X_q,
       X_m = X_m, X_rel = X_rel, m_series = m_series, e_embedded = e_embedded,
       E = E, Q = Q, N = N, T = T, s_m = s_m_eff, s_perp = s_perp_eff)
end

# ---------------------------------------------------------------------------
# 四块切分（行块 = 输出 mode、列块 = 输入 mode）
# ---------------------------------------------------------------------------

"""
    block_views(B) -> (; b0, B_mm, B_mperp, B_perpm, B_perpperp)

统一系数矩阵 `B` 的 DC 列 + 四块视图（共享内存的 view，非 copy）：

- 行块 = **输出** mode：行 1 = m（macro 输出）、行 2:N = q（relative 输出）；
- 列块 = **输入** mode：无 DC 形态（`P = 14N`）列 1:14 = B_m、列 15:14N =
  B_⊥；DC 形态（`P = 1+14N`，D-031）**列 1 = b₀**、四块从列 2 起
  （列 2:15 = B_m、列 16:1+14N = B_⊥）。
- `b0 = B[:,1] ∈ R^N`（N 维 view；D-031：统一 mode 输出空间的
  intercept——行 1 可含 macro 无条件漂移、行 2:N 可含 zero-sum relative
  长期漂移）。**零中心纪律（D-032，Step 8 义务）**：未来任何 prior 版本
  的 DC 先验均值必须为 0；禁止人工正收益 prior、禁止按历史赢家指定正
  先验——b₀ 的非零 posterior 只能来自价格证据。非 DC 形态返回
  `b0 = nothing`。

块维度：`B_mm ∈ R^{1×14}`、`B_mperp ∈ R^{1×14(N−1)}`、
`B_perpm ∈ R^{(N−1)×14}`、`B_perpperp ∈ R^{(N−1)×14(N−1)}`。
N = 1 时 relative 块为空视图（退化合法）。

`P ∉ {14N, 1+14N}` 抛 `ArgumentError`（块切分的维度契约）。
"""
function block_views(B::AbstractMatrix{Float64})
    N, P = size(B)
    has_dc = (P == 1 + 14 * N)
    (!has_dc && P != 14 * N) &&
        throw(ArgumentError("block_views: column count must be 14N (no DC) or 1+14N (DC) (got P=$P, N=$N)"))
    o = has_dc ? 1 : 0
    (; b0 = has_dc ? view(B, :, 1) : nothing,
       B_mm = view(B, 1:1, (o + 1):(o + 14)),
       B_mperp = view(B, 1:1, (o + 15):P),
       B_perpm = view(B, 2:N, (o + 1):(o + 14)),
       B_perpperp = view(B, 2:N, (o + 15):P))
end

# ---------------------------------------------------------------------------
# 固定 ridge 拟合（统一回归，四块全由数据决定）
# ---------------------------------------------------------------------------

"""
    fit_full_block_ridge(X, Y; alpha = 1.0)
        -> (; B, B_mm, B_mperp, B_perpm, B_perpperp, alpha, n, P, N)

统一 mode-space 回归的**固定 ridge reference**（Step 4/5；无 EB、无
posterior——Step 8 的对象）：

```
min_B  ‖Y − X·Bᵀ‖²_F + α·‖B‖²_F ,
B ∈ R^{N×P}, P ∈ {14N（无 DC）, 1+14N（DC，D-031）}
```

正规方程直接求解（D-082/D-083 reference 级，单线程，无优化）：

```
Bᵀ = (XᵀX + α·I)⁻¹ · Xᵀ·Y        （Cholesky 求解，非显式求逆）
```

- **四块全部由数据决定**（D-025/D-026）：不施加 block-diagonal、不施加
  trace 约束（D-028）。`B_mperp`/`B_perpm` 两个 cross 块与对角块同一
  个正规方程解出。
- **DC 列处理（Step 5 简化，钉死）**：固定 α 作用于**全部列**（包括
  DC 列）。这在 Step 8 会被 `α₀/α_p` 双 group 精度取代（D-033：零频
  DC ≠ dynamic path response，两 group；禁 per-band/per-asset/ARD）。
  DC 形态（P=1+14N）下首列必须恒 1（D-031 的 x_{0,t}=1），否则
  fail loudly。
- **零中心纪律（D-032）**：本步固定 ridge 无先验项；未来 prior 版本的
  DC 先验均值必须为 0，非零 posterior 只能来自价格证据（Step 8 义务，
  docstring 钉死）。
- `alpha` 是调用方给定的固定正数（数值正则，非策略参数——本步无 EB）；
  `α > 0` 保证 `XᵀX + αI ≻ 0`（XᵀX 半正定）。
- 返回 `B`（N×P）、`b0`（DC 形态为 N 维 view，非 DC 为 `nothing`）与
  `block_views(B)` 的四块视图（顶层字段展开）。

fail-loudly：X/Y 行数不一致或非二维抛 `DimensionMismatch`；
`P ∉ {14N, 1+14N}`（块切分契约）抛 `ArgumentError`；DC 形态首列非
恒 1 抛 `ArgumentError`；`alpha` 非有限或 ≤ 0 抛 `ArgumentError`；
X/Y 含 NaN/Inf 抛 `DomainError`；正规方程数值奇异（Cholesky 失败）
抛 `ErrorException`（SPEC §56）。
"""
function fit_full_block_ridge(X::AbstractMatrix{Float64},
                              Y::AbstractMatrix{Float64};
                              alpha::Real = 1.0)
    n, P = size(X)
    n2, N = size(Y)
    n2 == n ||
        throw(DimensionMismatch("fit_full_block_ridge: X and Y must have the same number of rows (got $n and $n2)"))
    N >= 1 ||
        throw(DimensionMismatch("fit_full_block_ridge: Y must have at least one output column (got N=$N)"))
    has_dc = (P == 1 + 14 * N)
    if !has_dc && P != 14 * N
        throw(ArgumentError("fit_full_block_ridge: X must have 14N (no DC) or 1+14N (DC, D-031) columns (got P=$P, N=$N) — unified mode design [B_m | B_⊥] or [1 | B_m | B_⊥]"))
    end
    if has_dc
        all(==(1.0), view(X, :, 1)) ||
            throw(ArgumentError("fit_full_block_ridge: DC column (first column, D-031 x_{0,t}=1) must be constant 1 — fail loudly"))
    end
    n >= 1 ||
        throw(ArgumentError("fit_full_block_ridge: need at least one training row"))
    (isfinite(alpha) && alpha > 0) ||
        throw(ArgumentError("fit_full_block_ridge: alpha must be finite and positive (fixed ridge regularizer, got $alpha)"))
    all(isfinite, X) ||
        throw(DomainError("fit_full_block_ridge: X contains NaN/Inf — design matrix must be finite (fail loudly, SPEC §56)"))
    all(isfinite, Y) ||
        throw(DomainError("fit_full_block_ridge: Y contains NaN/Inf — targets must be finite (fail loudly, SPEC §56)"))
    A = Symmetric(X' * X + alpha * Matrix(I, P, P))
    F = try
        cholesky(A)
    catch err
        throw(ErrorException("fit_full_block_ridge: normal equations (XᵀX + αI) numerically singular — fail loudly (SPEC §56): $(sprint(showerror, err))"))
    end
    B = Matrix((F \ (X' * Y))')             # N × P（transpose of P×N solve）
    blocks = block_views(B)
    (; B = B, b0 = blocks.b0,
       B_mm = blocks.B_mm, B_mperp = blocks.B_mperp,
       B_perpm = blocks.B_perpm, B_perpperp = blocks.B_perpperp,
       alpha = Float64(alpha), n = n, P = P, N = N)
end

# ---------------------------------------------------------------------------
# mode 坐标条件均值预测
# ---------------------------------------------------------------------------

"""
    predict_mode(B, x_t) -> Vector{Float64}

mode 坐标条件均值 `ŷ_{t+1} = B · x_t`（N 维；行 1 = m、行 2:N = q——
与 `block_views` 行块语义一致）。asset 空间投影（`E·ŷ`，输出契约 2 的
`μ_asset = E_active·μ`）由调用方执行——response 层输出到 mode 坐标为止。

fail-loudly：`length(x_t) ≠ size(B,2)` 抛 `DimensionMismatch`；
B/x_t 含 NaN/Inf 抛 `DomainError`。
"""
function predict_mode(B::AbstractMatrix{Float64}, x_t::AbstractVector{Float64})
    size(B, 2) == length(x_t) ||
        throw(DimensionMismatch("predict_mode: x_t length must match B columns (got $(length(x_t)) vs $(size(B, 2)))"))
    all(isfinite, B) ||
        throw(DomainError("predict_mode: B contains NaN/Inf"))
    all(isfinite, x_t) ||
        throw(DomainError("predict_mode: x_t contains NaN/Inf"))
    B * collect(x_t)
end
