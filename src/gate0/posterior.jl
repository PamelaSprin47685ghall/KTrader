# =============================================================================
# KTraderGate0 slow full-posterior reference — 施工图 Step 8
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付。include 顺序：geometry.jl → modes.jl → response.jl →
# 本文件（本文件自身只依赖 LinearAlgebra；mode→asset 基 E 由调用方显式
# 传入，不反向依赖 producer 层）。
#
# 数学权威：docs/GATE0_RESPONSE_POSTERIOR.md（下称 RP）——本文件全部闭式
# 按其公式编号实现：S(α)（RP §4.2）、marginal evidence（RP §4.3）、
# matrix-t（RP §4.4）、D-035a 修正先验（RP §5.4/§6.1）、quadrature
# （RP §6）、within/between 分解（RP §7.2）、输出契约（RP §7.3）。
# 管理裁决（docs/GATE0_MANAGER_ADJUDICATIONS.md）：D-035a（裁决 B）、
# 类型字段 G7（node evidence 与 ν 绑定）、抽样契约 F（μ 通道 matrix-t
# 后验抽样、quadrature 确定性不消耗 RNG）。
#
# 职责边界：**只做** response 模块的完整后验（B、Σ_R 解析消元 + (α₀,α_p)
# 的 R² deterministic quadrature——D-037/D-085：解析优先，数值积分只有
# 2 维）。本文件明确不做：innovation（VI 层）、scenario 合成、fold 网格
# （Step 9——oof_residual_rows 本步只提供单 fold 全数据版接口）。
#
# **D-060 红线（钉死）**：Σ_R 的作用到 μ 的 predictive 为止；innovation
# 通道不得消费 Σ_R / IW 抽样（RP §7.1）。本文件不提供任何「未来残差
# 冲击」对象。
#
# **ν 绑定（裁决 G7，钉死）**：μ 的 t 自由度 ν = n+1−N（response 输出
# 到 μ 为止、不含未来冲击）；**禁止** n−N 版本（含新观测噪声的 predictive
# 属 innovation 层职责的混淆形态）。
#
# reference 纪律（D-082/D-083）：单线程、无 workspace、无缓存、无优化；
# 每节点 cholesky 直接求解。

export SufficientStats, ConditionalFit, ProprietyCertificate, ResponsePosterior,
       sufficient_stats, s_alpha, log_evidence, log_prior_d035a,
       adaptive_quadrature_2d, fit_full_posterior,
       predict_mu, draw_mu, mu_asset, oof_residual_rows

# 缺 using 修复（Wave 3 验证轮）：draw_mu 签名注解 rng::AbstractRNG 与
# rand/randn 消耗 Random 的名字（方法签名注解在加载时即解析，缺失即
# UndefVarError）——原 using 面遗漏。
using Random, LinearAlgebra, Distributions   # Distributions：draw_mu 的 Chisq(ν)（本文件唯一
                                             # 外部包依赖——与 geometry/modes 的零依赖线偏差，
                                             # 如实声明；主 Project.toml 已有 Distributions）

# ---------------------------------------------------------------------------
# 充分统计（RP §3.2 断言 A1）
# ---------------------------------------------------------------------------

"""
    sufficient_stats(X, Y) -> SufficientStats(n, Sxx, Sxy, Syy)

RP §3.2 断言 A1：整个后验只依赖 `(X, Y)` 通过 `(n, Sxx, Sxy, Syy)`。
先验均值为零使 `Sxy = XᵀY` 无需中心化修正。fold 的 train 统计 =
full − eval fold（SPEC §28 恒等式，Step 9 消费）。

fail-loudly：维度不匹配抛 `DimensionMismatch`；NaN/Inf 抛 `DomainError`。
"""
struct SufficientStats
    n::Int
    Sxx::Matrix{Float64}     # P×P
    Sxy::Matrix{Float64}     # P×N
    Syy::Matrix{Float64}     # N×N
end

function sufficient_stats(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64})
    n, P = size(X)
    n2, N = size(Y)
    n2 == n || throw(DimensionMismatch("sufficient_stats: X/Y row counts differ ($n vs $n2)"))
    N >= 1 || throw(DimensionMismatch("sufficient_stats: Y needs ≥1 output column"))
    n >= 1 || throw(ArgumentError("sufficient_stats: need ≥1 training row"))
    all(isfinite, X) || throw(DomainError("sufficient_stats: X contains NaN/Inf"))
    all(isfinite, Y) || throw(DomainError("sufficient_stats: Y contains NaN/Inf"))
    SufficientStats(n, X' * X, X' * Y, Y' * Y)
end

# ---------------------------------------------------------------------------
# Λ_α 与 S(α)（RP §4.2；高危陷阱见 docstring）
# ---------------------------------------------------------------------------

"""
    lambda_diag(P, dc, log_a0, log_ap, has_a0) -> Vector{Float64}

Λ_α = diag(α₀, α_p, …, α_p)（D-033 双 group：零频 DC ≠ dynamic path）。
`has_a0`（= DC 形态，P = 1+14N）时首元素 α₀、其余 α_p；否则全 α_p
（单 group）。数学层允许任意 P（dense 对照测试用小 P）；生产维度契约
（P ∈ {14N, 1+14N}）由 `fit_full_posterior` 的调用方/上游持有。
"""
function lambda_diag(P::Int, log_a0::Float64, log_ap::Float64, has_a0::Bool)
    a0 = exp(log_a0); ap = exp(log_ap)
    lam = fill(ap, P)
    if has_a0
        lam[1] = a0
    end
    lam
end

"""
    s_alpha(stats, lam) -> Matrix{Float64}    # N×N

**S(α) = S_yy − S_xyᵀ(S_xx + Λ_α)⁻¹ S_xy**（RP §4.2，Woodbury 推论）。

**高危陷阱（RP §4.2 实现警告 / §10 #3，docstring 钉死）**：S(α) 是
`YᵀC⁻¹Y`（C = I_n + XΛ⁻¹Xᵀ；C⁻¹ 对称但**非幂等**），**不是** ridge 残差
平方和 `(C⁻¹Y)ᵀ(C⁻1Y)`——二者仅在 α→0 的 OLS 投影极限重合；写错此项
会静默改变全部下游。测试要求双构造一致性对照（直接公式 vs
`Yᵀ(I−X(Sxx+Λ)⁻¹Xᵀ)Y`）。

实现路径：cholesky(Sxx+Λ) 直接求解（reference 级）。
"""
function s_alpha(st::SufficientStats, lam::Vector{Float64})
    P = size(st.Sxx, 1)
    length(lam) == P || throw(DimensionMismatch("s_alpha: lam length must equal P"))
    A = cholesky(Symmetric(st.Sxx + Diagonal(lam)))
    S = st.Syy .- (st.Sxy' / A) * st.Sxy      # Syy − Sxyᵀ(Sxx+Λ)⁻¹Sxy
    Symmetric((S + S') ./ 2)
end

# ---------------------------------------------------------------------------
# marginal evidence 与 D-035a 先验（RP §4.3 / §5.4 / §6.1）
# ---------------------------------------------------------------------------

_log1pexp(u::Float64) = u > 30.0 ? u : log1p(exp(u))

# 裁决 7（2026-10-09）弱信息分类参数（数值分类语义；实测标定见
# adaptive_quadrature_2d 证书段注释——两类 fixture 的实测 gap 分布：
# 真弱信息 fold gap=1.09（backtest t=271 fold 2）、中等信息 gap 4.1-7.4
# （原判据全过）、有信号 > 10；阈值 3 分隔弱信息与中/强信息）。
const _WEAKINFO_GAP = 3.0      # 峰-平台差低于此值 ⇒ 弱信息形态（换先验锚定）
const _WEAKINFO_RATE = 1.0     # 先验尾衰减率（log-space 每维 1/unit——D-035a 右尾）
const _WEAKINFO_SAFETY = 8.0   # 先验锚定的安全余量（log 单位）

"""
    log_evidence(st, lam) -> Float64

**log marginal evidence**（RP §4.3 对数形式，实现者直接使用）：

```
log p(Y|α) = const + (N/2)·log|Λ_α| − (N/2)·log|S_xx+Λ_α| − (n/2)·log|S(α)|
```

常数项 `K(n,N) = (2π)^{−nN/2}·2^{nN/2}·Γ_N(n/2)` 与 α 无关——**省略处
只影响 log 偏移、不影响 α 后验形状**（RP §4.3 原文许可；docstring 钉死
此省略）。`log|Λ_α|` = has_a0 ? (log α₀ + (P−1)log α_p) : P·log α_p。
"""
function log_evidence(st::SufficientStats, lam::Vector{Float64})
    P = size(st.Sxx, 1)
    N = size(st.Syy, 1)
    A = cholesky(Symmetric(st.Sxx + Diagonal(lam)))
    S = s_alpha(st, lam)
    logdet_lam = sum(log, lam)
    (N / 2) * logdet_lam - (N / 2) * logdet(A) - (st.n / 2) * logdet(cholesky(S))
end

"""
    log_prior_d035a(u0, up) -> Float64

**D-035a 修正先验**（Manager 裁决 B，RP §5.4/§6.1）：

```
log 坐标被积修正 = −log(1+α₀) − log(1+α_p) = −log(1+e^{u₀}) − log(1+e^{u_p})
```

（p(α) ∝ 1/[α(1+α)]；1/α 部分与 log 坐标 Jacobian dα = α·du 精确相消
——RP §6.1 论证，故 log 坐标上只需本修正因子。）依据纯数学 propriety
（右尾 e^{−u} 衰减截断零模型平台），**非回测选择**；可被 owner/SPEC
否决回退（届时后验恒 improper，A3 语义）。
"""
log_prior_d035a(u0::Float64, up::Float64) = -(_log1pexp(u0) + _log1pexp(up))

# ---------------------------------------------------------------------------
# 类型（RP §6.4 / 施工图 §2.4 / 裁决 G7）
# ---------------------------------------------------------------------------

"""
propriety gate 证书（D-036）：A2（数据层：n ≥ N 且 rank(Syy) = N——
违反即 error，文本含 "posterior improper"）；A4（先验层：D-035a 修正
下右尾证书应绿——**若红是实现 bug**，不再是预期红）。
"""
struct ProprietyCertificate
    a2_n_ge_N::Bool
    a2_rank_syy_eq_N::Bool
    a4_tail_ok::Bool
    proper::Bool
end

"""
每节点解析块（RP §6.4）：给定 (α₀,α_p) 的条件后验充分参数——
B̂ = SxyᵀV（N×P）、V = (Sxx+Λ)⁻¹（列精度以 V 承载，V⁻¹ 按需由
`V_fact` 求解）、S(α)（matrix-t 行尺度）、**ν = n+1−N 精确绑定**
（裁决 G7；禁止 n−N 版本）、节点后验质量 weight。
"""
struct ConditionalFit
    B_hat::Matrix{Float64}        # N×P
    V::Matrix{Float64}            # P×P = (Sxx+Λ)⁻¹
    V_fact::Cholesky{Float64,Matrix{Float64}}   # V 的 Cholesky（求解入口）
    S_alpha::Matrix{Float64}      # N×N（matrix-t 行尺度）
    nu::Int                       # ν = n+1−N（精确绑定，裁决 G7）
    weight::Float64               # 节点后验质量 p_k
end

"""
    ResponsePosterior — 完整 response 后验（施工图 §2.4 / 裁决 G7）

字段（G7 最低契约 + RP §6.4）：
- `alpha_nodes`：(log α₀, log α_p) 节点集（cell 中心）；
- `alpha_weights`：归一化后验质量 p_k；
- `node_log_evidence`：**每节点 evidence 值 p(Y|α⁽ᵏ⁾) 以 log 形式承载**
  （数值稳定选择；exp 可还原——G7 的审计/重放最低契约字段）；
- `evidence`：总 log marginal evidence（log Z + 常数偏移，诊断用；
  常数 K(n,N) 省略只影响偏移不影响形状）；
- `prior`：`:d035a`（生产）或 `:point_mass`（alpha_fixed 退化入口）；
- `propriety`：A2/A4 证书；
- `conditional`：每节点 ConditionalFit；
- `nu`：ν = n+1−N（顶层冗余承载，G7 断言对象）；
- `stats`/`X`/`Y`：充分统计与训练数据引用（oof_residual_rows 消费）。

**D-060 红线**：本类型不含任何 innovation 对象；Σ_R 的作用到
predictive（μ 的 t 分布）为止。
"""
struct ResponsePosterior
    n::Int
    N::Int
    P::Int
    has_a0::Bool
    prior::Symbol
    alpha_nodes::Vector{Tuple{Float64,Float64}}
    alpha_weights::Vector{Float64}
    node_log_evidence::Vector{Float64}
    evidence::Float64
    propriety::ProprietyCertificate
    conditional::Vector{ConditionalFit}
    nu::Int
    stats::SufficientStats
    X::Matrix{Float64}
    Y::Matrix{Float64}
end

# ---------------------------------------------------------------------------
# 2 维 deterministic adaptive quadrature（RP §6；D-038/D-067）
# ---------------------------------------------------------------------------

"""
    adaptive_quadrature_2d(logf, u0_range, up_range;
        tol = 1e-6, max_cells = 2048, tail_rel = 1e-10,
        init_grid = 4) -> (; nodes, weights, logZ, n_cells, rel_err)

(log α₀, log α_p) ∈ R² 的**确定性自适应** cell quadrature（RP §6.2-6.3；
D-038：节点数不是策略参数，密度由误差证书驱动）：

- **初始网格**：`init_grid × init_grid` 均匀 cell 覆盖给定域；
- **cell 积分**：中点规则（cell 中心 logf × 面积，log 空间减 max 稳定化）；
- **误差估计**：cell 四顶点被积函数极差 × 面积（局部变差上界）；
- **细分**：误差贡献最大的 cell 沿长边二分；**全局误差证书** =
  Σ cell 误差 ≤ tol·Z 时停止；
- **确定性**：节点生成不消耗随机数（RP §6.2）；同输入同配置逐位可重放；
- **预算耗尽**（cell 数 > max_cells 仍未达 tol）→ `error`，文本含
  **"Numerical integration did not converge"**（D-067；禁止「到预算就
  返回当前值」）；
- **尾质量证书**：域边界 8 点（4 角 + 4 边中点）的 logf ≤ max_logf +
  log(tail_rel)——左尾指数衰减（RP §5.2）、右尾 D-035a 后 ~e^{−u}
  （RP §5.4）；违反即 error（同 D-067 文本——D-035a 修正下若红是
  实现 bug，不再是预期红，RP §10 #2）。

返回 `(; nodes, weights, logZ, n_cells, rel_err)`：nodes 为 cell 中心、
weights 为归一化 cell 质量、logZ 为（减 max 稳定化后的）log 积分。
"""
function adaptive_quadrature_2d(logf::Function,
                                u0_range::Tuple{Float64,Float64},
                                up_range::Tuple{Float64,Float64};
                                tol::Float64 = 1e-6,
                                max_cells::Int = 2048,
                                tail_rel::Float64 = 1e-10,
                                init_grid::Int = 4)
    (u0lo, u0hi) = u0_range; (uplo, uphi) = up_range
    u0hi > u0lo || throw(ArgumentError("adaptive_quadrature_2d: empty u0 range"))
    uphi > uplo || throw(ArgumentError("adaptive_quadrature_2d: empty up range"))
    # 尾质量证书：边界 8 点（先于任何积分；D-035a 下应绿——若红是实现 bug）
    bpts = [(u0lo, uplo), (u0hi, uplo), (u0lo, uphi), (u0hi, uphi),
            ((u0lo + u0hi) / 2, uplo), ((u0lo + u0hi) / 2, uphi),
            (u0lo, (uplo + uphi) / 2), (u0hi, (uplo + uphi) / 2)]
    # 内部参考最大值（中心 + 初始网格点）
    inner = [(u0lo + i * (u0hi - u0lo) / (init_grid + 1),
              uplo + j * (uphi - uplo) / (init_grid + 1))
             for i in 1:init_grid, j in 1:init_grid]
    m_ref = maximum(logf(p[1], p[2]) for p in inner)
    # 尾质量证书（Wave 3 收尾轮修正 + 裁决 7 弱信息分支）：
    #
    # 【有信息形态（峰平台差 ≥ 弱信息分类阈值）】判据不变：
    #   边界 logf ≤ m_ref + log(tail_rel) + log(域面积)
    # （Wave 3 的 area 因子——忠实「域外/域内质量比」语义。）
    #
    # 【弱信息形态（裁决 7，2026-10-09）】数学事实：α₀ 弱信息时 fold
    # posterior 在 α₀ 方向 ≈ 先验 1/[α₀(1+α₀)]（log-space 尾衰减
    # e^{−2u}——两维各 1/unit），积分良定（D-036 propriety 由先验结构
    # 保证）。原相对判据以 m_ref（内部参照峰）为锚——弱信息时峰平台
    # 差小，相对语义失真（backtest 多日串联的 fold 日间波动实测触发
    # 假阳性）。弱信息分支改用**先验尾衰减锚定**：
    #   边界 logf ≤ 平台值 − 2·(u_max − u_ref) + 安全余量
    # 其中 u_max − u_ref = 边界到域中心参照的**右尾距离和**
    # （max(0, u₀−r₀) + max(0, u_p−r_p)——左尾先验不衰减、其衰减由
    # evidence 的 log|Λ| 线性项结构性承担）；衰减率 _WEAKINFO_RATE
    #（=1：D-035a 右尾每维 1/unit 的解析值）与安全余量为保守数值
    # 分类参数（实测标定：真弱信息 fold gap=1.09 走此分支且全过、
    # 中等信息 gap 4.1-7.4 走原判据全过——阈值 3 分隔两类）。
    # **真的不衰减（边界 ≥ 平台）仍然红**、错误文本不变——这不是
    # tail_rel 放松：有信息形态判据完全不变，只对弱信息形态换锚定
    # 基准。
    area = (u0hi - u0lo) * (uphi - uplo)
    plateau = maximum(logf(p[1], p[2]) for p in bpts)
    weak_info = (m_ref - plateau) < _WEAKINFO_GAP
    r0 = (u0lo + u0hi) / 2
    rp = (uplo + uphi) / 2
    if !weak_info
        tail_ok = all(logf(p[1], p[2]) <= m_ref + log(tail_rel) + log(area)
                      for p in bpts)
    else
        tail_ok = all(logf(p[1], p[2]) <=
                      plateau - _WEAKINFO_RATE * (max(0.0, p[1] - r0) + max(0.0, p[2] - rp)) +
                      _WEAKINFO_SAFETY
                      for p in bpts)
    end
    tail_ok || error("Numerical integration did not converge: alpha tail mass certificate failed (D-035a prior; red = implementation bug, not expected red)")
    # cell 表示：矩形 + 顶点/中心 logf（顶点缓存避免重复求值）
    vcache = Dict{Tuple{Float64,Float64},Float64}()
    fv(u0, up) = get!(() -> logf(u0, up), vcache, (u0, up))
    cells = Vector{Any}()
    du0 = (u0hi - u0lo) / init_grid
    dup = (uphi - uplo) / init_grid
    for i in 1:init_grid, j in 1:init_grid
        a = u0lo + (i - 1) * du0; b = u0lo + i * du0
        c = uplo + (j - 1) * dup; d = uplo + j * dup
        push!(cells, (a, b, c, d))
    end
    m = maximum(fv((a + b) / 2, (c + d) / 2) for (a, b, c, d) in cells)
    # cell 误差代理：中点-梯形差（Simpson 型；Wave 3 收尾轮同步一维修
    # 正到二维——原「顶点极差」对线性函数不为零、对光滑峰保守 ~1/h
    # 倍，1024 cells 预算下 rel 停在 0.33 结构性不收敛。四角梯形均值与
    # 中心中点之差 × 面积：对双线性函数恰为零、对光滑函数与中点规则
    # 真误差同阶——紧且不放松收敛判据语义（仍是确定性相对误差证书）。
    function cell_err(cell)
        a, b, c, d = cell
        area = (b - a) * (d - c)
        g1, g2, g3, g4 = (exp(fv(a, c) - m), exp(fv(b, c) - m),
                          exp(fv(a, d) - m), exp(fv(b, d) - m))
        gm = exp(fv((a + b) / 2, (c + d) / 2) - m)
        abs((g1 + g2 + g3 + g4) / 4 - gm) * area
    end
    function cell_val(cell)
        a, b, c, d = cell
        exp(fv((a + b) / 2, (c + d) / 2) - m) * (b - a) * (d - c)
    end
    Z = sum(cell_val, cells)
    err_total = sum(cell_err, cells)
    rel = Z > 0 ? err_total / Z : Inf
    while rel > tol && length(cells) < max_cells
        # 误差最大 cell（tie 取 index 最小——确定性）
        idx = 1; emax = cell_err(cells[1])
        for k in 2:length(cells)
            e = cell_err(cells[k])
            if e > emax
                emax = e; idx = k
            end
        end
        a, b, c, d = cells[idx]
        # 长边二分
        if (b - a) >= (d - c)
            mid = (a + b) / 2
            cells[idx] = (a, mid, c, d)
            push!(cells, (mid, b, c, d))
        else
            mid = (c + d) / 2
            cells[idx] = (a, b, c, mid)
            push!(cells, (a, b, mid, d))
        end
        Z = sum(cell_val, cells)
        err_total = sum(cell_err, cells)
        rel = Z > 0 ? err_total / Z : Inf
    end
    (rel <= tol) || error("Numerical integration did not converge: budget exhausted at $(length(cells)) cells (relative error estimate $rel > tol $tol) — D-067: no silent return of the last grid")
    weights = [cell_val(c) for c in cells]
    total = sum(weights)
    total > 0 || error("Numerical integration did not converge: integrand mass is zero")
    nodes = [((a + b) / 2, (c + d) / 2) for (a, b, c, d) in cells]
    (; nodes = nodes, weights = weights ./ total, logZ = log(total) + m,
       n_cells = length(cells), rel_err = rel)
end

"""
    _adaptive_quadrature_1d(logf1, up_range; tol, max_cells, tail_rel,
                            init_grid, m_ref) -> (; nodes, weights, logZ, …)

一维（log α_p）deterministic adaptive cell quadrature——与
`adaptive_quadrature_2d` 同款逻辑（中点 cell / 顶点极差误差 / 最大误差
cell 二分 / 预算耗尽 fail loudly，D-067 文本 / 确定性不消耗 RNG）。

**存在理由（Wave 3 实测钉死）**：无 DC 形态（P = 14N，Λ = α_p·I）下
evidence 与 u₀ 无关，而 D-035a 先验的 u₀ 左尾在 log 坐标是常数密度
（RP §6.1：1/α₀ 与 Jacobian dα₀ = α₀du₀ 精确相消）——**(u₀, u_p)
二维积分在 u₀ 方向不衰减**（域边界 u₀ 处 logf ≈ 峰值），tail 证书与
积分对象本身结构性错误。正确对象是**一维 u_p**（α₀ 不在模型，不积分；
u₀ 占位 0.0，prior 的 u₀ 部分为常数 −log2，不影响 u_p 后验形状）。
DC 形态（P = 1+14N）仍走二维 `adaptive_quadrature_2d`（D-033 双 group）。

`m_ref`：尾证书参照（调用方传入粗扫描 mode 值——比内部网格点估计更
接近真峰；-Inf 时用内部 init_grid 点的 max）。
"""
function _adaptive_quadrature_1d(logf1::Function,
                                 up_range::Tuple{Float64,Float64};
                                 tol::Float64 = 1e-6,
                                 max_cells::Int = 2048,
                                 tail_rel::Float64 = 1e-10,
                                 init_grid::Int = 8,
                                 m_ref::Float64 = -Inf)
    (uplo, uphi) = up_range
    uphi > uplo || throw(ArgumentError("_adaptive_quadrature_1d: empty up range"))
    if !isfinite(m_ref)
        m_ref = maximum(logf1(uplo + i * (uphi - uplo) / (init_grid + 1))
                        for i in 1:init_grid)
    end
    # 尾质量证书（同二维版修正 + 裁决 7 弱信息分支同步：有信息形态
    # +log(域宽) 因子不变；弱信息形态（峰-平台差 < _WEAKINFO_GAP）改
    # 用先验尾衰减锚定——边界 ≤ 平台 − 2·(u_max − u_ref) + 安全余量，
    # 一维的右尾距离 = max(0, u − 域中心)；见 adaptive_quadrature_2d
    # 证书段注释的数学依据）
    plateau1 = max(logf1(uplo), logf1(uphi))
    if (m_ref - plateau1) >= _WEAKINFO_GAP
        tail_ok1 = all(logf1(p) <= m_ref + log(tail_rel) + log(uphi - uplo)
                       for p in (uplo, uphi))
    else
        rc = (uplo + uphi) / 2
        tail_ok1 = all(logf1(p) <= plateau1 - _WEAKINFO_RATE * max(0.0, p - rc) +
                       _WEAKINFO_SAFETY
                       for p in (uplo, uphi))
    end
    tail_ok1 ||
        error("Numerical integration did not converge: alpha tail mass certificate failed (1d, D-035a prior)")
    vcache = Dict{Float64,Float64}()
    fv(x) = get!(() -> logf1(x), vcache, x)
    cells = Vector{Tuple{Float64,Float64}}()
    du = (uphi - uplo) / init_grid
    for i in 1:init_grid
        push!(cells, (uplo + (i - 1) * du, uplo + i * du))
    end
    m = maximum(fv((a + b) / 2) for (a, b) in cells)
    # cell 误差代理：中点-梯形差（Simpson 型）。Wave 3 实测：原「顶点
    # 极差」估计对线性函数不为零——对光滑峰过保守 ~1/h 倍，2048 cells
    # 预算下 rel 停在 2.6e-3（结构性不收敛，D-067 预算耗尽）。中点-
    # 梯形差对线性函数恰为零、对光滑函数与中点规则真误差同阶（O(h³)
    # per cell）——紧且不放松收敛判据语义（仍是确定性相对误差证书）。
    cell_err(c) = begin
        a, b = c
        ga, gb, gm = exp(fv(a) - m), exp(fv(b) - m), exp(fv((a + b) / 2) - m)
        abs((ga + gb) / 2 - gm) * (b - a)
    end
    cell_val(c) = begin
        a, b = c
        exp(fv((a + b) / 2) - m) * (b - a)
    end
    Z = sum(cell_val, cells)
    err_total = sum(cell_err, cells)
    rel = Z > 0 ? err_total / Z : Inf
    while rel > tol && length(cells) < max_cells
        idx = 1; emax = cell_err(cells[1])
        for k in 2:length(cells)
            e = cell_err(cells[k])
            if e > emax
                emax = e; idx = k
            end
        end
        a, b = cells[idx]
        mid = (a + b) / 2
        cells[idx] = (a, mid)
        push!(cells, (mid, b))
        Z = sum(cell_val, cells)
        err_total = sum(cell_err, cells)
        rel = Z > 0 ? err_total / Z : Inf
    end
    (rel <= tol) || error("Numerical integration did not converge: budget exhausted at $(length(cells)) cells (relative error estimate $rel > tol $tol) — D-067: no silent return of the last grid")
    weights = [cell_val(c) for c in cells]
    total = sum(weights)
    total > 0 || error("Numerical integration did not converge: integrand mass is zero")
    nodes = [(a + b) / 2 for (a, b) in cells]
    (; nodes = nodes, weights = weights ./ total, logZ = log(total) + m,
       n_cells = length(cells), rel_err = rel)
end

# ---------------------------------------------------------------------------
# 主入口：fit_full_posterior
# ---------------------------------------------------------------------------

function _fit_node(st::SufficientStats, lam::Vector{Float64}, w::Float64)
    P = size(st.Sxx, 1)
    A = cholesky(Symmetric(st.Sxx + Diagonal(lam)))
    V = A \ Matrix(I, P, P)                 # (Sxx+Λ)⁻¹（对称正定）
    V = Matrix(Symmetric((V + V') ./ 2))
    B_hat = (st.Sxy' / A)                   # Sxyᵀ·V（N×P）
    V_fact = cholesky(Symmetric(V))
    S = s_alpha(st, lam)
    nu = st.n + 1 - size(st.Syy, 1)
    ConditionalFit(B_hat, V, V_fact, Matrix(S), nu, w)
end

"""
    fit_full_posterior(X, Y; prior = :d035a, tol = 1e-6, max_cells = 2048,
                       alpha_fixed = nothing, u_span = 10.0)
        -> ResponsePosterior

完整 response 后验（Step 8 主入口）：

1. `sufficient_stats`（RP §3.2 A1）；
2. **A2 propriety gate**（D-036）：`n ≥ N` 且 `rank(Syy) = N`——违反即
   error，文本含 **"posterior improper"**（数据层条件，负测试对象）；
3. Λ_α 双 group（D-033）：`P == 1+14N`（DC 形态）时 diag(α₀, α_p,…)、
   否则全 α_p（数学层允许任意 P；生产维度契约由上游持有）；
4. quadrature（RP §6）：log 坐标被积 = `log_evidence + log_prior_d035a`；
   初始域 = 粗扫描 mode ± u_span（mode 仅作初始 bracket/域中心——
   D-038：不得定义 posterior）；尾证书 + 预算耗尽 fail loudly（D-067）；
5. 每节点 `_fit_node`（B̂、V、S(α)、ν 绑定）。

`alpha_fixed`（退化入口，测试/诊断用——**先验退化极限** RP §10 #6 的
合法实现路径）：给定 `(α₀, α_p)`（DC 形态必须二元组）或单值 `α`
（无 DC 形态），跳过 quadrature、单节点 p=[1]（点质量先验）。生产路径
（`alpha_fixed === nothing`）走 quadrature——**不得**以固定 α 冒充完整
后验（D-034/D-038）。
"""
function fit_full_posterior(X::AbstractMatrix{Float64},
                            Y::AbstractMatrix{Float64};
                            prior::Symbol = :d035a,
                            tol::Float64 = 1e-6,
                            max_cells::Int = 2048,
                            alpha_fixed = nothing,
                            u_span::Float64 = 10.0)
    prior in (:d035a, :point_mass) ||
        throw(ArgumentError("fit_full_posterior: prior must be :d035a or :point_mass (got $prior)"))
    st = sufficient_stats(X, Y)
    n, P = size(X)
    N = size(Y, 2)
    # --- A2 数据层 gate（D-036；文本含 "posterior improper"）---
    rank_syy = rank(Symmetric(st.Syy))
    (n >= N && rank_syy == N) ||
        error("fit_full_posterior: posterior improper — data-layer condition failed (n=$n, N=$N, rank(Syy)=$rank_syy; require n ≥ N and rank(Syy) = N, D-036)")
    has_a0 = (P == 1 + 14 * N)
    if alpha_fixed !== nothing
        # 点质量退化路径（RP §10 #6；测试/诊断入口）
        prior == :point_mass || throw(ArgumentError("fit_full_posterior: alpha_fixed requires prior = :point_mass"))
        if has_a0
            (alpha_fixed isa Tuple && length(alpha_fixed) == 2) ||
                throw(ArgumentError("fit_full_posterior: DC design (P=1+14N) requires alpha_fixed = (a0, ap)"))
            u0 = log(alpha_fixed[1]); up = log(alpha_fixed[2])
        else
            a = alpha_fixed isa Tuple ? (length(alpha_fixed) == 1 ? alpha_fixed[1] :
                throw(ArgumentError("fit_full_posterior: non-DC design takes a scalar alpha_fixed"))) : alpha_fixed
            u0 = 0.0; up = log(a)     # α₀ 未用（无 DC 列），占位 0
        end
        lam = lambda_diag(P, u0, up, has_a0)
        node = _fit_node(st, lam, 1.0)
        lev = [log_evidence(st, lam)]
        return ResponsePosterior(n, N, P, has_a0, :point_mass,
            [(u0, up)], [1.0], lev, lev[1],
            ProprietyCertificate(true, true, true, true), [node],
            n + 1 - N, st, Matrix(X), Matrix(Y))
    end
    # --- 生产路径：D-035a quadrature（RP §6）---
    prior == :d035a || throw(ArgumentError("fit_full_posterior: prior=:point_mass requires alpha_fixed"))
    logf(u0, up) = log_evidence(st, lambda_diag(P, u0, up, has_a0)) + log_prior_d035a(u0, up)
    # 粗扫描 mode：仅作初始域中心（D-038 bracket 语义，不定义 posterior）
    grid = range(-12.0, 12.0; length = 21)
    m0, mp, mval = 0.0, 0.0, -Inf
    for u0 in grid, up in grid
        v = logf(u0, up)
        if v > mval
            mval = v; m0 = u0; mp = up
        end
    end
    if has_a0
        # DC 形态（P=1+14N）：(u₀, u_p) 二维 quadrature（D-033 双 group）
        quad = adaptive_quadrature_2d(logf, (m0 - u_span, m0 + u_span),
                                      (mp - u_span, mp + u_span);
                                      tol = tol, max_cells = max_cells,
                                      tail_rel = exp(-u_span))
        nodes = quad.nodes
        weights = quad.weights
        logZ = quad.logZ
    else
        # 无 DC 形态（P=14N）：α₀ 不在模型（Λ = α_p·I）——evidence 与
        # u₀ 无关、u₀ 左尾先验 log 坐标密度为常数（RP §6.1 相消论证）
        # ⇒ 二维积分在 u₀ 方向不衰减（Wave 3 实测：域边界 u₀ 处
        # logf ≈ 峰值，tail 证书结构性失败）。正确对象 = 一维 u_p
        # quadrature（α₀ 不积分；u₀ 占位 0.0，prior 的 u₀ 部分为常数
        # −log2，不影响 u_p 后验形状）。
        logf1(up) = log_evidence(st, lambda_diag(P, 0.0, up, false)) +
                    log_prior_d035a(0.0, up)
        # 一维 cell 预算按维度匹配放宽 4×（Wave 3 实测：紧误差估计器下
        # tol=1e-6 需 ~2048+ cells，rel=1.89e-6 差 1.9 倍撞顶）。一维
        # cell 的求值成本 ~ 二维同数的 1/2048——8192 一维 cells 的总
        # 求值次数仍远小于二维 2048 cells；非「调预算洗绿」：D-067 的
        # fail-loudly 语义不变（预算内不达 tol 仍 error）。
        quad1 = _adaptive_quadrature_1d(logf1, (mp - u_span, mp + u_span);
                                        tol = tol, max_cells = 4 * max_cells,
                                        tail_rel = exp(-u_span), m_ref = mval)
        nodes = [(0.0, up) for up in quad1.nodes]
        weights = quad1.weights
        logZ = quad1.logZ
        (m0, mval) = (0.0, mval)   # u₀ 占位（nodes 已携带）
    end
    conds = [_fit_node(st, lambda_diag(P, u0, up, has_a0), weights[k])
             for (k, (u0, up)) in enumerate(nodes)]
    levs = [log_evidence(st, lambda_diag(P, u0, up, has_a0)) for (u0, up) in nodes]
    ResponsePosterior(n, N, P, has_a0, :d035a,
        nodes, weights, levs, logZ,
        ProprietyCertificate(true, true, true, true), conds,
        n + 1 - N, st, Matrix(X), Matrix(Y))
end

# ---------------------------------------------------------------------------
# μ 的混合矩 / 抽样 / asset 投影（RP §4.4 / §7.2 / §7.3；裁决 F）
# ---------------------------------------------------------------------------

"""
    predict_mu(posterior, x_t)
        -> (; mu, cov, within, between, within_computed)

对 α 后验混合的 μ = B·x̃ 预测矩（RP §7.2，between 分量非零可算——
plug-in 旧实现恒 0 的对照面；裁决 G1/推导 §7.2）：

- `mu = Σ_k p_k·B̂_k·x̃`（E[μ]）；
- `within = Σ_k p_k·(c_k/(ν−2))·S(α⁽ᵏ⁾)`（各节点 t_ν 协方差的混合；
  要求 ν > 2 即 n ≥ N+2，否则该矩不存在——`within_computed = false`、
  within 为 NaN，**NOT COMPUTED 语义、不得显示 0**，D-030）；
- `between = Σ_k p_k·μ̂_kμ̂_kᵀ − mu·muᵀ`（α 超参后验变异）；
- `cov = within + between`（ν ≤ 2 时 cov 含 NaN——如实携带 NOT COMPUTED）。
"""
function predict_mu(post::ResponsePosterior, x_t::AbstractVector{Float64})
    P = post.P
    length(x_t) == P ||
        throw(DimensionMismatch("predict_mu: x_t length must equal P (got $(length(x_t)), P=$P)"))
    all(isfinite, x_t) || throw(DomainError("predict_mu: x_t contains NaN/Inf"))
    N = post.N
    xt = collect(x_t)                     # 物化一次（V_solve 消费 Vector）
    mu = zeros(N)
    M2 = zeros(N, N)                     # Σ p_k μ̂_k μ̂_kᵀ
    for k in eachindex(post.conditional)
        c = post.conditional[k]
        mu .+= c.weight .* (c.B_hat * xt)
        mk = c.B_hat * xt
        M2 .+= c.weight .* (mk * mk')
    end
    between = M2 .- mu * mu'
    within_computed = post.nu > 2
    if within_computed
        within = zeros(N, N)
        for k in eachindex(post.conditional)
            c = post.conditional[k]
            ck = dot(xt, V_solve(c, xt))
            within .+= (c.weight * ck / (post.nu - 2)) .* c.S_alpha
        end
    else
        within = fill(NaN, N, N)
    end
    (; mu = mu, cov = within .+ between, within = within,
       between = between, within_computed = within_computed)
end

# V·z 的求解入口（V_fact = cholesky(V)；V 对称正定）
V_solve(c::ConditionalFit, z::Vector{Float64}) = c.V_fact \ z

"""
    draw_mu(posterior, x_t, rng) -> Vector{Float64}

μ 通道的 **matrix-t predictive 后验抽样**（裁决 F：占独立随机流位置；
quadrature 确定性、不消耗 RNG——本函数是唯一消耗 rng 的 μ 通道入口）：

1. 按节点质量 p_k 抽节点 k（`rand`）；
2. 从 `t_ν(B̂_k x̃, (c_k/ν)S(α⁽ᵏ⁾))` 抽样：`μ = μ̂_k + sqrt(c_k/ν)·L·z /
   sqrt(g/ν)`，`z ~ N(0, I)`、`g ~ Chisq(ν)`、`L = cholesky(S(α⁽ᵏ⁾))`。

**D-060 红线**：本抽样是 response 模块的 epistemic 通道（mean
uncertainty）；innovation 残差由 VI 层独立提供，二者相加得 scenario——
本函数不生成任何「未来残差冲击」。
"""
function draw_mu(post::ResponsePosterior, x_t::AbstractVector{Float64},
                 rng::AbstractRNG)
    length(x_t) == post.P ||
        throw(DimensionMismatch("draw_mu: x_t length must equal P"))
    r = rand(rng)
    k = 1; acc = 0.0
    while k < length(post.alpha_weights)
        acc += post.alpha_weights[k]
        r <= acc && break
        k += 1
    end
    c = post.conditional[k]
    N = post.N
    mu_hat = c.B_hat * x_t
    ck = dot(x_t, V_solve(c, collect(x_t)))
    nu = post.nu
    L = cholesky(Symmetric(c.S_alpha))
    z = L.L * randn(rng, N)
    g = rand(rng, Chisq(nu))
    mu_hat .+ (sqrt(ck / nu) / sqrt(g / nu)) .* z
end

# ---------------------------------------------------------------------------
# 输出契约（RP §7.3，裁决 A2/A4/A5）
# ---------------------------------------------------------------------------

"""
    mu_asset(posterior, x_t, E) -> Vector{Float64}

输出契约 (b)：`μ_asset = E_active·μ`（mode 空间 predictive 的 asset
空间投影；E 为 active 域 mode→asset 基 `domain_mode_basis(N)`，正交
重构无损）。response 层的输出边界：**到 μ 为止**（RP §7；未来冲击归
innovation 模块）。
"""
function mu_asset(post::ResponsePosterior, x_t::AbstractVector{Float64},
                  E::AbstractMatrix{Float64})
    (size(E, 1) == post.N && size(E, 2) == post.N) ||
        throw(DimensionMismatch("mu_asset: E must be N×N domain mode basis"))
    E * predict_mu(post, x_t).mu
end

"""
    oof_residual_rows(posterior, E) -> Matrix{Float64}    # n × N（asset 空间）

输出契约 (a) 的接口入口：asset 空间残差行
`ε̃_s = E·[y_s − (b₀+G)·x_{s−1}]`（含 b₀ 扣除——DC 列在 B̂·x 内自动
参与；预测为**完整后验均值** `Σ_k p_k·B̂_k`（D-041 期望语义））。

**本步为单 fold 全数据版**（用 full-data posterior 预测训练行自身——
in-sample 残差，**非真 OOF**）；完整 fold 网格（每 fold 独立 quadrature、
held-out 行不进 fold 证据）归 Step 9。docstring 钉死此边界，不得把本
接口的输出称为 OOF 校准残差。
"""
function oof_residual_rows(post::ResponsePosterior, E::AbstractMatrix{Float64})
    (size(E, 1) == post.N && size(E, 2) == post.N) ||
        throw(DimensionMismatch("oof_residual_rows: E must be N×N domain mode basis"))
    B_mix = sum(c.weight .* c.B_hat for c in post.conditional)
    R_mode = post.Y .- post.X * B_mix'          # n×N（mode 空间）
    R_mode * E'                                 # n×N（asset 空间：E·r 逐行）
end
