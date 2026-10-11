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
# matrix-t（RP §4.4）、quadrature（RP §6）、within/between 分解
# （RP §7.2）、输出契约（RP §7.3）。
# 先验：RP §5.4 的 D-035a（p(α)∝1/[α(1+α)]）已被 Gate-0 P0-8 裁决替换
# 为 proper hierarchical prior τ~HalfCauchy(0,1)（p(α)∝1/[√α(1+α)]，
# log 坐标 (1/2)u−log(1+e^u)）；RP §5.4 的 propriety 论证与 quadrature
# 语义不变（右尾仍衰减可积，tail 证书按统一判据签发，无弱信息特例分支）。
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
# 阶段 1：Λ 秩一预分解路径（FIRST-DAY-FIT-1；见
# docs/FIRST_DAY_FIT_COST_ACCELERATION.md）。仅内部使用、默认关闭。
# 数学恒等式：Λ = αp·I + (α0−αp)e1e1ᵀ ⇒ A = B + δe1e1ᵀ（B = Sxx+αp·I）；
# log|A| = log|B| + log(1+δ·e1ᵀB⁻¹e1)（行列式引理）；
# A⁻¹Sxy = B⁻¹Sxy − δ·(B⁻¹e1)((e1ᵀB⁻¹)Sxy)/(1+δ·e1ᵀB⁻¹e1)（SM）。
# 预分解 Sxx = Q·diag(λ)·Qᵀ 一次；逐节点 O(P) + O(P²N)。
# ---------------------------------------------------------------------------

struct SxxRankOneSpec
    Q::Matrix{Float64}
    lam::Vector{Float64}
    v::Vector{Float64}
    W::Matrix{Float64}
end

function _rank1_spec(st::SufficientStats)
    F = eigen(Symmetric(st.Sxx))
    Q = Matrix(F.vectors)
    lam = Vector{Float64}(F.values)
    SxxRankOneSpec(Q, lam, Q[1, :], Q' * st.Sxy)
end

function _log_evidence_rank1(st::SufficientStats, lam::Vector{Float64},
                             spec::SxxRankOneSpec)
    P = size(st.Sxx, 1)
    N = size(st.Syy, 1)
    α0 = lam[1]; αp = lam[2]
    all(==(αp), view(lam, 2:P)) ||
        error("_log_evidence_rank1: lam must be DC rank-one form [α0, αp…]")
    δ = α0 - αp
    d = spec.lam .+ αp
    all(>(0.0), d) ||
        error("_log_evidence_rank1: shifted eigenvalue ≤ 0 (numerical)")
    logdet_B = sum(log, d)
    # CS 层重排（2026-10-10；不改数学）：手写循环去分配（原 .^2 ./ d 双临时）。
    e1Be1 = 0.0
    @inbounds for i in 1:P
        e1Be1 += spec.v[i]^2 / d[i]
    end
    denom = 1.0 + δ * e1Be1
    (isfinite(denom) && denom > 0.0) ||
        error("_log_evidence_rank1: SM denominator nonpositive (numerical)")
    # CS 层重排（2026-10-10；不改数学）：两次 Q 乘合并为单次 GEMM（减少
    # BLAS 调用次数——小尺寸高频调用开销是 measured 瓶颈嫌疑）；修正项改
    # 广播（省一个 P×N 外积分配）。第二因子仍为 (B⁻¹e1)ᵀ·Sxy（修复保留）。
    QZ = spec.Q * hcat(spec.v ./ d, spec.W ./ d)
    Bie1 = QZ[:, 1]
    Bis = QZ[:, 2:(N + 1)]
    w1 = Bie1' * st.Sxy
    Ais = Bis .- (δ / denom) .* Bie1 .* w1
    Sraw = st.Syy .- st.Sxy' * Ais
    Sα = Symmetric((Sraw + Sraw') ./ 2)
    logdet_lam = sum(log, lam)
    # R4'''：Sα 的 logdet 改走 eigen-based 鲁棒路径（F1/F2；cholesky(Sα)
    # 在 n<P 早期行的低 α 端抛 PosDefException）。
    (N / 2) * logdet_lam - (N / 2) * (logdet_B + log(denom)) -
        (st.n / 2) * _robust_logdet_psd(Sα)
end

# 回退包装（阶段 1；fail-soft）：spec === nothing → 原路径；SM 防护断言
# 触发（文本含 "_log_evidence_rank1"）→ 逐点回退原路径；其它错误（含
# 原路径自身失败）原样外逸（fail loudly，不改现状语义）。
function _evidence_eval(st::SufficientStats, lam::Vector{Float64},
                        spec::Union{Nothing,SxxRankOneSpec})
    spec === nothing && return log_evidence(st, lam)
    try
        _log_evidence_rank1(st, lam, spec)
    catch err
        if err isa ErrorException && occursin("_log_evidence_rank1", err.msg)
            return log_evidence(st, lam)
        end
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# R4'''（2026-10-10；F1/F2：PosDefException @ cholesky(Sα)）——鲁棒 logdet + 防御
# ---------------------------------------------------------------------------
# 规格：docs/EARLY_ROW_FIT_FAILURE_DIAGNOSIS.md §8（R4' 前提被 F2 推翻后修订）。
# F2 实测：n=2000、P=911 粗扫描 441/441 全失败、中域 121 点；eigmin(Sα) 在
# [-2.65e-12, -2.70e-13]（数值噪声级）；n=14044 不崩（train 规模主导）。
# Sα 理论 PSD（n≫N），负特征值为舍入噪声 → eigen-based 鲁棒 logdet：
#   λ < -neg_tol（结构性非正定）→ fail loudly；
#   其余 λ floor 到 ε_floor = _PSD_FLOOR_REL·scale 后 sum(log)
#   （floor→0 refinement 要求，待 D-066 定稿）。
# 防御层：按类型捕获 PosDefException（A=Sxx+Λ 等其它分解）→ 该点 -Inf +
# 诊断；证书点/refval 点非有限 → fail loudly；中域/占比/全失败 → fail loudly；
# 最终节点构造 PosDef → fail loudly。数学对象不变（Sα 理论 PSD；floor 是
# 数值治理，参照 EB_COVARIANCE_FLOOR 先例但语义独立）。正常档零改变
# （eigen-logdet vs cholesky-logdet 差应 ~roundoff 级，验证核对）。
# 阈值：保守初始值，待 D-066 refinement 定稿；禁止由运行预算/回测选择。
const _PSD_FLOOR_REL = 1e-12    # ε_floor（λmax 相对）；floor→0 refinement 要求
const _PSD_NEG_TOL_REL = 1e-10  # 结构性负值判定（|λ| > 此值·scale ⇒ fail loudly）
const _POSDEF_MID_DOMAIN_WIDTH = 2.0
const _POSDEF_MAX_FAILURE_RATIO = 0.2
const _POSDEF_POINT_CAP = 32

# R4''' 核心：eigen-based 鲁棒 log|S|（N×N；N=65 时便宜、确定性）。
function _robust_logdet_psd(S::Symmetric{Float64,Matrix{Float64}})
    F = eigen(S)
    lam = F.values
    scale = max(maximum(abs, lam), 1.0)
    floor_abs = _PSD_FLOOR_REL * scale
    neg_tol = _PSD_NEG_TOL_REL * scale
    acc = 0.0
    @inbounds for i in eachindex(lam)
        li = lam[i]
        li < -neg_tol &&
            error("S_alpha structurally non-PSD: lambda_min=$li (< -$neg_tol, scale=$scale) — R4 fail-loud")
        li < floor_abs && (li = floor_abs)
        acc += log(li)
    end
    acc
end

# R4''' 诊断载体（不导出；经 fit_full_posterior 的 posdef_diag kw 就地接收）。
mutable struct PosDefDiagnostics
    attempts::Int
    failures::Int
    points::Vector{Tuple{Float64,Float64}}
end
PosDefDiagnostics() = PosDefDiagnostics(0, 0, Tuple{Float64,Float64}[])

# 防御软包装（logf 路径）：PosDef → -Inf + 记录；其余异常原样外逸。
function _soft_logf(raw_ev::Function, lam::Vector{Float64}, prior_val::Float64,
                    u0::Float64, up::Float64, diag::PosDefDiagnostics)
    diag.attempts += 1
    try
        return raw_ev(lam) + prior_val
    catch err
        err isa PosDefException || rethrow()
        diag.failures += 1
        length(diag.points) < _POSDEF_POINT_CAP && push!(diag.points, (u0, up))
        return -Inf
    end
end

# 防御软包装（纯 evidence 路径：levs 等）：点坐标由 lam 反推。
function _soft_evidence(raw_ev::Function, lam::Vector{Float64}, diag::PosDefDiagnostics)
    diag.attempts += 1
    try
        return raw_ev(lam)
    catch err
        err isa PosDefException || rethrow()
        diag.failures += 1
        u0 = log(lam[1]); up = log(lam[2])
        length(diag.points) < _POSDEF_POINT_CAP && push!(diag.points, (u0, up))
        return -Inf
    end
end

# 最终节点构造（输出对象）：PosDef → fail loudly（不得 -Inf）。
function _soft_fit_node(st::SufficientStats, lam::Vector{Float64}, w::Float64;
                        rank1::Bool, spec::Union{Nothing,SxxRankOneSpec})
    try
        return _fit_node(st, lam, w; rank1 = rank1, spec = spec)
    catch err
        err isa PosDefException || rethrow()
        error("PosDefException at final-node fit (S construction) — R4 fail-loud")
    end
end

# 拟合安全断言：中域失败 / 占比超限 → fail loudly（防御层）。
function _assert_posdef_safe(diag::PosDefDiagnostics, m0::Float64, mp::Float64;
                             check_u0::Bool = true)
    for (u0, up) in diag.points
        hit = check_u0 && abs(u0 - m0) <= _POSDEF_MID_DOMAIN_WIDTH
        hit |= abs(up - mp) <= _POSDEF_MID_DOMAIN_WIDTH
        hit && error("PosDefException inside quadrature mid-domain (point=($u0,$up), mode=($m0,$mp)) — R4 fail-loud")
    end
    diag.failures / max(diag.attempts, 1) > _POSDEF_MAX_FAILURE_RATIO &&
        error("PosDefException failure ratio $(diag.failures)/$(diag.attempts) > $_POSDEF_MAX_FAILURE_RATIO — R4 fail-loud")
    nothing
end

# RankOneMul（阶段 2；未导出）：A = Sxx+Λ = B + δe1e1ᵀ 的乘法上下文——
# A·z = Q(d .* (Qᵀz)) + δ·z₁·e1（O(P²)；替代物化 V/V_fact）。
# 定义位置必须先于 ConditionalFit 的字段类型引用（顺序修复）。
struct RankOneMul
    spec::SxxRankOneSpec
    d::Vector{Float64}
    δ::Float64
end

# ---------------------------------------------------------------------------
# marginal evidence 与 D-035a 先验（RP §4.3 / §5.4 / §6.1）
# ---------------------------------------------------------------------------

_log1pexp(u::Float64) = u > 30.0 ? u : log1p(exp(u))

# P0-8 裁决（2026-10-10）：删除 _WEAKINFO_GAP / _WEAKINFO_RATE /
# _WEAKINFO_SAFETY 弱信息特例机器——其阈值由回测 fixture 标定
# （backtest t=271 fold 2 gap=1.09、中等信息 gap 4.1-7.4），违反开发守则
# （禁止用回测 fixture 定数值分支）。HalfCauchy-τ 先验下 tail 证书按统一
# 判据签发（两端自然衰减），无分类分支。

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
    # R4'''：S 的 logdet 同走鲁棒路径（rank1/reference 一致）。
    (N / 2) * logdet_lam - (N / 2) * logdet(A) - (st.n / 2) * _robust_logdet_psd(S)
end

"""
    log_prior_d035a(u0, up) -> Float64

**proper hierarchical prior：τ ~ HalfCauchy(0,1) 对应的 log 坐标密度**
（Gate-0 P0-8 裁决恢复；函数名保留历史 `d035a` 标识——D-035a 的
p(α)∝1/[α(1+α)] 已被裁决换回 HalfCauchy-τ）：

```
p(α) ∝ 1/[√α·(1+α)]          (τ = α^{−1/2}，τ ~ HalfCauchy(0,1))
log p_u(u) = (1/2)·u − log(1+e^u) + C     (u = log α)
```

推导：p_u(u) = p(α)·|dα/du| = e^{u/2}/(1+e^u)，故 log p_u = (1/2)u −
log(1+e^u)。**两端自然衰减**：u→−∞ 时 ≈ e^{u/2}→0（指数衰减）；
u→+∞ 时 ≈ e^{−u/2}（可积）——proper prior，非经验标定、非回测选择。
右尾衰减率（每维 1/2 per unit）慢于旧 D-035a（每维 1/unit），但积分
仍良定；tail 证书按统一判据签发（P0-8 删除 _WEAKINFO_* 特例分支）。
"""
log_prior_d035a(u0::Float64, up::Float64) =
    (0.5 * u0 - _log1pexp(u0)) + (0.5 * up - _log1pexp(up))

# ---------------------------------------------------------------------------
# 类型（RP §6.4 / 施工图 §2.4 / 裁决 G7）
# ---------------------------------------------------------------------------

"""
propriety gate 证书（D-036）：A2（数据层：n ≥ N 且 rank(Syy) = N——
违反即 error，文本含 "posterior improper"）；A4（先验层：HalfCauchy-τ
proper prior 下右尾证书应绿——**若红是实现 bug**，不再是预期红）。
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
    V::Union{Matrix{Float64},Nothing}   # P×P = (Sxx+Λ)⁻¹；rank1=true 时未物化（nothing）
    V_fact::Union{Cholesky{Float64,Matrix{Float64}},RankOneMul}  # 求解/乘入口（lazy 时 = A·z 上下文）
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
        tol = 1e-6, max_cells = 8192, tail_rel = 1e-10,
        init_grid = 4, m_ref = -Inf) -> (; nodes, weights, logZ, n_cells, rel_err)

(log α₀, log α_p) ∈ R² 的**确定性自适应** cell quadrature（RP §6.2-6.3；
D-038：节点数不是策略参数，密度由误差证书驱动）：

- **初始网格**：`init_grid × init_grid` 均匀 cell 覆盖给定域；
- **cell 积分**：张量积 Gauss-Legendre 2 点规则（每维节点 ±1/√3、权 1；
  二维张量权积归一 /4；log 空间减 max 稳定化）——per-cell 误差 ~h⁶，
  光滑被积函数下相对误差指数从 O(1/N)（中点规则）提升到 O(N^{-2}) 起；
- **误差估计**：张量积 3 点 G-L 对照（节点 0,±√(3/5)、权 8/9,5/9）与
  2 点主规则之差 × 安全因子 s：`s·|I₃ₓ₃−I₂ₓ₂|`（s 初始 1，D-066 由 E2
  实验定稿）——紧且不放松收敛判据语义（仍是确定性相对误差证书）；
- **细分**：误差贡献最大的 cell 沿长边二分；**全局误差证书** =
  Σ cell 误差 ≤ tol·Z 时停止；
- **确定性**：节点生成不消耗随机数（RP §6.2）；同输入同配置逐位可重放；
- **预算耗尽**（cell 数 > max_cells 仍未达 tol）→ `error`，文本含
  **"Numerical integration did not converge"**（D-067；禁止「到预算就
  返回当前值」；运行超时的正确行为是 fail / 缩小 fixture，**不是放宽
  tol**——P0-7 纪律）。`max_cells` 默认 **8192**（Manager 裁决 2026-10-10：
  与一维路径 `_adaptive_quadrature_1d` 的 4× 预算对称对齐——1D 路径由
  调用方传 `4*max_cells`，2D 路径直接默认 8192；求值成本已因 u_span=5
  域面积 4 倍缩小而下降。这是消除 1D/2D 预算不对称，**不是放宽数学
  tolerance**（P0-7 仍要求 tol=1e-6）；8192 仍不收敛则 fail loudly。
- **尾质量证书**：域边界 8 点（4 角 + 4 边中点）的 logf ≤ 峰值参考 +
  log(tail_rel) + log(域面积)——左尾指数衰减（RP §5.2）、右尾
  HalfCauchy-τ 后 ~e^{−u/2}（可积）；违反即 error（同 D-067 文本——
  proper prior 下若红是实现 bug，不再是预期红，RP §10 #2）。
- **峰值参考 m_ref**：调用方应传粗扫描 mode（fit_full_posterior 的
  mval，与一维版 _adaptive_quadrature_1d 同语义）——内部 init_grid²
  个网格点在峰尖时可能显著低估峰值，使边界点相对落差不足而误报红；
  -Inf 时回退到内部 init_grid² 点 max（仅诊断/独立调用用）。m_ref
  是 D-038 bracket 语义（粗扫描 mode 作参考），不是经验阈值/特例分支。

返回 `(; nodes, weights, logZ, n_cells, rel_err)`：nodes 为 cell 中心、
weights 为归一化 cell 质量、logZ 为（减 max 稳定化后的）log 积分。
"""
function adaptive_quadrature_2d(logf::Function,
                                u0_range::Tuple{Float64,Float64},
                                up_range::Tuple{Float64,Float64};
                                tol::Float64 = 1e-6,
                                max_cells::Int = 8192,
                                tail_rel::Float64 = 1e-10,
                                init_grid::Int = 4,
                                m_ref::Float64 = -Inf)
    (u0lo, u0hi) = u0_range; (uplo, uphi) = up_range
    u0hi > u0lo || throw(ArgumentError("adaptive_quadrature_2d: empty u0 range"))
    uphi > uplo || throw(ArgumentError("adaptive_quadrature_2d: empty up range"))
    # 尾质量证书：边界 8 点（先于任何积分；D-035a 下应绿——若红是实现 bug）
    bpts = [(u0lo, uplo), (u0hi, uplo), (u0lo, uphi), (u0hi, uphi),
            ((u0lo + u0hi) / 2, uplo), ((u0lo + u0hi) / 2, uphi),
            (u0lo, (uplo + uphi) / 2), (u0hi, (uplo + uphi) / 2)]
    # 峰值参考 refval：调用方传粗扫描 mode（fit_full_posterior 的 mval，
    # 与一维版 _adaptive_quadrature_1d 同语义）——内部 init_grid² 个点
    # 在峰尖时可能显著低估峰值，使边界点相对落差不足而误报红（N=4 无
    # 信号 fixture 实测红）；-Inf 回退到内部点 max（仅诊断/独立调用）。
    # m_ref 是 D-038 bracket 语义（粗扫描 mode 作参考），非经验阈值。
    inner = [(u0lo + i * (u0hi - u0lo) / (init_grid + 1),
              uplo + j * (uphi - uplo) / (init_grid + 1))
             for i in 1:init_grid, j in 1:init_grid]
    inner_vals = [logf(p[1], p[2]) for p in inner]
    all(isfinite, inner_vals) ||
        error("PosDefException at quadrature refval point — R4 fail-loud")
    refval = max(isfinite(m_ref) ? m_ref : -Inf, maximum(inner_vals))
    # 尾质量证书（P0-8 裁决后统一判据）：边界 8 点（4 角 + 4 边中点）的
    # logf ≤ refval + log(tail_rel) + log(域面积)——忠实「域外/域内质量
    # 比」语义。HalfCauchy-τ 先验下两端自然衰减（左尾 ~e^{u/2}、右尾
    # ~e^{−u/2}），证书应绿；红是实现 bug（不再有弱信息特例分支——不
    # 以回测 fixture 定数值分支，开发守则）。
    area = (u0hi - u0lo) * (uphi - uplo)
    # R4''' 证书点防护：非有限（防御软包装吞后的 -Inf）→ fail loudly。
    bvals = [logf(p[1], p[2]) for p in bpts]
    all(isfinite, bvals) ||
        error("PosDefException at quadrature certificate point — R4 fail-loud")
    tail_ok = all(bvals[i] <= refval + log(tail_rel) + log(area)
                  for i in eachindex(bpts))
    tail_ok || error("Numerical integration did not converge: alpha tail mass certificate failed (HalfCauchy-tau prior; red = implementation bug, not expected red)")
    # cell 表示：矩形 + G-L 节点 logf 缓存（节点坐标缓存避免重复求值）
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
    # --- m 基准修复（真数据首日溢出，2026-10-10）---
    # 旧实现 m 只取初始 cell 中心 max——中点规则时代 m 与采样点同点自洽
    # （exp(fv−m) ≤ 1 恒成立）；G-L 节点偏离中心（±1/√3、±√(3/5)）在陡峭
    # logf（真数据大 P 实测节点超中心 +502.7）下使 exp(fv−m) 溢出 → Inf。
    # 修复：m 覆盖 cell_val/cell_err 实际求值的全部节点（cell_peak = 2 点
    # 4 节点 ∪ 3 点 9 节点的 max）；细分中若新节点超 m，循环内经 cell_peak
    # 检查后更新 m 并全量刷新（见细分段）。m 是纯归一化基准：logZ =
    # log(total) + m 补偿、rel 为公因子比值——数学对象与容差语义不变。
    function cell_peak(cell)
        a, b, c, d = cell
        mx = (a + b) / 2; hx = (b - a) / 2
        my = (c + d) / 2; hy = (d - c) / 2
        g2 = 1.0 / sqrt(3.0)
        g3 = sqrt(3.0 / 5.0)
        max(fv(mx - hx * g2, my - hy * g2), fv(mx - hx * g2, my + hy * g2),
            fv(mx + hx * g2, my - hy * g2), fv(mx + hx * g2, my + hy * g2),
            fv(mx - hx * g3, my - hy * g3), fv(mx - hx * g3, my),
            fv(mx - hx * g3, my + hy * g3),
            fv(mx, my - hy * g3), fv(mx, my), fv(mx, my + hy * g3),
            fv(mx + hx * g3, my - hy * g3), fv(mx + hx * g3, my),
            fv(mx + hx * g3, my + hy * g3))
    end
    m = maximum(cell_peak(c) for c in cells)
    isfinite(m) ||
        error("PosDefException: all quadrature cell nodes failed — R4 fail-loud")
    # --- 局部规则升级（方案 a；设计文档 §4）---
    # 2 点 G-L 主规则（每维 ±1/√3、权 1）per-cell 误差 ~h⁶；3 点对照
    # （0,±√(3/5)、权 8/9,5/9）提供 `s·|I₃ₓ₃−I₂ₓ₂|` 误差代理——光滑
    # 函数下与真误差同阶（E2 验证保守性），相对误差指数从 O(1/N) 提升
    # 到 O(N^{-2}) 起。s 初始 1（D-066 候选，E2 定稿）；判据仍是确定性
    # 相对误差证书。
    function cell_err(cell)
        a, b, c, d = cell
        mx = (a + b) / 2; hx = (b - a) / 2
        my = (c + d) / 2; hy = (d - c) / 2
        g2 = 1.0 / sqrt(3.0)
        i2 = (exp(fv(mx - hx * g2, my - hy * g2) - m) +
              exp(fv(mx - hx * g2, my + hy * g2) - m) +
              exp(fv(mx + hx * g2, my - hy * g2) - m) +
              exp(fv(mx + hx * g2, my + hy * g2) - m)) / 4
        g3 = sqrt(3.0 / 5.0)
        wl = 5.0 / 9.0; wh = 8.0 / 9.0
        xs = (mx - hx * g3, mx, mx + hx * g3)
        ys = (my - hy * g3, my, my + hy * g3)
        acc = 0.0
        for j in 1:3, i in 1:3
            wi = i == 2 ? wh : wl
            wj = j == 2 ? wh : wl
            acc += wi * wj * exp(fv(xs[i], ys[j]) - m)
        end
        i3 = acc / 4
        s = 1.0
        s * abs(i3 - i2) * ((b - a) * (d - c))
    end
    function cell_val(cell)
        a, b, c, d = cell
        mx = (a + b) / 2; hx = (b - a) / 2
        my = (c + d) / 2; hy = (d - c) / 2
        g = 1.0 / sqrt(3.0)
        (exp(fv(mx - hx * g, my - hy * g) - m) +
         exp(fv(mx - hx * g, my + hy * g) - m) +
         exp(fv(mx + hx * g, my - hy * g) - m) +
         exp(fv(mx + hx * g, my + hy * g) - m)) * (b - a) * (d - c) / 4
    end
    # （i）循环内重复计算消除（Manager 裁决 2026-10-10；测量证据：A3 栈
    # 卡点 sum + cell_err 重复调用、1929 cells 的 O(cells²) 全量重算）：
    # 每 cell (val, err) 只算一次并维护数组；细分后仅更新被拆两 cell 与
    # 两个全局和（O(1)）；max-err 扫描读已存值、不再重复 cell_err。
    # 细分顺序语义不变（同一 max-err 定义、tie 取 index 最小）；结果仅有
    # 求和路径的 roundoff 级差异（增量 vs 全量，验收对照）；证书/容差不变。
    vals = [cell_val(c) for c in cells]
    errs = [cell_err(c) for c in cells]
    Z = sum(vals)
    err_total = sum(errs)
    rel = Z > 0 ? err_total / Z : Inf
    while rel > tol && length(cells) < max_cells
        # 误差最大 cell（tie 取 index 最小——确定性；读已存 errs，不重算）
        idx = 1; emax = errs[1]
        for k in 2:length(errs)
            if errs[k] > emax
                emax = errs[k]; idx = k
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
        # m 覆盖性维护：新 cell 的 G-L 节点可能超过当前 m——先检查；若
        # 超过则以新 m 全量刷新 vals/errs（基准变化使旧归一值失效；O(N)，
        # fv 节点缓存命中；仅在 m 增长时发生）；否则保持原增量路径（O(1)）。
        pm = max(cell_peak(cells[idx]), cell_peak(cells[end]))
        if pm > m
            m = pm
            push!(vals, 0.0); push!(errs, 0.0)   # 与新 cells 同步长度
            for k in eachindex(cells)
                vals[k] = cell_val(cells[k]); errs[k] = cell_err(cells[k])
            end
            Z = sum(vals); err_total = sum(errs)
        else
            v1 = cell_val(cells[idx]); e1 = cell_err(cells[idx])
            v2 = cell_val(cells[end]); e2 = cell_err(cells[end])
            Z += v1 + v2 - vals[idx]
            err_total += e1 + e2 - errs[idx]
            vals[idx] = v1; errs[idx] = e1
            push!(vals, v2); push!(errs, e2)
        end
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
`adaptive_quadrature_2d` 同款逻辑（中点 cell / Simpson 型中点-梯形差
误差估计 / 最大误差 cell 二分 / 预算耗尽 fail loudly，D-067 文本 /
确定性不消耗 RNG）。

**存在理由**：无 DC 形态（P = 14N，Λ = α_p·I）下 α₀ 不在模型（不参与
Λ_α），u₀ 不是积分变量。u₀ 方向的先验密度在 HalfCauchy-τ 下两端衰减、
积分为常数（归一化后 =1），故二维积分与一维 u_p 积分**数学等价**；
一维路径避免对无关维度做无谓求值（效率，非数学必需）。u₀ 占位 0.0，
prior 的 u₀ 部分贡献常数（= −log2），不影响 u_p 后验形状。DC 形态
（P = 1+14N）仍走二维 `adaptive_quadrature_2d`（D-033 双 group）。

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
    # 尾质量证书（P0-8 裁决后统一判据，与二维版同步）：域边界端点 logf ≤
    # m_ref + log(tail_rel) + log(域宽)；HalfCauchy-τ 先验下应绿，红是
    # 实现 bug（无弱信息特例分支）。
    # R4''' 证书点防护（1d，与 2D 同步）：非有限 → fail loudly。
    tv1 = (logf1(uplo), logf1(uphi))
    all(isfinite, tv1) ||
        error("PosDefException at quadrature certificate point (1d) — R4 fail-loud")
    tail_ok1 = all(tv1[i] <= m_ref + log(tail_rel) + log(uphi - uplo)
                   for i in 1:2)
    tail_ok1 ||
        error("Numerical integration did not converge: alpha tail mass certificate failed (1d, HalfCauchy-tau prior)")
    vcache = Dict{Float64,Float64}()
    fv(x) = get!(() -> logf1(x), vcache, x)
    cells = Vector{Tuple{Float64,Float64}}()
    du = (uphi - uplo) / init_grid
    for i in 1:init_grid
        push!(cells, (uplo + (i - 1) * du, uplo + i * du))
    end
    m = maximum(fv((a + b) / 2) for (a, b) in cells)
    isfinite(m) ||
        error("PosDefException: all quadrature cell nodes failed (1d) — R4 fail-loud")
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

function _fit_node(st::SufficientStats, lam::Vector{Float64}, w::Float64;
                   rank1::Bool = false,
                   spec::Union{Nothing,SxxRankOneSpec} = nothing)
    P = size(st.Sxx, 1)
    N = size(st.Syy, 1)
    if rank1 && spec !== nothing && all(==(lam[2]), view(lam, 2:P))
        # 阶段 2 lazy 路径（FIRST-DAY-FIT-1）：全程经 spec、无 cholesky；
        # A⁻¹Sxy 用 SM（阶段 1 同式）、A·z 由 RankOneMul 承担。O(P²N)。
        α0 = lam[1]; αp = lam[2]; δ = α0 - αp
        d = spec.lam .+ αp
        if all(>(0.0), d)
            e1Be1 = sum(spec.v .^ 2 ./ d)
            denom = 1.0 + δ * e1Be1
            if isfinite(denom) && denom > 0.0
                Bis = spec.Q * (spec.W ./ d)
                Bie1 = spec.Q * (spec.v ./ d)
                Ais = Bis .- ((δ / denom) .* Bie1) * (Bie1' * st.Sxy)
                Sraw = st.Syy .- st.Sxy' * Ais
                S = Symmetric((Sraw + Sraw') ./ 2)
                nu2 = st.n + 1 - N
                return ConditionalFit(Matrix(Ais'), nothing, RankOneMul(spec, d, δ),
                                      Matrix(S), nu2, w)
            end
        end
        # 数值防护失败：落到物化路径（fail-soft；不改数学）
    end
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
                       alpha_fixed = nothing, u_span = 5.0)
        -> ResponsePosterior

完整 response 后验（Step 8 主入口）：

1. `sufficient_stats`（RP §3.2 A1）；
2. **A2 propriety gate**（D-036）：`n ≥ N` 且 `rank(Syy) = N`——违反即
   error，文本含 **"posterior improper"**（数据层条件，负测试对象）；
3. Λ_α 双 group（D-033）：`P == 1+14N`（DC 形态）时 diag(α₀, α_p,…)、
   否则全 α_p（数学层允许任意 P；生产维度契约由上游持有）；
4. quadrature（RP §6）：log 坐标被积 = `log_evidence + log_prior_d035a`
   （HalfCauchy-τ proper prior）；初始域 = 粗扫描 mode ± u_span（mode
   仅作初始 bracket/域中心——D-038：不得定义 posterior）；尾证书 +
   预算耗尽 fail loudly（D-067）；
5. 每节点 `_fit_node`（B̂、V、S(α)、ν 绑定）。

`alpha_fixed`（退化入口，测试/诊断用——**先验退化极限** RP §10 #6 的
合法实现路径）：给定 `(α₀, α_p)`（DC 形态必须二元组）或单值 `α`
（无 DC 形态），跳过 quadrature、单节点 p=[1]（点质量先验）。生产路径
（`alpha_fixed === nothing`）走 quadrature——**不得**以固定 α 冒充完整
后验（D-034/D-038）。

**numerical contract 纪律（P0-7）**：`tol` 默认即严格 reference 口径
（1e-6——D-066 初始候选，留待 refinement 定稿，**禁止由运行预算决定**）。
预算耗尽 fail loudly（D-067 文本 "Numerical integration did not converge"）
——运行超时的正确行为是 fail / 缩小 fixture，不是放宽 tolerance。

**max_cells 语义（Manager 裁决 2026-10-10）**：二维路径（DC 形态）不再
透传本参数——`adaptive_quadrature_2d` 用其自身默认 **8192**（与一维路径
对称，消除 1D/2D 预算不对称）；一维路径（无 DC 形态）用 `4*max_cells`
（本参数默认 2048 ⇒ 一维预算 8192）。两路径预算同为 8192 cells。

**u_span 选择（2026-10-10 静态推导）**：默认 `u_span = 5`（从 10 缩小）。
理由：(1) tail certificate 判据 `logf(boundary) ≤ refval + log(tail_rel)
+ log(area)` 中 `tail_rel=exp(−u_span)`、`area=(2·u_span)²`，使边中点判据
允许平台-峰差 Δ ≤ 0.5·u_span + log(4·u_span²) − u_span；u_span=5 时
Δ ≤ 2.895（覆盖实测纯噪声 Δ≈2.2 与无信号 Δ≈0），u_span=4 时 Δ ≤ 1.84
（对 Δ=2.2 红，不安全）——故 5 是静态推导支持的最小安全值。(2) 域缩小
（[-5,5]² vs [-10,10]²，面积 4 倍）减少低密度区 cell 浪费、改善收敛。
**注意**：这是数值域参数（D-066 初始候选），不是放宽 tolerance——tail
certificate 仍保护先验尾部不被截断（域端点质量可忽略由证书保证）。
收敛的**完整**解决（DC 形态下 α₀/α_p 尺度各向异性 + u₀ 方向平台结构）
属 Gate-1 坐标变换/初始网格范畴，本步只做安全域缩小。

**R4'''（2026-10-10；F1/F2：PosDefException @ cholesky(Sα)）**：Sα 的 logdet
改走 eigen-based 鲁棒路径（负噪声特征值 floor 到 ε_floor=_PSD_FLOOR_REL·scale；
结构性负值 fail loudly）；防御层按类型捕获 PosDefException → 该点 logf=-Inf
+ 诊断；证书点/refval 点非有限 → fail loudly；中域/占比/全失败 → fail loudly。
阈值 _PSD_* / _POSDEF_* 为保守初始值（待 D-066 定稿）；tol/门禁/max_cells 原样。
可选 posdef_diag kw 接收诊断。正常档零改变（eigen vs cholesky ~roundoff 级）。
"""
function fit_full_posterior(X::AbstractMatrix{Float64},
                            Y::AbstractMatrix{Float64};
                            prior::Symbol = :d035a,
                            tol::Float64 = 1e-6,
                            max_cells::Int = 2048,
                            alpha_fixed = nothing,
                            u_span::Float64 = 5.0,
                            rank1::Bool = false,
                            # R4'''（F1/F2）：可选诊断出口（不导出；传入的
                            # mutable 对象被就地填充）。不传时内部照常做安全断言。
                            posdef_diag::Union{Nothing,PosDefDiagnostics} = nothing)
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
    # 阶段 1（FIRST-DAY-FIT-1；默认关闭）：rank1=true 且 DC 形态时，对每点
    # log_evidence 走 Λ 秩一预分解路径（SM/行列式引理）；数值防护失败
    # （非正特征移位/SM 分母非正）逐点回退原路径（fail-soft，不改数学）。
    _spec = (rank1 && has_a0) ? _rank1_spec(st) : nothing
    # R4'''（F1/F2）：防御性软包装——PosDef → 该点 logf = -Inf + 记录；
    # 核心 Sα 路径已由 _robust_logdet_psd 承担（eigen + floor），见上。
    diag = posdef_diag === nothing ? PosDefDiagnostics() : posdef_diag
    _raw_ev(lam_::Vector{Float64}) = _evidence_eval(st, lam_, _spec)
    _ev(lam_::Vector{Float64}) = _soft_evidence(_raw_ev, lam_, diag)
    logf(u0, up) = _soft_logf(_raw_ev, lambda_diag(P, u0, up, has_a0),
                              log_prior_d035a(u0, up), u0, up, diag)
    # 粗扫描 mode：仅作初始域中心（D-038 bracket 语义，不定义 posterior）
    grid = range(-12.0, 12.0; length = 21)
    m0, mp, mval = 0.0, 0.0, -Inf
    for u0 in grid, up in grid
        v = logf(u0, up)
        if v > mval
            mval = v; m0 = u0; mp = up
        end
    end
    isfinite(mval) ||
        error("PosDefException: all coarse-scan points failed — R4 fail-loud")
    if has_a0
        # DC 形态（P=1+14N）：(u₀, u_p) 二维 quadrature（D-033 双 group）
        # max_cells 不显式传（默认 8192——Manager 裁决 2026-10-10 与 1D
        # 路径对称；fit_full_posterior 的 max_cells 参数仍透传给一维路径
        # 的 4*max_cells，二维路径用默认值，避免调用方误传旧 2048）。
        quad = adaptive_quadrature_2d(logf, (m0 - u_span, m0 + u_span),
                                      (mp - u_span, mp + u_span);
                                      tol = tol,
                                      tail_rel = exp(-u_span),
                                      m_ref = mval)   # 粗扫描 mode 作峰值参考
        nodes = quad.nodes
        weights = quad.weights
        logZ = quad.logZ
    else
        # 无 DC 形态（P=14N）：α₀ 不在模型（Λ = α_p·I），u₀ 不是积分变量。
        # u₀ 方向的 HalfCauchy-τ 先验密度两端衰减、积分为常数（归一化后
        # =1），二维积分与一维 u_p 积分数学等价——一维路径只避免对无关
        # 维度做无谓求值。u₀ 占位 0.0，prior 的 u₀ 部分贡献常数
        # （= −log2），不影响 u_p 后验形状。
        logf1(up) = _soft_logf(_raw_ev, lambda_diag(P, 0.0, up, false),
                                log_prior_d035a(0.0, up), 0.0, up, diag)
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
    conds = [_soft_fit_node(st, lambda_diag(P, u0, up, has_a0), weights[k];
                            rank1 = rank1, spec = _spec)
             for (k, (u0, up)) in enumerate(nodes)]
    levs = [_ev(lambda_diag(P, u0, up, has_a0)) for (u0, up) in nodes]
    _assert_posdef_safe(diag, m0, mp; check_u0 = has_a0)   # R4'''：防御断言
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


function _mul_A(rm::RankOneMul, z::Vector{Float64})
    y = rm.spec.Q * (rm.d .* (rm.spec.Q' * z))
    y[1] += rm.δ * z[1]
    y
end

# V·z 的求解入口（默认 = V_fact \ z = A·z；lazy = RankOneMul 的 A·z）
V_solve(c::ConditionalFit, z::Vector{Float64}) =
    c.V_fact isa RankOneMul ? _mul_A(c.V_fact, z) : (c.V_fact \ z)

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
