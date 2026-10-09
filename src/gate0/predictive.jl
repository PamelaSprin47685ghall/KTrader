# =============================================================================
# KTraderGate0 PredictiveLaw 组合 — 施工图 Step 13
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付。include 顺序：geometry.jl → modes.jl → response.jl →
# posterior.jl → oof.jl → innovation.jl → 本文件（组装点消费两侧 export
# 面：ResponsePosterior/predict_mu/draw_mu 与 InnovationState/
# draw_innovation/mp_sqrt_factors）。
#
# 职责（施工图 Step 13 / VI §8.2 / RP §7.3）：**只做**两侧组装——response
# 层（posterior.jl：μ 的 matrix-t predictive）与 innovation 层
# （innovation.jl：V_t(d)/q(d|H)/z pool）组合成 one-step predictive law
# 的 scenario 生成与方差分解报告。本文件不持有任何一层的数学（μ 的
# posterior 归 posterior.jl；innovation law 归 innovation.jl）——它是两条
# 独立开发线的组装点（oof→innovation 契约的消费端延伸）。
#
# 裁决 A4（R 子集执行路径，八步链条）：
#   1-2. μ_asset = E_active·E[μ|H]（含 b₀——DC 列在 B̂·x 内自动参与）；
#        μ_R = E_{R_t}ᵀ·μ_asset[R_t]（asset 空间 R 子集提取 + mode 变换
#        ——与 innovation 层 ε^{(R)} 同路径，裁决 A3 同源：不从 active 域
#        mode 坐标直接取子向量）；
#   3.  scenario 生成（独立流契约，裁决 F——顺序固定，同 seed 逐位可重放）：
#        μ draw（流位置 1，matrix-t posterior 抽样）→ d 抽样（流位置 2，
#        q(d|H)）→ z 行抽样（流位置 3，per-d pool）→ ε = V_t^{1/2}·z；
#   4-8. y_R = μ_R + ε_R（R 域 mode）→ u_R = E_{R_t}·y_R → ×s₁[R_t] →
#        exp 成 gross。
#
# 裁决 A5（方差分解报告口径，asset 空间 R 子集上同域可加）：
#   Var(u_R) = E_R·Var(μ_R)·E_Rᵀ + E_R·E[V_t]·E_Rᵀ
#   epistemic（response posterior 的 within+between，经 E_active 投影到
#   R 子集）与 aleatoric（innovation 的 E[V_t] = Σ_g q_g·V_t(d_g)——V 的
#   d 混合矩）分别报告；不可计算部分显式 NOT COMPUTED（ν≤2 时 within
#   不存在——NaN 携带 + not_computed 清单，不显示 0，D-030/§30 报告义务）。
#   如实标注：ε 的完整二阶矩 = Σ_g q_g·V_g^{1/2}·M_z(d_g)·V_g^{1/2}
#   （M_z = z pool 经验二阶矩；协议不假设 M_z = I——VI §5.4），与 E[V_t]
#   在 M_z ≠ I 时不等；报告口径按裁决 A5 字面，差异不掩盖。
#
# D-060 红线（构造性保证，docstring 钉死）：scenario 通道 = μ 的
# matrix-t posterior draw（posterior 的 **epistemic** 表达——S(α)/ν 是
# t 分布尺度参数，不是 Σ_R 的未来残差重复计数）+ innovation draw
# （V_t 的 **aleatoric**）。本函数不消费、不生成任何 Σ_R 的「额外未来
# 冲击」——「response posterior draw 负责 mean uncertainty、innovation
# draw 负责 residual uncertainty」，Σ_R 在 scenario 通道不出现。
#
# 接口契约（消费端；与 oof.jl 的 producer 端对接，测试头注释钉死全文）：
#   crossing representation：ε̃ 矩阵（asset 空间、列 = active 域）+ 行
#   mask（O_s，长度 = active 资产数 N）+ 行身份（u_s 严格递增）+ R_t
#   （⊆ active 域索引集）+ E_active（N×N）；
#   stable identity rule：行 s ↔ 目标日 u_s 递增（innovation_state 契约）；
#   failure semantics：两侧 asset 空间维度不一致 / R_t 越界 / propriety
#   红 / s1 非正 / S<1 一律 fail loudly；innovation 覆盖不足（T4）在
#   innovation_state 构造层已拦（"innovation coverage failure"）——本层
#   传导不吞、不降级。
#
# reference 纪律（D-082/D-083）：单线程、无缓存、无优化；scenario 循环
# direct。fail-loudly（SPEC §56）。全部判据与回测收益无关（SPEC §95 /
# 开发守则 §24）。

using Random, LinearAlgebra

export PredictiveLawResult, predictive_law

# ---------------------------------------------------------------------------
# 输出类型
# ---------------------------------------------------------------------------

"""
    PredictiveLawResult

one-step predictive law 的组装输出（施工图 Step 13 / 裁决 A4/A5）。

字段（构造后只读）：
- `R_t`：scenario 列集（= InnovationState.R_t：free ∪ locked，sorted
  unique——**列集 = R_t**，active 域中 ⊄ R_t 的资产不进 scenario）；
- `gross`：S×N_R gross return 矩阵（列序 = R_t；exp(log_gross)）；
- `log_gross`：S×N_R（= u_R⊙s₁[R_t]——方差分解蒙特卡洛对照的域）；
- `mu_R`：E[μ_R]（R 域 mode 坐标——predict_mu 的 asset 投影 R 子集再
  mode 变换；解析对照对象，非 scenario 统计量）；
- `var_epistemic` / `var_aleatoric`：N_R×N_R，asset 空间 R 子集（裁决
  A5 同域可加：total = epistemic + aleatoric；ν≤2 时 epistemic 含 NaN）；
- `var_within` / `var_between`：epistemic 的两分量（asset R 子集投影；
  ν≤2 时 within 为 NaN——**NOT COMPUTED，不显示 0**，D-030）；
- `within_computed`：ν > 2 与否；
- `not_computed`：显式未计算分量清单（如 [:within]）。

**D-060（接口层面钉死）**：本类型不含任何 Σ_R/未来残差冲击通道字段
——scenario 的不确定性只有两个来源：μ 的 matrix-t posterior draw
（epistemic）与 innovation draw（aleatoric）。
"""
struct PredictiveLawResult
    R_t::Vector{Int}
    gross::Matrix{Float64}
    log_gross::Matrix{Float64}
    mu_R::Vector{Float64}
    var_epistemic::Matrix{Float64}
    var_aleatoric::Matrix{Float64}
    var_within::Matrix{Float64}
    var_between::Matrix{Float64}
    within_computed::Bool
    not_computed::Vector{Symbol}
end

# ---------------------------------------------------------------------------
# 主入口：八步链条组装
# ---------------------------------------------------------------------------

"""
    predictive_law(posterior, innovation_state, x_t;
                   s1, E_active, rng, S = 256) -> PredictiveLawResult

八步链条组装（VI §8.2 / RP §7.3 两侧权威表述的汇合点；裁决 A4 的
R 子集执行路径）：

1. `μ_asset = E_active·E[μ|H]`（`predict_mu` 的 mode 均值经 E_active
   投影——含 b₀：DC 列在 B̂·x 内自动参与）；
2. `μ_R = E_{R_t}ᵀ·μ_asset[R_t]`（asset 空间 R 子集提取 + mode 变换）；
3. 每 scenario s（**独立流契约**，裁决 F——顺序固定，同 seed 同输入
   逐位可重放，SPEC §36）：
   - 流位置 1：μ 通道 matrix-t posterior draw（`draw_mu`：mode 空间
     N 维 → E_active 投影 → R 子集 → E_Rᵀ 变换）；
   - 流位置 2：`d ~ q(d|H_t)`（`draw_innovation` 内部，先 d 后行）；
   - 流位置 3：`z 行 ~ Uniform(L)`（同上）；
   - `ε_R = V_t(d)^{1/2}·z`（Moore-Penrose 对称 PSD 主根族）；
4. `y_R = μ_R^{(s)} + ε_R^{(s)}`（R 域 mode 坐标）；
5. `u_R = E_{R_t}·y_R`（R 域 asset normalized return）；
6. `r_R = u_R ⊙ s₁[R_t]`（asset log return；s₁ 为 per-asset 一日
   ruler——**调用方传入**，ruler 语义归 geometry/prepare 层，本层不
   重算、不推导）；
7. `gross = exp(r_R)`。

方差分解报告（裁决 A5，asset 空间 R 子集同域可加）：

```
Var(u_R) = E_R·Var(μ_R)·E_Rᵀ + E_R·E[V_t]·E_Rᵀ
         = var_epistemic + var_aleatoric
```

- `var_epistemic`：response posterior 的 within+between（`predict_mu`
  的 cov——mode 空间 → E_active 投影 → R 子块 → E_R 环绕；μ ⊥ ε 由
  独立流保证）。**ν ≤ 2 时 within 不存在**（t 矩无二阶矩）——
  `not_computed` 显式标注、NaN 携带，不得显示 0（D-030/§30）。
- `var_aleatoric`：`E[V_t] = Σ_g q_g·V_t(d_g)`（V 的 d 混合矩——
  裁决 A5 字面口径）。如实标注：ε 的完整二阶矩为
  `Σ_g q_g·V_g^{1/2}·M_z(d_g)·V_g^{1/2}`（M_z = z pool 经验二阶矩，
  协议不假设 M_z = I——VI §5.4），二者在 M_z ≠ I 时不等；报告口径
  是 E[V_t]，差异不掩盖（蒙特卡洛对照容差须容纳此差异）。

**D-060 红线（构造性保证）**：scenario 通道 = `draw_mu`（matrix-t
——posterior 的 **epistemic** 表达：S(α)/ν 是 t 分布尺度，**不是**
Σ_R 的未来残差重复计数）+ `draw_innovation`（V_t 的 **aleatoric**）。
本函数不引入任何 Σ_R 的额外消费——response posterior draw 负责
mean uncertainty、innovation draw 负责 residual uncertainty，两层职责
互不重复（D-060 原文：不得把 regression Σ_R 又作为额外 future
residual shock 重复加一次）。

**接口契约（消费端，fail-loudly 不吞错不降级）**：
- `posterior::ResponsePosterior`（posterior.jl——mode 坐标维 N =
  active 资产数）；
- `innovation_state::InnovationState`（innovation.jl——其输入
  `residual_rows` 的列数 N_a **必须等于** posterior 的 N：两侧 asset
  空间必须是同一个 active 域；违反即 DimensionMismatch。R_t ⊆ 1:N
  由 innovation_state 构造保证（已拦越界），本函数冗余防御再校验）；
- `x_t`：P 维 mode 输入（含 DC 列位——列 1 = 常数列）；
- `s1`：N 维 per-asset 一日 ruler（全部正有限）；
- `E_active`：N×N active 域 mode→asset 基（`domain_mode_basis(N)`）；
- `rng`：随机流 owner（调用方持有；同 seed 重放契约）；
- `S`：scenario 数（≥ 1；数值收敛证书归 Step 14 的 adaptive RQMC，
  本层固定 S 是 reference 语义——D-062 的固定 S 只允许 reference/
  测试用途，生产收敛由上层证书约束）。

错误传导：propriety 证书红（防御性再查——fit 层已 gate）、两侧维度
不一致、R_t 越界、s1 非正、S<1、x_t 非有限 → 抛错；innovation 覆盖
不足（T4）在 `innovation_state` 构造层已拦（错误文本
"innovation coverage failure"）——本层不吞。
"""
function predictive_law(post::ResponsePosterior,
                        st::InnovationState,
                        x_t::AbstractVector{Float64};
                        s1::AbstractVector{Float64},
                        E_active::AbstractMatrix{Float64},
                        rng::AbstractRNG,
                        S::Int = 256)
    # --- 契约校验（fail-loudly；消费端边界） ---
    N = post.N
    S >= 1 || throw(ArgumentError("predictive_law: S ≥ 1 需要（got $S）"))
    length(x_t) == post.P ||
        throw(DimensionMismatch("predictive_law: x_t 长度 $(length(x_t)) ≠ P=$(post.P)"))
    all(isfinite, x_t) || throw(DomainError(collect(x_t), "predictive_law: x_t 含 NaN/Inf"))
    size(E_active) == (N, N) ||
        throw(DimensionMismatch("predictive_law: E_active 应为 $(N)×$(N)（active 域基），实际 $(size(E_active))"))
    length(s1) == N ||
        throw(DimensionMismatch("predictive_law: s1 长度 $(length(s1)) ≠ N=$(N)（active 资产数）"))
    all(j -> isfinite(s1[j]) && s1[j] > 0, 1:N) ||
        throw(DomainError(collect(s1), "predictive_law: s1 必须全部正有限（per-asset ruler）"))
    length(st.row_masks) >= 1 ||
        error("predictive_law: innovation state 无行（构造已断言——状态损坏）")
    N_a = length(st.row_masks[1])
    N_a == N ||
        throw(DimensionMismatch("predictive_law: 两侧 asset 空间不一致——innovation 输入列数 N_a=$(N_a) ≠ posterior 的 active 域 N=$(N)（crossing representation 契约：ε̃ 矩阵列 = active 域，producer/consumer 必须同一 active 域）"))
    all(j -> 1 <= j <= N, st.R_t) ||
        throw(ArgumentError("predictive_law: R_t 含越界资产索引（active 域 1:$(N)）——R_t=$(st.R_t)"))
    post.propriety.proper ||
        error("predictive_law: response posterior propriety 证书红（posterior improper 传导——不吞错）")

    N_R = st.N_R
    E_R = st.E_R
    R = st.R_t
    xt = collect(Float64, x_t)
    s1_R = collect(Float64, s1)[R]

    # --- 方差分解（裁决 A5；μ ⊥ ε 由独立流保证） ---
    # mode 空间矩 → asset 空间（E_active 正交投影）→ R 子块 → E_R 环绕
    pmu = predict_mu(post, xt)                  # (; mu, cov, within, between, within_computed)——mode 空间
    cov_asset = E_active * pmu.cov * E_active'   # N×N（ν≤2 时含 NaN——如实携带）
    cov_R = cov_asset[R, R]                      # asset 空间 R 子块
    var_mu_R = E_R' * cov_R * E_R                # R 域 mode（Var(μ_R)）
    var_epistemic = E_R * var_mu_R * E_R'        # 裁决 A5 字面（E_R 正交 ⇒ 数值 = cov_R）
    W_asset = E_active * pmu.within * E_active'
    B_asset = E_active * pmu.between * E_active'
    var_within = W_asset[R, R]
    var_between = B_asset[R, R]
    EV = zeros(N_R, N_R)                         # E[V_t] = Σ_g q_g·V_t(d_g)（d 混合矩）
    for g in eachindex(st.d_weights)
        EV .+= st.d_weights[g] .* st.V_t[g]
    end
    var_aleatoric = E_R * EV * E_R'
    not_computed = Symbol[]
    if !pmu.within_computed
        push!(not_computed, :within)   # ν≤2：t 矩二阶矩不存在——NOT COMPUTED（D-030：不显示 0）
    end
    mu_asset_mean = E_active * pmu.mu
    mu_R = E_R' * mu_asset_mean[R]

    # --- scenario 循环（八步链条；独立流契约：μ 流 1 → d 流 2 → 行流 3） ---
    gross = Matrix{Float64}(undef, S, N_R)
    log_gross = Matrix{Float64}(undef, S, N_R)
    for s in 1:S
        # 流位置 1：μ 通道 matrix-t posterior draw（mode → asset → R 域 mode）
        mu_mode_s = draw_mu(post, xt, rng)
        mu_R_s = E_R' * (E_active * mu_mode_s)[R]
        # 流位置 2+3：d ~ q(d|H) → z 行 ~ Uniform(L)（draw_innovation 内部
        # 固定先 d 后行）→ ε = V_t(d)^{1/2}·z（Moore-Penrose 主根族）
        innov = draw_innovation(st, rng)
        # 步骤 4-6：y_R = μ_R + ε_R → u_R = E_R·y_R → r_R = u_R⊙s₁[R]
        y_R = mu_R_s .+ innov.eps
        u_R = E_R * y_R
        r_R = u_R .* s1_R
        # 步骤 7：gross
        log_gross[s, :] .= r_R
        gross[s, :] .= exp.(r_R)
    end

    return PredictiveLawResult(collect(R), gross, log_gross, mu_R,
                               var_epistemic, var_aleatoric,
                               var_within, var_between,
                               pmu.within_computed, not_computed)
end
