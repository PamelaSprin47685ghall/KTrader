# =============================================================================
# KTraderGate0 driver — Step 15: 端到端单日决策 driver
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付（骨架 include 归 DevOps 集成验证轮——本文件不修改
# KTraderGate0.jl）。include 顺序：geometry → modes → response →
# posterior → oof → innovation → predictive → kelly → quadrature → 本文件
# （消费全部九模块 + Step 14 的公共接口）。
#
# 职责（施工图 Step 15 / 裁决书 §52 最终理论链）：把九模块串成**单日决策
# driver**——D-088/D-089 阶梯的 single-day 级。多日回测（gate0/backtest.jl）
# 在 Gate-0 关闭后才允许，不在本文件。
#
# §52 理论链的逐步落点（docstring 钉死，供审计对照）：
#
#   H_t ──(因果截断 t，构造性保证)──> signal_prices ──> log/return
#     ──(active 域：model_admitted 的 prefix 语义，A^model)──> ruler s1
#     ──> E_active（domain_mode_basis）
#     ──> build_mode_problem（mode 坐标 [m; Qᵀe] / [1|B_m|B_⊥] 含 DC 列）
#     ──> rows/row_ids/row_masks（WARMUP 起有观察行；行身份 = 目标日 u_s）
#     ──> free/locked（D-016/D-017，universe→active 域映射）
#     ──> resolve_risk_domain（R_t = free ∪ locked；T4 判定次序，裁决 C3）
#     ──> fit_full_posterior（Π(b₀,G,α₀,α_p,Σ_R|H)——完整 quadrature）
#     ──> prequential_residual_rows（strict prequential 残差行，P0-1——
#         production 残差源；legacy 3-fold OOF 仅作 reference/diagnostic）
#     ──> resolve_risk_domain 二次调用（满秩 gate，P0-3——传 residual_rows）
#     ──> innovation_state（ε→q(d)→V_t→z pool，R 域 mode 坐标；
#         require_full_rank=true——sample null space 不得当零风险）
#     ──> x_now（决策行 feature）
#     ──> 决策（adaptive_scenario_kelly 双证书收敛【D-062 生产路径】或
#         predictive_law + cash_kelly 固定 S reference【D-062 合法用途】）
#     ──> w_t*（cash + locked base_s，D-068/D-017）
#     ──> D-076 concentration cause report（§22 义务）
#
# T4 驱动器语义（裁决 C3，docstring 钉死）：innovation 层的 resolve_risk_domain
# 对「locked 资产覆盖不足」fail loudly（错误文本 "innovation coverage failure"
# ——那是 innovation 层的正确行为）；**驱动器对该错误的响应是维持当日持仓
# 并记录诊断**（mode=:held_maintained：不重排权重、不拼残差、不 zero-fill、
# 不 crash 整个 driver）——把 innovation 层的 error 转译为驱动器决策语义，
# 这是裁决 C3 对「回测驱动器捕获该错误、记录诊断、当日维持持仓」的单日
# driver 落点。非 coverage 类错误一律重抛（不吞、不降级）。
#
# P0-4 收口（批次 2）：quadrature.jl 的 solve_layer 已修复 locked 通道——
# locked 列不进 free 优化列（切 free 子列传 cash_kelly、全列进
# locked_wealth_gate0 的 base），wealth 表示统一方案 2（X_free·w_free +
# w_cash + base_locked），全 locked 分支为 w_cash·1 + base_locked。本 driver
# 不再存在「adaptive 有 locked 就回落 :reference 固定 S」的 production
# fallback——adaptive_scenario_kelly 直接处理 locked（D-017/D-068 语义：
# locked 风险参与 wealth、不进 free 优化列）。reference 固定 S 路径仅保留
# 为 D-062 的合法用途（unit test / artifact replay / fixture）。
#
# reference 纪律（D-082/D-083）：单线程、无缓存、无优化；fail-loudly
# （SPEC §56）；固定 seed（SPEC §36 同 seed 重放）；全部判据与回测收益
# 无关（SPEC §95 / 开发守则 §24）。单日慢可接受。
# =============================================================================

using Random, LinearAlgebra, Statistics

export Gate0DayDecision, single_day_decision, equal_weight_benchmark

const _DRIVER_DEFAULT_SEED = 0x00D05EED00000001
const _DIAG_SEED_XOR = 0xD1A6571C5EED0000   # 诊断样本流 seed 派生常数
                                            # （与 audit 派生常数不同源）

# ---------------------------------------------------------------------------
# 报告类型（D-076 字段清单的具体 NamedTuple 形态）
# ---------------------------------------------------------------------------

"""决策证书报告（P0-6 收口）：A = 权重收敛 ‖w_{2M}−w_M‖₁（D-064 A）、
B = B_opt（fine-rule regret：I_2M(w_{2M}) − I_2M(w_M) ≤ ε_U，在优化 fine
rule 上求值——P0-6 修复后 certificate_B 语义即 B_opt，非旧单向 regret）、
B_audit = |I_audit(w_{2M}) − I_audit(w_M)|（独立 audit replicate 绝对目标
差，D-064 B / D-065）、C = cash_kelly_certificate（D-064 C 求解误差）、
converged = 积分收敛、kelly_objective = 决策层 (1/S)Σlog(wealth)。固定 S
reference 路径的 A/B/B_audit 为 NaN（无积分收敛证书——D-062 语义：固定
S 是 reference 对照，不冒充收敛）；held_maintained / all_cash 全 NaN
（NOT COMPUTED，不显示 0）。"""
const CertReport = NamedTuple{(:A, :B, :B_audit, :C, :converged, :kelly_objective),
    Tuple{Float64, Float64, Float64, NamedTuple, Bool, Float64}}

"""方差分解报告（裁决 A5 / 裁决书 §30）：epistemic/aleatoric 为 asset 空间
R 子集上的 trace（同域可加口径）；ν≤2 时 epistemic 为 NaN 且
within_computed=false（NOT COMPUTED，不显示 0）。"""
const VarSplitReport = NamedTuple{(:epistemic, :aleatoric, :within_computed,
    :not_computed), Tuple{Float64, Float64, Bool, Vector{Symbol}}}

"""D-076 concentration cause report（裁决书 §22/§23 字段清单）。
top_asset = universe 索引（0 = 无风险仓）；utility_margin_1pct = 1% 权重
从 top 移向次优的 utility regret（独立诊断样本求值——非决策证书）；
top_posterior_mean_* / top_innovation_* = μ 与 E[V_t] 的最大方向。"""
const ConcentrationReport = NamedTuple{(:top_asset, :top_weight, :risky_weight,
    :cash_weight, :posterior_log_growth, :epistemic_variance,
    :innovation_variance, :utility_margin_1pct, :integration_M, :converged,
    :top_posterior_mean_asset, :top_posterior_mean_value,
    :top_innovation_eigenvalue, :top_innovation_mode_asset),
    Tuple{Int, Float64, Float64, Float64, Float64, Float64, Float64,
          Float64, Int, Bool, Int, Float64, Float64, Int}}

"""驱动器诊断（mode 特定）：reason = 分支原因；error_text = T4 错误原文；
dropped_free_universe = resolve 剔除的 free 资产（资金留 cash，D-056 出
路 4）。P0-4 收口：不再有 quadrature_locked_fallback 字段——adaptive
请求一律直接处理 locked（quadrature.jl 已修复 locked 通道），无回落。"""
const DriverDiag = NamedTuple{(:reason, :error_text, :dropped_free_universe),
    Tuple{Symbol, String, Vector{Int}}}

"""
    Gate0DayDecision

单日决策记录（Step 15 交付物）。字段（构造后只读）：

- `mode`：`:adaptive`（生产路径，双证书收敛） / `:reference`（固定 S 对照，
  D-062 合法用途） / `:all_cash`（R_t 空——D-056 出路 4 / 无 active 资产） /
  `:held_maintained`（T4：locked 覆盖不足，维持当日持仓，裁决 C3）；
- `t` / `R_t` / `R_universe`：决策日、risk 域（active 域索引 / universe 索引）；
- `w_universe`：N 维 universe 坐标权重（free 优化 + locked 维持；其余 0；
  held_maintained 时 = 当日持仓状态本身）；
- `w_risky`：R 域列序总权重（free 优化 + locked 维持）；
- `w_cash` / `locked_exposure` / `budget`：cash 权重、locked 敞口、free
  优化预算（= 1 − Σlocked）；
- `certificates` / `M`：决策证书（CertReport）与积分规模（adaptive 收敛层
  M / reference 固定 S / 0）；
- `variance_split` / `concentration` / `dropped_free` / `diagnostics`：
  §30 方差分解、D-076 集中诊断、被剔除 free（active 域索引）、分支诊断。

财富守恒不变量（正常决策路径）：Σ w_universe + w_cash = 1（locked 计入
w_universe；budget + Σlocked + w_cash = 1 的 D-068/D-017 形态）。
"""
struct Gate0DayDecision
    mode::Symbol
    t::Int
    R_t::Vector{Int}
    R_universe::Vector{Int}
    w_universe::Vector{Float64}
    w_risky::Vector{Float64}
    w_cash::Float64
    locked_exposure::Float64
    budget::Float64
    certificates::CertReport
    M::Int
    variance_split::VarSplitReport
    concentration::ConcentrationReport
    dropped_free::Vector{Int}
    diagnostics::DriverDiag
end

# ---------------------------------------------------------------------------
# 内部辅助：占位报告（held / all_cash 分支——NOT COMPUTED，不显示 0）
# ---------------------------------------------------------------------------

_no_cert()::CertReport =
    (; A = NaN, B = NaN, B_audit = NaN, C = (;), converged = false,
       kelly_objective = NaN)

_no_varsplit()::VarSplitReport =
    (; epistemic = NaN, aleatoric = NaN, within_computed = false,
       not_computed = [:epistemic, :aleatoric])

_no_concentration(M::Int)::ConcentrationReport =
    (; top_asset = 0, top_weight = 0.0, risky_weight = 0.0, cash_weight = 1.0,
       posterior_log_growth = NaN, epistemic_variance = NaN,
       innovation_variance = NaN, utility_margin_1pct = NaN,
       integration_M = M, converged = false, top_posterior_mean_asset = 0,
       top_posterior_mean_value = NaN, top_innovation_eigenvalue = NaN,
       top_innovation_mode_asset = 0)

"""T4 驱动器语义（裁决 C3）：innovation 层 coverage 不足（locked 引起）时
维持当日持仓 + 记录诊断，不重排、不 crash。首次 resolve（基础判据，步骤 6）
与二次 resolve（满秩判据，步骤 8b）共用此辅助——held_maintained 不需要
任何模型计算（不重排 ⇒ 无后验消费），维持当日持仓状态本身。"""
function _held_maintained_result(held::AbstractVector{Float64},
                                 act::Vector{Int},
                                 free_act::Vector{Int},
                                 locked_act::Vector{Int},
                                 t::Int, msg::String)
    R_held = sort(union(free_act, locked_act))
    w_risky_held = Float64[held[act[j]] for j in R_held]
    return Gate0DayDecision(:held_maintained, t, R_held,
        Int[act[j] for j in R_held], Float64.(held), w_risky_held,
        1.0 - sum(held), sum(held[act[locked_act]]),
        1.0 - sum(held[act[locked_act]]),
        _no_cert(), 0, _no_varsplit(), _no_concentration(0), Int[],
        (; reason = :innovation_coverage_locked, error_text = msg,
           dropped_free_universe = Int[]))
end

# ---------------------------------------------------------------------------
# 主入口：single_day_decision
# ---------------------------------------------------------------------------

"""
    single_day_decision(mf, el, held; t, seed, mode, ...) -> Gate0DayDecision

端到端单日决策（施工图 Step 15；裁决书 §52 理论链的驱动器落点——逐步
对照见文件头）。**单日**：多日回测在 Gate-0 关闭前禁止（D-088）。

输入：
- `mf::MarketFacts`：完整（≥ t 行）市场事实。driver 内部**先做因果截断**
  （构造性保证：截断后的 MarketFacts 物理上不含 >t 行——「未来行不存在」
  不是约定而是构造事实；因果性测试据此断言「截断输入与全量输入在 t 决策
  逐位一致」）；
- `el::Eligibility`：三 mask 政策层（free = E^trade ∧ T^exec，D-016）；
- `held`：当日持仓权重（universe 维，非负——w^held_j > 0 ∧ free=0 ⇒
  locked risk，D-017）；
- `t`：决策日（1 ≤ t ≤ T）；
- `seed`：决策随机流 seed（SPEC §36 同 seed 重放契约）；
- `mode`：`:adaptive`（默认，生产路径——adaptive_scenario_kelly 双证书
  A/B_opt/B_audit/C 收敛，D-062/D-064/D-065）或 `:reference`（固定 S 对照
  ——D-062 的合法用途：unit test / artifact replay / fixture）。locked 非零
  时 `:adaptive` **直接处理**（quadrature.jl 的 solve_layer 已修复 locked
  通道——P0-4 收口，不再回落 fixed-S；locked 列不进 free 优化列、其
  scenario wealth 贡献由 base 承担，D-017/D-068）；
- `S_reference` / `diag_S` / `min_scenarios` / `max_scenarios` /
  `weight_tol` / `utility_tol` / `kelly_tol` / `posterior_tol` /
  `posterior_max_cells` / `u_span`：透传各层（数值配置语义，非策略参数——
  D-066：tolerance 为定稿流程初始候选，**禁止由运行预算决定**；运行超时
  的正确行为是 fail / 缩小 fixture，不是放宽 tolerance——P0-7 纪律）。
  `u_span` 是 2D 超参 quadrature 的数值域半宽（默认 5.0，与 posterior.jl
  的 fit_full_posterior 对齐），**不是 tolerance 放宽**——prequential 路径
  用它降低逐行 2D 求积成本（D-038 bracket 语义）。

分支（mode 字段）：
- `:all_cash`——无 active 资产（A^model 全空）或 R_t 空（free 剔尽且无
  locked，D-056 出路 4：无可生成风险的资产，资金全留 cash）；
- `:held_maintained`——T4：resolve_risk_domain 抛 "innovation coverage
  failure"（覆盖不足由 locked 引起，裁决 C3）。驱动器语义：**维持当日持仓
  （w_universe = held）、记录诊断（错误原文）、不重排、不拼残差、不
  zero-fill、不 crash**。这是 innovation 层 fail-loudly 的驱动器端转译
  （裁决 C3 原文：「回测驱动器捕获该错误、记录诊断、当日维持持仓」）；
- 其余错误（propriety 红、数据非法、维度不符、积分预算耗尽等）一律
  **重抛**——driver 只转译 T4，不吞任何其他失败（SPEC §56）。

返回 `Gate0DayDecision`（D-076 全字段——见类型 docstring）。

D-076 义务（§22）：高集中解释前置六项（D-075）由本记录的字段支撑——
certificates（后验对象正确性的数值面）、variance_split（epistemic/
innovation 分解）、concentration（top asset / 权重 / log-growth / margin /
M / 证书 / top posterior mean direction / top innovation eigenmode）。
utility_margin_1pct 在**独立诊断样本**（seed ⊻ _DIAG_SEED_XOR，S = diag_S）
上求值——诊断量，非决策证书（决策收敛证书是 certificates 的
A/B_opt/B_audit/C——P0-6 收口）。
"""
function single_day_decision(mf::MarketFacts, el::Eligibility,
                             held::AbstractVector{Float64};
                             t::Int,
                             seed::UInt64 = _DRIVER_DEFAULT_SEED,
                             mode::Symbol = :adaptive,
                             S_reference::Int = 256,
                             diag_S::Int = 64,
                             min_scenarios::Int = 64,
                             max_scenarios::Int = 512,
                             # P0-7 收口：weight_tol/utility_tol/kelly_tol 恢复
                             # 裁决起始 reference 口径（1e-4/1e-6/1e-6 为严格起点，
                             # 留待 D-066 refinement 定稿，**禁止由运行预算决定**；
                             # 运行超时的正确行为是 fail / 缩小 fixture，不是放宽
                             # tolerance）。
                             weight_tol::Float64 = 1e-4,
                             utility_tol::Float64 = 1e-6,
                             kelly_tol::Float64 = 1e-6,
                             # posterior quadrature 配置（P0-7 收口）：严格
                             # reference 口径 1e-6 / 2048 cells（与 posterior.jl
                             # / oof.jl 默认一致；D-066 初始候选，留待 refinement
                             # 定稿，禁止由运行预算决定）。
                             posterior_tol::Float64 = 1e-6,
                             posterior_max_cells::Int = 2048,
                             # u_span：数值域参数（非 tolerance 放宽）——2D
                             # (log α₀, log α_p) quadrature 的 bracket 半宽。
                             # 生产默认 5.0 与 posterior.jl 的 fit_full_posterior
                             # 默认对齐（eng-oof 同步后 oof.jl 默认亦为 5.0）；
                             # prequential 路径用它降低 2D 求积成本（D-038
                             # bracket 语义，非精度放宽）。
                             u_span::Float64 = 5.0,
                                 # b′（μ 通道部分 QMC，设计 MU_CHANNEL_QMC_DESIGN）：
                                 # E3 判据已成立（A 越过）；默认 false 逐位不变；
                                 # 60-day 前置待验证。仅作用于 :adaptive 分支。
                                 mu_qmc::Bool = false,
                                 # CHISQ-QMC（A 门过门配置，CQ_*）：仅作用于
                                 # :adaptive 分支；默认 false 逐位不变。
                                 mu_chisq_qmc::Bool = false)
    T_full, N = size(mf.observed)
    mode in (:adaptive, :reference) ||
        throw(ArgumentError("single_day_decision: mode must be :adaptive or :reference (got $mode)"))
    (1 <= t <= T_full) ||
        throw(ArgumentError("single_day_decision: 1 ≤ t ≤ T (got t=$t, T=$T_full)"))
    length(held) == N ||
        throw(DimensionMismatch("single_day_decision: held length $(length(held)) ≠ N=$N"))
    all(x -> isfinite(x) && x >= 0, held) ||
        throw(ArgumentError("single_day_decision: held must be non-negative and finite"))
    (size(el.model_admitted) == (T_full, N)) ||
        throw(DimensionMismatch("single_day_decision: Eligibility 尺寸与 MarketFacts 不符"))

    # --- 步骤 0：因果截断（构造性保证：物理上不含 >t 行） ---
    mf_t = MarketFacts(mf.dates[1:t], mf.symbols,
                       mf.close[1:t, :], mf.adj[1:t, :], mf.observed[1:t, :])
    el_t = Eligibility(el.model_admitted[1:t, :], el.trade_eligible[1:t, :],
                       el.executable[1:t, :])

    # --- 步骤 1：active 域（A^model 的 prefix 语义，D-013） ---
    act = findall(el_t.model_admitted[t, :])          # universe → active 域
    N_act = length(act)
    if N_act == 0
        return Gate0DayDecision(:all_cash, t, Int[], Int[],
            zeros(N), Float64[], 1.0, 0.0, 1.0, _no_cert(), 0,
            _no_varsplit(), _no_concentration(0), Int[],
            (; reason = :no_active_assets, error_text = "",
               dropped_free_universe = Int[]))
    end

    # --- 步骤 2：signal → log/return（相邻 finite 对，D-012/D-022） ---
    signal = signal_prices(mf_t)[:, act]              # t × N_act
    x_log = log.(signal)                              # observed=false 处 NaN
    f_first = [findfirst(view(el_t.executable, 1:t, j)) for j in act]
    # （active ⟹ 至少一条相邻观测对 ⟹ f_first 非零——构造保证；防御再查）
    all(i -> i !== nothing, f_first) ||
        error("single_day_decision: active 资产 $(findall(i -> i === nothing, f_first)) 无观测行——与 A^model prefix 语义矛盾")
    s1_act = vec(ruler(x_log, Int[i for i in f_first])[:, 1])   # 一日 ruler（D-024）

    # returns 覆盖时刻 1:t（行 s = 时刻 s 的 return；行 1 无 return → NaN），
    # 使 X_full 含决策行 t（x_now = X_full[t,:]）。行 1 空观察（mode_field
    # 契约：|O|=0 → m=0, e=0——零嵌入，不进统计由 rows 剔除承担）。
    r = Matrix{Float64}(undef, t, N_act)
    r[1, :] .= NaN
    r[2:t, :] .= x_log[2:t, :] .- x_log[1:t-1, :]
    obs_ret = isfinite.(r)                            # t × N_act；行 s = 时刻 s 的观察

    # --- 步骤 3：mode 坐标（[m; Qᵀe] / [1|B_m|B_⊥]，D-021~D-031） ---
    E_active = domain_mode_basis(N_act)
    mp = build_mode_problem(r, Matrix{Bool}(obs_ret), s1_act, E_active; dc = true)

    # --- 步骤 4：训练行（WARMUP 起有观察行；空观察行剔除——fold 行集入口） ---
    # 配对行 s：X 行 = 时刻 s feature、Y 行 = 时刻 s+1 输出（build_mode_problem
    # 契约）；行身份 u_s = s+1（目标日）。空观察按**目标日**判定（VI §1.2
    # 契约：O_s = {j : r[u_s,j] finite}；空目标日行不当 residual=0 进统计——
    # §34 OOF tests 语义）。
    t - 1 >= WARMUP ||
        throw(ArgumentError("single_day_decision: t=$t 不足以形成训练行（需 t-1 ≥ WARMUP=$WARMUP）"))
    rows = Int[s for s in WARMUP:(t-1) if any(view(obs_ret, s + 1, :))]
    length(rows) >= 2 ||
        error("single_day_decision: 有观察训练行不足（$(length(rows)) 行 < 2——WARMUP 后无观察行）")
    row_ids = Int[s + 1 for s in rows]                # 行身份 u_s（目标日，严格递增）
    row_masks = Vector{BitVector}(
        [BitVector(view(obs_ret, u, :)) for u in row_ids])   # O_s = 目标日观察

    # --- 步骤 5：free/locked → active 域映射（D-016/D-017） ---
    free_univ = free(el_t, t)                         # N 维 universe
    locked_univ = locked(held .> 0, el_t, t)          # held ∧ ¬free
    free_act = Int[i for (i, j) in enumerate(act) if free_univ[j]]
    locked_act = Int[i for (i, j) in enumerate(act) if locked_univ[j]]

    # --- 步骤 6：R 域判定（T4 判定次序，裁决 C3） ---
    # resolve 在 posterior 之前：R 域是 innovation 的域定义（裁决 A1），
    # T4 维持持仓不需要任何模型计算（不重排 ⇒ 无后验消费）。
    R_act = Int[]
    dropped_free_act = Int[]
    if isempty(free_act) && isempty(locked_act)
        return Gate0DayDecision(:all_cash, t, Int[], Int[],
            zeros(N), Float64[], 1.0, 0.0, 1.0, _no_cert(), 0,
            _no_varsplit(), _no_concentration(0), Int[],
            (; reason = :empty_risk_domain, error_text = "",
               dropped_free_universe = Int[]))
    end
    try
        R_act, dropped_free_act = resolve_risk_domain(row_ids, row_masks,
                                                      free_act, locked_act, t)
    catch err
        msg = err isa ErrorException ? err.msg : sprint(showerror, err)
        if occursin("innovation coverage failure", msg)
            # T4 驱动器语义（裁决 C3）：维持当日持仓 + 记录诊断，不 crash。
            return _held_maintained_result(held, act, free_act, locked_act,
                                           t, msg)
        end
        rethrow()                                     # 非 T4：不吞、不降级
    end
    isempty(R_act) && error("single_day_decision: resolve 返回空 R_t 但 free/locked 非空——状态矛盾")
    N_R = length(R_act)
    dropped_free_univ = Int[act[j] for j in dropped_free_act]

    # --- 步骤 7：full posterior（Π(b₀,G,α|H)——完整 quadrature，D-034/D-038） ---
    # posterior_tol/posterior_max_cells 透传（P0-7 收口）：默认 1e-6/2048
    # 严格 reference 口径（D-066 初始候选；禁止由运行预算决定——运行超时
    # 的正确行为是 fail / 缩小 fixture）。
    X_tr = mp.X[rows, :]
    Y_tr = mp.Y[rows, :]
    post = fit_full_posterior(X_tr, Y_tr; tol = posterior_tol,
                              max_cells = posterior_max_cells)  # propriety 红 → 重抛（D-036）

    # --- 步骤 8：strict prequential residual history（P0-1 收口） ---
    # production 残差源 = prequential_residual_rows（不 export，同 include
    # 链可见）：每历史日期 s 用截至 s−1 的信息拟合、预测 s、观察 r_s、得到
    # ε_s、追加——历史当时真正发生的 forecast surprise（H_t → predict
    # r_{t+1} → observe r_{t+1} → ε_{t+1} → append）。3-fold OOF
    # （oof_residual_rows_full）已降级为 legacy/reference diagnostic，不再
    # 被 production 决策链消费（F_folds 退出 production theory）。
    resid_all = prequential_residual_rows(X_tr, Y_tr, E_active;
                                          tol = posterior_tol,
                                          max_cells = posterior_max_cells,
                                          u_span = u_span)
    # resid_all: n × N_act（asset 空间）。早期行（train 不足 A2 propriety）
    # 输出 NaN——残差不可定义。innovation_state 的 J 行判定只看 row_masks
    # （NaN 行若 mask 覆盖 R 会进 J → eps_R 非有限 → 崩溃），故此处**过滤
    # NaN 行**（与 innovation_state 契约「非 J 行的缺失由 mask 表达、NaN
    # 合法——J 行不合法」一致）：残差不可定义的行不进任何统计（V/ℓ/z
    # pool / 满秩 gate 全部基于可定义残差行）。
    keep_res = Int[i for i in 1:size(resid_all, 1) if all(isfinite, resid_all[i, :])]
    isempty(keep_res) &&
        error("single_day_decision: prequential 无任何可定义残差行——全部行 train 不足（A2 propriety）或退化；残差历史不可定义")
    resid = resid_all[keep_res, :]                # n_ok × N_act（asset 空间）
    row_ids_ok = row_ids[keep_res]                # 过滤后行身份（仍严格递增）
    row_masks_ok = row_masks[keep_res]            # 过滤后行 mask（与残差行序一致）

    # --- 步骤 8b：二次 resolve（P0-3 满秩 gate） ---
    # 首次 resolve（步骤 6）在 posterior 之前、拿不到残差，只做基础判据
    # （|J|≥2）。prequential 残差就绪后二次 resolve 传 residual_rows——
    # 「足」还要求 rank{ε_s^(R) : s ∈ J} = |R|（fractional kernel 正权重 ⇒
    # V_t(d) ≻ 0 ∀d；sample null space 不得宣布为 physical zero-risk
    # space）。剔除 free 直至满秩；locked 导致不可收缩 → 维持当日持仓
    # （T4，裁决 C3）。注意二次 resolve 与 innovation_state 都消费**过滤后**
    # 的 row_ids_ok/row_masks_ok/resid（NaN 行不进 joint 行集）。
    free_after_first = setdiff(free_act, dropped_free_act)
    try
        R_act2, dropped2 = resolve_risk_domain(row_ids_ok, row_masks_ok,
                                               free_after_first, locked_act,
                                               t; residual_rows = resid)
        R_act = R_act2
        append!(dropped_free_act, dropped2)
        N_R = length(R_act)
    catch err
        msg = err isa ErrorException ? err.msg : sprint(showerror, err)
        if occursin("innovation coverage failure", msg)
            return _held_maintained_result(held, act, free_act, locked_act,
                                           t, msg)
        end
        rethrow()                                 # 非 T4：不吞、不降级
    end

    # --- 步骤 9：innovation state（R 域 V_t/q(d)/z pool，D-045a） ---
    # require_full_rank=true（P0-3 收口）：二次 resolve 已保证 joint
    # residual span 满秩（prequential 残差），此处构造层再检查一次
    # （R 域 mode 坐标的 residual_rank——正交变换不改变秩，等价）——
    # sample null space 绝不进入生产决策（不得当零风险）。
    st = innovation_state(resid, row_ids_ok, row_masks_ok, R_act, E_active;
                          t = t, require_full_rank = true)

    # --- 步骤 10：决策行 feature ---
    x_now = vec(mp.X_full[t, :])                      # P = 1 + 14·N_act（含 DC 列）

    # --- 步骤 11：locked 权重（R 域列序；free 资产既有持仓进预算重优化） ---
    locked_set = Set(locked_act)
    locked_w = zeros(N_R)
    for (k, j) in enumerate(R_act)
        if j in locked_set
            locked_w[k] = held[act[j]]
        end
    end
    budget = 1.0 - sum(locked_w)
    locked_exposure = sum(locked_w)

    # --- 步骤 12：决策分支 ---
    # P0-4 收口：adaptive 请求一律走 adaptive_scenario_kelly（locked 直接
    # 由 quadrature.jl 的 solve_layer 处理——free-column Kelly + base_locked，
    # 方案 2 表示）。不再存在「adaptive 有 locked 就回落 :reference 固定 S」
    # 的 production fallback；:reference 仅保留为 D-062 合法用途（unit
    # test / artifact replay / fixture）。
    local w_R::Vector{Float64}
    local w_cash::Float64
    local certs::CertReport
    local M_dec::Int

    if mode == :adaptive
        # 生产路径（D-062）：双证书 A/B_opt/B_audit/C 收敛。locked 由
        # adaptive_scenario_kelly 直接处理（locked 列不进 free 优化列、其
        # scenario wealth 贡献由 base 承担——D-017/D-068）。
        res = adaptive_scenario_kelly(post, st, x_now;
                                      s1 = s1_act, E_active = E_active,
                                      rule_seed = seed,
                                      locked = locked_w,
                                      min_scenarios = min_scenarios,
                                      max_scenarios = max_scenarios,
                                      weight_tol = weight_tol,
                                      utility_tol = utility_tol,
                                      kelly_tol = kelly_tol,
                                      mu_qmc = mu_qmc,
                                      mu_chisq_qmc = mu_chisq_qmc)
        w_R = res.w_risky
        w_cash = res.w_cash
        certs = (; A = res.certificate_A, B = res.certificate_B,
                 B_audit = res.certificate_B_audit,
                 C = res.certificate_C, converged = true,
                 kelly_objective = res.certificate_C.objective)
        M_dec = res.M
    else
        # 固定 S reference 路径（D-062 合法用途；locked 非零时语义正确组装
        # ——free-column Kelly + base_locked，与 quadrature.jl 的 solve_layer
        # 同构）。
        pl = predictive_law(post, st, x_now; s1 = s1_act,
                            E_active = E_active, rng = MersenneTwister(seed),
                            S = S_reference)
        free_remaining = setdiff(free_act, dropped_free_act)
        free_pos = Int[k for (k, j) in enumerate(R_act) if j in Set(free_remaining)]
        base = locked_wealth_gate0(pl.gross, locked_w)
        if isempty(free_pos)
            # free 剔尽（R = locked）：无可优化列——预算全留 cash（唯一决策）。
            # 方案 2（P0-5）：wealth = w_cash·1 + base_locked（无 X·w 项——
            # locked 贡献由 base 承担；旧写法 pl.gross*locked_w .+ budget .+
            # base 中 pl.gross*locked_w == base → locked 双重计入）。
            w_R = copy(locked_w)
            w_cash = budget
            obj = isempty(pl.gross) ? 0.0 :
                mean(log.(budget .+ base))
            certs = (; A = NaN, B = NaN, B_audit = NaN,
                     C = (; feasibility = 0.0, kkt_residual = 0.0,
                          objective_gap = 0.0, objective = obj, dual = NaN),
                     converged = false, kelly_objective = obj)
        else
            X_free = pl.gross[:, free_pos]            # locked 列不进优化列（D-017）
            w_free, w_cash_f, cert = cash_kelly(X_free; base = base,
                                                budget = budget, tol = kelly_tol)
            w_R = zeros(N_R)
            w_R[free_pos] .= w_free
            for k in 1:N_R
                if locked_w[k] > 0
                    w_R[k] = locked_w[k]              # locked 维持（进 base 不进优化）
                end
            end
            w_cash = w_cash_f
            certs = (; A = NaN, B = NaN, B_audit = NaN, C = cert,
                     converged = false, kelly_objective = cert.objective)
        end
        M_dec = S_reference
    end

    # --- 步骤 13：写回 universe 坐标 ---
    w_universe = zeros(N)
    for k in 1:N_R
        w_universe[act[R_act[k]]] = w_R[k]
    end

    # --- 步骤 14：方差分解 + D-076 concentration（独立诊断样本） ---
    pmu = predict_mu(post, x_now)
    cov_asset = E_active * pmu.cov * E_active'        # ν≤2 时含 NaN（如实携带）
    var_epi = tr(cov_asset[R_act, R_act])
    EV = zeros(N_R, N_R)
    for g in eachindex(st.d_weights)
        EV .+= st.d_weights[g] .* st.V_t[g]
    end
    var_alea = tr(EV)
    not_computed = Symbol[]
    pmu.within_computed || push!(not_computed, :epistemic)  # ν≤2：NOT COMPUTED
    variance_split = (; epistemic = var_epi, aleatoric = var_alea,
                      within_computed = pmu.within_computed,
                      not_computed = not_computed)

    # top posterior mean direction（asset 空间 R 子集）
    mu_asset = E_active * pmu.mu
    mu_R = mu_asset[R_act]
    if N_R > 0 && any(!iszero, mu_R)
        tm = argmax(abs.(mu_R))
        top_mu_asset = act[R_act[tm]]
        top_mu_val = mu_R[tm]
    else
        top_mu_asset = 0
        top_mu_val = NaN
    end
    # top innovation covariance eigenmode（R 域 mode → asset 方向）
    if N_R > 0
        Fv = eigen(Symmetric(EV))
        lam_max = Fv.values[end]
        v_top = Fv.vectors[:, end]
        av = st.E_R * v_top                          # R 域 asset 方向
        top_innov_asset = act[R_act[argmax(abs.(av))]]
    else
        lam_max = NaN
        top_innov_asset = 0
    end

    # top 集中度（D-076）：全 R 域最大权重（含 locked 维持）
    top_k = N_R > 0 && maximum(w_R) > 0 ? argmax(w_R) : 0

    # utility margin：独立诊断样本（seed ⊻ _DIAG_SEED_XOR；诊断量非证书）。
    # P0-5 方案 2 表示：I = mean(log.(X_free·w_free .+ w_cash .+ base_locked))
    # ——locked 贡献由 base 承担，绝不在 X_full·w_full（已含 locked 填回）上
    # 再加 base（locked double-count）。margin 只在 free 权重上操作（locked
    # 不可交易、不可重排——D-017）；无 free 可优化列时 NOT COMPUTED。
    free_pos_diag = Int[k for k in 1:N_R if locked_w[k] == 0]
    margin = NaN
    if !isempty(free_pos_diag)
        w_free0 = w_R[free_pos_diag]
        top_free_k = maximum(w_free0) > 0 ? argmax(w_free0) : 0
        if top_free_k > 0
            rng_diag = MersenneTwister(seed ⊻ _DIAG_SEED_XOR)
            pl_d = predictive_law(post, st, x_now; s1 = s1_act,
                                  E_active = E_active, rng = rng_diag,
                                  S = diag_S)
            base_d = locked_wealth_gate0(pl_d.gross, locked_w)
            X_free_d = pl_d.gross[:, free_pos_diag]
            I_of(w_free, wc) =
                mean(log.(vec(X_free_d * w_free) .+ wc .+ base_d))
            w_shift = copy(w_free0)
            wc_shift = w_cash
            w_shift[top_free_k] = max(w_free0[top_free_k] - 0.01, 0.0)
            order = sortperm(w_free0; rev = true)
            if length(w_free0) >= 2
                w_shift[order[2]] = w_shift[order[2]] + 0.01
            else
                wc_shift = w_cash + 0.01             # 次优 = cash（单 free 资产）
            end
            margin = I_of(w_free0, w_cash) - I_of(w_shift, wc_shift)
        end
    end

    concentration = (; top_asset = top_k > 0 ? act[R_act[top_k]] : 0,
        top_weight = top_k > 0 ? w_R[top_k] : 0.0,
        risky_weight = sum(w_R), cash_weight = w_cash,
        posterior_log_growth = certs.kelly_objective,
        epistemic_variance = var_epi, innovation_variance = var_alea,
        utility_margin_1pct = margin, integration_M = M_dec,
        converged = certs.converged,
        top_posterior_mean_asset = top_mu_asset,
        top_posterior_mean_value = top_mu_val,
        top_innovation_eigenvalue = lam_max,
        top_innovation_mode_asset = top_innov_asset)

    return Gate0DayDecision(mode, t, R_act, Int[act[j] for j in R_act],
        w_universe, w_R, w_cash, locked_exposure, budget, certs, M_dec,
        variance_split, concentration, dropped_free_act,
        (; reason = :ok, error_text = "",
           dropped_free_universe = dropped_free_univ))
end

# ---------------------------------------------------------------------------
# 等权 benchmark（D-020：候选集外生）
# ---------------------------------------------------------------------------

"""
    equal_weight_benchmark(el, mf, t) -> (; t, mask, weights, n, symbols)

等权 benchmark（施工图 Step 15 / D-020）：候选集合 = `benchmark_universe
(el, t)`（= E^trade ∧ T^exec，**与 model_admitted 完全无关**——实验组的
admission 不能决定对照组有哪些资产；Eligibility 的构造纪律保证 mask 只
依赖 MarketFacts.observed 与用户政策）。等权 1/n 于候选集；空候选集
（n = 0）返回零权重（benchmark 无持仓——如实，不注入默认资产）。

`mf` 仅提供 `symbols`（报告用）；权重计算不消费任何模型对象——外生性
是签名层面的构造保证（不接收 model/posterior/active 概念）。
"""
function equal_weight_benchmark(el::Eligibility, mf::MarketFacts, t::Int)
    T, N = size(el.model_admitted)
    (size(mf.observed) == (T, N)) ||
        throw(DimensionMismatch("equal_weight_benchmark: mf 与 el 尺寸不符"))
    (1 <= t <= T) ||
        throw(ArgumentError("equal_weight_benchmark: 1 ≤ t ≤ T (got t=$t, T=$T)"))
    mask = benchmark_universe(el, t)                  # BitVector（N 维）
    n = count(mask)
    weights = n == 0 ? zeros(N) : Float64[mask[j] / n for j in 1:N]
    (; t = t, mask = mask, weights = weights, n = n,
       symbols = [mf.symbols[j] for j in findall(mask)])
end
