# =============================================================================
# KTraderGate0 — Step 14: adaptive RQMC（nested Sobol + Owen scramble +
# independent audit replicate + 双证书）。
# =============================================================================
#
# 所属 module: KTraderGate0。本文件由集成轮 include（KTraderGate0.jl，
# 顺序：… → posterior.jl → oof.jl → innovation.jl → kelly.jl → 本文件；
# 本文件消费 posterior.jl 的 draw_mu/ResponsePosterior、innovation.jl 的
# InnovationState/z_pool_rows/mp_sqrt_factors、kelly.jl 的
# cash_kelly/cash_kelly_certificate/locked_wealth_gate0——不复制求解层）。
# 为保持文件自含，依赖在此声明：
#     using Random, LinearAlgebra
#
# 规范来源（唯一权威，本文件逐条对照）：
# - docs/NUMERICAL_INTEGRATION_SPEC.md（两类误差的区分：求解误差 vs 积分
#   误差；嵌套确定性 rule 的硬性要求 §2.2；证书 (a)(b)(c) 的分层 owner
#   §3.1；fail loudly §4；S=300 降级条款 §5）；
# - AGENTS.md 裁决书 §20（D-061~D-067）；
# - docs/GATE0_IMPLEMENTATION_PLAN.md 修订版 Step 14 段。
#
# 裁决落地对照：
# - D-061/D-062：本文件是生产决策入口——不存在固定 S 返回路径；每次返回
#   都必须通过 A+B+C 三证书（min_scenarios 只是循环初始规模，docstring
#   钉死）。
# - D-063：nested **Owen-scrambled Sobol** 为 production 默认 backend
#   （理由是数值积分结构，非回测表现）。Halton 保留 legacy comparison
#   地位——本文件不实现 Halton，见文件尾「legacy 对照（D-063）」注释。
# - D-064：Certificate A（权重收敛 ‖w_{2M}−w_M‖₁ ≤ ε_w）+ Certificate B
#   （utility regret：I_audit(w_{2M}) − I_audit(w_M) ≤ ε_U，**独立 audit
#   rule** 上求值）+ Certificate C（fine 解的 cash_kelly_certificate 全绿
#   ——复用 kelly.jl，不复制）。三者不可互相顶替（NUMERICAL_INTEGRATION_
#   SPEC §3.1 owner 表：A+B 是积分收敛证据，C 是求解误差证据）。
# - D-065：audit rule 与 optimization rule 分离——audit scramble seed 由
#   rule seed 确定性派生（audit_seed = rule_seed ⊻ 0x9E3779B97F4A7C15），
#   防 sample optimization optimism（w_M 是在 optimization scenarios 上
#   选出来的，同一集合上检查 objective 存在 optimism）。
# - D-066：tolerance 默认值是 **D-066 synthetic 定稿流程的初始候选，非拍
#   脑袋终值**——数值依据见 `adaptive_scenario_kelly` docstring（历史先例
#   值 + 量纲论证；定稿须走 synthetic 解析律 → realistic single-day
#   fixture → 减半 → 稳定 → 记录为 numerical configuration，全程与
#   Sharpe/PnL 无关）。
# - D-067：预算耗尽 fail loudly——错误文本含 "Numerical integration did
#   not converge"，**不返回最后一层权重**。注意循环终止结构：当
#   M == max_scenarios 时不再生成同层比较（同层比较 A=B=0 是假绿——已
#   在控制流中结构性排除），顶层 (M_prev, max) 比较未过即 error。
#
# μ 通道混合设计（docstring 钉死，供 SPEC 复审）：
#   RQMC 点驱动 d 抽样（离散 inverse-CDF：cumsum 搜索）与 z 行抽样
#   （floor 变换）——一维/离散变换的天然适配；μ 通道（matrix-t posterior
#   draw）用**固定 seed 的独立流**（MersenneTwister(mu_seed)）——μ 的
#   epistemic 不确定性维度高（N 维联合 matrix-t）、无解析逆 CDF。μ 通道
#   的嵌套性由「每层从同一 seed 重建流」保证（前 M 个 draws 与上一层
#   逐位相同）。这是 reference 级的合法设计（裁决书 §27/D-082 reference
#   不追求性能极限；数学正确优先）。
#
# reference 纪律（D-082/D-083）：单线程、无跨调用缓存、无 workspace；
# rule_points 每次从头生成（O(M·dim·32) 重算完全可接受）。fail-loudly
# （SPEC §56）。全部判据与回测收益无关（SPEC §95 / 开发守则 §24）。
# =============================================================================

using Random
using LinearAlgebra

export SobolOwenRule, rule_points, audit_seed,
       RQMCScenarioSource, predictive_rqmc_source,
       AdaptiveKellyResult, adaptive_scenario_kelly

# ---------------------------------------------------------------------------
# 1. Sobol 序列（自实现，2 维，Joe-Kuo 方向数）
# ---------------------------------------------------------------------------
#
# 实现路径与范围（自实现的选择依据：避免新外部依赖 Sobol.jl——Project
# 无该包；reference 级单线程可接受）：
#
# - 方向数：Joe & Kuo (2008) 表的前 2 维种子。
#   * d=1：m_k = 1（∀k）→ v_k = 2^{32-k}——van der Corput 基 2 根式逆
#     （第一维的标准 Sobol 构造，无争议）。
#   * d=2：种子 (s=1, a=0, m_1=1)——Joe-Kuo 表第 2 行「2 1 0 1」。s=1
#     的多项式 P(y)=y+1 递推唯一确定：v_1 = 2^31，
#     v_k = v_{k−1} ⊕ (v_{k−1} >> 1)。由此生成的方向数 m 序列
#     1, 3, 5, 15, 17, 51, 85, 255, … 是 Sobol 文献中广泛引用的第二维
#     标准方向数（可由递推逐步验证：v_2 = 2^31 ⊕ 2^30 = 3·2^30，
#     v_3 = 5·2^29，v_4 = 15·2^28，…）。
# - 本任务的 RQMC 通道恰好需要 2 维（d 抽样 + z 行抽样；synthetic
#   Gaussian 对照 fixture 也是 2 维）。更高维需求（未来 μ 通道 RQMC 化）
#   需要扩充方向数表——`SobolOwenRule` 构造对 dim > 2 fail loudly，
#   扩展点在 _SOBOL_DIRV。
# - 生成：增量 XOR（Antonov–Saleev 形式）——点 index i 从 0 起连续生成：
#   x_0 = 0；x_i = x_{i−1} ⊕ v[ctz(i)+1]（ctz = 二进制尾零个数）。按全局
#   索引确定性生成 ⟹ **天然嵌套**：任意 M' > M 的序列前 M 点与 M 序列
#   逐位相同（NUMERICAL_INTEGRATION_SPEC §2.2 硬性要求 1）。

const _SOBOL_BITS = 32

"""Joe-Kuo 2008 前 2 维方向数（2×32 UInt32；见上方实现路径注释）。"""
function _sobol_direction_numbers()
    V = zeros(UInt32, 2, _SOBOL_BITS)
    for k in 1:_SOBOL_BITS
        V[1, k] = UInt32(1) << (_SOBOL_BITS - k)
    end
    V[2, 1] = UInt32(1) << (_SOBOL_BITS - 1)
    for k in 2:_SOBOL_BITS
        v = V[2, k - 1]
        V[2, k] = v ⊻ (v >> 1)
    end
    V
end

const _SOBOL_DIRV = _sobol_direction_numbers()

# ---------------------------------------------------------------------------
# 2. Owen scramble —— matrusca 线性（GF(2) affine）形式
# ---------------------------------------------------------------------------
#
# 形式（Matoušek 1998 的线性 scramble，Owen 1995 树 scramble 的线性
# 子族；任务书明确允许「XOR-based matrusca/线性 scramble」）：
#
#   y = c ⊕ ⊕_{j: x_j = 1} Col_j
#
# 其中 32 位整数 x 看作 GF(2)^32 向量；Col_j 是「输入位 j 对各输出位的
# 贡献掩码」——Col_j 的位 i 仅在 i ≤ j 可非零（下三角），对角位 j 恒置
# 1（单位下三角 ⟹ 双射）。c 是每维的 32 位 XOR 常数。
#
# 性质（docstring 钉死，诚实边界）：
# - **逐点确定性函数**：scramble(x) 只依赖 x 与 rule 的固定矩阵/常数 ⟹
#   Sobol 的前缀嵌套在 scramble 后逐位保持（§2.2 硬性要求 1 的 scramble
#   后半句）；同 seed 同输入逐位可重放（§2.2 要求 2 的基础）。
# - **net 性质保持**：线性 scramble 保持 Sobol (t,m,s)-net 结构
#   （Matoušek 1998）。方差分析保证（Owen 完整树 scramble 的 scrambled
#   variance estimator）在线性子族上弱于完整形式——reference 级合法，
#   标注供 SPEC 复审。
# - **随机化来源**：矩阵与常数由 MersenneTwister(seed) 派生（stdlib，
#   无新依赖；seed 显式、确定性）。

"""audit rule seed 的确定性派生（D-065）：rule_seed ⊻ 黄金比例常数。

audit rule 与 optimization rule 共用方向数、仅 scramble seed 不同——
两个 rule 的点集是同一 Sobol net 的两次独立随机化（独立性来自
MersenneTwister 流不同）。派生公式在此钉死：同 rule_seed 下 audit
seed 确定、可重放；不同 rule_seed 的 audit seed 不同。"""
audit_seed(rule_seed::UInt64) = rule_seed ⊻ 0x9E3779B97F4A7C15

"""μ 通道独立流 seed 的默认派生（与 audit_seed 用不同混合常数，避免
两个派生通道碰撞）。"""
_default_mu_seed(rule_seed::UInt64) = rule_seed + 0x2545F4914F6CDD1D

"""构造第 j 维的 32 个列掩码（单位下三角 GF(2)）+ XOR 常数。"""
function _make_scramble(rng::AbstractRNG)
    cols = Vector{UInt32}(undef, _SOBOL_BITS)
    for b in 1:_SOBOL_BITS
        # 低位掩码（位 0..b-2）随机；对角位 b-1 恒 1。
        lowmask = b == 1 ? UInt32(0) : (UInt32(1) << (b - 1)) - UInt32(1)
        cols[b] = (rand(rng, UInt32) & lowmask) | (UInt32(1) << (b - 1))
    end
    (cols, rand(rng, UInt32))
end

"""
    SobolOwenRule(dim, seed) -> rule

嵌套确定性 RQMC rule（D-063：nested Owen-scrambled Sobol）。

**rule 身份 = seed**（scramble 的随机化种子；方向数固定内建）。同 seed
构造的 rule 逐位可重放；不同 seed 的 rule 是同一 Sobol net 的独立
随机化（audit replicate 即由此构造，D-065）。

`dim` 当前上限 2（方向数表范围，见文件头注释）；超出 fail loudly。
"""
struct SobolOwenRule
    dim::Int
    seed::UInt64
    cols::Vector{Vector{UInt32}}
    xors::Vector{UInt32}
    function SobolOwenRule(dim::Int, seed::UInt64)
        1 <= dim <= 2 ||
            error("SobolOwenRule: 本 reference 实现支持 dim ∈ {1,2}（Joe-Kuo 前 2 维方向数）；got dim=$(dim)——扩充需扩展 _SOBOL_DIRV 方向数表")
        rng = MersenneTwister(seed)
        cols = Vector{Vector{UInt32}}(undef, dim)
        xors = Vector{UInt32}(undef, dim)
        for j in 1:dim
            c, x = _make_scramble(rng)
            cols[j] = c
            xors[j] = x
        end
        new(dim, seed, cols, xors)
    end
end

"""matrusca 线性 scramble 的单点应用（32 位整数域）。"""
function _scramble_u32(cols::Vector{UInt32}, c::UInt32, u::UInt32)
    y = c
    for b in 1:_SOBOL_BITS
        if (u >> (b - 1)) & UInt32(1) == UInt32(1)
            y ⊻= cols[b]
        end
    end
    y
end

"""
    rule_points(rule, M) -> Matrix{Float64}（M × dim，[0,1)）

按全局索引 0..M−1 生成 Sobol 点并施加 Owen（matrusca 线性）scramble。

**嵌套性（硬保证，§2.2 要求 1）**：`rule_points(rule, M)[1:M,:]` 与
`rule_points(rule, M')`（M' > M）的前 M 行**逐位相同**——生成从 index 0
确定性推进、scramble 是逐点确定性函数。M=0 返回 0×dim 空矩阵。
"""
function rule_points(rule::SobolOwenRule, M::Int)
    M >= 0 || throw(ArgumentError("rule_points: M ≥ 0 需要（got $M）"))
    dim = rule.dim
    M == 0 && return Matrix{Float64}(undef, 0, dim)
    pts = Matrix{Float64}(undef, M, dim)
    x = zeros(UInt32, dim)
    # 点 index 0：x_0 = 0（Sobol 原点）——scramble 后为常数向量 c ≠ 0
    # （一般情形），原点退化由 scramble 消化。
    for j in 1:dim
        pts[1, j] = Float64(_scramble_u32(rule.cols[j], rule.xors[j], UInt32(0))) / 2.0^_SOBOL_BITS
    end
    for i in 1:M-1
        vcol = trailing_zeros(UInt32(i)) + 1   # v 的列索引 = ctz(i) + 1
        for j in 1:dim
            x[j] = x[j] ⊻ _SOBOL_DIRV[j, vcol]
            pts[i + 1, j] = Float64(_scramble_u32(rule.cols[j], rule.xors[j], x[j])) / 2.0^_SOBOL_BITS
        end
    end
    pts
end

# ---------------------------------------------------------------------------
# 3. RQMC scenario source（积分对象的抽象）
# ---------------------------------------------------------------------------
#
# 设计动机（接口摩擦如实声明）：`predictive_law`（predictive.jl）的
# scenario 循环内嵌 rng 消费（draw_mu / draw_innovation 各占流位置），
# **无法注入外部 RQMC uniform**——d/行抽样的驱动源不可替换。本文件因此
# 定义 source 抽象：`gen(M, rule, mu_rng) -> Matrix{Float64}`（M×N_R 的
# gross return 矩阵，正有限）。生产 source `predictive_rqmc_source` 的
# 八步链条与 predictive.jl 逐行同构（μ draw → d → z 行 → ε = V^{1/2}z →
# y_R = μ_R + ε_R → u_R = E_R·y_R → ×s₁[R] → exp）；**唯一差异**是 d/行
# 抽样由 rule 点的确定性变换驱动（inverse-CDF / floor），μ 通道由调用方
# 传入的 mu_rng（固定 seed 重建 → 前缀嵌套）驱动。predictive.jl 的契约
# 校验逻辑随之复制到 source 构造器（两侧边界一致）。

"""
    RQMCScenarioSource(gen, n_assets, rqmc_dim)

RQMC 驱动的 scenario 生成器抽象（adaptive 循环的积分对象）。

- `gen(M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG) -> Matrix{Float64}`：
  M×n_assets 的 gross return 矩阵（全部正有限——kelly.jl 输入契约）。
  gen 消费 rule 的前 M 个点（嵌套）与 mu_rng（μ 通道独立流；synthetic
  source 可忽略 mu_rng）。gen 必须是**确定性**的：同 (M, rule, mu_rng)
  同输出。
- `n_assets`：N_R（free 列数——cash_kelly 的 X 列数）。
- `rqmc_dim`：gen 消费的 RQMC 维数（本实现 ≤ 2，SobolOwenRule 上限）。
"""
struct RQMCScenarioSource
    gen::Function
    n_assets::Int
    rqmc_dim::Int
    function RQMCScenarioSource(gen::Function, n_assets::Int, rqmc_dim::Int)
        n_assets >= 1 || throw(ArgumentError("RQMCScenarioSource: n_assets ≥ 1（got $n_assets）"))
        1 <= rqmc_dim <= 2 ||
            throw(ArgumentError("RQMCScenarioSource: rqmc_dim ∈ {1,2}（Sobol 方向数上限；got $rqmc_dim）"))
        new(gen, n_assets, rqmc_dim)
    end
end

"""
    predictive_rqmc_source(post, st, x_t; s1, E_active) -> RQMCScenarioSource

生产 scenario source：predictive law 的 RQMC 驱动版本。

八步链条与 `predictive_law`（predictive.jl）逐行同构（μ draw → d 抽样
→ z 行抽样 → ε = V_t(d)^{1/2}·z → y_R → u_R = E_R·y_R → r_R = u_R⊙s₁[R]
→ gross = exp(r_R)）；**唯一差异**在抽样驱动源：

- **d 通道（流位置 2）**：rule 点第 1 维 u_d → 离散 inverse-CDF
  （`cumsum(d_weights)` 搜索——与 draw_innovation 的
  `min(searchsortedfirst(cw, u1), n_d)` 同式）。
- **z 行通道（流位置 3）**：rule 点第 2 维 u_z →
  `idx_row = min(floor(Int, u_z·|L|) + 1, |L|)`（与 draw_innovation
  同式）。
- **μ 通道（流位置 1）**：`mu_rng`（调用方传入——adaptive 循环每层从
  固定 seed 重建，前 M 个 draws 与上一层逐位相同 ⟹ μ 通道嵌套）。

**μ 通道混合设计的语义（钉死，供 SPEC 复审）**：RQMC 驱动 d/z 通道的
inverse-CDF（一维/离散变换天然适配）；μ 的 matrix-t posterior draw 是
N 维联合分布、无解析逆 CDF，用固定 seed 独立流——μ 的 epistemic 不确
定性以「固定随机偏移族」进入积分，RQMC 负责 aleatoric 通道的积分收敛。
reference 级合法设计（D-082/D-083）。

D-060 红线继承：scenario 通道 = μ 的 matrix-t posterior draw（epistemic）
+ innovation draw（aleatoric）——本 source 不引入任何 Σ_R 的额外未来
冲击。

契约校验与 `predictive_law` 同源（两侧 asset 空间同一 active 域、
R_t 越界、s1 正有限、E_active 维度、x_t 长度/有限、propriety 证书）——
fail loudly 不吞错。
"""
function predictive_rqmc_source(post::ResponsePosterior,
                                st::InnovationState,
                                x_t::AbstractVector{Float64};
                                s1::AbstractVector{Float64},
                                E_active::AbstractMatrix{Float64})
    N = post.N
    length(x_t) == post.P ||
        throw(DimensionMismatch("predictive_rqmc_source: x_t 长度 $(length(x_t)) ≠ P=$(post.P)"))
    all(isfinite, x_t) ||
        throw(DomainError(collect(x_t), "predictive_rqmc_source: x_t 含 NaN/Inf"))
    size(E_active) == (N, N) ||
        throw(DimensionMismatch("predictive_rqmc_source: E_active 应为 $(N)×$(N)，实际 $(size(E_active))"))
    length(s1) == N ||
        throw(DimensionMismatch("predictive_rqmc_source: s1 长度 $(length(s1)) ≠ N=$(N)"))
    all(j -> isfinite(s1[j]) && s1[j] > 0, 1:N) ||
        throw(DomainError(collect(s1), "predictive_rqmc_source: s1 必须全部正有限"))
    length(st.row_masks) >= 1 ||
        error("predictive_rqmc_source: innovation state 无行（状态损坏）")
    length(st.row_masks[1]) == N ||
        throw(DimensionMismatch("predictive_rqmc_source: 两侧 asset 空间不一致——innovation 列数 $(length(st.row_masks[1])) ≠ posterior N=$(N)"))
    all(j -> 1 <= j <= N, st.R_t) ||
        throw(ArgumentError("predictive_rqmc_source: R_t 越界（1:$(N)）——R_t=$(st.R_t)"))
    post.propriety.proper ||
        error("predictive_rqmc_source: response posterior propriety 证书红（不吞错）")

    xt = collect(Float64, x_t)
    R = st.R_t
    N_R = st.N_R
    E_R = st.E_R
    s1_R = collect(Float64, s1)[R]
    n_d = length(st.d_nodes)
    nL = length(st.L_t)
    cw = cumsum(st.d_weights)
    # per-d 节点的确定性缓存（z pool 行矩阵 / V 因子——纯函数结果复用，
    # 无跨调用状态；reference 级允许）。
    z_cache = Dict{Int,Matrix{Float64}}()
    f_cache = Dict{Int,Any}()

    function gen(M::Int, rule::SobolOwenRule, mu_rng::AbstractRNG)
        pts = rule_points(rule, M)   # M×2（rqmc_dim = 2）
        gross = Matrix{Float64}(undef, M, N_R)
        for s in 1:M
            # 流位置 1：μ 通道（独立流——固定 seed 重建保证嵌套）
            mu_mode_s = draw_mu(post, xt, mu_rng)
            mu_R_s = E_R' * (E_active * mu_mode_s)[R]
            # 流位置 2：d 通道——RQMC inverse-CDF（与 draw_innovation 同式）
            u_d = pts[s, 1]
            idx_d = min(searchsortedfirst(cw, u_d), n_d)
            # 流位置 3：z 行通道——RQMC floor 变换（与 draw_innovation 同式）
            u_z = pts[s, 2]
            idx_row = min(floor(Int, u_z * nL) + 1, nL)
            # 反标准化：ε = V_t(d)^{1/2}·z_{s'}(d)（与 draw_innovation 同构）
            Z = get!(() -> z_pool_rows(st, st.d_nodes[idx_d]), z_cache, idx_d)
            F = get!(() -> mp_sqrt_factors(st.V_t[idx_d];
                                           cls_floor = st.z_pool.cls_floor),
                     f_cache, idx_d)
            z = Z[idx_row, :]
            eps = F.half * z
            # 步骤 4-7：y_R → u_R → r_R → gross（与 predictive_law 同构）
            y_R = mu_R_s .+ eps
            u_R = E_R * y_R
            r_R = u_R .* s1_R
            gross[s, :] .= exp.(r_R)
        end
        gross
    end
    return RQMCScenarioSource(gen, N_R, 2)
end

# ---------------------------------------------------------------------------
# 4. adaptive 循环 + 三证书
# ---------------------------------------------------------------------------

"""
    AdaptiveKellyResult

`adaptive_scenario_kelly` 的返回对象。

字段（构造后只读）：
- `w_risky` / `w_cash`：收敛层的 Kelly 权重（D-068 显式 cash 列语义，
  由 cash_kelly 求解）；
- `M`：收敛层 scenario 数；
- `rule_seed` / `audit_seed`：rule 身份（D-065——audit seed 确定性派生
  自 rule seed，重放/审计可追溯）；
- `certificate_A`：最终轮 ‖w_{2M} − w_M‖₁（权重收敛证书，D-064 A）；
- `certificate_B`：最终轮 I_audit(w_{2M}) − I_audit(w_M)（utility
  regret 证书，D-064 B——独立 audit rule 上的两个权重之目标间隙，
  **不是** w_prev 在 X_new 上的 KKT gap 代理——NUMERICAL_INTEGRA-
  TION_SPEC §3.4 钉死的差距语义）；
- `certificate_C`：fine 解的 `cash_kelly_certificate`（求解误差证书，
  D-064 C——复用 kelly.jl，不复制；cash_kelly 内部 fail-loudly 保证
  全绿，此处携带数值供审计/报告）；
- `converged`：true（false 路径 fail loudly 不返回——字段为报告完整
  性保留）；
- `iterations`：翻倍轮数；
- `history`：每轮 `(; M, M2, A, B)` 诊断序列（D-076 concentration
  报告可消费；测试断言「A 过 B 红必须继续翻倍」的依据）。
"""
struct AdaptiveKellyResult
    w_risky::Vector{Float64}
    w_cash::Float64
    M::Int
    rule_seed::UInt64
    audit_seed::UInt64
    certificate_A::Float64
    certificate_B::Float64
    certificate_C::NamedTuple
    converged::Bool
    iterations::Int
    history::Vector{NamedTuple}
end

"""
    adaptive_scenario_kelly(src::RQMCScenarioSource;
                            rule_seed, mu_seed, locked,
                            min_scenarios, max_scenarios,
                            weight_tol, utility_tol, kelly_tol)
        -> AdaptiveKellyResult

**生产决策入口**（D-062：无固定 S 返回路径）。循环：

```
M = min_scenarios
loop:
    M2 = min(2M, max_scenarios)
    在 optimization rule（同 rule 对象，嵌套）上生成 M2 点 scenario
    → cash_kelly 求解 w_{2M}
    Certificate A:  ‖w_{2M} − w_M‖₁ ≤ weight_tol
    Certificate B:  I_audit(w_{2M}) − I_audit(w_M) ≤ utility_tol
                    （独立 audit rule——audit seed 确定性派生，D-065；
                    同一 audit 样本上求值两个权重的目标间隙）
    Certificate C:  fine 解的 cash_kelly_certificate 全绿
                    （cash_kelly 内部 fail-loudly；数值携带于返回值）
    A ∧ B ∧ C → 返回（三者不可互替：A+B 积分收敛证据、C 求解误差证据）
    M2 == M（预算顶）→ fail loudly（D-067）
    M ← M2
```

**嵌套性**：每层用**同一 rule 对象**（同 seed）生成 M2 点——前 M 点
与上一层逐位相同（Sobol 全局索引 + scramble 逐点确定性）；μ 通道由
固定 seed 重建流保证前缀嵌套。**不重新洗牌、不换 rule、不重抽 shift**
（NUMERICAL_INTEGRATION_SPEC §2.2 硬性要求）。

**Certificate B 的语义（§3.1 owner 表）**：I_audit(w) = (1/M_a)·Σ_s
log(X_audit_s·w + w_cash + base_audit_s)，X_audit 是 audit rule 的
M_a = M2 个点（与 fine 层同规模的独立随机化）。B > 0 表示 w_M 在独立
audit 样本上还落后 fine 解 ε_U（regret）；B ≤ 0 表示 w_M 已不差于
fine 解——两种情况都通过（regret 语义）。**这不是** w_prev 在 X_new
上的 KKT gap 代理（旧线的 `kelly_certificate(X_new, w_prev)` 形态——
§3.4 明确列为差距）。

**预算耗尽（D-067）**：错误文本含 `"Numerical integration did not
converge"`（D-067 统一文本），**不
返回最后一层权重**。控制流结构性排除同层假绿：当 M 已达 max 时不再
生成 (max, max) 比较（A=B=0 的假通过），顶层 (M_prev, max) 比较未过
即 error。

**无固定 S 捷径（D-062 负测试锚点）**：本函数**没有**固定 scenario
数的返回路径——`min_scenarios` 只是循环初始规模（数值初始值语义），
每次返回都必须通过当轮 A+B+C 证书。不存在「跳过证书直接返回」的
分支。

**tolerance 默认值——D-066 synthetic 定稿流程的初始候选，非拍脑袋
终值**。数值依据（合成理据，与回测无关）：

- `weight_tol = 1e-3`：旧线 `adaptive_scenario_weights` 的 weight_tol
  （旧 src/predict.jl:647 的历史实测门限先例）；量纲 = N_R+1 维权重
  向量的 L1 距离（Kelly 权重 ∈ [0, budget]，总预算 1）。
- `utility_tol = 1e-5`：旧线 integration 层门限（旧 predict.jl:647 的
  objective_gap ≤ tol 先例）；量纲 = 每日 log-wealth 期望值（log 域，
  典型量级 1e-2~1e-4 的日 edge）——比权重容差严两个量级，因为目标值
  是积分量、其绝对精度直接进入决策质量。
- `min_scenarios = 64, max_scenarios = 512`：旧线 adaptive 的实测
  范围（NUMERICAL_INTEGRATION_SPEC §1.3 记录的现状）。
- 定稿路径（D-066）：synthetic 解析律（本步测试的 known Gaussian
  2-asset analytic Kelly）→ realistic single-day fixture → 容差减半
  → 权重/目标稳定 → 记录为 numerical configuration。**全程与
  Sharpe/PnL 完全无关**（SPEC §95 / 开发守则 §24）。

**locked 语义（D-017/D-068）**：`locked`（默认全零）是 R 域持仓权重
向量；每层对当层 scenario 矩阵计算 `base = locked_wealth_gate0(X,
locked)`（locked 风险参与 wealth、不进 free 优化列），budget =
1 − Σ locked。locked 风险绝不当 cash。

**μ 通道 seed**：`mu_seed`（默认 = `_default_mu_seed(rule_seed)`——
确定性派生，docstring 钉死）；audit 集合的 μ 流 seed = mu_seed ⊻
audit 常数（与 audit rule 的随机化独立）。
"""
function adaptive_scenario_kelly(src::RQMCScenarioSource;
                                  rule_seed::UInt64 = 0x1D9B7A5C_E11A0F42,
                                  mu_seed::Union{Nothing,UInt64} = nothing,
                                  locked::Vector{Float64} =
                                      zeros(src.n_assets),
                                  min_scenarios::Int = 64,
                                  max_scenarios::Int = 512,
                                  weight_tol::Float64 = 1e-3,
                                  utility_tol::Float64 = 1e-5,
                                  kelly_tol::Float64 = 1e-8)
    # --- 输入校验（fail-loudly，SPEC §56） ---
    min_scenarios >= 1 ||
        throw(ArgumentError("adaptive_scenario_kelly: min_scenarios ≥ 1（got $min_scenarios）"))
    max_scenarios >= min_scenarios ||
        throw(ArgumentError("adaptive_scenario_kelly: max_scenarios ≥ min_scenarios（$max_scenarios < $min_scenarios）"))
    weight_tol >= 0 || throw(ArgumentError("adaptive_scenario_kelly: weight_tol ≥ 0（got $weight_tol）"))
    utility_tol >= 0 || throw(ArgumentError("adaptive_scenario_kelly: utility_tol ≥ 0（got $utility_tol）"))
    kelly_tol > 0 || throw(ArgumentError("adaptive_scenario_kelly: kelly_tol > 0（got $kelly_tol）"))
    length(locked) == src.n_assets ||
        throw(DimensionMismatch("adaptive_scenario_kelly: locked 长度 $(length(locked)) ≠ n_assets $(src.n_assets)"))
    all(x -> isfinite(x) && x >= 0, locked) ||
        throw(ArgumentError("adaptive_scenario_kelly: locked 必须非负有限"))
    budget = 1.0 - sum(locked)
    budget >= 0 ||
        throw(ArgumentError("adaptive_scenario_kelly: locked 总和超过预算（Σlocked = $(sum(locked))）"))

    # --- 双 rule（D-065：audit seed 确定性派生） ---
    opt_rule = SobolOwenRule(src.rqmc_dim, rule_seed)
    a_seed = audit_seed(rule_seed)
    audit_rule = SobolOwenRule(src.rqmc_dim, a_seed)
    m_seed = mu_seed === nothing ? _default_mu_seed(rule_seed) : UInt64(mu_seed)
    am_seed = m_seed ⊻ 0x9E3779B97F4A7C15

    # --- 单层求解（optimization rule；μ 流固定 seed 重建 → 嵌套） ---
    # locked 通道修复（Manager 裁决 5，Wave 4 执行轮）：locked 列不进
    # free 优化列——切 free 子列传 cash_kelly、全列（free+locked）进
    # base（locked_wealth_gate0 的调用约定「X is the FULL scenario
    # matrix (free and locked)」）。修复前把 R 域全列直接传 cash_kelly
    # （locked 列进优化列 + base 双重计入——与 kelly.jl 的 D-017 语义
    # 冲突，driver.jl 文件头「接口摩擦」段记录的分歧）。与 driver.jl
    # 的 reference 路径同构。
    free_pos = [k for k in 1:src.n_assets if locked[k] == 0]
    function solve_layer(M::Int)
        X = src.gen(M, opt_rule, MersenneTwister(m_seed))
        base = locked_wealth_gate0(X, locked)
        if isempty(free_pos)
            # free 列空（全 locked）：无可优化列——预算全留 cash（唯一
            # 决策，与 driver 的 reference 路径同构）。证书为平凡可行
            # 解（w=0、gap=0——cash_kelly 的 degenerate 形态）。
            return (zeros(0), budget, (; feasibility = 0.0,
                kkt_residual = 0.0, objective_gap = 0.0,
                objective = isempty(X) ? 0.0 :
                    mean(log.(vec(X * locked) .+ budget .+ base)),
                dual = NaN))
        end
        X_free = X[:, free_pos]
        w_free, w_cash, cert = cash_kelly(X_free; base = base,
                                           budget = budget, tol = kelly_tol)
        w_full = zeros(src.n_assets)
        w_full[free_pos] .= w_free
        for k in 1:src.n_assets
            if locked[k] > 0
                w_full[k] = locked[k]          # locked 维持（进 base 不进优化）
            end
        end
        (w_full, w_cash, cert)
    end

    M = min_scenarios
    w_M, wc_M, _ = solve_layer(M)
    history = Vector{NamedTuple}()
    iterations = 0
    while true
        M2 = min(2 * M, max_scenarios)
        if M2 == M
            # 已在预算顶（上一轮已是 (M_prev, max) 的有效比较）——
            # 不生成同层 (max, max) 比较（A=B=0 假绿），直接 fail。
            break
        end
        w_2M, wc_2M, cert_2M = solve_layer(M2)
        iterations += 1
        # Certificate C：cert_2M 已由 cash_kelly fail-loudly 保证全绿
        # （不过即 error——不返回 heuristic 权重）；数值携带供审计。
        # Certificate A（权重收敛，D-064 A）：
        A = norm(vcat(w_2M, wc_2M) .- vcat(w_M, wc_M), 1)
        # Certificate B（utility regret，D-064 B / D-065）：
        # 同一 audit 样本（audit rule 的 M2 点 + audit μ 流）上求值
        # 两个权重的目标值——I_audit(w_{2M}) − I_audit(w_M)。
        X_a = src.gen(M2, audit_rule, MersenneTwister(am_seed))
        base_a = locked_wealth_gate0(X_a, locked)
        wealth_f = X_a * w_2M .+ wc_2M .+ base_a
        wealth_c = X_a * w_M .+ wc_M .+ base_a
        (all(x -> isfinite(x) && x > 0, wealth_f) &&
         all(x -> isfinite(x) && x > 0, wealth_c)) ||
            error("adaptive_scenario_kelly: audit objective 求值遇到非正 wealth——scenario source 违反正有限契约")
        I_f = sum(log, wealth_f) / M2
        I_c = sum(log, wealth_c) / M2
        B = I_f - I_c
        push!(history, (; M = M, M2 = M2, A = A, B = B))
        if A <= weight_tol && B <= utility_tol
            return AdaptiveKellyResult(w_2M, wc_2M, M2, rule_seed, a_seed,
                                       A, B, cert_2M, true, iterations,
                                       history)
        end
        M = M2
        w_M, wc_M = w_2M, wc_2M
    end
    # D-067：预算耗尽——统一错误文本；不返回最后一层权重。
    h = isempty(history) ? "(no refinement layer fit in budget)" :
        "(last layer M=$(history[end].M)→$(history[end].M2): A=$(history[end].A), B=$(history[end].B))"
    error("Numerical integration did not converge by $max_scenarios scenarios $h — D-067: no last-layer weights returned")
end

"""
    adaptive_scenario_kelly(post, st, x_t; s1, E_active, kwargs...)

生产便捷入口：构造 `predictive_rqmc_source(post, st, x_t; s1, E_active)`
后进入通用 `adaptive_scenario_kelly(src; ...)`。kwargs 透传（rule_seed /
mu_seed / locked / min_scenarios / max_scenarios / weight_tol /
utility_tol / kelly_tol）。
"""
function adaptive_scenario_kelly(post::ResponsePosterior,
                                 st::InnovationState,
                                 x_t::AbstractVector{Float64};
                                 s1::AbstractVector{Float64},
                                 E_active::AbstractMatrix{Float64},
                                 kwargs...)
    src = predictive_rqmc_source(post, st, x_t; s1 = s1, E_active = E_active)
    adaptive_scenario_kelly(src; kwargs...)
end

# ---------------------------------------------------------------------------
# legacy 对照（D-063）：Halton 保留 reference/legacy comparison 地位
# ---------------------------------------------------------------------------
# D-063 裁决：production 默认 backend 是 nested Owen-scrambled Sobol；
# Halton（旧线 ScenarioQuadrature 的形态）保留为 legacy comparison。
# 本文件不实现 Halton——对照实验（Sobol vs Halton 的收敛速率比较）属于
# 数值 backend 变更评估（NUMERICAL_INTEGRATION_SPEC §6 张力 2：backend
# 替换须重走 §3 证书），留待需要时另立交付。接口扩展点：实现
# `HaltonRule(dim, seed)` 并提供与 `rule_points` 同签名的生成函数即可
# 接入本循环（rule 抽象只要求「同 rule 对象的嵌套确定性点列」）。
