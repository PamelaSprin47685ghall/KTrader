# ============================================================================
# src/gate0/innovation.jl —— vector innovation reference（Step 10/11/12）
# ============================================================================
#
# 文件地位（裁决 H1/H2，docs/GATE0_MANAGER_ADJUDICATIONS.md）：未来顶层
# module `KTraderGate0` 的组成文件；当前按裸文件交付（同 geometry.jl /
# modes.jl），由测试经临时 module include（test/gate0/innovation_tests.jl）。
# 主 module 的 include 骨架归 DevOps 验证轮（本任务不改 KTraderGate0.jl）。
# 依赖链：geometry.jl（domain_mode_basis）→ modes.jl
# （risk_domain_mode_basis）→ 本文件。文件内自含 `using Random,
# LinearAlgebra`（两条加载路径——临时 module 与未来主 module——均获得
# 所需名字，同 geometry.jl 模式）。
#
# 职责（施工图 docs/GATE0_IMPLEMENTATION_PLAN.md §2.5 与 Step 10/11/12）：
# - R 域实例化（裁决 A / D-045a）：asset 空间 OOF 残差行 → R_t 子向量提取
#   → E_{R_t} mode 变换 → ε_s^{(R)} ∈ R^{N_R}（禁止从 active 域 mode 残差
#   直接取子向量——裁决 A3：mode 坐标无资产子集索引，数学上无定义）；
# - V_t(d)（VI §2）：kernel 加权联合残差二阶矩；分子分母同步收缩到
#   joint 合法行集 J_t（裁决 C1）；全部因果历史无窗口（D-048）；burn
#   废除（D-049：有效起点 = V_{s−1} 可定义性）；
# - ℓ_d 与 q(d|H)（VI §3）：Gaussian covariance quasi-likelihood（D-57），
#   先验 Uniform(0,1)（D-047），η≡1（D-058），direct sums 无 FFT
# （Step 12 / D-085：reference 先证明数学）；
# - Moore-Penrose support 协议（VI §4；D-050/D-051）；
# - standardized empirical shape（VI §5/§6；D-052/D-053/D-055）：per-d
#   joint 行 pool，禁 own-cell stitching（D-054）；
# - 覆盖不足 T4 判定次序（D-056 / 裁决 C3）。
#
# 规范来源：docs/GATE0_VECTOR_INNOVATION.md（VI——规范推导，本文件逐节
# 对照）；docs/GATE0_IMPLEMENTATION_PLAN.md §2.5/Step 10-12；
# docs/GATE0_MANAGER_ADJUDICATIONS.md 裁决 A/A2/A3/C1-C3/F1。
#
# 接口锚（VI §0，与 GATE0_RESPONSE_POSTERIOR.md 逐字一致；裁决 D1）：
#   ε 为统一 mode 残差：源对象是 asset 空间 OOF 残差行
#   ε̃_s = E_active·[y_s − (b₀^{(−fold(s))} + G^{(−fold(s))}·x_{s−1})]
#   （含 b₀ 扣除，ŷ = E[y_s|train^{(−fold)}] 为 D-041 期望语义）；
#   innovation 层按决策日 risk 域 R_t 消费其 R 子向量经 E_{R_t} 的 mode
#   变换。**b₀ 扣除 / OOF 隔离语义由调用方（response 层 Step 8/9）保证
#   ——本层不验证、不重算**（OOF/b₀ 归 response 层唯一所有）。
#
# reference 纪律（D-082/D-083/施工图 Step 10）：单线程、无 FFT、无增量、
# 无缓存、无 workspace；direct O(T²·N_R²) 完全可接受。fail-loudly
# （SPEC §56）。全部判据与回测收益无关（SPEC §95 / 开发守则 §24）。
#
# 与 EB_COVARIANCE_FLOOR 的语义区分（VI §4.2 / SPEC §25）：response 层 EB
# 的 floor 是数值保护；innovation 层**不存在**理论 floor——本文件的
# `cls_floor` 仅用于浮点谱分类（区分真零特征与数值小特征），不进入
# V_t 的数学定义，且必须接受 floor→0 refinement 检验（D-050/D-051）。

using Random, LinearAlgebra

export InnovationState, innovation_state, SupportCertificate,
    DConditionedShapePool, draw_innovation, z_pool_rows, z_row_at, V_at,
    shape_moments, quasi_loglik, joint_row_indices, coverage_report,
    resolve_risk_domain, default_d_grid, frac_weights_gate0,
    mp_sqrt_factors

# ---------------------------------------------------------------------------
# kernel：统一 fractional family（D-046 / VI §2.1）
# ---------------------------------------------------------------------------

"""
    frac_weights_gate0(d, n) -> Vector{Float64}    # 长度 n

统一 fractional kernel 权重（D-046）：`k_d(τ) = π_{τ−1}`，递推
`π₀ = 1`、`π_k = π_{k−1}·(k−1+d)/k`。返回 `w` 满足 `w[τ] = k_d(τ)`（τ =
1..n，即 lag-τ 的 kernel 权重）。

- 语义来源：与旧线 `frac_weights`（src/predict.jl:12，SPEC §31 的 π_k）
  **同一递推、同一 family**（D-046：禁止 macro/relative/每资产各一套
  kernel）；本实现从旧线语义**独立实现**（显式循环，不依赖旧线代码——
  裁决 H1 互不 include）。
- 对一切 d ∈ (0,1]、一切 k：π_k > 0（递推系数 (k−1+d)/k > 0）——kernel
  权重严格为正，无硬截断（VI §2.1）。d=1：π_k ≡ 1（等权）；d→0⁺：
  π_k → 1/k（近期主导）。
- d 的合法域：(0,1]（D-47 先验支撑；d=1 为闭区间端点极限包含，d=0 不在
  支撑内）。n ≥ 1。
"""
function frac_weights_gate0(d::Real, n::Integer)
    dv = Float64(d)
    (isfinite(dv) && dv > 0.0 && dv <= 1.0) ||
        throw(DomainError(dv, "frac_weights_gate0: d 必须落在 (0,1]（D-47 先验支撑；d=0 不在支撑内）"))
    nv = Int(n)
    nv >= 1 || throw(ArgumentError("frac_weights_gate0: n ≥ 1 需要（n = $(n)）"))
    w = Vector{Float64}(undef, nv)
    w[1] = 1.0
    for k in 1:nv-1
        w[k+1] = w[k] * (k - 1 + dv) / k
    end
    w
end

# ---------------------------------------------------------------------------
# joint 合法行集（VI §1.3 推论 2 / §6.2；裁决 C2 单一行集）
# ---------------------------------------------------------------------------

"""
    joint_row_indices(row_ids, row_masks, R, t) -> Vector{Int}

joint 合法行集 `J(R, t) = {s : u_s ≤ t, O_s ⊇ R}`（VI §1.3 推论 2）。

- **单一行集概念（裁决 C2）**：V 统计 / z pool / ℓ 求和三行集同一定义
  ——都要求「覆盖 R + 因果性」（V 统计额外要求自身可定义、z/ℓ 要求
  V_{s−1} 可定义，均由 J 派生，无第二套 mask 判定）。
- `row_ids[i] = u_i`（行目标日，**严格递增**——VI §1.2 契约 3，由
  innovation_state 构造断言；本函数不重复断言）。
- `row_masks[i][j]`：行 i 的 O_s 资产观测掩码（VI §1.2 契约 4：O_s =
  {j : r[u_s,j] finite}，与 response 层逐字一致——裁决 D2）。
- 空观察行（O_s = ∅）在 R 非空时构造性不满足 O_s ⊇ R，不入 J（VI
  §1.4：空观察行不进任何统计——它不是「零残差」）。
- 未来行（u_s > t）构造性不入 J（VI §7 因果性定理一）。
"""
function joint_row_indices(row_ids::AbstractVector{<:Integer},
                           row_masks::AbstractVector{<:AbstractVector{Bool}},
                           R::AbstractVector{<:Integer},
                           t::Integer)
    n_rows = length(row_ids)
    length(row_masks) == n_rows ||
        throw(DimensionMismatch("joint_row_indices: row_ids 与 row_masks 长度不符（$(n_rows) vs $(length(row_masks))）"))
    isempty(R) && throw(ArgumentError("joint_row_indices: risk 域 R 非空（N_R ≥ 1）——空 R 由调用方按 cash 出路处理，不入统计"))
    N_a = length(row_masks[1])
    for m in row_masks
        length(m) == N_a ||
            throw(DimensionMismatch("joint_row_indices: 行 mask 长度必须一致（$(N_a)）"))
    end
    for j in R
        (1 <= j <= N_a) || throw(ArgumentError("joint_row_indices: R 含越界资产索引 $(j)（universe 列数 $(N_a)）"))
    end
    J = Int[]
    for i in 1:n_rows
        if row_ids[i] <= t
            ok = true
            for j in R
                if !row_masks[i][j]
                    ok = false
                    break
                end
            end
            ok && push!(J, i)
        end
    end
    return J
end

"""
    coverage_report(row_ids, row_masks, R, t) -> NamedTuple

覆盖不足诊断（裁决 C3：回测驱动器捕获错误时记录诊断——当日资产、覆盖
缺口、行集大小）。字段：

- `n_rows_total` / `n_causal`：输入行总数 / u_s ≤ t 的因果行数；
- `n_joint`：|J(R,t)|（joint 合法行数）；
- `per_asset_rows::Vector{Int}`：R 中每资产的因果覆盖行数（u ≤ t 且该行
  mask 含该资产），与 R 同序；
- `coverage_gap::Vector{Int}`：R 中因果覆盖行数 < 2 的资产（覆盖不足的
  嫌疑资产——不足以单独判定，供诊断）；
- `sufficient::Bool`：|J| ≥ 2（V 可定义且 ℓ 有信息的合并判据——VI §3.4
  两断言）。
"""
function coverage_report(row_ids::AbstractVector{<:Integer},
                         row_masks::AbstractVector{<:AbstractVector{Bool}},
                         R::AbstractVector{<:Integer},
                         t::Integer)
    J = joint_row_indices(row_ids, row_masks, R, t)
    n_causal = count(<=(t), row_ids)
    per_asset = [count(i -> (row_ids[i] <= t && row_masks[i][j]), 1:length(row_ids))
                 for j in R]
    (; n_rows_total = length(row_ids),
       n_causal = n_causal,
       n_joint = length(J),
       R = collect(R),
       per_asset_rows = per_asset,
       coverage_gap = [R[j] for j in eachindex(R) if per_asset[j] < 2],
       sufficient = length(J) >= 2)
end

# ---------------------------------------------------------------------------
# V_t(d)：direct kernel 加权联合二阶矩（VI §2；裁决 C1）
# ---------------------------------------------------------------------------

"""
    V_direct(eps_R, u_J, d, decision_time) -> Matrix{Float64}    # N_R×N_R

vector innovation memory 的 direct 实现（VI §2.2 主对象；**reference 级
O(|J|·N_R²) per call，无 FFT/无缓存——Step 10/D-085**）：

```
V(d) = Σ_{s∈J} k_d(τ_s)·ε_s ε_sᵀ / Σ_{s∈J} k_d(τ_s)，τ_s = decision_time+1−u_s
```

- **分子分母同步收缩到 J**（裁决 C1 / VI §2.2 要点 1）：分母不是裸
  Σ k_d——分母含缺失行权重而分子不含 = 把缺失伪装成零方差证据（SPEC
  §12.1/D-012 的观测污染在二阶矩层的翻版）。完整 panel（全部行 joint
  合法）时分母退化为裁决书 D-045 字面形式 Σ_τ k_d（VI §9 检查 5 的
  对照点）。
- **lag 约定（VI §2.1）**：τ = decision_time+1−u_s ≥ 1（u_s ≤
  decision_time 由构造保证；本函数防御性断言）。V_t 用
  `decision_time = t`；V_{s−1} 用 `decision_time = u_s − 1`（行集取 u
  严格早于 u_s 的前缀）。
- `eps_R` 行序必须与 `u_J` 一致（u 递增）；`u_J` 严格递增。
- 构造性 PSD（VI §2.3：半正定矩阵的正系数组合）；|J| ≥ 1（分母 > 0）
  否则 error（V 不可定义——D-049/D-050）。
- **本函数不含任何 floor/δ**：V 的数学定义无正则化（VI §4.2：禁止
  V+δI 偷偷创造理论风险）；浮点谱分类归 `mp_sqrt_factors`。
"""
function V_direct(eps_R::AbstractMatrix{Float64},
                  u_J::AbstractVector{<:Integer},
                  d::Real,
                  decision_time::Integer)
    nJ = length(u_J)
    size(eps_R, 1) == nJ ||
        throw(DimensionMismatch("V_direct: eps_R 行数 $(size(eps_R,1)) 与行集 $(nJ) 不符"))
    nJ >= 1 ||
        error("V_direct: 行集为空——V(d) 不可定义（|J|=0，D-049/D-050：fail 优先于任何拼接/补零）")
    N_R = size(eps_R, 2)
    for i in 1:nJ
        u_J[i] <= decision_time ||
            error("V_direct: 因果性违反——行 $(i) 的目标日 u=$(u_J[i]) > 决策时刻 $(decision_time)（VI §7 定理一）")
        # 严格递增检查（Wave 3 修复：原 `i >= 2 && (u_J[i] > u_J[i-1]) ||
        # throw(...)` 的优先级在 i=1 时求值为 (false && …) || throw——
        # 恒抛，V_direct 从未被成功调用过。i=1 无前元素，应跳过检查。）
        if i >= 2 && u_J[i] <= u_J[i - 1]
            throw(ArgumentError("V_direct: u_J 必须严格递增（VI §1.2 契约 3）"))
        end
    end
    τ_max = decision_time + 1 - u_J[1]
    w = frac_weights_gate0(d, τ_max)
    V = zeros(N_R, N_R)
    denom = 0.0
    @inbounds for i in 1:nJ
        τ = decision_time + 1 - u_J[i]
        wi = w[τ]
        denom += wi
        for a in 1:N_R, b in 1:N_R
            V[a, b] += wi * eps_R[i, a] * eps_R[i, b]
        end
    end
    V ./= denom
    return V
end

# ---------------------------------------------------------------------------
# Moore-Penrose support 协议（VI §4；D-050/D-051）
# ---------------------------------------------------------------------------

"""
    mp_sqrt_factors(V; cls_floor = nothing) -> NamedTuple

对称 PSD 矩阵 V 的 Moore-Penrose 平方根因子（谱定义，VI §4.1）：

```
V = U Λ Uᵀ（eigen，升序）；
V^{1/2}  = U_pos · Diag(√λ_pos) · U_posᵀ     # 唯一对称 PSD 主平方根
V^{+1/2} = U_pos · Diag(λ_pos^{−1/2}) · U_posᵀ  # Moore-Penrose 逆平方根
```

- **null support 语义（D-050/D-051）**：零特征方向**不逆**（V^{+1/2} 的
  null 块为零算子，不是 δ^{−1/2}）、**不注入随机噪声**、**保留为 null
  support**（该方向 variance 为 0 = 历史未提供该方向二阶矩证据的忠实
  表达）。**禁止 V+δI**——任何以可逆性为由加 δI 的做法都是偷偷创造
  理论风险（VI §4.2）。
- **cls_floor 仅浮点分类**（VI §4.3）：区分「真零特征」与「数值小特
  征」的阈值 `λ > cls_floor` 判正；**不进入 V 的数学对象**（V_direct
  无 floor 参数）。默认 `N_R·eps()·max(λ_max, realmin)`。**必须**接受
  floor→0 refinement 证书（floor 减半序列下 z/预测矩收敛；不收敛即
  fail-loudly）；禁止把 cls_floor 当理论参数或用回测选择其值。
- **与 EB_COVARIANCE_FLOOR 的区分（SPEC §25 / VI §4.2）**：response 层
  EB floor 是数值保护；innovation 层不存在理论 floor。二者语义不同，
  不得混同。
- 返回 `(; half, inv_half, rank, eigenvalues, positive, cls_floor)`：
  `half`/`inv_half` 为 N×N 对称矩阵（null 块为零）；`positive` 为正特
  征值列表（升序）；`rank = length(positive)`。
- 因子约定（VI §5.1）：对称 PSD 主根族（谱定理唯一性；简并特征子空间
  内的旋转与符号翻转被 √Λ 常数块吸收）——沿用 `principal_sqrt_root`
  的 Manager 裁决（src/numerics.jl:438-446 的唯一性论证同源）。标准化
  与反标准化**不同矩阵、同一约定**；「禁止两次独立分解」指对同一 V 不
  得用两个不同约定的因子相乘冒充 V 的函数——主根族天然免除该歧义。
"""
function mp_sqrt_factors(V::AbstractMatrix{Float64}; cls_floor::Union{Nothing,Real} = nothing)
    N = size(V, 1)
    size(V, 2) == N ||
        throw(DimensionMismatch("mp_sqrt_factors: V 必须方阵（$(size(V))）"))
    F = eigen(Symmetric(V))
    λ = F.values                       # 升序
    U = F.vectors
    λmax = isempty(λ) ? 0.0 : λ[end]
    # 自动 floor（Wave 3 修正）：N·sqrt(eps)·λmax。原 N·eps·λmax 对
    # 「数值零特征」（rank 亏 V 的浮点消去噪声，实测 ~1e-17 @ λmax~1e-4，
    # 相对量级 ~1e-13）分类失败——被当正特征后 inv_half 的 1/√λ ~ 3e8
    # 使 z 爆炸（9.1 rank 断言 / 9.7 M_z 峰度同根因）。sqrt(eps) 相对
    # 量级（~1e-8）稳健覆盖消去噪声；显式 cls_floor / floor→0
    # refinement 语义不变（VI §4.3：仅浮点分类，非理论参数）。
    fl = cls_floor === nothing ? N * sqrt(eps(Float64)) * max(λmax, floatmin(Float64)) : Float64(cls_floor)
    fl >= 0.0 || throw(DomainError(fl, "mp_sqrt_factors: cls_floor 必须非负（浮点分类阈值）"))
    posmask = λ .> fl
    Upos = U[:, posmask]
    λpos = λ[posmask]
    half = Upos * Diagonal(sqrt.(λpos)) * Upos'
    inv_half = Upos * Diagonal(1.0 ./ sqrt.(λpos)) * Upos'
    return (; half = half, inv_half = inv_half, rank = length(λpos),
            eigenvalues = λ, positive = λpos, cls_floor = fl)
end

# ---------------------------------------------------------------------------
# ℓ_d（D-57）与 d quasi-posterior 数值 quadrature（D-47/D-58）
# ---------------------------------------------------------------------------

"""
    quasi_loglik(eps_R, u_J, d; cls_floor = nothing) -> Float64

Gaussian covariance quasi-likelihood（D-57 / VI §3.2，**mode-space 精确
形式**；direct sums，无 FFT——Step 12）：

```
ℓ_d = −½ Σ_{s∈L} [ log det⁺ V_{s−1}(d) + ε_sᵀ V_{s−1}(d)⁺ ε_s ]
```

- **求和行集 L（裁决 C2 / VI §3.2）**：J 中 V_{s−1} 可定义者 = 行
  2..nJ（J 按 u 严格递增——第一行无更早 joint 合法历史，其 V 无定义，
  **不进入 likelihood**：D-049「有效起点由数学可定义性决定」的精确形
  式；无 burn、无窗口、无补零）。
- `V_{s−1}`：行集 = J 中 u 严格早于 u_s 的前缀，lag τ = u_s − u_{s'}。
- `log det⁺`：正特征值之和的 log（null 方向不计入行列式）；二次型中 ε
  的 null 分量被 V⁺ 零化——**未识别方向不评分**（D-57「只在有效
  support 上计算」的执行）。
- quasi 声明（D-57/D-42）：经验 shape 非高斯时这不是完整 likelihood；
  q 是 quasi-posterior，不是普通 Bayes posterior。
- nJ ≥ 2 否则 error（|L|=0：ℓ 无信息——不得静默返回均匀先验冒装有
  信息的 quasi-posterior，VI §3.4）。
"""
function quasi_loglik(eps_R::AbstractMatrix{Float64},
                      u_J::AbstractVector{<:Integer},
                      d::Real;
                      cls_floor::Union{Nothing,Real} = nothing)
    nJ = length(u_J)
    nJ >= 2 ||
        error("quasi_loglik: |L| = 0（行集仅 $(nJ) 行，V_{s−1} 不可定义）——ℓ 无信息；不得静默返回均匀先验（D-049/VI §3.4）")
    ll = 0.0
    for i in 2:nJ
        Vsm = V_direct(view(eps_R, 1:i-1, :), view(u_J, 1:i-1), d, u_J[i] - 1)
        F = mp_sqrt_factors(Vsm; cls_floor = cls_floor)
        ε = view(eps_R, i, :)
        quad = norm(F.inv_half * ε) ^ 2      # εᵀV⁺ε（null 分量被零化）
        ll -= 0.5 * (sum(log, F.positive) + quad)
    end
    return ll
end

"""
    default_d_grid() -> (nodes::Vector{Float64}, cells::Vector{Float64})

d 的数值 quadrature 默认网格（**实现期定稿占位**——裁决 C4/T7：节点与
收敛判据留实现阶段按 D-066 流程定稿：synthetic 解析对照、tolerance 减半
收敛、与 Sharpe/PnL 无关；**禁止以回测选择**）。

- `nodes`：`[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]`——旧线 `DGRID_V1`
  （src/predict.jl:10）的节点值复制语义；D-047：DGRID_V1 **不是理论对
  象**，只是数值 quadrature 节点候选。
- `cells`：midpoint cell mass（相邻节点中点为边界、左端 0（开区间端
  点，d=0 不在支撑）、右端 1（闭区间端点极限包含，D-47））——覆盖
  (0,1]、ΣΔ = 1（浮点容差内）。**先验质量不漂移**（SPEC §64）：离散
  权重 q_g ∝ exp(ℓ(d_g))·Δ_g（VI §3.3）。
"""
function default_d_grid()
    nodes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n = length(nodes)
    edges = Vector{Float64}(undef, n + 1)
    edges[1] = 0.0
    for i in 1:n-1
        edges[i+1] = (nodes[i] + nodes[i+1]) / 2
    end
    edges[n+1] = 1.0
    cells = diff(edges)
    return (nodes, cells)
end

# ---------------------------------------------------------------------------
# per-d standardized shape pool（D-052/D-053/D-055；VI §5/§6）
# ---------------------------------------------------------------------------

"""
    DConditionedShapePool

per-d standardized joint shape pool 的冻结输入（VI §5.1/§5.2：z_s(d) =
V_{s−1}(d)^{+1/2}·ε_s 依赖 d，故 pool 是逐 d 的——d-conditioned
accessor 形态，施工图 §2.5）。

- `eps_R`：R 域 mode 残差行（|J|×N_R，J 序——与 InnovationState 共享
  同一 owner，无外部别名）。
- `u_J`：J 行目标日。
- `L_rows`：L 行在 J 空间中的位置（= 2:nJ；单一行集概念，裁决 C2）。
- `cls_floor`：浮点谱分类阈值（仅分类；floor→0 refinement 接口）。

**joint row 规则（D-055/VI §6.2）**：pool 行集 = L ⊆ J——被用于联合
scenario 的历史 residual row 必须覆盖 R_t 全部 risky assets。**一个
scenario innovation 必须来自一个联合合法 shape row**（D-054/VI §6.1：
禁跨日期拼 cell——拼接破坏联合残差向量的横截面依赖，且 mode 坐标下
结构性无定义，VI §1.3 命题）。

**经验测度声明（D-053/VI §5.2）**：z 的 predictive shape 使用历史 OOF
standardized rows 的经验测度——semiparametric empirical predictive，
**不是**声称已知真实 tail distribution；保留 tail/skew/cross-mode
shock shape；不新增 Student-t 等手工参数（参数化候选被 D-053 否决）。
"""
struct DConditionedShapePool
    eps_R::Matrix{Float64}
    u_J::Vector{Int}
    L_rows::Vector{Int}
    cls_floor::Union{Nothing,Float64}
end

"""
    row_at(pool::DConditionedShapePool, d, i) -> Vector{Float64}

第 i 个 L 行（i = 1..|L|）的 standardized shape：`z = V_{s−1}(d)^{+1/2}·ε_s`
（VI §5.1；null 方向分量构造性为零——V^{+1/2} 零化）。
"""
function row_at(pool::DConditionedShapePool, d::Real, i::Integer)
    nL = length(pool.L_rows)
    (1 <= i <= nL) || throw(ArgumentError("row_at: i ∈ 1..$(nL)，实际 $(i)"))
    Jpos = pool.L_rows[i]
    Vsm = V_direct(view(pool.eps_R, 1:Jpos-1, :), view(pool.u_J, 1:Jpos-1),
                   d, pool.u_J[Jpos] - 1)
    F = mp_sqrt_factors(Vsm; cls_floor = pool.cls_floor)
    return F.inv_half * view(pool.eps_R, Jpos, :)
end

"""
    rows(pool::DConditionedShapePool, d) -> Matrix{Float64}    # |L| × N_R

给定 d 的完整 z pool（每行 = 一个联合合法历史行；行集 = L）。
"""
function rows(pool::DConditionedShapePool, d::Real)
    nL = length(pool.L_rows)
    N_R = size(pool.eps_R, 2)
    Z = Matrix{Float64}(undef, nL, N_R)
    for i in 1:nL
        Z[i, :] .= row_at(pool, d, i)
    end
    return Z
end

# ---------------------------------------------------------------------------
# SupportCertificate（D-050/D-051 的决策日诊断证书）
# ---------------------------------------------------------------------------

"""
    SupportCertificate

决策日 V_t 的 PSD support 诊断证书（施工图 §2.5 support_cert 字段）：

- `cls_floor`：浮点分类阈值（仅分类，非理论参数——floor→0 refinement
  接口见 mp_sqrt_factors）；
- `ranks`：per-d-node rank(V_t)（**秩引理的实证记录**——VI §2.3：rank
  与 d 无关，只由 J 行残差张成子空间决定；不存在「小 d 经秩亏获得似然
  优势」的偏置通道）；
- `lambda_max` / `lambda_min_positive`：per-node 特征值极值（rank=0 时
  λ_min_positive = NaN——无正特征，如实标注，不显示 0）；
- `null_dirs`：per-node null 方向数（N_R − rank；未识别 support——
  variance 为 0 是忠实表达，D-050）。
"""
struct SupportCertificate
    cls_floor::Float64
    ranks::Vector{Int}
    lambda_max::Vector{Float64}
    lambda_min_positive::Vector{Float64}
    null_dirs::Vector{Int}
end

# ---------------------------------------------------------------------------
# InnovationState（施工图 §2.5；裁决 A/B2'/G2——R 域重设计）
# ---------------------------------------------------------------------------

"""
    InnovationState

vector innovation 层的 typed 状态（施工图 §2.5；裁决 A/B2'/G2）。全部
对象定义在 **R 域 mode 坐标**（E_{R_t} = [e₀^{(R_t)}; Q^{(R_t)}]，正交，
N_R = |R_t|；裁决 A1/D-045a）。

字段（构造后只读；构造函数是唯一入口，全部内部数组新建/防御性 copy）：
- `R_t`/`N_R`/`E_R`：risk 域资产集（sorted unique）、维数、域 mode 基；
- `t`：决策日（因果边界：J 全部行 u_s ≤ t）；
- `eps_R`：ε_s^{(R)} = E_{R_t}ᵀ·ε̃_s^{(R)}（|J|×N_R，J 序；来源 =
  asset 空间 OOF 残差行的 R_t 子向量经 E_{R_t} 变换——裁决 A3 路径）；
- `row_ids`/`row_masks`：输入行身份（u_s，严格递增）与行 mask（O_s）
  的 owned copy（mask 溯源，D-055）；
- `J_t`：joint 合法行索引（输入行空间；单一行集概念——裁决 C2）；
- `L_t`：ℓ/z 行在 J 空间的位置（= 2:|J|；V_{s−1} 可定义行）；
- `d_nodes`/`d_cells`/`d_weights`：数值 quadrature 节点 / cell mass /
  q(d|H) 离散权重（q_g ∝ exp(ℓ_d)·Δ_g，η≡1）；
- `V_t`：per-node V_t(d)（N_R×N_R，构造性 PSD）；
- `support_cert`：决策日 support 诊断（秩引理实证/null 方向）；
- `z_pool`：per-d shape pool（DConditionedShapePool）。

**已删除对象（D-059/D-047/D-054，VI §10 差异表）**：`v_forecasts`/
`v_bootstrap`（vector law 自含绝对尺度——V_t 自身就是绝对 scale，锚定
通道整体删除）；`own_res_rows`（own-row cell stitching 被禁）；
DGRID_V1 的理论对象身份（只是数值节点候选）。
"""
struct InnovationState
    R_t::Vector{Int}
    N_R::Int
    E_R::Matrix{Float64}
    t::Int
    eps_R::Matrix{Float64}
    row_ids::Vector{Int}
    row_masks::Vector{BitVector}
    J_t::Vector{Int}
    L_t::Vector{Int}
    d_nodes::Vector{Float64}
    d_cells::Vector{Float64}
    d_weights::Vector{Float64}
    V_t::Vector{Matrix{Float64}}
    support_cert::SupportCertificate
    z_pool::DConditionedShapePool
end

"""
    innovation_state(residual_rows, row_ids, row_masks, R_t, E_active;
                     t = nothing, d_nodes = nothing, d_cells = nothing,
                     cls_floor = nothing) -> InnovationState

R 域实例化主入口（施工图 Step 10；裁决 A/A3）。

**输入契约**（接口锚，VI §0/裁决 D1）：
- `residual_rows::n_rows×N_a`：asset 空间 OOF 残差行 ε̃_s（response 层
  Step 8/9 输出契约形态——**含 b₀ 扣除**：ε̃ = E_active·[y − (b₀+Gx)]，
  ŷ 为 fold 完整后验均值。**b₀/OOF 语义由调用方保证，本层不验证、不
  重算**）。非 J 行的 R 坐标允许 NaN（缺失由 mask 表达——VI §1.2 契约
  4/5）；J 行的 R 子向量必须 finite（违反即 error）。
- `row_ids`：行目标日 u_s，**严格递增**（VI §1.2 契约 3：行序 = 目标
  日序；违反 error）。
- `row_masks`：每行 O_s（长度 N_a 的 Bool 向量）。
- `R_t`：决策日 risk 域（free ∪ locked 的资产索引集；判定来源
  src/gate0/market.jl 的 free/locked——裁决 G3）。内部 sort(unique)。
- `E_active`：active 域基（ε̃ 的列坐标基，response 层输出契约）——本层
  仅做维度一致性校验（size = (N_a, N_a)），**不参与 V/z 的数值计算**
  （R 域对象自洽；mode 变换用内部构造的 E_{R_t}）。

**keyword**：
- `t`：决策日（因果边界）。默认 `maximum(row_ids)`——**调用方契约**：
  传入行必须全部 u_s ≤ t（response 层 OOF 输出即决策日因果前缀）。
- `d_nodes`/`d_cells`：数值 quadrature 节点与 cell mass（默认
  `default_d_grid()`——实现期定稿占位，D-066 流程）。校验：节点 ∈
  (0,1]、cells > 0、Σcells ≈ 1、长度相等。
- `cls_floor`：浮点谱分类阈值（默认 mp_sqrt_factors 的自动值；仅分类，
  floor→0 refinement 接口）。

**构造流程（fail-loudly，SPEC §56）**：行集判定 J → propriety 断言
（|J_t| ≥ 1 且 |L| ≥ 1——错误文本含 "innovation coverage failure"，
裁决 C3；|J|=1 时 V 可定义但 ℓ 无信息，同样 error，不得静默返回均匀先
验冒装有信息的 quasi-posterior——VI §3.4）→ R 子向量提取与 E_{R_t}
mode 变换（eps_R = ε̃[J,R]·E_R；**禁止从 active 域 mode 残差取子向量**
——裁决 A3）→ 有限性断言（VI §1.2 契约 5）→ per-node V_t/ℓ_d →
d_weights = softmax(ℓ + log cells)（先验质量不漂移，SPEC §64）→
support 证书 → z_pool 冻结。
"""
function innovation_state(residual_rows::AbstractMatrix{Float64},
                          row_ids::AbstractVector{<:Integer},
                          row_masks::AbstractVector{<:AbstractVector{Bool}},
                          R_t::AbstractVector{<:Integer},
                          E_active::AbstractMatrix{Float64};
                          t::Union{Nothing,Integer} = nothing,
                          d_nodes::Union{Nothing,AbstractVector{<:Real}} = nothing,
                          d_cells::Union{Nothing,AbstractVector{<:Real}} = nothing,
                          cls_floor::Union{Nothing,Real} = nothing)
    # --- 维度与输入校验（fail loudly） ---
    n_rows, N_a = size(residual_rows)
    n_rows >= 1 || error("innovation_state: residual_rows 至少一行（实际 $(size(residual_rows))）")
    length(row_ids) == n_rows ||
        throw(DimensionMismatch("innovation_state: row_ids 长度 $(length(row_ids)) ≠ 行数 $(n_rows)"))
    length(row_masks) == n_rows ||
        throw(DimensionMismatch("innovation_state: row_masks 长度 $(length(row_masks)) ≠ 行数 $(n_rows)"))
    size(E_active) == (N_a, N_a) ||
        throw(DimensionMismatch("innovation_state: E_active 应为 $(N_a)×$(N_a)（active 域基），实际 $(size(E_active))"))
    masks = [begin
                 length(m) == N_a ||
                     throw(DimensionMismatch("innovation_state: 行 $(i) mask 长度 $(length(m)) ≠ 资产数 $(N_a)"))
                 BitVector(m)
             end
             for (i, m) in enumerate(row_masks)]
    ids = collect(Int, row_ids)
    all(i -> ids[i] > ids[i-1], 2:n_rows) ||
        error("innovation_state: row_ids 必须严格递增（VI §1.2 契约 3：行序 = 目标日序）")
    R = sort(unique(Int.(collect(R_t))))
    isempty(R) && error("innovation_state: R_t 非空（N_R ≥ 1）——空 risk 域由调用方按 cash 出路处理（D-056 出路 4）")
    for j in R
        (1 <= j <= N_a) ||
            throw(ArgumentError("innovation_state: R_t 含越界资产索引 $(j)（universe 列数 $(N_a)）"))
    end
    N_R = length(R)
    tdec = t === nothing ? ids[end] : Int(t)
    tdec >= ids[1] ||
        error("innovation_state: 决策日 t=$(tdec) 早于首行目标日 $(ids[1])——J 必为空")

    # --- d 网格校验 ---
    if d_nodes === nothing || d_cells === nothing
        (d_nodes === nothing && d_cells === nothing) ||
            error("innovation_state: d_nodes 与 d_cells 必须同时提供或同时缺省")
        nodes, cells = default_d_grid()
    else
        nodes = Float64.(collect(d_nodes))
        cells = Float64.(collect(d_cells))
    end
    length(nodes) == length(cells) ||
        throw(DimensionMismatch("innovation_state: d_nodes（$(length(nodes))）与 d_cells（$(length(cells))）长度不符"))
    isempty(nodes) && error("innovation_state: d 网格非空")
    for g in eachindex(nodes)
        (isfinite(nodes[g]) && nodes[g] > 0.0 && nodes[g] <= 1.0) ||
            throw(DomainError(nodes[g], "innovation_state: d 节点必须落在 (0,1]（D-47 先验支撑）"))
        (isfinite(cells[g]) && cells[g] > 0.0) ||
            throw(DomainError(cells[g], "innovation_state: d cell mass 必须为正（SPEC §64：先验质量不漂移）"))
    end
    isapprox(sum(cells), 1.0; atol = 1e-9) ||
        error("innovation_state: cell mass 总和 $(sum(cells)) ≠ 1（覆盖 (0,1]，VI §3.3）")

    # --- 行集判定（单一行集，裁决 C2）与 propriety 断言 ---
    J = joint_row_indices(ids, masks, R, tdec)
    nJ = length(J)
    nJ >= 1 ||
        error("innovation coverage failure: |J_t| = 0——无 joint 合法历史行（O_s ⊇ R_t 且 u_s ≤ t），V_t 不可定义（D-050/D-056：fail 优先于任何拼接/补零；free 资产覆盖不足请先经 resolve_risk_domain 收缩 R_t）")
    nJ >= 2 ||
        error("innovation coverage failure: |L| = 0——仅 1 行 joint 合法历史，ℓ_d 无信息（D-049：有效起点 = V_{s−1} 可定义性）；不得静默返回均匀先验冒装有信息的 quasi-posterior（VI §3.4）")

    # --- R 子向量提取 + E_{R_t} mode 变换（裁决 A3 路径） ---
    E_R = risk_domain_mode_basis(R)
    eps_R = Matrix(residual_rows[J, R]) * E_R
    all(isfinite, eps_R) ||
        error("innovation_state: J 行的 R 子向量含非有限值（VI §1.2 契约 5：被消费的 ε 分量必须有限；非 J 行的缺失由 mask 表达，NaN 合法——J 行不合法）")

    # --- per-node V_t / ℓ_d / d_weights ---
    u_J = ids[J]
    V_list = [V_direct(eps_R, u_J, nodes[g], tdec) for g in eachindex(nodes)]
    lls = [quasi_loglik(eps_R, u_J, nodes[g]; cls_floor = cls_floor)
           for g in eachindex(nodes)]
    log_w = lls .+ log.(cells)
    mx = maximum(log_w)
    w_un = exp.(log_w .- mx)
    d_weights = w_un ./ sum(w_un)          # η ≡ 1（D-58：禁 temperature）

    # --- support 证书（秩引理实证 / null 方向）---
    ranks = Int[]; lmax = Float64[]; lminp = Float64[]; nulls = Int[]
    for g in eachindex(nodes)
        F = mp_sqrt_factors(V_list[g]; cls_floor = cls_floor)
        push!(ranks, F.rank)
        push!(lmax, isempty(F.eigenvalues) ? 0.0 : F.eigenvalues[end])
        push!(lminp, isempty(F.positive) ? NaN : F.positive[1])
        push!(nulls, N_R - F.rank)
    end
    cert = SupportCertificate(cls_floor === nothing ?
        mp_sqrt_factors(V_list[1]).cls_floor : Float64(cls_floor),
        ranks, lmax, lminp, nulls)

    # --- z pool 冻结（单一行集 L = J[2:end]）---
    pool = DConditionedShapePool(eps_R, u_J, collect(2:nJ),
                                  cls_floor === nothing ? nothing : Float64(cls_floor))

    return InnovationState(R, N_R, E_R, tdec, eps_R, ids, masks, J,
                           collect(2:nJ), nodes, cells, d_weights,
                           V_list, cert, pool)
end

# ---------------------------------------------------------------------------
# 消费接口：V_at / z / draw / shape 矩
# ---------------------------------------------------------------------------

"""
    V_at(st::InnovationState, d) -> Matrix{Float64}

给定 d 的决策日 V_t(d)：d ∈ d_nodes 时返回预计算矩阵；任意 d ∈ (0,1]
时现算（direct，供极限对照 d→0⁺ 等诊断使用）。
"""
function V_at(st::InnovationState, d::Real)
    i = findfirst(==(Float64(d)), st.d_nodes)
    if i === nothing
        return V_direct(st.eps_R, st.row_ids[st.J_t], Float64(d), st.t)
    end
    return st.V_t[i]
end

"""
    z_row_at(st::InnovationState, d, i) -> Vector{Float64}

第 i 个 L 行的 standardized shape（`z = V_{s−1}(d)^{+1/2}·ε_s`；null
方向分量构造性为零）。
"""
z_row_at(st::InnovationState, d::Real, i::Integer) = row_at(st.z_pool, d, i)

"""
    z_pool_rows(st::InnovationState, d) -> Matrix{Float64}    # |L| × N_R

给定 d 的完整 z pool（联合合法行集 L 上的经验测度——semiparametric
empirical predictive，D-053；**一个 scenario innovation 必须来自一个
联合合法 shape row**，D-054）。
"""
z_pool_rows(st::InnovationState, d::Real) = rows(st.z_pool, d)

"""
    draw_innovation(st::InnovationState, rng) -> NamedTuple

预测创新的抽样（VI §5.1 反标准化；裁决 F1：**d 抽样与行抽样各占独立
随机流位置，顺序固定——先 d 后行**；同 seed 同输入逐位可重放，SPEC
§36）：

1. `d ~ q(d|H_t)`（离散 quasi-posterior 权重，cumsum 搜索——流位置 1）；
2. 行 `s' ~ Uniform(L)`（流位置 2）；
3. `ε_{t+1} = V_t(d)^{1/2}·z_{s'}(d)`（对称 PSD 主根族因子约定）。

返回 `(; eps, d, z_row, z_row_id)`——ε ∈ R^{N_R}（R 域 mode 坐标）；
溯源字段供 concentration 诊断（D-076：d 与行身份）。

**绝对 scale（D-059/VI §5.4）**：ε 的量纲由 V_t^{1/2} 显式恢复——
`scale_d = sqrt(v/v_bootstrap)` 锚定通道已整体删除，本层无 v_bootstrap。
"""
function draw_innovation(st::InnovationState, rng::AbstractRNG)
    n_d = length(st.d_nodes)
    n_d >= 1 || error("draw_innovation: d 网格非空")
    # 流位置 1：d 抽样（quasi-posterior 离散权重）
    u1 = rand(rng)
    cw = cumsum(st.d_weights)
    idx_d = min(searchsortedfirst(cw, u1), n_d)
    d = st.d_nodes[idx_d]
    # 流位置 2：行抽样（L 上均匀）
    nL = length(st.L_t)
    nL >= 1 || error("draw_innovation: |L| = 0（构造已断言非空——状态损坏）")
    u2 = rand(rng)
    idx_row = min(floor(Int, u2 * nL) + 1, nL)
    # 反标准化：ε = V_t(d)^{1/2}·z_{s'}(d)
    z = row_at(st.z_pool, d, idx_row)
    F = mp_sqrt_factors(st.V_t[idx_d]; cls_floor = st.z_pool.cls_floor)
    eps = F.half * z
    return (; eps = eps, d = d, z_row = idx_row,
            z_row_id = st.row_ids[st.J_t[st.L_t[idx_row]]])
end

"""
    shape_moments(st::InnovationState, d) -> Matrix{Float64}    # N_R×N_R

z pool 的标准化二阶矩诊断 `M_z(d) = (1/|L|)·Σ z zᵀ`（VI §9 检查 7 /
裁决 G6：**诊断量，非断言 = I**——VI §5.4：V_{s−1} 随 s 漂移，M_z ≈ P_s
而非恒等；协议不假设 E[zzᵀ] = I，绝对量纲由 V_t^{1/2} 携带）。
"""
function shape_moments(st::InnovationState, d::Real)
    Z = z_pool_rows(st, d)
    nL = size(Z, 1)
    nL >= 1 || error("shape_moments: |L| = 0")
    return (Z' * Z) ./ nL
end

# ---------------------------------------------------------------------------
# 覆盖不足 T4 判定次序（D-056 / 裁决 C3；VI §6.3）
# ---------------------------------------------------------------------------

"""
    resolve_risk_domain(row_ids, row_masks, free_assets, locked_assets, t)
        -> (R_t::Vector{Int}, dropped_free::Vector{Int})

覆盖不足的 T4 判定次序（裁决 C3，VI §6.3 判定次序规范）：

给定初始 `R_t⁽⁰⁾ = free ∪ locked`：

- **(a) 覆盖不足由 free 资产引起**（|J(R)| < 2）：逐个剔除 free 资产
 （不进 free risky set），R 收缩，行集 J/pool 恢复。剔除次序：每次剔
  除「因果覆盖行数最少」的 free 资产（tie 按资产索引升序——确定性），
  直到 |J| ≥ 2 或 free 剔尽。被剔除资产的资金可以留在 cash（D-056 出
  路 4——这正是 cash 入 Kelly feasible set 的原因之一，D-068/D-070）。
- **(b) 剔除后仍不足**（覆盖不足由 locked 资产引起，或 free 剔尽仍
  不足）→ **error**（文本含 "innovation coverage failure"——裁决
  C3）。回测驱动器（gate0/backtest.jl，后续 Step）捕获该错误、记录诊
  断、**当日维持持仓**（不重排权重、不拼残差、不 zero-fill——裁决
  C3/T4 终态）。
- **free 剔尽且 locked 为空**：返回**空 R_t**（调用方按「无可生成风险
  的资产——资金全留 cash」处理；`innovation_state` 对空 R error）。

判定「足」= |J| ≥ 2（V 可定义且 ℓ 有信息的合并判据——VI §3.4 两断
言；|J| ≥ 1 但 = 1 时 ℓ 无信息，同属不足）。free/locked 语义由
src/gate0/market.jl 的 free/locked 判定提供（裁决 G3：Step 1/2 前置）。
"""
function resolve_risk_domain(row_ids::AbstractVector{<:Integer},
                             row_masks::AbstractVector{<:AbstractVector{Bool}},
                             free_assets::AbstractVector{<:Integer},
                             locked_assets::AbstractVector{<:Integer},
                             t::Integer)
    N_a = length(row_masks[1])
    for v in (free_assets, locked_assets), j in v
        (1 <= j <= N_a) ||
            throw(ArgumentError("resolve_risk_domain: 资产索引 $(j) 越界（universe 列数 $(N_a)）"))
    end
    isempty(intersect(free_assets, locked_assets)) ||
        error("resolve_risk_domain: free 与 locked 必须互斥（D-017：locked = held ∧ ¬free）")
    cur_free = sort(unique(Int.(collect(free_assets))))
    locked = sort(unique(Int.(collect(locked_assets))))
    dropped = Int[]
    # 空 R 的行集语义：O_s ⊇ ∅ 恒真 → J = 全部因果行（D-056 出路 4 的
    # 极端形态：free 剔尽且无 locked——资金全留 cash，无 innovation 对象，
    # innovation_state 对空 R error，由调用方处理）。
    njoint(R::Vector{Int}) = isempty(R) ? count(<=(t), row_ids) :
        length(joint_row_indices(row_ids, row_masks, R, t))
    R = sort(union(cur_free, locked))
    while !isempty(cur_free) && njoint(R) < 2
        # 剔除因果覆盖行数最少的 free 资产（tie → 索引序，确定性）
        counts = [count(i -> (row_ids[i] <= t && row_masks[i][j]), 1:length(row_ids))
                  for j in cur_free]
        cmin, jpos = findmin(counts)
        jstar = cur_free[jpos]
        push!(dropped, jstar)
        deleteat!(cur_free, jpos)
        R = sort(union(cur_free, locked))
    end
    if njoint(R) < 2
        error("innovation coverage failure: 剔除 free（$(dropped)）后行集仍不足——覆盖缺口由 locked 资产（$(locked)）引起，或 free 剔尽仍不足（D-056/裁决 C3：fail loudly，回测驱动器当日维持持仓，不拼残差、不 zero-fill）")
    end
    return (R, sort!(dropped))
end
