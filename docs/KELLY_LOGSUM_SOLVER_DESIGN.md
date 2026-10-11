# cash Kelly 求解核心重写设计：log-sum simplex 专用内点法（LOGSUM-SIMPLEX-IPM-1）

**状态：设计文档 + 原型探针（本批两交付物）。未运行任何命令；未修改 `src/`、`test/` 与任何既有文件；生产接线属后续受控施工。**
**适用对象：`src/gate0/kelly.jl` 的 `_cash_kelly_solve`（L617-698）主求解段。**
**纪律：同一 log-Kelly 目标、同一可行集、同一 `cash_kelly_certificate` 验收（L102-126）；tol 不放松；fail-loud 保留；无 RNG；不引入由回测选择的参数（SPEC §95 / 开发守则 §24）。**

---

## 0. 结论摘要

1. 选定算法：**原始对数障碍内点法（primal barrier IPM）**，每次迭代在 (m+1)×(m+1) 的 KKT 系统上工作，主成本为一遍 O(S·m) 数据扫描。对 S=131072、m=2 预期几十次迭代、亚秒级、RSS 增量近零。
2. 取代范围：仅 `_cash_kelly_solve` 内部的 Convex.jl 建模 + Clarabel 求解；函数签名与返回契约 `(; w_val, status_ok, reason, status)` 不变。
3. 保留：输入校验、原空间证书（唯一验收口径）、KELLY-SCALE-RETRY、设计 A 候选/取优、M2 出口、Newton polish、TIE-FAST-1、B2B-SPLIT 拆分、tie-break 主框架、fail-loud。
4. Clarabel 保留为 reference backend 与回退（同目标、同证书）。
5. tie-break Stage 2/3：m=2（当前实际链 free=1）给出精确一维专用化（W_ε 区间端点二分，O(S·二分) 成本）；一般 m 保持现状，扩展登记。
6. 探针 `dev/kelly_ipm_probe.jl`：自含、确定性，打印迭代数 / 证书三量 / 目标值 / 每迭代与总耗时，含解析对照与 131072 规模合成验证。

---

## 1. 问题、规模与现状

### 1.1 数学对象

`cash_kelly`（L1021-1028）的数学问题（L8-24；docstring L948-1020，目标式 L964）：

    max f(w) = (1/S) Σ_s log(a_s·w + b_s)
    s.t.  w ≥ 0,  1'w = budget

其中 A = [X | 1] ∈ R^{S×m} 为增广场景矩阵（cash 为显式末列），w ∈ R^m，b ∈ R^S 为 locked scenario wealth（D-017/D-068）。S ∈ {65536, 131072}；当前生产链 free=1，故 m=2。

### 1.2 证书（唯一验收口径）

`cash_kelly_certificate`（L102-126）在**原始 log 目标**上定义三量，`cash_kelly_certified`（L129-130）要求全部 ≤ tol：

- `feasibility` = max(|Σw_aug − budget|, max(0, −min(w_aug)))；
- `kkt_residual` = max_i |w_i·(ν − g_i)|，其中 g = ∇f（增广列，含 cash 列 (1/S)Σ 1/wealth_s），ν = max(g)；
- `objective_gap` = max(0, budget·ν − g'w)（凹性上界，SPEC §40）。

**本设计的一切返回/采纳均以该证书为准，solver 内部状态不参与验收。**

### 1.3 现状成本与阻塞（任务背景，已核实）

Convex.jl `sum(log.(A*w))` 建模 + Clarabel 求解：131072 场景单次 8–25s、RSS>1.4GiB；高成本日主求解 >50s 超有效窗口（t=339 `_cash_kelly_prelude_solve` 三次 48/48/50s 全切）；tie-break 两阶段在 131072 层 24.1s 且常返回 nothing（t=338）。

结构原因（静态）：Convex/Clarabel 把每个场景抬升为 exp-cone 约束与辅助变量，KKT 系统规模 ~O(S) 维，稀疏分解成本随 S 增长；而问题对 w 的实际自由度只有 m+1。**把工作从 O(S) 维锥问题降回 m+1 维 KKT，是数量级收益的来源。**

---

## 2. 算法选择

### 2.1 选定：原始对数障碍 IPM

对 μ > 0，解障碍子问题

    max φ_μ(w) = (1/S) Σ_s log(a_s·w + b_s) + μ Σ_i log(w_i)
    s.t.  1'w = budget

障碍项把 w ≥ 0 编码为严格内点（w > 0），等式约束用拉格朗日乘子 ν 处理。沿 μ ↓ 0 的中心路径跟踪；每一步用阻尼牛顿；终止由 1.2 节原空间证书判定。

### 2.2 选定理由

1. **成本结构与问题匹配。** 梯度 g = (1/S)A'(1/z)、Hessian H = (1/S)A'diag(1/z²)A 均为一遍 O(S·m)（m=2 时 3 个标量累加）；m+1 维求解 O(m³) 可忽略。通用锥求解器的 O(S) 维线性代数被彻底去掉。
2. **与证书天然对齐。** 障碍解上互补性 ≈ μ，μ 收到 tol 以下即三量可控；终止判据直接复用原空间证书（同一公式、同一 A/b/budget），"返回 ⟺ 证书绿"可在构造上保证。
3. **对退化/平局鲁棒。** 内点法不需要支撑集枚举/切换（主动集的组合管理风险），D-091 连续平局、D-090 对称射线、Case A/B 角点都不需要特判——严格内点路径自动穿越。
4. **确定性、无新依赖。** 全部为确定性浮点运算与手写小线性代数；不含 RNG、不含 BLAS 归约顺序依赖（主循环手写单线程循环），不受 BLAS 线程配置影响。

### 2.3 备选与拒绝理由

| 备选 | 结论 | 理由 |
|---|---|---|
| 乘性更新（Cover/EM: w_i ← w_i·g_i/Σw_j g_j） | 拒绝为默认；登记为可选初始化 | 线性收敛，到达证书 1e-8/1e-6 量级需极多轮；KKT 证书对齐依赖收敛质量 |
| 投影梯度 / 加速投影 | 拒绝 | 同样收敛率不足，边界投影与证书互补性不一致 |
| 主动集牛顿（单独） | 拒绝为默认；登记为后续可选 | 支撑切换策略（进出支撑）需组合管理；现有 `_cash_kelly_newton_polish`（L151-235）已覆盖"固定支撑精修到机器精度"，主求解只需好起点 |
| Quasi-Newton / 对角 Hessian 近似 | 拒绝为默认 | 不改变目标与证书（迭代数可能增多），但 m 极小、精确 Hessian 成本已可忽略；无收益动机。**若未来启用须论证不改变终止判据的证书语义且默认关闭** |
| 对偶坐标上升（S 维对偶） | 拒绝 | 对偶维数 S，与"m 小"结构相反 |
| 截断场景 / 采样 / 近似目标 | 拒绝 | 改变数学对象（SPEC 禁令）；不属数值层 |
| 保留 Convex+Clarabel 为唯一路径 | 拒绝 | 成本即本任务要解决的阻塞；但保留为 reference/回退 backend |

---

## 3. 算法规格

### 3.1 一阶/二阶量（一遍扫描）

给定 w（严格正、1'w = budget）：

    z_s = a_s·w + b_s            （s = 1..S；必须有限正）
    g   = (1/S) Σ_s a_s / z_s
    H   = (1/S) Σ_s a_s a_s' / z_s²      （对称 PSD）
    φ 的障碍对角项：μ / w_i²

### 3.2 牛顿系统（Schur 形式）

对 (Δw, Δν) 的线性化（L = φ_μ − ν(1'w − budget)）：

    [ H + μ W⁻²    1 ] [Δw]   [ r_dual ]
    [ 1'           0 ] [Δν] = [ −r_prim ]

其中 W = diag(w)，r_dual = g + μ./w − ν·1，r_prim = 1'w − budget。

用 Schur 补实现（M = H + μW⁻² 正定）：

    x1 = M⁻¹ r_dual,  x2 = M⁻¹ 1
    Δν = (1'x1 + r_prim) / (1'x2)
    Δw = x1 − Δν·x2

两个线性解用**手写 Cholesky**（m×m，确定性、无 LAPACK 依赖）；主元 ≤ 0 或非有限即 fail 为 `:singular_kkt`。

### 3.3 步长、Armijo、预算清理

- 分数步长：α_max = min(1, min_{i:Δw_i<0} −w_i/Δw_i)，α = τ·α_max（τ=0.99；无负分量时 α=1）。严格保持 w > 0。
- Armijo 回溯：以 φ_μ 为判据，`φ_try ≥ φ_cur − 1e-10(1+|φ_cur|)`；最多 12 次减半；仍不接受 → fail 为 `:line_search_failed`。
- 预算清理（primal residual cleanup，O(m)）：`w[m] += budget − Σw`；若 cash 变负则整体按比例缩放。不改变数学对象（证书复核可行性；量级为剩余残差，远小于步长）。

### 3.4 中心化路径（μ 序列）

- μ0 = 1.0（默认；数值配置，待 D-066 式 refinement 定稿）。
- 每轮最多 max_newton=4 步牛顿；轮末 μ ← max(σμ, μ_floor)，σ=0.2。
- μ_floor = max(0.01·tol, 1e-12)。
- max_outer=60。迭代上界 240 步。

理由：σ=0.2 是障碍法常规保守收缩；σ 越小每轮需更多牛顿步、越大收敛越慢。μ_floor 对齐 tol 两个数量级裕量：障碍解互补性 ≈ μ，tol=1e-8 要求 μ ≲ 1e-9。

### 3.5 终止判据与证书对齐（钉死项）

每一步（扫描后、更新前）用**当步梯度 g 与目标值**构造证书（`cert_from_g`：feasibility/kkt/gap/objective 与 L102-126 逐式一致；objective = Σlog z / S）。三量 ≤ tol → 立即返回 `converged=true`。**不存在"内部收敛"或"近似达标"的第二口径。**

### 3.6 数值防护

- wealth 非有限/非正 → 立即 fail（`:nonfinite_wealth`），不返回 heuristic。
- Cholesky 失败 / Schur 分母 ≤ 0 / 方向非有限 → fail（`:singular_kkt` / `:nonfinite_direction`）。
- 线搜索失败 → fail（`:line_search_failed`）。
- 预算/迭代耗尽 → fail（`:budget_exhausted`）。
- 初始点强制正性与预算归一（floor = 1e-14·max(budget,1)）。

全部失败路径只返回诊断 + 未认证的当前 w；**装配层不得采纳未认证解**（见 §4.4）。

### 3.7 初始点、确定性

- 默认 w0 = (budget/m)·1（均匀、严格内、预算精确）。
- 可选传入 w0（生产不默认引入跨日 warm start——保持逐日无状态、可重放；跨日加速属增量层议题，另议）。
- 全流程无 RNG；不依赖线程数、不依赖系统时钟（时钟仅用于探针计时打印）。

### 3.8 预条件 / 行缩放复用

新核心**不自行缩放**：直接消费调用方决定的 `(Xt, bt, cash_col)`（`_cash_kelly_prelude_solve` L909-934；含条件缩放与 KELLY-SCALE-RETRY 的缩放装配）。行缩放是数学恒等（KELLY_NUMERICAL §2.1），IPM 在两种装配上解同一问题；证书一律原空间。

### 3.9 复杂度与预期成本

- 每迭代：时间 O(S·m + m³)，空间 O(m²)（不物化 z 向量，逐行计算）；主循环单线程手写。
- 每迭代数据流量：S·m·8 字节读入一次（131072×2 → 2MB）+ 少量状态。
- 迭代数：μ 从 1.0 到 ~1e-10（tol=1e-8）约 15 轮 × 2–3 牛顿步 ≈ 30–45 步。
- 预期（待探针验证，不预填实测）：S=131072 单次亚秒级；RSS 增量近零。

---

## 4. 与现有语义的交互图

### 4.1 保留（零语义变化）

| 构件 | 位置 | 说明 |
|---|---|---|
| `cash_kelly_inputs` / 证书 / `cash_kelly_certified` | L71-130 | 唯一验收口径 |
| KELLY-SCALE-RETRY-1 + 条件缩放 | L238-298、L756-776、L886-906、L920-929 | 缩放装配与重试触发不动 |
| 设计 A 候选/取优 | L290-298、L732-800 | 新核心只是候选来源替换；比较键、tie 序、fail 文本不动 |
| M2 status 出口 | L670-686、L716-729 | Clarabel backend 保留；IPM 失败语义映射见 §4.3 |
| Newton polish | L151-235 | 对 IPM 解同样适用（固定支撑精修）；作为链上既有环节保留 |
| B2B-SPLIT-1/2/3 拆分 | L700-863、L865-946 | `_cash_kelly_solve` 仍是单一原子段；段边界不变 |
| TIE-FAST-1 / tie-break 框架 | L279-282、L814-833、L836-854、L1063-1167 | 语义与采纳条件不动 |
| fail-loud | L725-727、L804-812、L879-880、L1188-1189 | 全部错误路径与文本族保留 |

### 4.2 取代

`_cash_kelly_solve` 内部的 `Variable`/`hcat`/`solve!`（L652-669）与 status 白名单逻辑（L672-686）**原为默认主求解**；本设计以 IPM 取代之，接线后 Clarabel 变为 reference/回退。

### 4.3 `_cash_kelly_solve` 双 backend 装配（接口边界）

```julia
# 新内部核心（本设计交付为探针原型；生产接线后位于 kelly.jl）
_cash_kelly_ipm(Xt, bt, cash_col, budget, tol)
    -> (; w_val, converged::Bool, cert, iters::Int, reason::Symbol)

# 装配层：签名不变，新增可选 backend（默认 :auto）
_cash_kelly_solve(Xt::Matrix{Float64}, bt::Vector{Float64},
                  cash_col::Vector{Float64}, budget::Float64, tol::Float64;
                  backend::Symbol = :auto)
    -> (; w_val, status_ok::Bool, reason::Symbol, status)
```

装配规则（钉死）：

- `backend=:auto`（默认）：先 `_cash_kelly_ipm`；若 `converged` 且证书绿（原空间口径在调用层再验一次，保持候选链统一）→ `(w_val, true, :ok, :ipm)`；否则走现状 Clarabel 路径（逐字保留，含 status 白名单 / M2 出口 / clip / 归一）。
- `backend=:ipm`：只跑 IPM 并返回其状态（对拍/诊断用）。
- `backend=:clarabel`：只跑现状路径（回归对拍基准）。
- IPM 返回的 `w_val` 已由内部证书验证；装配层仍以 `status_ok=true` 表示"可进入候选链"，最终采纳权在 `_cash_kelly_finish_main` 的原空间证书。
- 可选诊断 kw `solver_diag`（同 `KellyScaleDiagnostics` 模式，默认 nothing 零行为变化）记录 backend、IPM 迭代数、μ_final；**不改返回契约、不改证书字段**。
- `status` 字段对 IPM 路径取符号 `:ipm`（Clarabel 路径仍为其原生 status 对象）；现有消费者只在 `!status_ok` 时读 `status`（错误文本），白名单路径不受影响。

### 4.4 fail-loud 与 Clarabel 回退条件

- IPM 在任何内部防护失败（§3.6）或预算耗尽时：`:auto` 装配回退 Clarabel（数学等价路径，非新决策权）；Clarabel 也失败 → 现有 error 文本族原样。
- 回退不是"放宽验收"：回退后仍需原空间证书，采纳规则不变。
- 若未来 IPM 被证明为唯一可靠路径，再单独审议是否收紧回退；**本设计不改变任何现有 fail 条件**。

---

## 5. tie-break Stage 2/3 专用化

### 5.1 现有语义（引用，不动）

`_cash_kelly_finish_tiebreak`（L836-854）→ `_cash_kelly_canonical_tiebreak`（L1063-1167）：

1. U* = 主解证书 objective；
2. Stage 2：max w_c s.t. w_aug ≥ 0、Σw = budget、U(w) ≥ U* − tie_eps；
3. Stage 3：min ‖w_risky‖₂² 同约束 + w_c = wc_max；
4. canonical 点过**原空间证书**才采纳，否则返回 nothing、保留主解（现状）。

### 5.2 m=2 精确一维专用化（本设计新增）

m=2 时增广权重参数化 w = (u, budget − u)，u ∈ [0, budget]（u 为 risky 分量），h(u) = U(w)：

1. h 的每次求值 O(S)（同一扫描）。
2. W_ε 是区间 [u_lo, u_hi]：u_lo = 最小 u 使 h(u) ≥ U* − tie_eps（h(0) 已满足则 u_lo = 0；否则在 [0, u*] 确定性二分，80 次迭代——成本 ~80×O(S)，131072 场景下毫秒级）。
3. Stage 2（max cash ⟺ min u）与 Stage 3（min ‖w_risky‖² = min u²，同向）在 m=2 下同点：w_tb = (u_lo, budget − u_lo)。
4. 输出仍过同一原空间证书；不过则返回 nothing、保留主解（与现状采纳语义逐点同口径）。

**行为等价性论证（关键）**：现状 Stage 2/3 求的就是同一数学集合 W_ε 上的同一 canonical 点（严格凹时 h 唯一极小 cash-max 在左端点；平局时端点为 u=0）；专用化只改变"如何求这个候选点"的成本（Clarabel 两次大解 → 确定性二分），不改变候选点的数学身份与采纳门槛。若数值上现状解与二分点存在 O(证书容差) 差异，双方都须过同一证书才被采纳，行为在证书口径下一致。

**诚实边界（静态分析）**：对严格凹、唯一最优的输入，W_ε 的 cash-max 点有 gap ≈ (1−u)·√(2·ε·|h''|) 量级；ε=1e-8、|h''|=O(0.03) 时约 4e-6，通常 **超过** tol=1e-8 证书门槛 → canonical 点被拒、保留主解。这既解释现行行为（tie-break 在严格凹案例多返回 nothing/保留主解），也说明专用化不改变采纳结果，只把"总在跑的 24s 候选计算"降为毫秒级；真正会被采纳的是平局/近平局（gap≈0）情形——与 P0-9 的规范目的吻合。

### 5.3 一般 m（>2）的边界

Stage 2/3 保持现状 Clarabel（语义已锁定；活跃集低维化 §9.12 已因本链无零列而否定）。若未来 free 资产数增长，候选扩展为「IPM + U 约束障碍通道」（把 U ≥ U*−ε 作为附加障碍、同一证书验收），属后续独立设计，不在本批。

### 5.4 TIE-FAST-1

全 cash 快速路径（L281-282；skip_tb 构造 L830、消费 L844）不变；m=2 专用化同样以它为前提短路。

---

## 6. 探针交付说明（`dev/kelly_ipm_probe.jl`）

自含脚本（仅 `LinearAlgebra`/`Printf`），不加载 Convex/Clarabel 与生产模块：

- 实现 §3 全部规格；证书公式与 L102-126 同式（内联对照版）。
- 打印：迭代数、converged/reason、证书三量、objective、每迭代 mean/max、总耗时；`w` 与解析参考（若有）的最大偏差。
- 小 fixture（解析对照）：Case C（内部解 5/6,1/6）、Case D（locked，5/12,17/60）、Case A（全 cash 角点）、D-090(a)（对称三资产 1/3 各 + cash 0）、m=1（唯一可行点）。
- tie-break m=2 演示：严格凹（Case C）与退化平局（risky 列恒 1）两种候选构造与证书判定。
- 大规模合成（确定性 splitmix64，无 RNG）：S=65536/131072、m=2、温和 spread=0.04 与宽尾 spread=40；宽尾附「行缩放前/后」对照（生产 cs 定义）。
- 可选 `ARGS[1]`：文本输入文件（`S m budget [tol]` 头 + S 行 m 列 + 可选 S 行 base），供 DevOps 用真实捕获输入复跑。

---

## 7. 未决与待验证 / revisit 条件

1. **数值验证待 DevOps**：探针在 131072 真实/合成输入上的迭代数、耗时、证书三量与宽尾稳健性；本设计未运行，全部成本为静态推断。
2. **数值配置定稿**：μ0=1.0、σ=0.2、μ_floor_factor=0.01、τ=0.99、max_outer/max_newton——属数值配置，按 D-066 式 refinement（合成解析 → 单日 fixture → 减半稳定）定稿，禁止由回测选择。
3. **m=2 tie-break 专用化**：落地前需与现状 Clarabel 两阶段在受控输入上对拍候选点与采纳结果（登记为后续施工项）。
4. **生产接线**：`_cash_kelly_ipm` 入 `kelly.jl`、`:auto` 装配切换、测试锚（kelly_cash 全量回归 + B2B-SPLIT/设计 A/负控）——独立受控施工，不在本批。
5. **revisit 条件**：探针显示某输入类无法在证书 tol 下收敛（且 Clarabel 能）时，重估回退策略；free 资产数显著 >1 时重估 O(S·m²) 与 tie-break 扩展；若未来证书定义扩展（字段/口径变化），重对齐 §3.5。

---

## 8. 引用

- `src/gate0/kelly.jl`：L102-126（证书）、L151-235（polish）、L617-698（`_cash_kelly_solve`）、L700-863（finish/main/tiebreak）、L865-946（prelude/split）、L1021-1028（入口）、L1063-1167（canonical tiebreak）。
- `test/gate0/kelly_cash_tests.jl`：Case A–D、P0-9、行缩放、K-S-R-1（B2B-SPLIT-1 26/26、TIE-FAST-1 13/13 为任务背景核实事实）。
- `docs/KELLY_NUMERICAL_ROW_SCALING.md` §2.1/§8/§9.10（缩放恒等与拆分链）。
- `docs/KELLY_ADAPTIVE_CERTIFICATE_DRIVEN.md` §10（M2 出口框架，与本设计正交）。
- 调用点：`src/gate0/driver.jl:516`、`src/gate0/quadrature.jl:749`（签名与三元组返回不变）。

*（本文件为设计 + 原型交付；未运行任何命令、未修改 `src/`、`test/` 或任何既有文件。）*

---

## 9. 实现备注（2026-10-11；生产接线后追加）

本节记录生产接线（`src/gate0/kelly.jl`）与本设计的差异、最终形态与受控
验证结果；行号引用已按接线后版本刷新，其余正文保持设计时点原样。

### 9.1 与设计的差异：返回点绑定深度

- 设计（§3.4/§3.5）：μ_floor 因子 0.01，返回条件 = 证书三量 ≤ tol（证书过即返）。
- 实现：μ_floor 因子 **1e-3**，且返回条件 = **证书过 AND μ ≤ μ_floor**。
- 理由：角点类解的主解残余小权重 ≈ μ/(ν−g)；原「证书过即返」时 μ≈tol
  量级，残余将达 ~1e-7，破坏下游 tie-break 的既有 1e-8 差值契约
  （TIE-FAST-1：`maximum(abs.(wA .- wA0)) ≤ 1e-8`）。1e-3 因子给约 10 倍
  裕量（tol=1e-8 时残余估 ≲1e-9）。
- 代价：迭代数略增（μ 需多收缩数轮）；秒级结论不变（见 §9.4）。证书仍是
  唯一验收口径——更深的返回点只会更优，不放松任何门槛。
- 探针头已登记「生产 vs 探针」的这两处已知差异（探针保持验证时点的
  0.01/证书过即返；其算法一致性由人复核）。

### 9.2 双 backend 最终形态

- `_cash_kelly_solve(..., tol; backend = :auto)`：
  - `:auto`（默认）：先跑 `_cash_kelly_ipm`；认证（内部证书 ≤ tol）→ 直接
    返回；未认证/运行性异常 → 回退既有 Convex+Clarabel 路径（原实现逐字
    保留，含 status 白名单 / M2 出口 / clip / 归一）。
  - `:ipm` / `:clarabel`：只跑单路径（诊断 / 回归对拍用）。
- 返回契约 `(; w_val, status_ok, reason, status)` 不变（:ipm 路径 `status`
  取符号 `:ipm`；失败 `reason ∈ :ipm_failed / :ipm_error`）。`backend` 为
  只增可选 kw，现有调用面零改动。
- 验收链不变：最终采纳仍由 `_cash_kelly_finish_main` 的原空间
  `cash_kelly_certificate` 决定。

### 9.3 探针的冻结独立验证器角色

- `dev/kelly_ipm_probe.jl` 自 2026-10-11 起为**冻结的独立验证器**（头部
  声明）：不随生产演进、无需同步维护；其算法已由生产 `_cash_kelly_ipm`
  取代；生产 vs 探针的两处已知差异（μ_floor 因子 1e-3 vs 0.01、返回点绑定
  μ≤floor vs 证书过即返）登记于探针头。

### 9.4 受控验证结果摘要（接线批次，引述）

- 测试：`kelly_cash` **23 testset / 222 断言全绿**（含新增 IPM/装配/
  tie-break/行缩放 testset；B2B-SPLIT-1 与 TIE-FAST-1 契约保持）。
  **后继口径（2026-10-11 同步；AGENTS.md 第 18 任段）**：登记曾为 24/227（经核实为
  23/227）；F5 测试锚追加后终态 **25 testset / 238 断言**全绿（`docs/GATE0_EXIT_CHECKLIST.md`
  L194/L195）。
- 真实 t=338 输入（131072 层、m=2）：主求解 0.159s（iters=65）、finish
  1.285s、tie-break 0.178s；对拍历史 mid2 maxdiff 2.42e-10。
- tie-break 严格凹语义不变（候选被原空间证书拒 → 保留主解）；平局采纳
  全 cash canonical 点。

### 9.5 未决（对 §7 的更新）

- §7-2 的数值配置定稿（D-066 式 refinement）仍待定；本节的 μ_floor 因子
  与返回点深度属该流程的待定稿项。
- §7-3 的一般 m（>2）tie-break 扩展仍为后续项（m=2 专用化已落地并被测试
  覆盖）。
- §9.4 为单机、当次运行口径；跨配置重放以确定性设计为前提，但未逐配置
  复核。

### 9.6 与 §4.3 声明的实现差异（登记；2026-10-11）

- §4.3 声明的 `_cash_kelly_ipm` 返回字段 `cert` 与 `_cash_kelly_solve` 的可选诊断 kw
  `solver_diag`，生产接线**未实现**（当前无消费者：采纳链只读取 `w_val / status_ok /
  reason / status`；诊断需求由既有 `KellyScaleDiagnostics` 承担）。如未来需要，另立
  受控施工；本登记不改变任何当前行为。
