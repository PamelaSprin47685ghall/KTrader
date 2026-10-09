# KTrader 数值积分收敛规范（Numerical Integration Spec）

**文档状态：SPEC 级定义草案（Normative Draft）——不是实现描述，不是性能承诺。**
**适用对象：`generate_scenarios_v1` / `adaptive_scenario_weights` / `kelly_*` 决策链。**
**权威关系：本文档细化 `AGENTS.md` §37（AGENTS.md:1176-1191）与 §65（AGENTS.md:2133-2141）的执行语义。与 `AGENTS.md` 冲突时以 `AGENTS.md`（Common Law / SPEC）为准。本文档不扩大任何角色职权，不授权任何命令执行，不修改源码、测试与既有文档。**

---

## 0. 目标对象与两类误差

决策目标（AGENTS.md:1201-1203）：

- I(w) = E[log(base + Rᵀw) | H]，连续（无限样本）目标；
- I_M(w) = (1/M) Σₛ log(base_s + (X_M w)_s)，给定 rule 下 M 个 scenario 的有限采样近似；
- w_M = argmax_{w∈W_t} I_M(w)，W_t = { w ≥ 0, Σw = budget }（预算与 locked base 语义见 AGENTS.md:1208-1218）。

目标：

- 积分收敛：沿嵌套 rule 细化 M 时 I_M(w) → I(w)（对每个固定 w）；
- 决策收敛：w_M → w* = argmax I(w)。

必须区分两类独立误差：

1. **求解误差**：给定 X_M，solver 相对 w_M 的误差。由 `kelly_certificate` 界定（src/kelly.jl:21-37），是「凸问题解好了没有」的证书。
2. **积分误差**：X_M 的采样律相对真实 posterior predictive 的离散化误差。由本规范 §3 的 (a)(b) 界定，是「M 够不够大」的证书。

现有 Kelly 证书只覆盖第 1 类。把它当成积分收敛证明，是把样本内的最优性当成样本外的逼近，属于层级错误。

---

## 1. 当前实现映射

### 1.1 有限采样近似在哪里发生

`generate_scenarios_v1`（src/predict.jl:546-625）按 §35 的六要素（AGENTS.md:1124-1131）生成 X_M ∈ R^{M×N_universe}：

- posterior draws 与 residual bootstrap 在 src/predict.jl:564-572 与 576-624；
- 目标函数求值发生在 kelly 层：wealth = X*w .+ b（src/kelly.jl:26），g = Xᵀ(1./wealth)/S（src/kelly.jl:30），objective = sum(log, wealth)/S（src/kelly.jl:102、110）；
- I_M(w) 就是 `sum(log, X*w .+ base)/S`（src/kelly.jl:30、102）。

### 1.2 S（scenario 数）的出现位置与默认

| 位置 | 默认 | 语义 |
|---|---|---|
| src/predict.jl:546 | `S=500` | `generate_scenarios_v1` 的 IID 默认 scenario 数 |
| src/kelly.jl:173 | `S=300` | `path_kelly_v1` 默认；`adaptive=false` 时直接使用 |
| src/backtest.jl:117 | `S=300` | `backtest_v1` 默认；`adaptive=false` 时逐日使用 |
| src/backtest.jl:249-251 | `S`（调用方） | 正式回测逐日 `generate_scenarios_v1(model; S, rng=MersenneTwister(seed+t))` |
| README.md:36、44 | `300` | 文档示例与 `SCENARIOS=300` 环境变量 |
| dev/m1_artifact_replay.jl:626-627 等 | `300` | M1 历史对照基准，非生产路径 |

### 1.3 adaptive 机制真实存在

不是占位，是已接线的实现：

- `ScenarioQuadrature`（src/predict.jl:507-521）：构造时取一组素数基与**一份共享随机 shift**（`rand(rng,dim)`，src/predict.jl:520）；嵌套性来自 `quadrature_uniform`（src/predict.jl:522-531）按**全局索引 s** 生成点：任意 M' > M 的序列前 M 个点与 M 序列逐位相同，不重新洗牌。
- quadrature 模式下 `generate_scenarios_v1` **不消耗调用方 rng**（src/predict.jl:552-563：`uniforms=nothing`，用 `quantile(Normal(), quadrature_uniform(...))` 逐点确定性变换；契约由 test/posterior_contract_tests.jl:34-45、241+ 覆盖）。
- `adaptive_scenario_weights`（src/predict.jl:627-653）：默认 `min_scenarios=64, max_scenarios=512`（src/predict.jl:628），从 S=64 起按 `S=min(2S,max_scenarios)` 倍翻（src/predict.jl:635），每轮用**同一个 rule 对象**重生成样本（src/predict.jl:636），因此样本是嵌套前缀。
- 收敛判据（src/predict.jl:647）：`‖w_new − w_prev‖₁ ≤ weight_tol(=1e-3)` 且 `kelly_certificate(X_new, w_prev).objective_gap ≤ tol(=1e-5)`。
- 预算耗尽：`error`（src/predict.jl:652）——fail-loud 已是现状。
- 测试证据：test/numerical_tests.jl:160-182（`small ≈ large[1:64,:] atol=1e-14` 证前缀嵌套；零响应 world 收敛到 S=128；`max_scenarios=128, tol=0, weight_tol=0` 时确定性抛错）；test/backtest_target_tests.jl:25-37（adaptive 接线、seed 保持、结果计数）；test/posterior_contract_tests.jl（quadrature 下 lazy/dense 一致）。

### 1.4 正式回测实际使用的参数

`backtest_v1` 默认 `adaptive=false`（src/backtest.jl:117-121）。非 adaptive 路径：worker 每日生成 `X = generate_scenarios_v1(model; S=300, rng=MersenneTwister(seed+t))`（src/backtest.jl:249-251），consumer 调 `scenario_weights`（src/backtest.jl:79-82、291-292）。**正式回测的默认不是嵌套 quadrature，而是固定 S=300 的 IID 采样**。`adaptive=true` 时经 `_backtest_target` 调 `adaptive_scenario_weights(...; rng=MersenneTwister(seed+t), tol=quadrature_tol, max_scenarios)`（src/backtest.jl:72-78）。默认参数下 §37 的细化判据根本不执行。

### 1.5 现有 Kelly 证书证明什么、不证明什么

`kelly_certificate`（src/kelly.jl:21-37）在给定矩阵 X_M 上验证：

- feasibility（simplex 越界，src/kelly.jl:25）；
- kkt_residual（互补性 `w .* (dual .- g)`，src/kelly.jl:32）；
- objective_gap = budget·max(g) − gᵀw ≥ F_M(w*) − F_M(w)（由 concavity，src/kelly.jl:17-19、33）。

它证明的是：**在固定的这一批 X_M 上，w 离 argmax I_M 有多远**。它不证明 I_M → I，也不证明 w_M → w*。若 X_M 有采样偏差，证书全绿的同时最优决策仍可能有系统误差。`fast_kelly_solver` 以 tol=1e-8 内部自证（src/kelly.jl:125-126），失败则 `clarabel_kelly_solver` 解同一目标并再次自证（src/kelly.jl:129、55-56）——两者都是**同一 X_M 上的求解误差**，样本层面的收敛不在其职责内。

---

## 2. 目标

### 2.1 收敛目标

沿同一嵌套 rule 细化：M → 2M → 4M → …

- I_M(w) → I(w)；
- 决策收敛：w_M → w*。

### 2.2 嵌套确定性 rule 的硬性要求

1. 一次 refinement 全程同一 rule 身份：primes 与 shifts 固定（src/predict.jl:512-521 的共享 shift）；2M 序列的前 M 个点必须逐位等于 M 序列的点；禁止重新洗牌、重抽 shift、或换 rule 后再比较。
2. quadrature 模式下不消耗调用方 rng（现状满足；见 §1.3）。
3. 调度无关：同一决策日、同一 seed 下，worker/线程调度不得改变 rule 或点序（AGENTS.md:1151-1172）。
4. rule 替换（如 Sobol/Owen，AGENTS.md:1191）是数值 backend 变更，不改变 posterior law；替换后必须重新走 §3 证书。

---

## 3. 收敛证书

对一次 refinement 的相邻两级 M、2M：

- (a) **权重稳定性**：‖w_{2M} − w_M‖₁ ≤ ε_w。
- (b) **目标间隙**：I_{2M}(w_{2M}) − I_{2M}(w_M) ≤ ε_U，用更细的同一 rule 求值。
- (c) **可选 KKT**：在 X_{2M} 上复核 kelly_certificate(X_{2M}, w_{2M}) 的全部指标 ≤ solver tol。

### 3.1 各自在哪一层计算、谁是 owner

| 判据 | 计算层 | owner | 理由 |
|---|---|---|---|
| (a) | scenario refinement 循环（持有 w_M 与 w_{2M}） | `predict.jl`（`adaptive_scenario_weights`） | kelly 层看不到跨 M 的另一半 |
| (b) | scenario generator 层求值（样本在 generator 层，目标语义在 kelly 层） | `predict.jl` 计算差；目标求值语义归 `kelly.jl` | 必须在同一更细样本上比较两个权重 |
| (c) | 给定 X 的 KKT/最优性 | `kelly.jl`（`kelly_certificate`） | 已有唯一实现，禁止复制 |

(a)(b) 是积分收敛证据；(c) 是求解误差证据。二者必须分别成立、分别记录，不得互相顶替。任何实现都不得把「certificate 绿了」说成「积分收敛了」。

### 3.2 通过条件

一次 adaptive 返回必须满足：fine 解 w_{2M} 自身有有效 solver 证书（现状：fast 内部 tol=1e-8 或 Clarabel 自证，src/kelly.jl:61、126、55-56），且 (a) ≤ ε_w、且 (b) ≤ ε_U。若启用 (c)，在 fine 样本上复核。

### 3.3 为什么 (b) 不能由 (a) 替代

仅有 (a)：w_M 与 w_{2M} 可能同步漂移（同一采样偏差下近似相等），‖·‖₁ 小但都远离 w*；仅有 (c)：I_M 可以因 M 不足而整体偏离 I，每步都「解得好」但解的是错的积分。所以需要 (b) 在更细同一 rule 上直接比较目标值。以 Sharpe/回测收益作判据违反 AGENTS.md:1183-1189 与 2133-2141。

### 3.4 现状与证书定义的差距

- 现状（src/predict.jl:646-647）：`certificate = kelly_certificate(X_new, w_prev)` 是 **w_M 在 X_{2M} 上的次优间隙**（(b) 的一种上界变体），不是直接的 I_{2M}(w_{2M}) − I_{2M}(w_M)；返回的是 w_{2M}（`weights`），证书却属于 w_M——语义不对称。
- `tol=1e-5`（integration 层）与 solver 内部 `tol=1e-8`（src/kelly.jl:61）是两个不同层次的阈值，必须分开命名、分开记录。
- 非 adaptive 默认路径（src/backtest.jl:117、249-251）完全不做跨 M 检查。

---

## 4. Fail loudly

- 预算内不收敛必须抛错，错误文本必须包含 `"Numerical integration did not converge"`（现状 src/predict.jl:652 为 `"posterior quadrature did not converge by $max_scenarios scenarios"`，语义一致、措辞待统一到规范文本）。
- 严禁「到 512 就算了」：不得在耗尽 `max_scenarios` 时返回 `converged=false` 的默认权重、静默回退 IID、或把最后一次权重当成结果。
- 严禁：靠调大 `max_scenarios` 让某次运行通过而不记录证书；靠删除 (b) 让循环提前退出；靠重跑直到碰巧通过。
- 与 AGENTS.md:2012（§56 fail-loud）一致。

---

## 5. S=300 降级

- `S=300` 只允许作为：数值初始规模、测试/诊断规模、历史对照（如 dev/m1_artifact_replay.jl:626-627 的 M1 基准）、或 adaptive 细化的中间起点。
- 生产决策（backtest/live 最终权重）不得使用未经 §3 证书的固定 S。当前「`S=300` + `adaptive=false`」仍是默认（src/kelly.jl:173、src/backtest.jl:117），属于与本文档目标的已知差距（§8），其修复必须走独立评审与实现流程，不得在本草案阶段声称已达成。
- 一旦 adaptive 成为生产默认，生产签名里的 `S=300` 必须被重新审查或删除。

---

## 6. 与 SPEC §37 的关系

- §37（AGENTS.md:1176-1191）：细化判据是 ‖w_{2S}−w_S‖₁ 与 Kelly objective certificate，不是 Sharpe。本文档把该句细化为：(a) 即 L1 判据；(b)(c) 把「Kelly objective certificate」拆成积分目标间隙与求解 KKT 两个层次。
- §65（AGENTS.md:2133-2141）：‖w_{2S}−w_S‖₁ → 0；不得用回测收益选 S。与本文档一致。
- 张力（必须标注）：
  1. §37 表述「Kelly objective certificate」是单数，现实现中它同时被当成 (b) 的代理与 (c) 的阈值，而 solver 内部另有 1e-8——层次混淆。
  2. §37 同时支持 IID 与 quadrature，但默认回测走 IID 且固定 S（src/backtest.jl:117）；生产若声明依赖 §37 的收敛保证，默认路径必须显式走证书路径。
  3. §37 允许未来替换 Sobol/Owen（数值 backend，不改变 posterior law）——本文档要求替换后重新走 §3 证书，并保留 rule 身份用于重放。

---

## 7. 未决决定（判据均不依赖 Sharpe）

| # | 未决项 | 决定所需证据 | 判据 |
|---|---|---|---|
| 1 | `max_scenarios` 默认（现状 512） | 不同 N 与响应强度下 M→2M→4M 的 (a)(b) 序列；撞顶比例 | 证书通过所需 M 的分布；撞顶即暴露给重新审查，不静默放过 |
| 2 | ε_w 数值（现状 weight_tol=1e-3） | 同上；量纲（full N_universe 空间 L1） | 缩小 ε_w 时 w 变化符合收敛趋势 |
| 3 | ε_U 数值与定义（绝对/相对） | (b) 的直接实现与实验 | ε_U 减半时返回 M 单调不减/证书仍可复现 |
| 4 | 嵌套下 RNG 确定性细节 | rule shift 抽取与返回对象的身份记录 | 同 seed、同 rule 逐位可重放；跨日期 `MersenneTwister(seed+t)` 与调度无关 |
| 5 | adaptive 返回对象是否携带 rule 身份与证书数值 | 审计、重放与复现要求 | 返回 primes+shifts+M+证书，或明确记录可重放来源 |
| 6 | adaptive 在 backtest 中的序列化/并行成本 | src/backtest.jl:105-111 记录的现状约束 | 只序列化数学所需的最小对象；不得改变任何数学 |

全部未决项的裁决必须回答：证书在预算内是否通过。禁止以「回测更好/更快」裁决数值参数。

---

## 8. 现状差距清单

| # | 项目 | 现状 | 规范目标 | 引用 |
|---|---|---|---|---|
| 1 | 嵌套 rule | 已实现（Halton + 共享 shift） | 保持，受证书约束 | src/predict.jl:507-531、627-653 |
| 2 | 生产默认路径 | `adaptive=false`, S=300 IID | 生产须走证书路径，或显式声明降级 | src/backtest.jl:117、249-251 |
| 3 | (a) 权重 L1 | 已实现 | ε_w 经实验确定 | src/predict.jl:647 |
| 4 | (b) 目标间隙 | 代理：w_prev 在 X_new 上的 KKT gap ≤ 1e-5 | 显式 I_{2M}(w_{2M}) − I_{2M}(w_M) ≤ ε_U | src/predict.jl:646-647 |
| 5 | (c) KKT | fast/Clarabel 内部 1e-8 | 可选 fine 样本复核 | src/kelly.jl:61、126 |
| 6 | 错误文本 | `posterior quadrature did not converge by ...` | 含 `"Numerical integration did not converge"` | src/predict.jl:652 |
| 7 | rule 身份/重放 | 返回对象不含 primes/shifts | 记录或可重放 | src/predict.jl:643-648 |
| 8 | 高维 rule 有效性 | dim = 2N+3；N 大时未证收敛速率 | 收敛实验覆盖真实 N | src/predict.jl:630 |

---

## 9. 一句话

证书必须回答两个不同的问题——「这批 scenario 上的凸问题解好了吗」(c) 与「这批 scenario 够代表真实积分了吗」(a)(b)；前者已有唯一 owner（kelly.jl），后者必须由 scenario refinement 层在预算内证明，否则 fail loudly，而不是「到 512 就算了」。
