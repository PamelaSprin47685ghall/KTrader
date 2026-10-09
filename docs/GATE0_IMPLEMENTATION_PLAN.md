# KTrader Gate-0 纠偏施工图与类型设计（GATE0_IMPLEMENTATION_PLAN）

**文档状态：Gate-0 纠偏施工图（纯设计文档）。本文是裁决书 §55 第一阶段产物——「新数据 mask 类型设计 + static implementation plan + mode-space response 维度图」——的 owner。**
**修订记录：2026-10-09 Manager 终审修订版（裁决 A/B2'/B/G4/C2/C3/G1~G8/H1~H9；权威文本 docs/GATE0_MANAGER_ADJUDICATIONS.md）。原开放决策 9 项全部获裁决（10.1）；InnovationState 按 R 域重设计（2.5）；propriety 先验修正 D-035a 已裁决（2.4）。**
**性质：不写代码、不运行命令、不修改任何源码/测试/既有文档。所有行号来自对当前工作树源码的只读核对。**
**优先级：AGENTS.md 裁决书（D-001~D-097 与 §40/§41）+ docs/GATE0_MANAGER_ADJUDICATIONS.md（终审裁决）> 旧 SPEC 不冲突条款 > 本文件。本文件与裁决冲突时以裁决为准；本文只做落实与展开，不做新裁决。**
**证据基准：src/ 十三文件结构核对 + docs/ 八份文档（MODEL_LEDGER / OLD_TO_CURRENT_SEMANTIC_DIFF / TRACE_NEUTRALITY_DERIVATION / POSTERIOR_DEFINITION / INNOVATION_LAW / NUMERICAL_INTEGRATION_SPEC / GATE0_RESPONSE_POSTERIOR / GATE0_VECTOR_INNOVATION；后两份为 Gate-0 姊妹推导交付物）。凡引用既有文档处只给章节与 `file:line`，不复制其内容。**

---

## 0. 阅读约定

- 「裁决书」= AGENTS.md 前半的 Gate-0 裁决书（D-001~D-097、§40 施工顺序、§41 Exit 清单）。
- 「终审裁决」= docs/GATE0_MANAGER_ADJUDICATIONS.md（2026-10-09 Manager 终审，裁决号 A/B2'/B/G4/C2/C3/G1~G8/H1~H9；本文所有「已裁决」标注的权威出处）。
- 「旧 SPEC」= AGENTS.md 其后的 SPEC / ROADMAP / 开发守则；与裁决书冲突的条款已被 D-001 压制。
- 「旧线」= 当前 `src/` 的 2.0 RC 实现（trace 条件化、EB plug-in、scalar innovation、fixed-S）。
- 「新线」= 本文规划的 Gate-0 reference 纠偏实现。
- 编号澄清：裁决书内 **D-046** 是「fractional kernel 继续统一」的 innovation 裁决；旧资产保留清单出自裁决书 **§46**（章节号，非裁决号）。**D-035a** 是终审裁决新增的 α 先验尾部修正（p(α) ∝ 1/[α(1+α)]，见 2.4 节）；**D-045a** 是 V_t 的 R 域实例化（见 2.5 节）。

---

## 1. 总路线决策：独立 reference 模块线

### 1.1 两条路线对比

| 维度 | 路线甲：改旧文件（在 src/ 上渐进纠偏） | 路线乙：新建独立 gate0 模块线（旧 src/ 保留为 historical fixture） |
|---|---|---|
| 与 D-082 的关系 | 违背「先写新的 slow reference，不在当前 fast core 上继续补丁」的字面与精神 | 直接落实 D-082 |
| 数学对象边界 | 旧 `PreparedProblem`/`V1Model`/`ResponseOperator` 的字段与旧数学绑定（14N asset 坐标 design、trace 条件化、EB 点估计、scalar innovation），渐进改会制造「半新半旧」混合对象 | 新类型从裁决书的数学对象直接生长，字段即数学 |
| 旧资产保全（裁决书 §46） | 旧 2.0 source snapshot 被逐次改写，历史 fixture 语义丢失 | 旧线整体冻结，D-002/D-004 的历史证据链零扰动 |
| 测试隔离 | 现有 23 项 REQUIRED 契约测试守护旧数学；改旧文件会让它们与新数学互相打架 | 旧测试继续守护旧线 fixture；新线有自己的 constitutional suite（Step 16） |
| 性能纪律（D-083） | 旧线的 workspace/incremental/scheduler 复杂度诱导「顺手优化」 | 新线单线程、无 incremental、无 scheduler |
| 成本 | 短期 diff 较小 | 新线代码量较大；几何原语按 H7 复制 |

### 1.2 建议

**路线乙已确认（裁决 H3，2026-10-09）。** 理由（原建议理由，终审采纳）：

1. D-082 的裁决语言是「先写新的 slow reference」，不是「把旧的改成新的」；路线甲把重写级纠偏降级为补丁级。
2. 裁决书 §46 明确旧 2.0 source snapshot 是必须**保留**的历史资产；路线乙是唯一能同时满足「保留旧物」与「长出新数学」的结构。
3. 旧线类型字段（`src/predict.jl:16-36` 的 `V1Model`、`src/prepare.jl:83-125` 的 `PreparedProblem`）与被废止对象（trace 条件化、plug-in EB、`v_bootstrap`、own-row stitching）深度耦合；在同一名下换血，等于让每个后续读者都要考古「这个字段现在是哪个数学」。
4. 开发守则 §20 单一真源不受破坏：确定性几何原语的单一真源由「来源标注 + 静态审查」承担（H7 复制方案，见 1.3）。

### 1.3 新模块结构（已裁决，裁决 H1/H2/H7）

```text
src/gate0/                     # 新 reference 线（单线程、无 incremental、无 scheduler、无 GPU）
    market.jl                  # MarketFacts + Eligibility + free/locked（D-012~D-017）
    mode_problem.jl            # 统一 mode 坐标 producer + 新 ModeProblem（D-021~D-030, D-080）
    response.jl                # 统一 full-block response + ResponsePosterior（D-025~D-039, D-085/D-086）
    oof.jl                     # full-mode OOF folds（D-043/D-044）
    innovation.jl              # vector V_t(d) / standardized shape / d quasi-posterior（D-040~D-059）
    predictive.jl              # PredictiveLaw 组装 + 方差分解（D-060）
    quadrature.jl              # nested Sobol + 独立 audit replicate（D-061~D-067）
    kelly.jl                   # cash feasible set Kelly（D-068~D-071）
    backtest.jl                # reference sequential runner + 三 benchmark（D-072~D-076）
test/gate0/                    # 新 constitutional suite（Step 16；独立测试入口，裁决 H1）
```

- **模块形态（裁决 H1）**：独立顶层 `module KTraderGate0` + `src/gate0/` 目录线；旧 `src/` 冻结；**主 module 与新线互不 include**；新线测试独立入口（`test/gate0/`）。文件切分按上表九文件（裁决 H2，按原建议）。
- **几何原语边界（裁决 H7）**：`ruler`（src/geometry.jl:26）、`path_basis_1d`（src/response.jl:2-16）、`relative_gauge`、`center_of_mass`（src/geometry.jl:20）、TAUS/BANDS/WARMUP（src/geometry.jl:11-14）是确定性数据变换（D-024：ruler/gauge 非 posterior 对象），以**纯函数复制**方式进入 `src/gate0/`（源码内标注来源 `file:line`）、**不跨 module 依赖**旧线。理由：H1 的互不 include 前提下，跨 module 依赖（`using KTrader`）会重新耦合两线；复制的是无 posterior 语义的确定性变换，单一真源由「来源标注 + Step 3 静态审查逐函数确认纯函数性」承担。带旧数学耦合者（`fit_response_operator`、`condition_trace_neutrality`、`optimize_conditioned_eb` 等）一律不进新线。§2.3/Step 3 等处的「复用」字样一律按本条读作「纯函数复制 + 来源标注」。
- **数据桥（H1 互不 include 的直接推论）**：`MarketFacts` 的构造函数接受裸矩阵参数（dates/symbols/close/adj/observed），**不引用旧 `Bars` 类型**；与旧 `Bars` 的字段对应转换在调用方完成。旧 `Bars` 零改动。
- 旧线处置：`src/` 其余文件不动、不优化、不删除（裁决书 §46 + D-084，见第 6 节）。

---

## 2. 新核心类型设计（字段级）

设计原则（Kolmogorov 之书「用类型系统排除非法状态」）：每个 mask 一个字段、每个数学概念一个 owner、「用观测 mask 表达政策」这类非法组合在类型层面不可表达。以下均为**设计规格**：字段名与语义是规范，内部表示细节留给施工。

### 2.1 MarketFacts（owner：`src/gate0/market.jl`；裁决 D-010/D-012）

```julia
struct MarketFacts
    dates::Vector{Date}
    symbols::Vector{String}
    close::Matrix{Float64}      # marking series（NaN before first bar, carried forward）——沿用 Bars 语义
    adj::Matrix{Float64}        # marking series——同上
    observed::BitMatrix         # O_{t,i}：D-012 唯一语义「日期 t 资产 i 有真实市场 bar」
end
```

| 字段 | 类型 | 语义 | 与旧字段映射/来源 |
|---|---|---|---|
| `dates` | `Vector{Date}` | 交易日历 | `Bars.dates`（src/data.jl:8）直传 |
| `symbols` | `Vector{String}` | universe 符号 | `Bars.symbols` 直传 |
| `close`/`adj` | `Matrix{Float64}` | 账户 marking（carry-forward），**不进入理论 signal** | `Bars.close`/`Bars.adj` 直传；marking 消费点见 src/backtest.jl:296 |
| `observed` | `BitMatrix` | **O_{t,i}**，物理观测事实。改名即钉死语义：`bar` 之名连同「was tradable」注释语义一并废除 | `Bars.bar`（src/data.jl:12）字段对应转换（裸矩阵桥，§1.3）；4 参构造 `bar=isfinite.(rawclose).&isfinite.(rawadj)`（src/data.jl:28）的推导逻辑沿用 |

**不变量（可证伪）**：`observed` 构造后只读；「因不想交易/历史不足/模型不 admission/风控而改 observed」的写入在类型与测试上都不存在（D-012 四条绝对禁止）。理论 signal 恢复沿用旧语义：`signal = ifelse.(observed, adj, NaN)`（等价 src/data.jl:37 的 `signal_prices`，新线以 `MarketFacts` 方法重新拥有）。

**语义修正点**：src/data.jl:2-5 docstring「bar[t,j] is true iff a real bar printed that day — i.e. the asset was tradable」把观测与可交易混写为同义——这是 D-012/D-014/D-015 拆分对象的历史根源；新类型用字段名与 docstring 分离三者。

### 2.2 Eligibility（owner：`src/gate0/market.jl`；裁决 D-013/D-014/D-015/D-016/D-017/D-018）

```julia
struct Eligibility
    model_admitted::BitMatrix   # A^model_{t,i}
    trade_eligible::BitMatrix   # E^trade_{t,i}
    executable::BitMatrix       # T^exec_{t,i}
end
```

| 字段 | 判定规则（规范） | 与旧实现的关系 |
|---|---|---|
| `model_admitted[t,j]` | =1 当且仅当 prefix 1:t 中资产 j **至少存在一条有效 daily return**（相邻两日观测对）。弱证据由 posterior 表达，不用 MINROWS 类门槛（D-013） | 旧 `active_universe_indices`（src/predict.jl:125-135）已是此语义；新线从「fit 时派生计算」升格为独立逐 (t,j) mask；全 NaN dummy 构造性排除 |
| `trade_eligible[t,j]` | 是否允许当日**新建/增加风险仓位**。默认全 true；用户 252 有效交易日规则在此表达：`trade_eligible[t,j] = (Σ_{τ≤t} observed[τ,j]) ≥ 252`（D-018）。计数按 O 累计（观测 bar 数，非 return 数）。**口径已裁决（H4）：累计第 252 个有效 bar 当日解除（当日 E^trade=1）**；Step 1 的 tiny analytic 测试钉死此边界（裁决书原文「直到累计 252 个有效 bar」；docs/MODEL_LEDGER.md §11 的 off-by-one 分析属旧 active 的 return-pair 语义，新口径按 O 计数无歧义） | **全新对象**：当前代码无独立 E^trade（docs/MODEL_LEDGER.md §4：承担者是 backtest free 掩码，且与观测共用 bar 字段） |
| `executable[t,j]` | 物理可执行性。理论 backtest 简单形式至少 `T^exec = O`（D-015）；live/broker 层由报价决定（src/broker.jl:136 的 `tradable`：bid/ask/last>0 是独立 T^exec 语义） | 旧线无独立对象（bar 被借用）；与 broker.tradable 各自拥有、不互相校验（沿 docs/MODEL_LEDGER.md §5 分层） |

**派生量（函数拥有，非字段）**：

- `free(t) = trade_eligible[t,:] .& executable[t,:]`（D-016）。**model_admitted 不得混入 free**——直接纠正 src/backtest.jl:242-245（`free[j]=b.bar[t,j]` 仅对 active 循环，即 free = active ∩ bar）把 admission 混进可交易集合的现状。
- **held/locked 单独处理（D-017）**：`w^held_j > 0 ∧ free_j=0` ⇒ **locked risk**，其 scenario wealth contribution 保留在 Kelly 的 base_s（沿用 `locked_wealth` 的 0*NaN 防护语义，src/kelly.jl:137-148），绝不当 cash。locked 判定由 Kelly 层拥有（`current .* .!free`，语义同 src/kelly.jl:155）；Eligibility 只提供 free。**risk 域 R_t = free ∪ locked** 是 innovation 层与 scenario 合成的域定义（2.5/2.6 节，裁决 A/B2'）。

**构造纪律**：三个 mask 的构造函数只依赖 `MarketFacts.observed` 与用户政策参数；任何把 posterior/模型输出写回 Eligibility 的路径都是类型错误（benchmark 外生性的根基，见第 8 节）。

### 2.3 ModeProblem（新 PreparedProblem；owner：`src/gate0/mode_problem.jl`；裁决 D-021~D-024/D-030/D-080/D-081）

```julia
struct ModeProblem
    # identity / universe
    admitted::Vector{Int}           # A^model 准入资产（model_admitted 前缀派生）
    N_universe::Int
    T::Int                          # prefix 行数
    N::Int                          # admitted 数（= mode 空间资产维）
    # geometry & scaling（D-024：确定性历史泛函，非 posterior 对象；实现按 H7 复制）
    s1::Vector{Float64}             # 每资产 1 日 ruler（ruler 语义）
    s_m::Vector{Float64}            # per-TAU macro ruler（src/predict.jl:276 fast_s_m 语义）
    s_perp::Vector{Float64}         # per-band gauge-invariant scalar relative ruler（D-024；SPEC §15）
    gauge::Matrix{Float64}          # 固定 Helmert Q ∈ R^{N×(N-1)}（D-023；relative_gauge 语义，H7 复制）
    # mask / observation 溯源（D-080）
    observed::BitMatrix             # return 级观测 mask（isfinite.(r)，语义同 src/predict.jl:68）
    alive_now::Vector{Bool}         # 决策日观测支撑（语义同 src/prepare.jl:131，owned copy）
    # 统一 mode 坐标（核心新对象，维度见第 3 节）
    X_mode::AbstractMatrix{Float64} # n_res × (14N+1)：[DC=1 | B_m | B_⊥]
    Y_mode::AbstractMatrix{Float64} # n_res × N：[m | Qᵀe]
    x_now::Vector{Float64}          # 决策行 feature（长度 14N+1）
    # fold 统计（统一 mode 空间）
    folds::FoldStatistics           # ranges + per-fold/full Grams（语义同 src/prepare.jl:50-58，维度换新）
    # 训练布局
    ts_total::Vector{Int}
    n_res::Int
    F_folds::Int
end
```

| 关键字段 | 语义 | 与旧 PreparedProblem（src/prepare.jl:83-125）的映射 |
|---|---|---|
| `X_mode` | 统一设计矩阵，**含 DC 列**，relative 部分在 gauge 坐标 | 替代旧 `X_rel_stacked`（n×14N，asset 坐标）与旧 `B_m`（n×14，独立回归）——两条 design 合一 |
| `Y_mode` | 统一目标 [m; Qᵀe] ∈ R^N | 替代旧 `macro_stats`（m 标量目标）与 `Y_target_rel`（N 维 embedded field 目标，src/predict.jl:298） |
| `folds` | 统一 mode 空间 fold Grams | 旧 `FoldStatistics`/`MacroStatistics`（src/prepare.jl:50-70）合并为一套（macro 不再独立成回归，D-039）；充分统计恒等（fold train = full − eval fold，逐元素）沿用 SPEC §28 |
| `observed`/`alive_now` | mask 溯源 | 沿用旧字段语义（src/prepare.jl:94-99），来源钉死为 MarketFacts.observed 派生 |
| 删除字段 | — | `adj_act`（caller-view 面板，solve 不读）、`relative_embedding`（被 Y_mode 取代）、`X_rel`（cumulative asset 坐标，被 gauge 累积 X_q 取代）、`ws_owner/ws_generation`（新线无 workspace，D-082/D-083） |

**不变量（可证伪，Step 3 测试）**：asset↔mode 往返 reconstruct（`m·e0 + Q·q` 恒等还原 embedded field）；资产置换协变；gauge 旋转（Q→QR）不变；zero embedding 只在 field 代数（D-022，语义沿 src/predict.jl:38-42 注释）；dummy all-NaN 资产严格不影响其余资产的 ModeProblem；线性交换性（QᵀX_rel = X_q：先投影再累积 = 先累积再投影——B_⊥ 构造语义，GATE0_RESPONSE_POSTERIOR.md §1.4）。

### 2.4 ResponsePosterior（owner：`src/gate0/response.jl`；裁决 D-031~D-039/D-085/D-086 + 终审 G7/D-035a）

统一 mode-space response 模型（推导已交付：**docs/GATE0_RESPONSE_POSTERIOR.md（裁决 H5 定名）**——Step 7 纸面推导，含 §4 解析积分路线、§5 propriety 完整静态推导、§10 实现者检查清单 14 条；实现 Step 8）：

$$
y_{t+1} = b_0 + G\,x_t + \epsilon_{t+1},\qquad
G\in\mathbb R^{N\times(14N+1)},\ b_0\in\mathbb R^N
$$

（$b_0$ 与 $G$ 的 DC 列合写即 x 首列为 1；第 3 节维度图统一按 14N+1 列含 DC。）

```julia
struct ResponsePosterior
    alpha_nodes::Vector{Tuple{Float64,Float64}}  # (log α_0, log α_p) 自适应 quadrature 节点（D-038）
    alpha_weights::Vector{Float64}               # 节点权重（含 cell mass / tail 证书）
    node_evidence::Vector{Float64}               # 每节点 evidence 值 p(Y|α⁽ᵏ⁾)（裁决 G7；log p(Y|α) 按推导 §4.3 闭式）
    conditional::Vector{ConditionalFit}          # 每 node：给定 (α_0,α_p) 时 B,Σ_R 的解析积分表示（D-037；matrix-t：位置 B̂、行尺度 S(α)、列精度 V⁻¹、尾部幂 (n+P)/2）
    propriety::ProprietyCertificate              # D-036 posterior propriety gate（不满足 → fail loudly）
    predictive::DecisionPredictive               # 决策时 E[G x_now]、Cov(G x_now)（超参混合后的边缘，D-034）；μ 的 t 边缘自由度 ν=n+1−N 精确绑定（裁决 G7）
    evidence::Float64                            # 总 log marginal evidence（诊断用）
end
```

| 字段 | 语义 | 规范来源 |
|---|---|---|
| `alpha_nodes/weights` | $(\log\alpha_0,\log\alpha_p)\in\mathbb R^2$ 的 deterministic adaptive quadrature；不得 profile argmax 后当已知、不得固定 `EB_ALPHA_MIN/MAX` 为 prior support、grid 节点数不得成为策略参数 | D-038 |
| `node_evidence` | 每节点 evidence 值 $p(Y\mid\alpha^{(k)})$（裁决 G7）；log 形式闭式见推导 §4.3（$|S(\alpha)|^{-n/2}$ 项族），供审计/重放 | 裁决 G7 |
| `conditional[i]` | 第 i node 的条件后验**解析**表示：$B,\Sigma_R$ 给定 $(\alpha_0,\alpha_p)$ 的共轭/reference-prior 联合积分结果——$B\mid Y,\alpha$ 为 **matrix-t**（位置 $\hat B$、行尺度 $S(\alpha)=S_{yy}-S_{xy}^{\top}(S_{xx}+\Lambda_\alpha)^{-1}S_{xy}$、列精度 $V^{-1}$、尾部幂 $(n+P)/2$）；$\Sigma_R\mid Y,\alpha\sim\mathcal{IW}(S(\alpha),n)$。**高危陷阱（推导 §4.2 实现警告）：$S(\alpha)$ 是 $Y^{\top}C^{-1}Y$，不是 ridge 残差平方和**——Step 8 有双构造一致性断言 | D-037/D-085/裁决 G5 |
| `propriety` | posterior proper 判定证书；不满足时**不得 clamp、不得改 EB、不得套 inverse-Wishart 超参**，必须 fail loudly。**D-035a 修正已裁决（裁决 B/G4）：p(α) ∝ 1/[α(1+α)]，纯数学 propriety 依据（推导 §5.3-§5.4：1/α 在无界支撑上右尾发散于零模型平台 × log-uniform 测度），非回测选择；修正先验下右尾收敛，Step 8 propriety gate 应绿，若红是实现 bug**（A2 数据层条件 n≥N 且 S(α)≻0 仍是 fail loudly 负测试） | D-036 + 裁决 B/G4 |
| `predictive` | 决策时线性形式 $Gx_{now}$ 的边缘矩：$\mu_{t+1}$ 的 mean/cov，含超参混合。**t 分布自由度 ν=n+1−N 精确绑定（裁决 G7：显式拒绝 n−N 版本——那是「含新观测噪声」的 predictive，被 D-040/D-041/D-060 的模块边界拒绝；推导 §4.4）**。D-060 方差分解中 **epistemic (response)** 部分的唯一来源 | D-034/D-040/D-060 + 裁决 G7 |
| 先验规格（类型外规范常量） | **D-035a 修正已裁决（裁决 B/G4）**：$p(\alpha_0)\propto1/[\alpha_0(1+\alpha_0)]$、$p(\alpha_p)\propto1/[\alpha_p(1+\alpha_p)]$（小 α 处保持 reference 局部行为、截断无穷远零模型平台；纯数学 propriety 依据，非回测选择）；$p(\Sigma_R)\propto|\Sigma_R|^{-(N+1)/2}$（Jeffreys）；$\Lambda_\alpha=\mathrm{diag}(\alpha_0,\alpha_p,\ldots,\alpha_p)$ 两 group precision（禁 per-band/per-asset/ARD/BF） | D-035 + D-035a（裁决 B/G4）/D-033/D-032 |

**与旧对象的映射**：取代 `ResponseOperator`（src/response.jl:92-104）+ `pred_moments`（V1Model 字段）+ macro EB 链（`optimize_matrix_normal_eb`、`sig2` 点估计——D-039 取消）。旧 `optimize_conditioned_eb`（src/response.jl:934-1121）降级为 legacy/hypothesis（D-084，第 6 节），只可提供 mode/initial bracket/diagnostic，**不得定义 posterior**（D-038）。

**输出契约（裁决 A2/A4，response 层的两个规范输出）**：

1. **asset 空间 OOF 残差行 ε̃_s（含 b₀ 扣除）**：fold posterior 对 held-out 行的预测含 DC 列——ε̃_s = E_active·(y_{s+1} − B̂⁽ᶠ⁾x̃_s)（active 域 mode→asset 重构后的残差行；E_active = [e₀; Q] ∈ R^{N_a×N}，正交）。innovation 层按 R_t 消费其子向量（2.5 节）；行身份契约沿 ResidualOracle 语义（src/residual_oracle.jl:17-22：行 idx ↔ 全局日 ts_total[idx]，目标 r[t+1,:]）。
2. **μ_asset = E_active·μ**：决策时 mode 空间 μ 经 E_active（正交，重构无损）映射为 asset 空间条件均值——response 层的输出边界（推导 §7：输出到 μ 为止，未来冲击归 innovation 模块）。

### 2.5 InnovationState（owner：`src/gate0/innovation.jl`；裁决 D-040~D-059 + 终审 A/B2'——R 域重设计）

**域定义（裁决 A/B2'，D-045a）**：决策日 t 的 innovation 服务于 Kelly 的 **risk 域** R_t = free ∪ locked（D-016/D-017），N_R = |R_t|。域 mode 基 E_{R_t} = [e₀^{(R_t)}; Q^{(R_t)}] ∈ R^{N_R×N_R}（正交）。历史行 s 的 R 域 mode 残差 ε_s^{(R)} = E_{R_t}ᵀ·ε̃_s^{(R)}，其中 ε̃_s^{(R)} 是 response 层 asset 空间 OOF 残差行（2.4 输出契约 1）的 R_t 子向量——行覆盖 R_t ⟹ 全观测 ⟹ 无信息损失的联合实现值。**行集判定（裁决 A/B2'）**：J_t = {s : u_s ≤ t, O_s ⊇ R_t}（V_t 统计行集）；L = {s ∈ J_t : V_{s-1} 可定义}（ℓ_d 求和行集；V_{s-1} 可定义 ⟺ 存在至少一行更早的 joint 合法历史——D-049「有效起点由数学可定义性决定」的精确形式）。完整推导见 **docs/GATE0_VECTOR_INNOVATION.md**（§1.3 R 域、§2 V_t/J_t、§5 shape 协议、§6 joint row 规则）。

```julia
struct InnovationState
    R_t::Vector{Int}                   # risk 域资产集（free ∪ locked，裁决 A/B2'）
    N_R::Int                           # |R_t|
    E_R::Matrix{Float64}               # 域 mode 基 E_{R_t} = [e₀^{(R_t)}; Q^{(R_t)}]（正交，N_R × N_R）
    eps_R::Matrix{Float64}             # R 域 mode 残差 ε_s^{(R)}（行集 J_t 上，|J_t| × N_R；来源 = 2.4 输出契约的 R_t 子向量变换）
    row_support::Vector{BitVector}     # 每行 joint 观测支撑（O_s ⊇ R_t 判定的 mask 溯源，D-055）
    d_nodes::Vector{Float64}           # d 的 adaptive quadrature 节点（连续，含 cell mass）
    d_weights::Vector{Float64}         # q(d|H) quasi-posterior 权重（D-047/D-057）
    V_t::Vector{Matrix{Float64}}       # 每 d node 的 V_t(d) ∈ R^{N_R×N_R}（R 域 mode 坐标，构造性 PSD）
    support_cert::SupportCertificate   # PSD support / 伪逆平方根证书（floor→0 refinement，D-050/D-051）
    z_pool::DConditionedShapePool      # standardized joint rows（per-d 结构或 d-conditioned accessor：z_s 依赖 V_{s-1}(d)；D-052~D-054）
    # propriety 断言（构造时强制，裁决 A/B2'）：|J_t| ≥ 1（V_t 可定义）、|L| ≥ 1
    # （ℓ_d 有信息的必要条件）——违反即 error，不得静默返回均匀先验冒装有信息的 quasi-posterior
end
```

| 字段 | 语义 | 规范来源与旧对象映射 |
|---|---|---|
| `R_t`/`N_R`/`E_R` | risk 域资产集及其 mode 基；全部 innovation 对象（V_t、z_pool、ℓ_d）定义在 R 域 mode 坐标上 | 裁决 A/B2'（D-045 的 R 域实例化 **D-045a**）；推导见 GATE0_VECTOR_INNOVATION.md §1.3 |
| `eps_R` | ε_s^{(R)} = E_{R_t}ᵀ·ε̃_s^{(R)}（**N_R 维 R 域 mode 向量**，非 macro 标量投影、非 active 域全量） | D-045a；取代 `res_history` 的 macro 标量序列用法（docs/INNOVATION_LAW.md §1.1 第一步的 scalar 投影现状） |
| `V_t` | V_t(d) = Σ_{s∈J_t} w_s·ε_s^{(R)}ε_s^{(R)ᵀ}，w_s = k_d(t+1−u_s)/Σ_{s'∈J_t} k_d(t+1−u_{s'})——**分子分母同步收缩到 J_t**（推导 §2.2 要点 1：分母含缺失行权重而分子不含 = 把缺失伪装成零方差证据，SPEC §12.1/D-012 的观测污染在二阶矩层的翻版）；kernel 与旧 `frac_weights`（src/predict.jl:12）**同一族**（D-046）；无固定 window（D-048）；burn 废除（D-049：可定义性 ⟺ \|J_t\|≥1）；**秩与 d 无关**（只由 J_t 残差张成子空间决定——不存在「小 d 经秩亏获得似然优势」的偏置通道） | D-045/D-045a/D-046/D-048/D-049 |
| `support_cert` | V^{−1/2} 为 **Moore-Penrose inverse square root on observed positive support**；零特征方向不逆、不注噪、留 null support；数值 eigen floor 仅用于浮点分类且须 floor→0 refinement；不允许固定 covariance floor 成为理论 | D-050/D-051 |
| `z_pool` | z_s = V_{s-1}^{+1/2}(d)·ε_s **依赖 d** ⇒ per-d 结构（或 d-conditioned accessor）；**单一行集概念（裁决 C2）**：pool / V 统计 / ℓ 求和**三行集同一**（都要求「覆盖 R_t + 因果性」；V 统计额外要求自身可定义、z/ℓ 要求 V_{s-1} 可定义）——单一 mask 判定，无三处不一致的实现面；**一个 scenario innovation 必须来自一个联合合法 shape row**（禁跨日期拼 cell） | D-052/D-053/D-054/D-055 + 裁决 C2 |
| `d_weights` | ℓ_d = −½ Σ_{s∈L}[log det⁺ V_{s-1}(d) + ε_sᵀV_{s-1}(d)⁺ε_s]（**只在 L 上求和**），q(d\|H) ∝ exp(ℓ_d)·1_{(0,1)}(d)；先验 Uniform(0,1)；η≡1（禁 temperature）；**propriety 自动成立**（ℓ_d ≤ 0 恒可积），前提 \|L\| ≥ 1（构造断言） | D-057/D-047/D-058 |
| 删除对象 | — | `v_forecasts`/`v_bootstrap`（D-059：vector law 自含绝对尺度）、`own_res_rows`（D-054）、`d_posterior` 的 DGRID_V1 离散网格身份（D-047：DGRID_V1 只是数值节点候选，非理论对象） |

**覆盖不足时（D-056，T4 判定次序，裁决 C3）**：给定初始 R_t⁽⁰⁾ = free ∪ locked——(a) 覆盖不足由 **free** 资产引起：逐个剔除该 free 资产（不进 free risky set），R_t 收缩，行集 J_t/pool 恢复；(b) 剔除后仍不足（覆盖不足由 **locked** 资产引起，或 free 剔尽仍不足）→ **fail loudly**；回测驱动器（gate0/backtest.jl）对该日**维持持仓**（不 stitching、不 zero-fill、不假装 full rank）。资金可留 cash——这正是 cash 入 Kelly 的原因之一。

### 2.6 PredictiveLaw（owner：`src/gate0/predictive.jl`；裁决 D-040~D-042/D-060 + 终审 A2/A4）

```julia
struct PredictiveLaw
    response::ResponsePosterior
    innovation::InnovationState
    x_now::Vector{Float64}              # 当前决策 feature（ModeProblem.x_now 冻结）
    variance_split::NamedTuple          # (response_epistemic, innovation_aleatoric)——可报告的方差分解
end
```

- 职责：one-step 总 predictive law $P(r_{t+1}\mid\mathcal H_t)$ 的组装——scenario draw = response posterior draw（mean uncertainty）+ innovation draw（$V_t(d)^{1/2}z$）；mode→asset 映射沿裁决书 §19 八步链条。
- **R 子集执行路径（裁决 A2/A4，scenario 合成的规范执行序列）**：scenario 合成在 risk 域 R_t（= free ∪ locked）上执行——μ_R = E_{R_t}ᵀ·μ_asset[R_t]（μ_asset 的 R_t 子向量的 R 域 mode 投影）→ y_R = μ_R + ε_R（ε_R 来自 innovation 的 R 域通道）→ u_R = E_{R_t}·y_R（R 域 mode→asset 重构）→ ×s₁[R_t]（恢复 log return）→ exp（gross）；**Kelly 的 scenario 列集 = R_t**（free 子集解 Kelly、locked 子集进 base_s，D-017）。
- **不重复计算铁律（D-060）**：response posterior draw 负责 mean 不确定性、innovation draw 负责残差不确定性；$\Sigma_R$ 是 response 模块的 working-likelihood covariance / posterior scaling nuisance，**不得**再作为未来 residual shock 加一次（推导 §7：$\Sigma_R$ 不出现在 innovation 通道）。
- 顶层名称（D-042）：`modular posterior predictive`（Bayesian response posterior + cross-fitted semiparametric innovation predictive）；禁称 fully Bayesian generative posterior predictive（README 状态声明补此名称，归 Wave 2 文档轮——见 10.3）。
- 方差分解（裁决书 §30）：$Var(r\mid H)=Var_{response}+E[V_\epsilon\mid H]$；response 项在 α 混合下可再分解 within/between（推导 §7.2 闭式，between 分量非零可算——plug-in 旧实现恒 0 的对照面）；无法计算部分写 NOT COMPUTED，不得显示 0。

### 2.7 V1Model 拆解方案（D-081）

旧 `V1Model`（src/predict.jl:16-36）十四字段逐项去向：

| 旧字段 | 新 owner / 处置 |
|---|---|
| `active_indices` | ModeProblem.admitted（坐标空间准入）+ Kelly 层 free 掩码（Eligibility 派生）——「进入模型」与「可交易」两道门分离 |
| `N_universe` | ModeProblem |
| `e0` | ModeProblem（几何；决策日 alive 支撑上定义，语义沿 src/predict.jl:473-478） |
| `s1`/`s_macro`/`s_perp` | ModeProblem（几何，D-024 确定性；实现按 H7 复制） |
| `relative_observed` | ModeProblem.observed（mask 溯源） |
| `resp`（ResponseOperator） | ResponsePosterior |
| `mu_pred` | PredictiveLaw（decision-time 边缘的 asset 空间投影：μ_asset = E_active·μ，2.4 输出契约 2） |
| `res_history`（ResidualOracle） | InnovationState.eps_R（R 域 N_R 维 mode 向量；lazy oracle 的按行求值加速属旧线性能资产，新线 reference 直接 dense，D-082/D-083） |
| `own_res_rows` | **删除**（D-054 禁 own-row cell stitching） |
| `pred_moments` | ResponsePosterior.predictive |
| `d_posterior` | InnovationState.d_weights |
| `v_forecasts` | **删除**（D-059） |
| `v_bootstrap` | **删除**（D-059） |

拆解后每个数学概念一个 owner：response/epistemic → gate0/response.jl；mean 边缘 → ResponsePosterior.predictive；residual provider → gate0/innovation.jl；d posterior → InnovationState。

---

## 3. mode-space response 维度图（D-025/D-026/D-030/D-031/D-033）

### 3.1 精确维度

**输入**（统一 mode 输入，$x_t\in\mathbb R^{14N+1}$）：

$$
x_t=\begin{bmatrix}1\\ B_m(t)\\ B_\perp(t)\end{bmatrix},\qquad
B_m\in\mathbb R^{14},\quad B_\perp\in\mathbb R^{14(N-1)}
$$

| 分量 | 维数 | 内容 | 尺度 |
|---|---|---|---|
| DC 列 | 1 | $x_{0,t}=1$（D-031 恢复的统一 intercept 通道） | 无（常数） |
| $B_m$ | 14 | macro Q/P path features：对 $X_m(t)=\sum_{s\le t}m_s$ 用 paired path basis（7 band × Q/P，语义同 src/response.jl:2-16 的 `path_basis_1d`） | $s_m$（per-band macro ruler） |
| $B_\perp$ | 14(N−1) | **gauge 坐标** relative Q/P path features：$q_t=Q^{\top}e_t\in\mathbb R^{N-1}$，累积 $X_q(t)=\sum_{s\le t}q_s$，每 gauge 坐标 k=1..N−1 做 paired path basis（7 band × Q/P） | $s_\perp$（per-band gauge-invariant scalar ruler） |

**总输入维：$1+14+14(N-1)=14N+1$。**

**输出**（统一 mode 输出，$y_t\in\mathbb R^N$）：

$$
y_t=\begin{bmatrix}m_t\\ q_t\end{bmatrix},\qquad q_t=Q^{\top}e_t\in\mathbb R^{N-1}
$$

（$m_t$ 为 center-of-mass macro 标量——D-021 保留分解；$e_t$ 为 zero-embedded relative field，其行和构造性为零，故 $Q^{\top}e$ 无损且 $e_0$ 分量由 $m$ 单独承载。）

**统一响应**：

$$
y_{t+1}=b_0+Gx_t+\epsilon_{t+1},\qquad G\in\mathbb R^{N\times(14N+1)}
$$

四 block 分解（行 = [macro 行；relative 行]，列 = [DC；macro 列；relative 列]）：

| block | 维度 | 语义 |
|---|---|---|
| $G_{mm}$ = G[1, 2:15] | $\mathbb R^{1\times14}$ | macro path → macro（旧线已有） |
| $G_{m\perp}$ = G[1, 16:14N+1] | $\mathbb R^{1\times14(N-1)}$ | **relative path → 整体市场**（横截面 rotation 预测 macro；D-026） |
| $G_{\perp m}$ = G[2:N, 2:15] | $\mathbb R^{(N-1)\times14}$ | **macro path → 横截面 rotation**（D-026） |
| $G_{\perp\perp}$ = G[2:N, 16:14N+1] | $\mathbb R^{(N-1)\times14(N-1)}$ | relative → relative（旧线已有，但坐标不同） |
| $b_0$ = G[:, 1] | $\mathbb R^{N}$ | DC intercept：$b_0[1]$ macro unconditional drift；$b_0[2{:}N]$ zero-sum relative long-run drift（D-031）；prior 零中心（D-032，经 $\alpha_0$ group precision 表达） |

四个 block 全部由数据/posterior 决定（D-025）；先验精度 $\Lambda_\alpha=\mathrm{diag}(\alpha_0,\alpha_p,\ldots,\alpha_p)$ 恰好两 group（DC vs 全部 path coefficients，D-033）。

### 3.2 与现有 block-diagonal design 的差异表

| 维度 | 现有实现（证据） | 新设计（裁决） |
|---|---|---|
| macro 输入 | $B_m\in\mathbb R^{14}$，**独立回归**（src/predict.jl:281；src/response.jl:1130-1155） | 同维，但进入**统一**回归 |
| relative 输入 | **14N** 列，asset 坐标 embedded field（`fill_design_matrix!` 列布局 offset=(b−1)2N、col_q/col_p 逐资产，src/response.jl:55-75、1164-1172；P_features=2·\|BANDS\|·N，src/prepare.jl:145） | **14(N−1)** 列，gauge 坐标 $Q^{\top}e$（D-030：直接在 gauge 坐标工作，构造性删除 input common unreachable direction 与 output common/relative 重复表示） |
| DC 通道 | 无（0 列） | 1 列（D-031） |
| 总输入维 | 两条独立 design：14（macro）与 14N（relative） | 统一 14N+1 |
| 输出 | m 标量（macro 回归）+ $e\in\mathbb R^N$（relative 回归，Y=embedded field 行） | 统一 $[m;q]\in\mathbb R^N$ |
| cross block | $G_{m\perp}=G_{\perp m}=0$（block-diagonal 假设，无理论推导——D-025 点名；docs/MODEL_LEDGER.md §6 的分别拟合事实） | 全部放开 |
| trace 约束 | 14 条 per-band/per-channel 硬条件化（src/response.jl:106-143、1199-1202；docs/TRACE_NEUTRALITY_DERIVATION.md 裁决：candidate hypothesis） | **不施加**（D-028；D-030 的 gauge 坐标构造性删除 common/重复表示——推导 §8 的构造性论证） |
| output covariance | macro `sig2` 点估计（src/response.jl:1154，D-039 取消）+ relative Σ（N−1 gauge support） | 统一 $\Sigma_R\in\mathbb R^{N\times N}$（mode 空间 output covariance，D-035 Jeffreys 先验） |
| 超参 | $\hat\alpha_m,\hat\alpha_{rel}$ EB 点估计 plug-in（docs/POSTERIOR_DEFINITION.md §2） | $(\log\alpha_0,\log\alpha_p)$ quadrature 积分（D-034/D-038） |

### 3.3 语义精度注记（如实记录的张力，不替裁决书补推导）

- docs/TRACE_NEUTRALITY_DERIVATION.md A.2 指出：旧 C 约束杀死的是**对角和**（$\sum_i G[i,\text{asset }i\text{ 的列}]$），与「移除 universal common timing response」的旧 SPEC §21 措辞在「共同响应含资产间对称传导分量」的解释下存在语义缺口。D-030 以 gauge 坐标「构造性删除 common/重复表示」替代该需求——本文按裁决书呈现目标结构，不声称两者数学等价；trace 假设的反例测试（D-094）独立覆盖其降级依据。
- ragged 行的 zero-embedding：旧 `embedded_relative_field`（src/predict.jl:43-70）对 observed 坐标减 observed 均值、缺失坐标写 0，行和构造性为零（$\sum_j=\text{total}-c\cdot\text{total}/c=0$），故 $e_t\in\mathbf 1^\perp$ 恒成立、$Q^{\top}e$ 无损。这是 Step 3 reconstruct 测试的断言对象；观察 mask 永久单独保留（D-022）。

---

## 4. Step 0–16 施工图（§40 的落实）

**顺序纪律（显式编码，违反即停工）**：

1. **Step 0 只改文档，不改 runtime**（§40 Step 0 原文）；静态审查无冲突后才开 Step 1。
2. **Step 7 之前严禁写复杂 optimizer**（§40 Step 7 原文）；Step 4 用固定 ridge/simple Gaussian reference 只验证 design/target algebra。
3. 每步只开一个层次（开发守则 §19/条款 H）；后 Gate 破坏前 Gate 立即回退（条款 C）。
4. 测试按 D-089 阶梯：static → tiny analytic → single day → 5 → 20 → 60 → 501 days；任何一级红立即停止升级。
5. **第一阶段完全禁止两年重跑**（D-088）：§41 全部完成前不碰两年/十年全量。
6. 所有容差/先验/结构选择不得以回测收益/Sharpe 裁决（D-044、开发守则 §24）。
7. Engineer 不执行命令；运行验证由 Manager 安排 DevOps（≤60s/RSS 护栏沿用）。

### Step 0 — 只改文档

- **目标**：裁决书并入 SPEC/ROADMAP/README release status；状态改 `2.0-RC-G0 / Gate 0 Reopened`（D-003）；D-096 禁用名称与 D-097 推荐名称落文档；§56 五句旧表述改写（D-073「Volatility Pump」、D-074「脱离 beta」等）；D-042 顶层名称写入。
- **涉及文件**：AGENTS.md（裁决书已在）、README.md、RELEASE.toml、RELEASE_CPU_2_0.md、SPEC 合并稿。**不改 runtime、不改测试。**
- **前置依赖**：无。
- **测试**：无 runtime；静态交叉引用审查。
- **验收判据**：静态审查无冲突；D-002 历史证据零改写（旧报告/tag/hash/artifact 原字节）；新增状态文件声明「2.0.0 Final 数学闭合声明已被撤销」。

### Step 1 — 拆 observation / eligibility

- **目标**：`MarketFacts`+`Eligibility` 落地（2.1/2.2 规格）；252 日规则改写为 `trade_eligible` 构造规则（D-018/D-019，口径按 H4）；删除一切「改 bar 表达政策」的代码路径（第 7 节清单）；free = E^trade ∧ T^exec（D-016）；locked 语义单独处理（D-017）。
- **涉及文件**：新建 `src/gate0/market.jl`；数据桥为裸矩阵构造函数（§1.3，H1 互不 include 推论）。旧 src/ 零改动。
- **前置依赖**：Step 0。
- **测试**（tiny analytic）：mask 语义四禁（政策性写入在类型/测试上不存在）；252 计数口径边界（H4：累计第 252 个有效 bar 当日解除）；dummy all-NaN、IPO、单日 missing（§35 ragged 1–3）；资产置换协变；桥的字节级一致（observed ≡ bar）。
- **验收判据**：D-012~D-020 落测试；§41 前四项（bar 只表示 observation；三 mask 分离；benchmark 外生化接口就绪；free 不含 admission）。

### Step 2 — Kelly cash feasible set

- **目标**：把 $\sum_i w_i=1$（risky-only，src/kelly.jl:45、150-170 的 budget 语义）改成 $\sum_i w_i\le1$ 或显式 cash 列 gross=1（D-068）；**保持完全相同的 log objective**；fully-invested risky simplex 降级为显式实验约束（D-069）；cash 不是 fractional Kelly（D-070）；不允许仓位上限/entropy/风险平价修集中（D-071）；locked base_s 保留（D-017）。
- **涉及文件**：新建 `src/gate0/kelly.jl`（solver 可复用旧 `kelly_weights_v1`/Clarabel 的**目标函数形态**——cash 列 gross=1 时同一 solver 直接适用；这是数值工程复用，非数学复用）。
- **前置依赖**：无数学依赖（可与 Step 1 并行开发；施工顺序按 §40 串行验收）。
- **测试**（tiny analytic）：§37 四 case——A 全 risky 确定性 <1 ⇒ w_cash=1；B 单资产确定性 >1 ⇒ w_asset=1；C 高均值高风险内部解；D locked 的 cash+free+locked 财富正确相加；D-090 对称世界的 Kelly 侧（N=3 exchangeable + cash）。
- **验收判据**：cash Kelly 四 case 全绿；KKT/可行性证书（沿用 src/kelly.jl:21-37 证书形态）通过。

### Step 3 — 统一 mode coordinate producer

- **目标**：只做 $[m;Q^{\top}e]$ 与 $[1;B_m;B_\perp]$ 的 producer（2.3 节）；**暂不改 posterior**（§40 Step 3 原文）。
- **涉及文件**：新建 `src/gate0/mode_problem.jl`；几何原语按 H7 纯函数复制进入新线（1.3 节边界，来源标注 + 静态确认纯函数性）。
- **前置依赖**：Step 1。
- **测试**（tiny analytic）：reconstruct（asset→mode→asset 往返恒等）；permutation 协变；gauge 旋转（Q→QR）不变；zero embedding 语义（D-022）；线性交换性（$Q^{\top}X_{rel}=X_q$：先投影再累积=先累积再投影）；维度断言（X_mode 列数=14N+1、Y_mode 列数=N）。
- **验收判据**：D-021~D-024、D-030 的坐标部分全绿。

### Step 4 — Full block response reference（固定 ridge）

- **目标**：四 block 统一 $G$ 的 **design/target algebra** 验证；固定 ridge / simple Gaussian reference（**不上 full posterior**——§40 Step 4 原文）。
- **涉及文件**：新建 `src/gate0/response.jl`（固定 ridge 版 fit；无 EB、无 trace 条件化）。
- **前置依赖**：Step 3。
- **测试**（tiny analytic）：**D-092 known cross-mode world**——人工 $G_{\perp m}\neq0$ 或 $G_{m\perp}\neq0$，新 full response 必须 recover；**旧 block-diagonal 实现应故意失败**（旧线 fixture 跑同一合成数据，差异记录为修复证据，证明新模型确实修复被删 block）。
- **验收判据**：D-025/D-026 落测试；四 block recover 精度达 tiny analytic tolerance。

### Step 5 — 加 DC channel

- **目标**：统一 response 中加常数 feature 列（$x_{0,t}=1$）。
- **涉及文件**：同 Step 4。
- **前置依赖**：Step 4。
- **测试**：**D-093 known DC world**——人工 constant drift，新 DC channel recover，dynamic Q/P coefficients 保持 0；无 DC 数据时 $b_0$ 估计收缩到 0 量级正确。
- **验收判据**：D-031 落地；D-032 的 prior 零中心在 Step 8 的 posterior 层复验。

### Step 6 — 移除 production trace neutrality

- **目标**：新 core 不调用 14 constraints（D-028）；旧 trace-conditioned solver 不删除、移 legacy/hypothesis 路径（D-029；处置见第 6 节）。
- **涉及文件**：`src/gate0/response.jl` 不含 `condition_trace_neutrality` 调用；旧线 src/response.jl 冻结不动。
- **前置依赖**：Step 4/5。
- **测试**：**D-094 trace-hypothesis counterexample**——构造真实 diagonal common response 的合成世界：unconstrained 新 core recover；旧 trace-neutral branch 系统性删除该分量（反例证据）。
- **验收判据**：D-027/D-028/D-029 落地；§41「14 trace constraints 不在 default core」打钩。

### Step 7 — 推导 full response posterior（纸面，严禁 optimizer）

- **目标**：纸面推导已交付：**docs/GATE0_RESPONSE_POSTERIOR.md（裁决 H5 定名）**——含 §4 解析积分路线（B 共轭消元 → Σ_R Jeffreys-IW 精确消元 → 2 维 quadrature）、§5 propriety 完整静态推导（小尾无条件收敛 / 大尾饱和发散）、§10 实现者检查清单 14 条。**先验修正已裁决（D-035a，裁决 B/G4），无规范阻塞。**
- **涉及文件**：docs/GATE0_RESPONSE_POSTERIOR.md（已存在）。**不写任何 runtime optimizer。**
- **前置依赖**：Step 4/5/6 的代数已验证；推导已经 Manager 终审（裁决 B/G4）。
- **测试**：无 runtime；推导静态 review 已由终审完成。
- **验收判据**：已达成——推导成文（含「何时不满足」的 propriety 判别式：A2 数据层 n≥N 且 S(α)≻0）；解析路线完全成功（推导 §4.5：Σ_R 整体解析消元，无需数值积分——D-037 顺序裁决的最强满足）；D-035a 修正已裁决（p(α)∝1/[α(1+α)]，纯数学 propriety 依据，非回测选择）。**Step 8 无规范阻塞。**
- **纪律**：本步是「先想清楚再讲」的强制落点；跳过推导直接写 solver 是 Gate 违规。

### Step 8 — 写 slow full-posterior reference

- **目标**：**先验修正已裁决（D-035a，裁决 B/G4），无规范阻塞**；单日 reference 实现（**先单日、不优化**）；$(\log\alpha_0,\log\alpha_p)$ deterministic adaptive quadrature（D-038：adaptive、tail/refinement 有证书、不收敛 fail loudly）；propriety gate fail loudly（D-036）；解析优先、MCMC 不作第一版（D-085/D-086）；统一 $\Sigma_R$（D-039）。
- **涉及文件**：`src/gate0/response.jl`（posterior 部分）+ `src/gate0/quadrature.jl` 雏形（alpha 维度）。
- **前置依赖**：Step 7 静态 review 已由终审完成（裁决 B/G4：先验修正已裁决，无规范阻塞）。
- **测试**（tiny analytic，裁决 G5 扩充）：**A2 数据层 fixture**（n<N 与退化 Y → error，文本含 "posterior improper"——数据层条件仍是负测试）；**S(α) 双构造一致性断言**（S(α) = S_yy − S_xyᵀ(S_xx+Λ_α)⁻¹S_xy 与直接构造 Yᵀ(I−XA⁻¹Xᵀ)Y 逐 α 对照；推导明示高危陷阱：S(α) **不是** ridge 残差平方和 (C⁻¹Y)ᵀ(C⁻¹Y)——写错会静默改变全部下游）；**matrix-t vs dense 小系统数值积分对照**（p(B\|Y,α) 核与显式数值积分一致）；**μ 的 t 边缘矩 vs dense 对照**（均值 = B̂x̃_t、协方差 = [c/(ν−2)]S(α)，ν=n+1−N）；**quadrature 同 seed 逐位重放**；**预算耗尽统一错误文本**（含 "Numerical integration did not converge"）；先验退化为点质量时与 Step 4 固定 ridge 一致（回归锚）；**propriety gate 应绿**（D-035a 修正先验下右尾收敛——若红是实现 bug，不再是预期红；裁决 B/G4）；quadrature 细化时边缘矩收敛；旧 `optimize_conditioned_eb` 只提供 mode/bracket/diagnostic 对照（D-038 尾句）。
- **验收判据**：D-034~D-039、D-035a、D-085/D-086 落地；§41「response hyperparameters 不再 point plug-in」「posterior propriety 有证明」两项打钩。

### Step 9 — 重建 OOF full-mode folds

- **目标**：每 fold 独立 fit、独立 $(\alpha_0,\alpha_p)$ posterior integration、不读 held-out rows、不复用 full-data posterior、昨日同 fold state 仅作 numerical warm-start（D-043）；dense residual reference 先实现（lazy 以后再说——§40 Step 9）。
- **涉及文件**：新建 `src/gate0/oof.jl`。
- **前置依赖**：Step 8。
- **测试**（tiny analytic，§34 全项）：held-out row 不进 response posterior；不进 alpha quadrature evidence；fold residual 与 dense reference 一致；**unified macro-relative cross blocks 也严格 OOF**；无观测 row 不得当 residual=0 进 likelihood。
- **验收判据**：D-043/D-044 落地（OOF 只用于 innovation 校准、不作模型选择工具——D-044 纪律进测试注释）。

### Step 10 — 写 vector innovation reference

- **目标**：$V_t(d)$ 的 R 域 N_R×N_R 直接实现（**O(T²N_R²) 可以接受，这是 reference；先证明数学，禁 FFT/incremental**——§40 Step 10）；全部因果历史（D-048）；burn 不再是理论参数（D-049）；PSD/support（D-050/D-051）；kernel 与旧同一 fractional family（D-046）；R 域实例化按 2.5 节（D-045a）。
- **涉及文件**：新建 `src/gate0/innovation.jl`。
- **前置依赖**：Step 1（Eligibility：R_t = free ∪ locked 需要 free/locked 语义，裁决 G3）、Step 2（locked 语义）、Step 9（OOF vector 残差）。
- **测试**（tiny analytic）：**D-095 vector volatility world**（人为某 relative 方向波动上升 ⇒ V_t 只放大该方向；旧 scalar law 错误放大全场——旧线对照）；极限一致性（d=1 等权二阶矩；d→0 最近行）；PSD 构造性；floor→0 refinement 收敛；**shape 语义类断言（裁决 G6）**——M_z 诊断报告（标准化残差二阶矩 E[zzᵀ] 的诊断输出）、tail/skew 保留（z_pool 不改变高阶矩）、cross-mode shock 分量间经验相关 = 历史相关（标准化不破坏联合结构）；**\|J_t\|=0 fixture**（→ fail loudly：V_t 不可定义）；**rank deficient fixture**（\|J_t\|<N_R 或行向量共线：null 方向 variance=0、z 的 null 分量=0、无 δI 注入）；**空观察行不进统计**（c=0 行不入 J_t/L——对照旧线 e[idx]=0.0 的处理，src/residual_oracle.jl:239）；**未来行注入构造性反例**（u_s > t 的行构造性不可入 J_t——因果性定理的负测试）；**N_R=1 标量对照**（kernel 与旧 scalar 族一致性 vs v_bootstrap 锚定删除的分离报告——D-059 的对照证据）；**innovation 层 dummy 资产 V_t 逐字节不变**（全 NaN 资产不进 R_t）。
- **验收判据**：D-045/D-045a/D-046/D-048~D-051、D-059 落地。

### Step 11 — standardized empirical shape

- **目标**：z_s = V_{s-1}^{+1/2}(d)·ε_s 的 joint row 池（D-052/D-053/D-055；R 域坐标，2.5 节）；**删除 own-cell stitching**（D-054）；覆盖不足时按 T4 判定次序处理（D-056，裁决 C3）。
- **涉及文件**：同 Step 10。
- **前置依赖**：Step 1/2（R_t = free ∪ locked 语义，裁决 G3）、Step 10。
- **测试**（§35 ragged 4–6）：held but unobservable；current free set 是历史 subset；no compatible joint residual shape rows（→ 按 2.5 节 T4 判定次序：free 剔除 → R_t 收缩；locked 不足 → fail loudly + 回测驱动器维持持仓，裁决 C3）；**stitching 拼接向量反例**（跨日期拼 cell 的构造向量不得出现在任何 pool——D-054 负测试，裁决 G6）；资产置换；gauge 旋转。
- **验收判据**：D-052~D-056 落地；「no own-cell stitching」进 §41 清单。

### Step 12 — continuous d quasi-posterior

- **目标**：$\ell_d$ 的 direct sums（**不要 FFT**——§40 Step 12）；$q(d\mid H)$ 连续 quasi-posterior（D-057）；先验显式 Uniform(0,1)（D-047：d=1 可作闭区间端点极限包含；DGRID_V1 退为数值节点候选，非均匀网格带 cell mass）；η≡1（D-058）。
- **涉及文件**：同 Step 10。
- **前置依赖**：Step 10。
- **测试**：quasi-likelihood 只在行集 L 上计算（2.5 节单一行集，裁决 C2）；网格细化先验质量不漂移（SPEC §64 语义在 d 维度的复现）；adaptive 节点收敛证书。
- **验收判据**：D-047/D-057/D-058 落地。

### Step 13 — 组合 PredictiveLaw

- **目标**：one-day scenario/reference draws 组装（2.6 节，R 子集执行路径按裁决 A2/A4）；response draw 与 innovation draw 职责分离（D-060）；方差分解可报告（§30）。
- **涉及文件**：新建 `src/gate0/predictive.jl`。
- **前置依赖**：Step 8/9、Step 11/12。
- **测试**（single day 级）：固定 seed 重放；mode→asset 映射往返（R 域执行路径）；方差分解两分量分别非零且可报告；$\Sigma_R$ 不作为额外 residual shock 重复出现（负测试：innovation 置零后 scenario 方差只剩 epistemic 分量）。
- **验收判据**：D-040/D-041/D-042/D-060 落地。

### Step 14 — adaptive RQMC

- **目标**：nested **Owen-scrambled Sobol**（D-063：production 默认 backend；Halton 保留 reference/legacy comparison，理由是数值积分结构而非回测表现）；**两个独立 numerical certificate**（D-064：A 权重收敛 $\|w_{2M}-w_M\|_1\le\epsilon_w$；B utility regret $I_{audit}(w_{2M})-I_{audit}(w_M)\le\epsilon_U$；KKT 第三证书，三者不互替）；**独立 audit replicate**（D-065：optimization rule 与 audit scrambled rule 分离，seed 确定性派生，防 sample optimization optimism）；容差经 numerical refinement 定稿（D-066：synthetic 解析律→realistic fixture→减半→稳定→与 Sharpe/PnL 无关→记录为 numerical configuration）；预算耗尽 fail loudly（D-067：统一错误文本 "Numerical integration did not converge"，不得返回最后一层权重）。
- **涉及文件**：新建 `src/gate0/quadrature.jl`（完整版）。
- **前置依赖**：Step 13。
- **测试**（tiny analytic → single day，§36 全项）：known Gaussian 2-asset analytic Kelly；nested M→2M；optimization/audit 独立 replicate；same seed replay；不同 seed 收敛到同权重；预算耗尽 fail-loud；无 fixed-S production 捷径（负测试：固定 S 路径在生产签名不可达）。
- **验收判据**：D-061~D-067 落地；§41「fixed S production path 删除」「integration 有 independent audit certificate」打钩。

### Step 15 — 端到端 Kelly

- **目标**：full posterior + vector innovation + converged integration + cash 的完整链条 $\mathcal H_t\to P(r_{t+1}\mid\mathcal H_t)\to w_t^*$；reference sequential backtest runner（每日严格 prepare→solve→scenarios→kelly→advance，宪法 oracle 语义沿旧 SPEC §42/§85；覆盖不足日按 T4 维持持仓，裁决 C3）；**三 benchmark**（D-072：cash R=1、daily equal-weight risky、initial equal-weight buy-and-hold——首日定义按 H8：首个决策日等权（E^trade∧T^exec 候选集）；2 vs 3 才能讨论 rebalancing premium——D-073）；concentration cause report（D-076：top asset、risky/cash weight、posterior expected log-growth、epistemic/innovation variance、utility regret if 1% shifted、M、certificate、top posterior mean directions、top innovation eigenmodes——不能只打 "Kelly concentrated"）；高集中解释前置条件（D-075 六项）。
- **涉及文件**：新建 `src/gate0/backtest.jl`。
- **前置依赖**：Step 1/2/13/14。
- **测试**（5 → 20 → 60 → 501 days 阶梯）：每级绿才升下一级；benchmark universe 外生（第 8 节）；无未来 bar 进决策（因果性）。
- **验收判据**：D-068~D-076 落地；501 days 通过后仍**不自动开两年**（**放行条件已裁决 H9：Gate-0 Exit 十八项 + D-088 清单**）。

### Step 16 — Gate-0 constitutional suite

- **目标**：全部 synthetic worlds + 不变量 + fail 条件的完整 suite 收口；§41 Exit 清单逐项核对。
- **涉及文件**：`test/gate0/`（新测试树；独立入口，裁决 H1；与旧 REQUIRED registry 并存，旧测试守护旧线 fixture）。
- **前置依赖**：Step 1–15。
- **测试**：第 9 节完整矩阵。
- **验收判据**：**§41 十八项全部为真，少一项 Gate 0 不得关闭**（映射见 5.2）。

---

## 5. D-001..D-097 覆盖矩阵

### 5.1 主体矩阵

落地形态缩写：〔S#〕= Step #；〔文档〕= 纯规范措辞/文档工作；〔资产〕= 旧资产处置（第 6 节）；〔测试〕= 测试设计（第 9 节）。

| 裁决 | 内容（一句话） | 落地 |
|---|---|---|
| D-001 | 裁决书临时最高规范 | 〔文档〕Step 0 并入；本文 §0 优先级声明 |
| D-002 | 不改写历史证据 | 〔文档〕Step 0；旧 artifact/tag/hash 原字节（§6） |
| D-003 | 撤销 Final 地位，2.0-RC-G0 | 〔文档〕Step 0 |
| D-004 | -62.22% 是 failure fixture | 〔资产〕保留为 regression fixture；禁作 target/tuning |
| D-005 | 不回滚早期赢家 | 〔文档〕Step 0 写入 SPEC |
| D-006 | 立即回 Gate 0 | 〔文档〕本文档整体 |
| D-007 | 顺序不变，后层破坏前层回退 | 〔文档〕§4 纪律第 3 条 |
| D-008 | 规律固定 | 〔文档〕理论保留条款 |
| D-009 | 全历史是原语 | 〔文档〕理论保留条款 |
| D-010 | 价格唯一 alpha 输入 | 〔文档〕MarketFacts 字段体现（无非价格输入） |
| D-011 | 日频不重开 | 〔文档〕Step 3 继承 step=one trading day |
| D-012 | Bars.bar 只表示 O | 〔S1〕MarketFacts.observed；第 7 节清单 |
| D-013 | model-admission mask | 〔S1〕Eligibility.model_admitted |
| D-014 | trade-eligibility mask | 〔S1〕Eligibility.trade_eligible |
| D-015 | physical-execution mask | 〔S1〕Eligibility.executable |
| D-016 | free = E^trade ∧ T^exec | 〔S1〕free 函数；纠正 src/backtest.jl:242-245 |
| D-017 | held 单独处理 | 〔S1/2〕locked 语义 + base_s（Step 2 Case D）；R_t = free∪locked（裁决 A/B2'） |
| D-018 | 252 规则属 trade eligibility | 〔S1〕trade_eligible 构造规则（**口径已裁决 H4：累计第 252 个有效 bar 当日解除**） |
| D-019 | 禁止篡改 Bars.bar | 〔S1〕第 7 节改写方案；政策写入类型上不可表达 |
| D-020 | benchmark universe 外生 | 〔S1/15〕第 8 节 |
| D-021 | center-of-mass/relative 保留 | 〔S3〕ModeProblem |
| D-022 | zero embedding 只能是 field algebra | 〔S3〕reconstruct/观察 mask 测试 |
| D-023 | fixed Helmert gauge 保留 | 〔S3〕ModeProblem.gauge |
| D-024 | ruler/gauge 非 posterior 对象 | 〔S3〕几何字段确定性；不恢复 geometry bootstrap |
| D-025 | 允许 macro↔relative cross-response | 〔S4〕统一 G 四 block；第 3 节 |
| D-026 | rotation 跨 macro-relative | 〔S4〕G_m⊥/G_⊥m 放开 + D-092 |
| D-027 | trace 降级 candidate hypothesis | 〔文档/S6〕TRACE_NEUTRALITY_DERIVATION 裁决引用 |
| D-028 | 默认关闭 14 条 trace | 〔S6〕gate0 core 不调用 |
| D-029 | 旧 trace solver 不删除 | 〔资产〕§6 legacy/hypothesis 保留 |
| D-030 | hard enforce 只有可达 support | 〔S3/4〕gauge 坐标 14(N−1)/N−1 |
| D-031 | 恢复 DC/intercept | 〔S5〕DC 列 + b_0 |
| D-032 | DC prior 零中心 | 〔S5/8〕α₀ group 零中心先验 |
| D-033 | DC/dynamic 两 group precision | 〔S7/8〕Λ_α=diag(α₀,α_p,…,α_p) |
| D-034 | α、Σ 不再 point estimate | 〔S8〕ResponsePosterior |
| D-035 | reference prior | 〔S7〕推导；〔S8〕实现。**D-035a 修正已裁决（裁决 B/G4）：p(α) ∝ 1/[α(1+α)]，纯数学 propriety 依据（右尾零模型平台 × log-uniform 测度发散，推导 §5.3-§5.4），非回测选择；Step 8 propriety gate 应绿，若红是实现 bug** |
| D-036 | propriety gate fail loudly | 〔S7/8〕ProprietyCertificate + 负测试。**D-035a 修正已裁决（裁决 B/G4）：Step 8 propriety gate 应绿，若红是实现 bug**（A2 数据层条件 n≥N 与 S(α)≻0 仍是 fail loudly 负测试） |
| D-037 | Σ_R 优先解析积分 | 〔S7〕顺序裁决（推导 §4.5：解析路线完全成功，无需数值积分） |
| D-038 | alpha log-space adaptive quadrature | 〔S8〕alpha_nodes/weights + 证书 |
| D-039 | macro σ² 独立点估计取消 | 〔S4/8〕统一 Σ_R |
| D-040 | response 负责 epistemic | 〔S13〕PredictiveLaw.response 职责 |
| D-041 | innovation 负责 OOF aleatoric | 〔S10/13〕InnovationState 职责 |
| D-042 | 顶层名称 modular posterior predictive | 〔文档/S0〕+〔S13〕docstring（README 补名称归 Wave 2，10.3） |
| D-043 | OOF 扩展到统一 response | 〔S9〕gate0/oof.jl |
| D-044 | OOF 不是模型选择工具 | 〔文档〕纪律；〔S9〕测试注释 |
| D-045 | innovation memory 是 vector/mode-space | 〔S10〕V_t(d)。**R 域实例化（D-045a，裁决 A/B2'）：V_t 为 N_R×N_R，R_t = free∪locked 的 mode 坐标 E_{R_t}**（推导见 GATE0_VECTOR_INNOVATION.md §1.3/§2） |
| D-046 | fractional kernel 继续统一 | 〔S10〕与旧 frac_weights 同族 |
| D-047 | d 先验 Uniform(0,1) | 〔S12〕d_nodes/d_weights |
| D-048 | kernel 全部因果历史 | 〔S10〕无固定 window |
| D-049 | burn=30 不再是理论参数 | 〔S10〕有效起点由数学可定义性（行集 L，2.5 节） |
| D-050 | 不允许固定 floor 成理论 | 〔S10〕SupportCertificate |
| D-051 | 伪逆平方根 on support | 〔S10/11〕Moore-Penrose |
| D-052 | standardized residual shape | 〔S11〕z_pool（d-conditioned，2.5 节） |
| D-053 | empirical shape 是 2.x 规范 | 〔S11〕semiparametric 语义 |
| D-054 | 禁止 own-row cell stitching | 〔S11〕joint row 池；own_res_rows 删除 |
| D-055 | ragged shape 按 mask/support | 〔S11〕row_support。**单一行集概念（裁决 C2）：pool/V 统计/ℓ 求和三行集同一（都要求覆盖 R_t + 因果性），单一 mask 判定，无三处不一致实现面** |
| D-056 | 无 joint rows 时 fail/cash | 〔S11/2〕。**T4 判定次序（裁决 C3）：free 资产覆盖不足 → 逐个剔除出 free risky set、R_t 收缩；locked 覆盖不足（或 free 剔尽仍不足）→ fail loudly + 回测驱动器维持持仓** |
| D-057 | d 用 Gaussian quasi-likelihood | 〔S12〕ℓ_d |
| D-058 | 不引入 temperature | 〔文档〕η≡1；无字段 |
| D-059 | 删除 v_bootstrap | 〔S10〕V_t 自含绝对尺度 |
| D-060 | response/innovation 不重复计算 | 〔S13〕variance_split + 负测试 |
| D-061 | Full Kelly 需收敛证书 | 〔S14/15〕 |
| D-062 | 生产必须 adaptive | 〔S14/15〕无固定 S 路径 |
| D-063 | Sobol 默认 | 〔S14〕backend |
| D-064 | 两个独立 certificate | 〔S14〕A/B |
| D-065 | 独立 audit replicate | 〔S14〕双 rule |
| D-066 | 容差 refinement 定稿 | 〔S14〕五步流程 |
| D-067 | 预算耗尽 fail loudly | 〔S14〕统一错误文本 |
| D-068 | cash 是 numeraire | 〔S2〕Σ risky ≤ 1 / cash 列 |
| D-069 | risky-only simplex 降级实验 | 〔S2〕显式实验约束 |
| D-070 | cash 不是 fractional Kelly | 〔文档/S2〕exact feasible set 语义 |
| D-071 | 不允许仓位上限修集中 | 〔文档〕禁令 |
| D-072 | 三 benchmark | 〔S15〕（buy-and-hold 首日按 H8） |
| D-073 | 禁称 Volatility Pump | 〔文档/S0〕§56 改写 |
| D-074 | correlation 不替代 beta | 〔文档/S0〕 |
| D-075 | 高集中不是独立 bug | 〔S15〕前置六项 |
| D-076 | concentration cause report | 〔S15〕报告字段 |
| D-077 | 不恢复旧 geometry bootstrap | 〔文档〕D-024 同源 |
| D-078 | 几何成未知 law 须另立理论 | 〔文档〕未来条款 |
| D-079 | BANDS/TAUS 是模型类定义 | 〔文档〕改动=理论版本变更 |
| D-080 | 本轮不研究 continuum bands | 〔文档〕Step 3-16 范围排除 |
| D-081 | V1Model 拆解 | 〔S8/13〕§2.7 映射表 |
| D-082 | 先写 slow reference | 〔路线〕§1 路线乙（H3 确认） |
| D-083 | reference 不追求 1 分钟 | 〔文档〕§4 纪律 |
| D-084 | optimize_conditioned_eb 移 legacy | 〔资产〕§6.3（H6：不复制） |
| D-085 | 能解析绝不采样 | 〔S7/8〕 |
| D-086 | 不允许 MCMC 第一版 | 〔S8〕deterministic only |
| D-087 | S=300 保留 regression fixture | 〔资产〕§6.4 |
| D-088 | 禁两年重跑 | 〔纪律〕§4 第 5 条（放行条件 H9） |
| D-089 | 测试升级阶梯 | 〔测试〕§9.1 |
| D-090 | Symmetric world | 〔测试〕§9.2-1 |
| D-091 | No-signal world | 〔测试〕§9.2-2 |
| D-092 | Known cross-mode world | 〔测试〕§9.2-3 |
| D-093 | Known DC world | 〔测试〕§9.2-4 |
| D-094 | Trace 反例 | 〔测试〕§9.2-5 |
| D-095 | Vector volatility world | 〔测试〕§9.2-6 |
| D-096 | 名称禁用 | 〔文档/S0〕 |
| D-097 | 推荐名称 | 〔文档/S0〕 |

### 5.2 §41 Exit 清单（18 项）→ Step/测试映射

| §41 项 | 落点 |
|---|---|
| Bars.bar 只表示 observation | S1（MarketFacts.observed + 第 7 节） |
| model/trade/execution mask 分离 | S1（Eligibility） |
| benchmark universe 外生 | S1/S15（第 8 节） |
| full mode response 含 cross blocks | S4 + D-092 |
| DC channel 存在且零中心 | S5/S8 + D-093 |
| 14 trace constraints 不在 default core | S6 + D-094 |
| response hyperparameters 不再 point plug-in | S8（对照 docs/POSTERIOR_DEFINITION.md §2） |
| posterior propriety 有证明 | S7（推导已交付 + D-035a 已裁决） |
| vector innovation law 实现 | S10 + D-095（R 域，D-045a） |
| no own-cell stitching | S11 |
| d prior 显式 | S12 |
| fixed S production path 删除 | S14（负测试） |
| integration 有 independent audit certificate | S14 |
| cash 在 feasible set | S2 + §37 |
| OOF full isolation | S9 + §34 |
| permutation/gauge/dummy invariance | S3/S11 |
| synthetic worlds 全部通过 | S16 |
| 所有 fail condition fail loudly | 各步负测试汇总（D-036/D-056/D-067 等） |

---

## 6. 旧资产处置清单

依据：裁决书 **§46（保留）**、**§47（删除/降级）**、**D-084**、**D-087**。（§46/§47 是章节号，与 D-046/D-047 两个裁决号是不同对象。）

### 6.1 保留（historical / 反例资产，非 production owner）

| 资产 | 位置 | 保留理由 |
|---|---|---|
| 原 2.0 source snapshot | `src/` 全部 + `dev/evidence/manager13/old_src_*/`（V0 `1d9b7ad`、V1.0 `21017b9`、`602b897` 快照，见 docs/OLD_TO_CURRENT_SEMANTIC_DIFF.md §0） | 历史 fixture（§46/D-082） |
| two-year failure artifact（-62.22%） | 发布物与 log（原字节原 hash 原时间线，D-004） | regression fixture：未来回答「哪个纠偏改变什么」（D-087） |
| hashes / manifest | `dev/evidence/manager*/`（final_snapshot、manifest 等） | 证据链（D-002） |
| conditioned EB tests | `test/conditioned_eb_tests.jl` 等（registry REQUIRED 内） | 守护旧线 fixture 语义 |
| trace hypothesis branch | 旧 `src/response.jl` 的 `condition_trace_neutrality` 链 + `test/conditioned_*` | 理论反例 / ablation / 未来复活（D-029） |
| 旧 response 语义 diff | docs/OLD_TO_CURRENT_SEMANTIC_DIFF.md | 历史语义（§46） |
| residual dense reference | `dense_oof_residuals`（src/residual_oracle.jl 底部，测试专用） | dense oracle；S9 新线另建自己的 dense reference |
| 旧增量 equality fixtures | `test/incremental_*` 等 | 历史等价性证据 |

### 6.2 默认 core 删除/降级（§47 清单 → 新线处置）

| §47 项 | 旧实现证据 | 新线处置 |
|---|---|---|
| per-band trace hard constraint | src/response.jl:106-143、1199-1202 | 不进 gate0 core（S6） |
| separate macro-only response | src/response.jl:1130-1155 | 统一回归（S4） |
| separate relative-only response | src/response.jl:1164-1208 | 统一回归（S4） |
| plug-in EB predictive | docs/POSTERIOR_DEFINITION.md §2 | ResponsePosterior（S8） |
| scalar fractional scale | src/predict.jl:615 | V_t(d)（S10） |
| `v_bootstrap` | src/predict.jl:499 | 删除（D-059） |
| own-row cell stitching | src/predict.jl:596-602 + src/residual_oracle.jl:264-274 | joint row 池（S11） |
| fixed-S production | src/backtest.jl:117-121、249-251（S=300, adaptive=false） | adaptive + 证书（S14） |
| risky-only forced full investment | src/kelly.jl:45、150-170 | cash feasible set（S2） |
| `Bars.bar` policy mutation | 宿主不在仓库（第 7 节） | MarketFacts.observed（S1） |
| benchmark dependency on model.active | src/backtest.jl:241-245→295 | 外生化（第 8 节） |
| causal story "low SNR → concentration" | 旧报告措辞 | §56 改写（S0） |

### 6.3 D-084：`optimize_conditioned_eb` 处置

保留 source/history 与 tests；**已裁决（H6）：旧 solver 不复制，旧线 src/ 冻结即保留**（不另建 experiments/ 目录副本）；不再继续性能优化（不以 sunk cost 保理论地位）；在新线中仅可提供 mode / initial bracket / diagnostic comparison（D-038），**不得定义 posterior**。

### 6.4 D-087：`S=300` 历史基准

两年 -62.22% 永久保留为 regression fixture；`dev/m1_artifact_replay.jl`、`dev/release_freeze.jl`、`dev/earlier_window_replay.jl` 等历史对照基准（S=300）原样保留；禁止作为 performance target / tuning objective / acceptance threshold。

---

## 7. Bars.bar 政策篡改的静态定位与改写方案

**检索方法**：全仓 grep 模式 `b_bar|\.bar\[[^\]]*\]\s*\.?=|bar\[1:[^\]]*\]\s*=` 与 `PONY|252|251|MINROWS`、`\.bar\[1:`。本次独立复核与 docs/MODEL_LEDGER.md §11 的结论一致。

### 7.1 直接篡改写入（D-019 点名形态 `b_bar[1:t_252-1, pony_j] .= false`）

**当前工作树零命中。** 该形态仅出现于 AGENTS.md:462（裁决书引用的示例代码）与 docs/MODEL_LEDGER.md:13（账本记录）。PONY 字符串唯一命中 universe.txt:46。

**结论**：篡改脚本的宿主不在本仓库可见位置（历史外部/会话脚本，或从未提交）。D-019 的工程意义因此是**防复发**而非清剿存量：Step 1 的 MarketFacts 类型把「政策写入 observed」在类型与测试上变为不可表达，是本裁决的落地形态。此结论与 MODEL_LEDGER §13「wrapper 脚本不在本仓库可见位置」的 [未知] 记录一致；若未来该脚本浮出（用户域或外部归档），按 7.3 方案重写。

### 7.2 现存 bar 语义混用面（D-012/D-016 的实际违规载体，file:line）

| 位置 | 现状 | 语义问题 | 改写方案（Step 1） |
|---|---|---|---|
| src/backtest.jl:242-245 | `active=copy(model.active_indices); free=falses(N); for j in active; free[j]=b.bar[t,j]` | free = active ∩ bar：观测 mask 被借用作「当日可交易」，且混入 model admission | 新线 backtest 从 Eligibility 取 free = trade_eligible[t,:] ∧ executable[t,:]；admission 只影响模型坐标空间 |
| src/data.jl:2-5 | docstring「bar[t,j] is true iff a real bar printed that day — i.e. the asset was tradable」 | 观测（O）与可交易（T）混写为同义 | MarketFacts.observed 的 docstring 只声明 O 语义；tradable 语义移交 Eligibility |
| src/kelly.jl:150-153 | `scenario_weights` 的 `free = tradable .& active` | tradable 由调用方传入（backtest 传的是 decision.free，即 active∩bar），admission 混入 free | gate0/kelly.jl 的 free 直接来自 Eligibility；admission 不再作为 Kelly 掩码输入（模型不预测的资产由 PredictiveLaw 覆盖域处理，非 free 语义） |
| src/broker.jl:136 | `tradable(q)=q.bid>0 && q.ask>0 && q.last>0` | 独立 T^exec 语义（报价级），与 bar 级 free 不互相校验 | 保留分层（docs/MODEL_LEDGER.md §5）；Eligibility.executable 在 backtest 简单形式 = O，live 层由 broker 语义供给 |

### 7.3 前缀复制（合法操作，非篡改——如实区分）

| 位置 | 操作 |
|---|---|
| dev/earlier_window_replay.jl:37、50；dev/release_failure_probe.jl:15；dev/release_freeze.jl:104 | `Bars(dates[1:t], ..., b.bar[1:t,:])`——构造**新** Bars 截断历史窗口，语义是「历史窗口选择」，不改写任何观测事实；保留不动 |

### 7.4 252 相关命中辨析

src/backtest.jl:348 的 `summarize(;periods=252)` 是年化常数（252 交易日/年），与 252 日可交易规则无关，不改。252 日规则的唯一合法表达位置是 Step 1 的 `trade_eligible` 构造规则（D-018，口径 H4）。

---

## 8. benchmark universe 外生化方案（D-020/D-072）

### 8.1 现状（违规证据链）

src/backtest.jl:241-245：`free` 由 `model.active_indices`（每决策日由 fit 产生的模型 admission）与 `b.bar` 决定；src/backtest.jl:295：`we = equal_weights_v1(free, he)`——**EW benchmark 的候选集依赖实验组的 active set**。实验组（Path Kelly 的 admission）决定对照组（benchmark）有哪些资产，违反 D-020。

### 8.2 方案

1. **候选集**：EW benchmark 的候选集合**只来自** `trade_eligible[t,:] ∧ executable[t,:]`（D-020），与 model admission 完全无关；Eligibility 构造纪律（§2.2：mask 构造只依赖 MarketFacts.observed 与用户政策）在类型上保证外生性。
2. **三 benchmark（D-072，Step 15；首日定义按 H8）**：
   - **Cash**：R=1（每期收益 0）；
   - **Daily equal-weight risky**：每日把 budget（= 1 − locked）均分给当日 free 集合（语义沿旧 `equal_weights_v1` 的 budget/locked 处理，src/backtest.jl:336-346，但候选集换外生 free）；
   - **Initial equal-weight buy-and-hold**：首决策日等权买入当日 free 集合，此后**不再重新分配**，仅按 gross 结转（现有 `he=(we.*gross_clean)./(1+ew[k])` 的结转公式即 buy-and-hold 持仓语义，去掉每日 `equal_weights_v1` 重分配即可）。**首日定义已裁决（H8）：首个决策日等权（E^trade∧T^exec 候选集）**。
3. **对照语义**：benchmark 2 vs 3 的差异才是 rebalanning premium 的可讨论对象（D-073：分解完成前禁称 Volatility Pump）；benchmark 1 提供 cash 基线（cash 入 Kelly 后的天然对照）。
4. **口径**：benchmark 与实验组共用同一 marking 序列（`adj`）与同一 Eligibility；不共用 model、不共用 scenarios、不共用 seed 流。

---

## 9. 测试升级阶梯与 synthetic worlds 设计

### 9.1 阶梯（D-089，全局纪律）

```text
static（Step 0/7：文档与推导审查——Step 7 推导已终审）
  → tiny analytic（Step 1-6/8-12 各自单元；N=2,3,4 小系统、固定 RNG、单线程）
  → single day（Step 13：真实几何 + 完整 law 的单决策日）
  → 5 days → 20 days → 60 days → 501 days（Step 14/15）
```

任何一级红立即停止升级（D-089）；501 之后到两年之间隔著 **Gate-0 Exit 十八项 + D-088 清单（放行条件已裁决 H9）**。

### 9.2 六个 synthetic worlds（D-090~D-095）

1. **D-090 Symmetric world**：N=3 资产 predictive law 完全 exchangeable。有 cash 且 risky expected log-growth > cash 时，risky 权重必须对称；risky law 无正优势时允许全 cash（Step 2 Kelly 侧 + Step 15 端到端侧）。
2. **D-091 No-signal world**：response posterior E[G]=0、innovation 对称。integration refinement 后 Kelly 必须收敛到对称 solution / cash，不得因 finite scenario 偶然赢家稳定产生 95% 单票（Step 14/15；与 Certificate A/B 直接挂钩）。
3. **D-092 Known cross-mode world**：人工 G_⊥m≠0 或 G_m⊥≠0；新 full response recover，旧 block-diagonal 故意失败（Step 4）。
4. **D-093 Known DC world**：人工 constant drift；DC channel recover，dynamic Q/P 系数保持 0（Step 5）。
5. **D-094 Trace-hypothesis counterexample**：真实 diagonal common response；unconstrained 新 core recover，旧 trace-neutral branch 系统性删除（Step 6）。
6. **D-095 Vector volatility world**：某 relative 方向近期波动显著上升；V_t 只放大该方向，旧 scalar law 全场放大（Step 10，旧线对照；R 域坐标按 D-045a）。

### 9.3 §34 OOF 测试（Step 9）

held-out row 不进 response posterior；不进 alpha quadrature evidence；fold residual 与 dense reference 一致；unified macro-relative cross blocks 严格 OOF；row with no observations 不得当 residual=0 进 likelihood。

### 9.4 §35 ragged 测试（Step 1/3/11）

(1) dummy all-NaN asset；(2) IPO；(3) 单日 missing；(4) held but unobservable；(5) current free set 是历史 subset；(6) no compatible joint residual shape rows（→ T4 判定次序，裁决 C3）；(7) 资产置换；(8) gauge 旋转。

### 9.5 §36 数值积分测试（Step 14）

known Gaussian 2-asset analytic Kelly；nested M→2M；optimization/audit 独立 replicate；same seed replay；不同 seed 收敛到同权重；预算耗尽 fail-loud；无 fixed-S production 捷径。

### 9.6 §37 cash Kelly 测试（Step 2）

Case A 全 risky 确定性 <1 ⇒ w_cash=1；Case B 单资产确定性 >1 ⇒ w_asset=1；Case C 高均值高风险内部解；Case D locked 的 cash+free+locked 财富正确相加。

---

## 10. 裁决记录与登记

### 10.1 开放决策处置（已裁决，裁决 H1~H9，2026-10-09）

原九项开放决策全部获 Manager 终审裁决（全文见 docs/GATE0_MANAGER_ADJUDICATIONS.md）：

| # | 原决策 | 裁决结果 |
|---|---|---|
| 1 | 新线模块形态 | **H1**：独立 module KTraderGate0 + src/gate0/ 目录线；旧 src/ 冻结；主 module 与新线互不 include；新线测试独立入口 |
| 2 | 新线文件切分粒度 | **H2**：按本文建议切分（§1.3 九文件） |
| 3 | 路线乙确认 | **H3**：路线乙确认 |
| 4 | 252 计数口径 | **H4**：累计第 252 个有效 bar 当日解除（当日 E^trade=1） |
| 5 | Step 7 推导文档命名 | **H5**：docs/GATE0_RESPONSE_POSTERIOR.md |
| 6 | D-084 experiments 副本 | **H6**：旧 solver 不复制（旧线冻结即保留） |
| 7 | 几何原语复用 | **H7**：纯函数复制到 src/gate0/ 并标注来源、不跨 module 依赖（§1.3 已联动修订） |
| 8 | buy-and-hold benchmark 首日 | **H8**：首个决策日等权（E^trade∧T^exec 候选集） |
| 9 | 501 后两年放行 | **H9**：Gate-0 Exit 十八项 + D-088 清单 |

### 10.2 推导级待裁项处置记录（裁决 G1）

- **T1~T6（GATE0_VECTOR_INNOVATION.md §11 张力清单）与 response §12 第 1 条（α 先验尾部修正）**：已裁决（裁决文本见 docs/GATE0_MANAGER_ADJUDICATIONS.md；其中 response §12 第 1 条即 D-035a：p(α)∝1/[α(1+α)]，裁决 B/G4）。
- **response §12 其余 6 条（第 2~7 条）**：按推导默认执行、标注「实现期可复审」——path group 划分（α_p 覆盖全体）、ragged 行 mode 分解权重语义（Step 3）、quadrature 输出对象审计字段、μ 通道 scenario 抽样语义（Step 8/13）、common response 措辞收敛、ε_quad 数值（与 T7 同走 D-066）。
- **T7**：留实现期（D-066 流程定稿）。

### 10.3 登记待办（Wave 2，非本轮 Step 0-16 范围）

- **README 状态声明补 D-042 顶层名称**：modular posterior predictive——旧线不得称 fully Bayesian generative posterior predictive、新线目标名称（归 Wave 2 文档轮）。
- **verify_release.jl 遗留问题登记**：已 defer；Wave 2 后由 DevOps 受控查验 bin/verify_release.jl 对改动后 RELEASE.toml/README 的行为。

---

## 附：证据索引（本次静态核对的关键锚点）

- src/data.jl:2-5（bar/tradable 混写 docstring）、:7-13（Bars）、:27-30（构造）、:37（signal_prices）
- src/backtest.jl:117-121（S=300/adaptive=false 默认）、:241-245（free=active∩bar）、:294（强制风险仓）、:295（EW 依赖 free）、:336-346（equal_weights_v1）、:348（periods=252）
- src/kelly.jl:21-37（证书）、:45（simplex 约束）、:137-148（locked_wealth）、:150-170（scenario_weights risky-only）
- src/prepare.jl:50-70（Fold/MacroStatistics）、:83-125（旧 PreparedProblem）、:131（alive_now 默认）、:145（P_features=14N）
- src/predict.jl:10-14（DGRID/DELTA_D）、:16-36（V1Model）、:43-70（embedded field）、:76-99（fractional）、:125-135（active_universe）、:286（ts_total）、:499（v_bootstrap）、:546-625（generate_scenarios：591 行 bootstrap、596-602 own-row、615 scale、620 loggross）、:627-653（adaptive）
- src/response.jl:2-16（path_basis）、:55-75（fill_design 列布局）、:106-143（constraint/conditioning）、:207-209（EB 域界）、:1124-1209（fit_response_operator：1130-1155 macro 独立 EB、1164-1185 relative、1199-1202 trace 条件化）、:1211-1229（predictive_moments）
- src/residual_oracle.jl:17-22（行身份契约）、:54-56（own rows）、:239（空观察行 e=0.0）
- src/geometry.jl:11-14（TAUS/BANDS/WARMUP）、:20（center_of_mass）、:26（ruler）
- src/KTrader.jl:14-25（include 顺序）
- dev/earlier_window_replay.jl:37,50、dev/release_failure_probe.jl:15、dev/release_freeze.jl:104（前缀复制，非篡改）
- universe.txt:46（PONY 唯一命中）
- docs/MODEL_LEDGER.md §4（E^trade 缺失）、§6（plug-in 现状）、§7（scalar innovation）、§9（fixed-S）、§11（PONY 静态推演）
- docs/POSTERIOR_DEFINITION.md §1-2（未知对象清单与 plug-in 结构）、§5（方差分解缺口）
- docs/INNOVATION_LAW.md §1（A1-A7 强假设）、§2（V_t 目标定义）
- docs/TRACE_NEUTRALITY_DERIVATION.md（candidate hypothesis 裁决、A.2 语义缺口）
- docs/NUMERICAL_INTEGRATION_SPEC.md §1（现状映射）、§3（证书层次）
- docs/OLD_TO_CURRENT_SEMANTIC_DIFF.md §0（旧侧基准）、§1（十对象 diff）
- docs/GATE0_RESPONSE_POSTERIOR.md（Step 7 纸面推导：§4 解析积分、§5 propriety/D-035a、§10 实现者检查清单）
- docs/GATE0_VECTOR_INNOVATION.md（innovation 层规范推导：§1.3 R 域/D-045a、§2 V_t/J_t、§5 shape 协议、§6 joint row、§11 张力清单 T1~T7）

*（本文件为 Gate-0 重开后的施工图交付物；2026-10-09 按 Manager 终审裁决（A/B2'/B/G4/C2/C3/G1~G8/H1~H9）修订。未运行任何命令，未修改 src/、test/、README.md、AGENTS.md 或任何既有文档。）*
