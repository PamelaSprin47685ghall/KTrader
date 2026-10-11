# Kelly 输入数值可达性：逐行正尺度规范化设计（KELLY_NUMERICAL）

**状态：只读调查 + 设计文档。未运行任何命令、未修改 `src/`、`test/` 或既有证据。**
**证据基准：当前工作树 + `archive/evidence/gate0_innovation_attribution_20261010/`（B 节归因：summary.md 与 01_probe.log，hash 已由该批登记）+ 前序 `gate0_t327_fix_20261010`（A/B 判别）与 `gate0_t4_recovery_20261010`（场景 8 形态）。**
**上游事实（B 节）：场景 8 t=271 的极端值完全来自 μ 的 posterior draw（maxabs full=130.19，mu-only=130.12，eps-only=0.070）；within cov eigmax=2.03e6（主因子 `x'Vx`=23330 × `S_α` eigmax=22.65）；mean-only loggross=0.007。数学诚实、数值不可用（预测标准差 ~727 标准化单位）；kelly 侧证书驱动（A/B）已证不能解决。**

---

## 0. 结论先行

1. **推荐：逐行正尺度规范化（row scaling），作为纯数值求解器输入变换。** 对 `cash_kelly` 的每行场景（含 cash 列）除以确定性正尺度 `c_s = max(|base_s|, max_j |X_sj|)`（`X>0` 保证 `c_s>0`）。数学上是**精确恒等**：目标差常数 `C=(1/S)Σlog c_s`，梯度/Hessian/KKT 与 argmax 逐点不变；改变的是内点法中间量（每行 `wealth` 的自然尺度）的行间动态范围。**证书与对外报告全部在原始空间评估**，故证书口径、objective 语义、tie-break 语义零变化。
2. **必须钉死的实现陷阱**：cash 列不能保留 `ones(S)`——整行缩放要求 cash 系数同时变为 `1 ./ c`。否则数学被改变（等价性失效），绝不允许。
3. **样本层/风险域规则（含 winsorize 类）不在本方案内**：它们改变数学对象（可行集/样本/风险域），属 D-013/D-036/D-056 语义与 SPEC 级议题；且 B 节的宽尾是「诚实」的 posterior 产物，「是否应由 Kelly 消费宽尾」是决策理论问题，不应由数值修复顺带裁决。登记为独立议题（§6）。
4. **判别先行**：本方案是否解决场景 8，取决于实验（缩放是否让 Clarabel 在 1e-10 下收敛且原空间证书全绿）。文档给出 E1–E5 实验设计；在 E1 通过前不写生产规格的"已解决"结论。

---

## 1. 问题精确形态（证据）

场景 8 t=271（N_R=2，其中资产 2 locked）进入 `cash_kelly` 的输入：

- free 列 `X_free` = 32 场景 gross，`exp(±130)` 量级 → `1e-56 ~ 1e56`；
- locked base = `0.3 × exp(μ+ε)` 同样跨 ~1e±56；
- `log` 目标的逐行曲率 `1/wealth²` 与行系数尺度动态范围 ~1e112，Clarabel 在 `1e-10` 容差下 `SLOW_PROGRESS`（前序证据），且 kelly 侧 A/B（状态出口证书化、容差分级）已证不能解决。

B 节归因明确：这不是 bug、不是 ε 通道、不是均值问题；是 **μ posterior draw 的诚实宽尾**（within 协方差 `x'Vx` 外推不确定性）。因此数值修复的合法空间是：**不改变样本、不改变目标、不改变证书**，只改善求解器所见的条件数。

---

## 2. 候选对比

### 2.1 候选 A：逐行正尺度规范化（推荐）

**变换定义**（对 `cash_kelly` 求解装配的输入；`s = 1..S`）：
```
c_s = max( |b_s| , max_j |X_sj| )        # X 全部正有限 ⇒ c_s > 0 有限
X̃   = X ./ c                             # 行缩放（Julia 广播：每行除以 c[s]）
b̃   = b ./ c
cash_col = 1.0 ./ c                      # 关键：cash 列同缩，不得保留 ones(S)
增广装配：Ã = [ X̃   cash_col ]，wealth̃ = Ã·w_aug + b̃
```

**正确性论证（本文独立推导）**：
- `wealth̃_s(w) = (X_s·w + w_c + b_s)/c_s = wealth_s(w)/c_s`；
- `J̃(w) = (1/S)Σ log wealth̃_s = J(w) − C`，`C = (1/S)Σ log c_s`（常数）；
- `argmax J̃ = argmax J`（常数偏移）；`∇J̃ = (1/S)Σ Ã_s/wealth̃_s = (1/S)Σ A_s/wealth_s = ∇J` 逐分量恒等；`∇²J̃ = ∇²J`（逐项 `A_sA_sᵀ/wealth²` 中 `c_s` 两次相消）；
- KKT 条件（`w≥0`、`Σw=budget`、互补性）在 `(J̃, ã约束)` 与 `(J, a约束)` 间逐点对应 ⇒ **证书三指标 feasibility/kkt/gap 数学不变**；`objective` 差常数 `C`。
- 因此：**任何最终解都必须在原始空间证书下验收**（`cash_kelly_certificate(X, w, w_c; base=b, ...)` 原样调用），报告 objective 也来自原空间——不存在"还原"歧义。缩放只服务求解器与（可选）polish 的中间数值。

**对病态的改善与它的边界（诚实声明）**：
- 改善的是**行间**尺度：缩放后每行最大系数为 1，exp-cone 各行 `y_s` 的自然量级从跨 `1e112` 收敛到 `O(1)` 邻域——这是内点法 KKT 缩放病态的主要来源之一。
- **不改善**行内动态范围（同一行内两资产系数比）、也不改变样本宽尾本身；若 Clarabel 的停滞还含其它机制（例如 exp-cone 建模对 `1/c_s` 列尺度异质敏感），缩放可能只部分有效——由 E1 判别。

**影响面**：`kelly_cash_tests`（77，数值路径变化需重跑；负控必须保持）、场景 2（回归；其最终解质量由原空间证书兜底，最坏=现状）、场景 8（E1 判别）、driver/quadrature（签名与调用不变）、报告与 tie-break（见 §3.4）。

### 2.2 候选 B：列尺度 + 行尺度组合

列缩放 `X → X·D`（`w → D⁻¹w`）可进一步均衡列尺度，但改变约束几何（`Σ D⁻¹w̃ = budget`），需要重写约束并在原空间复验最优解映射；复杂度显著上升。**不首选**；仅当 E1 显示行缩放对列尺度部分无效时再评估。

### 2.3 候选 C：样本层/风险域规则（不推荐作为本批修复）

- 形态：`x'Vx` 或预测标准差超阈值资产的降级/截断/winsorize（B 节 V3 诊断）。
- 论证：这改变**数学对象**（样本测度、可行集或风险域），违反"不改数学对象"；且与 D-013（model admission prefix 语义）、D-036（fold propriety）、D-056（free 收缩出路）、D-017（locked 不可当 cash）纠缠，任何一处改动都需 SPEC 级审查。
- 「宽尾是否有意义」是决策理论问题（诚实不确定性是否应被 Kelly 消费、locked 资产如何处理），**不应由数值修复顺带裁决**。登记为独立议题（§6），本设计不实施、不建议其数值形态。

---

## 3. 推荐方案与精确实现规格（候选 A）

### 3.1 施加点

`cash_kelly` 入口、`cash_kelly_inputs` 校验通过之后（`src/gate0/kelly.jl:265-284` 区间），在构造 `Variable`/`X_aug` 之前。**只改求解输入，不改任何证书函数。**

### 3.2 公式与代码结构（规格草案，待 E1 后定稿）

```julia
# --- 行尺度规范化（纯数值；数学恒等，设计文档 KELLY_NUMERICAL §2.1） ---
c = Vector{Float64}(undef, S)
for s in 1:S
    m = abs(b[s])
    for j in 1:n
        a = abs(X[s, j]); a > m && (m = a)
    end
    c[s] = m                      # X>0 ⇒ m>0；无需 floor（防守断言 c[s] > 0）
end
Xt = X ./ c                       # 行缩放
bt = b ./ c
cash_col = 1.0 ./ c               # 关键：cash 列同缩（不得 ones(S)）
```
- 主求解：`X_aug = hcat(Xt, cash_col)`；`wealth = X_aug * w_aug + bt`。
- 主求解容差保持现状（`min(tol/10, 1e-10)`、max_iter=1000、status 白名单）——本方案不触碰前序 A/B 已否定的容差路线。
- Newton polish：可在缩放空间执行（更好的 KKT 条件数）；**采纳条件不变**：精修解必须通过**原始空间**证书。
- **证书与报告**：`cert = cash_kelly_certificate(X, w[1:n], w[n+1]; base=b, budget=budget)`（原始 `X/b`）→ `objective`、`dual`、三指标全部原空间语义，零偏移。
- tie-break：默认保持在原始空间（现状 `_cash_kelly_canonical_tiebreak(X, b, ...)`）；若 E1 显示 tie-break 在场景 8 也停滞，备选为"缩放空间求解 + 原空间复验"，作为后续增量（不在本批）。

### 3.3 不变量检查（设计自检）

| 不变量 | 结论 |
|---|---|
| 数学对象（目标/可行集/证书） | 不变（argmax/∇/∇²/KKT 逐点恒等；证书原空间评估） |
| fail-loud | 不变（所有 error 文本与条件保留） |
| 确定性 | 不变（`c` 是输入的确定性函数；无 RNG） |
| 无新依赖 | 满足（仅 `max`/逐元素除法） |
| cash 列同缩 | 规格钉死；实现审查项 |
| 1D 无关性 | kelly 与 1D 求积无共享代码，互不影响 |
| `kelly_tol`/`tie_eps` 语义 | 不变（证书按原始 `tol` 验收） |

### 3.4 报告字段

- 无强制新字段。建议在实验期（非生产）加诊断：`c` 的 `minimum`/`maximum`/`categorical span` 与缩放前后 `status`/迭代数，用于 E1 归因；若证明有用，再按 D-076 报告口径登记。

---

## 4. 实验设计（供 DevOps；不改生产源码，探针/实验副本）

- **E1（场景 8 判别，核心）**：从 `gate0_t4_recovery_20261010`/`gate0_innovation_attribution_20261010` 的探针方法拦截 t=271 的 `(X_free, base, budget)`；在实验副本上对比：
  - 缩放前：复现 `SLOW_PROGRESS`（基线）；
  - 缩放后：status、迭代数、原空间证书三指标、polish 后证书、返回 `w` 与 `objective`。
  - 判据：原空间证书全绿（feasibility/kkt/gap ≤ 1e-6）→ 方案有效；仍红 → 记录停滞位置（是否仍是 SLOW_PROGRESS、kkt 残差量级），升级裁决。
- **E2（场景 2 回归）**：两日链（`28_diag_bt2_twoday` 同款）+ 单日扫描（`27`）：缩放前后最终解与证书对照；要求 kkt 仍 ≤1e-6（现状 8.37e-8），并记录缩放是否顺带缓解"ktol=1e-10 档 23.6s"的耗时。
- **E3（模块回归）**：`kelly_cash_tests` 77/77 + 负控（证书红必须红）；`driver_tests` 相关断言。
- **E4（位级/容差对照）**：对已通过案例，缩放解 vs 现状解：`‖w̃−w‖₁`、证书差、`objective` 差（应为 O(1e-12) 级或更好；若有系统性差异超出证书容差，按 bug 处理）。
- **E5（确定性）**：同输入同 seed 两次运行逐位一致；`c` 的计算与平台无关（基本算术）。
- 护栏：≤60s/RSS 2048/串行；超时如实记录，不得放宽 tol。

---

## 5. 待 Manager 裁决点

1. 是否批准"先 E1（只读探针）"；若 E1 绿，批准方案 A 的实现规格与 E2–E5 回归。
2. E1 红时的升级路径：是否评估列尺度组合（候选 B）或转向样本层议题（候选 C / SPEC）。
3. 是否接受"证书与报告全部原空间"的口径（本设计推荐；若要求在缩放空间评估 kkt 以获取更小数值，需另行论证，不建议）。
4. tie-break 是否在第二批同步进入缩放空间（本批建议保持原空间）。

---

## 6. 独立议题登记（不在本设计内）

**「诚实宽尾的后验预测是否应由 Kelly 原样消费」**——B 节的 within 宽尾来自 15 行样本/极短历史资产的真实不确定性；把它降级/截断改变数学对象（D-013/D-036/D-056 域）。该议题的裁决需要 SPEC 级审查（例如：模型准入的统计资格、风险域规则、或承认该 fixture 场景 8 在现有数学下不可解）。本设计只提供数值可达性修复，不替该议题作决定。

---

## 7. 层归属

逐行正尺度规范化属于**数值工程层**：无限算力下该变换消失（求解器直接解原问题）；它不改变数学对象、不改变证书、不改变样本。所有接受/拒绝仍由原始空间 `cash_kelly_certificate` 判定；fail-loud 保持；不含任何由回测收益选择的参数（`c_s` 是输入的确定性泛函）。

---

*本文件为只读调查 + 设计文档；未运行任何命令、未修改 `src/`、`test/`、`archive/` 既有文件。E1–E5 为设计草案，执行归 DevOps 受控流程。*

---

## 8. 证书驱动的缩放重试（KELLY-SCALE-RETRY-1；2026-10-10，G2 实证后设计）

**状态：只读 + 设计（本节）；未运行、未改 src/test。证据：archive/evidence/gate0_multiday_run_20261010/（G2_summary.md、G2_offline.log、G1_inputs.bin 的 cap#5 输入）。**

### 8.1 G2 事实（决定性）

| | cap #4（512 层） | cap #5（1024 层 = 失败点） |
|---|---|---|
| span | 2.01e8（阈值下侧） | **3.03e9（阈值下侧）** |
| 生产 cash_kelly | OK（11.02s） | **证书失败 kkt=1.9557e-6** |
| 原路径（RAW） | OPTIMAL，kkt=2.05e-10 ✓ | **ALMOST_OPTIMAL，kkt=1.9557e-6**（与生产失败值逐位一致）✗ |
| RAW + polish | kkt=2.03e-10 ✓ | kkt=1.92e-6 **仍不过** |
| 缩放路径（SCL） | OPTIMAL，kkt=9.93e-10 ✓ | **OPTIMAL，kkt=5.475e-9 ✓**（过 1e-6） |
| SCL + polish | kkt=9.84e-10 ✓ | kkt=5.38e-9 ✓ |

**判定**：存在中间窗口 [~1e8, 4.5e15)——原路径质量已不足、条件缩放尚未启用；缩放路径在同一输入上给出合格解。polish 无法补足（根因在原路径解质量）。上轮"中间地带无证据"的登记现已被 G2 推翻。

### 8.2 方案对比与论证

**(a) 下调阈值（否）**：把浮点结构边界（1/eps ≈ 4.5e15）换成经验阈值——任何新阈值下方仍存在下一段中间窗口（G2 本身证明：2e8 过、3e9 败，中间窗口连续存在）；且阈值邻域出现路径归属漂移（span 跨阈值 → 解路径变化；D-091 平局形态路径敏感性教训）。治标、无原则、不封口。

**(b) 恒双路径（否）**：每次求解 ×2；两解都过证书时需引入"选哪条"的新规则（解差 ~1e-9 级），无原则；对绝大多数已绿案例纯增成本。

**(c) 证书驱动的缩放重试（采纳）**：
- 触发条件是**证书质量本身**（对真实失败模式的直接响应），不是输入尺度的猜测；
- **阈值不动**：既有已缩放/未缩放案例的路径归属零漂移（D-091 等已绿锚不变）；
- 原路径通过时零变化（绝大多数）；仅失败时一次额外求解（罕见；G2 场景为秒级量级）；
- 方向性论证：G2 证明同输入下缩放路径质量严格更优（5.475e-9 vs 1.9557e-6）；"原→缩放"方向有实证支撑；反向（缩放 fail → 原路径）无证据，不做；
- **解一致性**：原路径 fail、缩放 pass 意味着缩放解是唯一合格解（不存在二选一）；两路径都过证书的情形（测试双跑对照）解差 ~1e-9 级（批次 4 D 节证据，转引）；
- **确定性**：cs 为输入的确定泛函（§2.1）；触发分支由证书值（确定性）决定；无 RNG；同输入重放逐位一致；
- **性能**：额外成本 = 失败时一次求解；成功路径零开销。

### 8.3 实现规格（cash_kelly 重试结构）

**当前插入点（src/gate0/kelly.jl 现状）**：首选装配（:332-340 阈值分支）→ solve（:341-358）→ status 白名单（:359-360）→ 后处理（:365-368）→ 原空间证书（:370-371）→ **polish 块（:372-386）** → 最终 fail（:390-391）→ tie-break（:399-407）。

**新结构（重试在证书 fail 之后、polish 之前）**：

1. 提取内部 helper _cash_kelly_solve(Xt, bt, cash_col, budget, tol) -> Union{Nothing,Vector{Float64}}：装配（Variable/hcat/wealth）→ 容差（min(tol/10,1e-10)）→ max_iter=1000 → solve → **status 非白名单返回 nothing**（由调用方决定语义）→ clip+预算归一。只重构、不改数值语义。
2. 首选求解：helper 返回 nothing → error("Clarabel cash log-Kelly failed: ...")（**保持现状**：首选 status fail 不重试——G2 形态是 status 过而证书 fail；status-fail 属另一议题，见 §8.5）。
3. 首选证书 fail 且首选为**未缩放**路径（span ≤ 阈值）→ 用缩放装配（cs 已算好）再调用 helper 一次：
   - 返回 nothing → 放弃重试（继续 polish 链）；
   - 成功 → 原空间证书（cash_kelly_certificate(X, ...)，原 X/b）→ certified 则采纳（rescued）并替换 cert。
4. 仍有证书 fail → 进入现有 polish 块（不变、原空间）→ 最终 fail（不变）。
5. 首选为**已缩放**路径（span > 阈值）且证书 fail → 不重试（无更优替代）→ polish → fail（现状）。

**建议提取的纯决策 helper（可单测）**：_cash_kelly_needs_retry(preferred_scaled::Bool, cert, tol) = !preferred_scaled && !cash_kelly_certified(cert; tol)。

**诊断**：新增 kw scale_retry_diag::Union{Nothing,KellyScaleDiagnostics} = nothing（不导出；mutable 对象就地填充；与 R4''' 的 posdef_diag 同模式）。字段：preferred::Symbol（:scaled/:unscaled）、span::Float64、attempted::Bool、rescued::Bool。默认 nothing → 零行为变化、零返回契约变化。

**不变量表**：

| 不变量 | 结论 |
|---|---|
| 数学对象（目标/可行集/证书） | 不变（两路径恒等；证书原空间验收） |
| kelly_tol / 门禁 | 不放松（全部 tol 透传；certified 用同一 tol） |
| fail-loud | 保持（首选 status fail / 最终证书 fail 文本与路径不变） |
| 确定性 | 不变（无 RNG；分支由证书值决定） |
| tie-break | 不变（消费已认证主解；重试只改变被认证的解来源） |
| polish | 不变（原空间、采纳条件不变） |
| 条件缩放阈值 | **不动**（1/eps 分支保持） |
| 返回契约 | 不变（三元组；诊断经可选 kw） |

**明确不做**：status 非白名单的出口证书化（KELLY_ADAPTIVE 历史议题，另论）；tie-break 进缩放空间（KELLY_NUMERICAL §5-4 遗留）；polish 进缩放空间（可选后续，须单独证据）。

### 8.4 测试设计（test/gate0/kelly_cash_tests.jl 追加）

- **T1a（纯逻辑真值表）**：_cash_kelly_needs_retry 四组合：(unscaled, fail)→true；(unscaled, pass)→false；(scaled, fail)→false；(scaled, pass)→false。便宜、确定。
- **T1b（零回归）**：既有通过案例（场景 2 档、D-091 平局、Case A–D）带 diag kw 运行 → 解与不带 kw 时逐位一致、attempted=false。
- **T1c（已缩放不重试）**：极端 span 案例（场景 8 同型，span>阈值）→ preferred=:scaled、attempted=false。
- **T1d（G2 同型集成，受控/登记）**：将 G1_inputs.bin 的 cap#5 输入固化为受控 fixture（DevOps 生成；fixture 化决定归 owner）→ 断言：原路径 fail 被重试救回、原空间证书全绿、rescued=true、kkt ≤ 1e-6。这是"中间地带"的真实边界 witness；未 fixture 化前，T1a+T1b/T1c 承载机制语义。
- **T2（负控）**：重试也 fail 的输入 → 最终 error（不得假绿）；由 T1a 与既有"证书红必红"负控承载；若可构造双败 fixture 则加集成负控。
- **T3（确定性）**：同输入两次运行（带 diag）逐位一致 + diag 相同。
- **T4（诊断）**：kw 填充字段；不传时无副作用。
- 护栏：≤60s/RSS2048、串行；不放松断言/容差。

### 8.5 与既有裁决/文档的关系

- 本重试是 §2.1 行缩放恒等的"条件启用"补丁：阈值语义不动、证书口径不动；G2 待裁决点（下调阈值 vs 重试）——本设计选重试并给出论证。
- 与 KELLY_ADAPTIVE：本设计**不复活** status 出口证书化路线（那条被实测否定）；只处理"status 过、证书 fail"。
- 与 D-091：平局形态不受影响（其 span 温和、走原路径且通过，重试不触发）。
- 与 D-066：本设计不引入新的数值参数（无新阈值；cs 与触发均为输入的确定泛函）。

### 8.6 未决与待观察

1. 实现与 T1a–T4 为源码交付；运行（含 T1d 集成与 kkt 实测）归 DevOps 受控流程。
2. T1d 的 fixture 化决定（owner/DevOps）。
3. 重试与 polish 的先后顺序：本设计按 Manager 方向（重试先于 polish）；顺序不影响正确性，若实验显示 polish 先行更省/更稳可重评。
4. "首选已缩放仍 fail → 再试别的路径"的扩展：无证据，登记不做。

---

## 9. K-S-R-1 重试取优（设计 A）与 polish 精度边界评估（B）（2026-10-10，E5b 后）

**状态：只读 + 设计（本节）；未运行、未改 src/test。证据：archive/evidence/gate0_multiday_run_20261010/（E5b）。**

### 9.1 E5b 事实

- **cap#3**（S=256、span=7.015e6）：RAW OPTIMAL、kkt=1.9812e-6；RAW+polish kkt=**1.0779e-6**（仍 >1e-6，差约 7.8%）；解 w=[0.499,0.501]。
- **SCL**：SLOW_PROGRESS、kkt=1.0137、**远解** w=[0.949,0.050]——缩放重试在该输入上负改善；非平局（解差 0.45）。

### 9.2 现状定位（先精确，不夸大）

- 当前采纳规则：仅当缩放解通过**原空间证书**才采纳（未过证书的解从不被采纳）——cap#3 的远解**不会**被采纳；现状在该输入上最终 fail-loud（原解 polish 后仍不过）。
- 现存缺口（设计 A 的靶）：**当两路径都过证书时**，现状是「先到先得」——缩放解一旦过证书即采纳，**跳过对原解+polish 的比较**，可能丢弃数值更优候选；且最终失败的错误文本只反映单路径证书质量，双路径诊断缺失。cap#3 是一个「差点命中该模式」的输入（SCL 只差 status 一步），设计 A 把它结构性封口。

### 9.3 设计 A 规格（重试取优——正确性修复）

1. **候选收集**：raw 必入；scaled 仅当 raw 未过证书、未缩放首选、且 scaled 求解 status 过时入集。
2. **对各未过证书的候选尝试 polish**（原空间、_cash_kelly_newton_polish、同一门槛；未过者才做，最多两次）。
3. **统一比较键（钉死、确定性）**：
   - 档 1：是否过证书（过者优先）；
   - 档 2：证书质量 q = max(feasibility, kkt_residual, objective_gap) 小者优先；
   - 档 3：q 差小于 1e-15 视为 tie → 路径偏好 **raw > scaled**（固定序）。
   - objective 不进比较键：两候选同为原空间、同一 scenario 集，kkt 残差是更直接的数值质量；避免浮点噪声阈值。
4. **采纳较优者**；若较优者未过证书 → fail loud（D-067 文本族前缀不变），错误文本含**双路径状态**（raw/scaled 的 status、kkt、gap、polish 结果、较优者标注）。
5. **诊断扩展**：KellyScaleDiagnostics 增加 adopted::Symbol（:raw/:scaled/:raw_polished/:scaled_polished）、raw_quality::Float64、scaled_quality::Float64（无候选为 Inf）；保留 preferred/span/attempted/rescued（rescued 语义 = adopted 属 scaled 系）。默认构造同步更新。
6. **不变量**：门槛不变（只有过证书者可被采纳——「取优」只决定谁被采纳/如何诊断，**绝不**放宽验收）；fail-loud 保持；确定性（比较键无 RNG）；tie-break 不变（消费已认证的最终解）；条件缩放阈值不动；对「raw 直接过」的输入**零变化**（不触发任何新路径）。
7. **与现状的行为差异**（预期仅两处）：
   - G2 cap#5：raw+polish 1.92e-6 不过、scaled 5.475e-9 过 → 取优=scaled（**不回退**，与现状一致）；
   - cap#3：scaled 无候选（SLOW_PROGRESS）→ raw 唯一候选、fail-loud，诊断含双路径事实（**不再负改善、不再片面**）；
   - 唯一可能改变既有数值的形态：「raw 未过但 raw+polish 过且优于 scaled」——现状会采 scaled、设计 A 采 raw+polish（修复丢优）。

### 9.4 测试设计（设计 A）

- **T-A1（比较键纯函数）**：把 _pick_better_candidate(ca, cb) 提炼为纯 helper，四组合真值表（过/不过 × 质量大/小）+ tie 路径偏好；确定性、无 RNG。
- **T-A2（G2 cap#5 集成）**：adopted=:scaled、rescued=true、原空间证书绿（不回退）。
- **T-A3（cap#3 集成）**：fail-loud；错误文本含双路径诊断（raw 1.0779e-6、scaled unavailable）；**远解不被采纳**。
- **T-A4（零回归）**：kelly_cash 全量（Case A–D、D-091、场景 2/8、tie）逐位不变（raw 过者不触发新路径）；确定性两跑一致。
- **T-A5（两候选都过时的取优）**：受控构造（mock 或 fixture）验证「raw_polished 优于 scaled 时采纳 raw」。
- 护栏：不放松任何容差；≤60s/RSS2048；串行。

### 9.5 评估 B（polish 8% 差距——不放松门禁下的改进路径）

问题：cap#3 的 polish 后 kkt=1.0779e-6，超出 1e-6 约 7.8%。候选路径与量化评估：

- **(B1) 更多 Newton 轮**：若 10 轮耗尽 → 可能有效；若已到 best_res 平台（步长/支撑限制）→ 无效。**需 P1 探针**记录 polish 轨迹（每轮 res、break 原因）。
- **(B2) 缩放空间 polish**：在 (Xt,bt) 上跑同一 Newton（行缩放恒等、KKT 条件数更好），解映射回原空间、**原空间证书验收**。KELLY_NUMERICAL §3.2 已登记的可选增量。**P2 对照**：kkt 是否 ≤1e-6；成本（同量级）；风险低（同一数学）。
- **(B3) 支撑敏感性（最可疑）**：当前 polish 固定 Clarabel 解的支撑（findall(>(1e-6), w)）；若真最优接近支撑边界（某分量在 1e-6 级），固定支撑的 KKT 系统可能不含真驻点 → 残差平台在 ~1e-6。候选修法：polish 后未过时**从 polish 解重启一次**（支撑更新）；或支撑边界探测（成本高、须谨慎）。**P3 实验**：把支撑判定 1e-6 换 1e-8/1e-10 观察 kkt 轨迹（仅诊断，不改生产）。
- **(B4) 更高精度内部容差/更长主迭代**：主求解容差已至 min(tol/10,1e-10) 上限；收益有限，登记。
- **判定树**：P1–P4 任一使原空间 kkt ≤1e-6 且不放松门禁 → 实施（另立受控施工）；**全失败 → 登记「该输入类在 1e-6 门禁下的数值精度边界」为 SPEC 面**（与 KELLY_ADAPTIVE 的 tol 讨论、D-066 流程衔接），不硬修、不静默。

### 9.6 与 D-091 / 条件缩放 / E5b 的关系

- D-091：设计 A 的比较键为纯数值 + 固定路径 tie——不引入 seed/路径漂移；平局形态不受影响（raw 过 → 零变化）。
- 条件缩放：阈值与启用规则不动；设计 A 只改「重试采纳/诊断」层。
- E5b/cap#3：中间地带的新形态（原路径数值上优于缩放）；设计 A 封口为「取优 + 完整诊断」。
- 实施须由 Manager/SPEC 裁决（设计 A 改变 K-S-R-1 采纳规则；本节只给设计）。

### 9.7 边界

- 只读；未运行；全部行为差异为静态推导；cap#3/cap#5 数字转引 E5b；比较键的 1e-15 tie 阈值属数值配置（待 D-066 定稿）。

### 9.8 P1–P4 判决与 polish 轮数修复（2026-10-10，已实施）

- 根因（P1 轨迹，archive/evidence/gate0_multiday_run_20261010/P1_P4_*）：cap#3 的 polish 停滞 = max_iters=10 上限（约 0.93/轮线性收敛）；50 轮 → 7.2e-8（过证书）、200 轮 → 2.8e-12（机器精度）。B3 支撑敏感性证伪（四起点全过）；缩放路径劣（P2）；主求解紧容差（P4）为备选。
- 实施形式：max_iters 默认 10 → 200，stationarity（1e-13）早退保留——正常用例零额外成本（典型几步退出；每轮 k 小、微秒级），病态用例获得所需深度；证书/门禁/确定性/失败语义均不变。
- 测试锚：cap#5 内联输入上的 10 vs 200 轮结构对照（单调、确定性、默认值语义）；cap#3 强锚（kkt 1.98e-6 → ≤1e-6 救回）归验证批次用 E5b_cap3.bin 复跑。

### 9.9 tie-break 成本诊断与优化设计（2026-10-10，段 B 热点）

**背景**：段 B 两次切在 _cash_kelly_canonical_tiebreak（kelly.jl:656）；完成态压窗口边（CQ 探针同参数 40.8s 曾通过；段 B 多加载/构建几秒越线）。tie-break（P0-9 两阶段）在 65536/131072 大层疑似 5-15s 热点。

#### 9.9.1 现状精确读数

- cash_kelly 的调用点：if tie_eps > 0 后**无条件**调用 _cash_kelly_canonical_tiebreak——**没有无 tie 预检、没有快速路径**；每次 cash_kelly（默认 tie_eps=1e-8）都构建并求解两个额外 Clarabel 问题（Stage 2 max cash、Stage 3 min risky L2），场景规模与主问题同为 S×n。
- Stage 3 的输出再经原空间证书复验（cert3；不过则返回 nothing、保留主解）——语义闭环已有；成本是两次大求解。

#### 9.9.2 无 tie 快速路径的语义分析（关键，先堵死错路）

- **通用「等价集为空→跳过」不可行**：对严格凹的 U，ε-超水平集 W_ε = {U ≥ U*−ε} 对任何 ε>0 都有体积；canonicalization 的输出偏移为 O(√ε) 级（既有 Case C 注释：√ε_tie ≈ 1.4e-4）。这是 P0-9 的**预期行为**（在 ε-集上重选规范代表），任何输入都适用——预检跳过会改变输出，违反「语义不动」。
- **精确快速路径存在但窄**：若主解已全 cash（w_risky ≈ 0）——cash=budget 已是 ε-集内最大、risky L2=0 已是最小 → canonicalization 恒等 → 跳过两次求解不改变任何输出（精确）。这是可实现的零风险快速路径。
- **t=330 是否有真 tie：无法从「被切在 tie-break」推断**——当前实现对任何输入都执行该段；诊断须记录（主解是否全 cash、零列 KKT 间隙分布、Stage 2/3 的迭代数与耗时）才能定论。

#### 9.9.3 候选

- **(a) 全 cash 精确快速路径（推荐，先做）**：主解 risky 权重全部小于 1e-12 → 直接返回主解（canonicalization 恒等，解析证明见上）。实现面：一处条件；数学逐点等价；确定性不变。覆盖 D-090(b)/D-091 的 canonical 端点情形（输出一致）与深段可能的全 cash 日。
- **(b) 活跃集低维化（主候选，收益大、需论证）**：ε-集内的可移动方向受 KKT 间隙控制——主解支撑列 ∪ 零列中 g_i > ν − margin 者为活跃列；其余远列（移动上界 ε/(ν−g_i) 低于数值分辨率）固定为 0。在活跃列子问题（小规模 Clarabel）上解 Stage 2/3；U 的评估仍全场景（O(S·活跃列)）；最终仍过 cert3（原空间、tol 不变）。收益取决于活跃集大小（若深段解稀疏/近全 cash → 活跃集小 → 显著）。margin 与「低于分辨率」的判据属数值配置（待 D-066 定稿，须 refinement 与反例测试）。
- **(c) 解析判据/合并**：与 a/b 融合；「通用跳过」已证不可行（9.9.2）。主解热启（Clarabel 无稳定接口）不采用；内部精度放宽会移动规范代表——不采用（语义不动）。

#### 9.9.4 规格草图与测试设计

1. (a)：cash_kelly 的 tie-break 调用前插入「主解全 cash」判据；跳过时诊断/返回与 canonicalization 恒等情形逐位一致。
2. (b)：构造活跃列（含 margin 判据）、小问题组装（变量 = 活跃列 + cash）、全场景 U 评估、cert3 复验、失败回退全量两阶段（保留现状路径为 reference）。
3. 测试：D-090(b)/D-091（canonical 端点仍为全 cash；a 跳过与现状输出逐位一致）；Case C（非全 cash → 不跳过，行为不变）；确定性；段 B 复跑判据：A 逐位复现 6.2585e-5、耗时 ≤45s；回归（kelly 全量）。
4. 待验证：诊断先行——记录 tie-break 耗时分解与主解/间隙数据，判定热点真实构成后再定 (b) 的收益与 margin。

#### 9.9.5 与 P0-9 / 设计 A / M2 的关系

- P0-9：canonicalization 语义完全不动；本设计只消除「恒等情形」的求解成本与（b）等价缩减，不改变任何输入的规范代表。
- 设计 A/M2：正交（那是采纳/候选层；这是 final 解的规范化层）。
- 边界：只读；未运行；margin/跳过阈值属数值配置（D-066）。

#### 9.9.6 诊断登记（探针侧，实施可后置）

段 B 复跑时记录：tie-break 是否进入（全 cash 判据 all_cash）、Stage 2/3 各自耗时与迭代数、主解 cash 程度（Σw_risky）与零列 KKT 间隙分布——用于判定 (b) 活跃集低维化的收益与 margin。TIE-FAST-1 已实施：全 cash 精确快速路径（1e-12 安全界）。

### 9.10 b2b 性能修复：活跃集 vs cash_kelly 拆分（2026-10-10，六段窗口失守后设计）

**背景**：b2b（纯 kelly 131072：X2 加载 + cash_kelly）逐日上升——330:37s → 331:45s（重试后）→ 332:>48s×4（稳定出窗）。b2b 内部：主求解（Clarabel 131072 场景）+ polish（200 轮上限）+ tie-break（两阶段、§9.9 已诊断无预检无条件跑）。

#### 9.10.1 候选对比

- **(a) §9.9 (b) 活跃集低维化**：把 tie-break 的两阶段化为小规模等价问题（主支撑 ∪ 近 KKT 间隙零列；margin 待 D-066）。等价性：ε-集内远列固定为 0 的 KKT 间隙论证。实现面：_cash_kelly_canonical_tiebreak 内部重构（kelly.jl）。收益：tie-break 从 131072 场景两次大解压到小解（10-15s → 秒级，若活跃集小）。风险：margin 判据的正确性（需反例测试与 refinement）。
- **(b) cash_kelly 内部拆分（接口设计）**：把尾部提炼为两个可调用阶段——_cash_kelly_main（从头到「最终证书检查之后、tie-break 之前」的全部逻辑，产出 w_val/cert/u_star/skip_tb）与 _cash_kelly_tiebreak_apply（tie-break 调用 + 采纳逻辑；skip_tb 时返回主解）；cash_kelly = 两者组合（逐点一致）。批量工具可将 b2b 再拆两段：b2b1（X2 → main → 中间产物落盘）、b2b2（中间产物 → tiebreak_apply → seg2）。收益：把 kelly 切成「主求解段 + tie-break 段」，各自独立窗口；**前提是主求解段 ≤48s**（Clarabel 调用不可再拆）。风险：与 NODE/JOINT/NEST 的求解敏感**无关**——拆分不改任何送入 Clarabel 的输入（同 X、同 cash_kelly 逻辑），只是把同一计算安排在不同进程；无新数学路径、求解行为逐位一致（调度级变化）。
- **(c) a+b 组合（推荐）**：a 把 tie-break 压到秒级、b 把它分出去；b2b1（主求解）与 b2b2（小 tie-break）各自进窗。若主求解 ≤45s 则 t=332 闭合；若 45-48s 则边缘；>48s 则 b 不够（见 d 登记）。
- **(d) 其它（登记）**：polish 若在病态日跑满 200 轮需专项诊断（轮数/耗时）；主求解的 CS 层（Convex 建模开销、MOI 直连）属 D-082/D-083 reference 定位之外/需裁决；主求解本身**不可拆**（Clarabel 是单一调用）——若其 >48s，剩余选项是批量环境放宽（长跑授权，owner 域），不是生产数学变更。

#### 9.10.2 诊断先行（b2b 分段计时）

t=332 的 >48s 组成未测：主求解 / polish / tie-break 三者占比决定 a/b 的先后与是否足够。记录点：X2 加载、主求解（至 cert）、polish（轮数与耗时）、tie-break（Stage2/3 各自）、总时长——探针侧实施。

#### 9.10.3 拆分规格（b）

1. _cash_kelly_main(X; base, budget, tol, tie_eps, scale_retry_diag) -> (; w_val, cert, u_star, skip_tb)：现 cash_kelly 至「最终证书检查之后」的全部逻辑（含候选/polish/status-exit，均不导出）；失败语义与现函数一致。
2. _cash_kelly_tiebreak_apply(X, b, budget, w_val, u_star, skip_tb, tie_eps, tol) -> (; w_risky, w_cash, cert)：现 tie-break 调用 + 采纳逻辑逐字复刻；skip_tb 时返回主解。
3. cash_kelly = main + tiebreak_apply 组合；公共签名/语义/确定性不变。批量侧：b2b1 落盘中间产物（w_val/cert/u_star/skip_tb 的 NamedTuple），b2b2 从 prep 重算 b、读 X2 与中间产物、调 tiebreak_apply、落 seg2——产物与现 b2b 的 seg2 逐点一致。

#### 9.10.4 测试设计与待验证

- 拆分等价性（核心）：对 Case C、D-091、cap#5 与真实 X2 fixture（若可得），cash_kelly 与 main+tiebreak_apply 组合的 w/cert 全字段逐点一致（skip_tb 两分支都覆盖）。
- 批量对拍：b2b1/b2b2 的 seg2 与 b2b 的 seg2 逐点一致；b3 照旧。
- 回归：kelly 全量。
- 判据：t=332 的 b2b 或其拆分版进窗；333+ 推进恢复；A/B/C 与既有对拍一致（FG2 的 332 未有 A，以「拆分前后逐点一致」为对拍基准）。

#### 9.10.5 边界

- 只读；未运行；全部收益为分析与外推；margin/分段阈值属数值配置（D-066）。

#### 9.10.6 B2B-SPLIT-2 与 tie-break 零产出登记（2026-10-10）

- S4 分解（t=338）：raw_polish 0.7s、scaled_solve 16.7s、tie_break 24.1s（**失败返回 nothing**）、其余 ~8.5s。
- 已实施 B2B-SPLIT-2：_cash_kelly_finish 再拆为 _cash_kelly_finish_main（候选构建段，至 tie-break 前）与 _cash_kelly_finish_tiebreak（tie-break + 终检）；批量 b2b2 改产 mid 产物、新增 b2b3 消费 mid 产 seg2。
- **待 SPEC 审视（不在本轮实施）**：tie-break 在深场景「24.1s 零产出」（失败返回 nothing 仍耗 24s）——门控候选（如需先验上界/降级判据/预算门）属数值配置层的新裁决；本条只登记现象与候选，不预判。

### 9.11 tie-break 范围/门控：SPEC 提案（2026-10-10，未收敛日加深的前提）

**证据汇总**：S4 分解（t=338）tie-break 24.1s 且返回 nothing（零产出）；deep@262144 出窗（S5_t336d_b2b3{,b}.log：124×2、50s+ 切）；336/340 未收敛（A=1.355e-4 / 1.283e-4，超门 35.5% / 28%），其加深（max=262144）被 tie-break@262144 阻断。关键语义事实：_cash_kelly_finish_tiebreak 的失败（返回 nothing）**保留主解**——即「跑 tie-break 失败」与「跳过 tie-break」在输出上相同。

**提案（给 SPEC/owner 的选项，不预判）**：

- **(a) 保持现状**：无条件跑 tie-break。语义影响：无。证据：现行行为。实现面：零。与加深的关系：**阻断**（262144 的 tie-break 出窗）。
- **(b) 范围门控**（如场景数 ≤ 阈值才做 tie-break）：语义影响——对「总会失败」的输入（深场景、S4 型）**零行为差异**（跳过=失败=保留主解）；对「本会成功」的输入**改变输出**（P0-9 规范化缺失）——**是语义变更，需 SPEC 裁决**。实现面：小（finish_tiebreak 一处条件）。与加深的关系：**可解锁**（262144 场景直接跳过 tie-break；其输出退化为主解——与 tie-break 失败同果）。
- **(c) 快速失败/迭代上限**（tie-break 求解器设置）：语义影响——迭代上限使提前停止→cert3 不过→保留主解，与失败同果；风险面与 (b) 相同（某些「本会成功」输入变失败）。**但**：失败若发生在迭代内，仍要跑满上限才停——除非上限极小，否则不解决「24s 零产出」；收益有限。实现面：小。登记。
- **(d) 活跃集低维化（§9.9(b)）**：语义影响——数学等价目标（小规模问题求同一 canonical 点）；「本会成功」输入给出同一输出、「本会失败」输入同样失败——**无门控语义风险**。实现面：中（_cash_kelly_canonical_tiebreak 重构 + margin 判据，待 D-066 + 反例测试）。与加深的关系：**可解锁**（tie-break 化为毫秒-秒级，262144 场景不再是大解）。

**推荐与关系（不预判）**：

- **(d) 是首选**（无语义裁决需求、直接解锁加深、对既有输出零改变）；(b) 是低成本应急（若 SPEC 接受「深场景跳过规范化」的语义变更）；(c) 收益有限、登记；(a) 为基线。
- 与 P0-9 的关系：canonicalization 的语义定义不动是 (d) 的前提；(b) 是对该定义的例外条款——两者性质不同。
- 与 SPLIT-2 的关系：tie-break 已被隔离到独立段（b2b3）——本提案决定的是「该段跑不跑/怎么跑」，与调度层正交。
- 解锁判据：未收敛日的加深验证路径 = (b) 或 (d) 落地后，tie-break@262144 不再占用窗口预算。

### 9.12 TIE-ACTIVE-1：tie-break 活跃集低维化——设计与可行性再分析（2026-10-10）

**目标（§9.11 选项 d）**：tie-break 24.1s 零产出（@338）、deep@262144 出窗（阻断未收敛日加深）——把两阶段化为小规模等价问题，无门控语义风险。

#### 9.12.1 设计（活跃集构造与两阶段小问题）

1. 主解 KKT 梯度：g = vcat(X' * (1 ./ wealth), [sum(1 ./ wealth)]) ./ S（wealth = X*w_risky + w_cash + 0；ν = max(g)）。
2. 活跃列 = 主解正支撑 ∪ 零列中 gap_i = ν − g_i < margin 者；cash 列恒活跃；其余远列固定为 0。
3. ε-集内小规模 Stage2（max cash）/Stage3（min risky L2）：变量 = 活跃列 + cash，同目标同约束；U 的评估仍**全场景**（O(S·活跃列)）；输出仍过原空间 cert3。
4. 全量两阶段保留为 fallback / 对拍基准（活跃集不适用或小问题失败时回退）。

#### 9.12.2 可行性再分析（本轮的关键发现）

- 逐字段一致（测试要求）的可达性依赖 margin 的强度：远列在 ε-集内的可移动量一阶界 t ≤ ε/gap。
- **精确界**（要求 t < 数值分辨率，如 1e-12）：margin = ε/1e-12 = 1e4 量级——而实际 gap 分布量级为 1e-3 到 1，**几乎所有零列都会被判为活跃** → 活跃集=全列 → **无收益**。
- **宽松界**（t ≤ 1e-5 视为可忽略）：产生 O(1e-5) 的输出偏移——**是近似而非等价**，需 SPEC 认知（风险面小于选项 b，但仍非零）。
- **结论**：d 的收益与等价强度**取决于 t=338 等深场景的实际 gap 分布**——这是一个尚未测量的量；在缺它之前，任何实现的收益都是未证的，且强等价与收益之间存在真实张力。

#### 9.12.3 实现规格（供施工，待前置诊断）

1. 前置诊断（第一步）：记录深场景（t=338 型）主解的零列 gap 分布（分位）——判定活跃集规模与 margin 可行区间。
2. _cash_kelly_canonical_tiebreak 内加活跃集路径：构造活跃集 → 若规模显著小于全列且精确判据（按定稿 margin）成立 → 小问题；否则全量两阶段。
3. 命名常量 _TIE_ACTIVE_MOVE_EPS（待 D-066）；fallback 与对拍基准保留。
4. 测试：D-090/D-091/Case C 的活跃集路径 == 全量路径逐字段；margin 边界反例；回归。

#### 9.12.4 状态与关系

- **本轮交付：设计 + 规格 + 可行性分析；实现未做**——因为实现的正确性参数（margin）与收益前提（gap 分布）均未定稿/未测，草率实现会以近似污染生产数值路径。
- 与 §9.11(d)：本节的发现收窄了 d 的适用条件（gap 分布决定收益），并给出了施工前的第一步（诊断）。
- 与 SPLIT-2：调度层已把 tie-break 隔离；本节决定其内部算法是否可低维化——两者正交。
- 待验证：gap 分布诊断；t=338 tie-break 24.1s 的改善空间；deep@262144 进窗；未收敛日加深复验。

### 9.13 未收敛日处理：最终决策包（2026-10-10，综合 §9.11/§9.12/S10）

**证据链汇总**：S10 诊断（本链 n=1——只有 1 个 risky 列 + cash，**无零列** → 活跃集=全列 → §9.12 选项 d 无收益）；S4（tie-break 24.1s 零产出 @338）；deep@262144 出窗（S5_t336d_b2b3{,b}：124×2）；未收敛日序列（336/340/343；十四日三超、约 20-25%，门在 ~78% 分位）；SPLIT 链调度稳定（各段独立进窗）。

**决策包（给 owner/SPEC；不预判）**：

| 路径 | 当前状态 | 证据 | 语义影响 | 实现面 | 后果 |
|---|---|---|---|---|---|
| **b 范围门控** | 唯一「解锁加深」的短路径 | S4（深场景 tie-break 零产出）；S5（262144 出窗） | **语义例外**：只影响「tie-break 本会成功」的输入（丢失 P0-9 规范化）；对「总失败」的深场景零差异 | 小（finish_tiebreak 一处条件） | 加深可跑；**加深的预期效果无直接证据**（被阻断至今）——需 b 落地后实测 |
| **c D-066 复核** | 待定稿 | weight_tol=1e-4；深段噪声底与门同量级 | **规范层**定稿程序（不得由回测选择） | 程序性 | 给「门」一个依据；不直接解锁加深 |
| **d 活跃集** | **已否定** | S10：本链无零列 | — | — | 若未来 n>1 输入出现可复议 |
| **其它：μ 通道方差缩减** | 登记 | 已到 Sobol 现状（节点/z/chisq 的 QMC 化已做尽） | 无 | 未知 | 无近路 |

**推荐与顺序（不预判）**：

- 三条路径性质不同：b = 语义例外（需 SPEC 裁决）、c = 规范定稿（程序）、其它 = 工程登记。
- 顺序建议：c 与 b 可并行推进——c 为门给出定稿依据，b 为加深给出通路；b 落地后先实测「加深是否真的改善未收敛日」（预期效果至今无直接证据），再决定其长期化。
- **最终裁决留 owner/SPEC**；本决策包不实施任何路径。

**引用链**：§9.11（b/c/d 选项与推荐）→ §9.12（d 的可行性再分析）→ 本节（S10 对 d 的否定与最终整理）。

**边界**：只读；未运行；全部为选项分析与证据转引；实施须 SPEC 裁决（b）或独立施工（d）。

### 9.9 补充：M2 出口证书化的语境分离（2026-10-10）

kelly adjudicate 批次（archive/evidence/gate0_kelly_adjudicate_20261010/summary.md）对**场景 8（t=271、S=32、极端 base 1e±56）**的判决是：**M2 否定**——7 档 Clarabel 容差全部 SLOW_PROGRESS、返回解逐位相同（cert kkt=gap=1.35），polish 仅到 0.416；E2 模拟的「方案 A（SLOW_PROGRESS 出口证书化）」在该输入上**不过**。主根因是样本极端动态范围（V3 winsorize 后 kkt=1.88e-7 过——修复方向在样本/尺度层）。因此：**出口证书化框架不能承诺救回场景 8**；它的价值与边界见 §10。

---

## 10. SLOW_PROGRESS 出口证书化（M2 框架）——设计（2026-10-10）

**状态：只读 + 设计；未运行、未改 src/test。**

### 10.1 背景与语境

- 触发：S=512 层的 kelly 求解 status 非白名单（SLOW_PROGRESS）时，_cash_kelly_solve 走 status 白名单路径直接判失败，解从未获得证书验收的机会。
- 原则：证书是外部数学验收（原空间 feasibility/kkt/gap），status 是求解器内部收敛指示。KELLY_ADAPTIVE 文档的判读仍然成立：「让 status 退回诊断、让证书当门槛」是把验收权放回它该在的地方。
- **与 §9.9 的关系（必须分开）**：adjudicate 批次证明了「场景 8 的出口解不可救」（M2 否定、具体输入）；本设计是**通用框架**，不预设任一输入可救——其验收完全由证书决定，最坏情形=现状（fail-loud），最好情形=救回「status 非白名单但解已达证书质量」的输入。

### 10.2 代码现状（file 级）

- _cash_kelly_solve（kelly.jl）：solve 后 status in (OPTIMAL, ALMOST_OPTIMAL) 检查不通过时 return (; w_val=nothing, status_ok=false, reason=:status, status=...)——status 非白名单**不提取解**；total > 0 检查不通过时 return reason=:degenerate。
- cash_kelly：首选 res.status_ok == false → 按 reason 走既有 error（Clarabel …failed / degenerate）；设计 A 候选集（p0 首选 → 未过 polish → need_scl 缩放重试 + polish → fold 取优 → 双路径 fail-loud）只覆盖「status 过、证书 fail」形态。

### 10.3 设计：出口证书化

1. **出口解提取（仅数值层）**：_cash_kelly_solve 在 status 非白名单时，尝试提取当前迭代解：try max.(vec(evaluate(w_aug)), 0.0) catch；有限且 sum > 0 → 归一（budget）后作为**出口候选**返回（reason=:status_exit，携带 status）；提取失败/非有限/sum≤0 → 维持既有 return（reason=:status，无解）。degenerate 分支不动。
2. **候选融合（与设计 A 统一）**：首选出口候选进入候选集（路径标签沿用装配：raw 出口 → :raw 系；scaled 出口 → :scaled 系），未过证书则跑同一 polish（:raw_polished / :scaled_polished）；need_scl 判断照旧（首选未过且未缩放 → 允许缩放重试；缩放重试的 status 出口同理入集）。取优规则、比较键、tie 序全部不变。
3. **采纳**：任一候选（含出口系）过原空间证书且质量最优 → 采纳；诊断标注 adopted_from_exit::Bool（新增字段）。
4. **fail-loud（不过时）**：出口候选未过（含 polish 后）→ error，**保留既有前缀** Clarabel cash log-Kelly failed: $status，追加：exit candidate not certified: adopted=... raw_quality=... scaled_quality=...（诊断可读）。**无出口解**时保持原文（仅前缀句）。degenerate 分支文本不变。
5. **不变量**：门槛/容差一格不放宽（出口解与 OPTIMAL 解用同一证书验收）；确定性（提取与比较无 RNG）；对 status 白名单路径零变化（分支不进）；tie-break 不变；fail-loud 保持。
6. **诊断**：KellyScaleDiagnostics 增加 adopted_from_exit::Bool（默认 false）；exit 的 status 记入错误文本（字段不爆炸）。

### 10.4 成本与风险

- 成本：仅 status 非白名单时多一次出口提取 + 一次 polish（k 小、微秒-毫秒级）；场景 8 类输入每层多一次 polish（现路径直接 error、无 polish）。可接受。
- 风险：采纳的解虽过证书但来自非白名单迭代——证书已含 KKT/gap 验收（凹问题近驻点即近全局最优），与 OPTIMAL 解的验收标准同一；残余风险=证书本身的分辨力（与既有采纳逻辑同一假设）。
- 风险（诚实）：对场景 8 类极端输入无改善（§9.9）；对 S=512 的形态未测——由并行诊断数据定。

### 10.5 测试设计

- T-M2a（结构）：把出口提取提炼为纯 helper（或对 _cash_kelly_solve 的白名单内路径做零变化回归）；出口分支的集成锚依赖 S=512 输入（并行诊断提供），登记。
- T-M2b（负控）：无有限解（提取失败）→ 原 error 文本（仅前缀句）；degenerate → 原文本。
- T-M2c（确定性）：出口路径两次同输出；候选比较无 RNG。
- T-M2d（零回归）：既有 135 项（kelly_cash 口径）全绿；status 白名单路径逐位不变。
- T-M2e（集成，待数据）：S=512 输入复跑——若出口解（含 polish）过证书则采纳（adopted_from_exit=true、证书全绿、诊断链完整）；不过则 fail-loud（文本含双路径 + status）。

### 10.6 与 M1/M2、设计 A、D-091 的关系

- M1/M2（KELLY_ADAPTIVE）：本设计是方案 A 的形式化与通用化；M1（极端尺度）输入不因此改变（§9.9 的 M2 否定保持）。
- 设计 A：本设计只是候选集的**来源扩展**（status 出口解），取优/采纳/fail-loud 规则完全复用设计 A，不新建第二套验收逻辑。
- D-091：平局形态 status 白名单内、零影响。

### 10.7 边界

- 只读；未运行；形态与收益为静态推导与转引；S=512 的判别数据由并行诊断提供后复核。实现须另立受控施工。
