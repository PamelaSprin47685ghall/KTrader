# 真数据早期行（n<P）fit 失败——机理诊断设计（EARLY-ROW-FIT-1）

**状态：只读调查 + 诊断设计（2026-10-10）。未运行任何命令、未修改 `src/`、`test/` 或任何既有证据；本文为新增文档。**
**事件来源：devops 成本基线的转述——真实 P=911 下 n=256/2000 的 `fit_full_posterior` 在粗扫描某点即 cholesky 失败（被描述为"后验不正定"）。本文对该转述作机理候选分解，不宣称已复现。**
**证据基准：`src/gate0/posterior.jl`、`src/gate0/oof.jl`、`src/gate0/driver.jl` 当前工作树只读；行号为本次阅读时点。**
**关联裁决点：`docs/MULTIDAY_COST_OPTIMIZATION.md` §7 的 U1（F6 晚期窗口结构不可达——已裁决）、U4（warm start 的 reference equality 口径）；P0-3（创新满秩 gate）与 D-066（数值容差定稿）。**

---

## 0. 结论先行

1. **数学上 n<P 的 fit 是良定义的**：`Sxx + Λ` 的正定性由 Λ=diag(α)>0 保证（先验正则化），n<P 只意味着 Sxx 秩 ≤ n（后验宽、先验主导）——这不是"后验不正定"，而是**数值条件数**在小 α 端恶化后的分解失败候选。
2. **抛点有三个可能，且当前错误类型处置不同**：
   - `log_evidence` 的 `cholesky(Sxx+Λ)`（P×P，`posterior.jl:234`）；
   - `s_alpha` 内重复的 `cholesky(Sxx+Λ)`（P×P，`:118`）；
   - `logdet(cholesky(Sα))`（N×N，`:178` 与 `:237`）——**rank1 路径的最后一步仍是 N×N cholesky**。
3. **关键处置事实（静态）**：`_evidence_eval` 的 fail-soft 回退只捕获 `ErrorException` 且文本含 `"_log_evidence_rank1"`（`:189-193`）；`LinearAlgebra.PosDefException` 不满足该条件 → **rethrow**。同样，`oof.jl` 的行级 NaN 判定（`_row_undefined_error`，`:446-451`）只认 "posterior improper" 与 "Numerical integration did not converge" 两类文本 → **PosDefException 会穿透行级捕获，作为链级错误重抛**。因此"该行不可 fit"与"多日链崩溃"是两种不同后果，必须先分清。
4. **判定候选偏"数值治理可修"**：最可疑的是低 α 端（u≈−12，α≈6.1e-6）的 `S(α)=Syy−SxyᵀA⁻¹Sxy` 消差与 `A` 的谱；n=14044（n≥P）不失败与"n<P 结构性分界"一致。但**未复现前不下结论**；若诊断显示数值治理无法在不动数学的前提下消除失败，才升级为 SPEC 语义裁决（第 4 节判定树）。
5. **影响面**：若失败不可修且穿透为链级错误，真数据前段（P=911 已固定、train 行 <911 的约 t≤912 段）的多日验证与 U1 裁决的"以早期窗口承载"路径都受损；若可修，则该段恢复为正常（先验主导的）残差行。本设计给出最小探针与判定树，不给出修复实现。

---

## 1. 事件与影响面

### 1.1 事件（转述，未复核原始 log）

- 真实 P=911：`fit_full_posterior` 在 **n=256/2000**（train 行数）时，**粗扫描某点即 cholesky 失败**，被描述为"后验不正定"。
- 粗扫描形态（`posterior.jl:761-768`）：`(u0,up) ∈ [-12,12]²` 的 21×21 网格逐点 `logf = _ev(λ(u0,up)) + log_prior`；每点 `log_evidence` 含 **1 次 P×P cholesky（A）+ 1 次 P×P cholesky（s_alpha 内重复）+ 1 次 N×N cholesky（Sα 的 logdet）**，rank1=true 时 P×P 两项被 SM 路径替换，但 **N×N 的 `cholesky(Sα)` 仍在**（`:178`）。
- 无 rank1 时（`spec === nothing`）走原路径 `log_evidence`（`:186`），三个 cholesky 全量在。

### 1.2 为什么"早期行"必然大量 n<P

- prequential 每行 s 的 train = `X[1:s-1]`（`oof.jl:392`）；决策日的 `P = 1+14·N_act`。
- 若 N_act=65（P=911）在数据较早处已固定，则 **t≲912 的所有决策日其 prequential 行都 n=s−1<911**——即"真实 P=911 下 n=256/2000"正是这段的典型形态。
- 反之，n=14044 档（prequential 后段与 full-fit）满足 n>P，Sxx 满秩、条件数由数据主导——这解释了"正常档不失败"。

### 1.3 两种后果必须分开

| 后果 | 触发条件 | 严重度 |
|---|---|---|
| 该行残差 NaN（既有语义） | 抛错文本命中 `_row_undefined_error` 两类 | 低（行级，设计内） |
| **链级崩溃（rethrow）** | `PosDefException` 或其它未列入的异常 | 高（多日链直接失败，非行级） |

当前静态证据指向：cholesky 的 `PosDefException` 属于第二类。**诊断的第一目标就是确认实际抛出的异常类型与抛点**。

---

## 2. 静态机理候选（按可疑度排序）

### M1（最可疑）低 α 端的 S(α) 消差非正定 —— N×N cholesky

- `S(α) = Syy − Sxyᵀ(Sxx+Λ)⁻¹Sxy`（`:115-121`、RP §4.2）。n<P 时 `A⁻¹=(Sxx+Λ)⁻¹` 的谱在小 α 端被 1/α 放大（A 的最小特征值 ≈ α + 零特征值）；`SxyᵀA⁻¹Sxy` 量级增大，与 Syy 相减后的 `Sα` 在浮点上可能失去正定性（对称化 `Symmetric((Sraw+Sraw')/2)` 只治对称不治正定）。
- 与现象吻合：失败集中在**低 α 点**（u≈−12，α≈6.1e-6）；rank1 路径也在同一步（`:178`）失败；n≥P 时 A⁻¹ 良态、消差温和。
- 判定含义：偏数值治理候选（R1/R3，见第 5 节）。

### M2 Sxx 的数值谱与 A 的 cholesky —— P×P

- n<P ⇒ `Sxx=XᵀX` 秩 ≤256；`eigen(Symmetric(Sxx))`（`_rank1_spec:139-144`）给出 ~P−n 个数值零（可能微负 ~−ε·λmax）。
- 理论：`Sxx+Λ` 最小特征值 ≥ min(α)>0 ⇒ 正定。数值：若 α 小且 λ 数值负值的绝对值接近 α（需 λmax·ε ≳ α，即 λmax ≳ α/ε ≈ 6e-6/2e-16 ≈ 3e10——**通常不成立**），才会失败。**本项可疑度低于 M1**，但诊断须实测 λmin 与负值计数。
- 判定含义：若成立，数值治理（谱处理/域）候选。

### M3 rank1 路径的防护盲区

- `d = spec.lam .+ αp` 的 `all(>(0.0),d)` 与 SM denom 防护（`:155-156`、`:164-165`）抛 `ErrorException` 且文本含函数名 → `_evidence_eval` 正确回退原路径（`:189-193`）。
- 但 `:178` 的 `cholesky(Sα)` 抛 `PosDefException` → **不回退、不转译** → 外逸。这是处置逻辑的盲区候选（设计问题或需修复点），诊断须记录"回退是否发生"。

### M4 小 train 的退化列

- 早期资产/坐标可能产生零列或近似零列（mode features 零嵌入）；加 Λ 后不破坏正定，但放大条件数。属加重因子而非独立根因。

### M5 粗扫描的其它数值路径

- `log_prior_d035a` 与 evidence 的组合不会导致 cholesky；但 `exp`/`log` 的极值（u=±12）会影响 A 的尺度。归入 M1/M2 的对照扫描。

---

## 3. 最小诊断探针设计（供 DevOps；不改 src/test）

**输入构造**：固定真实 P=911 的 X/Y 前缀——`n ∈ {256, 2000, 14044}`（三个档位）；来源优先复用真数据 mode problem（`build_mode_problem` 输出）的前 n 行；若需独立脚本，按既有受控脚本模式放 `archive/evidence/...` 或临时目录。

**探针记录项（每档）**：

1. **失败定位**：粗扫描 21×21 逐点执行，记录第一个抛错点的 `(u0,up)`、`α0/αp`、是否 rank1（`spec !== nothing`）、是否发生过 fail-soft 回退、以及异常类型（`PosDefException` vs `ErrorException`）与完整栈；栈须落到 `:118` / `:234` / `:178` / `:237` 中的具体一处。
2. **矩阵数值**：`Sxx` 谱（λmin/λmax、负值个数、数值秩估计）；`A=Sxx+Λ` 在失败点的最小特征值估计（可用 `eigen` 小系统或 cholesky info）；`Sα` 对称化前后的特征值 min/max 与负值计数；`κ` 估计。
3. **α 扫描曲线**：固定 `up` 于粗扫描 mode 附近，沿 `u0∈[-12,12]` 与 `up∈[-12,12]` 各取 25 点，记录 `λmin(Sα)` 与 `λmin(A)` 曲线——确认失败是否只在低 α 端（M1）或另有热点。
4. **对照**：n=14044 同套量；rank1 on/off 两路径；`s_alpha` 直接调用 vs `log_evidence` 内路径（定位是否是 s_alpha 的重复分解暴露的问题）。
5. **NaN/异常传播**：将探针接到 `_row_undefined_error` 判定与 `prequential_residual_rows` 的行级路径上（只读诊断副本），确认实际抛错是"行级 NaN"还是"链级 rethrow"。

**护栏**：单命令 ≤60s/RSS 2048MiB；串行；不改 `src/test`；超时如实记录；不得为"跑通"放宽任何容差。

---

## 4. 判定树

```
失败点与异常类型？
├─ PosDefException @ Sα（:178/:237）
│    ├─ λmin(Sα) ≈ −c·ε·‖Sα‖（消差量级） → 数值治理候选（R1：数值 PSD 化 + floor→0 refinement；R3：等价重写）
│    └─ λmin(Sα) 显著负（非舍入量级） → 数学/实现审查（Sα 公式或输入）
├─ PosDefException @ A（:118/:234）
│    ├─ λmin(A) ≥ 0 理论（α>0）但数值 <0 → 数值治理候选（R2/R3）
│    └─ α 被构造为 0/负 → 代码缺陷（另立）
├─ ErrorException（SM 防护）且未回退 → 处置逻辑缺陷（M3）
└─ 全部档位（含 n≥P）同样失败 → 与 n<P 无因果关系，另查
```

**结论归属规则**：能在**不动数学对象/门禁/容差**的前提下通过数值治理（含 D-066 口径的数值配置）消除失败 → 判"数值治理可修"；若失败点对应先验质量非零区域且数值治理无法在不缩小积分域/不放宽语义的前提下消除 → 升级为 SPEC 语义裁决（例如 n<P 的数值合同定义）。

---

## 5. 修复候选登记（非本批实现）

| # | 候选 | 层 | 备注 |
|---|---|---|---|
| R1 | `Sα` 的数值 PSD 化（极小特征值 floor + floor→0 refinement） | 数值治理 | 必须与 `EB_COVARIANCE_FLOOR` 语义区分；禁止静默 clamp |
| R2 | α 域/粗扫描的数值配置（D-066 refinement 定稿） | 数值配置 | 不得由运行预算驱动；改变积分域须 tail 证书复核 |
| R3 | `Sα`/`A⁻¹Sxy` 的等价重写（如 Woodbury 对偶形式降低消差） | 数学加速（Gate-1） | 须先证明恒等 |
| R4 | n<P 行的语义定义（若数值治理不可达） | SPEC 裁决 | 影响 U1 的早期窗口承载路径 |

---

## 6. 与既有裁决点的关系

- **U1（F6 晚期窗口结构不可达——已裁决）**：本项不改变晚期 1.4e4 行的量级结论；但决定 U1 所述"多日验证以早期窗口承载"的**真数据前段是否可用**——若早期行失败不可修且穿透为链级错误，则前段验证同样受阻，需要上报重议承载窗口。
- **U4（warm start 的 reference equality 口径）**：warm start 可能引入更小的 α 起点，触发同类数值失败；E6 实验应增加本项探针的失败点检查（冷/暖两路径的 NaN 行集与异常集对照）。
- **P0-3（创新满秩 gate）**：若早期行被大量置 NaN，会缩减创新可用行；诊断应同时记录"修前/修后 NaN 行集"与 J 行覆盖，供满秩 gate 复核。
- **D-066**：任何数值域/floor 的定稿都必须走 refinement 证据，不得由回测或预算选择。

---

## 7. 边界

- 本文未运行任何命令；事件描述来自 devops 转述，机理候选为静态分析，**均未复现**。
- 第 2 节候选按可疑度排序，仅用于指导探针，不构成根因结论。
- 第 5 节修复候选为登记，实现须另立受控施工；数学对象、门禁与容差不在本设计的可改面内。

---

## 8. F1/F2 实测后的修复设计（R4' 前提被 F2 推翻；现行 R4'''，2026-10-10）

> **F2 修订（2026-10-10）**：R4' 的前提不成立——见本节末「§8.7 F2 修订与 R4'''」。

### 8.1 F1 事实（devops 实测，转述）

- **PosDefException(info=2) @ N×N `cholesky(Sα)`**（`posterior.jl:177`，rank1 路径最后一步）；首个失败点 = 网格远角 `(u0=−12, up=−12)`（α≈6.1e−6 双方）。
- **rank1=true 与 rank1=false 两条路径都穿透 fail-soft**——`_evidence_eval` 的文本捕获（`:189-193`）只认 `ErrorException` 且含 `"_log_evidence_rank1"`；`PosDefException` 不满足 → rethrow → 链级崩溃。
- 风险集中于**大 P 的小 train 行**（n<P 早期段）。
- 推论：M1（低 α 端 Sα 消差，本文件 §2）为主机理候选——与首个失败点在远角一致。

### 8.2 语义论证：−Inf 忽略的偏误量级与安全边界

**偏误判据**。设失败点相对峰值的 logf 落差为 Δ。把该点（及其 cell）置 −Inf 的积分偏误量级 ≤ C·exp(−Δ)（C 为常数/归一化因子）。要求偏误 ≪ tol=1e-6，即

> **Δ ≥ log(1/tol) + margin ≈ 13.8 + margin（建议实测门槛 ≥ 20）**。

**静态数量级估计（远角）**。`(u0,up)=(−12,−12)` 处 logf 的两个大项 `(N/2)log|Λ|` 与 `−(N/2)log|Sxx+Λ|` 的抵消残差含 `−(N/2)·log|I+Λ⁻¹Sxx|`，其量级为 O(N·P·|Δu|)（P=911、N=65、|Δu| 为与 mode 的 log 尺度差）——即 10⁴–10⁵ 量级，**远大于 20**。但该估计依赖 |Λ| 抵消结构，**精确值必须由 F1 的 α 扫描数据核对**（本设计不把该估计当结论）。若核对显示失败点 Δ < 门槛，或失败点落于中域，则改走 R4″（8.5）。

**−Inf 在各下游路径的安全性核对（当前代码静态行为）**：

| 路径 | 对 logf=−Inf 的行为 | 安全性 |
|---|---|---|
| `cell_val`/`cell_err`（`:446-478`） | `exp(−Inf−m)=0` → 该点贡献 0 | 安全（当 Δ 够大）；err 代理可能放大 → 细分隔离，fail-closed |
| m 基准 `cell_peak`（`:425-439`） | max 中 −Inf 不污染（除非全 −Inf） | 安全 |
| 粗扫描 m0/mp/mval（`:761-768`） | max 不受单点 −Inf 影响；**全 −Inf → mval=−Inf** → 后续 refval/判据红 | fail-loud 方向，安全 |
| tail 证书边界 8 点（`:403-405`） | −Inf ≤ RHS 恒成立 → **放行** | **危险方向**：数值失败 ≠ 真实低贡献；证书点失败必须 fail loudly，不得 −Inf 放过 |
| rel 估计（`:489`） | Z>0 检查 + 预算耗尽 fail loudly | 安全（fail-closed） |
| `log|S|` 的 reference 路径（`:237`） | 同点级捕获 | 与 rank1 一致 |

**结论（条件式）**：R4' 的语义成立条件 = (i) 失败点 Δ ≥ 20（F1 扫描核对）；(ii) 失败点仅出现在域边界/远角且占比小；(iii) tail 证书点与 refval 相关点**不得**走 −Inf（必须 fail loudly）。任一不满足 → R4″。

### 8.3 实现规格（R4'）

1. **捕获层（统一、按异常类型）**：在 `fit_full_posterior` 构造的 `logf` 闭包层（以及 `_evidence_eval`）捕获 `LinearAlgebra.PosDefException`：
   - 记录失败点 `(u0,up)`、路径（rank1/reference）、n/P、`λmin` 估计（可选）；
   - 返回该点 `logf = −Inf`；
   - `_evidence_eval` 现有文本捕获**升级为类型捕获**：`ErrorException` 含 `"_log_evidence_rank1"` → 回退 reference（现语义保留）；`PosDefException` → −Inf + 记录（新语义）。**不得**把其它非 PosDef 异常吞成 −Inf。
2. **覆盖所有 Sα/S 分解点**：`_log_evidence_rank1:177`、`log_evidence:237`、`s_alpha:118`（P×P 预防性）、`draw_mu:903`（抽样期——登记，不在 fit 修复面）。rank1/reference 两路径共用同一捕获层，保证行为一致。
3. **诊断载体**：`PosDefDiagnostics`（count、失败点列表上限 K、最大 |u−mode|、是否含中域、占比、路径）；作为 fit 的诊断字段/日志出口，**不进生产数学对象**（D-060 红线不涉及本项，但保持 fit 输出契约最小）。
4. **fail-loud 阈值**（D-066 数值配置，须 refinement 定稿，禁止由预算/回测选择）：
   - (a) **中域失败**：任一失败点满足 `|u_i − mode_i| ≤ dmid`（dmid 待定稿）→ error；
   - (b) **占比**：失败点数 / 求值点数 > ρ_max → error；
   - (c) **证书点失败**：tail 边界 8 点、refval 计算点中任一 PosDef → error；
   - (d) **全失败**：粗扫描/求积全 −Inf → error（沿用既有 fail-loud 文本族，D-067）。
5. **容差/门禁不变**：tol、tail_rel、max_cells、certificate 语义全部原样；−Inf 只影响数值偏误且受 8.2 条件约束。

### 8.4 测试设计（供 DevOps）

- **T1（早期行不穿透）**：n=2000、P=911 探针 fit —— 要么完成、要么按既有行级 NaN 语义（不得 PosDefException 穿透）；断言无异常外逸。
- **T2（极端角忽略的一致性）**：远角失败 fixture —— 忽略后 `alpha_nodes/weights/logZ` 与"该点用极大负值（如 −1e6）替代"的参照在 tol 内一致；合成已知解上偏误 ≪ tol。
- **T3（正常档零改变）**：n≥P 既有 fixture（无失败）逐字段不变；诊断计数为 0；不触发任何新路径（回归）。
- **T4（负控）**：合成中域失败（人为非正定 Sα）→ fail loudly；边界失败 + 高占比 → fail loudly。
- **T5（证书点防护）**：tail/refval 点 PosDef → fail loudly（不得 −Inf）。
- 命令与护栏沿用既有 scoped 规程（≤60s/RSS2048/串行；不改 src/test）。

### 8.5 备选

- **R4″（保守）**：`eigen(Symmetric(Sα))` + 负特征值 PSD 投影（floor→0）保留该点证据值。语义：Sα 数学上 PSD，投影属数值治理；须 refinement 证书（floor 减小结果稳定），并与 `EB_COVARIANCE_FLOOR` 语义区分。启用条件：8.2 条件 (i)/(ii) 不成立，或 R4' 的 fail-loud 在真实早期行上频繁触发。
- **R3（propriety gate 拦小 train）**：把 n<P 行判为不可定义（NaN）。语义反驳：n<P 的 fit 数学良定义（Λ≻0），用计算困难冒充语义边界会把真数据前段大量行砍成 NaN（影响创新行集与 P0-3 满秩 gate）；仅当证明 PosDef 失败数学上无法避免时才留 SPEC 裁决——当前证据不支持该前提。

### 8.6 与 U1/U4/D-066 的关系

- **U1**：本修复使真数据前段（t≲912 的 n<P 行）恢复可 fit，增强"多日验证以早期窗口承载"的可行性；不改变晚期 1.4e4 行的量级结论。
- **U4**：warm start 可能引入更小 α 起点 → E6 须对照冷/暖失败点集与 `PosDefDiagnostics`；本诊断字段即对照对象。
- **D-066**：dmid、ρ_max、Δ 门槛均属数值配置，须 refinement 定稿；不得以"能跑通"为目标放宽容差或改积分域语义。

*（本节为 F1 实测后的设计追加；未运行任何命令，未修改 `src/`、`test/` 或既有证据。）*

---

### 8.7 F2 修订与 R4'''（2026-10-10）

**F2 实测（决定性）**：n=2000、P=911 粗扫描 **441/441 点全失败、中域 121 点**；
eigmin(Sα) ∈ [−2.65e−12, −2.70e−13]（1e−13 量级、**数值噪声级**）；n=14044 不崩
（train 规模主导）。

**结论**：R4' 原前提（仅远角、小占比、Δ≥20 可忽略）**不成立**；R4' 实现停止。

**R4''' 设计（现行）**：

1. **核心**：Sα 的 logdet/正定性路径改走 eigen-based 鲁棒路径（N×N=65，
   便宜、确定性）：负特征值 λ<0 且 |λ|≤ε_num 时 floor 到 ε_floor；结构性负值
   （|λ| 超阈值）→ fail loudly（含诊断）。覆盖 cholesky(Sα) 的全部
   logdet/正定性使用点（rank1 :178 与 reference :237）——统一 helper
   _robust_logdet_psd；s_alpha 预防面（A 分解）由防御捕获处理。
2. **阈值（保守初始值，待 D-066 定稿；附 floor→0 refinement 要求）**：
   _PSD_FLOOR_REL = 1e-12（ε_floor，相对 λmax；scale=max(|λ|max,1)）；
   _PSD_NEG_TOL_REL = 1e-10（结构性负值判定：|λ| > 此值·scale ⇒ fail loudly）。
   F2 噪声量级 ~1e−12（绝对）相对阈值余量 ≥4 个数量级。
3. **保留防御层**：按类型捕获 PosDefException（A 等其它分解）→ 该点 −Inf +
   诊断（PosDefDiagnostics）；证书点/refval 点非有限 → fail loudly；中域/占比/
   全失败 → fail loudly；最终节点构造 PosDef → fail loudly。正常档零改变
   （eigen-logdet vs cholesky-logdet 差应 ~roundoff 级，验证核对）。
4. **数学对象不变**：Sα 理论 PSD；floor 是数值治理（参照 EB_COVARIANCE_FLOOR
   先例、语义独立），不得静默放宽；floor→0 时结果应在容差内收敛（D-066）。
5. **测试更新（test/gate0/posterior_tests.jl 追加 testset）**：正常 PSD 与
   cholesky 路径 roundoff 级一致；F2 量级负噪声 → floor 有限；结构性负值 →
   fail loudly；防御软包装/中域/占比/证书/refval/全失败负控；正常档零改变
   （诊断零失败 + 逐位重放）。n=2000/P=911 真实验证归 DevOps 探针。
6. **待验证清单**（交 DevOps）：(i) 早期行 n=2000/P=911 粗扫描 441 点不再失败、
   fit 完成或行级 NaN；(ii) 正常档 zero-change 回归（eigen 替换的数值差核对）；
   (iii) 结构性负值负控 fail loudly；(iv) floor→0 refinement（ε_floor 减半时
   下游权重/残差稳定）；(v) 全量 test/gate0 回归。

---

## 9. prequential 逐行 fit 的 active 集/P 语义核查（2026-10-10；只读）

### 9.1 代码事实（file:line）

- src/gate0/driver.jl:292：act = findall(el_t.model_admitted[t, :])——**决策日 t** 的 active 集；
- driver.jl:303-321：signal / r / mode problem 均取 act 列（t × N_act；build_mode_problem 入口）；
- driver.jl:375-376：X_tr = mp.X[rows, :]、Y_tr = mp.Y[rows, :]（rows = WARMUP:(t-1) 有观察行）；
- driver.jl:387：prequential_residual_rows(X_tr, Y_tr, E_active; …)；
- oof.jl:372：n, P = size(X)；oof.jl:392-393：X_tr = X[1:(s-1), :]、Y_tr = Y[1:(s-1), :]；oof.jl:398：fit_full_posterior(X_tr, Y_tr)。

**结论**：每行因果 fit 的列数 P/N **恒定 = 决策日 active 集**（P = 1+14·N_act），只有 train 行数 s−1 随行变化；**不存在"行 s 时刻 active 集"的动态重算**。"P 恒定 911"（当决策日 N_act=65）是代码构造事实。

### 9.2 A2 propriety gate 的触发链与 rank(Syy) 语义

- posterior.jl:728-730：gate 条件 n ≥ N 且 rank(Syy) == N；违反 → error（文本 "posterior improper"）。
- oof.jl:400-416：行级 catch；_row_undefined_error（oof.jl:446-451）只吞 "posterior improper" 与 "Numerical integration did not converge" → 该行 NaN；其余异常重抛。
- Y 行 = 各目标日 u_s 的 mode field（driver.jl:321 build_mode_problem；NaN 零嵌入 D-022 + observed mask 单独保留）。故 **rank(Syy) 由"目标日观察覆盖"决定**：若前 s−1 个目标日累计只张成 k 个方向，则 rank = k。
- 两个必要条件独立：n ≥ N（即 s−1 ≥ 65）先拦最前约 65 行；rank(Syy) = 65 另需观察覆盖满 65 个方向。

### 9.3 探针 rank(Syy)=1/3 的判定（条件式，未断案）

- "P 恒定"只决定列数（911），不决定 Y 的行结构——**rank = 1/3 不是 P 语义的产物**，只能由 Y 行内容解释。
- 两种候选：(a) 探针 Y = 真实 mp.Y 前 2000 行，且该段观察覆盖确实只有 1–3 维 → A2 拦截是**真实语义**（propriety 真条件），非 artifact；(b) 探针 Y 为人工构造（少列/少方向）→ artifact。
- **交叉证据**：F2 的"粗扫描 441/441 失败"探针已进入求积（A2 已过 → 其 Y 满足 rank=N）；本次"rank=1/3"探针停在 A2 之前。两者是不同构造。F2 的 Sα 结论不受影响（它证明的是 A2 通过后的行仍有 Sα 数值问题 → R4''' 仍必要）；但"n=2000 被拦"不能与 F2 失败混为一谈——它是**另一道门**（设计内的行级 NaN 语义）的表现。
- 区分所需的窄观察（归 DevOps）：真实 mp.Y[1:2000, :] 的 rank(Syy) 与各目标日观察集大小分布。

### 9.4 晚期窗口（t=14305）的量级估算

- rows ≈ 14049（f6 实数据探针：X 14049×911、Y 14049×65；该 full-fit 已过 A2 → rank(Syy)=65 在 14049 行上成立）。
- preq 拦截：前约 65 行因 n < N 必拦；rank(Syy) < 65 的附加区间长度取决于上市结构（panel 未读；量级可能数百行）。
- 若拦截约 300 行：有效残差约 13700 行（>97%）→ 创新历史充足；D-055/D-056 的 NaN 行过滤语义（driver.jl:397-402）不变。
- **U1 结论不变**：即便 Sα 问题全部清零，1.4 万行的逐行 fit 规模仍结构性超单命令预算。
- 早期窗口（501-day 等）：行总量小、拦截占比相对更高；R 域满秩所需行数（≤65）在 200+ 有效行下仍可满足，但需真实观察结构核对。

### 9.5 与既有裁决的关系

- D-036 propriety gate：不变（A2 拦截 = 设计内；被拦行 NaN）。
- D-055/D-056：NaN 行不进 V_t / z pool / 满秩 gate——语义不变。
- U1：不变（结构性量级结论）。
- R4'''：必要（F2 的 A2-通过后 Sα 问题）；本核查不改变它；但探针需统一 Y 构造（rank = 65）后才能复现目标问题。

### 9.6 待观察（DevOps）

1. 真实 mp.Y[1:2000, :] 的 rank(Syy) 与观察集增长曲线（判定 9.3 (a)/(b)）。
2. 真实数据第一个 rank(Syy)=65 的目标日 s*（决定 A2 拦截段长度）。
3. F2 探针的 Y 构造说明（其 Y 是否 rank=65）。

*（本节为只读核查；未运行任何命令，未修改 src/、test/ 或既有证据。）*
