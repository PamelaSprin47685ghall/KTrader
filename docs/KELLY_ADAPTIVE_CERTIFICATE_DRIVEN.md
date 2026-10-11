# Kelly 证书驱动自适应：调查报告与设计（KELLY_ADAPTIVE）

**状态：只读调查 + 设计文档。未运行任何命令、未修改 `src/`、`test/` 或任何既有证据。**
**证据基准：当前工作树（HEAD 994fb708 时点）+ `archive/evidence/gate0_t327_fix_20261010/`（27/28 logs）、`archive/evidence/gate0_t4_recovery_20261010/`（场景 8 探针）、`archive/evidence/gate0_merged_verify_20261010/`（合并验证）。**
**上游裁定：`gate0_t327_fix_20261010/summary.md` §6-2——静态容差公式路线被实测否定；若解耦恒定 `1e-10`，需「证书驱动自适应」（先宽后紧、证书闭环）。**
**边界：本文件不宣布场景 8 一定可解；若判别实验证明其在该证书门槛下数值不可达，如实升级为 SPEC 级裁决项。**

---

## 0. 结论先行

1. 当前 `cash_kelly` 的阻塞结构是**状态白名单**而非证书：主求解容差被恒钉 `1e-10`（`src/gate0/kelly.jl:297-300`），`SLOW_PROGRESS` 在证书与 Newton polish 有机会运行之前直接 error（`:304-305`）。证书（`:315-316`）与 polish（`:317-331`）只服务于「status 已通过」的解。
2. 判别证据支持「两条腿」设计：
   - 场景 2（t=271）证明：**最终解质量由解本身决定，不由 status 决定**——solver 1e-10 → kkt 8.37e-8（过 1e-6）；solver ≈1e-9 → kkt 2.17e-6（不过）。证书才是真门槛（`28_diag_bt2_twoday.log`）。
   - 场景 8（t=271，新几何）证明：**status 卡死时当前直接 error，证书完全没有机会说话**；且其输入含极端尺度（free 列 gross `1.57e-9 ~ 4.73e9`；locked base 所在列 `2.26e-55 ~ 3.46e56`，`merged_verify/summary.md` §B2）。
3. 推荐路线（待判别后定稿）：**方案 A（SLOW_PROGRESS 出口证书化）为最小主干**——把「status 白名单」改为「有有限解 → 证书（含 polish）验收 → 过则接受、不过则原样 fail-loud」；**方案 D（问题数值尺度诊断/等价预处理）为条件候选**，仅当判别实验显示极端尺度是场景 8 的主因时启用。方案 B（宽→紧分级闭环）是 A 的超集，成本更高，作为 A 不够时的升级路径。
4. 全部方案保持：证书门槛不放松、fail-loud 保持、确定性、无新依赖、数学对象不变（同一 log-Kelly 目标与同一 `cash_kelly_certificate`）。
5. **判别实验必须先做**：场景 8 的 `SLOW_PROGRESS` 解是否可用、polish 后 kkt 能否 ≤1e-6，是 A 成败的唯一判据；在此未知前不写生产规格。

---

## 1. 现状精确读数（file:line）

`cash_kelly(X; base, budget, tol, tie_eps)`（`src/gate0/kelly.jl:265-355`）：

1. 输入校验（`:269`，`cash_kelly_inputs` `:71-83`：正有限 gross、非负有限 base）。
2. 退化 `budget == 0` 分支（`:277-284`）。
3. 主求解装配：`Variable(n+1)` + `X_aug = hcat(X, ones(S))`（`:286-288`）；`max (1/S)Σlog(wealth) s.t. w_aug ≥ 0, Σw_aug = budget`（`:289-290`）。
4. **容差设置**：`for setting in ("tol_gap_abs","tol_gap_rel","tol_feas"); MOI.set(..., min(tol/10, 1e-10))`（`:297-300`）——对任何 `tol ≥ 1e-9` 恒等于 `1e-10`（比 Clarabel 默认 1e-8 严 100 倍）。
5. `max_iter=1000`（`:301-302`）；`solve!`（`:303`）。
6. **状态白名单（当前直接阻塞点）**：`problem.status in (OPTIMAL, ALMOST_OPTIMAL) || error("Clarabel cash log-Kelly failed: $(problem.status)")`（`:304-305`）。`SLOW_PROGRESS` 在此被拒。
7. 后处理（`:307-313`：负尘埃截零 + 预算恢复）。
8. 证书：`cash_kelly_certificate`（`src/gate0/kelly.jl:102-126`：feasibility `:108`、kkt `:117-123`、gap `:120`）；`cash_kelly_certified`（`:129-130`）。
9. **Newton polish**：条件为「status 已通过 ∧ 证书红」（`:317-321`）；实现 `_cash_kelly_newton_polish`（`:147-211`）：固定支撑、解 `g_S(u)=ν·1` + 预算、10 步 Newton、支撑阈值 `1e-6`（`:161`）、精修解过同一证书才采纳（`:322-330`）。
10. 最终 fail-loud（`:335-336`）；tie-break（`:344-352`，`_cash_kelly_canonical_tiebreak` `:390-451`，两阶段各用 Clarabel 默认容差、无 `1e-10` 钉死）。

对照：CLI/quadrature 侧调用点 `src/gate0/quadrature.jl:616`、`src/gate0/driver.jl:507` 都传 `tol = kelly_tol`（driver 默认 1e-6）。

---

## 2. 判别证据汇总（上轮实验与合并验证）

- **场景 2 两日链（`28_diag_bt2_twoday.log`）**：
  - `ktol=1e-8`（solver=1e-10）→ feasibility 2.2e-16、kkt 8.365e-8、gap 8.366e-8 —— **过 1e-6 证书**；
  - `ktol=1e-6`（solver≈1e-9）→ kkt 2.175e-6、gap 2.175e-6 —— **不过**。
  - 含义：solver 容差 → 解质量的关系极陡（1e-10 vs 1e-9，kkt 差 ~26 倍）；静态放宽公式必死；但「更紧的求解」确实能产出过证书的解。
- **场景 2 单日（`27_diag_bt2_tol_scan.log`，t=270）**：ktol 1e-10~1e-6 全 OK、解逐位相同；耗时 23.6s（ktol=1e-10）vs 1.0s（其余）——紧容差在个别日子显著更慢，宽容差单日质量无差。
- **场景 8（`gate0_t4_recovery_20261010/summary.md`）**：ruler 修复后预测律全 finite（s1(资产2)=0.05），但链路推进后**撞 kelly SLOW_PROGRESS**（第二阻塞）。输入形态：`X 32×2`、`locked=[0.0, 0.3]`（资产 2 locked，base = 0.3×col2）；新几何 gross `col1 1.57e-9~4.73e9`、`col2 2.26e-55~3.46e56`（宽尾但 finite）。
- **合并验证（`gate0_merged_verify_20261010/summary.md`）**：场景 2 在新 posterior 规则下直接通过（数值改变了下游形态）；场景 8 仍红；t=327 的 kelly 层在当前字节未被直接观察（prequential 成本挡路）。
- **当前 kelly.jl 无 `_clarabel_tolerances` helper**（上轮改造已字节级回退；本文 §1 的行号即当前字节）。

---

## 3. 机制分析

### 3.1 为什么 `SLOW_PROGRESS` 下直接 error 是「状态门」，不是「质量门」

证书（feasibility/kkt/gap ≤ tol）在**原始 log 目标**上独立计算，与 Clarabel 的内部 status 无关。`SLOW_PROGRESS` 只表示「未达到请求的内部容差/进展缓慢」，它通常仍返回当前迭代的近似解。当前代码在证书有机会评估之前就拒绝了它——这就是「状态门」。而场景 2 的教训恰好说明：**status=OPTIMAL 也不保证证书过；最终质量必须由证书说了算**。因此「让 status 退回诊断、让证书当门槛」在语义上不是放松，而是把验收权放回它本来该在的地方。

### 3.2 场景 8 为何可能卡死（两个候选机制）

- **M1 极端尺度病态**：locked base 场景值 `2.26e-55 ~ 3.46e56`，跨 ~111 个数量级；`log` 在这些场景的梯度 `1/wealth` 动态范围极端。exp-cone 表示 + 内点法在 `1e-10` 容差下的 KKT 残差可能需要机器精度之外的条件数。若是此机制，宽容差解+polish 可能仍过不了 1e-6 证书，**且任何纯求解器配置都无法解决**——需要数值层处理或上位裁决。
- **M2 慢收敛但可达**：极端值只在少数场景，最优解可能由其余场景主导，`SLOW_PROGRESS` 只是 1e-10 不可达；宽容差解经 polish 可过 1e-6 证书。若是此机制，方案 A 直接可解。
- **判别点**：若在场景 8 单日上 solver 放到 1e-8 后 status=OPTIMAL、且解 + polish 后 kkt ≤ 1e-6 → M2 成立；若 kkt 仍 >>1e-6（或解含 NaN）→ M1 成立，需升级裁决。

### 3.3 与「恒定 1e-10」的关系

场景 2 的实测（`:297-300` 的 `1e-10`）说明：**对某些问题族，主求解容差必须紧到 1e-10 才能让 OPTIMAL 解过证书**。因此方案不能把 `1e-10` 简单替换为宽容差——正确形态是「多级尝试 + 证书闭环」：宽容差先试（便宜、且能规避 SLOW_PROGRESS），不行再落回 1e-10（现状行为），并把 SLOW_PROGRESS 解也纳入证书验收。**对场景 2 的最坏情形 = 现状**（紧路径保留），最好情形 = 宽路径也过（性能收益）。

---

## 4. 候选方案对比

| # | 方案 | 改动面 | 场景 2 | 场景 8 | 证书/纪律 | 风险 |
|---|---|---|---|---|---|---|
| A | SLOW_PROGRESS 出口证书化 | `kelly.jl:303-331`（状态检查 → 有解则走证书+polish） | 不受影响（其路径 status 已通过） | M2 可解；M1 仍红 | 证书不变、fail-loud 不变 | SLOW_PROGRESS 解可能不可用；须防「解为 NaN/负」 |
| B | 宽→紧分级闭环 | `:286-336`（两次求解 + 证书闭环） | 最坏=现状、最好更快 | 覆盖 M2；M1 仍红 | 同上 | 成本 ×2~3；需要确定性排序（宽度顺序写死） |
| C | polish 强化（缩支撑重试/支撑枚举） | `:147-211` | 无害 | 只在「有解」时有用（配合 A/B） | 同一证书 | 复杂化；上轮已显示缩支撑改造能修部分红 |
| D | 数值尺度预处理（free 列对角缩放，等价） | 求解装配层 | 近乎恒等（尺度相近） | **对 base 极端值无效**（base 非求解变量）；只抗 free 列尺度失衡 | 需等价性证明 + 原空间证书 | 对场景 8 的 base 病态大概率无效；不首选 |
| E | 输入层处理（预测律/几何） | 上游 | — | 改变数学样本，超出 kelly 授权面 | — | 不在此设计范围 |

**关键判读**：场景 8 的极端尺度主要来自 **locked base**（外生常数、不参与优化），因此方案 D（列缩放）大概率治不了它；能救它的只有「宽容差/停滞解的证书化验收」（A/B）或更上位的样本层裁决（E）。A 是信息增益最高、改动最小的下一步。

---

## 5. 推荐与实现规格草案（待判别后定稿）

### 5.1 推荐顺序

1. **先做 §6 的判别实验 E1-E3**（不改源码，探针脚本）。
2. 若 M2 成立：实现方案 A（主）+ 可选 B（A 不够时）。
3. 若 M1 成立：停止 kelly 侧修补，把「场景 8 极端 base 的数值可达性」升级为 SPEC 级裁决（可能的方向：样本层宽尾处理、证书数值度量、或承认该 fixture 的 kelly 层不可解并调整场景构造——均由上位决定，不由本设计擅自改）。

### 5.2 方案 A 规格草案（若判决定稿）

- **改动**：`cash_kelly` 主求解段（`kelly.jl:303-331`）。
- **控制流**：
  1. `solve!` 后，若 `status ∈ {OPTIMAL, ALMOST_OPTIMAL}` → 现状路径（后处理 → 证书 → polish → 证书）。
  2. 否则，若 `status == SLOW_PROGRESS`（或 Clarabel 其他「返回解」状态）：`evaluate(w_aug)`；若解全有限且 `total > 0` → 后处理（截零、预算恢复）→ 证书 → polish → 证书；**过则接受并记录 status 来源；不过则 `error(原文本 + status)`**。
  3. 解不可用（非有限/退化）→ 原 error。
- **不变量**：任何被接受解都必须通过 `cash_kelly_certificate`（tol 不变）；错误文本保留原句；无 RNG；tie-break（`:344-352`）不变（它只消费已认证主解）。
- **诊断字段（建议）**：证书携带 `solver_status`（符号），供报告与测试断言用；不改变证书三指标的语义。
- **风险控制**：`kelly_cash_tests` 负控（证书红必须红）必须保持；新增「SLOW_PROGRESS 解过证书 → 接受」的定向测试需在判别实验确认可达后写。

### 5.3 方案 B 规格草案（条件项，A 不足时）

- 求解顺序固定为：宽容差（如 Clarabel 默认 1e-8）→ 证书/polish；红 → 1e-10（现状）→ 证书/polish；红 → error。
- 宽容差仅作为「先到解附近」的快速阶段；**最终接受永远是证书**。
- 成本与耗时增加须由实验量化；`ktol` 路径不变（调用者 `tol` 语义不变）。

---

## 6. 判别性实验设计（供 DevOps；不改生产源码，探针脚本落受控区）

- **E1（场景 8 停滞形态）**：复现 `archive/evidence/gate0_t4_recovery_20261010` 的场景 8 t=271 输入（可用其探针方法从 `single_day_decision` 拦截 X_free 与 base，或重放 `03_diag_s8_gross` 的构造）。在实验副本上把主求解容差依次放到 `1e-8`、`1e-6`，观察：status、迭代数、`evaluate` 解是否有限、`cash_kelly_certificate` 三指标、`_cash_kelly_newton_polish` 后三指标。
  - 判据：status=OPTIMAL 且（原始或 polish 后）kkt/gap ≤ 1e-6 → M2；否则 M1。
- **E2（SLOW_PROGRESS 解拦截）**：保持 1e-10 不变，仅捕获 SLOW_PROGRESS 后的解（不采纳，诊断）→ 同 E1 的证书与 polish 评估。
- **E3（尺度归因）**：打印场景 8 输入的列尺度、base 分布（min/max/分位）、每场景对 kkt 梯度的贡献排序——确认病态是 base 极端值主导还是 free 列尺度。
- **E4（场景 2 回归）**：若实现 A/B，重跑两日链（`28` 同款构造）：确认宽路径失败时回退紧路径后 kkt 仍 8.4e-8 过；确认最终解与现状一致或变化在证书容差内。
- **E5（模块回归）**：`kelly_cash_tests` 77/77 + 负控（证书红必红）；`driver_tests`/`backtest` 相关分组。
- **E6（确定性/重放）**：同输入同 seed 两次运行逐位一致；tie-break 行为不变。
- 全部实验 ≤60s/命令、RSS 2048MiB、串行；超时如实记录，不为赶进度放宽 tol。

---

## 7. 影响面盘点

- **`kelly_cash_tests`（77）**：方案 A 不改任何已过路径的语义；负控必须保持。新增断言只能加在「判别实验确认行为之后」。
- **场景 2（t=271）**：保护目标是「最终解 kkt ≤1e-6」——由 A/B 的证书闭环与紧路径回退保证（最坏=现状）。
- **场景 8（t=271）**：A 的可解性判据见 E1/E2；若 M1 则如实红（升级裁决），不制造假绿。
- **driver**：`driver.jl:507` 调用契约不变（`cash_kelly(X_free; base, budget, tol)` 签名与返回三元组不变）。
- **tie-break 契约**：只消费「已通过证书的主解」；A/B 不改变其输入契约（`cert.objective` 语义不变）。若证书 NamedTuple 新增 `solver_status` 字段，需同步消费者与测试。
- **quadrature.jl:616**：同一 `cash_kelly`，自动受益；无独立改动。

---

## 8. 待 Manager 裁决点

1. 是否批准先做 E1-E3 判别（不改源码）；判别的执行安排。
2. 若 M2：批准方案 A 的实现规格（§5.2）与回归计划。
3. 若 M1：场景 8 的升级路径（样本层/证书度量/场景构造）——本文件不擅自选择。
4. 证书是否携带 `solver_status` 诊断字段（影响测试断言与报告口径）。
5. `ktol=1e-10` 在个别日子的 23.6s 级耗时（`27` 证据）是否值得纳入「宽容差先试」的性能收益评估（B 方案动机之一；不作性能承诺）。

---

## 9. 层归属

证书驱动自适应属于**数值工程层/求解器链路控制**：数学对象（同一 log-Kelly 目标、同一可行集、同一证书）不变；改变的是「求解器状态如何映射到验收流程」。验收权始终在 `cash_kelly_certificate`（原目标），fail-loud 文本与门槛不放松；不引入任何由回测收益选择的参数。判别实验的目的是让机制选择基于数值事实（status/解质量），而不是猜测。

---

*本文件为只读调查 + 设计文档；未运行任何命令、未修改 `src/`、`test/`、`archive/` 既有文件。E1-E6 为设计草案，执行归 DevOps 受控流程。*
