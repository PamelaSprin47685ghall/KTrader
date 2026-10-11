# KTrader — D-066 数值容差定稿实验设计（ε_w / ε_U / M_max）

**文档状态：预注册设计草案。只读撰写——未运行任何命令、未修改 `src/`、`test/` 或任何既有文档。**
**权威关系：本文细化 `AGENTS.md` D-064/D-065/D-066/D-067（§20，行 2594-2680）与 `docs/NUMERICAL_INTEGRATION_SPEC.md` 的执行语义；与 `AGENTS.md` 冲突时以 `AGENTS.md` 为准。本文不授权任何运行、不裁决任何数值、不改变当前门禁与容差。**
**范围锚点：`docs/KELLY_NUMERICAL_ROW_SCALING.md` §9.13 的 c 项（「weight_tol D-066 复核——规范定稿程序」）；`docs/GATE0_EXIT_CHECKLIST.md` 未收敛日决策包。**
**落盘位置说明：置于 `docs/` 与 `docs/NUMERICAL_INTEGRATION_SPEC.md` 同目录——二者同属数值层规范文档，§9.13 c 项与本文直接对接。**

---

## 0. 任务、范围与可交付物

### 0.1 定稿对象

D-066 要求 `ε_w、ε_U、M_max` 通过 numerical refinement 定稿（五步：synthetic 解析答案 → realistic fixture → tolerance 减半 → weight/utility 稳定 → 与 Sharpe/PnL 无关）。对应当前实现（`src/gate0/quadrature.jl` 四证书，行 776-826）：

| D-066 符号 | 实现字段 | 当前 driver 生产值 | 当前函数默认值 |
|---|---|---|---|
| ε_w（Certificate A） | `weight_tol` | 1e-4（`driver.jl:257`） | 1e-3（`quadrature.jl:687`） |
| ε_U（B_opt，fine-rule regret） | `utility_tol` | 1e-6（`driver.jl:258`） | 1e-5（`quadrature.jl:688`） |
| （B_audit，D-066 未点名但实现面存在） | `utility_tol_audit` | 1e-5（driver 未透传→函数默认） | 1e-5（`quadrature.jl:689`） |
| M_max | `max_scenarios`（+ `min_scenarios`=64） | driver 默认 512；批次运行 `GATE0_MAX_SCENARIOS=131072` | 512（`quadrature.jl:686`） |

ε_U,audit 的定稿地位见 §11-2（待裁决）。

### 0.2 可交付物（执行完成后）

1. **证据包**：`archive/evidence/gate0_tol_finalize_<date>/`（A/B/C/D 四组产物 + summary + hash 清单）。
2. **定稿记录**（numerical configuration，格式见 §6.3）：ε_w★、ε_U★、ε_U,audit★、M_max★ 及依据分位表。
3. 本文档在定稿后的状态更新（「执行结果」节）与 `AGENTS.md` 第 N 任段登记（由文档 owner 执行）。

### 0.3 非目标

- 不改任何门、容差、源码、测试；不改 `src/gate0/quadrature.jl` / `driver.jl` 默认值。
- 不做任何 SPEC 裁决；不跑回测/多日窗口；不采集任何收益指标。
- 不评估 §9.13 的 b（范围门控）与 d（活跃集——S10 已否定，n=1 无零列）；仅在依赖处引用。

---

## 1. 现状基线（只读核对）

### 1.1 实现位置（当前字节）

- 循环与四证书：`src/gate0/quadrature.jl:761-833`（A 在 777；B_opt 在 785-798；B_audit 在 804-819；联合判据 822；fail-loud 830-833）。
- 返回对象 `AdaptiveKellyResult`（含 `history`：每轮 `(; M, M2, A, B_opt, B_audit)`）：`quadrature.jl:553-566`。
- tolerance 默认值的「初始候选」声明与定稿路径：`quadrature.jl:648-668`。
- driver 生产透传：`src/gate0/driver.jl:252-279`（`weight_tol=1e-4 / utility_tol=1e-6 / kelly_tol=1e-6 / posterior_tol=1e-6 / max_cells=2048 / u_span=5.0`；`min_scenarios=64 / max_scenarios=512`；`mu_qmc/mu_chisq_qmc` 默认 false）。
- adaptive 调用透传面：`driver.jl:473-483`（未传 `utility_tol_audit` → 函数默认 1e-5；未传 `utility_tol_audit` 即 B_audit 门 1e-5）。
- 入口环境变量：`bin/backtest.jl:130-142`（`GATE0_WEIGHT_TOL=1e-4 / GATE0_UTILITY_TOL=1e-6 / GATE0_MAX_SCENARIOS=512 / GATE0_MU_QMC=false / GATE0_CHISQ_QMC=false`）。
- 审计注记（已知）：`dev/gate0_static_audit_2026.md:208`（B_opt 无显式 `>=0` 断言，非负性由 fine 解最优性 + C 证书支撑）；`:238`（函数级默认与生产入口的差异，未来第二调用方需审查传参）。

### 1.2 已有测量（只读转引，均为既有字节上的观测）

**(a) 合成世界（`test/gate0/quadrature_tests.jl`）**

- 收敛率：RQMC 权重 L1 的实测收敛率约 **M^{-1}**；`weight_tol=1e-3` 需 M~2048+（该文件 216-217 行注释）。
- B 的二阶性：实测 **B/A² ≈ 1.5e-7 恒定**（B ~ H_eff·A²/2，H_eff≈2σ²；263-298 行注释）——光滑凸 fixture 上 B 的二阶收敛恒先于 A 的一阶收敛，「B 红而 A 已过」不可构造。
- 假收敛实例：`min=8` 时 8/16 点 RQMC 偏差使解落在满仓角点（w=[1,0]），两角点解相同 → **A=1.8e-8 假收敛**；M=32 才回内部解（该文件 264-272 行注释）。这是「A 小 ≠ 解对」的既有直接证据。
- tolerance 减半既有测试（243-254）：1e-3→5e-4 时 M 非降、w 差 ≤0.02（粗断言——本设计将其细化）。

**(b) 真实链 17 日 A 序列（过门配置：`mu_qmc=true + chisq_qmc=true + max=131072`；`weight_tol=1e-4`；`S11_summary.md`）**

```
t:   330     331     332     333     334     335     336      337     338     339   340      341     342    343      344      345     346
A:  6.26e-5 5.97e-5 6.25e-5 3.40e-5 5.31e-6 9.33e-5 1.355e-4  9.48e-5 4.67e-5 卡    1.283e-4 9.83e-5 4.40e-5 1.246e-4 1.346e-4 8.32e-5 3.42e-5
```

- 十六日完成里 **12 过 4 超**（336/340/343/344），通过率 75%；门（1e-4）位于 A 分布约 75% 分位；未收敛日为波动聚集（343→344 连续），非固定周期。
- 全部 A 来自同一层对 **65536→131072**（批次的 min/max 语义；`dev/batch_two_stage.md:52-54` 登记「min=65536 跳过 64..65536 的中间层」）。
- B 证书水平（`batch_out/batch_t33X_solve3.txt`）：过门日 330：B_opt=4.66e-10、B_audit=1.08e-6；未收敛日 336：A=1.3550e-4、B_opt=2.87e-9、B_audit=6.70e-7。**B_audit 与门（1e-5）同量级但未 binding；A 是唯一瓶颈。**
- 339 卡点：solve 段三次 48/48/50s 全切（单次主求解原子不可拆；AGENTS.md 第 16 任段 / `S7_t339_b2b1b{,2,3}.log`）。

**(c) 深段历史（历史布局（NODE）时代的观测；`DB_summary.md`/`DS_summary.md`）——其与当前过门配置（P-batch）的逐项差异未在本次只读核对中厘清；只作「现象存在」证据，不得转写为定稿数字**

- 非单调：8192→16384: 3.694e-4；16384→32768: 1.053e-4；**32768→65536: 7.193e-4（反弹 6.8×）**。
- 通道主导：固定 z1（μ 中心方向）→ A 塌缩约 7 个数量级；噪声底 ~1e-4–7e-4，**与门（1e-4）同量级**。
- 结论（当时）：纯预算/单通道固定都不能可靠过门；「门本身值得独立复核」（DS 未决第 2 条）——即本设计的动机来源。

**(d) 成本基线（只作预算口径）**

- 131072 档 kelly 整段 ~41s / ~1.6GiB（`bin/backtest.jl:139-141` 注）；六段形态 capture 27s / b1a 10s / b1b 28s / b2a 11s / b2b 37s / b3 10s（t=330）；八/九段后最大段 40s、合计 168-201s/日（`S11_summary.md`）。
- 262144 档：main 11s 进窗、tie-break 段出窗（`S5_t336d_b2b3{,b}.log`，124×2）——**深于 131072 的完整生产解当前被 tie-break 阻断**（与 §9.13 的 b/d 耦合，登记为可选延伸的依赖）。

### 1.3 关键缺口（本设计要填的）

1. **64..32768 的逐层 A/B 轨迹不存在**——全部真实证据只有最后一层对；「M 需求」无法从单层对推断。
2. **「A 作为误差代理」的有效性未测**——A 小的时候真实误差多大？合成世界可给出答案（真实误差 ≤ k·A 的经验 k）。
3. **减半稳定性未测**——D-066 第 3/4 条未执行。
4. **B_audit 的观测分布未系统化**——只有 3 个单点（1.08e-6 / 3.79e-7 / 6.70e-7），门 1e-5 的余量与 binding 临界未知。
5. **首次穿越 vs 稳定满足的差异未测**——深段非单调意味着「单层 ≤ε 即返回」存在假收敛窗口（合成世界的角点实例已证存在性），其频率未量化。

---

## 2. 预注册纪律（执行前冻结）

1. **判据先于数据**：本文经评审/Manager 确认后冻结。执行批次不得修改判据、候选集、seed 清单；发现判据不可行时，另立登记并交裁决（不得静默改口径）。
2. **无收益字段**：全部产物**禁止**出现 `pnl / sharpe / return / net_value / drawdown` 任何列；证据只含 A / B_opt / B_audit / M / w / 证书三项 / 耗时 / 配置。任何分析不得以收益为条件筛选 fixture 或候选。
3. **固定 seed**：真实 fixture 主 seed = `0x00D05EED00000001`（驱动默认，与批次证据链一致）。合成世界额外 seed（预注册）：`0x0000C0FFEE / 0x0000BEEF / 0x0000AAAA / 0x0000BBBB`（均在既有测试中使用过；如需增减须在冻结时确认）。
4. **字节登记**：执行前记录 `src/gate0/`、`test/gate0/`、`dev/` 工具、`batch_out/` 输入产物的 SHA256 清单（`pre_snapshot`）；结论绑定字节；不跨字节转写数字。
5. **重放与如实记账**：同 seed 同字节逐位可重放；失败/未收敛/超时如实入账，**不得重跑择优**。
6. **配置双口径**：所有 fixture 与产物必须标明配置标记：`P-batch`（`mu_qmc=true + chisq_qmc=true`，与既有证据链一致）或 `P-default`（driver 默认，`mu_qmc=false`）。主定稿口径的选择见 §11-1。
7. **工具不动数学**：新/旧 dev 工具只允许复刻生产调用链与离线拼装统计；不得引入第二套 posterior、不得改求解参数（`kelly_tol` 等一律照生产值）。

---

## 3. 实验 A：synthetic 解析世界（D-066 第 1 条）

### 3.1 世界清单（预注册）

| # | 世界 | 分布 | 解析锚 | 用途 |
|---|---|---|---|---|
| A1 | known Gaussian 2-asset | `r_j ~ N(μ_j, σ_j²)` 独立，复用 `quadrature_tests.jl` 的 `MU_FIX=[0.0015,0.0008]`, `SIG_FIX=[0.08,0.06]` | w* 由 GH 数值积分 + KKT 求根（精度目标见 A.2） | 主收敛曲线 |
| A2 | Gaussian 2-asset 弱 edge 变体 | μ 减半（[0.00075,0.0004]），σ 不变 | 同 A1 | SNR 敏感性 |
| A3 | known Gaussian 3-asset 独立 | μ=[0.0012,0.0009,0.0006], σ=[0.09,0.07,0.05] | 3D GH + 局部 KKT 求根（不做全网格） | 维度扩展（轻量） |
| A4 | D-090(a) 离散对称世界 | 三行 (1.25,0.9,0.9) 及其置换，等权 1/3（复用 `kelly_cash_tests.jl:187-189`） | w*=(1/3,1/3,1/3)、cash=0（解析角点） | 结构判据（对称性/精确解） |
| A5 | D-090(b)/D-091 平局世界 | `kelly_cash_tests.jl:196-198` 与 `:228-233` | 平局集解析（objective ≡ 0） | 无集中/端点判据 |

说明：A4/A5 是离散行世界，M≥行数后 RQMC 完全枚举——**只作结构检查（对称、平局、无虚假集中），不产生「M 需求曲线」**（无深度信号）。曲线职责在 A1-A3。

### 3.2 构造（RQMC 包装规格）

- **A1-A3**：`RQMCScenarioSource`，gen 用 inverse-CDF（与 `quadrature_tests.jl:81-94` 的 `gaussian_rqmc_source` 同模式）：`r_j = μ_j + σ_j·Φ⁻¹(pts[s,j])`，`gross = exp(r)`。无 μ epistemic 通道（`mu_rng` 忽略）——**这是与真实链的结构差异（真实链有 matrix-t μ 通道），必须作为边界标注**。
- **A4/A5**：gen 从 1 维 rule 点做行选择（`idx = ceil(u·R)`），返回该行 gross 向量。
- **解析基准方法升级**（相对既有测试的 40 节点+网格）：基准 = 高精度数值积分（GH 2D：80 节点/维；3D：40³ + 局部精化）上的目标函数，以 KKT 方程数值求根（Newton）解出 w*；**基准自洽验证**：GH 节点 80→120（或网格+求根两法）给出的 w* 差 < 1e-6 方可使用；3D 只做局部最优性验证（一阶条件残差 < 1e-6）。目标：使「与解析 w* 的距离」可测到 1e-5 量级。

### 3.3 测量量（每层对、每 world×seed）

对层集 `L = {64,128,256,512,1024,2048,4096,8192,16384,32768,65536}`（可选延至 131072）与层对 `(M→2M)`：

```
M, M2, A=‖w_2M−w_M‖₁（含 cash 增广）, B_opt, B_audit, secs,
w（完整向量）, wc, |w_{2M}−w*|₁（A1-A3）, objective(M2)
```

A4/A5 附加：置换对称差 `max(w)−min(w)`、cash 值、objective 与平局值之差。

### 3.4 判据/分析（定稿输入，全部预注册）

1. **收敛率拟合**：`log A vs log M` 与 `log|w_M−w*| vs log M` 的斜率（分段：浅段 64-4096、深段 4096-65536）；报告两斜率之差。
2. **包络比**（A 作为误差代理的有效性）：逐层对计算 `k = |w_{2M}−w*|₁ / A`；报告 p50/p90/max 与是否随 M 稳定。**若 k 无稳定上界，必须如实标记「A 不能单独作为误差代理」**。
3. **M_first(ε) 与 M_req(ε) 表**（定义见 §7.1），ε 候选集：
   `{1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5}`；对每候选给出所需 M（或「超测程」）。
4. **假收敛率**：`P(|w_{2·M_first(ε)}−w*|₁ > ε)`（首次穿越层对的 fine 解；跨 seed/world；ε 同上）——直接量化「首次穿越即返回」的错误率。
5. **B 的 binding 临界**：max B_opt 与 max B_audit 在全部产物上的观测分布；给出使 B 门开始 binding 的 ε_U 临界值。
6. **tolerance 减半**（并入实验 C 的合成部分，见 §5）。

### 3.5 证据形态

- `tolA_<world>_<seed>.csv`（逐层对一行，列见 §3.3）。
- `tolA_<world>_summary.md`：M_req 表、包络比分布、假收敛率、收敛率拟合。
- `tolA_summary.md`：跨世界总表。

### 3.6 分段执行（每段 ≤60s/RSS2048）

新 dev 工具规格：**`dev/tol_trace.jl`**（与 `batch_t_solve3.jl` 逐点同式复刻；退出码独立）：

```
gen   --world <W> --seed <hex> --M <M> [--audit] --out <dir>   # 生成并落盘 X_M（opt/audit）
solve --world <W> --seed <hex> --M <M> --out <dir>              # 读 X_M → cash_kelly → seg_M
join  --world <W> --seed <hex> --M <M> --out <dir>              # 读 seg_M/seg_2M/X_2M + 现场 gen audit → A/B + 行落盘
```

- 小层（M≤4096）：`gen+solve+join` 可合并进单命令（多 M 循环）。
- 8192..65536：单层单段（gen 一段、solve 一段、join 一段）；
- 2 资产世界预计单层 solve 显著低于真实链（真实 131072 整段 ~41s 为上限参考）；如单命令超窗按 gen/solve/join 拆分。
- 每 world×seed 预计 14-18 段；A1-A3 × 4 seed 共 12 组 ≈ 170-210 段（小层合并后）；A4/A5 结构检查各 1-2 段。

---

## 4. 实验 B：realistic single-day fixtures（D-066 第 2 条）

### 4.1 fixture 清单（预注册）

**核心 6 日**（覆盖过门/贴边/超门三种形态）：

| t | 已有 A（65536→131072） | 形态 |
|---|---|---|
| 330 | 6.2585e-5 | 过门代表（首次过门日） |
| 331 | 5.9696e-5 | 过门 |
| 335 | 9.3344e-5 | 贴边（距门 6.7%） |
| 336 | 1.3550e-4 | 超门代表（+35.5%） |
| 344 | 1.346e-4 | 超门（与 343 连续） |
| 346 | 3.4211e-5 | 最深过门 |

**扩展 5 日（可选）**：340（1.283e-4）、343（1.246e-4）、341（9.83e-5 贴边）、342（4.40e-5）、345（8.32e-5）。

已有产物（`archive/evidence/gate0_multiday_run_20261010/batch_out/`）：`batch_t<day>_{prep,X1,X2,seg1,seg2}.bin` 与 `_solve3.txt`（330-346 全体）。**65536→131072 层对直接复用；32768→65536 的 X 侧已有（X1）；更小层全新计算。**

### 4.2 方法

- **逐层对轨迹**：对 `M ∈ {64,128,...,32768}` 逐个执行层对（opt gen + cash_kelly + join），与既有 65536→131072 拼成**完整 11 层对序列**。
- 求解调用 = 生产同式（`cash_kelly(X_free; base, budget, tol=kelly_tol)` + `locked_wealth_gate0`），与 `quadrature.jl:732-759` 的 `solve_layer` 逐点一致；不得绕过 locked/base 语义。
- 配置：**主口径 P-batch**（与既有证据链一致）；**P-default 对照子集**（330、336 两日，全部层）——量化配置差异（见 §11-1）。
- audit X：join 步现场 gen（每层对一次，成本 ≈ opt gen）。
- 大层对（32768→65536 起）沿用现有 SPLIT 链（八/九段拆分；`dev/batch_two_stage.md:94-103`），保持「每段独立进窗」。

### 4.3 测量量

同 §3.3（但无 w*）；附加每层 `w` 的摘要（top 分量、cash）、status、耗时。**禁止**任何收益字段。

### 4.4 判据（定稿输入）

1. 每 fixture × ε 候选：`M_first(ε)`、`M_req(ε)`（或「≤131072 内未稳定」）。
2. **「A 是唯一瓶颈」假设检验**：全部层对上 B_opt/B_audit ≤ 门/10 的比例；B binding 的层对清单（若有）。
3. 未收敛日的深层走势：A 在 65536→131072 之上的可观测行为（若 262144 段可行则延伸；否则如实记录「超测程」）。
4. 跨层 B_audit 的分布（ε_U,audit 定稿输入）。

### 4.5 证据形态

- `tolB_t<day>_layers.csv`（P-batch；P-default 子集另加 `_pdefault` 后缀）。
- `tolB_t<day>_summary.md`；跨日总表 `tolB_crossday_summary.md`（每行：day、配置、11 层对的 A/B、M_first/M_req 表）。

### 4.6 分段执行

- 小层对（M≤4096）：单命令逐对（复用 `batch_t_solve3.jl` 的 `GATE0_BATCH_MIN=M GATE0_BATCH_MAX=2M`，或 `tol_trace.jl` 的批量模式）；
- 8192..32768：每层对 1-3 段；
- 32768→65536：solve(32768) 一段 + join（X1 已有）一段；
- 65536→131072：join 一段（seg1/seg2/X2 已有；audit 现场 gen）。
- 每 fixture ≈ 14-20 段；核心 6 日 ≈ 90-120 段；扩展日可选。

---

## 5. 实验 C：tolerance 减半（D-066 第 3/4 条）

### 5.1 设计

**形态 1（离线读出，主；零新增求解）**：从 A/B 轨迹（实验 A/B 的产物）对每 world/fixture、每候选 ε 读出：

- `w_first(ε) = w_{2·M_first(ε)}`（该层对的 fine 解；现行 adaptive 语义会返回的解）；
- `w_stable(ε) = w_{2·M_req(ε)}`（保守语义的解；M_req 超测程则不可评估）；
- 对 ε 与 ε/2（及可选 ε/4）：计算 `‖w(ε) − w(ε/2)‖₁` 与 `‖w_stable(ε) − w_stable(ε/2)‖₁`。

**形态 2（真跑抽检，次；预算可控）**：对合成 A1（2 资产，便宜）与 2-3 个真实 fixture 用 `adaptive_scenario_kelly` 以 ε/2、ε/4 重跑（真实 fixture 需 max 放大或如实 fail）。

### 5.2 判据（预注册）

- **(c1) 稳定性**：`‖w(ε) − w(ε/2)‖₁ < ε/2` → 该 ε 在「已定稿稳定」候选。
- **(c2) 单调性**：`M_req(ε/2) ≥ M_req(ε)`（允许相等；违反须归因）。
- **(c3) 形态 2 一致性**：重跑返回的 M 非降、解差复现 (c1)。
- **(c4) ε_U 减半**：B 门减半后的 binding 检查——若减半后 B 开始 binding，说明 ε_U 的有效约束区已到；记录临界。
- 每 (ε, world/fixture) 记：通过 / 失败 / 超测程（三态，超测程本身是 M_max 证据）。

### 5.3 证据形态与分段

- `tolC_halving_summary.md`（候选矩阵：行=ε，列=world/fixture，格=三态 + 数值）；引用 A/B 轨迹文件，不复制。
- 形态 1 零新命令；形态 2 每 fixture 1-2 段（合成含在 A 的段内）。

---

## 6. 实验 D：M_max（D-066 第 4/5 条）

### 6.1 分析输入

A/B/C 的 `M_first(ε)`、`M_req(ε)` 分布（per-day、per-world、per-seed）；B binding 数据；成本实测（每层对耗时）。

### 6.2 输出（供 owner 定稿）

1. **「ε → M 需求」表**：对每候选 ε，给出 `M_req` 与 `M_first` 的 p50/p90/p95/max（合成按 world、真实按日）。
2. **撞顶率**：对 M_max 候选 `{512, 4096, 32768, 131072, 262144}` 计算「M_req(ε★) 超出该顶」的比例（对合成与真实分别给出）。
3. **首次穿越 vs 稳定满足的间距**：`M_req − M_first` 分布——量化当前「单层判据」的假收敛暴露面（合成世界另有真误差校准，§3.4-4）。
4. **成本表**：每层对/段的实测耗时与 RSS（只作预算依据，不作「优化」目标）。

### 6.3 定稿记录格式（numerical configuration）

```
weight_tol★      = ...   依据：M_req 表（p95/max）+ 包络比 + 假收敛率
utility_tol★     = ...   依据：B_opt 观测分布 + binding 临界
utility_tol_audit★ = ... 依据：B_audit 观测分布 + binding 临界（D-066 未点名——见 §11-2）
max_scenarios★   = ...   依据：M_req(ε★) 分位 + 成本上界 + fail-loud 语义
min_scenarios    = ...   依据：首个非退化层（合成 A4/A5 与既有角点教训）
```

### 6.4 证据形态与分段

`tolD_Mmax_summary.md`（表 + 图数据）；纯离线（除可选的 262144 延伸段）。**262144 延伸依赖 tie-break 段进窗（当前出窗）——见 §11-4。**

---

## 7. 统一定义与证据规范

### 7.1 定义（钉死，执行批次不得改）

- **层集 L**：`{64,128,...,131072}`（可选 262144）；**层对 i**：`(M_i → 2M_i)`。
- **A_i** = `|vcat(w_{2M},wc_{2M}) − vcat(w_M,wc_M)|₁`（含 cash 增广；与 `quadrature.jl:777` 同式）。
- **B_opt_i / B_audit_i**：与 `quadrature.jl:798/819` 同式；audit X 每次现场生成（同 `am_seed` 派生链）。
- **M_first(ε)** = 首个使 `A_i ≤ ε` 的层对之 M（现行 adaptive 的返回点语义）。
- **M_req(ε)** = 首个使「自该层对起全部已测层对均 ≤ ε」的 M；若末层对不满足 → **「未稳定」**。
- **稳定区** = `M ≥ M_req(ε)` 的已测区间。
- **假收敛**（仅合成，有 w*）：`A_i ≤ ε` 而 `|w_{2M_i}−w*|₁ > ε`。
- **超测程** = `M_req(ε)` 或 `M_req(ε/2)` 超出已测最大层。

### 7.2 命名与落盘

- 目录 `archive/evidence/gate0_tol_finalize_<date>/`。
- 文件：`tolA_* / tolB_* / tolC_* / tolD_* / summary.md / pre_snapshot.txt`（16+ 关键对象 SHA256）。
- 每 log 尾行沿用 scoped 形态（rc / elapsed / deadline / killed / rss_peak）。

### 7.3 禁止字段（重申）

`pnl / sharpe / return / net_value / drawdown` 及任何收益派生列——全产物禁止。

---

## 8. 受控执行分段设计（总表）

| 实验 | 命令形态 | 段数估计 | 单段约束 |
|---|---|---|---|
| A（合成） | `dev/tol_trace.jl {gen,solve,join}`（新工具） | ~180-220 | ≤60s / RSS2048 |
| B（真实） | `batch_t_solve3.jl`（复用）+ `tol_trace.jl join` | ~90-120（核心 6 日） | 同上；大层对八/九段拆分 |
| C（减半） | 离线为主；形态 2 抽检 | ~20 | 同上 |
| D（M_max） | 离线 | 0（+可选 262144 段） | — |

复用清单：`batch_out/` 330-346 的 prep/X1/X2/seg1/seg2；`batch_t_solve3.jl` 的 rule 派生与拆分链；`FB_diag.jl / S4_finish_diag.jl / S10_gap_diag.jl` 的「读已有产物离线诊断」形态。

新工具（唯一新增）：`dev/tol_trace.jl` —— **规格要求**：

1. 与 `batch_t_solve3.jl` 的 rule 派生逐点一致（`opt_rule=SobolOwenRule(2, rule_seed)`；`a_seed=audit_seed(rule_seed)`；`m_seed=_default_mu_seed(rule_seed)`；`am_seed=m_seed ⊻ 0x9E3779B97F4A7C15`）。
2. solve 与 `solve_layer`（`quadrature.jl:732-759`）同式（free 列 Kelly + `base_locked`；方案 2 表示；**禁止** X_full·w_full + base 双重计入）。
3. 支持「world 模式」（合成）与「prep 模式」（真实）；均落盘 X/seg/行统计；原子写入；退出码独立可辨。
4. 不引入任何新数学、不缓存跨调用、不改任何生产容差。

---

## 9. 与 D-066 五条的逐条对应

| D-066 条款 | 本设计落点 |
|---|---|
| 1. synthetic laws 有解析答案 | §3（A1-A5；解析基准 = GH + KKT 求根，精度自洽 <1e-6；结构世界解析对称/平局） |
| 2. realistic single-day fixtures | §4（330/331/335/336/344/346 核心 + 扩展；复用批次产物补全 11 层对） |
| 3. tolerance 减半 | §5（形态 1 离线读出 + 形态 2 抽检；(c1)-(c4) 判据） |
| 4. weight / utility 稳定 | §3.4-3/4（M_req、假收敛率）+ §4.4（M_first/M_req、B binding）+ §5.2（(c1) 稳定性） |
| 5. 与 Sharpe/PnL 完全无关 | §2 纪律 + §7.3 禁止字段；判据与 fixture 选择不含任何收益输入 |
| 定稿后「记录为 numerical configuration」 | §6.3 格式 + §0.2 可交付物 |

---

## 10. 预期工作量与边界

### 10.1 工作量（段数口径）

- A：3 个连续世界 × 4 seed × 11-12 层对（小层合并后 ~14-18 段/组）≈ 170-210 段；A4/A5 结构检查 ~6 段。
- B：核心 6 日 ×（补 10 层对 + join）≈ 90-120 段；P-default 对照子集（2 日 × 11 层）≈ 25-35 段；扩展 5 日可选 ≈ +80 段。
- C/D：以离线为主；形态 2 抽检 ~20 段。
- 合计（核心范围）：**~300-400 段**，按 batch 节奏为多批次串行执行。

### 10.2 哪些结论依赖真实数据 / 哪些可用 synthetic

| 结论 | 来源 |
|---|---|
| A 与真误差的包络关系、假收敛率、收敛率拟合 | **synthetic**（唯一可得：真实链无 w*） |
| 合成世界的 M_req(ε) 曲线 | synthetic |
| 真实 fixture 的 M_first/M_req 分布、未收敛日形态 | **真实数据** |
| B_opt/B_audit 在真实链的分布与 binding 临界 | **真实数据** |
| 配置差异（P-batch vs P-default） | 真实数据（对照子集） |
| 减半稳定性 | 两者（合成主、真实抽检） |

### 10.3 已知边界（如实）

1. 深段非单调（§1.2c）——M_req 的「稳定满足」判据比「首次穿越」保守，二者差值是设计输出而非缺陷。
2. z1 通道的采样噪声底在深段与门同量级——若定稿 ε_w 落在噪声底内，须由 owner 在「加深 / 门口径 / 方差缩减」三者间裁决（§11-3）。
3. 262144 层的完整生产解当前被 tie-break 出窗阻断（§1.2d）——深于 131072 的验证有环境依赖。
4. A1-A3 无 μ 通道，与真实链结构不同——配置差异须由 B 的真实轨迹承担，不能由合成外推。
5. 339 类「solve 段超窗」日的轨迹可能无法完整观测（单次主求解原子不可拆）——如实记「超测程」。

---

## 11. 待 Manager 裁决的取舍

1. **配置口径（最重要）**：生产 driver 默认 `mu_qmc=false`（`driver.jl:276-279`），而全部过门证据与批次运行使用 `P-batch`（`mu_qmc=true+chisq=true`）。定稿数字应以哪个配置为基准？若以 P-batch 为准，P-default 的收敛行为需要单独测量/裁决（或要求生产默认切 P-batch——那是另一个决定）。**本设计建议：主定稿以 P-batch 为准（与证据链一致），同时强制 P-default 对照子集，把差异作为裁决输入。**
2. **ε_U,audit 的定稿地位**：D-066 只写 ε_U；实现面存在独立 audit 门（当前 driver 未透传→1e-5，比 ε_U 宽 10×）。定稿是否覆盖它、是否要求收紧到 ε_U 同值，请裁决（本设计默认纳入定稿对象）。
3. **「稳定满足」的评估请求**：§7.1 同时定义 `M_first`（现行语义）与 `M_req`（保守语义）。若测量显示二者差异显著（假收敛窗口大），是否把「连续 k 层」类判据纳入 SPEC 评估（**语义变更，不在本设计内实施**），请裁决。
4. **262144 延伸的依赖**：深于 131072 的验证需要 tie-break 段进窗（§9.13 的 b 或 d）。是否把 F2 的 262144 延伸与 tie-break 处理批次并联/串行，请裁决。
5. **合成世界与 seed 规模**：A3（3 资产）与 4 个附加 seed 的成本/收益权衡；是否删减以缩批次。
6. **执行冻结方式**：谁确认预注册（Manager/SPEC）、`pre_snapshot` 的 hash 清单范围（建议：`src/gate0/**`、`dev/tol_trace.jl`、`dev/batch_t_solve3.jl`、`batch_out/` 输入产物）。

---

## 12. 不做清单（negative scope）

- 不改 `src/`、`test/`、任何门/容差/默认值/证书语义。
- 不运行回测/多日窗口/GPU；不采集 PnL/Sharpe；不用收益数字做任何选择。
- 不做 SPEC 裁决；不实施 §9.13 的 b/d；不改 `tie_eps`、`kelly_tol`、posterior 求积口径。
- 不把任何「为了让某日过门」的调整写入设计（D-066 第 5 条、SPEC §95、开发守则 §24）。

---

*（本文件为 Gate-0 重开后 D-066 定稿实验的预注册设计；只读撰写，未运行任何命令，未修改 `src/`、`test/` 或任何既有文档。冻结与执行须另行受控授权。）*
