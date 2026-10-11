# t=327 posterior 2D 超参求积：数值合同调查与设计（ADJUDICATION / QUAD）

**状态：设计文档（只读调查 + 设计落盘）。本文件写作过程未运行任何命令、未修改 `src/` 与 `test/`、未差遣任何执行者。**
**证据基准：当前工作树（只读）+ `archive/evidence/gate0_t327_fix_20261010/`（summary.md 与其引用的 00/01/02/20 logs；本文转述其数字，不复制原文）。**
**上位规范：D-038（adaptive / tail 证书 / refinement 证书 / fail loudly）、D-066（数值容差须经 numerical refinement 定稿）、P0-7（不得为运行时间放宽数学容差）、`docs/NUMERICAL_INTEGRATION_SPEC.md`（积分误差与求解误差分层）。**
**边界：kelly 层不在本文范围。`gate0_t327_fix_20261010/summary.md` §3 已实测否定「solver 容差跟随证书 tol」路线并字节级回退——恒定 `1e-10` 对既有场景（backtest 场景 2：solver 1e-9 → kkt 2.17e-6 > 1e-6）有实证必要性。本文不触碰 kelly。**
**纪律：本文全部数值主张若未标注「已实测」，均为待验证预期；方案选择（包括容差定稿）禁止由回测收益/预算驱动。**

---

## 0. 结论先行

1. t=327 当前第一阻塞是 **2D alpha 求积（`adaptive_quadrature_2d`）预算耗尽**：8192 cells 下 `relative error estimate 8.22401003043056e-4 > tol 1.0e-6`（00/02 logs；`src/gate0/posterior.jl:376` 的 fail-loud 文本正是此路径）。
2. 根因是**数学性的**：中点 cell 规则 + 当前误差代理在二维下的固有相对收敛率 ~`O(1/N)`（02 log 轨迹 1024→8192：6.36e-3 → 8.22e-4，外推 1e-6 需 ~6.7e6 cells）。这不是预算参数、不是实现 bug（峰形态正常、尾证书绿），中点规则换预算也到不了 1e-6。
3. **推荐最小 coherent 改动**：把 2D 局部规则从中点升级为**张量积 Gauss-Legendre（2 点主规则 + 3 点对照做误差代理）**，保持自适应框架、tail 证书、确定性、fail-loud、`max_cells=8192` 预算与返回契约不变。数量级论证：相对误差指数从 `N^{-1}` 提升到 `N^{-2}` 起（光滑被积函数），8192 cells 内达 1e-6 是现实预期；常数需 E1/E2 实验确认（§7）。
4. 次选：细分循环性能（堆化，当前全局扫描为 O(N²)，是第二结构性障碍）与分裂方向准则；域截断/变量变换收益低（§4c）；容差定稿为**最后备选**（违反首选方向，需 §7 全套证据 + P0-7 论证）。
5. **重大未决（交 Manager）**：posterior 解除后，t=327 是否仍会在 kelly 层撞 SLOW_PROGRESS **未被当前字节观察**——01:22 的 kelly 证据出自 posterior.jl 13:30 改动之前的旧字节；当前字节「过不了 posterior、到不了 kelly」（summary §1/§6-3）。60/60 可能需要 posterior 与 kelly 两处各有其解。

---

## 1. 现状实现精确读数（file:line + 公式）

### 1.1 2D 求积实现（`src/gate0/posterior.jl:286-383`）

- 调用链：`single_day_decision`（`src/gate0/driver.jl:377-378` full fit；`:387-390` prequential 每历史行）→ `fit_full_posterior`（`posterior.jl:536-540`）→ 2D 路径（`posterior.jl:587-593`，**不显式传 `max_cells` → 用默认 8192**）或 1D 路径（`:612`，`4*max_cells`）。
- 初始网格：`init_grid=4` → 16 cells（`posterior.jl:324-330`）。
- cell 值（中点规则）：`I_c = exp(logf(center) − m)·area`（`:345-348`）。
- 误差代理（Simpson 型）：`E_c = |(四角 exp(logf−m) 均值) − exp(logf(center)−m)|·area`（`:337-344`）。
- 全局：`Z = ΣI_c`、`rel = ΣE_c / Z`（`:349-351`）；停止条件 `rel ≤ tol` 或 cells ≥ `max_cells`（`:352`）；预算耗尽 → `error("Numerical integration did not converge: budget exhausted …")`（`:376`）。
- 细分：全局扫描找最大 `E_c` 的 cell、**沿长边二分**（`:353-371`），每轮重建 `Z`/`err_total`（`O(N)`/轮 ⇒ 全程 `O(N²)` 扫描）。
- 尾质量证书：域边界 8 点（4 角 + 4 边中点）`logf ≤ refval + log(tail_rel) + log(area)`（`:297-319`）；`m_ref` 为粗扫描 mode 参考（`:301-310`）。
- 峰值稳定化：所有值以 `m = max cell 中心值` 归一（`:331`），logZ 补偿（`:381`）。
- 确定性：无 RNG；tie 取 index 最小（`:353-354`）。

### 1.2 被积函数结构与光滑性

`logf(u0, up) = log_evidence(st, lambda_diag(P, u0, up, has_a0)) + log_prior_d035a(u0, up)`（`posterior.jl:575`）。

- `log_prior_d035a`：HalfCauchy-τ（P0-8 后无弱信息特例分支），两端指数衰减（`posterior.jl:158-175`、`:311-315` 注释）。
- `log_evidence`（`posterior.jl:148-155`）：对 `Λ = diag(e^{u0}, e^{up}, …, e^{up})` 的 Cholesky/log-det/二次型组合。`exp(u)` 在 log 坐标全阶光滑且无零点 → **被积函数在 (log α₀, log α_p) 坐标下是光滑函数**（这正是该坐标设计的意义），高阶多项式规则适用。
- 实测形态（02 log）：峰宽 2–4（log 单位）、域 ±5（`u_span=5`）、峰形状正常、尾证书绿。域面积 `(2·5)²=100`，峰外质量经 tail 证书证明可忽略。

### 1.3 收敛率与成本结构（本文推导，与 02 log 互证）

- 令 `N` cells、`h ≈ 1/√N`。对光滑 `f`：
  - 中点规则 per-cell 误差 ≈ `(h²/24)(f_xx+f_yy)·area`；当前误差代理 ≈ `(h²/8)(f_xx+f_yy)·area`——**同阶、约 3× 保守**（待 E2 复核）。
  - 全局 `rel ≈ ΣE_c / Z ~ N·h⁴ = O(1/N)`。✅ 与实测一致（cells ×8、误差 ÷7.7）。
- 外推：从 8192 cells 出发达 1e-6 需 ~6.7e6 cells（02 log）；即使容忍，当前 O(N²) 全局扫描细分在百万级 cells 下不可行——**收敛率与细分成本是两个独立的结构性障碍**。

### 1.4 预算与 fail-loud（现状）

- 2D 默认 `max_cells=8192`（`posterior.jl:290`、Manager 裁决 2026-10-10 与 1D 的 `4*max_cells` 对齐；`:268-272` 注释）。
- `tol` 默认 1e-6（`posterior.jl:289`）；`tail_rel=1e-10`（`:291`）。
- fail-loud 纪律：预算耗尽不返回、文本统一、P0-7 禁止放宽（`:265-272`）。

---

## 2. 层归属论证

**这是数值工程层，不是数学对象层变更。** 判断依据（开发守则 §7 的三层划分）：

- 数学对象是连续积分 `Z = ∫ exp(logf(u)) du` 与节点后验质量 `p_k`；中点/G-L、误差代理、细分顺序、预算全部是「若无限算力则应消失或趋极限」的离散化手段。
- **不改变**：`ResponsePosterior` 契约字段（`alpha_nodes`=cell 中心、`alpha_weights`=归一化质量、`node_log_evidence`、`evidence`，`posterior.jl:228-244`）；`rel ≤ tol` 证书语义；tail 证书判据；确定性；fail-loud 文本与行为；D-060 红线（本层不含 innovation）。
- **会改变**：数值值——节点/权重/`logZ` 随任何规则变化而变。这属于 D-066 的「numerics 配置」范畴：允许，但必须经 refinement 实验定稿，且 tol 减半时结果稳定；禁止由回测选择规则。
- 与 NUMERICAL_INTEGRATION_SPEC 的分层一致：这是「积分误差」侧（规则精度），与「求解误差」侧（Kelly 证书）无关。

---

## 3. 方案对比（a/b/c/d）

| 方案 | 预期收敛率 | 成本 | 改动面 | 证书兼容 | 主要风险 |
|---|---|---|---|---|---|
| (a) 张量 G-L（2 点主 + 3 点对照） | `rel ~ N^{-2}` 起（光滑）；每 cell 误差 `~h⁶` | 求值 13/cell（当前 5/cell，约 ×2.6） | `adaptive_quadrature_2d` 两个内部函数 + docstring | `rel ≤ tol`/tail/fail-loud 原样 | 代理常数未标定；数值值变化需回归重基线 |
| (a′) 3×3 单规则 + 保守差分 | 更高（`~N^{-3}`） | 9–25 求值/cell | 同上 | 同上 | 误差代理需独立规则，成本更高；收益待 E1 |
| (b) 细化策略（堆 + 方向） | 不改收敛率 | 降低大 N 的 O(N²) 扫描 | 循环重构 | 不变 | 改动大；N≤8192 时收益有限（第二阶段） |
| (c) 域截断 / 变量变换 | 不改收敛率量级 | 中性 | 参数/变换 | tail 证书需重验 | 收益低；改变数值值；截断需重新证明 |
| (d) 容差定稿（1e-6 → 更大） | 不适用 | 降预算压力 | 仅数值配置 | 需重定义合同值 | 违反首选方向；P0-7 举证负担重；需 §7 全套 |

**(a) 细节**：一维 G-L 节点/权：2 点 `±1/√3`（权 1,1）、3 点 `0, ±√(3/5)`（权 8/9, 5/9）。张量积到 2D。误差代理 `E_c = s·|I_{3×3} − I_{2×2}|`，安全因子 `s ≥ 1`（初始 1，D-066 定稿；E2 验证代理对真误差的保守性）。数学依据：2×2 张量 G-L 精确每维 ≤3 次单项式，per-cell 误差 `~h⁶`；相对误差指数比中点提升两个数量级起步。

**(b) 细节**：首选保持「最大误差 cell + 长边二分」（确定性、最小改动）；堆化与「按二阶导选择分裂方向」列为第二阶段（仅当 E4 显示扫描/分裂成为热点）。

**(c) 细节**：域从 ±5 收窄受 `tail_rel=exp(-u_span)` 与 tail 证书约束（`posterior.jl:525-531`）；峰宽 2–4 下收窄可省面积但收敛率不变，不解决 1e-6 问题。不建议首版做。

**(d) 细节**：若走此路，需要：(i) 多 fixture 的 rel-vs-cells 收敛曲线证明 tol* 在预算内可达；(ii) tol* 减半时下游权重稳定的敏感性实验（D-066 精神）；(iii) 明确声明 tol* 不是预算驱动而是数值合同定稿，并更新 SPEC。此为后备，不是首选。

---

## 4. 推荐方案与实现规格

**推荐：方案 (a)，保持其余一切不变。**

### 4.1 公式与替换点

- `cell_val(cell)`（替换 `posterior.jl:345-348`）：
  `I_c = area · Σ_{i,j} w_i w_j · exp(logf(x_i, y_j) − m) / 4`，其中 `x_i = (a+b)/2 + (b−a)/2·g_i`，`g = (±1/√3)`，`w = (1,1)`；`area = (b−a)(d−c)`。等价地 `w_i w_j / 4` 归一。
- `cell_err(cell)`（替换 `posterior.jl:337-344`）：`E_c = s · |I_c^{(3,3)} − I_c^{(2,2)}|`；3 点节点 `(0, ±√(3/5))`、权 `(8/9, 5/9)`。
- `vcache`（`posterior.jl:321-322`）保留：键为浮点坐标，G-L 节点坐标确定，缓存行为可复现。
- 一切其余结构——初始网格、最大误差选择、长边二分、`Z`/`err_total` 重算、`rel` 停止、预算 fail-loud、tail 证书、`nodes = cell 中心`、`weights` 归一化、`logZ = log(total)+m`——**原样保留**。

### 4.2 契约保持

- `nodes` 仍是 cell 中心（下游 `ConditionalFit` 每节点一个后验的契约不变）；`weights` 为 G-L 质量归一；`rel` 仍是「相对误差证书」；tail 与错误文本不变。
- 1D 路径（`_adaptive_quadrature_1d`）**首版不动**（现状在 1e-6 下绿；对称升级列为裁决点，避免不必要变更）。
- `max_cells=8192` 不变；若 E1 显示个别形态 8192 不够，登记为新数值配置裁决（不得由预算放宽 tol）。

---

## 5. 测试与验证计划

1. **合成解析答案**：`logf` 取已知 2D 高斯（解析 `Z` 已知）与混合峰（两峰）；断言 (i) 收敛到 1e-6 内；(ii) `rel_est ≥ 真误差`（代理保守）；(iii) 与细网格参考一致。
2. **t=327 同款**：`archive/evidence/gate0_step16/seg6_day_diag.jl`（N=4、seed=2026、T=337、`kelly_tol=5e-3`、`posterior_tol` 默认 1e-6）单跑 t=327：预期过 posterior；记录 rel/cells；若随后撞 kelly（见 §8-2）如实记录。
3. **60-day 段 6 与全链**：`backtest_batched.jl` 段 6（t=327..336）→ 60/60；分段护栏与 held 桥接不变。
4. **模块回归**：`test/gate0/posterior_tests.jl`（40 项）为首要；连带 `quadrature_tests`（55）、`oof_tests`（25）、`prequential_tests`（12）、`driver_tests`（71）、`backtest_tests`（49，按 `BACKTEST_SCENARIO` 分组）。
5. **全量回归口径**：`test/gate0/` 890 项（写作时口径；合并验证实际见 `archive/evidence/gate0_merged_verify_20261010/`——849+49 已绿、场景 8 未执行；913 全量口径见 `AGENTS.md` 第13任段）（按既有分组/独立文件运行策略）；数值值变化导致的既有断言失败必须逐条归因（规则升级的预期效应 vs 真实缺陷），不得靠放宽断言过关。

---

## 6. refinement 实验设计（供 DevOps 执行；命令级草案）

护栏：单命令 ≤60s / RSS 2048MiB；existing 配置；单线程；脚本放受控区（`archive/evidence/...` 或临时目录），不改 `src/test`。

- **E1 收敛曲线**：对 t=327 full-fit 输入（02 诊断脚本可复用）与 2–3 个其他决策日，画 `rel` vs cells（512→8192，新旧规则各一条）。判据：新规则指数显著 > 1（期望 ≥1.7）；8192 内达 1e-6。
- **E2 代理保守性**：合成高斯上比较 `rel_est` 与真误差；要求 `rel_est/真误差 ≥ c`（c≥1 或至少稳定不低估）；确定安全因子 `s`（D-066 定稿）。
- **E3 决策敏感性**：`posterior_tol=1e-6` vs `1e-7`（若预算允许）下 `alpha_weights`、下游 `mu/cov`、Kelly 权重差——验证 1e-6 已足够（D-066 精神：tol 减半则结果稳定）。
- **E4 成本**：单日 posterior 时长（before/after）、60-day 段 6 总时长（护栏内）；`prequential` 路径（每历史行一次 2D 求积）的成本放大单独记录。
- **E5 尾证书回归**：全部 fixture 上 tail 证书保持绿；红即实现缺陷（P0-8 后无预期红）。
- 命令草案形态（示例，实际由 Manager/DevOps 定）：`bin/scoped_run.sh 50 <log> --rss-guard=2048 julia --startup-file=no --project=. <受控诊断脚本>`；t=327 基线用 `archive/evidence/gate0_step16/seg6_day_diag.jl`（其 include 相对路径在新位置仍指向仓库根）。

**与容差定稿的关系**：E1–E3 的目的正是让 `tol=1e-6` 在合同内可达（P0-7 首选路线）；E3 同时为「1e-6 是否足够」提供数值合同证据。若 E1 证明某些形态 8192 内不可达，登记为 fail-loud 的正常红 + 新预算/规则裁决，**不是**放宽 tol。

---

## 7. 现状事实精确读数（driver 默认 vs 60-day vs 测试）

| 位置 | 配置 | 备注 |
|---|---|---|
| `src/gate0/driver.jl:264-265` | `posterior_tol=1e-6`、`posterior_max_cells=2048`（默认） | 透传至 `fit_full_posterior`（`:377-378`、`:387-390`） |
| `src/gate0/posterior.jl:536-540` | `fit_full_posterior(...; tol=1e-6, max_cells=2048, u_span=5.0)` | 2D 路径不显式传 `max_cells`（用默认 8192，`:587-593`）；1D 用 `4*max_cells`（`:612`） |
| `archive/evidence/gate0_step16/backtest_batched.jl:60-63` | 只传 `kelly_tol=5e-3`；`posterior_tol` 未传 → **1e-6** | 60-day 全程严格口径；段 1–5 绿、段 6 t=327 红 |
| `archive/evidence/gate0_step16/seg6_day_diag.jl:31-34` | `posterior_tol` 未传 → 1e-6 | t=327 复现输入 |
| `test/gate0/backtest_tests.jl:92-97` 等 | `posterior_tol=1e-2`（fixture 级降级） | 注释自认「2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）」 |
| 00 log 错误文本 | `budget exhausted at 8192 cells (relative error estimate 8.22401003043056e-4 > tol 1.0e-6)` | 当前字节第一阻塞 |

即：**生产/60-day 用 1e-6，测试层用 1e-2 绕过**——本设计要消灭的正是这个分裂。

---

## 8. 未决与裁决点（交 Manager）

1. **1D 路径是否同步升级**：当前 1D 在 1e-6 下绿；首版 2D-only 是「最小 coherent 改动」，但 1D/2D 将不同款。对称升级（成本低、风险中）作为裁决点。
2. **t=327 的第二阻塞（kelly）**：posterior 解除后是否仍撞 SLOW_PROGRESS **未被当前字节观察**（01:22 证据出自旧 posterior.jl 字节；summary §6-3 明说「当前无法在原始字节上观察 kelly 层的任何修复效果」）。若仍撞，kelly 侧需要独立设计（summary §6-2 已裁定：静态公式路线被否定；若解耦 1e-10 需要证书驱动的自适应机制——须另立设计与验证）。
3. **安全因子与误差代理定稿**：`s` 与「代理 vs 真误差」的保守性由 E2 定稿（D-066），本文初始 `s=1` 只是候选。
4. **max_cells 预算**：8192 对新规则是否足够由 E1 回答；不够则走数值配置裁决，不放宽 tol。
5. **数值值变化的回归面**：规则升级改变 `alpha_weights`/`logZ`，所有消费后验的既有证据需重跑与重基线（预期效应，须显式登记）。

---

## 9. 一句话

t=327 的 posterior 阻塞是「中点规则 O(1/N) × 1e-6 合同 × 8192 预算」的结构性错配；最小 coherent 修法是局部规则升级到张量 Gauss-Legendre（2 点主 + 3 点对照），保留全部证书与 fail-loud 语义，用 E1–E5 定稿数值配置——而不是放宽 tol。

---

*本文为只读调查 + 设计文档；未运行任何命令、未修改 `src/`、`test/`、`archive/` 既有文件。所有实验为设计草案，执行归 DevOps 受控流程。*
