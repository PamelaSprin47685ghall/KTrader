# prequential 残差生成成本——数学加速设计（PREQUENTIAL-COST-1）

**状态：设计文档（2026-10-10）；只读调查 + 设计，未实现、未运行。登记供 SPEC / Manager 裁决。**
**层级：数学加速（D-007 第二层）——同一数学对象的更优计算；不改数学、不改证书、不得放宽容差或减点；reference equality 必须可验证。**
**证据边界：源码引用来自此前只读核对（file:line 以「约」标注）；测量数字按 Manager 转述的 `archive/evidence/gate0_merged_verify_20261010/`（A3/A3b）引用，本文未逐一复核日志正文。**

---

## 0. 摘要

t=327 单日链超时（55s/58s，rc=124）的根因是 **prequnetial 残差生成对 67 个历史行逐行调用 `fit_full_posterior`**，
每行一次完整 2D 求积（train=4/20/70 行 ≈ 0.85s/3.12s/4.66s；67 行累计外推 ≈ 200s/日；新求积规则每 cell 13 点 ×2.6）。
成本归因：单行耗时随 train 行数的增长远慢于行数增长（行数 ×17.5 → 耗时 ×5.5），
**主导是求积固定开销（粗扫描 + adaptive cells 的评估次数），不是 Gram/充分统计规模**。
推荐组合：**(A) 前缀充分统计增量 + (B) 跨行域中心/初始划分 warm start（证书闭环 + 冷路径回退）+ (C) 同 (st, λ) 评估精确缓存**；
预期 3–6× 单日 prequential 成本下降（外推估算，须实现后实测）；若不足，再进入 CS 加速层（D 项，非本层）。

---

## 1. 成本结构解剖（基于已读源码）

### 1.1 调用链

- `driver.jl` 步骤 8（约 L364–380）：`prequential_residual_rows(X_tr, Y_tr, E_active; tol, max_cells, u_span)`——`X_tr/Y_tr = mp.X[rows,:]/mp.Y[rows,:]`。
- `oof.jl` 的 `prequential_residual_rows`（约 L302–395）：`for s in 1:n`，每行 `fit_full_posterior(X[1:s-1,:], Y[1:s-1,:]; ...)`，
  然后 `r_mode = Y[s,:] - B_mix * X[s,:]`、`out[s,:] = E*r_mode`；单行 propriety/求积失败 → 该行 NaN（行级语义不变）。
- `posterior.jl` 的 `fit_full_posterior`（约 L536 起）：
  1. `sufficient_stats(X, Y)`（每行重算，O(s·P·N)）；
  2. A2 数据层 gate（L548–551：`n ≥ N && rank(Syy) = N`，fail loudly）；
  3. DC 形态（P=1+14N）粗扫描 `21×21 = 441` 次 `logf = log_evidence + log_prior_d035a`（每次含谱/Cholesky 级成本）；
  4. `adaptive_quadrature_2d`（每 cell 13 点评估；tail 证书 + 预算耗尽 fail loudly）；
  5. 每节点 `_fit_node`（O(P³) Cholesky + O(N³) 项）。

### 1.2 测量与归因（Manager 转述）

| train 行数 | 单次 fit 耗时 |
|---|---|
| 4 | 0.85s |
| 20 | 3.12s |
| 70 | 4.66s |

- 67 行累计外推 ≈ 200s/日；t=327 单日链两次超时（55s/58s）是 50s deadline 先杀，远未到外推全量。
- 每 cell 13 点（新求积规则，×2.6）——求积评估次数再次抬升。
- **归因**：0.85s → 4.66s 的斜率对应「每行多一行数据」的边际成本小；固定块（粗扫描 441 点 + cells × 13）主导。
  因此加速优先级：**求积层（B/C）> 统计层（A）**；A 仍值得做（去掉 O(s·P·N) 重算，且为 B 提供干净接口）。

---

## 2. 候选方案对比

### A. 前缀充分统计增量（采纳，低风险）

- **数学**：`(n, Sxx, Sxy, Syy)` 对行追加是精确线性更新（rank-one/一行外积）；与 `sufficient_stats` 从头重算在浮点上仅有求和序差异。
- **实现**：`prequential_residual_rows` 内维护前缀统计序列（起点 = 第 1 行统计），逐行 `append!`；
  `fit_full_posterior` 增加**内部**入口接受预计算统计（不扩大公共 API；经未导出 kw 或同 module helper）。
- **收益**：去除每行 O(s·P·N) 重算；对 s≈70、N_act 中等是常数因子（占单行成本的小头，估计 5–15%）。
- **风险**：低。验收：增量统计 vs 重算统计逐字段 atol（沿用既有差分对照口径 1e-10 量级）。

### B. 跨行域中心 / 初始划分 warm start（采纳，主收益）

- **规范依据**：D-038 明确「粗扫描 mode 仅作初始 bracket/域中心，不定义 posterior」；开发守则 §47「允许 warm start / fewer iterations，条件是达到同一个 mathematical certificate」。
  即：**初始域/初始 cell 划分的复用不改变数学对象**，只要最终收敛判据与冷路径相同。
- **做法**：行 s+1 以行 s 的粗扫描 mode（m0, mp）为初始域中心（跳过 441 点全扫描，改小半径验证扫描 + 直接 adaptive）；
  并可将行 s 的最终 cell 划分作为行 s+1 的 adaptive 初始表。
- **安全闭环（必须）**：tail 证书与 `rel_err ≤ tol` 判据**不放松**；若 warm 域导致证书红/预算耗尽 → **回退一次冷启动全路径**；仍失败 → 该行 NaN（现行行级语义）。
- **收益（外推）**：粗扫描 441 → ~25 点；更窄初域通常显著减少 adaptive cell 数（收敛到同一 tol 所需 cells 减少）——两者相加是 3–6× 的主来源。
- **风险**：跨行复用改变求积**路径**（节点集可能不同）→ 不是逐位等价，须按「字段级 tolerance + 证书一致」验收（口径见 §4；请 Manager 裁决）。

### C. 同 (st, λ) 评估精确缓存（采纳，保守子集）

- **内容**：单行内部，粗扫描与 adaptive 求积若在**同一 (u0, up) 点**评估 `logf`，精确复用（浮点键相等才命中）；不跨行复用（st 不同）。
- **收益**：消除粗扫描与 adaptive 的重复评估（现状各自独立评估）。中等偏低。
- **风险**：零数学变化（同点同值）。

### D. CS 层（登记，不在本层执行）

- 441 点 logf 的批量向量化、BLAS 调度、线程拓扑：属 D-007 第三层（计算机科学加速）；本设计不承诺、不混入。
  仅登记为「若 A+B+C 后仍不足 60-day 护栏」的下一层候选。

---

## 3. 推荐与预期收益

**推荐：A + B（证书闭环 + 冷路径回退）+ C（精确缓存）**，分三步落地，每步独立可验证：

1. **Step-1（A+C）**：统计增量 + 评估缓存。无求积路径变化 → 可做**逐字段强对照**（近逐位）。
2. **Step-2（B）**：warm start + 证书闭环。验收按字段级 tolerance + 证书一致性 + 冷/暖双跑对照。
3. **Step-3**：实测（微基准 + 单日链 + 10-day 分段），若仍未回 60s 护栏，开 D 层评估。

**预期收益（外推，须实测）**：
- 现状：单行 ≈ 0.85–4.66s（依 train 行数）、单日 67 行 ≈ 200s（外推）。
- 目标：单日 prequential 部分 ≤ 30–40s（含其它环节后单日链 <50s），即 **5–7×** 下降；
  其中 B 为主（3–5×）、A/C 为辅（合计 1.3–1.6×）。
- 60-day 分段：若单日 ≈ 5–8s，则 5–8 日/段 <60s 可行；501-day 仍按批次策略（不在本设计承诺内）。

---

## 4. 实现规格（契约）

1. **接口（不扩大公共面）**：
   - `prequential_residual_rows` 内部：前缀统计序列 + warm state（`m0/mp/初始 cells`）作为**局部变量**在行循环内传递；
   - `fit_full_posterior` 增加未导出入口参数（如 `stats = nothing`、`warm = nothing`）——默认 `nothing` 时行为与现状**逐字节一致**；
   - 公共 API、导出面、测试可见签名不变。
2. **warm 契约**：warm 仅提供「初始域中心 / 初始 cell 表」；收敛判据、tail 证书、fail-loud 文本全部不变；
   证书红 → 冷路径重试一次；二次失败 → NaN（行级不可定义语义不变）。
3. **reference equality 验收（字段清单）**：对同一输入，冷（现状路径）vs 暖双跑，逐行对照：
   - `alpha_nodes` / `alpha_weights`（按 tol 口径）；`logZ`；`B̂_mix`（或残差行 `ε_s`，atol 从严）；
   - NaN 行集合（必须逐一相同）；证书字段（valid/收敛标志）；
   - tolerance 禁止放宽（D-066）；对照测试进 CI（冷路径为 reference 保留）。
4. **fail-loud 保持**：预算耗尽、证书失败、非有限值的错误文本与路径不变。
5. **不得用回测收益选择**：warm 半径、缓存策略、回退策略均为数值工程选择，其验收仅由字段对照 + 证书决定。

---

## 5. 未决点（待 Manager / SPEC 裁决）

1. **reference equality 口径**：跨求积路径不可能逐位相同；建议「字段级 tolerance（沿用 D-066 定稿口径）+ 证书一致 + NaN 行集相同」；是否接受，请裁决。
2. **warm start 半径/回退策略参数**：属数值配置（非数学），须经 refinement 定稿、禁止由运行预算决定。
3. **A3/A3b 测量复核**：本文按转述引用（4/20/70 行 → 0.85/3.12/4.66s；每 cell 13 点 ×2.6）；实现前请 DevOps 复核原始 log 与口径。
4. **收益目标确认**：5–7× 为外推估算；若实测不足，D 层（CS 加速）的开启标准与预算由 owner 定。
5. **与既有裁决的关系**：本设计不动 D-034/D-038/D-043/D-067 的任何语义；warm start 的规范性引用为开发守则 §47 与 D-038 的「mode 仅作域中心」语义——若 SPEC 另有更严口径，以 SPEC 为准。

---

## 6. 参考

- 源码：`src/gate0/driver.jl`（步骤 8，约 L364–380）；`src/gate0/oof.jl`（`prequential_residual_rows`，约 L302–395）；
  `src/gate0/posterior.jl`（`fit_full_posterior` 约 L536 起、A2 gate L548–551、粗扫描与 `adaptive_quadrature_2d`、`_fit_node`）；
- 规范：AGENTS.md D-038（mode 仅作域中心）、D-066（tolerance 禁由预算决定）、D-067（fail loud）、SPEC §42 数学加速清单（#1 解析积分、#2 alpha quadrature cache、#3 sufficient statistics 等）；开发守则 §47/§70（warm start 证书语义）；
- 证据：`archive/evidence/gate0_merged_verify_20261010/`（A3/A3b 测量；按转述引用）。

---

## 7. 边界

本设计未运行任何命令、未改 `src/`、`test/`；实现与实测须另立受控施工与 DevOps 运行。


---

## 8. 实现落盘记录（2026-10-10）

- **实现（两文件五处，锚点快照校验）**：`src/gate0/posterior.jl` 1 处——
  `adaptive_quadrature_2d` 细分循环改增量维护（vals/errs 数组、O(1) 全局和修正、
  读存量 max-err 扫描）；`src/gate0/oof.jl` 4 处——`Threads.@threads` 行级并行、
  错误槽位（非行级错误循环后主线程按行序 throw，避免 TaskFailedException 包装）、
  `filled_flags`（Vector{Bool}，避 BitVector 位打包竞态）、结尾 `any(filled_flags)`。
- **默认设计**：并行无条件启用（单线程会话自动退化顺序执行）；无新随机与自由参数；
  每行只读共享输入、按行索引写 out ⇒ 线程数不改变结果。
- **测试**：`test/gate0/prequential_tests.jl` 末尾新 testset（10 条）——增量 vs 全量
  参照对照（rel≤tol、|ΔlogZ|≤1e-8、解析锚 log(2π)、cell 数差 ≤2）；手动逐行逐位
  对照（isequal，NaN 安全）；确定性重放。
- **风险点**：停止点 roundoff 差 ≤2 cells（对照口径已按 tolerance）；多线程会话下
  BLAS 嵌套可能抵消并行收益（正确性无碍）；错误槽位语义待运行证明。
- **验证**：见 §5 清单 + 本记录（before/after 单日链、67 行总时长、60-day 分段、RSS）。