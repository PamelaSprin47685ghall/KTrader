# 真数据首日 full-fit 成本——数学加速设计（FIRST-DAY-FIT-1）

**状态：只读调查 + 设计（2026-10-10）；未实现、未运行。登记供 Manager/SPEC 裁决。**
**层级：数学加速（D-007 第二层）。不改数学/证书/容差；不碰 kelly.jl（并行工作线）。**
**证据：`archive/evidence/gate0_merged_verify3_20261010/A2_fit_timing.log`（read-only 复核）。**

---

## 0. 摘要

首日 full-fit（rows=14049、X=14049×911）在 58s 内未完成；栈热点 = 求积内层
每次 `log_evidence` 的 **P×P cholesky（911³）**，而求积每 cell 需 13 个此类求值。
关键结构：`Sxx + Λ` 里的 `Λ = αp·I + (α0−αp)·e1e1ᵀ` 是**秩一扰动**——
对固定的 `Sxx` 预做一次谱分解，则每节点 `log det` 可 O(P)、Schur 补可 O(P²N)，
单个求值点从 P³ 级降到 P²N 级（P=911、N=4 时约 **50–75×**）。
分两阶段：**阶段 1**（求积内层，纯函数边界内）为本批推荐；
**阶段 2**（最终节点 `_fit_node` 的 V/V_fact 表示）需下游接口裁决，单独登记。

---

## 1. 事故与证据（A2 实测）

- `reconstruct done: rows=14049 X=(14049, 911)`；58s deadline 杀（rc=124、RSS 1715MiB）。
- 栈：`fit_full_posterior(607/661) → adaptive_quadrature_2d(429) → cell_peak(348) → fv(325) →
  logf(646) → log_evidence(151) → cholesky`。BLAS 侧 `dsyrk + dpotrf` 证实是 P×P 分解。
- 含义：**每个求积求值点**（内层 ~6.5×cells 个去重点）触发一次 911³ cholesky；
  加上收敛后**每个最终节点**一次 `_fit_node`（两次 P³ 级：V 与 V_fact）。
- 注：验证批次后的 posterior.jl 形态（`cell_peak` 等命名/行号）与本文引用一致；
  实现时以工作树当前字节为准。

---

## 2. 成本解剖（公式级）

`log_evidence(st, lam)`（posterior.jl:148 起）：

```
log p(Y|α) = const + (N/2)·log|Λ| − (N/2)·log|Sxx+Λ| − (n/2)·log|S(α)|
A = cholesky(Sxx + Λ);  S(α) = s_alpha(st, lam);  （逐点）
```

- 逐点成本：cholesky(A) ≈ P³/3；`s_alpha` 需 A⁻¹·Sxy ≈ P²N；cholesky(S) ≈ N³（小）。
- 求值点数：内层 ≈ 13×cells（vcache 去重后 ~6.5×cells）+ 粗扫描 441 + 最终节点 K=cells。
- P=911、N≈4：P³/3 ≈ 2.5e8 flops ≈ 25–50ms（BLAS 多线程）；6.5×K≈数千点 → 分钟级；
  这正是 58s 内完不成的原因。**与 n 无关**（sufficient_stats 一次性 O(nP²) 不主导）。

---

## 3. 候选 a（推荐）：固定谱预分解 + 秩一/SM 逐点更新

### 3.1 数学机制（恒等式，非近似）

`Λ = αp·I + (α0−αp)·E11`（E11 = e1e1ᵀ），故

```
A(α) = (Sxx + αp·I) + (α0−αp)·e1e1ᵀ = B + δ·e1e1ᵀ
```

预做一次对称谱分解 `Sxx = Q·diag(λ)·Qᵀ`（P³ 一次），并预计算：
`v = Qᵀe1`（P 维）、`W = Qᵀ·Sxy`（P×N）。则每个节点：

- **log|B|** = Σ log(λᵢ+αp) ：O(P)；
- **e1ᵀB⁻¹e1** = Σ vᵢ²/(λᵢ+αp) ：O(P)；
- **log|A|** = log|B| + log(1 + δ·e1ᵀB⁻¹e1)（行列式引理）：O(1)；
- **A⁻¹Sxy** = B⁻¹Sxy − δ·(B⁻¹e1)·((B⁻¹e1)ᵀSxy)/(1+δ·e1ᵀB⁻¹e1)（SM）；
  其中 `B⁻¹Sxy = Q·(W ./ (λ+αp))` ：O(P²N)；`(B⁻¹e1)ᵀSxy = e1ᵀ(B⁻¹Sxy)` ：O(N)；
- **S(α)** = Syy − Sxyᵀ(A⁻¹Sxy)：O(PN²)。

逐点总成本 **O(P²N + P)**（对比现状 O(P³ + P²N)）→ P=911、N≈4 时约 **50–75×** 降幅。

### 3.2 阶段划分

- **阶段 1（本批推荐）**：只改 `log_evidence` 的求值路径——纯标量函数、无接口变更。
  预分解载体在 `fit_full_posterior` 内构造一次（Sxx 已知后），经内部 kw 传入求积的 logf。
  默认关闭时行为与现状逐字节一致（对照路径保留）。
- **阶段 2（登记，待裁决）**：最终节点 `_fit_node` 的 `V`（P×P 显式）与 `V_fact` 每节点两次 P³；
  K≈数百–2000 时其单项即可达数十秒。干净解法是把 `ConditionalFit` 的 V/V_fact 改成
  lazy 谱/SM 表示（消费方 `predictive_moments` 等改为作用口径）——**这是接口/表示层的变更**，
  需要消费链核对与更大验证面；本设计只登记，不擅自纳入本批。

---

## 4. 候选 b/c/d

- **b（primal/dual）**：n=14049 > P=911，现状已在 primal（P×P）空间；dual 为 n×n（2e8 元素）
  **不可行**。结论：不动。
- **c（求积层缓存/统计）**：vcache 已存在于求积内部；粗扫描与 adaptive 的点不重合，无可共享。
  SM 化后粗扫描 441 点 ≈ 0.4–1s，不再是问题；**跨节点共享被数学禁止**（各节点 st 相同但 α 不同——
  预分解共享已由 3.1 承担）。结论：随阶段 1 一并解决，无独立动作。
- **d（其他）**：向量化/SIMD/BLAS 拓扑/线程属 CS 层，登记不执行；kelly.jl 不碰（另线）。

---

## 5. 实现规格（阶段 1 契约）

1. 新增内部载体（未导出，posterior.jl 内）：`SxxSpec = (Q, λ, v=Qᵀe1, W=QᵀSxy)`；
   在 `fit_full_posterior` 的 Sxx 就绪后构造一次（O(P³) 谱分解）。
2. `log_evidence` 增加内部入口（kw `spec = nothing`）：`nothing` 时走现状分支（逐字节不变）；
   非空时走 3.1 的恒等式路径。**仅此一处新增分支；证书/容差/错误文本不变。**
3. 数值纪律：`λᵢ+αp > 0`（αp 由先验支撑）；SM 分母 `1+δ·e1ᵀB⁻¹e1 > 0`（A 正定保证），
   实现时对该分母做有限性/正性断言（fail loudly，不 clamp）；条件数监控为诊断项。
4. 对照验收（沿用已定口径）：同一输入下「SM 路径 vs 逐点 cholesky 路径」对
   `node_log_evidence`、`alpha_weights`、`logZ`、最终残差/预测字段做字段级 tolerance 对照；
   含 NaN/失败行集一致性；tolerance 不得放宽。
5. fail-loud、确定性（无随机）、warm/缓存语义：与既有裁决一致，不新增自由参数。

---

## 6. 预期收益（基于 A2 外推）

- 内层求值（占主导）：单点 P³ → O(P²N)；P=911、N=4 约 50–75×；
- 粗扫描 441 点：分钟级 → 亚秒级；
- 首日 full-fit：内层部分预计从分钟级降到 **数秒**；若阶段 2 不做，`_fit_node`（K×2P³）
  可能仍占数十秒——**首日全链能否落回 58s 取决于 K 的实际值与阶段 2 的裁决**。
- 以上为外推；实测由验证批次（A2 复测 + 对照）给出。

---

## 7. 未决点（待 Manager 裁决）

1. **阶段 1 是否本批实施**（推荐是）；实现范围仅 `posterior.jl` 的 `log_evidence` 分支 +
   `fit_full_posterior` 的载体构造。
2. **阶段 2（`_fit_node`/ConditionalFit 的 lazy 表示）**：涉及下游消费接口，是否立项单独裁决。
3. **SM 数值稳定性**：极端 α（e.g. αp→0⁺、δ 幅度大）下的 roundoff/病态——对照与断言方案见 §5；
   若对照超差，回退逐点 cholesky（默认关闭路径即回退）。
4. **谱分解成本与条件数**：Sxx 近奇异时的分解质量（αp 正则保证正定）；成本一次性 1–3s 外推。
5. **K（最终节点数）未知**：A2 未完成，无法核对 `_fit_node` 项的实际量级；建议复测时记录 K。

---

## 8. 边界

未运行任何命令；未改 `src/`、`test/`；不碰 `kelly.jl`。本文档为设计登记，实现须另立受控施工。
