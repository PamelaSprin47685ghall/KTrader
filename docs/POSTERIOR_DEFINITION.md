# KTrader Posterior Definition — SPEC 级定义草案

**文档状态：草案（Gate 0 重开后 Deliverable 4），交 SPEC 审查。**
**性质：定义与现状盘点，不是实现；本文件的任何目标条款都不改变当前运行时行为，落地须另立受控施工。**
**证据基准：当前工作树源码（HEAD 602b897 + dirty 47 时点）只读核对；全部引用为 `file:line`。**
**纪律：本文中所有「具体选择」均标注「留待 SPEC 审查决定，禁止由回测选择」——任何先验、积分网格、收敛判据都不得以回测收益/Sharpe 作为裁决依据。**

---

## 0. 阅读约定

- $\mathcal H_t$ 为价格历史信息集；$Y$ 为训练目标（relative field 的下一日值）；$\Theta$ 泛指全部未知固定对象。
- 「带先验的未知对象」= 有显式测度（连续后验、离散后验或经验测度）并进入预测积分的对象。
- 「点估计 plug-in」= 由某个判据选出单一数值、当作已知常数代入下游的对象；其自身不确定性不进入任何预测方差。
- 「条件后验」= 在超参固定这一条件下精确积分掉的对象；它是 plug-in 结构的一部分，不是完整 posterior。
- 凡本文件与源码冲突，以源码事实为准；凡本文件与 SPEC 冲突，以 SPEC 为规范，本文件如实记录张力供审查。

---

## 1. 未知固定对象清单

| # | 对象 | 当前身份 | 进入预测的方式 | 关键证据 |
|---|------|----------|----------------|----------|
| 1 | $G$（相对 response，$N\times P$） | 给定 $(\hat\alpha,\hat\Sigma)$ 下的**条件后验**（半积分对象） | 预测均值 $\mu_{rel}=G_c x_t$ 与预测协方差中的 $x'Vx\cdot\hat\Sigma$ 项 | `src/response.jl:1199-1202`、`src/response.jl:1220-1228` |
| 2 | $G_{macro}$（macro response） | 给定 $\hat\alpha_m,\hat\sigma^2$ 下的**条件后验** | $\mu_m=\langle G_m,B_m\rangle$、$var_m=B_m'\hat{Cov}_m B_m$ | `src/response.jl:1146-1155`、`src/response.jl:1218-1219` |
| 3 | $\alpha_m,\alpha_{rel}$（ridge 强度） | **点估计 plug-in**（evidence profile 的 argmax） | 仅以 $\hat\alpha$ 代入谱权重 $d=1/(\lambda+\hat\alpha)$ | `src/response.jl:252`、`src/response.jl:999`、`src/response.jl:1147`、`src/response.jl:1186` |
| 4 | $\Sigma$（relative innovation 协方差，gauge support 内 $N-1$ 维） | **点估计 plug-in** | 作为 $G$ 条件后验的 column covariance；预测协方差 `dot(x,v).*Sigma_rel` | `src/response.jl:999`、`src/response.jl:179-182`、`src/response.jl:1226` |
| 5 | $\sigma^2$（macro innovation 方差） | **点估计 plug-in**（无先验、无后验） | `covm = sig2 .* (...)`、scenario 的 macro 方差 | `src/response.jl:1154-1155`、`src/predict.jl:612` |
| 6 | $d$（fractional memory 指数） | **离散后验**（grid quadrature），先验未显式化 | $p_d\propto\exp(\ell(d))\Delta d$，scenario 离散抽样 | `src/predict.jl:10`、`src/predict.jl:14`、`src/predict.jl:94-96`、`src/predict.jl:613-615` |
| 7 | innovation 残差分布 | **经验测度 plug-in**（OOF row bootstrap） | scenario 逐场景行抽样 + own-row 回退 | `src/predict.jl:452-454`、`src/predict.jl:590-620` |
| 8 | 尺度 $scale_d$ / $v_{bootstrap}$ | $\hat v_{bootstrap}$ 点估计；$scale_d$ 条件于抽中的 $d$ | `scale=sqrt(v_forecasts[d]/v_bootstrap)` | `src/predict.jl:499`、`src/predict.jl:615` |
| 9 | 几何/尺度量：$s_1,s_m,s_\perp,e_0$、`alive_now`、Helmert gauge | 确定性数据变换，**非推断对象** | 作为已知坐标/度量代入 | `src/predict.jl:244-279`、`src/predict.jl:473-481`、`src/numerics.jl:368-380` |

逐项说明：

1—**$G$**。相对通道的完整链条是：先验 $G\mid\Sigma,\alpha\sim MN(0,\Sigma,\alpha^{-1}I)$（SPEC §19），ridge 后验因子 $V=(X'X+\alpha I)^{-1}$（`src/response.jl:169-174`），未约束均值 $\hat G$（`src/response.jl:1187-1191`），再做精确 trace-neutral conditioning $G_c=\hat G-\Omega C'(C\Omega C')^{-1}C\hat G$（`src/response.jl:1192-1202`，实现见 `src/response.jl:131-143`）。`predictive_moments` 对 $Gx_t$ 做精确边缘化（`src/response.jl:1211-1230`），其中 `mu_rel` 是后验均值、`L_rel` 含条件后验的 predictive covariance。结论：$G$ 的不确定性**在 $\hat\alpha,\hat\Sigma$ 固定这一条件下**真实进入了预测；但 $\hat\alpha,\hat\Sigma$ 本身无后验，所以是半积分对象。

2—**$G_{macro}$**。macro 链平行：EB 或固定 ridge 得 $\alpha_m$（`src/response.jl:1146-1149`），$\hat\sigma^2$ 点估计（`src/response.jl:1154`），`covm` 是条件后验协方差（`src/response.jl:1155`），`var_m` 进入 scenario（`src/response.jl:1219`、`src/predict.jl:612`）。

3—**$\alpha_m,\alpha_{rel}$**。`maximize_logalpha` 在 log α 网格上以 evidence 的 argmax 选点（`src/response.jl:212-273`，选定处 `src/response.jl:252`），`optimize_conditioned_eb` 返回 `(alpha,Sigma)` 标量/矩阵对（`src/response.jl:999`）。存在 KKT/边界证书（`src/response.jl:276-282`、`src/response.jl:789-816`）与域界 `EB_ALPHA_MIN/MAX`（`src/response.jl:207-208`）——**证书是驻点证明，不是后验**。SPEC §24 明确 `EB_ALPHA_MIN/MAX` 是数值域界、不是金融超参数。

4—**$\Sigma$**。同上由 `optimize_conditioned_eb` 返回（`src/response.jl:999`），经 `supported_covariance` 投影到 gauge support（`src/response.jl:179-182`）。`EB_COVARIANCE_FLOOR=1e-8`（`src/response.jl:209`）是数值保护（SPEC §25 明示不得当理论风险参数）。

5—**$\sigma^2$**。`sig2=max(sse/max(n-gamma_m,1.0),1e-8)`（`src/response.jl:1154`）是频率派点估计；无 prior、无 posterior、无证书。它通过 `covm`、`var_m`、以及 scenario 的 macro draw 进入预测（`src/predict.jl:612`）。

6—**$d$**。当前是清单中唯一真正「对超参做离散积分」的对象：`causal_fractional_posterior` 返回 $p_d\propto\exp(\ell(d))\cdot\Delta d$（`src/predict.jl:94-96`），网格与 cell 宽度见 `src/predict.jl:10,14`；scenario 从 `cumsum(d_posterior)` 离散抽样（`src/predict.jl:613-615`）。先验目前**隐式**为网格上的均匀×$\Delta d$；SPEC §31 预留「若未来有明确连续 prior $p(d)$，再乘 $p(d_g)$」。

7—**innovation 残差**。`ResidualOracle` 冻结本次 solve 的输入（`src/predict.jl:452-454`），scenario 按观察 mask 决定行抽样、缺失 cell 走 own-row 回退（`src/predict.jl:576-620`）。这是经验测度（empirical predictive），无参数后验；其抽样与数值公式精确（`src/predict.jl:533-545` 的契约注释）。

8—**尺度**。$v_{bootstrap}=\max(\mathrm{Var}(e^{macro,OOF}_{history}),10^{-8})$ 点估计（`src/predict.jl:499`）；$scale_d=\sqrt{v_{T+1}(d)/v_{bootstrap}}$ 在抽中 $d$ 后是条件点值（`src/predict.jl:615`）。

9—**几何量**。`s1`、`s_m`、`s_perp`、`e0`、`alive_now`、Helmert gauge 都是确定性构造（`src/predict.jl:244-279`、`src/numerics.jl:368-380`），不承担后验；它们的错误属于工程 bug，不属 epistemic 缺口。

---

## 2. 当前事实：$\Pi(dG\mid\hat\alpha,\hat\Sigma,\mathcal H)$ 的 plug-in EB 结构

**精确陈述（作为源码事实，非规范断言）：**

$$
\Pi(dG\mid \hat\alpha,\hat\Sigma,\mathcal H)=MN(\hat G_c,\ \hat\Sigma,\ \hat V),\qquad CG=0,
$$

其中 $\hat V$ 是 $(\alpha I+X'X)^{-1}$ 的谱表示（`src/response.jl:169-174`），$\hat G_c$ 是 trace-neutral 条件均值（`src/response.jl:1192-1202`），$\hat\Sigma$ 是 EB 返回的单一矩阵（`src/response.jl:999`）。决策时刻只解析边缘化 $Gx_t$（`src/response.jl:1211-1230`）：不抽整张 $G$，符合 SPEC §33。

- **$\hat\alpha$、$\hat\Sigma$ 的来源**：`optimize_conditioned_eb` 在 trace-neutral support 下交替优化 evidence，直至 KKT/floor 证书通过或 fail-closed 抛错（`src/response.jl:934-1122`；证书 `src/response.jl:789-816`）。这是 profile EB 的最大化，不是对 $\alpha,\Sigma$ 的后验积分。
- **macro 侧同构**：`optimize_matrix_normal_eb` 返回 $\hat\alpha_m$、$\hat\Sigma_m$ 与证书（`src/response.jl:351-380`），`sig2` 另行点估计（`src/response.jl:1154`）。
- **fold 路径**：每个 fold 用 train Grams 独立求自己的 $\hat\alpha^{(-f)},\hat\Sigma^{(-f)}$（`src/predict.jl:405-447`），held-out 行不进入 fold EB；warm start 取上一决策日同 fold 的 α（`src/predict.jl:441`），属严格过去信息。增量路径经 `solve_current!` → `_fit_prepared_v1` → `solve` 共用同一装配（`src/incremental.jl:951-957`、`src/predict.jl:338-340`），不存在第二套 posterior。
- **scenario 的其余层**：$d$ 离散后验（`src/predict.jl:613-615`）、残差行经验 bootstrap（`src/predict.jl:590-620`）、fractional quasi-likelihood（`src/predict.jl:101-119`）。

**必须钉死的声明：**

> **当前结构是 epistemic approximation，不是完整 posterior。** 准确地说：它是条件于超参点估计 $(\hat\alpha,\hat\Sigma,\hat\sigma^2,\hat v_{bootstrap})$ 与经验残差测度的「部分后验预测」；完整对象 $\Pi(dG,d\alpha,d\Sigma,d\sigma^2,\dots\mid\mathcal H)$ **未被计算，也未被证明可由当前结构近似**。任何把它称为「完整 posterior」「Bayesian posterior predictive」的措辞，除非另行给出证明，均不成立。

差距的精确清单：
1. $\alpha,\Sigma,\sigma^2$ 无后验测度（第 1 节 #3/#4/#5）——其不确定性未进入任何预测方差；
2. $d$ 有离散后验但先验未显式化（#6）；
3. innovation 是经验测度而非参数化后验（#7）；
4. fractional likelihood 是 quasi-likelihood（`src/predict.jl:101-119`），SPEC §30 本就只声称 KISS 版。

---

## 3. 目标定义：$\Pi(dG,d\alpha,d\Sigma\mid\mathcal H)$ 的层次模型抽象形式

以下为**目标抽象**，用于 SPEC 审查；每一条具体选择都留白。

### 3.1 层次结构（抽象形式）

$$
\begin{aligned}
&\text{先验（结构固定，形式留白）：}\\
&\qquad G\mid \Sigma,\alpha \sim MN(0,\Sigma,\alpha^{-1}I),\qquad C\,g=0,\quad g=\mathrm{vec}(G);\\
&\qquad \Sigma \sim p_\Sigma(\cdot)\ \text{（支撑在 relative gauge support 内、PSD + 正定下界）};\\
&\qquad \alpha \sim p_\alpha(\cdot)>0;\\
&\text{似然：}\qquad Y = XG' + E,\qquad E\ \text{的行结构按 innovation 模型（见第 6/7 项）};\\
&\text{后验：}\qquad \pi(G,\alpha,\Sigma\mid Y)\ \propto\ p(Y\mid G,\Sigma)\,p(G\mid\Sigma,\alpha)\,p_\Sigma(\Sigma)\,p_\alpha(\alpha)\,\mathbf 1[Cg=0];\\
&\text{预测：}\qquad p(r_{t+1}\mid\mathcal H)=\int p(r_{t+1}\mid G,\alpha,\Sigma,\text{innovation})\,d\Pi(G,\alpha,\Sigma\mid\mathcal H).
\end{aligned}
$$

### 3.2 留白与纪律

| 待定项 | 候选方向（**留待 SPEC 审查决定，禁止由回测选择**） | 当前状态 |
|---|---|---|
| $p_\alpha(\alpha)$ | log-uniform / half-Cauchy / 其他适当先验；`EB_ALPHA_MIN/MAX` 只作数值域界，不得充当先验截断的金融理由 | 不存在 |
| $p_\Sigma(\Sigma)$ | gauge support 上的 inverse-Wishart（需定自由度/尺度）或 non-informative + 与 floor 明确区分的正定约束 | 不存在 |
| $\sigma^2$ 是否升格 | 独立 inverse-gamma 或 profile | 点估计 |
| $d$ 的显式先验 | SPEC §31 预留的 $p(d)$ | 隐式均匀 |
| innovation 参数化 | 保持经验 bootstrap（声明为 empirical predictive）或参数后验 | 经验测度 |

**纪律条款**：上表任何选择都不得依据回测 Sharpe、收益或「哪个先验让曲线好看」来决定；候选评估只能用统计恰当性、refinement 收敛与性质测试（SPEC §95、开发守则 §24）。

### 3.3 目标必须保持的不变量

- trace-neutral 约束必须在**联合后验的支撑上**成立（SPEC §21-22、§61），不能退化为「只对均值减 trace」；
- $\alpha$ 的后验必须带 evidence/先验的联合密度，不能退回「profile argmax + 声称积分」；
- 任何积分近似都必须有 refinement certificate 且 fail-closed（SPEC §56）；
- $G$ 的解析边缘化（§33）仍然是首选，不因引入超参后验而退化为抽整张 $G$。

---

## 4. 积分策略顺序

按「便宜→贵」与证据阶梯一致排序；能用便宜阶梯裁明的事，不劳烦贵的（开发守则条款 B、SPEC §67）。

**第 1 阶（解析边缘化，必须做尽）。**
$G\mid\Sigma,\alpha$ 是高斯，$Gx_t$ 的 predictive moments 已有解析式（`src/response.jl:1211-1230`）；trace-neutral conditioning 只在 14×14 的约束空间求解（`src/response.jl:1192-1199` 注释；实现 `src/response.jl:131-143`）。目标：对 $(\alpha,\Sigma)$ 固定下的所有高斯块继续解析积分，禁止对 $G$ 整体做随机采样再求平均。

**第 2 阶（scalar 超参确定性 quadrature）。**
- $\alpha$（1 维、正）：目标为 $\pi(\alpha\mid Y)\propto E(\alpha)\,p_\alpha(\alpha)$ 的确定性 quadrature；网格/节点密度与收敛判据（细化到 evidence 曲线与积分量稳定）**留待 SPEC 审查决定**。当前 `maximize_logalpha` 的 33 节点+bracket 是**优化器的候选机制**（`src/response.jl:212-273`），不是后验积分，不得直接挪用为「已有积分」。
- $d$ 已在此层（`src/predict.jl:94-96`），保持 $\Delta d$ 与显式先验；细化时先验质量不得漂移（SPEC §64）。

**第 3 阶（低维剩余积分，收敛受控）。**
- $\Sigma$ 在 gauge support 内为 $(N-1)$ 维 PSD 对象；候选为 Laplace（曲率可用现有自然梯度/Jacobian 结构，`src/response.jl:646-680`、`src/response.jl:583-624`）或对参数化（特征值/Cholesky 坐标）做确定性 quadrature。
- 必须有 refinement certificate；不收敛即 fail-closed，不得静默返回非驻点/非收敛结果（SPEC §56）。

**第 4 阶（随机后验采样，最后手段）。**
仅当 1—3 阶被证明不可行时启用；固定 seed、可复现（SPEC §36），且采样误差必须独立报告、不得把「scenario 数」当作后验积分精度的替代。

**本模型的具体边缘化顺序（目标建议，供 SPEC 裁决）：**
1. 内层：给定 $(\alpha,\Sigma)$ 解析积分 $G$（已具备）；
2. 次内层：$\alpha\mid\Sigma$（或 $\alpha$ 的联合 profile）做 1 维确定性 quadrature；
3. 外层：$\Sigma$ 做 Laplace/低维 quadrature；
4. 再外层：$d$ 与 innovation 的后验（$d$ 已是 1 维；innovation 视第 3.2 节决定）；
5. macro 链（$\alpha_m,\sigma^2$）平行处理。
注意：当前实现的交替**优化**（`src/response.jl:962-1114`）与上述积分顺序是不同对象；不得把优化器迭代路径当作积分路径。

---

## 5. 预测方差分解

**规范目标分解（SPEC §26 的明确形式）：**

$$
\mathrm{Var}(r\mid\mathcal H)
=
\underbrace{\mathbb E_\Theta[\mathrm{Var}(r\mid\mathcal H,\Theta)]}_{\text{within}}
+
\underbrace{\mathrm{Var}_\Theta[\mathbb E(r\mid\mathcal H,\Theta)]}_{\text{between}},
\qquad \Theta=(G,\alpha,\Sigma,\sigma^2,\dots).
$$

**当前实现的对应（诚实映射）：**

- 已做：在**条件后验**近似下，$G$ 的条件不确定性进入了 predictive covariance——相对通道 `dot(x,v).*Sigma_rel` 与 trace-conditioning 修正（`src/response.jl:1226`），macro 通道 `var_m`（`src/response.jl:1219`）；innovation 方差由残差 bootstrap × $scale_d^2$ 近似（`src/predict.jl:615,620`）。
- **未做（根因）**：$\alpha,\Sigma,\sigma^2$ 固定于点估计（`src/response.jl:999`、`src/response.jl:1154`），因此条件均值 $\mathbb E(r\mid\mathcal H,\hat\alpha,\hat\Sigma,\dots)$ 之上不存在超参后验的变异——**between 项中来自超参的贡献恒为零**。这不是数值误差，结构上就没有测度可积。
- 同时注意：条件后验对 $G$ 的积分所贡献的方差，严格属于「对 $G$ 的 between 部分已在条件层完成」；一旦把 $\Theta$ 定义为含超参的完整对象，当前实现只完成了 $\Theta$ 的一个切片。

**报告要求：**
1. 分解必须**可报告**：within（innovation，含 bootstrap/scale）与 between（$G$ 条件部分 + 超参部分）分别给出数值；
2. 当前状态下必须如实标注「超参 between 分量：未计算（=0 by construction，非统计结论）」；
3. 任何声称「posterior uncertainty 已完整进入 scenarios」的陈述，必须先给出 between 项的非零超参分量或证明其可忽略——否则撤回该措辞。

---

## 6. 去留判据：保留 plug-in vs 实现完整积分

两条路线都允许，但各自的最小证据形态不同；选择不得由回测驱动。

**路线 A：保留 plug-in（显式标记为 epistemic approximation）。**
最小证据形态（全部可与回测无关地取得）：
1. SPEC 文本显式声明本近似（本文第 2 节的形式）；
2. 超参后验集中性证据：$\alpha$、$\Sigma$ 在最优点的局部曲率/证据敏感性报告（例如 evidence 沿 $\log\alpha$ 的平坦度、$\Sigma$ 特征方向的 Laplace 尺度），并明确「集中 ⇏ 方差为零」；
3. 至少一个可复现的差异界：plug-in predictive 方差 vs 条件后验方差的相对差距，说明被忽略的超参分量量级；
4. 若做不出 2/3，则以更弱的形态保留：「未测量，仅声明近似」——这也允许，但不得再称完整 posterior。

**路线 B：实现完整积分。**
最小证据形态：
1. 先验 $p_\alpha,p_\Sigma$（及 $\sigma^2$、$d$、innovation 的选择）经 SPEC 审查定稿，**禁止由回测选择**；
2. 积分实现带 refinement certificate（类似 scenario 的 $|w_{2S}-w_S|_1$ 收敛口径，SPEC §37/§64），不收敛 fail-closed；
3. 与解析/plug-in 对照的回归测试：固定 seed 下，$E\Pi[\cdot]$ 与条件矩在「先验退化为点质量」极限下一致；
4. predictive variance 分解（第 5 节）双分量可报告；权重/方差的差异在受控 tolerance 内说明；
5. 任何情况下保留 reference 路径（开发守则条款 21）。

**共同铁律：**
- **不得用 Sharpe / 回测收益/换手 决策 A 或 B，或决定任何先验、网格、积分精度**（SPEC §95、开发守则 §24）；
- 路线切换必须走 SPEC 审查，不接受「优化器顺路升级为积分」；
- 不论哪条路线，当前源码行为不变更——本文件不触发任何施工。

---

## 7. 未决问题清单（交 SPEC 审查）

1. **$\alpha$ 先验**：形式（log-uniform / half-Cauchy / 其他）、域与 `EB_ALPHA_MIN/MAX` 的关系；若保留边界，边界撞墙时的报告义务。
2. **$\Sigma$ 先验**：gauge support 上的具体分布（inverse-Wishart 参数？non-informative？）以及它与 `EB_COVARIANCE_FLOOR`（数值保护，`src/response.jl:209`）的边界如何措辞区分。
3. **$\Sigma$ 积分的表示与收敛判据**：Laplace vs 参数化 quadrature；$(N-1)$ 维 PSD 域的积分误差如何证书化。
4. **$\sigma^2$**：是否升格为带 inverse-gamma 的未知量；macro `covm`（`src/response.jl:1155`）随之是否变化。
5. **$d$ 的连续先验**：SPEC §31 预留项落地与否；若不落地，显式声明「均匀网格先验」为当前规范。
6. **innovation**：保持经验 bootstrap（并正式声明为 empirical predictive）还是参数化；经验测度的 plug-in 性质是否写入规范。
7. **报告格式**：第 5 节方差分解的字段、精度与「不可计算」的标注规范。
8. **A/B 决策门槛**：谁批准、以什么证据翻案、refinement 到何种程度算「做尽」。
9. **API 表达**：`predictive_moments` 返回的 NamedTuple（`mu_m,var_m,mu_rel,L_rel,x_features`，`src/response.jl:1229`）是否需要携带「哪些分量含超参后验」的结构性标注，供下游 scenario/报告使用。
10. **SPEC §26 措辞对齐**：$\Pi$ 的定义域（含不含超参）必须在 SPEC 全文钉死；当前「posterior uncertainty 进入 predictive」的表述在两种定义域下含义不同，是本清单里最需要先裁决的一项。

---

## 8. 与既有 SPEC 的关系（一致点与张力）

**一致（作为规范执行无误）：**
- SPEC §19/§33：Matrix-Normal 条件后验与决策时刻精确边缘化——实现相符（`src/response.jl:1192-1202`、`src/response.jl:1211-1230`）；
- SPEC §21-22：trace neutrality 在 covariance support 内、非 post-hoc 减均值——实现相符（`src/response.jl:1192-1199` 注释明确拒绝欧氏减均值）；
- SPEC §25/§24：EB 证书与边界为数值域界——实现相符（`src/response.jl:207-209`、`src/response.jl:789-816`）；
- SPEC §30/§31：fractional KISS 与 $\Delta d$ quadrature mass——实现相符（`src/predict.jl:94-96`）。

**张力/待裁决（本文件的中心议题）：**
- SPEC §26 写 $\bar\mu=\mathbb E_\Pi[\mu_G]$，$\bar\Sigma=\mathbb E_\Pi[\Sigma_G]+\mathrm{Cov}_\Pi(\mu_G)$。若 $\Pi$ 读作**完整联合后验**，当前实现不满足（超参无测度，第二个 $\mathbb E_\Pi[\Sigma_G]$ 用 $\hat\Sigma$ 代入）；若 $\Pi$ 读作**条件于超参点估计的条件后验**，实现满足。两种读法给出的规范义务（是否必须实现完整积分）完全不同——这是与 SPEC 的实质冲突点，需审查裁决。
- SPEC §95 禁止「因某选择让回测好看而选择」——本文件全部留白项遵守；但若 SPEC 不补先验形式，完整积分在规范上不可执行。
- 历史注记：本文不改写任何既有 SPEC 文本、不改动 README/AGENTS；冲突以本清单形式提交，不自行裁决。

---

*（本文件为 Gate 0 重开后 Deliverable 4 交付物；未运行任何命令，未修改 src/、test/、README.md、AGENTS.md 或任何既有文档。）*
