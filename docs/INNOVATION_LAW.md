# KTrader Innovation Law — SPEC 级定义草案

**文档状态：草案（Gate 0 重开后 Deliverable 5），交 SPEC 审查。**
**性质：定义与现状盘点，不是实现；本文件的任何目标条款都不改变当前运行时行为，落地须另立受控施工。**
**证据基准：当前工作树源码只读核对（HEAD 602b897 + dirty 47 时点）；全部引用为 `file:line`。**
**纪律：本文中所有「具体选择」均标注「留待 SPEC 审查决定，禁止由回测选择」——任何先验、kernel 形式、正则化与收敛判据都不得以回测收益/Sharpe 作为裁决依据（SPEC §95、开发守则 §24）。**

---

## 0. 阅读约定

- $\mathcal H_t$ 为价格历史信息集；$\varepsilon$ 为 OOF 残差，$\varepsilon_t = r_{t+1}-\hat\mu_t^{(-fold)}$（`src/predict.jl:5` 的模块定义）。
- $V_t(d)$ 为本文目标对象：vector innovation memory；$z$ 为标准化形状（standardized shape）。
- 「带先验的未知对象」「点估计 plug-in」「条件后验」等阅读约定沿用 `docs/POSTERIOR_DEFINITION.md`（下称 D4，Deliverable 4）第 0 节；本文不复制 D4 的未知对象清单，只处理 innovation 这一层。
- 凡本文件与源码冲突，以源码事实为准；凡本文件与 SPEC 冲突，以 SPEC 为规范，本文件如实记录张力供审查。
- 本文的「目标」一词指**定义草案中的规范对象**，不声称已实现，也不构成施工授权。

---

## 1. 当前实现的精确映射

### 1.1 链条：从 macro OOF 残差到场景缩放（四步）

**第一步：macro OOF 标量残差序列。**
`solve(prep)` 在决策时刻调用 `macro_residual_series(res_history)`（`src/predict.jl:493-495`），得到 $e \in \mathbb R^{n_{res}}$。该序列由 `ResidualOracle` 每次调用重算（`src/residual_oracle.jl:216-260`），公式为

```text
e[idx] = sum(r[O])/sqrt(c) - mu_macro*sw/c - dot(h, x_t)/sqrt(c)
```

（`src/residual_oracle.jl:257`；契约注释 `src/residual_oracle.jl:39-49`）；空观察行记 $e_{idx}=0.0$（`src/residual_oracle.jl:239`）。关键事实：**这是 macro 通道的一个标量投影，不是 $N$ 维残差向量**——全 $N$ 维残差只在被显式请求的行上计算（`src/residual_oracle.jl:50-53`、`150-211`）。

**第二步：fractional posterior。**
`causal_fractional_posterior(e_res_m)`（`src/predict.jl:496-498`）在网格 `DGRID_V1`（`src/predict.jl:10`）上经 FFT 求 quasi-log-likelihood 与 per-$d$ 预测方差。实现为 `fractional_likelihood!`（`src/predict.jl:101-119`）：

- `variance = max(work.output[t-1]/cs[t-1], 1e-10)`（`src/predict.jl:112`）——这是 causal conditional variance $v_t(d)$ 的有限卷积实现；
- `counts[g] = cumsum(frac_weights(d, L÷2))`（`src/numerics.jl:425`），`kernels[g]` 取 `frac_weights` 的正变换（`src/numerics.jl:419-426`）；
- fractional kernel 由 `frac_weights(d,n)` 递推给出（`src/predict.jl:12`；SPEC §31 的 $\pi_k$）。

后验权重 $p_d \propto \exp(\ell(d))\cdot \Delta d$，其中 $\Delta d$ = `DELTA_D_V1`：`log_weights = ll .+ log.(DELTA_D_V1)`（`src/predict.jl:14`、`94-96`）。

**第三步：per-$d$ 预测方差与 bootstrap 锚点。**

- `forecasts[g] = max(work.output[T]/cs[T], 1e-10)`（`src/predict.jl:116`）——即 $v_{T+1}(d)$ 的当前实现（SPEC §32）；
- `v_bootstrap = max(var(e_res_m), 1e-8)`（`src/predict.jl:499`）。

**第四步：场景缩放（`generate_scenarios_v1`，`src/predict.jl:546-625`）。**

- 均匀行抽样：`row = min(floor(Int, ur_of(s)*T)+1, T)`（`src/predict.jl:591`），其中 `ur_of(s) = uniforms[s,2]`（`src/predict.jl:578`）、`uniforms = rand(rng, S, N+2)`（`src/predict.jl:552`）。**同一场景的全部资产共享同一个历史行**（`src/predict.jl:616` 的 `k0 = position[row_of[s]]`，`src/predict.jl:618-619`），即「横截面 shape = 该历史行的经验 shape」。
- 缺失 cell 的 own-row 回退：`back_of[s,j]` 指向该资产自身的有效残差行（`src/predict.jl:596-602`），有效行定义 `own[j] = {idx : isfinite(r[ts_total[idx]+1, j])}`（`src/residual_oracle.jl:54-56`、`264-274`）。
- $d$ 抽样：`d = min(searchsortedfirst(cumsum(d_posterior), ud), length(cumulative))`（`src/predict.jl:613-614`）。
- 缩放与合成：`scale = sqrt(model.v_forecasts[d]/model.v_bootstrap)`（`src/predict.jl:615`）；`loggross = (mu_m*e0[j] + relative[j,s] - shift)*s1[j] + residual*scale`（`src/predict.jl:620`）；`gross = exp(loggross)`（`src/predict.jl:621`）。

数学形式即 SPEC §35 的公式：

$$
\log R_i^{(s)}
=
\left(\mu_m^{(s)} e_{0,i}+\mu_{rel,i}^{(s)}-\overline{\mu_{rel}^{(s)}}\right)s_{1,i}
+
\underbrace{\sqrt{\tfrac{v_{T+1}(d_s)}{v_{bootstrap}}}\,\varepsilon_{i,row_s}^{OOF}}_{\text{当前 innovation 通道（标量缩放 × 经验行）}}.
$$

### 1.2 该结构所编码的强假设（逐条）

**A1 — common scalar scaling。** innovation 的记忆结构只有一个标量自由度 `scale`（`src/predict.jl:615`），所有资产、所有方向共用同一个缩放。等价于假设：风险的时间变化是横截面均匀的；不允许「今天某类资产的波动比另一类更可预测」。

**A2 — 横截面 shape = 历史无条件分布。** 行 bootstrap 直接复用历史行的完整横截面向量（`src/predict.jl:591`、`608`、`619`），不做逐行标准化。任意时间点的 innovation 形状都是历史无条件行分布的一个样本；条件信息（当前 $V_t$ 的形状）不改变 shape。

**A3 — 无 mode/asset-specific conditional risk。** $d$ 只进入标量 `scale`（`src/predict.jl:615`）；relative 方向的协方差与 $d$ 无关；唯一的记忆序列是 macro 标量投影（`src/predict.jl:493-495`）。资产 $j$ 的条件方差隐含地只有 $s_{1,j}^2 \cdot scale^2 \cdot$（该资产残差的历史经验二阶矩），没有资产特定的记忆结构。

**A4 — tail shape 不变。** bootstrap 原样取历史行（`src/predict.jl:608` 的 `residual_rows`），不改变任何高阶矩——尾部、偏度、峰度全部由经验测度的无条件分布决定。

**A5 — 行间 i.i.d.。** 场景之间行抽样独立均匀（`src/predict.jl:590-604` 的 pass 1 完全由 uniforms 与 mask 决定），无跨时间的条件依赖；时间记忆只通过标量 $v(d)$ 一条通道体现。

**A6 — $d$ 与 shape 独立。** $d$ 抽样（`src/predict.jl:613-614`）与行抽样（`src/predict.jl:591`）、relative draw（`src/predict.jl:570-572`）相互独立；$d$ 不重塑横截面。

**A7 — 空观察行零填充。** 历史中 $c=0$ 的行以 $e=0$ 进入 fractional likelihood 与 $v_{bootstrap}$（`src/residual_oracle.jl:239`）；这是一个被显式接受的经验处理，不是缺失残差的概率模型。

### 1.3 目标 vs 当前：逐项差距表

| # | 目标条款（本文第 2/3 节） | 当前事实 | 差距性质 | 证据 |
|---|--------------------------|----------|----------|------|
| 1 | 记忆为矩阵 $V_t(d)$（asset 坐标 $N\times N$） | 只有标量 $v_t(d)$，且仅作用于 macro 投影 | 结构缺失 | `src/predict.jl:116`、`493-495` |
| 2 | 逐行标准化 $z_s=V_{s-1}^{-1/2}\varepsilon_s$ | 无标准化；直接使用未标准化的历史行 | 结构缺失 | `src/predict.jl:608`、`619` |
| 3 | 反标准化因子 $V_t^{1/2}$ | 全局标量 $\sqrt{v_{T+1}(d)/v_{bootstrap}}$ | 结构缺失 | `src/predict.jl:615` |
| 4 | 方向性条件风险（relative/asset 记忆） | relative 方向与历史无条件分布绑定 | 结构缺失 | `src/predict.jl:619-620` |
| 5 | $d$ 可重塑 innovation 形状 | $d$ 只进入标量缩放 | 结构缺失 | `src/predict.jl:613-615` |
| 6 | ragged 缺失 cell 的 own-row fallback | 已有结构性回退（只依赖 mask 与 uniform） | 语义兼容，标准化语义待钉死 | `src/predict.jl:596-602`、`src/residual_oracle.jl:264-274` |
| 7 | 因果性：$V_t$ 只用 $\le t$ 信息 | 已满足（行抽样只在历史行内） | 无差距，目标须继承 | `src/predict.jl:590-604` |
| 8 | 唯一对称 PSD 因子约定 | relative draw 已用 `principal_sqrt_root` | 可沿用同一规范 | `src/predict.jl:564-570`、`src/numerics.jl:452-460` |

### 1.4 与 SPEC 的关系

SPEC §30 已明确：当前 innovation 是「OOF residual cross-sectional row bootstrap + macro residual 上 fractional-memory volatility scaling + missing cell 用该资产 own observed residual row 回退」，是 KISS 版 non-Markov risk，**不声称是最终完整无限维 innovation law**。本文的目标对象就是这个 KISS 版之后的规范对象：把 scalar 记忆升级为 vector memory。本文件不改变任何运行时行为。

---

## 2. 目标对象：vector innovation memory

### 2.1 定义

对 `DGRID_V1` 的每个 grid point $d$，定义 causal vector innovation memory：

$$
\boxed{
V_t(d)
=
\frac{\sum_{\tau\ge 1} k_d(\tau)\,\varepsilon_{t+1-\tau}\,\varepsilon_{t+1-\tau}^{\top}}
{\sum_{\tau\ge 1} k_d(\tau)}
}
,\qquad \varepsilon_s \in \mathbb R^{N_{active}}
$$

其中：

- $k_d$ 是与当前 `frac_weights` **同一个** fractional kernel（`src/predict.jl:12`；SPEC §31 的 $\pi_k$）；不得在 vector 版另立第二套 kernel（domain-language 纪律：一个概念一个词）。
- $\varepsilon_s$ 为 OOF 残差行，行身份语义沿用 `ResidualOracle` 契约：行 $idx$ 对应全局日 $t=ts\_total[idx]$，目标为 $r[t+1,:]$（`src/residual_oracle.jl:17-22`）。
- 下标 $t+1-\tau \le t$ 对 $\tau \ge 1$ 恒成立，保证 $V_t$ 只用决策时已知信息（见 3.2）。

**性质：**

- **PSD**：$k_d(\tau)\ge 0$（`frac_weights` 递推 $(k-1+d)/k$，$d \in (0,1]$，`src/predict.jl:12`），故 $V_t(d)$ 是半正定矩阵的加权和，构造性 PSD。
- **极限一致性**：$d=1$ 时 $k_1 \equiv 1$，$V_t(1)$ 退化为历史等权二阶矩；$d\to 0$ 时 kernel 快速衰减，$V_t$ 趋于最近少数行的二阶矩——与当前 scalar $v_t(d)$ 的极限一致（同一 kernel，`src/predict.jl:116`）。
- **归一化**：分母 $\sum_\tau k_d(\tau)$ 与当前实现 `counts=cumsum(frac_weights)`（`src/numerics.jl:425`）同源；其精确形式（截断长度、是否含 burn）列未决（6.1、6.3）。

### 2.2 common/relative gauge 分解

沿用项目既有分解（SPEC §12-15）：

$$
\varepsilon = e_0\,\varepsilon_m + Q\,\varepsilon_\perp,
\qquad
\varepsilon_m = e_0^{\top}\varepsilon,\quad
\varepsilon_\perp = Q^{\top}\varepsilon,
$$

其中 $e_0$ 为 center-of-mass 单位向量（alive 坐标上等权，`src/predict.jl:473-478`），$Q$ 为 Helmert fixed gauge（零和支撑；`relative_gauge(N)`，`src/predict.jl:371` 的构造性 witness 注释引用之，定义见 `src/numerics.jl`，D4 §1 #9 引用其在 `src/numerics.jl:368-380` 的使用）。

在 asset 坐标下 $V_t$ 相应分解为

$$
V_t
=
e_0 V_m e_0^{\top}
+
e_0 c_t^{\top}
+
c_t e_0^{\top}
+
Q\,V_\perp\,Q^{\top},
\qquad
V_m = e_0^{\top} V_t e_0,\quad
V_\perp = Q^{\top} V_t Q,\quad
c_t = Q^{\top} V_t e_0 .
$$

**要求：**

- **PSD**：$V_t(d) \succeq 0$（构造保证）。若标准化需要可逆，必须显式加正则化（6.2），不得静默丢弃零特征方向。
- **support**：relative 分量必须落在 $Q$ 的零和支撑内；ragged 语义沿用 zero-embedded 场的定义（`src/predict.jl:38-70`）：缺失坐标在 field 代数里嵌入为 0，但**观察 mask 必须单独保留**（`src/predict.jl:68`），$V_t$ 的统计只由实际观测的行定义。不得把 zero-embedding 当成「缺失收益 = 0」的概率模型（SPEC §12.1）。
- **permutation covariance**：资产列置换 $\Pi$ 下 $V_t \to \Pi V_t \Pi^{\top}$、目标权重 $\to \Pi^{\top} w$（SPEC §58）。任何依赖列编号的构造（例如固定的第一个资产角色）都是 bug。
- **gauge covariance**：$Q \to QR$（$R$ 正交）下 asset-space 对象不得改变（SPEC §59）。relative 归一必须用 per-band gauge-invariant scalar（`s_perp` 的既有教训，`src/predict.jl:22`、`279`；SPEC §15），**不得逐 coordinate 归一**。
- **alive 集合与扩维**：$V_t$ 只在当前 active/alive 坐标上定义；新资产上市（IPO、active-space 扩维）不得改变已有资产的 $V_t$ 子块（SPEC §11、§60 的 dummy/inactive invariance 在 innovation 层的推广；SPEC §50 的 exact coordinate embedding）。

### 2.3 与当前 scalar 对象的关系（诚实映射）

当前 `v_t(d)` 的标量序列是 $V_t(d)$ 在 macro 方向的一个**投影**：`src/predict.jl:116` 的 `output[T]/cs[T]` 实现的是 $\big(\sum_\tau k_d(\tau)\, e_{t+1-\tau}^2\big)/\big(\sum_\tau k_d(\tau)\big)$，其中 $e$ 是 macro 标量残差（`src/predict.jl:493-495`）。目标对象把同一 kernel 作用到完整向量残差上。

措辞纪律：scalar 版是**当前实现的合法 KISS 版**（SPEC §30 原话），vector 版是本文提交审查的**目标定义**。二者之间不是「近似 → 精确」的自动升级，而是规范选择（见第 4 节禁止项 1）。

---

## 3. 采样定律

### 3.1 standardized-shape 协议

目标场景的 innovation 部分定义为：

1. **标准化（历史形状）**：对历史每一行 $s \le t$，
   $$
   z_s = V_{s-1}^{-1/2}\,\varepsilon_s
   \quad\text{（含正则化，见 6.2）}.
   $$
2. **反标准化（未来抽样）**：在决策时刻 $t$，
   $$
   \varepsilon_{t+1} = V_t^{1/2}\, z_{s'},
   $$
   其中 $z_{s'}$ 是从历史标准化形状 $\{z_s\}$ 的经验测度中抽样的一个样本（或参数化分布的一个 draw，见 6.9）。

**要求：**

- 因子 $V^{1/2}$ 必须用唯一对称 PSD 主根（canonical convention）：`principal_sqrt_root` 的既有裁决——同 $\Sigma$ 在同 seed 有限场景下逐点到 roundoff 一致，不截秩、不掉小正特征值（`src/numerics.jl:438-460`）。这是 `src/predict.jl:564-570` 已为 relative draw 采纳的规范耦合；innovation 通道沿用同一约定，消除特征向量符号/简并基旋转自由度。
- 标准化与反标准化必须共用**同一次、同一个** $V$ 分解（同一个正则化 $\delta$、同一个因子约定）；禁止用两次独立分解的两个因子相乘。
- 形状抽样必须与 response 通道的 draw 使用独立随机流位置，保持 `rand`/`randn!` 消耗顺序的确定性契约（`src/predict.jl:533-545` 注释；SPEC §36）。

### 3.2 因果性（严格论证）

- $V_t(d)$ 的构造只含 $\varepsilon_{t+1-\tau}$，$\tau \ge 1$，即全局日 $\le t$ 的已实现残差；预测日 $t+1$ 的残差不在项内（2.1 定义）。
- 行抽样只在历史行 $1..T$ 内（`src/predict.jl:591`）；own-row 回退只从该资产历史有效行集合抽（`src/residual_oracle.jl:264-274`）。两者都满足「$V_t$ 只用 $\le t$ 信息」。
- $z_s = V_{s-1}^{-1/2}\varepsilon_s$ 中 $V_{s-1}$ 只含 $\le s-1$ 的行，$\varepsilon_s$ 自身属于 $s$ 时刻已实现（$s \le t$），故 $z_s$ 在 $t$ 时刻可算；$\varepsilon_{t+1} = V_t^{1/2} z_{s'}$ 严格因果。
- 禁止：把 $\varepsilon_{t+1}$ 自己、或任何 $>t$ 的行放进 $V_t$；禁止用未来行参与标准化集合的构造。

### 3.3 support 条件

- 若 $V_t$ 奇异（秩 $<N$ 或数值近奇异），$V_t^{-1/2}$ 沿零特征方向无定义。协议必须声明正则化：$V_t + \delta_t I$ 或特征值 floor；$\delta$ 的选择必须带 refinement 证据（6.2），且与 `EB_COVARIANCE_FLOOR` 的数值保护语义明确区分（SPEC §25；D4 §1 #4 引用 `src/response.jl:209`）。
- relative 分量的标准化必须在 $N-1$ 维 gauge support 内做；不得在含 $e_0$ 的 $N$ 维空间里做再投影（会产生假的 macro 泄漏）。
- permutation/gauge 协变性必须在标准化算子层面成立：$V^{-1/2}$ 与 $V^{1/2}$ 必须共享同一 gauge 约定；否则 $Q\to QR$ 会改变抽样分布（SPEC §59）。

### 3.4 ragged 缺失 cell 的 own-row fallback 语义

当前语义（事实）：场景 $s$ 抽中行 `row` 后，对每个资产 $j$，若 $(row,j)$ 缺失，则回退到该资产自己的历史有效残差行（`src/predict.jl:596-602` 的 `back_of`；`src/residual_oracle.jl:264-274`）；回退行的选择只依赖观察 mask 与 uniform，不依赖残差数值（`src/predict.jl:576-604` 的 pass 1 设计，与 `src/predict.jl:544-545` 的契约注释）。

目标协议保留这个结构性回退，但要求其语义在标准化协议下显式化：

- 缺失 cell **不是** 「0 innovation」，也**不是** 「用跨资产 $V_t$ 的相关结构去插补」；它是「该资产自身的边缘 innovation」。
- 若对缺失 cell 直接使用 $V_t$ 的跨资产标准化（含 $j$ 与其他资产的相关方向），会把未观测资产的相关结构强加给它——禁止。
- 允许的语义（候选，见 6.4）：对缺失 cell 使用该资产**边缘**的标准化形状——由该资产自身的有效 $z_s$ 行构成的一维经验测度反标准化；等价于 own-row fallback 的标准化版本。
- 选择判据不依赖 Sharpe：dummy asset invariance（SPEC §60）、ragged 语义保持（SPEC §12.1）、fallback 经验分布与标准化集合的一致性检验。

---

## 4. 禁止项

以下路径被明确定义为**不合规**：

1. **「先用 scalar、之后再升级」。** 不得把当前 scalar 版定位为「过渡路径」并借此为后续工作预留含糊空间；不得用「以后会升级」作为当前结构的辩护词。规范对象要么是 scalar（如实声明为 KISS 版、SPEC §30 的措辞），要么是本文的 vector 目标；两者的取舍必须走 SPEC 审查，不得作为实现顺路的默认。
2. **不得引入 factor bag。** innovation memory 不得表示为「若干手造因子」的叠加。$V_t(d)$ 是残差二阶矩的 kernel 加权对象；其自然结构（若有）应从 $V_t$ 的谱中导出，而不是预先命名。
3. **不得为资产手造因子。** 不得给资产或行业添加人工标签、板块表、手造协方差结构；所有结构必须来自价格历史与残差本身（SPEC §4、§5）。

（说明：这三条与 SPEC §4/§5/§95 一致；本文只是把 discipline 钉在 innovation 层，防止「升级路径」成为绕过 SPEC 审查的借口。）

---

## 5. 与 $d$ 后验 quadrature 的一致性

- $V_t(d)$ 必须与 $v_t(d)$ 共用**同一** fractional kernel $k_d$ 与**同一**归一化（第 2.1/2.2 节）。$d$ 后验保持 $p_d \propto \exp(\ell(d))\cdot\Delta d$ 的测度语义（`src/predict.jl:14`、`94-96`；SPEC §31）。
- 场景采样顺序保持「先抽 $d$、后取 per-$d$ innovation 因子」：与当前 `scale = sqrt(v_forecasts[d]/v_bootstrap)`（`src/predict.jl:615`）在同一位置使用 $d$ 的方向一致——即 $d$ 抽样先于 innovation 形状的反标准化。
- **网格细化时先验质量不得漂移**（SPEC §64）：$\Delta d$ 的测度语义在 vector 版保持不变。若 vector 版的 $\ell(d)$ 定义需要变化（例如从 scalar quasi-LL 变为 matrix quasi-LL），必须显式声明为规范变更、经 SPEC 审查，不得静默替换。
- `v_bootstrap` 的角色：当前绝对尺度锚点（`src/predict.jl:499`），SPEC §32 明确禁止用 $\sqrt{v_d/\operatorname{mean}_d v_d}$ 之类消掉总体 volatility level 的形式。vector 版必须保留「绝对尺度锚点」语义——每 $d$ 因子与历史锚点一致标定；锚点的具体形式列未决（6.6）。
- 收敛判据不得用回测收益选择 $S$ 或网格（SPEC §37、§65）：必须用 $\|w_{2S}-w_S\|_1$ 与 Kelly certificate。

---

## 6. 未决决定清单（每项给不依赖 Sharpe 的判据）

1. **kernel 归一与截断。** $k_d(\tau)$ 是否归一（$\sum k_d = 1$ 的显式口径）、有效截断长度（当前 FFT 用 `L÷2` 长度生成 kernel，`src/numerics.jl:421`）。判据：$d=0$/$d=1$ 极限与最近行/全历史等权二阶矩的一致性；与当前 $v_t(d)$ 在相同输入上的逐点对照（reference equality）；SPEC §64 的 refinement 语义。
2. **PSD floor/正则化。** $V_t^{-1/2}$ 的 $\delta$ 或特征值 floor。判据：$\delta \to 0$ 时目标权重/预测矩收敛（SPEC §25 已有 floor refinement 先例）；与 `EB_COVARIANCE_FLOOR` 的数值保护语义明确区分；失败时 fail-closed（SPEC §56）。
3. **标准化窗口与 burn。** $V_t$ 的有效历史长度（当前 `burn = 30`，`src/predict.jl:76-79`、`111`）。判据：窗口增大时标准化残差二阶矩收敛到单位矩阵的诊断检验；因果性不受影响（3.2）。
4. **ragged 缺失 cell 语义。** 保留原样 fallback（own-row 原值）还是 own-row 的标准化版本（3.4）。判据：dummy asset invariance（SPEC §60）、ragged 语义保持（SPEC §12.1）、fallback 分布与标准化集合的一致性。
5. **$V_t$ 的 macro/relative 交叉项。** 保留自然交叉协方差 $c_t$ 还是强制 block-diagonal（$e_0$ 与 $Q$ 块对角）。判据：gauge/permutation 协变性（SPEC §59、§58）与分解的语义纯洁性；不得按回测效果选择。
6. **绝对尺度锚点。** `v_bootstrap` 在 vector 版的对应物（标量锚 / 矩阵锚 / macro 通道锚）。判据：SPEC §32 的「不得消掉总体 volatility level」；锚点定义的 $d$-无关性；与 $\sigma^2$ 点估计的关系按 D4 §2 的 plug-in 分类处理，不得混同。
7. **$d$ 先验显式化。** 与 D4 第 7 节第 5 项对齐（SPEC §31 预留的 $p(d)$）。判据：均匀网格先验显式声明，或连续先验 + refinement 语义；先验质量不得漂移。
8. **实现表示与预算。** 每 $d$ 一个 $N\times N$ 矩阵的存储/计算 vs 低秩/谱/递推表示；是否属于 incremental 层（SPEC §48、§70）。判据：reference equality、预算 route 不改输出（SPEC §50）、静态复杂度论证；性能不得成为改数学的理由（SPEC §67）。
9. **标准化形状的分布。** $z$ 保持经验测度（bootstrap 标准化行）还是参数化。判据：tail shape 假设（A4）的显式声明；经验测度的 plug-in 性质按 D4 §2 分类；若参数化，必须有先验/似然的显式形式与证书（SPEC §56）。
10. **与 response 不确定性的层级关系。** innovation 层的标准化不得重复或抵消 response 层的 posterior predictive 结构（`src/predict.jl:570-572` 的 `canonical_root * z + mu_rel`）。判据：方差分解可报告性（D4 §5）；两层各自进入 predictive 的方式在 SPEC 中钉死。

---

## 7. 与既有文档和 SPEC 的关系

- 本文不复制 SPEC §30-32、§35-37 的原文；它们仍是规范来源。本文的定位是 innovation 层的 SPEC 级目标定义草案，与 D4 互补：D4 处理 $G/\alpha/\Sigma$ 的超参与 posterior 结构，本文处理 innovation law 的表示维度。
- 与 D4 保持一致的点：$d$ 的离散后验与 $\Delta d$ 测度（D4 #6）、innovation 的经验测度性质（D4 #7）、尺度的点估计性质（D4 #8）在本文全部保留；本文的目标条款不改变 D4 的清单。
- 纪律条款：本文所有未决项的选择都不得依据回测 Sharpe/收益（SPEC §95、开发守则 §24）；路线切换必须走 SPEC 审查；当前源码行为不变更——本文件不触发任何施工。

---

*（本文件为 Gate 0 重开后 Deliverable 5 交付物；未运行任何命令，未修改 `src/`、`test/`、`README.md`、`AGENTS.md` 或任何既有文档。）*
