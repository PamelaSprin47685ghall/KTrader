# KTrader Gate-0 统一 Mode-Space Response 的完整后验推导

**文档状态：Gate-0 Step 7 纸面推导（Deliverable）。规范草案，交 SPEC 审查；不是实现描述，不改变任何运行时行为。**
**任务来源：AGENTS.md 裁决书 §7-§11（D-021~D-039）、§40 Step 7（「推导 full response posterior……静态 review 完成以前严禁写复杂 optimizer」）。**
**证据基准：HEAD 602b897 + dirty 工作树时点的只读源码核对；全部行号引用为 `file:line`。本文未运行任何命令、未修改任何源码。**
**纪律：所有先验/积分/容差选择不得以回测收益为裁决依据（D-035 原文、SPEC §95、开发守则 §24）。凡无法闭合的步骤显式标注「未决，需 SPEC 审查」。**

---

## 0. 定位与阅读约定

本文档是裁决书 Step 7 的数学交付物：给出统一 mode-space response 的**完整后验**（full posterior）——即 `docs/POSTERIOR_DEFINITION.md`（下称 D4）第 2 节所精确陈述的 plug-in EB 结构的**替代对象**——的纸面推导，密度目标为：一名陌生工程师只读本文档即可实现 slow reference，无需再读旧 EB 实现（`src/response.jl:934-1122` 的 `optimize_conditioned_eb`、`src/response.jl:351-380` 的 `optimize_matrix_normal_eb`）。

阅读约定：

- $\mathcal H_t$：价格历史信息集；$n$：训练行数；$N$：mode 维数（见 §1 记号表——$N$ 同时等于资产数，语义不同、数值巧合相等）；$P$：feature 维数。
- $\Sigma_R$：response working-likelihood 的 row covariance（输出 mode 空间）；$\alpha_0,\alpha_p$：DC / path 两个 group 的精度超参。
- 「解析积分」指闭式边缘化；「数值积分」指 deterministic quadrature；二者与随机采样（MCMC）严格区分（D-086：MCMC 不作为第一版）。
- 与 D4 的关系：D4 是**现状盘点**（plug-in 结构的事实陈述），本文是**目标定义**（完整后验的推导）。§11 给出条款级差异表。
- 与 `docs/TRACE_NEUTRALITY_DERIVATION.md`（下称 D3）的关系：D3 裁决 14 条 trace 约束为 candidate hypothesis；本文 §8 给出默认 core 不施加它的构造性理由。
- 与 `docs/GATE0_VECTOR_INNOVATION.md`（下称 VI）的关系：innovation 层不在本文范围；本文 §7 钉死 response 模块的输出边界（$\mu$ 为止）与输出契约（§7.3），不定义 $\varepsilon$ 的定律。
- **接口锚（与 VI 逐字一致，Manager 裁决 D）**：$\varepsilon$ 为统一 mode 残差：源对象是 asset 空间 OOF 残差行 $\tilde\varepsilon_s = E_{active}\cdot\big[y_s-\big(b_0^{(-fold(s))}+G^{(-fold(s))}\cdot x_{s-1}\big)\big]$（含 $b_0$ 扣除，$\hat y=E[y_s\mid train^{(-fold)}]$ 为 D-041 期望语义）；innovation 层按决策日 risk 域 $R_t$ 消费其 $R$ 子向量经 $E_{R_t}$ 的 mode 变换。行 mask 契约：$O_s=\{j : r[u_s,j]\ \text{finite}\}$（ResidualOracle 行身份语义，VI §1.2 契约）。

---

## 1. 统一 mode-space 模型（D-021/D-025/D-026/D-031）

### 1.1 记号与维度总表

| 符号 | 维度 | 语义 |
|---|---|---|
| $N_a$ | 标量 | active 资产数（`active_universe_indices`，`src/predict.jl:125-135`） |
| $N$ | $=N_a$ | **mode 维数** $=1+(N_a-1)$（macro 1 维 + relative $N_a-1$ 维；与 $N_a$ 数值相等、语义不同，全文以 mode 语义使用） |
| $Q$ | $N_a\times(N_a-1)$ | Helmert fixed gauge（$Q^\top Q=I$，$Q^\top\mathbf 1=0$；`relative_gauge`，`src/numerics.jl:368-380`） |
| $e_0$ | $N_a$ | center-of-mass 单位向量（alive 坐标等权，`src/predict.jl:473-478`） |
| $u_{s}$ | $N_a$ | ruler 标准化收益（$u_{s,i}=r_{s,i}/s_{1,i}$，SPEC §12） |
| $m_s$ | $1$ | macro 标量（当日观察集 $O_s$ 上的 center of mass，`src/predict.jl:43-70`） |
| $e_s$ | $N_a$ | zero-embedded relative field（缺失坐标嵌 0，观察 mask 单独保留；SPEC §12.1） |
| $q_s=Q^\top e_s$ | $N_a-1$ | relative field 的 gauge 坐标 |
| $y_s$ | $N$ | 统一 mode 输出 $=[m_s;\,q_s]$ |
| $B_m(s)$ | $14$ | macro path basis（Q/P 各 7 band；`path_basis_1d`，`src/response.jl:2-16`） |
| $B_\perp(s)$ | $14(N_a-1)$ | relative path basis 的 gauge 坐标（对 $Q^\top X_\perp$ 的时间滤波；见 1.4） |
| $\tilde x_s$ | $P=1+14N_a$ | 统一输入 $=[1;\,B_m(s);\,B_\perp(s)]$（首列 $1$ 为 DC 常数列，D-031） |
| $B=[\,b_0\ G\,]$ | $N\times P$ | 统一系数矩阵：首列 $b_0\in\mathbb R^N$ 为 DC，其余 $G\in\mathbb R^{N\times 14N_a}$ |
| $\Sigma_R$ | $N\times N$ | 输出 mode 空间 row covariance（含 macro/relative 交叉块） |
| $X$ | $n\times P$ | 训练行堆叠的设计矩阵（行 $s\in$ 训练集，$\tilde x_s^\top$） |
| $Y$ | $n\times N$ | 训练目标（行 $s$ 为 $y_{s+1}^\top$，即 SPEC §28 的 $ts\_total$ 语义：feature 在 $s$、目标在 $s+1$） |

### 1.2 模型

$$
\boxed{\;Y = X\,B^{\top} + E,\qquad \mathrm{rows}(E)\ \overset{\text{working}}{\sim}\ \mathcal N(0,\Sigma_R)\;}
$$

即对每条训练行 $s$：

$$
y_{s+1} = b_0 + G\,x_s + \varepsilon^{w}_{s+1},\qquad \varepsilon^{w}_{s+1}\sim\mathcal N(0,\Sigma_R)\ \text{（working likelihood 噪声）}.
$$

**测度语义**：这是 response 模块的 **working likelihood**——$\mathcal N(0,\Sigma_R)$ 是系数后验的共轭缩放核，不是对真实 innovation 的分布声明（D-040/D-041/D-060；详见 §7）。真实未来残差由 innovation 模块（vector law $V_t(d)$）独立提供。

**记号区分（Manager 裁决 D4）**：全文以可区分记号防止「$\Sigma_R$ 抽样被误当 innovation」——$\varepsilon^w$：working likelihood 噪声（**分布对象**，仅存在于似然/共轭结构，其协方差 $\Sigma_R$ 的后验抽样**不是** innovation）；$\tilde\varepsilon_s$（即 $\varepsilon^{OOF}$）：OOF 残差（**实现值**，asset 空间行，§7.3(a) 输出契约）；$\varepsilon_{t+1}=V_t(d)^{1/2}z$：innovation 层的未来残差（VI 定义）。

### 1.3 四块 cross-response（D-025/D-026）

把 $G$ 按输入（macro path $14$ 列 / relative path $14(N_a-1)$ 列）与输出（macro $1$ 行 / relative $N_a-1$ 行）分块：

$$
G=\begin{bmatrix}G_{mm}&G_{m\perp}\\[2pt]G_{m\perp}&G_{\perp\perp}\end{bmatrix},
\qquad
\begin{aligned}
G_{mm}&\in\mathbb R^{1\times14},&\quad G_{m\perp}&\in\mathbb R^{1\times14(N_a-1)},\\
G_{\perp m}&\in\mathbb R^{(N_a-1)\times14},&\quad G_{\perp\perp}&\in\mathbb R^{(N_a-1)\times14(N_a-1)}.
\end{aligned}
$$

（Manager 裁决 2026-10-10：本句块记号原文作 G_{\perp m}，与本文矩阵分块定义（行=输出）相反——relative 输入预测整体市场应为 G_{m\perp}，已订正。Wave 2 Step 4 交付者发现；response.jl docstring 已按矩阵语义实现并记录同一勘误。）

**四块全部由数据/后验决定**。当前实现（`fit_response_operator`，`src/response.jl:1124-1209`）将 macro 与 relative 拆成两个独立回归（macro：`src/response.jl:1130-1155`；relative：`src/response.jl:1164-1208`），等价于强制

$$
G_{m\perp}=0,\qquad G_{\perp m}=0
$$

——block-diagonal 假设，无理论推导支持（D-025 原文）。本模型删除该假设：relative path 可以预测整体市场（$G_{\perp m}$ 方向），macro path 可以预测横截面 rotation（$G_{m\perp}$ 方向）。合成世界恢复性测试见 §10 第 8 条（D-092）。

### 1.4 与当前 14N 维 design 的精确差异对照

| 维度 | 当前实现 | 本模型 | 差异性质 |
|---|---|---|---|
| 输入总维数 | relative $14N_a$（`src/predict.jl:290`：$P_{features}=2|BANDS|\cdot N$）+ macro $14$ **分开两个设计** | $1+14N_a$ **单一设计**（DC 1 + macro 14 + relative $14(N_a-1)$） | 结构 |
| relative 列坐标 | asset 坐标（每 band $2N_a$ 列，$j=1..N_a$；`src/response.jl:62-63` 的 `col_q/col_p`） | **Helmert gauge 坐标**（每 band $2(N_a-1)$ 列） | 结构（D-030） |
| DC 列 | 无 | 首列 $1$，独立精度 group $\alpha_0$ | 理论恢复（D-031/D-032） |
| relative 目标 | $Y_{target}=e_{s+1}\in\mathbb R^{N_a}$（zero-embedded asset 坐标，$N_a$ 维表示 $N_a-1$ 维对象，`src/predict.jl:298`） | $q_{s+1}=Q^\top e_{s+1}\in\mathbb R^{N_a-1}$（无冗余表示） | 结构 |
| macro 通道 | 独立回归 + $\hat\sigma^2$ 点估计（`src/response.jl:1154`） | 并入统一 $Y$ 的第一列，$\Sigma_R$ 的 $(1,1)/(1,\cdot)$ 块承担 | 理论变更（D-039） |
| $\Sigma$ 结构 | relative $\Sigma$（$N_a-1$ 维 gauge support，`supported_covariance`，`src/response.jl:179-182`）+ macro $\hat\sigma^2$ 分离 | 统一 $\Sigma_R\in\mathbb R^{N\times N}$，**含 macro-relative 交叉协方差块** | 理论变更 |
| $\alpha$ 结构 | $\hat\alpha_m,\hat\alpha_{rel}$ 两个独立 EB 点估计（`src/response.jl:1147,1180`） | $\alpha_0,\alpha_p$ 两个 group，**联合后验积分**（§4-§6） | 认识论升级（D-033/D-034/D-038） |

**$B_\perp$ 的构造语义**：时间滤波（`path_basis_1d` 的窗口均值/差分）是线性算子，与 $Q^\top$ 可交换，故

$$
B_\perp(s)=\mathcal F_\tau\!\big[Q^\top X_\perp\big](s),\qquad X_\perp(t)=\sum_{s\le t}e_s,
$$

其中 $\mathcal F_\tau$ 为 SPEC §16 的 paired Q/P 滤波、$s_\perp(\tau)$ 归一沿用 gauge-invariant per-band scalar（SPEC §15；逐坐标归一会破坏 $Q\to QR$ 协变性，禁止）。这是裁决书 Step 3（统一 mode coordinate producer）的职责；本文只钉死其语义契约：**$B_\perp$ 必须与 asset 坐标版本满足 $B_\perp = \mathcal L_Q(B_{asset})$ 的精确线性关系（$Q^\top$ 逐 band 作用于列块），使 permutation/gauge 不变量可测**。

**ragged 语义**：$m_s$ 用当日观察集 $O_s$（`embedded_relative_field` 现有语义，`src/predict.jl:48-66`）；$e_s$ 缺失坐标嵌 0、观察 mask 单独保留（SPEC §12.1）；$q_s=Q^\top e_s$ 照常投影（缺失资产的贡献为 0 是 field 代数，不是收益 imputation）。**已裁决（Manager 裁决 E，2026-10-09，记录于 `docs/GATE0_MANAGER_ADJUDICATIONS.md`）**：$m_s$ 按行级观察集 $O_s$ 归一（$1/\sqrt{|O_s|}$，SPEC §12 现状语义）；$E_{active}$ 用 active 集等权、$E_{R_t}$ 用 $R_t$ 等权——**域基（domain basis，随域等权）与行级 field 定义（随行观察集归一）是两个概念**，并存不矛盾。Step 3 的其余 ragged 实现细节按推导默认建议执行、实现期可复审（裁决 G1，§12 第 3 条）。

---

## 2. 先验（D-032/D-033/D-035）

### 2.1 系数矩阵先验

$$
\boxed{\;B\ \big|\ \Sigma_R,\alpha_0,\alpha_p\ \sim\ \mathcal{MN}_{N\times P}\!\left(0,\ \Sigma_R,\ \Lambda_\alpha^{-1}\right),\qquad
\Lambda_\alpha=\mathrm{diag}\!\left(\alpha_0,\ \alpha_p\mathbf 1_{14N_a}\right)\;}
$$

密度（$\mathcal{MN}$ 的行/列协方差参数化）：

$$
p(B\mid\Sigma_R,\alpha)= (2\pi)^{-NP/2}\,|\Sigma_R|^{-P/2}\,|\Lambda_\alpha^{-1}|^{-N/2}\,
\exp\!\left(-\tfrac12\,\mathrm{tr}\!\left[\Lambda_\alpha\,B^\top\Sigma_R^{-1}B\right]\right).
$$

- **DC 零中心（D-032）**：先验均值 $0$ 对 $b_0$ 成立——先验不注入任何正漂移；$b_0$ 的非零后验只能来自价格证据（似然项）。先验均值向量为零是「零中心」的精确含义；它**不**等价于「后验均值为零」。
- **两个 group（D-033）**：$\alpha_0$ 管零频 DC（1 列），$\alpha_p$ 管**全部** path 系数（macro path 14 列与 relative path $14(N_a-1)$ 列共用）。禁止每 band / 每资产 / macro-vs-relative path 各一个精度（D-033 原文；若 SPEC 未来要拆 macro/relative path group，属理论版本变更，见 §12 第 2 条）。
- 与旧实现的对照：旧先验是 $G\mid\Sigma,\alpha\sim\mathcal{MN}(0,\Sigma,\alpha^{-1}I)$（SPEC §19），单一 $\alpha$、无 DC 列、$\Sigma$ 限 relative support——本先验是其 $P\to1+14N_a$、$\alpha I\to\Lambda_\alpha$、$\Sigma\to\Sigma_R$（全 $N$ 维）的三重推广。

### 2.2 超参与协方差先验（D-035）

$$
\boxed{\;p(\alpha_0)\propto 1/\alpha_0,\qquad p(\alpha_p)\propto 1/\alpha_p,\qquad
p(\Sigma_R)\propto|\Sigma_R|^{-(N+1)/2}\;}
$$

- $\alpha$ 的 $1/\alpha$ 先验即 **log-scale 上的均匀测度**（$d\alpha/\alpha=d\log\alpha$）——reference prior 的标准选择；$\alpha_0,\alpha_p$ **独立**。
- $\Sigma_R$ 的 $|{\cdot}|^{-(N+1)/2}$ 是 $N$ 维多元正态协方差的 **Jeffreys/reference prior**（对 working likelihood 的参数 $\Sigma_R$ 取 Jeffreys，非对整个模型）。
- **测度语义**：三者均为 improper；后验 propriety 由 §5 的 gate 单独证明——improper prior 不自动导致 improper posterior，但也不豁免检查（D-036）。
- **这些先验不是通过回测选出来的**（D-035 原文）；任何替换（含 §5 发现所要求的尾部修正）必须走 SPEC 审查，不得以回测表现为依据。
- 禁止（D-036 原文，实现红线）：propriety 不满足时偷偷 clamp、改回 EB、套 inverse-Wishart 超参数。必须 fail loudly。

---

## 3. 联合后验与充分统计表示

### 3.1 联合后验

$$
p(B,\Sigma_R,\alpha_0,\alpha_p\mid Y)\ \propto\
p(Y\mid B,\Sigma_R)\;p(B\mid\Sigma_R,\alpha)\;p(\Sigma_R)\;p(\alpha_0)\,p(\alpha_p).
$$

### 3.2 充分统计（关键断言 A1）

**整个后验只依赖 $(X,Y)$ 通过四个统计量**：

$$
\boxed{\;n,\qquad S_{xx}=X^\top X\in\mathbb R^{P\times P},\qquad S_{xy}=X^\top Y\in\mathbb R^{P\times N},\qquad S_{yy}=Y^\top Y\in\mathbb R^{N\times N}\;}
$$

（$B$ 的先验均值为零使 $S_{xy}$ 无需中心化修正。）这是 reference 实现、增量引擎、OOF fold 共享同一数学对象的基础：fold 的 train 统计 = full 统计 − eval fold 统计（SPEC §28 现有恒等式，逐元素成立）。**实现含义**：slow reference 只需持有这四个量（加 $\Lambda_\alpha$ 的两个标量）即可完成 §4-§6 的全部计算；任何「需要整块设计矩阵」的说法只可能是数值工程选择（如 dual 分解），不是数学需要。

---

## 4. 解析积分路线（D-037/D-085：能解析绝不采样）

积分顺序（由内到外）：$B$（矩阵正态共轭）→ $\Sigma_R$（Jeffreys 下的逆 Wishart 核积分）→ $(\alpha_0,\alpha_p)$（2 维 deterministic quadrature，§6）。**结论先行：给定 $(\alpha_0,\alpha_p)$，$(B,\Sigma_R)$ 可以整体解析消掉；剩余积分恰为 $\mathbb R^2$ 上的 2 维对象**——D-085 的「低维 hyperparameters 做 deterministic quadrature」在此精确成立，无需对 $\Sigma$ 做任何数值积分（本节末尾论证）。

### 4.1 第一步：$B\mid\Sigma_R,\alpha,Y$（矩阵正态共轭）

对 $\log p$ 关于 $B$ 配平方（$\mathrm{tr}(\Sigma_R^{-1}[E^\top E+B\Lambda_\alpha B^\top])$，$E=Y-XB^\top$）：

$$
\boxed{\;B\ \big|\ \Sigma_R,\alpha,Y\ \sim\ \mathcal{MN}_{N\times P}\!\left(\hat B,\ \Sigma_R,\ V\right),\qquad
V=(S_{xx}+\Lambda_\alpha)^{-1},\qquad \hat B=S_{xy}^\top V\;}
$$

（$\hat B^\top=VS_{xy}$ 是 ridge 解的转置；与旧实现 `V=(X^TX+αI)^{-1}`、`G=B_scaled'*basis'`（`src/response.jl:1186-1191`）同构，差别仅在 $\Lambda_\alpha$ 的两 group 结构与 $P$ 的扩维。）

**用到的恒等式**：Gaussian–Gaussian 共轭（矩阵正态先验 × 矩阵正态似然 → 矩阵正态后验）。

### 4.2 第二步：$Y\mid\Sigma_R,\alpha$（对 $B$ 的边缘化 = 矩阵正态卷积）

$Y=XB^\top+E$，$B\sim\mathcal{MN}(0,\Sigma_R,\Lambda_\alpha^{-1})$ 与 $E$ 行独立 $\mathcal N(0,\Sigma_R)$。用 vec 技巧（$\mathrm{vec}(XB^\top)=(I_N\otimes X)\,\mathrm{vec}(B^\top)$，$\mathrm{cov}(\mathrm{vec}(B^\top))=\Sigma_R\otimes\Lambda_\alpha^{-1}$）：

$$
\boxed{\;Y\ \big|\ \Sigma_R,\alpha\ \sim\ \mathcal{MN}_{n\times N}\!\left(0,\ \Sigma_R,\ C\right),\qquad
C = I_n+X\Lambda_\alpha^{-1}X^\top\;}
$$

密度：

$$
p(Y\mid\Sigma_R,\alpha)=(2\pi)^{-nN/2}\,|\Sigma_R|^{-n/2}\,|C|^{-N/2}\,
\exp\!\left(-\tfrac12\,\mathrm{tr}\!\left[\Sigma_R^{-1}\,S(\alpha)\right]\right),
$$

其中（Woodbury：$C^{-1}=I_n-X(S_{xx}+\Lambda_\alpha)^{-1}X^\top$；Sylvester：$|C|=|\Lambda_\alpha|^{-1}|S_{xx}+\Lambda_\alpha|$）：

$$
\boxed{\;S(\alpha)\ :=\ Y^\top C^{-1}Y\ =\ S_{yy}-S_{xy}^\top\,(S_{xx}+\Lambda_\alpha)^{-1}\,S_{xy}\;}
$$

（$N\times N$，对称 PSD。**实现警告**：$S(\alpha)$ 是 $Y^\top C^{-1}Y$（$C^{-1}$ 对称但非幂等），**不是** ridge 残差平方和 $(C^{-1}Y)^\top(C^{-1}Y)$——二者仅在 $\alpha\to0$ 的 OLS 投影极限重合；写错此项会静默改变全部下游。）

**用到的恒等式**：矩阵正态卷积（独立矩阵正态之和）、Sylvester 行列式恒等式、Woodbury 矩阵求逆。

### 4.3 第三步：$p(Y\mid\alpha)$（对 $\Sigma_R$ 的 Jeffreys 积分）

被积核：$|\Sigma_R|^{-n/2}\exp(-\tfrac12\mathrm{tr}(\Sigma_R^{-1}S))\times|\Sigma_R|^{-(N+1)/2}=|\Sigma_R|^{-(n+N+1)/2}\exp(\cdot)$——逆 Wishart 核。用标准积分（$\nu>N-1$、$\Psi\succ0$ 时）：

$$
\int_{\Sigma\succ0}|\Sigma|^{-(\nu+N+1)/2}\exp\!\left(-\tfrac12\mathrm{tr}(\Sigma^{-1}\Psi)\right)d\Sigma
=2^{\nu N/2}\,\Gamma_N(\nu/2)\,|\Psi|^{-\nu/2},
$$

其中 $\Gamma_N(a)=\pi^{N(N-1)/4}\prod_{j=1}^{N}\Gamma\!\big(a+\tfrac{1-j}{2}\big)$（多元 Gamma）。取 $\nu=n$、$\Psi=S(\alpha)$：

$$
\boxed{\;
p(Y\mid\alpha_0,\alpha_p)
=K(n,N)\;\underbrace{|\Lambda_\alpha|^{N/2}\,|S_{xx}+\Lambda_\alpha|^{-N/2}}_{=\,|I_P+\Lambda_\alpha^{-1}S_{xx}|^{-N/2}}\;\big|S(\alpha)\big|^{-n/2}
\;}
$$

$$
K(n,N)=(2\pi)^{-nN/2}\,2^{nN/2-1}\,\Gamma_N(n/2)\quad(\text{与 }\alpha\text{ 无关}).
$$

（Manager 裁决 2026-10-10：marginal evidence 常数 K 的定义原文为 2^{nN/2}，订正为 2^{nN/2-1}（指数减一）——对齐 Jeffreys σ 先验 dense 直接积分的计算事实，Wave 3 裁决 1；posterior_tests 的 logK 断言已按此对齐并全绿。Manager 确认 2026-10-10：dense 对照为 N=1（nN/2=n/2），测试断言 (n/2−1)·log2 与订正在 N=1 下精确一致，N 维推广按指数结构自然。）

**这就是 marginal evidence**（D-037 要求的对象）。对数形式（实现者直接使用）：

$$
\log p(Y\mid\alpha)=\mathrm{const}+\tfrac{N}{2}\big[\log\alpha_0+14N_a\log\alpha_p\big]-\tfrac N2\log|S_{xx}+\Lambda_\alpha|-\tfrac n2\log|S(\alpha)|.
$$

**用到的恒等式**：逆 Wishart 归一化积分（Jeffreys 先验下 $\Sigma$ 的精确共轭消元）。

### 4.4 第四步：$B\mid\alpha,Y$ 与决策时线性形式（matrix-t）

**系数后验（对 $\Sigma_R$ 积分后的边缘）**：混合 $\mathcal{MN}(\hat B,\Sigma_R,V)$ 与 $\Sigma_R\mid\alpha,Y\sim\mathcal{IW}\!\big(S(\alpha),\,n\big)$（第三步的直接副产品）：

$$
\boxed{\;
p(B\mid Y,\alpha)\ \propto\ |V|^{-N/2}\;\Big|S(\alpha)+(B-\hat B)\,V^{-1}(B-\hat B)^\top\Big|^{-(n+P)/2}
\;}
$$

——**matrix-t**（Dawid 分布族；本文参数化钉死为：位置 $\hat B\in\mathbb R^{N\times P}$、行尺度 $S(\alpha)\in\mathbb R^{N\times N}$、列精度 $V^{-1}\in\mathbb R^{P\times P}$、自由度参数 $n$；密度核如上，尾部幂 $(n+P)/2$ 来自 $\Sigma$ 积分的有效 $\nu'=n+P$）。**用到的恒等式**：正态–逆 Wishart 混合 → matrix-t（对矩阵变元）。

**决策时线性形式 $\mu=B\tilde x_t$**（$\tilde x_t\in\mathbb R^P$ 为决策日 feature；这是 D-037 的「decision-time predictive」——不抽整张 $B$，只边缘化 $B\tilde x_t$，SPEC §33 原则的推广）：

条件于 $(\Sigma_R,\alpha)$：$\mu\sim\mathcal N\!\big(\hat B\tilde x_t,\ c\,\Sigma_R\big)$，$c=\tilde x_t^\top V\tilde x_t$。再对 $\Sigma_R$ 积分（正态–IW 混合 → 多元 t，matrix determinant lemma $|S+c^{-1}\delta\delta^\top|=|S|\,(1+c^{-1}\delta^\top S^{-1}\delta)$）：

$$
\boxed{\;
\mu\ \big|\ Y,\alpha\ \sim\ t_{\nu}\!\left(\hat B\tilde x_t,\ \ \frac{c}{\nu}\,S(\alpha)\right),
\qquad \nu=n+1-N,\qquad c=\tilde x_t^\top V\tilde x_t
\;}
$$

（$t_\nu(\mu_0,\Sigma_t)$ 按密度 $\propto\big(1+\tfrac1\nu\delta^\top\Sigma_t^{-1}\delta\big)^{-(\nu+N)/2}$ 参数化；幂 $-(\nu+N)/2=-(n+1)/2$ 与 4.4 推导一致。）**要求 $\nu>0$ 即 $n\ge N$**——与 §5 propriety 条件同源。

**注意（epistemic 边界）**：此 $t$ 的尺度**只**覆盖 response 模块的 epistemic 不确定性（系数条件方差 + $\Sigma_R$ 后验）；它**不含**未来一日的 aleatoric 冲击。若要「含新观测噪声」的 predictive（$\tilde y=\mu+\varepsilon_{t+1}^{new}$），自由度降为 $n-N$——但按 D-040/D-041/D-060，**response 模块的输出到 $\mu$ 为止**，未来冲击由 innovation 模块提供，故本推导钉死 $\nu=n+1-N$ 版本（§7 展开）。

### 4.5 $\Sigma_R$ 能否整体解析消掉：论证

能。给定 $(\alpha_0,\alpha_p)$：(i) $B$ 经共轭消元（4.1）；(ii) $\Sigma_R$ 经 Jeffreys–IW 积分精确消元（4.3，条件 $n>N-1$、$S(\alpha)\succ0$）；(iii) 决策量 $\mu$ 有 matrix-t/多元-t 闭式（4.4）。**不存在需要数值积分 $\Sigma$ 的维度**——D-037「高维 $\Sigma$ 的 Laplace/quadrature 不作为第一方案」在 reference 先验下被更强地满足：解析路线完全成功，Laplace/quadrature/MCMC 均无必要。数值积分只剩 $(\log\alpha_0,\log\alpha_p)\in\mathbb R^2$（§6）。

---

## 5. Posterior propriety（D-036：完整静态推导）

目标：$p(\alpha_0,\alpha_p\mid Y)\propto p(Y\mid\alpha)\,/\,(\alpha_0\alpha_p)$ 在 $(0,\infty)^2$ 上可归一。分三个区域推导（$u_i=\log\alpha_i$ 坐标，$d\alpha_i/\alpha_i=du_i$，故 log 坐标被积函数就是 $p(Y\mid\alpha)$ 本身）。

### 5.1 内部（$\alpha$ 有限远离 0）

$p(Y\mid\alpha)$ 处处有限正（$S_{xx}+\Lambda_\alpha\succ0$ 恒成立、$S(\alpha)\succ0$ 时 $|S(\alpha)|^{-n/2}$ 有限）。无内部奇点。

### 5.2 小 $\alpha$ 尾部（$u\to-\infty$）：收敛

用 $|\Lambda_\alpha|^{N/2}|S_{xx}+\Lambda_\alpha|^{-N/2}=\big|I_P+\Lambda_\alpha^{-1}S_{xx}\big|^{-N/2}$。设 $\mathrm{rank}(X)=r\le\min(n,P)$，则 $\Lambda_\alpha^{-1}S_{xx}$ 有 $r$ 个 $\sim\alpha_p^{-1}$ 的发散特征值（DC 块同理对 $\alpha_0$），故该项 $\sim\alpha_p^{rN/2}\to0$。$S(\alpha)\succeq\lambda_{\min}(C^{-1})\,S_{yy}$，而 $\lambda_{\min}(C^{-1})=1/\lambda_{\max}(C)\sim\alpha/\lambda_{\max}(S_{xx})$，故 $|S(\alpha)|^{-n/2}\lesssim(\alpha_p^{-1})^{nN/2}$。log 坐标被积 $\lesssim\exp\big[(rN/2-nN/2)\,u\big]$，$u\to-\infty$ 时只要 $n>r$ 即指数衰减。**$n>r$ 由 $n\ge r+1$ 恒成立（$n$ 行数据的秩不超过 $n$）——小尾无条件收敛**（含 $X$ 秩亏情形：秩亏只让 $r$ 更小、衰减更快）。

### 5.3 大 $\alpha$ 尾部（$u\to+\infty$）：**发散（关键发现）**

$\alpha_p\to\infty$（$\alpha_0$ 任意固定或同趋于 $\infty$）：$\Lambda_\alpha^{-1}\to0$，$C\to I_n$，$C^{-1}\to I_n$，故

$$
S(\alpha)\to S_{yy},\qquad |I_P+\Lambda_\alpha^{-1}S_{xx}|^{-N/2}\to1,\qquad
p(Y\mid\alpha)\to K(n,N)\,|S_{yy}|^{-n/2}\ >\ 0.
$$

（机制：先验精度 $\to\infty$ 把 $B$ 钉到 $0$，模型退化为「$Y$ 与 $X$ 独立」的零模型，其 evidence 是与 $\alpha$ 无关的正常数。）log 坐标上被积函数趋于**非零常数**，故

$$
\int^{+\infty}p(Y\mid e^u)\,du=+\infty.
$$

$\alpha_0\to\infty$ 单独发散同理（饱和于「无截距」子模型的 evidence，仍与 $\alpha_0$ 无关）。

### 5.4 结论与 gate 断言

$$
\boxed{\ \text{在 D-035 先验（}1/\alpha\ \text{log-uniform）与无界支撑下，后验对 }(\alpha_0,\alpha_p)\ \text{恒不 proper}\ }
$$

（与 g-prior 文献中 Jeffreys 先验的尾部行为同族：似然在「零模型」处的饱和平台 × log-uniform 测度 = 无穷尾部质量。）此外还有数据层条件：$\Sigma_R$ 积分（4.3）本身要求 $n>N-1$ 且 $S(\alpha)\succ0$；$S(\alpha)\succ0$ 在 $Y$ 列满秩（$n\ge N$ 且目标行不退化）时对一切 $\alpha>0$ 成立（$C^{-1}\succ0$ 恒成立——正则化使 $S$ 不受 $X$ 秩亏影响，这与 OLS 投影残差不同）。

**可写成实现的断言集**：

- **A2（数据层）**：$n\ge N$（等价 $\nu=n+1-N>0$）且 $S(\alpha)\succ0$（实现：`rank(Syy) == N` 检查）。违反 → error，错误文本含 "posterior improper"（D-036）。
- **A3（先验层，静态数学事实）**：D-035 原始先验（$1/\alpha$、无界支撑）下，A2 通过**仍不**proper——大尾恒发散（§5.3）。此为推导事实，不随实现状态变化；若实现运行在原始先验下，必须 fail loudly，禁止 clamp / 回退 EB / 套 IW 超参（D-036 原文红线）。
- **A4（先验修正，已裁决采纳）**：Manager 已裁决采纳 **D-035a（2026-10-09，记录于 `docs/GATE0_MANAGER_ADJUDICATIONS.md`）**：

$$
p(\alpha)\propto\frac{1}{\alpha(1+\alpha)}\quad(\text{log 坐标上}\ \sim e^{-u}\ \text{衰减，}\ \int^{+\infty}\!\text{const}\cdot e^{-u}du<\infty;\ \text{小 }\alpha\text{ 处仍为 }1/\alpha,\ \text{保 Jeffreys 局部行为}).
$$

  **裁决依据**：纯数学 propriety——保持 reference 先验的局部（小/中 $\alpha$）语义、截断无穷远的零模型平台质量；**非回测选择**（D-035 纪律延续）。**可推翻条件**：owner/SPEC 否决 D-035a 则回退 $1/\alpha$，后验回到恒 improper（A3 语义，实现恒红）。有界域截断路径仍被 D-038 禁止且数学上无效（被积不衰减 ⇒ 截断误差不可忽略）。

**诚实陈述**：propriety 推导（小尾收敛、大尾饱和、数据层条件）是严格闭合的静态事实；先验修正形式已由 D-035a 裁决闭合（可推翻条件如上）。D-036 要求的「静态推导 + 最小解析测试验证」中，静态部分由本节完成，测试用例见 §10（第 2 条在 D-035a 下应绿）。

---

## 6. $(\log\alpha_0,\log\alpha_p)\in\mathbb R^2$ 的 deterministic adaptive quadrature（D-038）

### 6.1 被积对象与坐标

$$
p(\alpha\mid Y)\ \propto\ p(Y\mid\alpha)\,/\,(\alpha_0\alpha_p)
\quad\Longrightarrow\quad
\text{log 坐标密度}:\ \ w(u_0,u_p)=p(Y\mid e^{u_0},e^{u_p})\ /\ Z,
$$

（$1/\alpha$ 先验与 $d\alpha=\alpha\,du$ 的 Jacobian 精确相消——log 坐标上被积函数就是 marginal evidence 本身；**D-035a 已采纳（§5.4）**：需对两个 $\alpha$ 各乘修正因子，即 $w(u_0,u_p)\propto p(Y\mid e^{u_0},e^{u_p})\,\big/\big[(1+e^{u_0})(1+e^{u_p})\big]$。）任何后验泛函（$\mu$ 的矩、$\alpha$ 的后验矩、predictive 混合权）都是 $w$ 的 2 维积分。

### 6.2 自适应细分策略（设计，非实现）

- **初始 bracket**：旧 EB optimizer 的 mode（`maximize_logalpha` 的解，`src/response.jl:212-273`）**仅作**初始网格中心与诊断对照（D-038 原文：提供 mode / initial bracket / diagnostic comparison；**不得定义 posterior**）。初始网格覆盖 mode 两侧若干倍程。
- **细分**：2 维 cell 网格（log 坐标均匀或按 evidence 曲率加密）；每 cell 的误差估计 = 被积函数在 cell 内的局部变差上界（如顶点极差或内嵌更高阶规则之差）；对误差贡献最大的 cell 二分；**全局误差证书** $=\sum_{\text{cells}}\text{err}(\text{cell})$，小于 $\epsilon_{quad}$ 时停止。
- **确定性**：节点生成不消耗随机数（与 `ScenarioQuadrature` 的共享 shift 语义对照，`src/predict.jl:507-531`）；同输入、同配置逐位可重放。
- **节点数不是策略参数**（D-038 原文）：网格密度由误差证书驱动，不得固定。

### 6.3 尾部质量证书与 fail loudly

- **左尾**（$u\to-\infty$）：§5.2 证明指数衰减，尾质量证书可签发（$\int_{-\infty}^{u_{\min}}\!\lesssim e^{(rN/2-nN/2)u_{\min}}$，显式可算）。
- **右尾**（$u\to+\infty$）：**D-035a 修正后**（§5.4），被积函数以 $\sim e^{-u}$ 衰减（§5.3 的饱和平台被修正因子压制），尾质量证书可签发：$\int_{u>U}\mathrm{const}\cdot e^{-u}du\le e^{-U}$，显式可算。**回退情形**（owner/SPEC 否决 D-035a、回到 $1/\alpha$）：§5.3 的饱和使证书恒失败——那是后验 improper 的直接表现，须转译为 fail loudly（错误文本含 "posterior improper: alpha tail"），**不得**通过截断积分域、调大预算或返回最后网格蒙混（D-067 语义；错误文本规范见 `docs/NUMERICAL_INTEGRATION_SPEC.md` §4——统一为含 "Numerical integration did not converge" 的 fail 形态）。
- **refinement 证书**：相邻两级网格（$h\to h/2$）上，全部目标泛函（至少：$\mu$ 的均值向量、协方差迹、$\alpha$ 后验矩）的变化量 $\le\epsilon_{quad}$；预算（最大 cell 数/求值数）耗尽仍未达 → error。**禁止**「到预算就返回当前值」。
- $\epsilon_{quad}$ 的定稿走 D-066 流程（synthetic 解析答案 + tolerance 减半 + 与 Sharpe/PnL 无关），本文不拍数字。

### 6.4 输出对象

quadrature 输出：(i) 节点集 $\{(u_0^{(k)},u_p^{(k)})\}$、(ii) 后验质量 $\{p_k\}$（归一化）、(iii) 每节点的解析块（$\hat B,V,S(\alpha^{(k)})$ 及其 Cholesky/特征因子）。下游 predictive 是**节点上的解析块之混合**（mixture over $\alpha$），保留为数值对象（无闭式，§7）。

**类型字段建议（Manager 裁决 G7）**：实现者应在 `ResponsePosterior` 类型中承载**每节点 evidence 值 $p(Y\mid\alpha^{(k)})$** 与 **$t$ 分布自由度 $\nu=n+1-N$ 的精确绑定**（推导 §4.4 钉死 $n+1-N$ 版本并拒绝 $n-N$ 版本；实现者检查清单第 15 条显式重申此区分）——二者是审计/重放与下游 t 抽样的最低契约字段。

**未决，需 SPEC 审查（按推导默认建议执行，实现期可复审——裁决 G1）**：返回对象是否携带节点集与证书数值（重放/审计要求，与 `docs/NUMERICAL_INTEGRATION_SPEC.md` §7 第 5 项同一未决）。

---

## 7. Epistemic / aleatoric 分离（D-040/D-041/D-060）

### 7.1 $\Sigma_R$ 的角色边界

$\Sigma_R$ 在本模型中**只**是：(a) working likelihood 的 row covariance（似然的噪声参数）；(b) 系数先验/后验的行协方差（posterior scaling nuisance）；(c) 经 4.3 的 Jeffreys 积分后，以 $\mathcal{IW}(S(\alpha),n)$ 的身份进入 matrix-t 的行尺度——**其作用到 response posterior 的 predictive（$\mu$ 的 $t$ 分布）为止**。

**禁止（D-060 原文，scenario 红线）**：把 $\Sigma_R$（或其 IW 后验抽样）作为**未来残差冲击**再生成一次。scenario generator 的构成必须是（**抽样契约已裁决，Manager 裁决 F**：$\mu$ 通道用 matrix-t predictive **后验抽样**、占独立随机流位置；§6 的 quadrature 确定性、不消耗 RNG）：

$$
\text{response 通道：}\ \mu^{(s)}\sim t_{n+1-N}\big(\hat B\tilde x_t,\tfrac{c}{\nu}S(\alpha)\big)\ \text{（matrix-t predictive 后验抽样；或先抽 }(\alpha_0,\alpha_p)\text{ 再按该节点解析块抽 }\mu\text{）}
$$
$$
\text{innovation 通道：}\ \varepsilon^{(s)}_{t+1}=V_t(d)^{1/2}z\ \text{（vector innovation law，VI：}docs/\text{GATE0\_VECTOR\_INNOVATION.md}\text{）}
$$

二者相加得 $y^{(s)}_{t+1}=\mu^{(s)}+\varepsilon^{(s)}$（asset 域合成链见 §7.3(c)）。$\Sigma_R$ **不出现**在 innovation 通道（记号区分见 §1.2：working 噪声 $\varepsilon^w\neq$ OOF 残差 $\tilde\varepsilon\neq$ innovation $\varepsilon$）。

### 7.2 方差分解的可报告形式（D-030）

$$
\mathrm{Var}(r\mid\mathcal H)=\underbrace{\mathrm{Var}_{response}(\mu)}_{\text{epistemic}}+\underbrace{\mathbb E\,V_{\epsilon}}_{\text{aleatoric（innovation 模块）}},
$$

其中 response 项在 $\alpha$-混合下可再分解（全部可由 §6 的节点集数值算出）：

$$
\mathrm{Var}_{response}(\mu)=
\underbrace{\sum_k p_k\,\frac{c_k}{\nu-2}\,S\!\big(\alpha^{(k)}\big)}_{\text{within：系数条件方差 + }\Sigma_R\text{ 后验（各节点 }t\text{ 协方差之混合）}}
\;+\;
\underbrace{\sum_k p_k\big(\hat B_k\tilde x_t\big)\big(\hat B_k\tilde x_t\big)^\top-\bar\mu\,\bar\mu^\top}_{\text{between：}\alpha\text{ 超参后验变异}},
\qquad
\bar\mu=\sum_k p_k\,\hat B_k\tilde x_t
$$

（within 项为各节点 $t_\nu(\hat B_k\tilde x_t,\tfrac{c_k}{\nu}S(\alpha^{(k)}))$ 的协方差 $\tfrac{\nu}{\nu-2}\cdot\tfrac{c_k}{\nu}S=\tfrac{c_k}{\nu-2}S$ 之混合，要求 $\nu>2$ 即 $n\ge N+2$，否则该矩不存在、报告 NOT COMPUTED；两侧均为 $N\times N$ 矩阵。within/between 记号按 D4 §5 的口径；**与 D4 现状的差别**：D4 记录「超参 between 分量恒为零（by construction）」，本模型中该分量为非零可算——这是 plug-in→完整后验升级的直接后果。）无法计算的部分必须标注 NOT COMPUTED，不得显示 0（D-030）。

### 7.3 输出契约（Manager 裁决 A2/A4/A5，2026-10-09）

response 模块对下游的全部规范输出（与 VI 的接口，锚文本见 §0）：

**(a) asset 空间 OOF 残差行——innovation 层（VI）的规范输入**。对每条历史训练行 $s$：

$$
\boxed{\;\tilde\varepsilon_s\ =\ E_{active}\cdot\Big[y_s-\Big(b_0^{(-fold(s))}+G^{(-fold(s))}\,x_{s-1}\Big)\Big]\;}
$$

- **含 $b_0$ 扣除**：DC 通道参与 OOF 均值预测，残差不得遗漏截距（否则 $b_0$ 的估计误差泄漏进 innovation 层）。
- **D-041 期望语义**：$\hat y=E[y_s\mid train^{(-fold)}]$ 是 fold posterior 的**完整后验均值**（含 $\alpha$ 积分——§6 fold 节点集上 $\hat B^{(-f)}_k\tilde x_{s-1}$ 按 $p_k^{(-f)}$ 的混合均值），不是 plug-in 点估计。
- $\tilde\varepsilon_s\in\mathbb R^{N_a}$ 为 **asset 空间**行；行 mask 契约 $O_s=\{j:r[u_s,j]\ \text{finite}\}$（ResidualOracle 行身份语义，VI §1.2）。innovation 层按决策日 risk 域 $R_t$ 消费其 $R$ 子向量经 $E_{R_t}$ 的 mode 变换（$E_{active}$/$E_{R_t}$ 的维度契约以 VI §1.2 为准；$E_{active}$ 为 active 域等权 mode→asset 重构、$E_{R_t}$ 为 $R_t$ 域等权对应物，裁决 E）。

**(b) $\mu$ 的 asset 空间规范输出**：$\mu_{asset}=E_{active}\cdot\mu\in\mathbb R^{N_a}$（$\mu=B\tilde x_t$ 的 mode 空间 predictive，§4.4）。

**(c) scenario 合成在 risk 域 $R_t$ 子集执行**（Kelly scenario 列集 $=R_t$）：

$$
\mu_R=E_{R_t}^{\top}\,\mu_{asset}[R_t],\qquad
y_R=\mu_R+\varepsilon_R,\qquad
u_R=E_{R_t}\,y_R,\qquad
\text{gross}_R=\exp\!\big(u_R\circ s_1[R_t]\big).
$$

（$\varepsilon_R$ 为 innovation 层在 $R_t$ 域 mode 坐标提供的残差；加法在 mode 子空间、经 $E_{R_t}$ 映回 asset $R$ 子集、乘 $s_1$ 恢复资产尺度、exp 得 gross——SPEC §35 合成链的域收缩版本。）

**(d) 方差分解报告口径（裁决 A5）**：§7.2 的 epistemic/aleatoric 分解在 **asset 空间 $R_t$ 子集上同域可加**（两侧均先经 $E_{R_t}$ 映到同一 $R$ 域再相加报告）；active 域全量 epistemic 分解为**诊断附加项**，标注 **NOT REQUIRED**（不构成验收义务）。无法计算的部分仍标 NOT COMPUTED、不得显示 0（D-030）。

---

## 8. 与 trace neutrality 的关系（D-028/D-029/D-030）

### 8.1 默认 core 不施加 14 条约束：构造性理由

本模型在 gauge 坐标 $q=Q^\top e$ 中工作，这**构造性**实现了 D-030 的两点：

1. **input common 方向不可达**：$B_\perp$ 的列是 relative path 的 gauge 坐标（$Q^\top$ 系），其张成空间天然不含 asset 空间的 common 方向 $\mathbf 1$（$Q^\top\mathbf1=0$）；macro 输入由独立的 $B_m$ 通道显式承载。「所有资产同步的 path 特征」不是被约束**删除**，而是被分解为显式建模对象（$m$ 通道 + $G_{\perp m}$ 块）——它可被数据拟合、也可被后验收缩，不再需要先验禁止。
2. **output common/relative 无冗余表示**：旧架构用 $N_a$ 维 zero-embedded $e$ 表示 $N_a-1$ 维 relative 对象（冗余 1 维），relative $\Sigma$ 需 `supported_covariance` 投影（`src/response.jl:179-182`）；新架构输出 $[m;q]$ 恰为 $N$ 维非冗余坐标，$\Sigma_R$ 直接定义在满维空间。

### 8.2 旧 14 条约束在新坐标中的地位

旧约束 $\mathrm{tr}A_b=\mathrm{tr}B_b=0$（per band per channel 的系数对角和，D3 文档 A 节的精确语义）作用在 **asset 坐标**的 $G$ 上：「资产 $i$ 对自身 Q/P 特征的响应之和为零」。在 mode 坐标中，「资产 $i$ 对自身的响应」不再是 $G$ 的对角元，而是 $M^\top G\tilde M$ 型组合（$M$ 为 mode→asset 的重构矩阵）——**旧约束在新坐标系中没有直接对应物**，无法平移。因此：

- 默认 core：不施加（D-028）；support 的正确性由坐标构造保证（8.1），而非系数约束。
- 旧 trace-conditioned solver（`condition_trace_neutrality`，`src/response.jl:131-148`；`optimize_conditioned_eb`，`src/response.jl:934-1122`）：保留为 hypothesis 实验路径（D-029），其 identification 地位按 D3 裁决为 **candidate hypothesis**（无推导支撑其 identification theorem 地位）；不得成为默认 production core。
- **反例测试义务（D-094）**：构造真实 diagonal common response 的合成世界，验证 unconstrained 新 core 能恢复、旧 trace-neutral branch 系统性删除（§10 第 9 条）。

### 8.3 一个必须钉死的语义区分

「gauge 坐标构造性删除 common 方向」（8.1，**坐标/support 层面**）与「trace 约束删除共同响应」（**系数层面**）是不同几何对象：前者删除的是**不可达的表示方向**，后者删除的是**可达但被假设为零的系数方向**。D-030 的「无需 14 条 trace constraint」指的是前者已接管 support 语义；它不蕴含「共同响应不存在」——共同响应在新模型中由 $m$ 通道与交叉块表达，交给数据与先验精度（$\alpha$ 后验）裁决。**未决，需 SPEC 审查**：D3 文档未决问题 3（「universal common timing response」措辞在含资产间传导解释下过宽）在新架构下的对应措辞收敛。

---

## 9. 与 OOF 的关系（D-043）

每 fold 的 posterior integration 语义：

1. **fold train Gram 独立**：$S_{xx}^{(-f)}=S_{xx}-S_{xx}^{(f)}$（逐元素差，SPEC §28 现有恒等式；$S_{xy},S_{yy},n^{(-f)}$ 同理）。由 §3.2 断言 A1，fold 的**全部**后验计算（evidence、matrix-t、$\mu$ predictive）只需这组差分统计——held-out 行不进入 fold 证据，结构性成立（不是需要额外检查的实现巧合，而是充分统计表示的直接推论；但实现仍须以 §10 第 7 条测试钉死）。
2. **fold 独立 $(\alpha_0,\alpha_p)$ 积分**：每 fold 用自己的 $p(Y^{(-f)}\mid\alpha)$ 走 §6 quadrature——**不复用** full-data 后验、不把 full-data 的 mode 当 fold 超参（D-043 原文）。昨日**同 fold** 的 quadrature 网格可作 numerical warm-start（初始 bracket 收窄），但节点集与质量必须由本 fold evidence 重新证书化。
3. **OOF 残差的消费（输出契约见 §7.3(a)）**：fold posterior 的完整后验均值（含 $\alpha$ 积分，D-041 期望语义）用于 held-out 行，asset 空间 OOF 残差行 $\tilde\varepsilon_s=E_{active}\cdot\big[y_s-(b_0^{(-fold(s))}+G^{(-fold(s))}x_{s-1})\big]$（含 $b_0$ 扣除）交给 innovation 模块——该对象是 VI（`docs/GATE0_VECTOR_INNOVATION.md`）的规范输入；行身份与行 mask 契约以 VI §1.2 为准（$O_s=\{j:r[u_s,j]\ \mathrm{finite}\}$，ResidualOracle 行身份语义）。**OOF 不是模型选择工具**（D-044）：不得据 OOF Sharpe 选 bands/priors/d/universe。
4. **交叉块也严格 OOF**：统一设计使 macro/relative 交叉块在同一 fold 语义内（不存在「macro 用全数据、relative 用 fold」的混合路径——旧架构分拟合使这种混合在结构上可能，新架构单设计使它不可能）。

---

## 10. 实现者检查清单（slow reference 的数学断言）

每条给出：断言、验证方法、失败时的预期行为。全部与回测收益无关。

| # | 断言 | 验证 | 失败行为 |
|---|---|---|---|
| 1 | **A2 数据层**：$n\ge N$ 且 $\mathrm{rank}(S_{yy})=N$ | 构造 $n<N$ 与退化 $Y$ 的 fixture | error，文本含 "posterior improper" |
| 2 | **A3/A4 先验层**：D-035a 修正下右尾证书应通过、后验 proper、quadrature 收敛 | 合成小系统（$N=2,3$；$n$ 小），跑 §6 quadrature，断言收敛且尾质量/refinement 证书通过 | **若红是实现 bug**（修正因子 $1/(1+e^u)$ 遗漏、尾证书实现错误等）；仅当 owner/SPEC 否决 D-035a 回退 $1/\alpha$ 时，红才恢复「恒 improper 的预期红」语义（错误文本含 "posterior improper: alpha tail"） |
| 3 | **充分统计恒等**：$S(\alpha)=S_{yy}-S_{xy}^\top(S_{xx}+\Lambda_\alpha)^{-1}S_{xy}$ 与直接构造 $Y^\top(I-XA^{-1}X^\top)Y$ 数值一致（含 $\alpha$ 扫描） | 小系统逐 $\alpha$ 对照，atol 按条件数 | 不一致即实现错误（无合法容差） |
| 4 | **matrix-t 边缘化 vs dense**：$p(B\mid Y,\alpha)$ 的核 $\big|S+(B-\hat B)V^{-1}(B-\hat B)^\top\big|^{-(n+P)/2}$ 与小系统上对 $(B,\Sigma)$ 的**显式数值积分**一致 | $N=2,P\le3$ 网格数值积分对照 | 相对误差不收敛即公式错 |
| 5 | **$\mu$ 的 t 边缘**：$\mu\mid Y,\alpha$ 的均值 $=\hat B\tilde x_t$、协方差 $=\tfrac{c}{\nu-2}S(\alpha)$（$\nu>2$ 时；由 $\tfrac{\nu}{\nu-2}\times\tfrac{c}{\nu}S(\alpha)$ 而来）与 dense 数值积分一致 | 同上小系统 | 同上 |
| 6 | **先验退化极限**：$p(\alpha)\to\delta(\alpha^\*)$（点质量）时，$B$ 后验回到 ridge 条件后验 $\mathcal{MN}(\hat B,\hat\Sigma,V)$（$\hat\Sigma$ 为固定 $\Sigma$ 版本）；进一步固定 $\Sigma_R=\hat\Sigma$ 时与旧 `fit_response_operator` 的条件矩（`src/response.jl:1199-1202`）在同 design 上一致 | 用旧 design（无 DC、block-diag）构造桥接测试，仅验证公式族连续性 | 极限不连续即推导错 |
| 7 | **OOF 隔离**：改动 fold $f$ 的 held-out 目标行，$S^{(-f)}$、fold evidence、fold $\hat B^{(-f)}$ 不变 | 直接改数对照 | 任何变化即泄漏 |
| 8 | **交叉块恢复（D-092）**：合成世界设 $G_{\perp m}\neq0$（或 $G_{m\perp}\neq0$），新 core 的 $\hat B$ 恢复该块（后验均值方向正确、幅度随 $n$ 收敛）；旧 block-diagonal 实现故意失败（对照面证明修复了被删块） | 合成数据 + 参数恢复 | 新 core 不能恢复即实现错 |
| 9 | **trace 反例（D-094）**：合成真实 diagonal common response，unconstrained core 恢复；旧 trace-neutral branch 系统性删除 | 合成数据对照 | （预期红绿分工，见 D-094） |
| 10 | **DC 零中心（D-093）**：合成 constant drift 世界，$b_0$ 后验均值恢复 drift、dynamic path 系数保持 0；零信号世界 $b_0$ 后验均值 $\to0$ | 合成数据 | 先验注入即错（D-032） |
| 11 | **quadrature 收敛（D-066/D-067）**：网格加密时 $\mu$ 矩、$\alpha$ 矩稳定；同 seed 逐位重放；预算耗尽 error（含 "Numerical integration did not converge"） | 细化序列 + 重放 | 任何静默返回即违规 |
| 12 | **permutation/gauge 协变**：资产列置换 $\Pi$ 下 $\mu$（asset 空间重构后）$\to\Pi^\top\mu$；$Q\to QR$ 下不变 | 构造置换/旋转 fixture | 依赖列编号或 gauge 选择即 bug |
| 13 | **epistemic/aleatoric 报告（D-030）**：§7.2 分解可输出，within/between 分量非零可算（between 在 plug-in 旧实现下恒 0——对照面）；不可算字段标 NOT COMPUTED | 报告字段检查 | 显示 0 即违规 |
| 14 | **$\Sigma_R$ 不进 innovation 通道（D-060）**：scenario 代码审查级断言——innovation 通道的输入不含 $\Sigma_R$/IW 抽样 | 静态审查 + 接口签名 | 出现即重复计算违规 |
| 15 | **$\nu$ 绑定（裁决 G7）**：`ResponsePosterior` 类型承载每节点 evidence 值 $p(Y\mid\alpha^{(k)})$ 与 $t$ 自由度 $\nu=n+1-N$ 的精确绑定；$\nu$ **必须**为 $n+1-N$（§4.4 钉死：response 输出到 $\mu$ 为止、不含未来冲击），**禁止** $n-N$ 版本（那是含新观测噪声的 predictive，属 innovation 层职责的混淆形态） | 类型字段静态检查 + 小系统 t 矩对照（均值 $=\hat B\tilde x_t$、$\nu>2$ 时协方差 $=\tfrac{c}{\nu-2}S$） | $\nu$ 取 $n-N$ 即层级混淆 bug |

---

## 11. 与 `docs/POSTERIOR_DEFINITION.md` 的条款级差异表

D4 是现状盘点（plug-in EB 的事实陈述）；本文是其替代对象的定义。逐条：

| D4 条款 | D4 现状 | 本文 | 关系 |
|---|---|---|---|
| §1 #1（$G$ 条件后验，半积分对象） | $G\mid\hat\alpha,\hat\Sigma$ 条件 Gaussian | $B$ 有完整 matrix-t 边缘（$\Sigma_R$ 解析积分 + $\alpha$ 数值积分） | **升级** |
| §1 #2（$G_{macro}$） | macro 独立链 + $\hat\sigma^2$ | 并入统一 $B/\Sigma_R$（D-039） | **推翻**（结构合并） |
| §1 #3（$\alpha_m,\alpha_{rel}$ 点估计） | EB argmax plug-in | $(\alpha_0,\alpha_p)$ 联合后验，2 维 quadrature | **推翻**（D-034/D-038）——propriety 发现的先验修正已由 **D-035a** 裁决闭合（§5.4） |
| §1 #4（$\Sigma$ 点估计） | $\hat\Sigma$ plug-in | $\Sigma_R\sim\mathcal{IW}(S(\alpha),n)$，解析积分 | **推翻**（升级为完整积分） |
| §1 #5（$\sigma^2$ 点估计） | `sse/(n-γ)` | 取消独立点估计，由 $\Sigma_R$ 的 $(1,1)$ 块承担 | **推翻**（D-039） |
| §1 #6（$d$ 离散后验） | — | 本文不处理（innovation 层） | 不变（归 GATE0_VECTOR_INNOVATION.md） |
| §1 #7（残差经验测度） | — | 同上 | 不变 |
| §1 #8（$scale/v_{bootstrap}$） | — | 同上（D-059 属 innovation 裁决） | 不变 |
| §1 #9（几何量非推断对象） | 确定性变换 | 沿用（$Q,s_\perp$ 等仍为固定 gauge，D-024） | 一致 |
| §2（plug-in 结构精确陈述） | 「epistemic approximation」 | 被 §3-§6 的完整后验取代（D-035a 已裁决落地） | **推翻** |
| §3（目标层次模型） | 抽象形式 + 全部留白 | 先验形式按 D-035 钉死、积分路线闭合；propriety 发现新增一个必须裁决的留白 | **部分升级**（留白大幅收敛，但新增 §12 第 1 条） |
| §4（积分策略顺序） | 4 阶建议 | 第 1-2 阶（解析 + 2 维 quadrature）完全实现；第 3 阶（$\Sigma$ Laplace）**不需要**；第 4 阶（MCMC）不需要 | **升级并简化** |
| §5（方差分解） | between 超参分量 = 0 by construction | between 分量非零可算（§7.2） | **升级** |
| §6（路线 A/B） | 二选一待裁决 | 裁决书已选 B（D-034）；本文给出 B 的完整数学，propriety 前提已由 D-035a 闭合 | **执行**（B 的数学与前提均闭合） |
| §7 未决 1/2（$\alpha,\Sigma$ 先验） | 留白 | D-035 已钉（Jeffreys $\Sigma$）；$\alpha$ 尾部修正经 D-035a 裁决采纳 | **钉死**（§12 第 1 条已裁决闭合） |
| §7 未决 10（$\Pi$ 定义域含不含超参） | 最需先裁决 | 本文按「含超参」定义域推导（$\Pi$ 覆盖 $B,\Sigma_R,\alpha_0,\alpha_p$） | **钉死**（供 SPEC 确认） |

---

## 12. 未决事项清单（终态：含 Manager 裁决闭合项，裁决记录 `docs/GATE0_MANAGER_ADJUDICATIONS.md`）

1. **$\alpha$ 先验尾部修正——已裁决闭合（D-035a）**：本文最重要发现（§5.3-§5.4）：D-035 的 $1/\alpha$ log-uniform 先验在无界支撑上使后验恒不 proper（$\alpha\to\infty$ 处 evidence 饱和于零模型平台 × log-uniform 尾测度 = 无穷质量）。**Manager 已裁决采纳 D-035a（2026-10-09）**：$p(\alpha)\propto1/[\alpha(1+\alpha)]$，依据纯数学 propriety（保 reference 局部行为、截断零模型平台），非回测选择；可推翻条件：owner/SPEC 否决则回退 $1/\alpha$ + 恒红（§5.4）。有界域截断仍被 D-038 禁止且数学无效。
2. **path group 划分**（保持未决，按推导默认建议执行，实现期可复审——裁决 G1）：$\alpha_p$ 覆盖 macro path 与 relative path 全体（本文按 D-033 字面：是）；拆分属理论版本变更。
3. **ragged 行的 mode 分解权重语义**（核心已裁决闭合——裁决 E）：$m_s$ 按行级观察集 $O_s$ 归一（$1/\sqrt{|O_s|}$）、$E_{active}$/$E_{R_t}$ 用各自域等权（域基 ≠ 行级 field 定义，§1.4）；Step 3 的其余实现细节保持未决，按推导默认建议执行、实现期可复审（裁决 G1）。
4. **quadrature 输出对象的审计字段**（§6.4；保持未决，按推导默认建议执行，实现期可复审——裁决 G1；G7 的 evidence/$\nu$ 绑定字段已是最低契约）：节点集/证书是否随对象返回。
5. **$\mu$ 通道的 scenario 抽样语义——已裁决闭合（裁决 F）**：$\mu$ 通道用 matrix-t predictive 后验抽样（独立随机流位置）；quadrature 确定性、不消耗 RNG（§7.1）。
6. **D3 未决 3 的措辞收敛**（§8.3；保持未决，按推导默认建议执行，实现期可复审——裁决 G1）：新架构下「common response」的规范措辞。
7. **$\epsilon_{quad}$ 数值**（§6.3；保持未决，按推导默认建议执行，实现期可复审——裁决 G1）：走 D-066 流程。

---

## 13. 关键断言摘要（供评审）

1. **A1**：整个后验由 $(n,S_{xx},S_{xy},S_{yy})$ 充分决定；$S(\alpha)=S_{yy}-S_{xy}^\top(S_{xx}+\Lambda_\alpha)^{-1}S_{xy}$。
2. **A2**：$\Sigma$ 积分有限要求 $n>N-1$ 且 $S(\alpha)\succ0$（$n\ge N$ + $Y$ 列满秩即足）。
3. **A3/A4**：D-035 原始先验下后验对 $\alpha$ **恒不 proper**（右尾饱和发散；左尾无条件收敛）——静态数学事实；修正 $1/[\alpha(1+\alpha)]$ 已由 Manager 裁决采纳为 **D-035a**（2026-10-09，依据纯数学 propriety；可被 owner/SPEC 否决回退，届时回到恒红）。
4. **A5**：给定 $\alpha$，$(B,\Sigma_R)$ 整体解析消元：evidence $p(Y\mid\alpha)\propto|I_P+\Lambda_\alpha^{-1}S_{xx}|^{-N/2}|S(\alpha)|^{-n/2}$；$B\mid Y,\alpha$ 为 matrix-t（核 $|S+(B-\hat B)V^{-1}(B-\hat B)^\top|^{-(n+P)/2}$）；$\mu=B\tilde x_t$ 为 $t_{n+1-N}(\hat B\tilde x_t,\tfrac{c}{\nu}S(\alpha))$。数值积分只剩 $\mathbb R^2$。
5. **A6**：log 坐标上 quadrature 被积函数 = marginal evidence 本身（$1/\alpha$ 与 Jacobian 相消）。
6. **A7**：$\Sigma_R$ 作用到 $\mu$ 的 predictive 为止；innovation 通道不得再消费 $\Sigma_R$（D-060）。
7. **A8**：gauge 坐标工作构造性删除 common 表示方向（support 层），trace 约束（系数层）在新坐标无可平移对应物——默认 core 不施加，旧 solver 留 hypothesis（D3 裁决）。

---

## 14. 证据索引

- 裁决依据：AGENTS.md §7-§11（D-021~D-039）、§19（D-060）、§29（D-085/D-086）、§30（D-087-D-089）、§33（D-090-D-095）、§40 Step 7；Manager 终审裁决（2026-10-09，`docs/GATE0_MANAGER_ADJUDICATIONS.md`：D-035a 先验修正、输出契约 A2/A4/A5、接口锚 D、记号区分 D4、ragged 归一 E、抽样契约 F、类型字段 G7、复审口径 G1）。
- 现状对照源码：`src/response.jl:1124-1209`（`fit_response_operator`：分拟合、EB 点估计、trace conditioning）、`src/response.jl:131-148`（`condition_trace_neutrality`）、`src/response.jl:212-273`（`maximize_logalpha`，仅作 bracket）、`src/response.jl:351-380, 934-1122`（被降级的 EB optimizer）、`src/predict.jl:43-70`（`embedded_relative_field`）、`src/predict.jl:286-298`（现有 design/target 构造）、`src/predict.jl:473-478`（$e_0$）、`src/prepare.jl:83-165`（`PreparedProblem` 契约）、`src/numerics.jl:368-380`（Helmert gauge）。
- 姊妹文档：`docs/POSTERIOR_DEFINITION.md`（现状盘点，§11 差异表的对象）、`docs/TRACE_NEUTRALITY_DERIVATION.md`（candidate hypothesis 裁决）、`docs/GATE0_VECTOR_INNOVATION.md`（VI：innovation 层边界，§0/§7.3 接口锚的对象）、`docs/NUMERICAL_INTEGRATION_SPEC.md`（scenario 层收敛证书与错误文本规范）。

---

*（本文件为 Gate-0 Step 7 纸面推导交付物，并已按 Manager 终审裁决（2026-10-09）完成修订：D-035a 先验修正（§5.4/§6/§10/§11/§12/§13）、§7.3 输出契约（A2/A4/A5）、VI 接口锚与引用更名（D）、记号区分（D4）、ragged 归一（E）、抽样契约（F）、类型字段建议（G7）。未运行任何命令，未修改任何源码、测试或既有文档；所有先验与积分选择均未以回测表现为依据；剩余未决项以 §12 终态清单为准。）*
