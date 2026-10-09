# KTrader Gate-0 Vector Innovation Law — 规范推导

**文档状态：规范推导（Normative Derivation）——innovation 层重写的数学基石。**
**规范来源：AGENTS.md Gate-0 裁决书 §14–§19（D-045～D-060、D-57～D-59、D-040～D-042）；本文档是 docs/INNOVATION_LAW.md 目标条款按裁决书的最终升级版。**
**性质：纯数学推导与规范定义。本文档不写代码、不运行命令、不修改任何源码；落地须另立受控施工（裁决书 §40 Step 10–13）。**
**接口锚（与并行撰写的 docs/GATE0_RESPONSE_POSTERIOR.md 共用，两文档必须逐字一致；Manager 裁决 D 统一锚文字，2026-10-09）：**

> **ε 为统一 mode 残差：源对象是 asset 空间 OOF 残差行 $\tilde\varepsilon_s = E_{\mathrm{active}}\cdot\big[y_s-\big(b_0^{(-\mathrm{fold}(s))}+G^{(-\mathrm{fold}(s))}\cdot x_{s-1}\big)\big]$（含 $b_0$ 扣除，$\hat y=E[y_s\mid\mathrm{train}^{(-\mathrm{fold})}]$ 为 D-041 期望语义）；innovation 层按决策日 risk 域 $R_t$ 消费其 $R$ 子向量经 $E_{R_t}$ 的 mode 变换。**

**证据基准：当前工作树源码只读核对（src/predict.jl、src/residual_oracle.jl、src/numerics.jl），引用为 `file:line`。**
**纪律：本文档所有选择均由裁决书条文或数学可定义性推出；任何先验、kernel、support、收敛判据都不得以回测收益/Sharpe 作为裁决依据（裁决书 §48；SPEC §95；开发守则 §24）。初稿与裁决书的张力（T1–T7）已由 Manager 终审裁决（2026-10-09；裁决 A/A2/A4/A5/C/D/D2/E/F，全文落盘于 docs/GATE0_MANAGER_ADJUDICATIONS.md）闭环，终态见 §11。**

---

## 0. 阅读约定与文档地位

- $\mathcal H_t$ 为价格历史信息集（决策日 $t$ 收盘已知的一切价格事实）。
- $N$：active 资产数（admission 语义，D-013：prefix 中至少一条有效 daily return）；mode 空间维数同为 $N$（macro 1 维 + relative $N-1$ 维）。
- $R_t \subseteq \{1,\dots,N\}$：决策日 $t$ 的 **risk 域**——需要为其生成 scenario 风险的全部资产集合，即 free risky set 与 locked 持仓的并（D-016/D-017）。$N_R = |R_t|$。
- $\varepsilon_s \in \mathbb R^{N}$：统一 mode 残差（接口锚；§1.2 给出其最小语义契约）。下标 $s$ 为**残差的目标日**（该残差评价的 return 之实现日，见 §1.3）。
- $V_t(d) \in \mathbb R^{N_R \times N_R}$：本文档的主对象——vector innovation memory。
- $z$：standardized shape（标准化形状）。
- 「joint 合法行」：进入统计的历史残差行（§2.2、§6；本文档证明 $V$ 统计与 shape pool 在规范设计下使用**同一个行集**）。
- 记号 $\log\det^+$ 与 $V^+$：Moore-Penrose 语义（§4）。
- 凡本文档与 docs/INNOVATION_LAW.md 冲突：以本文档为准（差异表见 §10）；凡本文档与裁决书冲突：以裁决书为准（张力见 §11）；凡与源码冲突：源码事实是对照对象，不是规范。

本文档的目标：**陌生工程师只读本文档即可实现 vector innovation reference**（单线程、无 FFT、无增量、直接 $O(T^2N_R^2)$ 均可，裁决书 §82/D-083：reference 不追求速度，数学正确优先）。

---

## 1. 坐标系统与统一 mode 残差

### 1.1 mode 坐标与正交变换

对任一资产域 $A$（$|A| = n$），定义正交矩阵

$$
E_A = [\,e_0^{(A)}\;\; Q^{(A)}\,] \in \mathbb R^{n\times n},
\qquad
e_0^{(A)} = \tfrac{1}{\sqrt n}\mathbf 1_n,\qquad
Q^{(A)\top}Q^{(A)} = I,\;\; Q^{(A)\top}\mathbf 1_n = 0,
$$

$Q^{(A)}$ 为 $A$ 域的 Helmert 固定数值 gauge（D-023：纯数值 gauge，不是板块、不是 factor、不是市场模式）。任何资产空间向量 $v\in\mathbb R^{n}$ 的 mode 表示为

$$
\hat v = E_A^\top v = \begin{bmatrix} e_0^{(A)\top} v \\ Q^{(A)\top} v\end{bmatrix}\in\mathbb R^{n},
\qquad
v = E_A\hat v .
$$

因 $Q^{(A)\top}\mathbf 1 = 0$，有 $Q^{(A)\top}v = Q^{(A)\top}\big(v-\bar v\mathbf 1\big)$：mode 变换自动把 $v$ 分解为（等权均值部分，零和部分）。$E_A$ 正交，故内积、范数、协方差在两坐标系下严格守恒（正交相似）：

$$
\|\hat v\| = \|v\|,\qquad \hat u^\top\hat v = u^\top v,\qquad
\widehat{\Sigma} = E_A^\top \Sigma\, E_A .
$$

**归一语义注（裁决 E1，2026-10-09）**：$E_A$（含 $e_0^{(A)}=\mathbf 1_n/\sqrt n$，$n=|A|$ 为域大小）是**域级等权基**，与具体行的观测集无关。response 层行级 field 的归一（$m_s$ 按该行观察集 $O_s$ 归一，$1/\sqrt{|O_s|}$，SPEC §12）与 $E_{\mathrm{active}}/E_{R_t}$ 的域级等权基是**两个概念**：前者是 response 层 $y_s$ 的 field 定义，后者是本层 mode 变换的域基定义。本层只消费 asset 空间源对象 $\tilde\varepsilon_s$（§1.2），其上的 $E_{R_t}$ 变换按域级等权基执行，二者不得混同。

### 1.2 统一 mode 残差：最小语义契约（接口锚的实例化）

**ε 为统一 mode 残差（接口锚，§0 逐字；Manager 裁决 D 统一锚文字）。** 其完整拟合链条（目标 $y_{t+1}=[m_{t+1};\,q_{t+1}]$、统一 response $\hat y=b_0+Gx$、fold 隔离）由 docs/GATE0_RESPONSE_POSTERIOR.md 定义并唯一所有；本文档只钉死 innovation 层消费所需的**最小契约**：

1. **坐标与源对象**：mode 残差 $\varepsilon_s^{\mathrm{mode}} = y_s-\hat y_s^{(-\mathrm{fold}(s))}\in\mathbb R^{N}$，在 active 域 mode 坐标 $[\,m;\;Q^\top e\,]$ 下表达（D-025 的统一 mode output 坐标）；**源对象**为其 asset 空间表示 $\tilde\varepsilon_s = E_{\mathrm{active}}\,\varepsilon_s^{\mathrm{mode}}\in\mathbb R^{N}$（asset 空间 OOF 残差行，§0 锚文字）——innovation 层的一切消费从 $\tilde\varepsilon_s$ 出发（§1.3）。
2. **OOF 语义（含 $b_0$ 扣除）**：$\hat y_s^{(-\mathrm{fold}(s))} = b_0^{(-\mathrm{fold}(s))}+G^{(-\mathrm{fold}(s))}\cdot x_{s-1} = E[y_s\mid\mathrm{train}^{(-\mathrm{fold}(s))}]$（D-041 期望语义）由不含行 $s$ 的训练数据拟合（D-043：每 fold 独立后验、held-out 行不进 fold 证据）；行 $s$ 的残差是 out-of-fit 创新的实现值（D-041：innovation 层的输入是 cross-fitted residual）。**$b_0$ 扣除必须包含**：残差是相对完整条件期望（含 DC 通道）的偏差，不是相对动态路径部分的偏差。
3. **时间戳**：行 $s$ 的目标日 $u_s$ = 该行所评价的 return 之实现日。现有 ResidualOracle 契约的行身份（oracle 行 idx 对应全局日 $t=\texttt{ts\_total[idx]}$、目标为 $r[t+1,:]$，`src/residual_oracle.jl:17-22`）给出实例：$u_{\text{idx}}=\texttt{ts\_total[idx]}+1$，行序 = 目标日序，严格递增。
4. **mask**：每行携带其资产观测掩码 $O_s=\{j : r[u_s,j]\ \text{finite}\}$（**与 RP 逐字一致，裁决 D2**；`src/residual_oracle.jl:19-20` 的行自身历史 mask 语义）。mode 残差在缺失坐标上的数值**不承担概率语义**（§1.4）。
5. **有限性**：被消费的 $\varepsilon_s$ 分量必须有限；NaN/Inf 进入统计即 fail-loudly（SPEC §56）。

### 1.3 行合法性：mode 混合性 ⇒ 行级 joint 语义（D-054 的数学必然）

**命题（mode 混合性）。** 在 mode 坐标 $[\,m;\,Q^\top e\,]$ 下，$m=e_0^\top(\cdot)$ 与每个 $q_k=(Q_{:k})^\top(\cdot)$ 都是**全部 $N$ 个资产坐标的线性组合**。因此任何一个资产坐标的观测缺失，都使该行 mode 残差的**每一个坐标**失去联合意义：零嵌入（D-022）只是 field 代数表示，不是「缺失收益 $=0$」的概率断言；缺失坐标以 $e_j=0$ 代入后，$m$ 与全部 $q_k$ 的值都是「嵌入值」而非「残差实现值」。

**推论 1（joint 行规则的数学根源）。** 任何以 mode 坐标为对象的联合统计（$V_t$ 的外积、shape pool 的行、$\ell_d$ 的二次型）只能以**整行为单位**判定合法性：行 $s$ 联合合法 ⟺ $O_s\supseteq$ 统计所需的全部资产坐标。不存在「按坐标拼接」的合法形式——这不是直觉禁令，而是 mode 坐标代数的必然。**D-054 禁止 own-row cell stitching 在 mode 坐标下结构性不可能成立**（对照：现有 own-row fallback `src/predict.jl:590-604` 之所以可能，是因为它工作在 asset 坐标的逐 cell 消费上；该通道被 D-054 废除后，asset 坐标下的拼接同样被禁——scenario innovation 必须保持联合残差向量的横截面依赖，D-054 原文理由）。

**推论 2（域实例化）。** 决策日 $t$ 的 scenario innovation 服务于 Kelly 的 risk 域 $R_t$（free ∪ locked；D-016/D-017）。因此统计所需坐标集 $=R_t$：

$$
\boxed{\ \text{行 } s\ \text{joint 合法（对决策日 } t\text{）}\ \Longleftrightarrow\ O_s \supseteq R_t\ \ \text{且}\ \ u_s\le t .\ }
$$

innovation 层的全部对象定义在 **$R_t$ 域的 mode 坐标**上：$E_{R_t}=[\,e_0^{(R_t)}\,;\,Q^{(R_t)}\,]$，维数 $N_R$。历史行 $s$ 的 $R$-域 mode 残差

$$
\varepsilon_s^{(R)} = E_{R_t}^\top\,\tilde\varepsilon_s^{(R)}\in\mathbb R^{N_R},
\qquad
\tilde\varepsilon_s^{(R)} = \big(\text{asset 空间 OOF 残差行 } s\big)_{R_t}\ \text{子向量},
$$

其中 asset 空间 OOF 残差行由 response 层契约给出（现有对照公式 `src/residual_oracle.jl:26-28`；新 response 的残差公式由 GATE0_RESPONSE_POSTERIOR.md 所有）。行覆盖 $R_t$ ⟹ $R$ 子向量全观测 ⟹ $\varepsilon_s^{(R)}$ 是无信息损失的联合实现值。

**接口锚在两层的关系（裁决 A2 精确路径，2026-10-09）**：response 层定义源对象——asset 空间 OOF 残差行 $\tilde\varepsilon_s = E_{\mathrm{active}}\cdot\big[y_s-\big(b_0^{(-\mathrm{fold}(s))}+G^{(-\mathrm{fold}(s))}\cdot x_{s-1}\big)\big]$（含 $b_0$ 扣除）；innovation 层**从该 asset 空间残差行提取 $R_t$ 子向量，经 $E_{R_t}$ mode 变换构造 $V_t$/$z$/pool**（本层规范职责）。「mode 残差的 $R$ 子向量」这类表述在数学上无定义（mode 坐标无资产子集索引），一律以上述 asset 空间提取路径为准。锚点文字两文档逐字一致（§0）；$R$ 域实例化已由 Manager 裁决采纳为 **D-045a 澄清条款**（2026-10-09，裁决 A；§11 T3 终态：已裁决采纳）。

**为什么必须是 $R$ 域而非恒为 active 域（拒绝变体）。** 若 $V_t$ 恒取 active 域（$N$ 维），则行合法性须为 $O_s\supseteq$ 全部 active 坐标；此时 D-056 的出路（「该资产不进入 free risky set」从而缩小覆盖要求、恢复 joint 历史）**结构性失效**——剔除一个 free 资产不改变 active 域，行集不恢复，cash 语义无法兑现。$R$ 域实例化是同时兑现 D-055（「覆盖全部 risky assets」字面）与 D-056（cash 出路）的唯一自洽设计。active 域变体被本文档拒绝，理由如上（非回测理由）。

### 1.4 零嵌入的边界（重申 D-022）

未观测坐标在 field 代数中嵌入为 0 **只**表示「当前 active 坐标空间中 relative field 的嵌入值为零」，不表示 $r=0$。在 innovation 层：零嵌入值**永不进入** $V_t$ 的二阶矩统计、shape pool、$\ell_d$ 的二次型（§2.2 的行集收缩正是其执行机制）；observation mask 永久保留（§1.2 契约第 4 条）。空观察行（$O_s=\varnothing$，如现有 `src/residual_oracle.jl:239` 的 $c=0$ 情形）不进入任何统计——它不是「零残差」。

---

## 2. $V_t(d)$：vector innovation memory（D-045/D-046/D-048/D-049）

### 2.1 kernel：统一 fractional family（D-046）

对 $d\in(0,1]$：

$$
k_d(\tau) := \pi_{\tau-1},\qquad \tau\ge 1,
\qquad
\pi_0=1,\qquad
\pi_k=\pi_{k-1}\,\frac{k-1+d}{k}\ \ (k\ge1),
$$

与现有 `frac_weights`（`src/predict.jl:12`；SPEC §31 的 $\pi_k$）**同一递推、同一 family**。$d=1$：$\pi_k\equiv1$（等权）；$d\to0^+$：$\pi_k\to1/k$（近期主导）。对一切 $d\in(0,1]$、一切 $k$：$\pi_k>0$（递推系数 $(k-1+d)/k>0$）——**kernel 权重严格为正，无硬截断**。

**禁止**（D-046 原文）：macro 一套、relative 一套、每资产一套、每板块手工一套 kernel。$V_t$ 的全部坐标共用同一 $k_d$——这保持 Maxwell 式统一。

**lag 约定。** 行（目标日 $u$）在 $V_t$ 中的 lag 为 $\tau=t+1-u$（$u\le t\Rightarrow\tau\ge1$；最近已实现行 $u=t$ 的 $\tau=1$、权重 $\pi_0=1$）。与现有 scalar 实现的因果卷积方向一致（`variance=output[t-1]/cs[t-1]` 用 $<t$ 的信息预测 $t$，`src/predict.jl:112`）；oracle 行 idx 与目标日之间为严格单调平移，不影响 kernel 的相对结构。

### 2.2 定义（维度、测度、行集）

**主对象。** 对决策日 $t$、risk 域 $R_t$、$d\in(0,1]$：

$$
\boxed{\;
V_t(d)
=
\frac{\displaystyle\sum_{\tau\ge1}\ \mathbf 1\{\text{目标日 } t+1-\tau\ \text{有 joint 合法行}\}\ k_d(\tau)\ \varepsilon^{(R)}\varepsilon^{(R)\top}}
{\displaystyle\sum_{\tau\ge1}\ \mathbf 1\{\text{目标日 } t+1-\tau\ \text{有 joint 合法行}\}\ k_d(\tau)}
\;\in\;\mathbb R^{N_R\times N_R}\;}
$$

其中求和行集为

$$
J_t \;=\; \{\,s : u_s\le t,\ O_s\supseteq R_t\,\}
\qquad(\text{joint 合法行集，§1.3 推论 2}),
$$

$\varepsilon^{(R)}=\varepsilon_s^{(R)}$ 为该行的 $R$-域 mode 残差。要点：

1. **分子分母同步收缩到 $J_t$**。分母不是裸的 $\sum_{\tau\ge1}k_d(\tau)$。理由（数学可定义性，D-049 精神）：外积 $\varepsilon\varepsilon^\top$ 仅对联合合法行良定义；若分母含缺失行权重而分子不含，等于把「缺失」伪装成「零方差证据」——这正是 SPEC §12.1/D-012 禁止的观测污染在二阶矩层的翻版。裁决书 D-045 字面分母 $\sum_{\tau\ge1}k_d(\tau)$ 是「全部行 joint 合法（完整 panel）」情形的字面形式；ragged 情形按本条收缩（**已裁决采纳**，Manager 裁决 C，2026-10-09；§11 T1 终态）。
2. **全部可用因果历史（D-048）**：$J_t$ 无固定 rolling window；**禁止** 63 日/252 日/3 年/5 年等任何回测口径窗口。kernel 自身决定远历史权重（$\pi_k\sim k^{d-1}$ 多项式衰减，$d<1$ 时 $\sum\pi_k<\infty$，有效记忆有限但由 $d$ 数据驱动决定，不由人工窗口决定）。
3. **burn 废除（D-049）**：不存在硬编码 burn=30（现有 `src/predict.jl:76` 的 `burn=30` 退出理论）。**有效起点由数学可定义性决定**：$V_t(d)$ 可定义 ⟺ $|J_t|\ge1$（分母 $>0$）；$z_s$ 可定义 ⟺ $V_{s-1}$ 可定义（§5）；$\ell_d$ 的求和行可定义 ⟺ 同（§3）。早期没有足够过去行形成 $V_{s-1}$ 的行**直接不进入 likelihood/score**——不补零、不跳过计分、不加 burn 偏置。
4. **测度语义**：$V_t(d)$ 是 joint 合法历史行的 kernel 加权二阶矩（经验测度的加权平均）；分母使其成为加权平均（尺度归一），不是累计和。

### 2.3 PSD 构造性与秩（D-050）

$$
V_t(d)=\sum_{s\in J_t} w_s\,\varepsilon_s^{(R)}\varepsilon_s^{(R)\top},\qquad
w_s=\frac{k_d(t+1-u_s)}{\sum_{s'\in J_t}k_d(t+1-u_{s'})}>0 ,
$$

是半正定矩阵的非负系数组合，**构造性 PSD**（$x^\top V_t x=\sum w_s(x^\top\varepsilon_s)^2\ge0$）。

**秩引理（d 无关性）。** 因全部 $w_s>0$：

$$
\operatorname{rank}V_t(d)=\dim\operatorname{span}\{\varepsilon_s^{(R)} : s\in J_t\}\quad\text{对一切 }d\in(0,1]\text{ 成立}，
$$

即**秩与 $d$ 无关**，只由行集 $J_t$ 的残差向量张成子空间决定。推论：(a) 不存在「小 $d$ 通过秩亏获得似然优势」的偏置通道（§3 的 $\ell_d$ 比较在相同秩下进行）；(b) 数值上 kernel 权重极小时的有效秩差异属于浮点分类领域（§4.3），不是数学对象差异。

**rank deficient 的规范语义（D-050 原文）**：若某方向历史数据不足（$|J_t|<N_R$ 或行向量线性相关），该方向 variance 可以为 0——**未识别 support**。不得通过 $V+\delta I$ 偷偷创造理论风险（§4.2）；该方向的风险贡献为 0 是「历史未提供该方向二阶矩证据」的忠实表达。

**分母为零（$|J_t|=0$）**：$V_t$ 不可定义 → **fail loudly**（D-056 的 fail 优先；错误先于任何拼接/补零）。

### 2.4 mode 分解：macro / relative / cross 块与风险轮动（D-045）

$R$ 域 mode 坐标下按 (1, 其余) 分块：

$$
V_t(d)\ \text{的坐标结构}\ =\
\begin{bmatrix} V_m & c_t^\top \\ c_t & V_\perp\end{bmatrix},
\qquad
V_m\in\mathbb R,\ \ c_t\in\mathbb R^{N_R-1},\ \ V_\perp\in\mathbb R^{(N_R-1)\times(N_R-1)} .
$$

- $V_m$：**macro 方差**（$R$ 域等权 COM 方向的条件二阶矩）；
- $V_\perp$：**relative 协方差**（零和支撑内的条件横截面风险）；
- $c_t$：**macro↔relative 交叉协方差**——D-045 原文「天然包含 macro-relative cross covariance」，与 response 层 D-025/D-026（允许 $G_{m\perp},G_{\perp m}$ cross block）同构对应：innovation 层**不施加** block-diagonal 假设，$c_t\neq0$ 完全合法（INNOVATION_LAW.md §6.5 的「强制 block-diagonal」候选被裁决书否决）；
- **mode 风险轮动**：$V_\perp$（等价地 $Q^{(R)}V_\perp Q^{(R)\top}$）的谱随 $t$ 演化 = 风险在自然 mode 方向间的轮动；这是 vector law 相对 scalar law 的结构性新增信息（scalar 只见 $V_m$ 一维投影，见 §12 对照）。

### 2.5 极限行为

- **$d=1$**（闭区间端点极限，D-047）：$k_1\equiv1$，$V_t(1)=\frac1{|J_t|}\sum_{s\in J_t}\varepsilon\varepsilon^\top$（joint 合法历史的等权二阶矩）。
- **$d\to0^+$**（开区间极限，不包含 $0$；d→0⁺ 的真极限是最近行独占：frac_weights 递推 π₀=1 恒定、π_k→0（k≥1），故 V_t → ε_last ε_lastᵀ；调和衰减 d/τ 只是固定小 d 的中间行为，不是 d→0 的极限（Manager 裁决 2026-10-10，Wave 3 裁决 2；innovation_tests 9.5 已按真极限断言并全绿）。。$d=0$ 本身不在先验支撑内。
- **$N_R=1$**（单资产 risk 域）：$V_t(d)\in\mathbb R^{1\times1}$ 退化为标量 $v_t(d)=\frac{\sum k_d\,\varepsilon^2}{\sum k_d}$——与现有 scalar fractional 条件方差（`src/predict.jl:112` 的卷积公式）**同族同形**（§9 检查清单的极限对照点；锚定项差异见 §5.4）。

### 2.6 协变性

- **资产置换 $\Pi$（SPEC §58）**：$R$ 域内资产重排时 $E_{R}\to E_{R}\Pi^\top$（gauge 随置换同步），$\varepsilon^{(R)}$ 的 mode 坐标按正交变换协变；asset 空间对象 $V^{\text{asset}}=E_R V_t E_R^\top$ 满足 $V^{\text{asset}}\to\Pi V^{\text{asset}}\Pi^\top$，预测分布与 Kelly 权重协变 $\Pi^\top w$。任何依赖列编号的构造（如固定「第一个资产」角色）都是 bug。
- **gauge 旋转 $Q\to QR$（SPEC §59）**：$V_\perp\to R^\top V_\perp R$、$c_t\to R^\top c_t$；asset 空间对象与全部可观测断言（预测分布、权重）**不变**。逐 coordinate 归一（破坏旋转协变）禁止——沿用 $s_\perp$ 教训（SPEC §15）。
- **dummy / IPO（SPEC §60、§11）**：全 NaN 资产不进 active ⟹ 不进 $R$ ⟹ $V_t$ 逐字节不变。新资产进 $R$ 后：历史行需覆盖它才进 $J_t$（D-055）；覆盖不足时按 D-056 处理（§6.3）。

---

## 3. $d$ 的先验与 quasi-posterior（D-047/D-57/D-58）

### 3.1 先验

$$
\boxed{\ d\sim\mathrm{Uniform}(0,1)\ }
$$

$d=1$ 可作为闭区间端点极限包含（$k_1\equiv1$ 良定义）；$d=0$ 不在支撑内。**现有 `DGRID_V1`（`src/predict.jl:10`）不再是理论对象**（D-047 原文）——它只能作为数值 quadrature 节点；任何非均匀网格必须携带 cell mass（§3.3）。

### 3.2 quasi-likelihood $\ell_d$ 的 mode-space 精确形式（D-57）

对每个 $d\in(0,1]$，**Gaussian covariance quasi-likelihood**：

$$
\boxed{\;
\ell_d
=-\frac12\sum_{s\in L}
\Big[\ \log\det{}^+ V_{s-1}(d)\ +\ \varepsilon_s^{(R)\top}V_{s-1}(d)^+\,\varepsilon_s^{(R)}\ \Big]
\;}
$$

逐项语义：

- **求和行集 $L$**：$L=\{\,s : u_s\le t,\ O_s\supseteq R_t,\ V_{s-1}(d)\ \text{可定义}\,\}$——joint 合法行中，其标准化因子 $V_{s-1}$ 已可定义者。$V_{s-1}$ 可定义 ⟺ $\{\,s' : u_{s'}<u_s,\ O_{s'}\supseteq R_t\,\}\neq\varnothing$（存在至少一行更早的 joint 合法历史）。**这就是 D-049「有效起点由数学可定义性决定」的精确形式**：第一个 joint 合法行不进 $\ell$（其 $V$ 无定义），第二个起进入；无 burn、无窗口、无补零。
- **$V_{s-1}(d)$**：决策时刻 $s-1$（即目标日 $u_s$ 前一日）的 memory，行集 $J_{u_s-1}=\{s' : u_{s'}\le u_s-1,\ O_{s'}\supseteq R_t\}=\{s':u_{s'}<u_s,\ O_{s'}\supseteq R_t\}$，lag $\tau=u_s-u_{s'}$。
- **$\log\det^+$ 与 $V^+$**：Moore-Penrose 语义（§4）。$\log\det^+V=\sum_{\lambda_i>0}\log\lambda_i$（正特征之和；null 方向不计入行列式）；二次型中 $\varepsilon$ 的 null 分量被 $V^+$ 零化——**未识别方向不评分**（该方向的残差分量对 $\ell_d$ 无贡献，这是 D-57「只在有效 support 上计算」的执行）。
- **「row 的有效 support」的读法（本钉法）**：mode 坐标下（§1.3 命题），资产观测的缺失污染全部 mode 坐标，故「逐行子 support 的二次型」在 mode 坐标下**无定义**；唯一良定义读法是行集层面的（joint 合法行 = 全 $R$ 坐标覆盖，support 即 $V$ 的正 support）。该读法已由 Manager 裁决采纳（裁决 C，2026-10-09：**单一行集概念**——pool/$V$ 统计/$\ell$ 求和三行集同一定义；§11 T2 终态）；被拒绝的逐行子 support 变体（R2）在 mode 坐标下数学无定义，不再列为开放张力。

**quasi（非完整 likelihood）的声明（D-57 原文语义）**：经验 shape 非高斯时，Gaussian covariance likelihood 不再是完整 likelihood；$\ell_d$ 是 quasi-likelihood，$q$ 是 quasi-posterior，**不是普通 Bayes posterior**。正式名称（D-042）：**quasi-posterior over $d$**。

### 3.3 quasi-posterior 与数值 quadrature

$$
\boxed{\;
q(d\mid\mathcal H_t)\ \propto\ \exp(\ell_d)\cdot\mathbf 1_{(0,1)}(d)
\;}
$$

- **先验质量**：$\mathbf 1_{(0,1)}$ 即 Uniform(0,1) 的密度（D-47）。**网格细化时先验质量不得漂移**（SPEC §64）：数值离散化必须为「节点 + cell mass」形式——节点集 $\{d_g\}$、cell 宽度 $\Delta_g$（覆盖 $(0,1]$，$\sum_g\Delta_g=1$），离散权重
$$
q_g\ \propto\ \exp\big(\ell(d_g)\big)\,\Delta_g ,
$$
  与现有 `log_weights=ll.+log.(DELTA_D_V1)`（`src/predict.jl:94`）的测度语义同构，但节点集是**数值 quadrature 的选择**（收敛性由 cell 细化证书，SPEC §64），不是理论对象。adaptive quadrature / 收敛判据属数值层，须带 refinement 证书、不收敛 fail-loudly（SPEC §56）；判据不得涉及回测收益（SPEC §95）。
- **$\eta=1$（D-58）**：不引入 generalized-Bayes temperature / learning-rate。$\eta$ 固定为 1，不允许根据回测调整；若未来理论要求 calibration temperature，必须另立理论版本。

### 3.4 propriety

$q(d\mid\mathcal H)$ 在 $(0,1]$ 上的可积性：每项 $-\frac12[\log\det^+ + \text{quad}]$ 中 $\log\det^+$ 与二次型均非负（$\lambda_i>0$ 有限、$\varepsilon$ 有限），故 $\ell_d\le0$、$\exp(\ell_d)\le1$，在有限测度域 $(0,1]$ 上**恒可积**——propriety 自动成立，前提是输入有限性（§1.2 契约第 5 条）与 $L\neq\varnothing$。$L=\varnothing$（无任何行进入 $\ell$）时 $\ell_d\equiv0$、$q$ 退化为先验均匀——但此时 $V_t$ 同样不可定义（$J_t$ 至多含一行），innovation 层整体 fail-loudly（§2.3）。**propriety gate 的实现形态**：输入有限性断言 + $|J_t|\ge1$ 断言 + $|L|\ge1$ 断言（后者为 $\ell$ 有信息的必要条件；违反即 error，不得静默返回均匀先验冒充有信息的 quasi-posterior）。

---

## 4. PSD / support 协议（D-050/D-051）

### 4.1 Moore-Penrose inverse square root：谱定义

设 $V=U\Lambda U^\top$ 为谱分解（$U$ 正交，$\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_r,0,\dots,0)$，$\lambda_1\ge\cdots\ge\lambda_r>0$，$r=\operatorname{rank}V$）。定义：

$$
V^{1/2} := U\,\operatorname{diag}\big(\sqrt{\lambda_1},\dots,\sqrt{\lambda_r},0,\dots,0\big)\,U^\top,
\qquad
V^{+1/2} := U\,\operatorname{diag}\big(\lambda_1^{-1/2},\dots,\lambda_r^{-1/2},0,\dots,0\big)\,U^\top .
$$

- $V^{1/2}$ 是**唯一对称 PSD 主平方根**（谱定理；简并特征子空间内的任意正交旋转被 $\sqrt\Lambda$ 常数块吸收、特征向量符号翻转同理被吸收）——与现有 `principal_sqrt_root` 的唯一性论证同源（`src/numerics.jl:438-460`，Manager 2026-10-07 裁决：同 $\Sigma$ 在同 seed 有限场景下逐点到 roundoff 一致，不掉小正特征值、不降 rank）。**innovation 层的因子约定沿用该裁决**：$V^{1/2}$ 与 $V^{+1/2}$ 一律取对称主根族。
- $V^{+1/2}$ 是 $V^{1/2}$ 的 Moore-Penrose 逆（对称、PSD）。恒等式：
$$
V^{+1/2}\,V\,V^{+1/2}=P_s,\qquad
V^{1/2}V^{+1/2}=V^{+1/2}V^{1/2}=P_s,\qquad
P_s=U_rU_r^\top\ (\text{support 投影}).
$$

### 4.2 null support 语义（不逆、不注噪、不造风险）

- **零特征方向不逆**：$V^{+1/2}$ 在 null 方向为零算子（不是 $\delta^{-1/2}$）。
- **不注入随机噪声**：null 方向不补伪随机分量。
- **保留为 null support**：该方向 variance 为 0 / 未识别（D-050）；其下游语义见 §5.3。
- **禁止 $V+\delta I$**：任何以「可逆性」为由对 $V_t$ 加 $\delta I$ 的做法都是**偷偷创造理论风险**（把未识别方向的 variance 从 0 篡改为 $\delta$）——D-050/D-051 明文禁止。INNOVATION_LAW.md §3.3/§6.2 的「正则化候选」被推翻（§10 差异表）。
- **与 `EB_COVARIANCE_FLOOR` 的语义区分**（沿用 SPEC §25 裁决）：response 层 EB 的 floor 是数值保护；innovation 层**不存在**理论 floor。二者不得混同。

### 4.3 数值 eigen floor：仅浮点分类，必须 floor→0 refinement

浮点谱分解需要区分「真零特征」与「数值小特征」：分类阈值 $\delta_{\text{cls}}>0$（如 $\varepsilon_{\text{mach}}\cdot\|V\|$ 量级）**只用于浮点分类**（决定哪些 $\lambda_i$ 进入 $V^{1/2}$/$V^{+1/2}$ 的非零块），**不进入数学对象**（$V_t$ 的定义无 $\delta$）。**必须**做 floor→0 refinement 证书：$\delta_{\text{cls}}$ 减半序列下，标准化行 $z$、预测矩、（下游）权重收敛（SPEC §25 的既有先例：floor refinement 时目标权重收敛）。不收敛即 fail-loudly。**禁止**把 $\delta_{\text{cls}}$ 当作理论参数或用回测选择其值。

---

## 5. standardized shape 协议（D-052/D-053/D-059）

### 5.1 标准化与反标准化

**标准化（历史形状，per-$d$）**：对每个 $d\in(0,1]$、每个 $s\in L_d$（$\ell$ 行集；$z$ 与 $\ell$ 共用行集，因二者都需要 $V_{s-1}$ 可定义）：

$$
\boxed{\ z_s(d)=V_{s-1}(d)^{+1/2}\,\varepsilon_s^{(R)}\ \in\mathbb R^{N_R}\ }
$$

**反标准化（未来抽样）**：决策日 $t$，先抽 $d$（从 $q(d\mid\mathcal H_t)$），再从该 $d$ 的 pool 抽行 $s'$，然后

$$
\boxed{\ \varepsilon_{t+1}=V_t(d)^{1/2}\,z_{s'}(d)\ \in\mathbb R^{N_R}\ }
$$

- **per-$d$ pool**：$z_s(d)$ 依赖 $d$（标准化因子随 $d$ 变），故 shape pool 是逐 $d$ 的：$\mathcal Z(d)=\{z_s(d) : s\in L\}$。抽样顺序**先 $d$ 后 $z$**（裁决书 §19 八步链条第 2→3 步的顺序；与现有实现 `d` 先抽样、后 `scale` 的顺序同位，`src/predict.jl:613-615`）。
- **因子约定（「共用同一次同一个分解」的精确语义）**：标准化用 $V_{s-1}^{+1/2}$、反标准化用 $V_t^{1/2}$——**不同矩阵、同一约定**（对称 PSD 主根族，§4.1）。「禁止用两次独立分解的因子相乘」指：对**同一个** $V$ 不得用两个不同约定的因子（如 Cholesky 因子与特征根因子）相乘冒充 $V$ 的函数；主根族在谱定理唯一性下天然免除该歧义。INNOVATION_LAW.md §3.1 的「同一个正则化 $\delta$」表述随正则化候选一并废除（§10）。
- **随机流纪律**：$d$ 抽样、行抽样、response 层 draw 使用独立随机流位置，保持确定性契约（SPEC §36；`src/predict.jl:533-545` 的 rng 顺序契约精神沿用）。**μ 通道的抽样语义已由 RP 侧同步裁决（裁决 F1：response draw 为 matrix-t predictive 抽样，占用独立流位置）**——本条随机流声明经 Manager 裁决 F 确认有效（2026-10-09）。

### 5.2 经验测度：semiparametric empirical predictive 的正式声明（D-053）

$z$ 的 predictive shape 使用历史 OOF standardized rows 的**经验测度**：

$$
\mathcal Z(d)=\big\{\,z_s(d)\ :\ s\in L\,\},\qquad s'\sim\mathrm{Uniform}(\mathcal Z(d)) .
$$

**正式声明**：这是 **semiparametric empirical predictive**——明确**不是**声称已知真实 tail distribution。其规范地位与优点（D-053 原文）：

- 保留 **tail**（尾部形状不经 Gaussian 化压缩）；
- 保留 **skew**（偏度）；
- 保留 **cross-mode shock shape**（联合向量的横截面冲击形状——行级联合抽样使 $z$ 的分量间经验依赖完整保留，这正是 D-054 禁止 cell stitching 所守护的对象）；
- **不新增** Student-t 自由度等手工参数（参数化候选被 D-053 否决；INNOVATION_LAW.md §6.9 的「参数化分布」分支关闭）。

顶层名称（D-042）：整体 predictive law 称 **modular posterior predictive**（或「Bayesian response posterior + cross-fitted semiparametric innovation predictive」）；在 innovation 参数化之前**禁止**称 fully Bayesian generative posterior predictive。

### 5.3 null 分量与 $V_t$ support 的同步性

$z_s$ 在 $V_{s-1}$ 的 null 方向分量为 $0$（$V^{+1/2}$ 零化）。**同步性论证**：对固定 $d$，$J_{u_s-1}\subseteq J_t$（行集随时间单调不减），故 $\operatorname{span}\{V_{s-1}\}\subseteq\operatorname{span}\{V_t\}$（§2.3 秩引理），即历史 null 方向在 $V_t$ 下至多仍是 null 或已获证据。两种情形：

1. **方向在 $V_t$ 仍 null**：$z$ 的该分量恒 $0$，$V_t^{1/2}$ 的该方向也为 $0$ 算子——预测创新在该方向恒 $0$，与 D-050「未识别 support 的 variance 为 0」一致。
2. **方向在 $V_t$ 获得证据（早期行 null、后期行非 null）**：后期行的 $z$ 在该方向非零，pool 含非零样本；预测 $\varepsilon=V_t^{1/2}z$ 的该方向由非零 $z$ 样本与 $V_t^{1/2}$ 的非零行共同生成。

唯一「恒零」情形是**全部历史**在该方向无证据——此时 $V_t$ 本身 null（同一证据集），自洽。结论：协议不存在「历史未识别而 $V_t$ 有证据但 shape 恒零」的不自洽态。

### 5.4 绝对 scale：$V_t$ 自身就是绝对 scale，$v_{\text{bootstrap}}$ 删除（D-059）

**量纲论证**：$z$ 是标准化（无量纲）形状；$V_t$ 携带收益平方量纲（残差二阶矩）；$\varepsilon_{t+1}=V_t^{1/2}z$ 的量纲由 $V_t^{1/2}$ 显式恢复。条件二阶矩：

$$
\mathbb E\big[\varepsilon_{t+1}\varepsilon_{t+1}^\top\ \big|\ d,\ s'\big]=V_t^{1/2}\,z_{s'}z_{s'}^\top V_t^{1/2},
\qquad
\mathbb E\big[\,\cdot\,\big|\ d\big]\approx V_t^{1/2}\,M_z(d)\,V_t^{1/2},
$$

其中 $M_z(d)=\frac1{|\mathcal Z(d)|}\sum_s z_s z_s^\top$ 是 pool 的标准化二阶矩（$\approx$ 而非 $=$：$V_{s-1}$ 随 $s$ 漂移，$M_z$ 不恒为 $P_s$；**协议不假设** $\mathbb E[zz^\top]=I$——绝对量纲不依赖该假设，全部由 $V_t^{1/2}$ 携带）。

**删除项**：现有 scalar 通道的

$$
\text{scale}_d=\sqrt{v_{T+1}(d)/v_{\text{bootstrap}}}
\qquad(\texttt{src/predict.jl:615},\ v_{\text{bootstrap}}=\texttt{var}(e^{macro}),\ \texttt{src/predict.jl:499})
$$

**整体删除**（D-059 原文：删除 `v_bootstrap` scalar anchor）。这不是丢失绝对尺度——恰恰相反，$V_t(d)$ 自身就是绝对 scale：vector law 的二阶矩直接携带绝对量纲，无需「预测方差 / 历史方差」的比率锚定。删除的连带项：`V1Model` 的 `v_bootstrap` 字段语义、场景公式中的 `residual*scale` 通道（`src/predict.jl:620`）——由 $\varepsilon_{t+1}=V_t^{1/2}z$ 整体取代。**极限对照注意**：与现有 scalar 实现对照时（§9 检查 6），$V$ 的 $(1,1)$ 元（macro 方差）与现有 $v_t(d)$ 的 kernel 卷积部分同族可比，但锚定结构（除以 $v_{\text{bootstrap}}$）在新 law 中不存在——对照须分离「kernel 部分一致」与「锚定部分已删除」两个断言，不得混同为一个数值相等。

---

## 6. joint row 规则与 ragged 语义（D-054/D-055/D-056）

### 6.1 禁止 own-row cell stitching（D-054）

**规范**：一个 scenario innovation 必须来自**一个联合合法 shape row**——$\varepsilon_{t+1}=V_t^{1/2}z_{s'}$ 中 $z_{s'}$ 是单一历史行 $s'$ 的标准化向量；**不得**把不同日期的 asset cells 拼成一个「历史向量」。理由：(a) 裁决书 D-054 原文——拼接破坏联合残差向量的横截面依赖；(b) §1.3 命题——mode 坐标下拼接结构性无定义。现有实现的 own-row fallback（`src/predict.jl:596-602`；`src/residual_oracle.jl:264-274` 的 own rows）在新 law 中**无对应通道**。

### 6.2 shape pool 的行集与兼容判定（D-055）

$$
\boxed{\ \text{pool}(d)=\big\{\,z_s(d)\ :\ s\in L\,\},\qquad
L=\{\,s\le t : O_s\supseteq R_t\ \text{且}\ V_{s-1}\ \text{可定义}\,\}\ }
$$

**默认严格规则**（D-055 原文）：被用于联合 scenario 的历史 residual row 必须覆盖当前需要生成风险的**全部 risky assets**（$R_t=$ free ∪ locked）。不满足者不进入 joint empirical shape pool。注意本设计下 **pool 行集 = $V$ 统计行集 = $\ell$ 求和行集**（三者同一：都需要「覆盖 $R_t$」+ 因果性；$V$ 统计额外要求分母非零即自身可定义，$z$/$\ell$ 要求 $V_{s-1}$ 可定义）——单一行集概念，单一 mask 判定，无三处不一致的实现面。裁决书 D-055 的「默认严格规则」措辞暗示可能存在非默认的宽松变体：本文档只规范默认严格规则；任何宽松变体（若未来提出）须作为理论变更单独审查（D-055 原文语义），不得由实现顺路引入。

### 6.3 无足够 joint rows 时的处理（D-056）

**优先级序列（裁决书原文顺序）**：

1. **fail loudly**——fail 优先于任何拼接/补零/降级（D-056 第一词；SPEC §56）；
2. **该资产不进入 free risky set**——剔除导致覆盖不足的 free 资产，$R_t$ 收缩，行集 $J_t$/pool 恢复（§1.3 的 $R$ 域设计正是此出路的载体）；
3. **已持有则 locked**——locked 资产保留在 $R_t$（其 scenario wealth contribution 必须保留，D-017：绝不当 cash）；
4. **资金可以留在 cash**——这正是 cash 被纳入 Kelly feasible set（D-068/D-070）的重要原因之一：exact Kelly 在无足够 edge/无足够联合历史时自己选择 cash。

**判定次序（实现者规范，已裁决采纳——Manager 裁决 C，2026-10-09）**：给定初始 $R_t^{(0)}=$ free ∪ locked：(a) 若 pool 覆盖不足由 free 资产引起，逐个剔除该 free 资产（不进 free risky set），重算行集；(b) 若剔除后仍不足（覆盖不足由 locked 资产引起，或 free 剔尽仍不足）→ **fail loudly**，由回测驱动器捕获该错误并记录诊断（当日资产、覆盖缺口、行集大小）；**当日维持既有持仓**（不重排权重）；**不拼残差、不 zero-fill**（§11 T4 终态：已裁决采纳）。

### 6.4 locked 资产与 Kelly base 的衔接（接口，非本文档所有）

Kelly 层的 locked wealth（`locked_wealth`，`src/kelly.jl:137-148`；SPEC §38.1：locked 风险绝不当 cash，缺预测律即 error）需要 locked 列的 scenario gross。本协议下 locked 资产 ∈ $R_t$，其 gross 由八步链条（§8.2）生成——即 locked 资产的 scenario wealth 来自**同一个联合 innovation 通道**（无独立边缘通道）。若 locked 资产导致 pool 覆盖不足：**fail loudly + 驱动器捕获记录诊断 + 当日维持持仓**（Manager 裁决 C，2026-10-09；§6.3 判定次序 (b)）；替代 wealth 构造（如边缘化处理）被裁决排除在默认规范外——不得由实现自行发明，未来若提出须作为理论变更单独审查（§6.2 末段的同一纪律）。

---

## 7. 因果性：严格论证

**定理（$V_t$ 的因果性）。** $V_t(d)$ 的行集 $J_t\subseteq\{s : u_s\le t\}$：每个被加项 $\varepsilon_{s}^{(R)}$ 的目标日 $u_s\le t$，即其评价的 return 在决策日 $t$ 收盘**已实现**。预测日 $t+1$ 的 return（$r_{t+1}$）不出现在任何被加项中。∎（构造性：$J_t$ 定义含 $u_s\le t$；lag 约定 $\tau\ge1$。）

**定理（$z_s$ 的决策时刻可算性）。** $z_s=V_{s-1}^{+1/2}\varepsilon_s$：$V_{s-1}$ 的行集 $\{s' : u_{s'}\le u_s-1\}$（严格早于行 $s$ 的目标日），$\varepsilon_s$ 自身在 $u_s\le t$ 已实现。故对一切 $s\in L$（$u_s\le t$），$z_s(d)$ 在决策日 $t$ 是**已实现量的确定函数**——可算。∎

**定理（预测创新的严格因果）。** $\varepsilon_{t+1}=V_t^{1/2}z_{s'}(d)$：$V_t$ 因果（上）；$z_{s'}$ 因果（上）；$d$ 抽自 $q(d\mid\mathcal H_t)$，其 $\ell_d$ 的行集 $u_s\le t$——全部输入 $\subseteq\mathcal H_t$。∎

**禁止项**：(a) 把 $\varepsilon_{t+1}$ 自身或任何 $u>t$ 的行放进 $V_t$；(b) 用未来行参与标准化集合的构造；(c) 用决策日之后的信息（含「事后已知的 pool 大小」以外的任何未来统计）选择 $d$ 的 quadrature 节点。现有实现的因果卷积（`variance=output[t-1]/cs[t-1]`，`src/predict.jl:112`；行抽样只在历史行内，`src/predict.jl:591`）满足同构性质，作为对照基准。

---

## 8. 与 response posterior 的接口与总 predictive law（D-060/D-040/D-041/D-042）

### 8.1 两层职责：epistemic vs aleatoric，不得重复计算（D-060/D-040/D-041）

| 层 | 职责（D-040/D-041 原文） | 输出 | 禁止 |
|---|---|---|---|
| **response posterior** | epistemic uncertainty：给定有限历史，对条件均值算子 $G$（及 $b_0$、超参）有多不确定 | $P(\mu_{t+1}\mid\mathcal H_t)$ 或等价参数后验（GATE0_RESPONSE_POSTERIOR.md 所有） | 不得把 working-likelihood covariance $\Sigma_R$ 再作为额外 future residual shock 重复加一次 |
| **innovation law**（本文档） | out-of-fit aleatoric predictive law：OOF 意义下真实下一日残差可能是什么样 | $P(\varepsilon_{t+1}\mid\mathcal H_t)$：$q(d\mid\mathcal H)$ × $V_t(d)$ × 经验 $z$ 测度 | 不得混入 response 参数的不确定性 |

**不重复计算（D-060 原文）**：scenario generator 中 response posterior draw 只负责均值不确定性、innovation draw 只负责残差不确定性。$\Sigma_R$ 在 response 模块里是 working-likelihood covariance / posterior scaling nuisance；**真正未来 aleatoric risk 由 vector innovation law（$V_t$）唯一提供**。两层各出现一次、互不重叠。

**方差分解（报告义务，裁决书 §30）**：

$$
\operatorname{Var}(r\mid\mathcal H_t)=\underbrace{\operatorname{Var}_{\Pi}(\mu)}_{\text{epistemic（response）}}+\underbrace{\mathbb E\big[V_\varepsilon\big]}_{\text{aleatoric（innovation，}=V_t\text{ 的混合）}},
$$

在 mode 坐标成立，经正交变换 $E_{R_t}$ 在 asset 坐标同样成立（协变）。**报告口径（Manager 裁决 A5，2026-10-09）**：规范报告在 asset 空间 $R_t$ 子集上**同域可加**——

$$
\operatorname{Var}(u_R)=E_{R_t}\,\operatorname{Var}(\mu_R)\,E_{R_t}^\top+E_{R_t}\,\mathbb E[V_R]\,E_{R_t}^\top
\qquad\big(u_R\in\mathbb R^{N_R}\ \text{asset 坐标，}\ \mu_R=E_{R_t}^\top\mu_{\text{asset}}[R_t]\big),
$$

active 域全量 epistemic（$\operatorname{Var}(\mu_{\text{asset}})$ 的 $N$ 维全量）为**诊断附加项**，不进入规范分解的主报告。报告要求：两分量分别给出数值；无法计算的部分必须写 **NOT COMPUTED**，不得显示 0（裁决书 §30 原文）。

### 8.2 总 predictive law：八步链条（裁决书 §19）

**μ 契约（Manager 裁决 A4，2026-10-09）**：response 层规范输出 **asset 空间** $\mu_{\text{asset}}=E_{\mathrm{active}}\cdot\mu\in\mathbb R^N$（$E_{\mathrm{active}}$ 为 active 域 mode→asset 正交变换）；scenario 合成时 innovation/Kelly 层取 $\mu_R=E_{R_t}^\top\cdot\mu_{\text{asset}}[R_t]\in\mathbb R^{N_R}$（$R$ 域 mode 坐标）。

对每个 scenario $s$（$d_s$ 抽自 $q(d\mid\mathcal H_t)$；行 $s'$ 抽自 $\mathrm{pool}(d_s)$；$\mu^{(s)}$ 抽/积分自 response posterior——response 层职责，裁决 F1：matrix-t predictive 抽样）：

1. **均值**：$\mu_{\text{asset}}^{(s)}\in\mathbb R^N$（response 层输出，asset 空间）；本层取 $\mu_R^{(s)}=E_{R_t}^\top\,\mu_{\text{asset}}^{(s)}[R_t]\in\mathbb R^{N_R}$。
2. **记忆指数**：$d_s\sim q(\cdot\mid\mathcal H_t)$。
3. **形状**：$z^{(s)}=z_{s'}(d_s)\in\mathcal Z(d_s)$（compatible standardized residual empirical measure 的一个样本）。
4. **创新**：$\varepsilon_{t+1}^{(s)}=V_t(d_s)^{1/2}\,z^{(s)}$。
5. **mode 场**：$y_R^{(s)}=\mu_R^{(s)}+\varepsilon_{t+1}^{(s)}\in\mathbb R^{N_R}$（$R$ 域 mode 坐标）。
6. **映射回资产**：$u_R^{(s)}=E_{R_t}\,y_R^{(s)}\in\mathbb R^{N_R}$（asset normalized return field，即 $r/s_1$ 的标准化量）。
7. **恢复 log return**：$\rho^{(s)}=s_1[R_t]\odot u_R^{(s)}$（asset log return；$s_1$ 为 1-day fractal ruler，`src/predict.jl:244`）。
8. **gross**：$R^{(s)}=\exp(\rho^{(s)})$（gross return；正有限性由 Kelly 层 `kelly_inputs` 校验，`src/kelly.jl:6-15`）。

**Kelly scenario 列集**：$R_t$——free 列进入优化变量、locked 列进入 base_s（`locked_wealth`，§6.4）；active 域中 $\notin R_t$ 的资产不生成 scenario 列（其权重恒 0 / 不持仓）。

**积分语义（与抽样语义的对偶）**：八步给出 scenario 抽样形式；等价的积分形式（$\mu$ 与 $d$ 的积分而非抽样）在数值收敛证书（D-061～D-067，docs/NUMERICAL_INTEGRATION_SPEC.md 的证书体系）下与抽样形式逼近同一 predictive law。数值 backend（IID vs 嵌套 quadrature）不改变本定律（SPEC §37）。

### 8.3 与 Kelly 的衔接（消费侧，非本文档所有）

$R^{(s)}$ 进入 Kelly 的 exact log-growth 目标（$w\ge0$、$\sum_iw_i\le1$、cash 为 numeraire，D-068/D-070；locked base 语义 D-017/§38.1）。innovation 层的职责到 gross 为止；feasible set、证书、concentration 诊断归 Kelly 层（裁决书 §21/§23）。

---

## 9. 实现者检查清单

实现 vector innovation reference 时，以下测试/断言是**规范要求**（形态：tiny analytic / synthetic world / 性质测试；全部判据不依赖回测收益，SPEC §95/§89）：

1. **propriety / support 断言**
   - 输入有限性：$\varepsilon$ 含 NaN/Inf → error（fail-loud）。
   - $|J_t|=0$ → error（$V_t$ 不可定义）。
   - $|L|=0$ → error（$\ell$ 无信息，不得静默返回均匀先验）。
   - rank deficient fixture（构造 $|J_t|<N_R$ 或行向量共线）：null 方向 variance $=0$；$z$ 的 null 分量 $=0$；预测创新在 null 方向 $=0$；**无噪声注入、无 $\delta I$**。
   - 秩引理验证：同一行集、不同 $d$ → $\operatorname{rank}V_t(d)$ 相同。
2. **floor→0 refinement**
   - 浮点分类阈值 $\delta_{\text{cls}}$ 减半序列 → $z$、预测矩、（下游）权重收敛；不收敛 fail-loud。
   - 断言 $V_t$ 的数学定义不含 $\delta$（$\delta$ 只出现在分类代码路径）。
3. **joint row 兼容性的 mask 判定**
   - $R_t$ 变化（剔除 free 资产）→ $J_t$/pool 行集相应恢复（D-056 出路可兑现）。
   - own-row stitching 反例：不同日期 cells 拼成的向量不在任何 pool（且 mode 坐标下无定义）。
   - IPO 资产：进入 $R_t$ 前不影响 $V_t$（dummy invariance）；进入后历史行覆盖不足 → D-056 序列。
   - 空观察行（$O_s=\varnothing$）不进任何统计。
4. **因果性断言**
   - $V_t$ 的全部行 $u_s\le t$；$z_s$ 的 $V_{s-1}$ 行 $u_{s'}<u_s$。
   - 构造未来行（$u>t$）注入 → 断言拒绝（构造性反例测试）。
   - 同 seed 重放：$V_t$、pool、$z$、$\varepsilon_{t+1}$ 逐点可复现（SPEC §36）。
5. **极限对照（与现有 scalar 实现）**
   - $N_R=1$：$V_t(d)$ 的 $(1,1)$ 元 = 标量 $\frac{\sum k_d\varepsilon^2}{\sum k_d}$；与现有 `causal_fractional_posterior` 的 kernel 卷积部分（`src/predict.jl:112`）同族对照；**锚定差异分离报告**（$\sqrt{v/v_{\text{bootstrap}}}$ 通道已删除，§5.4——不得把「kernel 部分一致」与「整体数值相等」混同）。
   - $d=1$：$V_t(1)$ = joint 合法行等权二阶矩（解析可验）。
   - $d\to0^+$：近期行主导（权重比解析可验）。
   - 完整 panel（无 ragged）：$V_t$ 的分母 $=\sum_\tau k_d$（裁决书字面形式成立）。
6. **协变性**
   - 资产置换：asset 空间 $V^{\text{asset}}\to\Pi V\Pi^\top$；预测分布与权重协变（SPEC §58）。
   - gauge 旋转 $Q\to QR$：asset 空间对象与全部可观测断言不变（SPEC §59）。
   - dummy 资产：$V_t$ 逐字节不变（SPEC §60）。
7. **shape 语义**
   - $z$ pool 的二阶矩 $M_z(d)$ 报告（诊断量；非断言 $=I$）。
   - tail/skew 保留：$z$ 的分布 = 历史标准化 shape 的经验测度（非参数化拟合）；断言无 Student-t 等新参数出现。
   - cross-mode shock：联合行抽样的分量间经验相关 = 历史相关（无拼接破坏）。
8. **与 response 层的接口**
   - 接口锚文字一致性：两文档的 ε 契约逐字一致（§0）。
   - 不重复计算：scenario 的 aleatoric 通道唯一来自 $V_t$；$\Sigma_R$ 不作为额外 shock 出现（D-060）。
   - 方差分解可报告（§8.1）；不可计算项标 NOT COMPUTED。

---

## 10. 与 docs/INNOVATION_LAW.md 的条款级差异表

INNOVATION_LAW.md 是「定义草案（Deliverable 5）」；本文档是其按裁决书的升级版。分类：**保留**（语义不变或仅形式升级）、**升级**（裁决书已裁，草案的开放项收口）、**推翻**（草案的候选方向被裁决书否决）。

| # | INNOVATION_LAW.md 条款 | 处置 | 说明 |
|---|---|---|---|
| 1 | §1 现状映射（四步链条、A1–A7 假设、差距表） | **保留** | 作为历史对照与差距证据（本文档 §12 引用其 file:line） |
| 2 | §2.1 $V_t(d)$ 定义（asset 坐标主定义） | **升级** | 主定义改为 **mode 坐标**（D-045 字面；委任书指定）；asset 坐标作为正交等价表示（§1.1）。公式本体（kernel 加权二阶矩）不变 |
| 3 | §2.2 common/relative gauge 分解 | **保留** | 作为 mode 主定义的 asset 投影性质（§2.4 的 $V_m,c_t,V_\perp$ 分块） |
| 4 | §2.2「若标准化需要可逆，必须显式加正则化（6.2），不得静默丢弃零特征方向」 | **推翻** | D-050/D-051 裁定：不加 $\delta I$、不造理论风险；Moore-Penrose inverse square root on observed positive support（§4）；正则化候选废除 |
| 5 | §2.2 permutation/gauge/alive/扩维要求 | **保留** | §2.6 形式化（含 $R$ 域实例化下的重述） |
| 6 | §3.1 standardized 协议（z、反标准化、主根、随机流） | **升级** | per-$d$ pool 显式化（$z_s(d)$ 依赖 $d$）；「同一正则化 $\delta$」表述废除，改为「同一因子约定（对称主根族）」（§5.1）；主根沿用 `principal_sqrt_root` 裁决（`src/numerics.jl:438-460`） |
| 7 | §3.1「（或参数化分布的一个 draw，见 6.9）」 | **推翻** | D-053 裁定经验测度（semiparametric empirical predictive）；参数化候选关闭（§5.2） |
| 8 | §3.2 因果性论证 | **保留** | 升级为 §7 定理形式（$V_t$/$z_s$/预测创新三条，构造性证明） |
| 9 | §3.3 support 条件（「$V_t+\delta_tI$ 或特征值 floor」候选） | **推翻** | 同 #4；「与 EB_COVARIANCE_FLOOR 区分」**保留**（§4.2） |
| 10 | §3.4 ragged 缺失 cell 的 own-row fallback 语义（含「own-row 标准化版本」候选） | **推翻** | D-054 禁止 own-row cell stitching；D-055/D-056 的 joint row 规则与 cash 出路取代（§6）；mode 混合性给出数学必然性论证（§1.3） |
| 11 | §4 禁止项（scalar 过渡辩护 / factor bag / 手造因子） | **保留** | 三条与裁决书一致；本文档 §1.3/§2.4 进一步给出 mode 层执行 |
| 12 | §5「$d$ 后验保持 $p_d\propto\exp(\ell)\Delta d$」 | **升级** | 连续 quasi-posterior $q(d\mid\mathcal H)\propto\exp(\ell_d)\mathbf 1_{(0,1)}(d)$（D-47/D-57）；$\Delta d$ 降级为数值 quadrature cell mass（§3.3） |
| 13 | §5「$v_{\text{bootstrap}}$ 的角色：vector 版必须保留绝对尺度锚点语义（6.6）」 | **推翻** | D-059：删除锚点；$V_t(d)$ 自身就是绝对 scale（§5.4） |
| 14 | §6.1 kernel 归一与截断 | **部分升级** | 数学定义钉死（$k_d(\tau)=\pi_{\tau-1}$、全部因果历史、正权重无截断、分母归一，§2）；数值 quadrature 的节点/cell mass 细节保留为实现层未决（收敛证书体系所有） |
| 15 | §6.2 PSD floor / 正则化（$\delta\to0$ 收敛判据） | **升级** | 判据吸收进 §4.3 的 floor→0 refinement 证书；「$V+\delta I$ 候选」推翻 |
| 16 | §6.3 标准化窗口与 burn（burn=30） | **推翻** | D-048/D-049：无 rolling window、burn 废除；有效起点 = $V_{s-1}$ 可定义性（§2.2 要点 3、§3.2） |
| 17 | §6.4 ragged 缺失 cell 语义（原样 fallback vs 标准化 fallback） | **推翻** | 同 #10；两候选均关闭，joint row 规则取代 |
| 18 | §6.5 macro/relative 交叉项（保留 $c_t$ vs block-diagonal） | **已裁决** | D-045：天然包含 cross covariance；block-diagonal 候选否决（§2.4） |
| 19 | §6.6 绝对尺度锚点（标量锚/矩阵锚/macro 通道锚） | **推翻** | D-059（同 #13） |
| 20 | §6.7 $d$ 先验显式化 | **已裁决** | D-47：Uniform(0,1)（$d=1$ 端点极限包含）（§3.1） |
| 21 | §6.8 实现表示与预算（低秩/谱/递推、增量层） | **保留** | 实现层未决；reference 先直接 $O(T^2N_R^2)$（裁决书 Step 10/§82）；数学恒等加速属后 Gate（裁决书 §42） |
| 22 | §6.9 标准化 shape 的分布（经验 vs 参数化） | **已裁决** | D-053：经验测度（§5.2） |
| 23 | §6.10 与 response 不确定性的层级关系 | **已裁决** | D-060（§8.1）；方差分解报告义务钉死（§8.1、裁决书 §30） |
| 24 | —（新增） | **新增** | $R$ 域实例化（§1.3）：INNOVATION_LAW.md 无此条款；由 D-055/D-056 的自洽性要求推出 |
| 25 | —（新增） | **新增** | 秩引理（§2.3：rank 与 $d$ 无关）——消除「小 $d$ 免费似然」误读 |
| 26 | —（新增） | **新增** | null 同步性（§5.3）——消除「shape 恒零 vs $V_t$ 有证据」的不自洽疑虑 |

---

## 11. 与裁决书的张力与终审终态（Manager 裁决后闭环）

初稿列出的张力 T1–T7 已由 Manager 终审裁决（2026-10-09；裁决 A/A2/A4/A5/C/D/D2/E/F，全文落盘于 docs/GATE0_MANAGER_ADJUDICATIONS.md）闭环。终态：

| # | 张力（初稿） | 终态 | 裁决 |
|---|---|---|---|
| T1 | 分母的字面（$\sum_{\tau\ge1}k_d(\tau)$）vs ragged 收缩到 $J_t$ | **已裁决采纳**：分子分母同步收缩到 joint 合法行集（§2.2 要点 1） | 裁决 C |
| T2 | D-57「row 的有效 support」读法：行集层面 vs 逐行子 support | **已裁决采纳**：行集层面读法；pool/$V$ 统计/$\ell$ 求和**三行集同一定义**（单一行集概念，§3.2/§6.2） | 裁决 C |
| T3 | $V_t$/$z$/pool 的域实例化：active 域 vs risk 域 $R_t$ | **已裁决采纳（D-045a 澄清条款，2026-10-09）**：实例化在决策日 risk 域 $R_t=$ free ∪ locked 的 mode 坐标 $E_{R_t}$ 上（§1.3 推论 2；裁决 A2 精确路径：从 asset 空间源对象提取 $R$ 子向量） | 裁决 A/A2 |
| T4 | locked 资产覆盖不足时的 wealth 构造 | **已裁决采纳**：fail loudly + 回测驱动器捕获记录诊断 + 当日维持持仓；不拼残差、不 zero-fill；边缘化等替代构造排除在默认规范外（§6.3/§6.4） | 裁决 C |
| T5 | 坐标主从变更（asset→mode）的知会 | **部分升级条款已裁决确认**：mode 为主定义（§1.1）；数值 quadrature 细节仍属实现层 | 裁决 C |
| T6 | 接口契约行 mask 定义与 RP 的一致性 | **已对齐**：$O_s=\{j : r[u_s,j]\ \text{finite}\}$ 与 RP 逐字一致（§1.2 契约 4）；锚文字统一（§0，裁决 D 版本） | 裁决 D/D2 |
| T7 | 数值 quadrature 节点与 cell mass | **留实现期**：走 D-066 流程（refinement 证书定稿，不以回测选择）；连续对象 $q(d\mid\mathcal H_t)$ 已钉死（§3.3） | 裁决 C |

**遗留（非张力）**：T7 的节点集/收敛判据属数值层实现决定，须带 refinement 证书（SPEC §64/§56；D-066 流程）；除此之外本文档不含未决结构性歧义。

---

## 12. 现有实现对照锚（file:line 索引）

实现 reference 时用于对照的现有 scalar 通道位置（**对照对象，不是规范**）：

| 对象 | 位置 | 对照点 |
|---|---|---|
| fractional kernel 递推 | `src/predict.jl:12` | $\pi_k$ 递推（同一 family，D-046） |
| scalar causal 条件方差 | `src/predict.jl:112` | $v_t(d)$ 卷积（$V_t$ 的 $(1,1)$ 元同族极限，§9 检查 5） |
| scalar quasi-likelihood | `src/predict.jl:101-119` | $\ell_d$ 的标量形态（mode 版的 $N_R=1$ 极限） |
| $d$ 后验与 cell mass | `src/predict.jl:94-96`、`10`、`14` | $q_g\propto\exp(\ell)\Delta_g$ 测度语义（升级为连续对象，§3.3） |
| burn=30 | `src/predict.jl:76` | 被废除的理论参数（D-049；§2.2 要点 3） |
| $v_{\text{bootstrap}}$ 与 scale | `src/predict.jl:499`、`615` | 被删除的锚定通道（D-059；§5.4） |
| own-row fallback | `src/predict.jl:596-602`、`src/residual_oracle.jl:264-274` | 被禁止的 cell stitching（D-054；§6.1） |
| 行抽样（共享行） | `src/predict.jl:591`、`616-619` | 行级联合抽样的现存形态（joint row 规则的前身；其 own-row 例外被废除） |
| macro 标量残差序列 | `src/residual_oracle.jl:216-260` | $V_t$ 输入的 scalar 投影对照（mode 残差的 $(1,\cdot)$ 投影） |
| OOF 行契约 | `src/residual_oracle.jl:17-22` | 行身份/目标日/mask 语义（§1.2 契约 3-4 的实例） |
| 主根唯一性裁决 | `src/numerics.jl:438-460` | `principal_sqrt_root` 的谱定理唯一性（§4.1 因子约定沿用） |
| locked wealth | `src/kelly.jl:137-148` | locked 列 scenario gross 的消费侧（§6.4 衔接） |
| scenario 公式（旧） | `src/predict.jl:620-621` | 八步链条的 asset 空间前身（§8.2；scale 通道被 $V^{1/2}z$ 取代） |

---

## 13. 一句话总结

> **innovation 层的规范对象是 risk 域 mode 坐标下的 kernel 加权联合残差二阶矩 $V_t(d)$：统一 kernel、全部因果历史、构造性 PSD、Moore-Penrose support 协议、行级 joint 语义、经验标准化 shape、$V_t$ 自携绝对尺度；$d$ 带 Uniform(0,1) 先验与 Gaussian quasi-likelihood 的 quasi-posterior；与 response posterior 以 §0 接口锚（ε 源对象 = asset 空间 OOF 残差行，含 $b_0$ 扣除；innovation 层按 $R_t$ 消费其 $R$ 子向量经 $E_{R_t}$ mode 变换）锚接，epistemic 与 aleatoric 各出现一次、互不重叠。**

*（本文件为 Gate-0 innovation 层规范推导交付物；未运行任何命令，未修改 src/、test/、README.md、AGENTS.md 或任何既有文档。初稿张力 T1–T7 已由 Manager 终审裁决（2026-10-09，裁决 A/A2/A4/A5/C/D/D2/E/F）闭环，终态见 §11；本修订版为裁决落地版。）*
