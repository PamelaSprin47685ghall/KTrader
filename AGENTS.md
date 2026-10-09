# KTrader / Path Kelly — 最新权威 SPEC

**文档状态：Normative / Source of Truth**  
**适用版本：KTrader 2.0.0 Final（CPU）；数学合同仍为 Path Kelly 1.0，GPU 属于 2.1**
**仓库快照基准：`repomix-output(20261007-113017).xml`**  
**目标读者：第一次接触本项目、没有历史上下文、需要从零恢复全部代码与设计意图的工程师**

发布入口为 README.md、RELEASE.toml 和 RELEASE_CPU_2_0.md。下文历史交付段的
未发布、阻断或旧版本说明保留其当时语境，不覆盖当前 Final 元数据；原失败日志
不得改写。CPU 封版只变更 Project 版本行，已验收的生产代码、测试和依赖锁
保持原字节。以 `bin/verify_release.jl` 核验正式源码，不以修改旧凭据冒充新验收。

---

## 0. 如何使用本 SPEC

这不是 README，也不是“设计想法汇总”。它是本项目的**唯一权威规范**。陌生工程师应该可以只读本文档，在不依赖历史聊天的情况下，恢复：

1. 项目的理论目标与不变量；
2. 数据语义；
3. 数学模型；
4. Bayesian / OOF / innovation / Kelly 的关系；
5. 每个 Julia 模块应负责什么；
6. 当前仓库中的主要类型、常量、函数与调用链；
7. 哪些东西是理论、哪些是统计认识论、哪些只是数值工程；
8. reference / accelerator / incremental backend 的边界；
9. 正确性测试、性质测试、性能测试；
10. 从一个空目录重写本项目时的顺序。

若代码与本 SPEC 冲突，优先级如下：

> **本 SPEC 的明确规范 > 通过的数学性质测试 > 当前实现细节 > 历史注释。**

当前仓库仍处于架构整理期，存在性能诊断脚手架和过重的 accelerator 代码。本文会同时标注：

- **[规范]**：1.0 必须满足；
- **[当前实现]**：最新仓库现在如何做；
- **[过渡]**：当前存在但 1.0 架构整理后应迁移、收缩或删除；
- **[未来]**：2.0 或之后的优化，不得反向污染 1.0 数学定义。

---

# 第一部分：项目使命与不可破坏公理

## 1. 项目一句话定义

KTrader / Path Kelly 是一个：

> **只以价格历史为市场原始信息，直接构造未来收益的 posterior predictive distribution，并通过 exact log-growth / Kelly 决策得到目标持仓权重的因果量化系统。**

核心映射：

\[
\boxed{
\mathcal H_t
\rightarrow
\Pi(\text{unknown law / operator}\mid \mathcal H_t)
\rightarrow
P(r_{t+1}\mid\mathcal H_t)
\rightarrow
w_t^*
}
\]

其中：

\[
\mathcal H_t=\{x_i(s):s\le t,\ i=1,\ldots,N\},\qquad x_i(t)=\log P_i(t).
\]

项目不输出“因子分数”“股票排名”“Top-K”“多因子加权分”。理论最终输出必须是：

\[
\boxed{w_t^*}
\]

即当前信息集下的目标组合权重。

---

## 2. 真实规律不允许写成“时变理论”

[规范]

真实世界的规律写为一个固定映射：

\[
\boxed{
\mathcal F:\mathcal H_t\mapsto P(r_{\rm future}\mid\mathcal H_t)
}
\]

历史路径在变化，但规律本身不是“每天换一套理论”。因此项目统一语言是：

> **规律不时变，输入历史在演化。**

禁止把真实规律本体写成：

\[
\mathcal F_t,
\]

除非那只是代码接口索引、数值近似对象或显式的模型类记号，而不是本体论声明。

Bayesian posterior 会变化：

\[
\Pi(d\mathcal G\mid\mathcal H_t),
\]

这是因为我们对一个固定未知对象 \(\mathcal G\) 的认识在更新，不是因为 \(\mathcal G\) 每天变成另一个真理。

---

## 3. 全历史是理论原语，有限状态不是本体

[规范]

不能无条件假设存在有限充分状态 \(z_t\) 使：

\[
P(\text{future}\mid\mathcal H_t)=P(\text{future}\mid z_t).
\]

本项目的精确理论对象是：

\[
P(r_{t,t+h}\mid\mathcal H_t),
\]

以及条件均值泛函：

\[
m_h[\mathcal H_t]=E[r_{t,t+h}\mid\mathcal H_t].
\]

任何 \(Q/P\) 坐标、有限频带、有限矩阵 \(G\)、FFT、有限 posterior grid 都必须被理解为：

- 有限证据下的统计模型；或
- 无限维对象的可计算表示；或
- 数值离散。

不得把这些近似倒过来宣称成市场“真正只有这些状态”。

---

## 4. 价格历史是原始市场输入

[规范]

策略核心不得依赖：

- 人工行业标签；
- 财务报表因子；
- 新闻 NLP 因子；
- 手工宏观 regime；
- 人工命名的板块轮动表；
- 传统 factor bag。

天然结构必须从价格几何和响应算子中产生。

允许 execution 层使用账户、报价、订单状态，因为 execution 不是 alpha 理论。

---

## 5. 理论输出与执行必须分层

理论层：

\[
\mathcal H_t\mapsto w_t^*.
\]

执行层：

\[
w_t^*\mapsto \text{shares / orders / rebalance actions}.
\]

交易手续费、融资成本、订单撤销、限价单、部分成交等不进入当前理论目标函数。执行层从 `rebalance.mjs` / broker semantics 独立派生。

---

## 6. 经济目标是 log-growth / Kelly

[规范]

精确目标：

\[
\boxed{
w_t^*
=
\arg\max_{w\in\mathcal W_t}
E[\log(1+w^\top r_{t+1})\mid\mathcal H_t]
}
\]

若使用 gross returns \(R=1+r\)，在全投资 long-only simplex 中：

\[
\boxed{
\max_{w\ge0,\ \mathbf1^\top w=1}
E[\log(R^\top w)]
}
\]

scenario approximation：

\[
\max_w \frac1S\sum_{s=1}^S\log(R_s^\top w).
\]

Sharpe ratio 不是优化目标，只可作为事后报告；均值—方差只是小收益局部近似：

\[
U(w)\approx w^\top\mu-\frac\gamma2w^\top\Sigma w.
\]

禁止因 solver 失败退化到另一个经济目标。

---

# 第二部分：三层分离纪律

## 7. 理论 / 认识论 / 数值工程必须分开

所有设计决定都必须归入以下一层：

### 7.1 理论层

即使有无限计算也仍存在。例如：

- 价格历史是信息集；
- 固定路径规律；
- center-of-mass / relative decomposition；
- structural trace neutrality；
- exact log-growth objective；
- posterior predictive 是决策输入。

### 7.2 统计认识论层

无限算力也无法消除有限数据导致的不确定性。例如：

- \(\Pi(G,\Sigma\mid\mathcal H)\)；
- empirical Bayes / hyperprior；
- OOF residual；
- geometry / response posterior uncertainty；
- finite-sample complexity control。

### 7.3 数值工程层

如果无限计算存在则应消失或趋于极限。例如：

- FFT；
- primal / dual matrix identity；
- Cholesky / eigensolver；
- scenario count；
- quadrature grid；
- workspace；
- SIMD；
- CPU threads；
- incremental sufficient statistics；
- GPU backend。

判断问题归属的快速规则：

> **“若无限算力存在，这个对象是否仍必须存在？”**

- 是：理论或统计认识论；
- 否：数值工程。

禁止因为一个数值实现回测 Sharpe 更高，就把它升格为理论。

---

# 第三部分：数据与时间语义

## 8. Daily Bars 数据结构

[当前实现]

```julia
struct Bars
    dates::Vector{Date}
    symbols::Vector{String}
    close::Matrix{Float64}
    adj::Matrix{Float64}
    bar::BitMatrix
end
```

语义：

- `bar[t,j] == true`：资产 j 在日期 t 有真实 bar；
- `close` / `adj`：账户 marking 使用的 carried-forward series；
- 上市前保持 NaN；
- 上市后缺 bar 的 marking price 可 carry forward；
- `signal_prices(b)` 必须恢复理论输入：

```julia
signal_prices(b::Bars) = ifelse.(b.bar, b.adj, NaN)
```

### 8.1 不交易不等于零收益

理论 signal history 中：

\[
P^{signal}_{t,i}=NaN
\]

表示当天没有真实价格观测。

不得将 carried marking price 当成理论 signal price，从而伪造：

\[
r_{t,i}=0.
\]

### 8.2 Raw return

\[
r_{t,i}=x_i(t)-x_i(t-1)
\]

只有两个端点都真实可观测时才是 finite，否则为 NaN。

---

## 9. 日频语义

[规范]

本项目当前理论单位是 daily step。日频 path basis、response target、innovation law 和 backtest 决策均以日为离散步长。

此语义已经确定，不再作为 1.0 的争议项。

---

# 第四部分：多尺度价格几何

## 10. Fractal ruler

每个资产对尺度 \(\tau\) 的 path increment：

\[
\Delta_\tau x_i(t)=x_i(t)-x_i(t-\tau).
\]

经验尺度律：

\[
|\Delta_\tau x_i|\sim C_i\tau^{H_i}.
\]

当前离散尺度：

```julia
const TAUS = 2 .^ (0:8)
# 1,2,4,8,16,32,64,128,256
```

路径 basis temporal bands：

```julia
const BANDS = 2 .^ (1:7)
# 2,4,8,16,32,64,128
const WARMUP = 2 * maximum(BANDS) # 256
```

`ruler` 对每个资产用自身完整可用历史计算 RMS \(\tau\)-increments，再在 \(\log \tau\)-\(\log RMS\) 上拟合线性 scaling law，得到各 TAU 的平滑 ruler。

一天 ruler：

\[
s_{1,i}=s_i(1).
\]

它是价格尺度正规化，不是 alpha feature。

---

## 11. Active universe

资产只有在 prefix 中至少存在一条有效 daily return 才进入当日模型空间。

[当前实现]

```julia
active_universe_indices(adj)
```

全 NaN dummy asset 必须严格不影响已有资产模型：

\[
[P,NaN]\Rightarrow w_{1:N}\text{ unchanged}.
\]

这是必须测试的性质。

---

## 12. Macro / relative field

先用 ruler 标准化 observed returns：

\[
u_{t,i}=\frac{r_{t,i}}{s_{1,i}}.
\]

对第 t 行实际 observed set \(O_t\)，大小 \(n_t\)：

\[
m_t=\frac1{\sqrt{n_t}}\sum_{i\in O_t}u_{t,i}.
\]

relative field：

\[
e_{t,i}=
\begin{cases}
 u_{t,i}-\frac1{n_t}\sum_{j\in O_t}u_{t,j}, & i\in O_t,\\
 0,& i\notin O_t.
\end{cases}
\]

### 12.1 Zero embedding 的精确定义

[规范 / V1]

missing return 仍然是 missing；但为了定义当前 active coordinate space 中的 relative **field**，未观察坐标被嵌入为 0。

这不是：

- 将缺失收益 impute 成 0；
- Gaussian marginalization；
- 假设停牌收益为 0。

原始 observation mask 必须单独保留。

当前实现：

```julia
embedded_relative_field(r, s1)
```

返回：

- `macro_flow`
- `embedded`
- `observed`

---

## 13. 固定 relative gauge

理论 relative space 是：

\[
\mathbf1^\perp.
\]

数值上可用 Helmert basis：

\[
Q^TQ=I,\qquad Q^T\mathbf1=0.
\]

当前 `relative_gauge(N)` 返回确定性 Helmert basis。

重要：

> Helmert basis 是纯数值 gauge，不是市场天然板块。

自然结构是算子谱，不是“第 k 个 Helmert 坐标”。

模型应满足资产 permutation covariance，以及理想情况下 relative gauge rotation covariance。

---

## 14. Cumulative path coordinates

定义：

\[
X_m(t)=\sum_{s\le t}m_s,
\]

\[
X_{\perp,i}(t)=\sum_{s\le t}e_{s,i}.
\]

必须在 level/path 上构造 Q/P basis，不能在 numerator 用 returns、denominator 却用 integrated levels。

这是历史上已修正的一个关键 bug。

---

## 15. Gauge-invariant scalar relative ruler

每个 temporal band \(\tau\) 使用一个 relative scalar ruler：

\[
s_\perp^2(\tau)
\propto
\frac1{N-1}
\operatorname{tr}\operatorname{Cov}
[X_\perp(t)-X_\perp(t-\tau)].
\]

当前 `compute_s_perp(X_rel)` 实现为历史平方位移的平均，并乘 \(N/(N-1)\) normalization。

为什么必须是 per-band scalar，而不是每 Helmert coordinate 一个 scaler？

因为逐坐标 normalization 会破坏：

\[
Q\to QR
\]

下的旋转协变性。

---

# 第五部分：Q/P 路径 basis

## 16. 时间滤波定义

对某 path level \(X(t)\) 和 band \(\tau\)：

近期窗口均值：

\[
c_0(t,\tau)=\frac1\tau\sum_{s=t-\tau+1}^t X(s),
\]

前一期窗口均值：

\[
c_1(t,\tau)=\frac1\tau\sum_{s=t-2\tau+1}^{t-\tau}X(s).
\]

定义：

\[
Q(t,\tau)= -\frac{X(t)-c_0}{s(\tau)},
\]

\[
P(t,\tau)= \frac{c_0-c_1}{s(\tau)}.
\]

项目中 Q/P 是 paired local path coordinates：

- Q 类似“位置偏离”；
- P 类似“方向/确认”。

它们不是有限 Markov state 的本体声明，只是当前 response operator 的有限 path basis。

当前实现：

```julia
path_basis_1d
build_X_rel_stacked
fill_design_matrix!
compute_B_rel_at_t
```

relative feature dimension：

\[
P_{features}=2\times |BANDS|\times N=14N.
\]

---

# 第六部分：Response Operator

## 17. 理论响应

理想无限维局部线性响应：

\[
\mu_l(t)=\sum_k\int G_{lk}(\tau)a_k(t-\tau)d\tau.
\]

有限 basis 实现：

\[
\mu_l=
\sum_{k,b}
[A_{lk,b}Q_{k,b}+B_{lk,b}P_{k,b}].
\]

记：

\[
G_b=A_b+iB_b.
\]

解释：

- diagonal：mode 自身延续/回归；
- off-diagonal：跨资产/跨 mode 传递，即 sector rotation 的价格内生版本；
- paired Q/P coefficients 可组合为 amplitude / phase：

\[
\rho=\sqrt{a^2+b^2},\qquad
\theta=\operatorname{atan2}(b,a).
\]

相位解释只在明确的 paired basis 下成立，不得把任意 Fourier phase 无条件解释成同一金融语义。

---

## 18. Macro response

Macro basis \(B_m\) 为 14 维 Q/P path features。

目标：

\[
y^m_{t+1}=m_{t+1}.
\]

ridge / Bayesian linear regression：

\[
y=X\beta+\epsilon.
\]

当前 response fit 同时支持：

- direct statistics \(S_{xx},S_{xy},S_{yy}\)；
- full design matrix when dual branch needs rows。

---

## 19. Relative Matrix-Normal model

训练：

\[
Y=XG^T+E,
\]

其中：

- \(X\in\mathbb R^{n\times P}\), \(P=14N\)；
- \(Y\in\mathbb R^{n\times N}\)；
- \(G\in\mathbb R^{N\times P}\)。

Gaussian / matrix-normal prior：

\[
G\mid\Sigma,\alpha
\sim
MN(0,\Sigma,\alpha^{-1}I).
\]

无约束 posterior column covariance：

\[
V=(X^TX+\alpha I)^{-1}.
\]

形式上：

\[
G\mid\Sigma,Y
\sim
MN(\hat G,\Sigma,V).
\]

其中 \(\Sigma\) 是 relative innovation covariance，必须限制在 relative support 内。

---

## 20. Primal / dual 必须数学等价

若 \(n\ge P\)，可在 feature space：

\[
X^TX.
\]

若 \(n<P\)，可在 sample space：

\[
XX^T.
\]

通过 Woodbury / SVD identity 得到完全相同的 ridge solution 和 covariance action。

选择 primal 或 dual 属于数值工程，绝不允许改变统计模型。

当前 `ridge_spectrum` 根据 `size(X,1) < size(X,2)` 选择 dual。

---

# 第七部分：Structural Neutrality

## 21. 每 band trace neutrality

V1 明确采用：

\[
\boxed{
\operatorname{tr}A_b=0,
\qquad
\operatorname{tr}B_b=0,
\quad\forall b
}
\]

等价于每个时间尺度的 universal common timing response 被移除。

它是 structural / identification constraint，不是防过拟合的数值 regularizer。

含义：

> 预测必须来自 cross-asset / cross-mode heterogeneity，而不能依赖所有资产在某 band 上同样的趋势或同样的 anticipatory bias。

---

## 22. Neutrality 必须存在于 posterior support

禁止：

1. 先拟合无约束 posterior；
2. 只把 mean 的 trace 减掉；
3. covariance 仍允许离开 neutral subspace。

精确线性 constraint：

\[
Cg=0,
\qquad g=\operatorname{vec}G.
\]

若无约束：

\[
g\sim N(m,\Omega),
\]

则 constrained mean：

\[
m_c
=
m-\Omega C^T(C\Omega C^T)^{-1}Cm.
\]

constrained covariance：

\[
\Omega_c
=
\Omega-
\Omega C^T(C\Omega C^T)^{-1}C\Omega.
\]

当前 `condition_trace_neutrality` 利用 Matrix-Normal / RidgeCovariance 结构，避免形成巨大 \(\Omega\)，把核心约束 solve 压到 14×14。

---

# 第八部分：Empirical Bayes 与 posterior uncertainty

## 23. Bayesian anti-overfit 原则

真正允许的 anti-overfit 来源只有：

1. structural constraints；
2. finite-data posterior uncertainty；
3. evidence / complexity cost。

不允许把“为了计算快而截 rank”伪装成统计 anti-overfit。

若使用 rank truncation：

- 因 evidence 支持：统计层；
- 只因计算：数值层，必须做收敛。

---

## 24. Alpha 不得是交易策略 knob

固定 `ridge_alpha=10` 只能作为测试/注入接口，不应成为最终生产策略参数。

生产默认：

```julia
ridge_alpha = nothing
```

表示通过 evidence / EB 求 alpha。

当前允许边界：

```julia
EB_ALPHA_MIN = 1e-4
EB_ALPHA_MAX = 1e6
```

它们应被视为数值域界，不是金融超参数；若 optimum 长期撞边界，应报警并重新审查 prior/model。

---

## 25. Joint conditioned EB

[当前实现]

Relative fit 使用 `optimize_conditioned_eb`，目标是在 trace-neutral posterior support 下联合得到：

\[
\alpha^*,\Sigma^*.
\]

求解器必须：

- 具有 alpha stationarity certificate；
- 具有 covariance stationarity certificate；
- line search 失败但 gradient 未收敛时必须报错；
- 不允许静默返回非 stationary point。

`EB_COVARIANCE_FLOOR = 1e-8` 当前是数值保护，不得当成理论风险参数；1.0 测试必须验证 floor refinement 时目标权重收敛。

---

## 26. Total posterior predictive uncertainty

若模型参数/算子存在 posterior：

\[
\bar\mu=E_\Pi[\mu_G],
\]

\[
\bar\Sigma
=
E_\Pi[\Sigma_G]
+
\operatorname{Cov}_\Pi(\mu_G).
\]

因此 epistemic disagreement 会自然增加 predictive risk，Kelly 仓位自动缩小。

禁止单独再引入“置信度系数”乘仓位。

---

# 第九部分：Out-of-Fold 认识论校准

## 27. OOF 的目的

OOF 不是 execution cross-validation，不是超参选择器。

目的是构造真正 out-of-fit innovation：

\[
\epsilon_t^{OOF}
=
r_{t+1}-\hat\mu_t^{(-fold(t))}.
\]

如果同一个 observation 参与拟合 \(G\) 又用于计算自己的 residual，则 residual 会系统性过窄。

---

## 28. Fold 规范

V1 默认：

```julia
F_folds = 3
```

训练 rows：

```julia
ts_total = WARMUP:T-2
```

contiguous folds。

每 fold train sufficient stats：

\[
S^{(-f)}=S^{full}-S^{(f)}.
\]

每 fold 必须独立求自己的：

\[
\alpha_m^{(-f)},\quad\alpha_{rel}^{(-f)},\quad\Sigma^{(-f)}
\]

或任何规范中定义的 posterior nuisance parameter。

昨天同一 fold 的 optimum 可以作为 warm start；不得把今天 full-fit data-dependent optimum 当成固定 fold hyperparameter。

---

## 29. Lazy ResidualOracle

[当前实现 / 保留]

为避免每日物化整个 \(T\times N\) OOF residual matrix，生产路径使用 `ResidualOracle <: AbstractMatrix{Float64}`。

它冻结本次 solve 所需：

- fold models；
- fold ranges；
- `ts_total`；
- raw returns；
- `s1`；
- `s_perp`；
- macro basis；
- relative path prefix sums。

接口语义必须与 dense residual matrix 一致：

```julia
size(o)
getindex(o, row, asset)
residual_row(o, row)
residual_rows(o, rows)
own_residual_rows(o)
macro_residual_series(o)
```

reference 测试：

\[
ResidualOracle[row,j]
\approx
DenseOOF[row,j]
\]

允许 summation order 导致 roundoff 差异，不要求 ULP-identical。

当前源码中“NOT YET WIRED”类历史注释必须删除或更新；生产代码已实际使用该 oracle。

---

# 第十部分：Innovation / Conditional Risk

## 30. Response + innovation = predictive law

完整模型不是只有 conditional mean：

\[
\boxed{
\text{response operator}
+
\text{conditional innovation law}
=
P(r_{t+1}\mid\mathcal H_t)
}
\]

当前 V1 innovation 是：

1. OOF residual cross-sectional row bootstrap；
2. macro residual 上的 fractional-memory volatility scaling；
3. missing cell 用该资产 own observed residual row 回退。

这是 KISS 版 non-Markov risk，不声称是最终完整无限维 innovation law。

---

## 31. Fractional volatility posterior

当前 grid：

```julia
const DGRID_V1 = [0.05,0.1,0.2,0.4,0.6,0.8,1.0]
```

数值 quadrature cell widths：

```julia
const DELTA_D_V1 = [0.025,0.075,0.15,0.2,0.2,0.2,0.1]
```

这一点非常重要：posterior 权重必须包含 \(\Delta d\)，否则非均匀数值网格会偷偷改变 prior。

fractional weights：

\[
\pi_0=1,
\qquad
\pi_k=\pi_{k-1}\frac{k-1+d}{k}.
\]

causal conditional variance：

\[
v_t(d)
=
\frac{\sum_{j<t}\pi_{t-j}(d)e_j^2}
{\sum_{j<t}\pi_{t-j}(d)}.
\]

Gaussian quasi log-likelihood：

\[
\ell(d)
=
-\frac12\sum_t
\left[
\log v_t(d)+\frac{e_t^2}{v_t(d)}
\right].
\]

posterior quadrature：

\[
p_g
\propto
\exp(\ell(d_g))\,\Delta d_g
\]

（若未来有明确连续 prior \(p(d)\)，再乘 \(p(d_g)\)）。

当前使用 FFT 做完全相同的有限卷积，不改变数学结果。

---

## 32. Absolute volatility scale

不能用：

\[
\sqrt{v_d/\operatorname{mean}_d v_d}
\]

因为那会消掉总体 volatility level。

当前规范：

\[
\boxed{
scale_d
=
\sqrt{\frac{v_{T+1}(d)}{v_{bootstrap}}}
}
\]

其中：

\[
v_{bootstrap}=\operatorname{Var}(e^{macro,OOF}_{history}).
\]

---

# 第十一部分：Decision-time Posterior Predictive

## 33. 不抽整张 G

决策只关心当前 feature \(x_t\) 下：

\[
Gx_t.
\]

从 Matrix-Normal posterior 直接计算 predictive marginal：

\[
\mu_{rel}=E[G\mid H]x_t,
\]

\[
\Sigma_{rel,x}
=
\operatorname{Cov}(Gx_t\mid H,Cg=0).
\]

这是精确边缘化，不是降阶近似。

当前 `predictive_moments` 返回：

- `mu_m`
- `var_m`
- `mu_rel`
- `L_rel`
- `x_features`

`L_rel` 是 predictive covariance factor；scenario generator 当前进一步转换为唯一对称 PSD 主平方根，以消除 eigenvector sign / degenerate basis 的 RNG coupling 不确定性。

---

## 34. Asset-space mean

当前 active assets 下：

\[
\mu^{norm}
=
\mu_m e_0 + \mu_{rel}^{zero-sum}.
\]

最终 asset return mean：

\[
\boxed{
\mu_i
=
s_{1,i}\mu^{norm}_i
}
\]

然后 embed 回完整 universe；inactive asset 不得进入 model coordinate space。

---

# 第十二部分：Scenario Generator

## 35. Scenario law

每个 scenario 包含：

1. macro posterior draw；
2. relative posterior predictive draw；
3. fractional \(d\) posterior draw；
4. OOF residual row bootstrap；
5. ragged residual missing-cell own-row fallback；
6. exponentiate log gross return。

schematically：

\[
\log R_i^{(s)}
=
\left(
\mu_m^{(s)}e_{0,i}
+
\mu_{rel,i}^{(s)}
-
\overline{\mu_{rel}^{(s)}}
\right)s_{1,i}
+
scale_{d_s}\epsilon_{i,row_s}^{OOF}.
\]

---

## 36. RNG determinism

[规范]

给定：

- 完全相同 input history；
- 完全相同 seed；
- 完全相同 scenario count；
- 完全相同 posterior covariance；

scenario 应在合理 roundoff 内可重复。

当前固定回测使用：

```julia
MersenneTwister(seed + t)
```

从而 worker 调度不影响某个日期的随机流。

不得引入 thread-order-dependent global RNG。

---

## 37. IID 与 quadrature

`generate_scenarios_v1` 支持：

- IID Monte Carlo；
- nested randomized low-discrepancy quadrature（当前 Halton）。

数值 refinement 必须依据：

\[
\|w_{2S}-w_S\|_1
\]

和 Kelly objective certificate，而不是回测 Sharpe。

未来可替换为 Sobol/Owen，但这是数值 backend，不改变 posterior law。

---

# 第十三部分：Kelly Solver

## 38. exact sample Kelly

`kelly_weights_v1(X)` 解决：

\[
\max_{w\ge0,\ \sum w=budget}
\frac1S\sum_s\log(base_s+X_sw).
\]

`base_s` 表示 locked holdings 的 scenario wealth contribution。

### 38.1 Locked positions

若某资产不可交易但已有持仓，其风险不能当成 cash。

必须：

\[
base_s=R_{s,locked}^T w_{locked}.
\]

free assets 只优化剩余 budget。

---

## 39. 两个 solver，一个目标

当前：

- `fast_kelly_solver`：专用 Newton/barrier；
- `clarabel_kelly_solver`：Convex/Clarabel reference。

允许 fast solver 失败后调用 Clarabel，但必须解**完全相同的 log-Kelly objective**。

禁止 fallback 到 mean-variance。

---

## 40. Kelly certificate

利用 concavity：

\[
F(w^*)-F(w)
\le
budget\cdot\max(g)-g^Tw.
\]

必须检查：

- simplex feasibility；
- nonnegativity；
- KKT residual；
- objective upper gap。

只有 certificate 通过才能接受 fast solver 结果。

---

# 第十四部分：Execution 层

## 41. Broker / Rebalance 边界

理论只给 weights。

执行器处理：

- snapshot；
- available cash；
- tradability；
- foreign positions；
- cancellations；
- target shares；
- progressive liquidation；
- limit orders。

当前主要 API：

```julia
tradier
positions
orders
snapshot
tradable
held_weights
target_shares
liquidation
rebalance!
```

### 41.1 Leverage

目标 weights 满足 fully invested long-only；执行器通过 budget / cash 约束确保不借钱，目标 notional 不超过管理资本。

### 41.2 Foreign holdings

universe 外已有持仓只能渐进卖出，不允许理论模型给其新增目标仓位。

---

# 第十五部分：Backtest 语义

## 42. Backtest 基本循环

对 decision day t：

1. 只使用 `signal[1:t,:]`；
2. 构造 model；
3. 产生 posterior scenarios；
4. 根据当天 tradability + 当前 held 计算 Kelly target；
5. 用 t→t+1 marking gross 更新账户；
6. 进入下一天。

禁止未来 bar 进入 t 时刻推断。

---

## 43. Reference backtest 与 fast backtest

[规范]

1.0 必须保留一个极简 sequential reference runner。

Fast scheduler 可以存在，但必须只改变执行顺序和缓存，不能改变任何日期的数学输入或随机种子。

当前 backtest 已发展出：

- contiguous blocks；
- incremental checkpoints；
- producer/consumer；
- disk spool；
- adaptive mode channels。

这些都属于计算机科学加速或性能实验，不得进入模型定义。

1.0 架构整理后建议：

- reference sequential runner 常驻；
- fast block runner 独立；
- 性能 probe 不 export 到生产 API。

---

# 第十六部分：目标架构——PreparedProblem 边界

## 44. 为什么需要 PreparedProblem

当前复杂性最大的根源是 batch / incremental / probe / OOF / residual 之间共享太多内部状态。

1.0 必须引入一个明确的数学中间对象：

\[
\boxed{PreparedProblem_t}
\]

它代表“历史已经变成可供 posterior solver 使用的充分输入”。

### 44.1 建议字段

规范上至少包含：

```julia
struct PreparedProblem
    active_indices
    N_universe
    T
    N

    # geometry / scaling
    s1
    s_macro
    s_perp

    # path representation needed at decision time
    B_macro_now
    B_relative_now

    # full response sufficient statistics
    full_macro_stats
    full_relative_stats

    # OOF fold sufficient statistics / row ownership
    fold_ranges
    fold_macro_stats
    fold_relative_stats

    # data needed by residual provider / innovation
    residual_source_inputs
    observation_masks
    ts_total
end
```

字段具体类型可优化，但语义必须固定。

---

## 45. 唯一 solver

必须只有一个：

```julia
solve(prepared::PreparedProblem) -> V1Model
```

它负责：

- full response posterior；
- fold response posterior；
- exact trace conditioning；
- residual provider；
- fractional innovation posterior；
- decision-time predictive moments。

Batch、incremental、未来 GPU 只能改变：

```julia
prepare_xxx(history) -> PreparedProblem
```

不能复制 posterior solver。

---

## 46. Reference prepare

必须保留：

```julia
prepare_reference(history) -> PreparedProblem
```

特点：

- 最简单；
- 从完整 prefix 重算；
- 不追求最快；
- 是 executable specification；
- 测试所有 accelerator against it。

---

## 47. Accelerator prepare

可有：

```julia
prepare_incremental(state) -> PreparedProblem
```

未来也可有：

```julia
prepare_gpu(...)
```

但它们都必须逐字段满足：

\[
Prepared_{fast}\approx Prepared_{reference}.
\]

不是只比较最终收益曲线。

---

# 第十七部分：跨日增量化的规范边界

## 48. 增量化不是理论

跨日增量是 exact evaluation algorithm：

\[
State_t+\text{new day}
\rightarrow
State_{t+1}
\rightarrow
PreparedProblem_{t+1}.
\]

它不能改变：

- ruler；
- history length；
- folds；
- EB；
- posterior rank；
- neutrality；
- OOF；
- innovation；
- Kelly。

---

## 49. Ragged exact factorization

对第 t 行 mask \(m_t\)：

\[
M_t=\operatorname{diag}(m_t),
\qquad
n_t=\mathbf1^Tm_t,
\]

relative projector：

\[
P_t=M_t-\frac{m_tm_t^T}{n_t}.
\]

令当前 day metric：

\[
d_i=1/s_{1,i}.
\]

zero-filled raw return \(z_t\) 只作为 field algebra 表示，则：

\[
\boxed{
e_t(d)=P_t\operatorname{diag}(z_t)d=A_td
}
\]

其中 \(A_t\) 与当前 decision-day metric 无关。

这证明 ragged 不构成数学不可增量化的理由。

---

## 50. 当前 mask-run accelerator

[当前实现 / 过渡]

`incremental.jl` 已实现：

- unique observation-mask registry；
- maximal constant-mask runs；
- regime interior aggregated Grams；
- crossing-window row materialization；
- IPO active-space exact coordinate embedding；
- per-mask/per-fold statistics；
- resource budget route to exact batch reference；
- `async_seen` 仅保留为 diagnostics，不再永久关闭 fast path。

当前资源 guard：

```julia
REGIME_GRAM_BUDGET_DEFAULT = 512 MiB per state
REGIME_MATERIALIZE_ROW_LIMIT_DEFAULT = 4096
REGIME_WINDOW_DEPTH = 256
```

这些预算只决定 route，不得改变结果。

---

## 51. 增量 backend 的正确验收

逐 decision day 比较 reference：

\[
\frac{\|S_{xx}^{inc}-S_{xx}^{ref}\|_F}
{1+\|S_{xx}^{ref}\|_F}<\epsilon,
\]

同理：

- `s1`
- `s_macro`
- `s_perp`
- full/fold `Sxy/Syy`
- current basis
- posterior alpha
- Sigma
- predictive moments
- residual oracle cells
- scenarios（固定 canonical root 与 seed）
- final weights。

不能只做 end-to-end PnL 近似。

---

# 第十八部分：模块级恢复规范

## 52. 当前仓库模块与职责

当前快照约 5231 行，文件：

```text
backtest.jl
broker.jl
ceiling_probes.jl
data.jl
geometry.jl
incremental.jl
kelly.jl
KTrader.jl
live.jl
numerics.jl
predict.jl
residual_oracle.jl
response.jl
```

### 52.1 `data.jl`

负责：

- `Bars`；
- carried marking series；
- `signal_prices`；
- Yahoo ingestion；
- save/load CSV；
- market final-bar date logic。

不得出现 response / Kelly 逻辑。

### 52.2 `geometry.jl`

负责：

- TAUS/BANDS/WARMUP；
- per-asset ruler；
- prefix ruler stats；
- center-of-mass decomposition；
- general pairwise covariance utility。

不得做 posterior。

### 52.3 `numerics.jl`

负责：

- timing；
- matrix workspace；
- price cache；
- Helmert gauge cache；
- RidgeCovariance spectral action；
- FFT workspace pool；
- canonical PSD square root。

里面所有常量都必须被分类为数值参数，而不是金融理论参数。

### 52.4 `response.jl`

负责：

- Q/P basis；
- design matrix；
- response sufficient statistics solver；
- ridge spectrum；
- Matrix-Normal EB；
- conditioned trace neutrality；
- predictive response moments。

这是最核心数学模块之一。

### 52.5 `residual_oracle.jl`

负责：

- dense residual reference；
- lazy residual provider；
- exact row reconstruction；
- macro residual series。

最终应收缩成接口明确的小模块。

### 52.6 `predict.jl`

负责：

- embedded relative field；
- fractional posterior；
- active universe；
- reference prepare；
- full + OOF model assembly；
- V1Model；
- scenarios / adaptive quadrature。

架构整理后应把 diagnostic `oof_shadow` plumbing 移出 production。

### 52.7 `incremental.jl`

只应负责：

\[
History/State\rightarrow PreparedProblem.
\]

不得再拥有独立 response / posterior / Kelly 数学。

### 52.8 `kelly.jl`

负责唯一经济决策：

- certificate；
- fast exact-objective solver；
- Clarabel reference；
- locked wealth；
- scenario weights；
- convenience `path_kelly_v1`。

### 52.9 `backtest.jl`

只负责时序模拟与调度。

不要成为模型模块。

### 52.10 `broker.jl`

账户与订单 execution。

### 52.11 `live.jl`

Daily theory 与 broker execution glue。

### 52.12 `ceiling_probes.jl`

[已执行 — M1 施工；运行验证为部分通过，见 §53 的精确状态]

原[过渡]目标已执行：`src/ceiling_probes.jl` 已物理删除，
`KTrader.jl` 不再默认 include / export 任何探针；backtest 的 `probe`
参数与 collector plumbing 已删。诊断现位于：

```text
dev/probes.jl   # module DevProbes，显式加载（bin 经 include dev/probes.jl）
```

旧 counter factory（`oof_shadow`/`OOFCounters`/身份异常）已废除，
仅存于历史记录。本节保留原目标作为规范依据；当前源码接口以
M1 施工后的实际 source 为准。

---

# 第十九部分：当前公开 API 快照

## 53. KTrader.jl 当前依赖

```julia
using LinearAlgebra, Statistics, Random, Dates
using TimeZones
using PrecompileTools
using Convex, Clarabel
using HTTP, JSON3, CSV
using FFTW, Distributions
```

当前 include 顺序（M1 施工后实际 source；旧快照基准
`repomix-output(20261007-113017).xml` 时为含 `ceiling_probes` 的
13 文件链，见 §72 的历史行数记录）。运行状态（范围精确，局部绿
不泛化为发布）：Prepared 契约 56 项全过（B4b 以 `isequal` 比值，
非地址绑定）；timed 7、admission 17、model-API small 最后全部绿；
architecture 标准的真树检查与 fixture 红绿均已实测；604 项
timeblock 测试通过——真实根因是 root call 的 `Base.close` 修复加
L91 的 inc 期望改为相对 first/last−1（batch 计 0；原 7 红全部来自
inc 期望而非 close 时序，历史归因以此为准）；独立 reference R1+R2
单相 52 项与 R3 单相 33 项（N2 默认 EB / S24 / oneDay / Channel
会计）均 RC0——R3 以 `--compile=min` 纯语义配置运行（12.5 s），
该模式仅为语义验证配置、不是 throughput 口径（默认模式的超时
历史保留；40−12.5 之类"精确 compile 份额"的推论不写入——编译
模式同样改变 runtime 常数，且无 native compile_time 证据）。
required registry 已 12 项（历史时点陈述：12 是当时旧记忆、已撤回，现行 REQUIRED 21 + 1 architecture 以下文实际 registry 为准——18 是 2026-10-07 的真实验证范围、保留作时间线不改历史事实；三个 admission-integration 交付加入后为 21）；standard full 运行拒绝 partial 阶段 env，防 silent skip。没有 ALL 级命令绿。true-N65 typed 数理对照已
DevOps 实测 RC0（历史时间线注记 2026-10-07 文档审计：此句写作时残差 finite-check 尚待补，后文已记录其后续实测 RC0；原文保留作时间线，不回改）：旧 inc Prep NamedTuple 在 dev 显式映射 typed
PreparedProblem（完整 T14309/N65/P910/F3，raw/stat 无截断重算，
owner None），单 cold 默认 EB BLAS6 solve 7.449 s（保存
/tmp/model_m1_t14309.jls）；对旧 I0_inc 报告 alpha/Gmean/Sigma/
mu/covLprod/d/vforecasts 字段全 0diff，对 batch_final alpha_rel
相对 8.53e-14 / Gc 2.47e-13 / Sigma 2.82e-13 / mu 1.13e-14 / cov
6.13e-15，原 tol 过；单独 sameSeed canonical S300 X 9.77e-14 /
wL1 2.31e-12、两 KKT 约 1e-10 / feas 2.2e-16 / objective gap
6.29e-9 / min wealth ≈0.804 / swap loss 浮点零，无重复 fit。定性
边界：这是单例重构保真，不是性能达标也不是 whole M0/M1——7.449
无 controlled before-after 不可解释为"M1 优化所以更快"；0diff 仅
覆盖上述字段、同 input 本机，非机群字节级。残差 finite-check 已实测 RC0（只加载 artifacts、不复
solve/prepare）：F3 每折首中尾确定性 9 行
[1,2342,4684,4685,7026,9368,9369,11710,14052]，块 9×65=585 cell，
260 finite / 325 NaN；shape 相同，这 9 行的 isfinite 与 NaN 位置
精确一致，finite maxdiff 对 I0_inc 为 0.0、对 batch 3.0e-14（原
tol 1e-9 过）。定性：**样本行保真**，不是全 T×N 逐 cell、更不是
机器群 byte guarantee——不得把样本 mask 说成全历史 cell 通过。
最小重放说明见 dev/m1_typed_replay.md（源码/输入哈希含算法名、
artifact/model 路径供续行，原 8 旧物件保全）。最终 DevOps 实测
Julia 0、HEAD 602b897 未动、工作树 38 项未提交未清用户改动；
全部窄命令 60s/RSS 护栏、全 suite/多日/GPU 继续暂停。（2026-10-08 收口更新：工作树现同时保留既有用户改动与本轮工程改动——未 commit/reset/clean；具名最终快照 dev/evidence/manager3/final_snapshot.txt 记录 16 个关键 runtime 对象的完整 SHA256。）措辞修正（2026-10-07 文档审计）：撤回「此次 M1 最小 source 收口」的收口表述——已验证的仅是上文列出的单例对照本身；M1 边界仍有进行中或新交付、未运行验证的事项（workspace generation 不再笼统写 RUN PENDING——本任窄绿：prepared_problem_contract_tests 61/61 RC0 23.3s、prepared_lifecycle_tests 24/24 RC0 12.4s，scoped 45s/RSS2048 现有干净加载/单线程下无修复/无放宽；ArchitectureGate 同入口合法 minimal/nested RC0、bad/empty RC1、真树 --architecture-only RC0 且对象 hash 前后不变；caller finite↔NaN 旧 Prepared 冻结与 fresh 响应、显式 mask copy、workspace 复用 model 快照已由这 85 项证明该 fixture 范围——区分本任窄绿与前任 56 数字历史；shared workspace 仍 generation borrowed、default 无 ws 独立；B/collector 最终 274/274 RC0；CLI 已真实运行观察 RC、dev probe 重接线已落地运行；动态 include fail-closed 已由 registry REQUIRED 18 的 Gate 再读确认工作）。owner 工作按 source 事实与方向记录、不作运行验证结果：ReviewMath 已交付（source-written、测试未跑）PreparedProblem 新 field alive_now::Vector{Bool}（owned constructor copy，default=isfinite.(adj_act 末行) 仅在 constructor 执行；solve 以 findall(prep.alive_now) 计算 e0，不再读取 adj_act；当日观测支撑 owned mask，不复制 whole panel；旧 dev 22-key NamedTuple 可 keyword Ctor 自动补 mask；旧 typed JLS schema 不保证直接 deserialize，须显式重 wrap 不得假绿）；ReviewEngine 修 daily-mask（非 activation rho）；ReviewGate 修动态 include fail-closed；DevOps 只修 guard。最终结果将回报并再审计、不作绿预填。DevOps 后续实证（2026-10-07）：13 份 /tmp artifact 已保全；旧增量 wrapper 5 keys（prep, fast, prepare_seconds, limit, peak_rss_kb）；prep 22 keys 已实证（active_idx, N_universe, adj_act, T, N, r, s1, field, m, relative_embedding, X_rel, s_m, s_perp, B_m, ts_total, n_res, n_bands, P_features, X_rel_stacked, Y_target_rel, stats, macro_stats，observed 在 field 内）；旧 canonical checker seed=1、S=300；typed solve 记录形态为 KTrader.solve(prep) 默认 cold EB、F3、ws None/gen0；旧 m1_convert/m1_canon/m1_rescheck/i0_fit 文本脚本存在可读（精确路径由 Manager 向 DevOps 补，不猜）；guard wrapper 最终 8/8 RC0、15s、deadline50、RSSpeak 321MiB（ALL PASS 汇总也算一场、真实独立脚本场景仍 9，不是 10；bin/scoped_run.sh sha256=c3d776e40f39a32f0a144e3749a7ba2c3a9ccc6c7ddb3d4610d28277d04a26d6；test/scoped_run_tests.jl sha256=eb9dec1b71982ab5d639c42c912a3020526c1d383c9201bb7dcd23006a8eac5f）；完整 artifact hashes 待下次从已存 log 传入、勿编造；核心 src 新 field 行为已由上述 85 项窄绿覆盖（fixture 范围）；ArchitectureGate 已 RC0；CLI 已真实运行观察（合法 400×2 CSV roundtrip、合法 B/小 phase solve、非法 unknown flag/probe 矩阵均 RC；library cli_boundary 50/50 RC0、admission 17/17、modelApi 8/8）；B/collector 测试最终 274/274 RC0（13.6s，existing 配置；compile=min 失败保留为非吞吐口径数据点；此前 104/6/1 及三个 fixture 根因保留为修复时间线）；最终 dev oof_timing_split wrapper 真实透传 rep.observer_error 与 timing、测试恢复 observer_error 断言、撤销 warmup_error 顶替；admission 17/modelApi 8 复验绿。registry 实际 stdlib 与 Gate 再读均确认 REQUIRED 18（含生命周期/CLI/四 legacy；前报 12 是旧记忆已撤回）。——该 18 是 2026-10-07 时点的真实验证范围（历史记录，不改写为 21）；现行 registry 为 21 REQUIRED + 1 architecture：incremental_budget / history_cache / artifact_replay 三个 contract 测试已加入 REQUIRED（源码静态就绪），其运行验证由 DevOps 受控记录后补。replay 四 phase（check/rewrap/model-only/residual-sample）已 RC0；solve-once 当前源两次运行 7.202s（stdout 对象丢失，失误证据不抹）与 8.336s（含 ~2s 持久化新 artifact /tmp/model_solveonce_t14309.jl2s，安全不覆旧，双基线字段对照与 9 行残差已做）；新 artifact 完整 SHA256 ae54c564f3005cff022703a15637f21be4c0a44f6261c92791c3948db0a2016c 已入唯一 manifest，sameSeed 数值已实测：新 vs inc/m1 X/w/obj/swap 全 0；新 vs batch X 9.77e-14、wL1 2.31e-12、双 KKT ~1e-10、gap 6.29e-9、swap ±2.5e-17；9 行 585 cells finite 260/NaN 325、masks 同、finite 对 inc 0、对 batch 3e-14（仍样本范围）。7.202 对象未保存返工成本保留；8.336 含持久化、非提速 claim。主 mask 源协议（forward [t,t+255]、boundary_cost 独立、stats [from,T-1]、bar T 免疫）及分名 rationale（rho 用户度量 vs boundary_cost 物化边界、混同即错误归因）、owned O(N) alive mask + generation 借用 rationale、pkg using 序列化身份 rationale、B 拓扑数据与 provenance 边界均见 README 对应段。真实 B 只可称「已保存 artifact 有限端点 mask 拓扑」：T14309/N65 prefix+ignored 末行哨兵，全 K 14308/H 59/unique 60/median 121.5/p90 617.3/max 1966/rho ≈ .5454；尾 2000 H 8/unique 9/median 239/p90 441.6/max 460/rho .754/boundary .7535。原 panel 来源仍 unknown、无存活原调用链 signal_prices 证据，530055 NaN 不证明非 carried；历史 rho .563 差因 unknown 不能归因少 1 bar，不同指标/初始 mask 定义可能但未证。manifest 纠错已实证落档（文档 owner 直接只读核对一致）：错误分数 191/350、263.9/350 已删，授权单次纯 mask 精度补缺 check（零 prepare/solve/scenario/fit，scoped_run 50s/RSS2048 RC0）取得精确整数覆盖——from2 K14308 forward=boundary=7803/14308=0.5453592395862454；末 2000 from12310 forward 1508/2000=.754、boundary 1507/2000=.7535；ragged6 四位舍入 log 与 exact_cnt 精度 log 均留；manifest 最终 sha256 d290718e56fe1907196ffa62e3ef861dc11968d25005d315d6b441f6bdaa942d 为外部标识（不自引用循环）。原造错分数仅作一句时间线记账、不再当指标。旧 activation 表标 historical。全部窄绿不泛化为 whole M0/M1，fullsuite/多日/GPU 仍暂停。非 whole M0/M1 发布、非吞吐达标。compile=min 12.5 s 纯语义配置而非精确
compile 份额。用户所有 ≤60s/RSS 护栏、full suite/multiday/GPU
暂停与 Gate 顺序全部保留；旧 baseline 历史原值不动。whole
M0/M1 发布、吞吐与全 suite 多日仍禁止。

第3任修订（2026-10-08；(1)(2)(3) 均已受控运行窄绿（log 在 dev/evidence/manager3/），各项 scoped ≤50s/RSS2048/existing 配置）：
(1) IPO 预算边界：embed_active! 以实际持有的 FoldGram payload（V+Xc+R2 矩阵字节）为预算口径，目标字节先于任何扩张分配检查；超预算不分配、释放 Grams 并将行 demote 到同一数学的精确物化路线（route-only——raw history/坐标/fold membership 保留，预算恢复后重新聚合；账本只计持有 payload，非 OS RSS 上限、不称实测毫秒），账本与 payload 同步、checkpoint deepcopy 同步携带（test/incremental_budget_contract_tests.jl 以独立字节核算验证）。其运行结果（2026-10-08 DevOps，log 同目录）：1041/1041 RC0 9.4s（scoped 16s/RSS2048/peak 946MiB）。红历史保留：初次 1025 Pass / 10 Fail（total 1035，bud1_red_history.log）——10 个红并非单一根因：真实源码缺陷是 advance 首建 RawInferenceCore 时丢弃 initialize 校验过的 gram_budget/materialize_row_limit（死 kwargs，core 永用默认 512MiB/4096），另含两处独立的测试自身错误（错 active 日时序：首 finite price 360 vs 首有效相邻 return 361 恰差一日，embed/跨线只能在 361 验收；假笛卡尔积期望 6 Grams，实际 fold 驱动聚合为 4）；修复为 ExactInferenceState 在懒创建 core 前保存两配置、core 创建时传参、checkpoint 复制贯穿——两字段是 core 懒创建前保存已校验 resource options 的载体、非新理论 knob（默认值/route-only 语义不变，账本非 OS RSS 硬上限，tol/guard 未放宽）。
(2) cache 前缀守卫：verify_history_cache_prefix 对实际消费的 logs/returns/first_price/first_return 逐项拒绝异源、载荷与元数据突变（比对 logs 而非仅 returns，纯价格 level 平移亦拒；NaN 位置精确、Inf 拒绝），O(TN) 且无 panel 级分配；只读 1:T，合法未来 suffix 不读不设限；_prepare_v1 在首个 cache 读取前、initialize_inference 在任何 state 构建前接线（test/history_cache_contract_tests.jl）。其运行结果（2026-10-08 DevOps，ExactInferenceState 构造器新增两预算字段后复验）：4 个 testset（24+9+15+45）全绿 RC0、scoped 20s/peak 998MiB（其中 initialize_inference 统一守卫 testset 45/45、5.5s）。
(3) replay 硬断言：dev/m1_artifact_replay.jl 单一比较 owner error-on-failure；9 行残差块 finite-only（无 T×N 物化）；solve-once 拒绝覆写先于任何反序列化/拟合；tol 出处将历史阈值与 2026-10-08 Manager 新决策分开记录（test/artifact_replay_contract_tests.jl，已入 21 REQUIRED 标准 loop）。(3) 运行结果（2026-10-08 DevOps，log 在 dev/evidence/manager3/）：contract 测试 54/54 RC0 1.9s（首跑 52/2 为 fold 行手算索引期望错，修期望、不改门/容差）；check RC0（4/13 of-record hash 验证——不称全 13 库存验证）；model-only RC0 20s——load-only 新增 compare_model_fields 对 inc/batch 双基线 15 项 GATED（vs inc 全 0；vs batch alpha_rel abs 1.27e-9 / Gc 2.47e-13 / Sigma 3.97e-14 / mu 1.13e-14 / L*L' Frob 6.13e-15），路径绑定 manifest 对应对象（每 phase 先 sha256 验证、无换基线）、tol/seed 无变更，场景/Kelly 原 tol 仍过；residual-sample RC0 16s（9×65、finite 260/NaN 325、vs inc 0 / vs batch 3.0035e-14、mask 精确、Inf 拒，仍样本行范围）；solve-once RC1 预期红 17s——拒绝先于任何反序列化/拟合、无新输出路径、无 manifest 变更、旧输出磁盘 SHA 不变；真实坏字段红（producer→parser→consumer）：model_only_gate_probe.jl 对真实 deserialize 模型内存单 cell mu_pred +1 使同一 consumer 抛错后内存恢复、磁盘 artifact 不动。
共同约束与选择依据（链接 SPEC、不复制；非新 ADR、不逐 helper 记小决定）：三项修复共同守住三条线——原数学不变、可变输入/资源路径必须能红、≤60s 窄验证（SPEC §56 fail-loudly、§51 增量逐字段验收；资源 route 语义见 §50：预算只 route、不改输出）。预算修复选择"检查实际持有的 payload、超限即释放并 demote"，拒绝两个更廉价的假修——"先扩张分配再修打印账本"（护栏失效路径原样保留）与"以固定 mask×fold 理论最大数当实际占用"（与真实持有字节脱钩即误 route）。cache 修复选择对实际消费的 prefix 逐元素 O(TN) 精确校验、不分配整 panel，拒绝 shape/identity/抽样比对与靠 caller 自觉只读（各有静默漏检面）；其 O(TN) 代价尚无吞吐测量。replay 修复选择共用强比较器、真实生产证书与 9 行抽样，拒绝 println-only 与全历史 NaN-poisoned 物化。revisit 只由新真实证据触发（如证明 mutation 安全的更廉价 cache 契约、或获准 profile 显示 validation 成热点），公开重审而非偷偷改精度。
相邻窄绿复验（同批 DevOps，log 同目录）：incremental_primal_prep 46/46 RC0 10.2s、prepared_problem 契约 61/61 RC0 22.6s（含小合成 solve——“无 fit”仅指无 N65 真实 artifact 重拟合、无大 benchmark）、prepared_lifecycle 24/24 RC0 11.8s、真树 architecture-only 5 Summary 全 Pass RC0（21+1 准入口径）。原数学、历史数值、fullsuite/多日/GPU/吞吐暂停与 panel provenance unknown 边界全部保留不动；三项窄绿均为单机、无 tol/guard 放宽、无跨机群字节保证，不泛化为 whole M0/M1。
准入负例与 snapshot 收口（2026-10-08，log 同目录；harness 脚本已归档亲读核准）：required_negative.log 由 harness_required_negative.jl 产生——该 harness include 真实生产 test/contract_registry.jl 并调用生产 verify_required_contract_files：临时缺一 REQUIRED 文件确定性抛错（[NEG] threw=true、错误信息含 fail-closed）、all-present 通过（[POS] true），该次运行零 numeric testsets、不抛则 harness 自身失败——真原语、非 fake。fixture_root_4way.log 由 harness_fixture_root_4way.jl 产生——该 harness 构造四个 fixture 根并对其内容自检（filesize/include 计数/occursin dev/probes/isfile 断言），5/5、RC0；它本身不调用 test/runtests.jl --architecture-only --architecture-root（harness 自检不充标准入口 Gate 红）；标准入口对 fixture 根的红/绿（minimal/nested RC0、bad/empty RC1）由更早 DevOps 实测覆盖（见上文 workspace-generation 窄绿段）；as-run harness 源码及其 SHA256 绑定于 harness_snapshot.txt；期间 harness 临时 sed 误写 occursin==9 已修——属 harness 笔误、非生产 Gate 红，如实记录不混同。范围声明：「21 REQUIRED 登记与文件前置校验通过」不等于「21 套数值测试全部运行通过」——本任实际运行的 numeric 恰为上述 6 文件（budget/cache/primal/prepared/lifecycle/replay），各自独立 scoped 运行。具名最终快照 dev/evidence/manager3/final_snapshot.txt：16 个关键 runtime 对象（src/test/dev 脚本/manifest）的完整 SHA256（2026-10-08 01:54 +08:00，no commit/reset/clean）；README/AGENTS/dev note 为文档、不在 runtime snapshot 内；未变更既有 snapshot/manifest。
本任真实标准入口四向现已补齐（2026-10-08 02:03 +08:00；standard_entry_4way_summary.txt + std_entry_minimal/nested/bad/empty.log，四 log 的 SHA256 绑定于 summary）：实际命令 julia --startup-file=no --project=. test/runtests.jl --architecture-only --architecture-root <ROOT>——minimal/nested RC0（fixture-mode 声明 KTrader NOT loaded、全 Pass）；bad RC1 自然退出 killed=0（violations 含 unsupported include form 与 dev/probes.jl、dev/more_probes.jl，violations==String[] 断言失败于 architecture_contract_tests.jl:363）；empty RC1 自然退出 killed=0（fail-closed L356/357 src tree/module present 红加 include target not present: KTrader.jl）——拒因均从实际 consumer 输出可核、非路径错误。旧 harness_fixture_root_4way.jl 仅内容自检、仍保留为辅助证据；前次把 harness 自检升格为标准入口四向的措辞已纠正。超时时间线如实保留：初次 12s-deadline 尝试得 RC124（超时/停止、栈为 LLVM）——其成因无 native 编译计量、不以后来 3s 干净 RC1 反推，只记录为超时；25s-deadline 重跑 3s 自然退出 RC1 是直接证据，原超时记录不抹。

第4任文档校准（2026-10-08；有界文档工作——未运行任何命令/git/测试，未修改生产 src、测试、manifest、原 summary/log/snapshot；仅改 README.md/AGENTS.md/dev/m1_typed_replay.md 的 current 段并新增 dev/evidence/manager4/evidence_corrections.md）：
(1) 竞争口径消除：README workspace-generation 段的 "SOURCE-WRITTEN/tests updating, all unrun" 与 "RUN PENDING/nothing has been executed" 是交付时点记录，现明确标注为时间线并由已记录窄绿取代（prepared_problem_contract_tests 61/61 RC0 23.3s、prepared_lifecycle_tests 24/24 RC0 12.4s，scoped、existing 配置、无修复/无放宽——窄绿不泛化 whole M0/M1）；历史 "18/12" registry 数字保留为其时点真时间线，不抹成 21 全跑。
(2) 未证下界撤回：README "真实优化幅度 ≥32% 方向" 措辞撤回——17.844s（特化已预热）与 12.154s（含首次 JIT、未预热）是两个口径不对称的历史观测、无 controlled before/after，不构成优化幅度或下界证据；OOF_fit 15.1s→8.5s 同样只作记录；不添加新性能归因。
(3) 证据权重具名纠正：manager3/standard_entry_4way_summary.txt:7 的 "hit cold LLVM specialization latency" 把初次 12s-deadline RC124 的成因断言为 cold LLVM 特化延迟——RC124（超时/停止）与 LLVM 栈是观测事实，成因 unknown（无 native 编译计量），不得当计量证据或据以推导 compile share；25s-deadline 下 3s 自然退出 RC1 是重跑的直接证据、不反推初次成因。原 summary/log/snapshot 一字未改；具名纠正记录于 dev/evidence/manager4/evidence_corrections.md，README/本文件/dev note 的 current 段均引用之。
(4) 口径钉死（集中声明、与既有记录一致）：21 REQUIRED 登记/文件门通过 ≠ 21 套数值全运行——本任之前已存 numeric 恰为六文件（budget/cache/primal/prepared/lifecycle/replay）；样本残差 9×65 是样本行保真、不称全历史 T×N；浮点 tol 等价非字节/机群恒等。
(5) 两项修复最终运行同步（2026-10-08 DevOps 运行事实已归、log 在 dev/evidence/manager4/；此前「已托付/未回报」「source-written/待验证」均为交付时点时间线、保留为非竞争 current）：两项均窄绿 RC0、不泛化 whole M0/M1、任何新耗时只作观察不作性能对比。(a) ruler_stats 公开入口同源守卫（owner 交付）：numerics.jl verify_ruler_stats_prefix（L301）以本输入前缀流式重算实际消费的 T 行/active 列/每 tau 的 acc/cnt——计数精确硬等、acc 沿既有 64eps/atol=0 舍入口径、NaN/Inf 拒绝；守卫语义是「拒绝与本输入前缀消费统计不一致的缓存」（本轮同形不同振幅反例属其一一类拒绝），检验的是消费统计一致、非来源身份/price 字节指纹——不同历史若产生相同消费统计则接受且 ruler 不变（设计语义）；可观察未来 suffix 不读；_prepare_v1 于 ruler_from_stats 消费前接线（predict.jl:230），initialize_inference 沿既有同源语义对照 state 自身累计统计（incremental.jl:191-203，cnt 精确、同 64eps 口径）；kw/schema 不改；复杂度 O(N_active·|TAUS|·T) 时间、O(1) 标量工作空间、无完整 3D 副本——correctness guard、不称性能优化、不预填零动态 allocation 或吞吐数字；红绿例入 test/history_cache_contract_tests.jl（[R*] testset）。运行证据：history_cache_existing_final.log 五 testset 24+9+15+45+32=125/125 RC0（existing 纯语义/单线程、deadline 45s/elapsed 25s/RSS guard 2048/peak 1034MiB）、无自修无 tol 放宽、源/test 前后 sha256 固定；真树 arch_existing_final.log RC0/7s/guard2048/peak840MiB、Registry 21+1 文件门经真实入口 fail-closed 先后成立——不等于 21 numeric 全跑。(b) replay 验证/消费共用认证 .jls 路径（owner 交付）：path_of_record 唯一路径 owner，所有 consumer 统一经 load_verified——对同一次 read 得到的字节按本进程登记的 SHA/size 重验后才 deserialize(IOBuffer(data))、无第二次读，验证后同路径替换被拒；未登记 lookalike 在任何 read/parse 前拒绝；持久化 provenance 过 assert_source_of_record（历史 unbound '/tmp/prep_inc_t14309.jl' 按构造被拒）——拒绝「只改后缀」与「路径过去验证过就可盲读」两个假修；no-overwrite、manifest、旧 artifact、phase、tol、seed 均不变。运行证据（dev/evidence/manager4/）：artifact_replay_contract_guarded.log 82/82 RC0/4s/peak367MiB；契约测试头注释时点化（仅 test 头 55-57 改历史说明、dev 脚本/运行 source/断言未改）后，DevOps 对最终测试字节重验——artifact_replay_contract_final.log 82/82 RC0/3.1s/peak369MiB/killed0，两 log 为注释改动前/后两个时间点、非竞争数字；新 final_snapshot.txt（16 对象逐路径 full SHA）与 final_summary.txt（进程 0、HEAD 602b897、dirty45、未 commit/reset/clean）已落 manager4；replay_check_guarded.log check RC0/3s/peak501MiB 仍只 4/13（不称全 13）；rewrap_guarded.log RC0/16s/peak1265MiB——真实 .jls 经 load_verified 重 wrap 为 typed PreparedProblem（T14309/N65/P910/F3、owned alive_now 65、ws None/gen0），无 prepare/fit/solve/scenario、无新 artifact；solve_once_refusal_45s.log 自然 RC1/killed=0（deadline 45s/elapsed 15s/peak1213MiB），拒绝先于任何 load/fit、旧输出 sha256 不变（solve_once_refusal_45s_shaset.txt）；15s 拒绝不归因热 IO/cache（未测）、native compile share 未知。失败时间线保留不抹：arch 15s/cache 35s RC124；未获 arch 准入仍跑 cache 属顺序违规；早期 82/check guard0 行为证据非 RSS 合规（合规记录以 guarded log 为准）；RC127 参数误置创建 132-byte ./2048、已移 scoped_run_arg_misuse_2048.log、「零副作用」旧说撤回；solve-once 25s RC124 的 15s 工作窗在 verify/SHA 时中断、不算自然拒绝（后来 45s deadline 才 RC1）；不同 SHA 集合 diff 属构造错、已按路径相同集合补正。model-only/residual-sample/N65 真实 fit 本任未运行。重放脚本历史中硬编码的另一输入路径 /tmp/prep_inc_t14309.jl（无 s、manifest 未登记；其存在与内容格式均未知，.jl 扩展名不能证明它是文本）与已认证的 /tmp/prep_inc_t14309.jls 是两个对象、不得混淆；历史 rewrap RC0 与当前源码未逐时绑定，不得倒推「当时该 .jl 存在或内容已知」、亦不得倒推「RC0 的消费对象就是该未认证 .jl」。
全部 fullsuite/多日/宏基准/GPU/吞吐暂停、命令 ≤60s、旧 HEAD/用户未提交改动保护、panel provenance unknown 保留不变；本段不改动任何路线图内容。

第5任文档与证据引用校准（2026-10-08；纯文档任务——未运行任何命令/git/编译/测试、未差遣 DevOps；manager3/4 全部旧证据、日志、snapshot、manifest、源码与测试一字未改；具名纠正记录 dev/evidence/manager5/evidence_corrections.md，本段与其一致、不复制全文）：
(1) 四向 summary guard 转写纠偏：standard_entry_4way_summary.txt 第 2 行 "(scoped_run 25s, RSS2048)" 与四份 log 原文 scoped_run 尾行不一致——minimal/nested 实为 deadline=12、elapsed=1s（summary 第 3/4 行写 3s，与 log 不一致），bad/empty 为 deadline=25、elapsed=3s，四 log 均 rss_guard=0、rss_peak=0MiB（guard 未生效/未测量）。结论：RSS 合规与 25s/3s guard 配置不能从该 summary 或四 log 推出；红绿 verdict 本身（RC0/RC1、killed=0 自然退出、violations 内容）不受影响，由四份 SHA256 绑定的 log 原文支撑。原 summary/log 一字未改。
(2) final_summary.txt 字段归属纠偏：manager4/final_summary.txt（10 行）与 manager4/final_snapshot.txt（18 行）均未记录 process 0 / HEAD 602b897 / dirty 45 / no commit-reset-clean；第4任段、manager4/evidence_corrections.md 第 67-69 行、README、dev/m1_typed_replay.md 把该组字段归给 final_summary.txt 的引用，实为前任回报的转写、落盘未见。该组字段的真实性属尚未知事实（本任未穷尽全仓库搜索、不下肯定或否定结论）；引用口径改为「前任回报、落盘未见」。manager4/evidence_corrections.md 为旧证据一字不改，其偏差由 manager5 文件具名纠正。本任 current 资源/Git 可引用 manager5/preflight（本任文档 pass 亲读：julia_procs=0、guard 契约完整、HEAD 602b897/dirty 45/未 commit-reset-clean，文件自声明 observed NOW、不倒推 manager4）；manager4 历史口径不变（前任回报、final_summary 未记录）。
(3) ReferenceIngressRepair 源码已交付、运行已验证窄绿（源码 owner 静态交付；DevOps 受控运行；本任文档 pass 亲读 manager5 三份 log 与 ingress_repair_summary.txt 核准、落盘与回报一致无差异）：公开 prepare_reference 入口（原 src/prepare.jl:169 kwargs 裸透传）现为封闭显式 7-keyword 白名单（ridge_alpha/F_folds/ruler_stats/history_cache/alpha_initial/timing/workspace，默认值不变），Julia keyword dispatch 在函数体前拒绝一切其它关键字（三个 internal override 即使 =nothing 也拒，拒绝先于 workspace generation 变化与任何 builder/cache 执行，白名单非黑名单、无静默丢弃）；_prepare_v1 私有路径与 incremental state-owned override 未改（predict.jl 仅接口注释）；现有 REQUIRED test/history_cache_contract_tests.jl 追加 closed public keyword surface testset。运行证据（dev/evidence/manager5/，各一条 scoped 命令）：arch_gate.log RC0 7s/killed0/guard2048/peak840（真实标准入口先行）；history_cache_ingress.log RC0 28s/killed0/guard2048/peak1039、六 testset 210/210=24+9+15+45+32+85（新增 testset 85/85 含 P1/P2 真实负控：三 internal =nothing 必拒、拒绝时 generation 不变且 builder 留痕为空）；primal_prep_ingress.log RC0 16s/killed0/guard2048/peak1029、46/46 私有 route 保留。四对象验证前后 SHA256 固定于 ingress_repair_summary.txt；无自修、无 tol/guard 放宽、existing 单线程纯语义配置（非吞吐口径）。措辞校准：否定注入实测语义是「新接口下无效调用抛错且 ws/builder 副作用为零」，「旧宽接口下这些断言会红」属静态反事实（未实际运行旧源码或 mutant），不得写成旧接口红运行。边界：不称「全部外部必须只走 prepare_reference」——fit_v1/solve 正常公开链仍合法，封闭的只是公开 reference 入口的 internal override 面。集中决策段（context 约束、白名单选择、拒绝 kwargs 裸透传/三名黑名单/调用者自觉的理由、兼容后果、新证据触发重审；链接 SPEC §46/§55/§56、不复制规范、不长 ADR）见 manager5 文件第 3 节。
口径不变：21 REQUIRED 登记 ≠ 21 numeric 全跑；full suite/多日/GPU/吞吐暂停；每命令 ≤60s/RSS2048 硬约束不变；历史红与旧记录保留。本段不改动任何路线图内容。

第6任评审注记（2026-10-08；纯文档任务——未运行任何命令/git/编译/测试、未差遣 DevOps；manager3/4/5 旧证据、log、snapshot、manifest 一字未改；具名注记 dev/evidence/manager6/guard_scope_notes.md，本段与其一致、不复制全文、不制造第二份运行真源）：
(1) 评审事实与口径收窄：prepare_reference 七 keyword 封闭成立（prepare.jl:189-195；三个 internal 通道只在 _prepare_v1，公开链 fit_v1/path_kelly/prepare_incremental 均无注入面，KTrader.jl 不 export _prepare_v1；Julia qualified 调用仍可达 internal 通道——契约封闭非物理封闭）。时序精确口径：只有「未知 keyword 的 dispatch 拒绝」先于 _prepare_v1 的 generation bump（predict.jl:195）；合法 keyword 携带异源 cache/ruler_stats 的守卫拒绝在 bump 之后、首次数学消费之前（verify_history_cache_prefix 接线 predict.jl:206、verify_ruler_stats_prefix 接线 L241）——并非所有错误都在副作用前；failed/partial prepare 已 bump 使同 owner 旧 lease stale 本就是 prepare.jL lease 契约（L33-42、predict.jl:192-196，prepared_problem_contract_tests.jl:151-161 断言），非缺陷；本任前次回报「拒绝先于任何副作用」的过广句以此收窄。守卫面与消费面重合（numerics.jl:199-264/301-357 对 _prepare_v1 消费点 predict.jl:207/216/219/247 与 ruler_from_stats row-T 逐点对应；接受边界=消费统计一致⇒与 no-cache reference 数学结果不变面）。alive_now owned copy（prepare.jl:160）、solve 不读 adj_act（predict.jl:359-361/473）；ResidualOracle 构造全量冻结（residual_oracle.jl:110-112，X_rel 缩为新分配 prefix-sum 表）。workspace slot 疑问已由 Manager 亲读 response.jl 关闭：fit_response_operator 函数体不用 workspace、build_X helper 仅 :path_sums/:design、solve 重建调用（predict.jl:417）不传 workspace——不作为已证漏洞记录，仅留关闭事实。
(2) SixResourceGuardAudit 时间线更新（2026-10-08 本任文档 pass 亲读 bin/scoped_run.sh 全文 596 行核对）：源码 owner 已交付——取消/EXIT/pending-acquire trap（owner_cancel/scope_exit/pending 登记，取消 handler 在 spawn 前安装）、mono_ms /proc/uptime %.0f 单调时钟与显式坏钟 125（start/monitor/cleanup-window/elapsed 四处 fail-closed、无钟回退固定轮次上界）、有界 leader 确认后才 wait（确认不了 LEADER-STILL-ALIVE 125、绝不无界 wait）、终止/确认/收割三清理阶段共享同次 acquire 唯一绝对截止 SCOPE_CleanupDeadlineMs（min(发起时刻+TERM_GRACE+REAP_BUDGET, StartMs+deadline)，幂等、不叠加 5+5+5）；此前「owner 未交付」为交付时点时间线而非竞争 current，真运行完成状态待最终各相。首版缺陷经静态核对 source-written 已修（不声称每一个已运行）：SIG_IGN 跨 exec 继承污染子命令、无界 wait/坏钟隐式算术、32 位 uptime %d 饱和（mawk 等实现 uptime 超约 24.8 天饱和为常数、deadline 永不触发，现 %.0f 全程 double）、独立清理预算叠加。不可捕捉界限保留：SIGKILL/SIGSTOP/宿主断电/内核崩溃无法 trap、owned 组孤儿化、唯一兜底外层更大 deadline 硬杀（调用纪律、两者 ≤60s）；INT 仅当启动时 disposition 默认才可 trap（继承 SIG_IGN 时 POSIX/bash 不允许重设）；不写「任何外部终止原子自愈」；墙钟回拨由唯一 mono 时间源消解。具名 rationale（拒绝单靠外层 killPID——setsid 组孤儿化无人收尾；拒绝墙钟——回拨放宽 deadline；拒绝忽略信号挡窗口——SIG_IGN 跨 exec 污染子 TERM 宽限；拒绝延长 deadline 或把整锅运行当分相验证——60s 硬限启动前拒绝、每失败类需独立 phase 证据）。运行动态事实（当前仅此 scope，本任亲读 log）：basic 真入口 0/7 精确透传、>60 RC2、G1 真实 TERM marker/child_rc=0 而 wrapper RC=124；初期 TERM 临时 harness 错误保留于 dev/evidence/manager6/harness_fault_timeline.txt（127 为 harness 伪影、143 只是 script 自报；attempt2 的 30s 外层超时与 wrapper 自报 2s 并存，harness 源码清理错误可静态指认，但其因果链无 FD/native wait trace 直接计量、按推断/unknown 记——与 LLVM 超时口径一致、不抹失败），干净 parent RC 与 post-cancel bystander 仍待持久 C1 相——不采信「全部最小序列完成/零缺陷」；真实 arch_admission.log RC0/7s/45deadline/RSS2048/peak841/killed0/cancellednone、四 runtime/Gate 对象前后 hash 固定，准入仅此 scope。最终运行已成立（2026-10-08 05:09 final3，本任亲读 final3_closure_summary/negative_control_corrected_record/negative_control_hashes/final_runtime_snapshot/arch_after_final2/negative_control_clock_mono 核对）：最终对象 bin c7048b3be064d5ef60903198a5e8938dca4a7cf4e70df968c323715462817446、test 4b13b81fe3b54373e2acec4b4c093bb35d85659487239b9d7aa45223751adec6（完整 SHA 以实盘为准；05:05 snapshot 的 3a754edf 为其前一版；第7任按 source-of-record——final2_objects.txt/final3_runtime_snapshot.txt/negative_control_hashes.txt——校正此处一字符转写 87339→87239，本任未执行 hash 重算，见 dev/evidence/manager7/evidence_corrections.md）；最终 std arch RC0 7s/RSS2048/peak841；九相 53 项全 RC0（baseline 6 + owner-term/int/hup 各 6 + clock mutant/mono/fail/saturate 5/5/7/6 + selftest 6，unknown/duplicate 为预期 RC1 语义拒绝非测试红）；same-consumer 真注入负控 date→clock-mono 自然 RC1/killed0/16s、终核亲读 negative_control_date_real.log 为 2 Pass/3 Fail——两个时钟目标红之外第三个 no_live_in_group 断言也红（共 3 红；原 corrected record「two target assertions」只指时钟目标，总红数以原 log 为准）、%d→clock-saturate 自然 RC1/killed0/三红，生产 bin 对照绿——此前「phase 未交付/parentRC/post-cancel 待测」current 已被九窄相覆盖、降为时间线；guard_closure_summary 旧 NOT-done 由 final3 取代、旧 file 不改。保留真实失败不洗绿：soft-scope/private process API/.status 早期红、normalleader cleanup window 修复前红、harness 错误（30s 外层超时成因 unknown 不反推）、负控注入错行（sed 命中注释 L47 而非 mono_ms L100）与一次负控外层 124；negative 最初 date RC0 为未注入伪绿（注入工具错误、非 fixture 盲区）、「hold 掩护」说法已撤回。措辞约束：16s 自然 RC1 与 exitcode 0 只证明断言红与无 TIMEOUT——原「sleep 30 正常结束」说法未证、exitcode 0 非自然 wrapper 成功证明；no_live_in_group 红成因未经 native 计量、不作 SIGKILL 机制断言，仅静态说明 harness 在断言后清理（非源码/fixture 盲区）。证明范围是一个 required 文件分相、非 21 全跑；ALL 整跑明确不授权（授权边界非欠账）；模型/全 suite/多日/GPU/吞吐暂停保持；不写任意取消原子自愈/跨机群字节保证/性能达标。补齐（final3_runtime_snapshot.txt / final3_resource_state.txt，本任亲读）：当前仓库直接 SHA 实测 bin c7048b3b / test 4b13b81f 与 final3 as-run 一致、已绑定各 phase 原 log 与两真红记录 full SHA；negroot 现存 date 注入态 badbin、%d 旧态 hash 在 negative_control_hashes.txt；此前仅依 negroot 副本的边界降为补齐前时点记录；当前资源状态（julia_procs=0 / sleep_orphans=0 / terminals 0 / HEAD 602b897 / dirty 45 / git_writes none）以 final3_resource_state.txt 实际内容为准、不转写猜字段。
口径不变：前任 7kw 入口修订/210/46/arch 窄绿仍有效（以 manager5 log 为准，非本任运行）、21 REQUIRED 登记 ≠ 21 numeric 全跑、full suite/多日/GPU/吞吐暂停、每命令 ≤60s/RSS2048 硬约束不变、历史红与旧记录保留。本段不改动任何路线图内容。

第7任文档校准（2026-10-08；纯文档任务——未运行任何命令/git/hash/编译/测试、未差遣 DevOps；manager3/4/5/6 旧证据、log、snapshot（含 guard_scope_notes.md）一字未改；具名记录 dev/evidence/manager7/evidence_corrections.md，本段与其一致、不复制全文）：
(1) scoped_run 坏钟/准入口径撤回：上方第6任段「start/monitor/cleanup-window/elapsed 四处 fail-closed」过广——本任静态核对确认 begin_cleanup_window/mono_elapsed_ms 晚期坏钟仅降级/0 占位、可透传 rc0（假绿路径），start 坏钟 terminate 后 Cleaned=1 而无 leader/owned live group 回收确认，optional --rss-guard 后空 command 可被接纳（假绿准入）；现有 clock-fail 相只测 monitor 循环坏钟（该相 125 成立、以 final2_phase_clock-fail.log 与 final3 各 clock 相 log 为据）。三条为本任静态发现、未动态复现；源码 owner 的有界修复已另行托付、进行中，本任不预填绿。旧九相 53 项 RC0 与最终 std arch RC0 保留其已测范围有效（从未覆盖上述三条边界）；「四处 125 全部闭合/无静默降级已解决」不再作为可引用事实，静态复现与修复后验证留给后续 DevOps。
(2) test/scoped_run_tests.jl full SHA 转写差异：本文件第6任段原引 …59487339… 为单字符手误；source-of-record（final2_objects.txt、final3_runtime_snapshot.txt、negative_control_hashes.txt 三处一致）为 …59487239…，本段引用已按 source-of-record 校正。本任未执行 hash 重算、差异仅为已落盘字符串表面对比；前任「委任者亲跑」误归属已撤回、不再传递；guard_scope_notes.md:164 的同偏差属 manager6 旧证据、原文不动、由 manager7 勘误具名纠正。README.md 与 dev/m1_typed_replay.md 仅短前缀、不含 full hash、无转写错误。
口径不变：21 REQUIRED 登记 ≠ 21 numeric 全跑；full suite/ALL/多日/N65 真实 fit/GPU/吞吐暂停；每命令 ≤60s/RSS2048 硬约束；用户 dirty 工作树保护与 panel provenance unknown 保留。本段不改动任何路线图内容。

第8任评审与验证收口（2026-10-08；评审+受控运行+环境诊断三段工作；评审接纳后的两处 REVISE 义务——四项 REQUIRED 契约测试运行证据缺失、AGENTS.md 状态段滞后于 scoped_run 十六相闭合——驱动本任全部工作）：
(1) 评审结论：独立评审已接纳——structure/simplicity/granularity/language_algorithms/caller_ergonomics/logic_reliability_boundaries 七维 PERFECT；tests_evidence 与 completeness 两维 REVISE，即上述两项义务。
(2) scoped_run 十六相验证闭合确认（前任交付、本任评审核对证据链）：116 项测试 + 25 项 arch gate 检查全 RC0、零活进程残留、fail-closed 125 边界全部成立；bin/scoped_run.sh 交付物 hash c7048b3b…、test/scoped_run_tests.jl 4b13b81f…（以 final3_runtime_snapshot.txt 为 source-of-record）。
(3) 四项 REQUIRED 测试受控运行尝试（conditioned_eb_tests / conditioned_jcore_tests / conditioned_contraction_kernel_tests / relative_support_tests）：各一条独立 scoped 命令（bin/scoped_run.sh 50 <log> --rss-guard=2048 julia --startup-file=no --project=. test/<file>.jl），依次串行未并行；全部 rc=1、elapsed=1s、killed=0、RSS 峰值 321/325/325/326MiB（以各 log 尾行为准）、无 Test Summary——失败发生在 `using KTrader` 触发的预编译阶段，错误原文逐字一致 `ERROR: LoadError: failed to find source of parent package: "FillArrays"`。定性：环境层加载阻断，非测试断言红、非 KTrader 源码数学缺陷证据。
(4) 环境诊断闭环（全部只读）：根因为宿主默认 Julia depot `~/.julia/` 整体不存在（packages/compiled/registries/environments 全缺，家目录无任何 ~/.julia* 备份或替代）；Manifest.toml git tracked 且 clean（dirty 45 的 13 个 modified 文件里没有它），FillArrays 1.17.1 登记完整（uuid 1a297f60…/git-tree-sha1 086b5fbd…）；Project.toml 的 M 状态为用户既有改动、与本次故障无关；JULIA_DEPOT_PATH/XDG_DATA_HOME 均空，Julia 1.12.7 系统级二进制 /usr/bin/julia 独立于 depot；三条隔离加载测试（全局 using FillArrays → ArgumentError: Package FillArrays not found in current path；项目 using FillArrays → 同一 ArgumentError；项目 using KTrader → 复现主故障错误串 failed to find source of parent package: "FillArrays"）钉死最小复现面；时窗 2026-10-08 05:09（manager6 final3 arch RC0 成功，depot 必在）至 06:43（本任首测失败）之间 depot 被外部移除，移除者/方式未知（用户域事实，不做猜测性归因）。修复路径仅二：网络 Pkg.instantiate（未授权资源、拉取 registry 必然超 60s 命令硬限、且有触碰 tracked Manifest 的 resolve 风险）或用户提供 depot 备份（本机无备份源）——均属用户裁决域，本任未修复、DevOps 依授权边界停在修复前。
(5) 证据落盘：dev/evidence/manager8/ 共十份 log（4 运行 + 6 诊断）；运行前后 HEAD 602b897、porcelain 计数 45、无残留 julia 进程、无 commit/reset/clean/stash。
(6) 口径句（保留）：四项 REQUIRED 测试状态为「已登记 registry、受控运行尝试因环境阻断未取得绿证据、根因已钉死为 depot 整体缺失」；21 REQUIRED 登记 ≠ 21 numeric 全跑；full suite/ALL/多日/N65 真实 fit/GPU/吞吐暂停保持；每命令 ≤60s/RSS2048 硬约束不变；本段不改动任何路线图内容、不改动任何源码/测试/manifest/log/snapshot。

第9任评审与边界确认（2026-10-08；评审+受控探查两段工作——本任对第8任工作的评审提交并接纳，REVISE 处置经 DevOps 受控探查确认环境边界；纯文档落盘任务，未运行任何测试、未做任何修复、未差遣任何源码改动）：
(1) 评审结论：本任对第8任工作的独立评审已提交并接纳：八维中 language_algorithms/simplicity/structure/granularity/logic_reliability_boundaries/caller_ergonomics 七维 PERFECT；tests_evidence 与 completeness 两维 REVISE，实质为同一 mission 级欠账——四项 REQUIRED 契约测试（conditioned_eb/conditioned_jcore/conditioned_contraction_kernel/relative_support）的绿证据未取得，属如实记录的未完成而非第8任可归责缺陷。
(2) 评审核对确认：manager8 十份 log 与 AGENTS.md 第8任段数字逐字一致、无编造；scoped_run 十六相 source-of-record（manager6/7 证据链与 final3_runtime_snapshot.txt）真实存在、hash 引用一致。
(3) REVISE 的处置：DevOps 受控探查（分支 b）确认 depot 仍缺失，四条重跑未执行（条件不满足），不做任何修复（网络 instantiate 未授权且必超 60s 命令硬限、本机无备份源，修复属用户裁决域）。探查事实（dev/evidence/manager9/depot_probe.log，时点 2026-10-08 06:55:44 CST）：ls ~/.julia/、ls ~/.julia/packages/F/FillArrays、ls -d ~/.julia* 三段探查原文均为 No such file or directory；scoped_run 尾行 rc=2、elapsed=0s、deadline=30、killed=0、rss_peak=0MiB——rc=2 来自 ls 对不存在路径的退出码，是探查命令如实记录「缺失」的形态、非超时非护栏事件。结论：宿主 depot 仍整体缺失，与 manager8 诊断时点状态一致，无本地变体/备份。
(4) 证据落盘：dev/evidence/manager9/depot_probe.log（本轮唯一新增文件）；运行前后 HEAD 602b897、porcelain 45 不变，无残留 julia 进程，manager3-8 既有证据未触碰。
(5) 口径句（保留）：四项 REQUIRED 测试状态保持「已登记 registry、运行尝试与探查均因 depot 整体缺失而环境阻断、绿证据未取得」；解除路径：用户批准网络 instantiate（需单独授权口径，注意 resolve 触碰 tracked Manifest 的风险）或提供外部 depot 备份；恢复后以与第8任完全相同的四条 scoped 命令（50s/RSS2048）重跑、log 续落 manager9；21 REQUIRED 登记 ≠ 21 numeric 全跑；full suite/ALL/多日/N65 真实 fit/GPU/吞吐暂停；每命令 ≤60s/RSS2048 硬约束；本段不改动任何路线图内容、不改动任何源码/测试/manifest/log/snapshot。

第10任评审与全链收口（2026-10-08；评审+多波受托运行与交叉线修复收口；纯文档段——本任盘点并核对 manager10/manager11 落盘 log，不改任何源码/测试/manifest/log/snapshot）：
(1) 评审：对第9任工作的独立评审已提交并接纳——七维 PERFECT，tests_evidence 与 completeness 两维 REVISE（四项 REQUIRED 契约测试绿证据缺失），该义务驱动本任工作。
(2) depot 恢复：宿主 ~/.julia 重建过程见 dev/evidence/manager11/instantiate.log（registry 与包安装列表）；warmup rc=0、elapsed=2s（manager11/warmup.log）；恢复者与时点归属未逐字确证。
(3) 四项 REQUIRED 计数（manager11 落盘 log）：conditioned_eb 145（66+25+13+22+15+4）、conditioned_jcore 118（113+5）、conditioned_contraction_kernel 1359、relative_support 15（其中 relative_support.log 的 11 pass/1 error 为 200 预算旧态，after_iters 15/15 为对齐态）；manager10/four_required_rerun_summary.txt 记录四者 rc=0（当次运行、无自修）。时序裁定：manager11 多项 log（07:25/07:36）早于 11:22 response.jl 编辑，属编辑前时点证据（manager10/timing_adjudication.txt）；其后编辑引入 (7) 的两个新红，已由 (8) 修复。
(4) 其余 REQUIRED 与 arch gate：manager10 各 log 逐项 rc=0（timed/model_api/admission/primal_prep/prepared_contract/prepared_lifecycle/cli/posterior/residual_flow/reference_isolation/timeblock/ceiling_probes/incremental_budget/history_cache/artifact_replay 等；各自计数与 scoped_run 尾行见对应 log）；scoped_run_tests 整文件 ALL 因总时长超 wrapper 单命令上限，按文件自带 --phase 分相执行（16 相；manager10/scoped_run_phase_*.log）；arch gate 真实树模式经 corrected ROOT 后全 Pass（arch_gate_src.log；ROOT=仓库根触发 fixture-root 键空间不匹配确定性红，见 arch_gate_corrected_root_note.txt）。
(5) full suite 首跑与 round2：首跑 11515 passed / 5 failed / 302 errored（manager10/full_suite.log；红定位于 incremental_tests.jl 的 p.field.observed / Y_target_rel 滞后与 residual_oracle_tests.jl canonical root 未对齐）；修复后 round2 全绿 11822/11822（full_suite_round2.log L75），计数守恒 11515+5+302=11822、零删减。
(6) manager11 交叉线交付（本任盘点）：response.jl 代数重构——conditioned_alpha_cache 的 trace 直接收缩（ridge_constraint_traces）、sigma_stationary_fixed_point 的 J-only frozen_step、jcore 对称核懒装配、fit-local reuse（_conditioned_scalar_cache/_conditioned_scalar_geometry）；配套 conditioned_deadwork_tests.jl 228/228（manager10/conditioned_deadwork.log）与 fit_local_reuse_tests.jl 257/257（244+7+6；manager10/fit_local_reuse.log）；registry 升至 23 项 REQUIRED（manager10/wave8_summary.txt）；EB 局部标定观察值见 manager10/eb_local_calibration.log（非性能 claim）。
(7) round3（23 项 REQUIRED）：12289 passed / 2 errored（full_suite_round3.log L113）——Dense testset 的 .G 字段访问与 companion 的 line search failed（src/response.jl:961）；时序裁定：红在 7b8452… 版本上稳定复现（round3 + 五配置 + rerun，跨实际 6/12 线程）；同类盆地收敛需求 264 步见 manager11/eb_convergence_probe.log（iters=200 THREW，400+ CONVERGED@264）。
(8) 修复：Dense 按生产同式重建 G（deadwork 同式）；:961 fail 前新增确定性收敛候选（Riccati 闭式解 + conditioned_eb_certificate 全证书验收，仅在原必抛错控制流位置执行）；round4 总量守恒印证 12307=12289+18（Dense +14、companion +4）。
(9) round4：12307/12307 全 Pass、0 fail、0 error、rc=0、153s、RSS_PEAK_KB=1942380；五文件 PRE/POST sha256+mtime 一致（src/response.jl 4e206461…、test/conditioned_eb_tests.jl 622d7d93…、test/conditioned_deadwork_tests.jl e725abe7…、test/fit_local_reuse_tests.jl 4fa94034…、test/contract_registry.jl 543f3382…；见 full_suite_round4.snapshot.txt）；porcelain 47、HEAD 602b897 不变。
(10) 多日 backtest（合成数据口径）：20 决策日 rc=0/50s/RSS 2115116KB、60 决策日 rc=0/11s/RSS 1057860KB（multiday_backtest.log、multiday_backtest_60d.log）；wall time 只作观察值、非性能 claim。
口径句（保留）：全部绿证据为单机、当前时点（HEAD 602b897 + dirty 47）、existing 配置一次运行事实，非跨机群/跨时点保证；N65 真实 fit 当前时点证据未取得；GPU/吞吐正式基准不在本任范围（Gate 前置未满足）；本段不改动任何路线图内容。

第11任评审遗留封口（2026-10-08；评审遗留处置——本任定位评审痕迹、封口三条遗留项；未修改任何源码/测试/manifest/log）：
(1) 评审痕迹定位：git refs/localspace/review/ws_*/ 五个 LocalSpace 评审快照（最新 ws_e132a7f4，baseline c41e23a @ 11:31:05；open 3dbb4cb @ 11:04:53；更早四个含 bb17247/b33dc5f/17221fa 等；hash/时点为委派者双路调查所得，本机 .git 不可读、未在只读 log 中复核）；快照 commit message 仅「LocalSpace review snapshot」、git notes 空（wave13_summary.txt T1）——无评审问题清单文本；全仓关键词与外围路径双路检索零命中，清单文本不在本机可读位置。本段以仓库内可指认的三条实质未闭环项为处置对象。
(2) A 项封口：dev/evidence/response_deadwork_20261008/architecture.log 记录标准入口 arch 测试 RSS 超限被杀（guard=2048MiB、observed_peak=2059MiB、rc=124、killed=1）——系预编译与测试同发的叠加；预编译就绪后以 --architecture-root <repo>/src 复跑 rc=0、elapsed=4s、rss_peak=690MiB、killed=0，五组 Test Summary 全 Pass（3/3+2/2+2/2+15/15+3/3=25）（arch_gate_after_delivery.log）。
(3) B 项封口：dev/fit_local_reuse_20261008.md 自认的「末次 hash/进程/Git 组合核对被工具安全层拦截、未执行」由 fit_local_reuse_final_hash.log（2026-10-08 12:01:13 CST）补齐——六文件 sha256+mtime+size（src/response.jl 4e206461…、src/residual_oracle.jl 0b34847b…、test/fit_local_reuse_tests.jl 4fa94034…、test/conditioned_deadwork_tests.jl e725abe7…、test/contract_registry.jl 543f3382…、AGENTS.md fc40a2f1…，该 AGENTS.md hash 为 11:52 时点快照、非本段落盘后）+ porcelain=47 + HEAD=602b897 + julia_procs=0（同刻一次取得）。
(4) C 项封口：scoped_run 三边界四相复验（clock-start 9/9、clock-window-poll 9/9、clock-transient 7/7、command-admission 4/4）合计 29/29、outer rc=0、killed=0（scoped_run_boundary_recheck.log）；对应第7任三条静态边界（start 坏钟确认回收、晚期坏钟不假绿、空 command pre-spawn 拒绝）；clock-elapsed/clock-cleanup 两相本波未重跑（第三波 16 相已有 rc=0 记录）。
(5) 口径句（保留）：以上为单机、当前时点（HEAD 602b897 + dirty 47）一次运行/核对事实，非跨机群保证；评审问题清单文本不在本机可读位置；N65 真实 fit / GPU / 吞吐边界延续第10任段口径（无输入源、Gate 前置未满足）；本段不改动任何路线图内容。

```text
data
geometry
numerics
response
residual_oracle
prepare
predict
incremental
kelly
backtest
broker
live
```

`prepare` 在 `predict` 之前（typed `PreparedProblem` /
`FoldStatistics` / `MacroStatistics` / `MacroFoldBlock`，
`prepare_reference` / `prepare_incremental`；`solve(prepared)` 唯一
装配，`response.jl` 保持拟合数学唯一 owner）。无 `ceiling_probes`：
probes 由 `dev/probes.jl` 独立加载。原"最终 1.0 建议将 probes 移出
默认 module"已执行完毕。

---

## 54. 关键公共函数

数据：

```julia
Bars
download_bars
save_bars
load_bars
load_universe
signal_prices
```

几何：

```julia
ruler
center_of_mass
center_of_mass_decomposition
PrefixRulerStats
build_prefix_ruler_stats
ruler_from_stats
```

响应：

```julia
path_basis_1d
compute_s_perp
build_X_rel_stacked
compute_B_rel_at_t
fit_response_operator
predictive_moments
condition_trace_neutrality
```

准备与求解（M1 新公开 API）：

```julia
PreparedProblem        # typed 准备边界（含 FoldStatistics/MacroStatistics/MacroFoldBlock）
prepare_reference      # reference prepare：完整前缀重算
prepare_incremental    # exact 增量 prepare backend
solve                  # solve(prepared::PreparedProblem)：唯一 full+OOF+fractional 装配
```

`fit_v1` / `_fit_prepared_v1` 为此路径的 forwarder，不是第二实现。
诊断探针（DevProbes）不属于此公开 API——dev 独立加载。

预测：

```julia
fit_v1
generate_scenarios_v1
causal_fractional_posterior
active_universe_indices
```

残差：

```julia
ResidualOracle
dense_oof_residuals
```

增量：

```julia
ExactInferenceState
initialize_inference
advance_exact!
solve_current!
inference_checkpoint
inference_diagnostics
```

Kelly：

```julia
kelly_weights_v1
fast_kelly_solver
clarabel_kelly_solver
path_kelly_v1
path_kelly
```

Backtest：

```julia
backtest_v1
equal_weights_v1
summarize
```

Execution：

```julia
Broker
tradier
rebalance!
tradable
held_weights
target_shares
LiveState
live_step!
settle!
preview_history
```

---

# 第二十部分：Determinism、错误策略与 fallback

## 55. 数学 fallback 与工程 fallback

允许：

- fast numerical path 无法完成时调用**同一个数学问题**的 reference solver；
- incremental resource budget 超过时调用 batch prepare；
- fast Kelly certificate 不通过时调用 Clarabel exact objective。

禁止：

- 更换 objective；
- 改模型 rank；
- 截历史；
- 减少 folds；
- 固定 data-dependent alpha；
- 忽略 posterior uncertainty；
- 为了不报错返回 heuristic weights。

---

## 56. Fail loudly 原则

以下应抛错而非静默继续：

- conditioned EB 未达到 stationarity；
- Kelly certificate 失败且 reference solver 也失败；
- posterior quadrature 到 max scenarios 仍不收敛；
- input data 与 cache 不一致；
- checkpoint 非独立；
- NaN/Inf 进入必须 finite 的矩阵；
- theory invariant 被破坏。

---

# 第二十一部分：数学性质测试（宪法测试）

## 57. Price-scale invariance

对每资产常数 \(c_i>0\)：

\[
P_i(t)\to c_iP_i(t)
\]

目标 weights 应不变（允许数值 roundoff）。

---

## 58. Asset permutation covariance

若资产列按 permutation \(\Pi\) 重排：

\[
P\to P\Pi,
\]

则：

\[
w\to \Pi^Tw.
\]

任何与原始列编号有关的 hidden coordinate bias 都是 bug。

---

## 59. Relative gauge rotation covariance

对 relative gauge：

\[
Q\to QR,
\qquad R^TR=I,
\]

asset-space prediction 与 weights 不得改变。

---

## 60. Dummy inactive asset invariance

append 全 NaN asset：

\[
[P,NaN]
\]

已有资产 prediction / weights 不变。

---

## 61. Structural neutrality

每个 posterior mean / sample support 必须满足：

\[
\operatorname{tr}A_b=0,
\qquad
\operatorname{tr}B_b=0.
\]

测试 threshold 应接近 machine precision / solver tolerance，而不是人为金融阈值。

---

## 62. Synthetic response worlds

至少构造：

- pure mean reversion；
- pure trend；
- right-side confirmation；
- left-side anticipation；
- off-diagonal cross-mode rotation；
- zero-response world。

验证 posterior 恢复正确方向，并在零响应世界收缩到 0。

---

## 63. OOF leakage test

修改某 fold held-out target，确认该 fold 的训练 posterior hyperparameters 不通过不合法路径发生变化。

---

## 64. Quadrature grid refinement

对 fractional d grid：

\[
grid\ refinement
\Rightarrow
p(d),\ w
\text{ converge}
\]

而不是因增加 grid point 改变 prior mass。

---

## 65. Scenario convergence

增加 scenario / quadrature points：

\[
\|w_{2S}-w_S\|_1\to0.
\]

不得用回测收益选择 S。

---

## 66. Reference vs accelerator equality

任何 accelerator PR 必须跑：

```text
reference PreparedProblem
vs
accelerated PreparedProblem
```

逐字段 tolerance comparison。

---

# 第二十二部分：性能工程规范

## 67. 优化顺序是硬纪律

项目统一顺序：

\[
\boxed{
\text{数学正确性}
\rightarrow
\text{数学加速}
\rightarrow
\text{计算机科学加速}
\rightarrow
\text{增量化}
\rightarrow
\text{GPU}
}
\]

前一层没有做尽，不开后一层。

---

## 68. 数学加速示例

允许：

- predictive marginalization \(Gx\) 代替抽整个 G；
- Kronecker identity；
- 14 维 constraint solve；
- primal/dual identity；
- fold sufficient-stat subtraction；
- FFT exact convolution。

不允许以“更快”为理由改变统计模型。

---

## 69. 计算机科学加速示例

- workspace reuse；
- allocation removal；
- BLAS 3；
- contiguous layout；
- SIMD；
- task/BLAS topology；
- precompile；
- deterministic block scheduler；
- mmap / buffering；
- warm starts with same convergence certificate。

---

## 70. 增量化做尽的定义

“做尽增量化”不等于每个对象都递推。

定义是：

> 对所有能保持 exact 且具有正收益的对象完成跨日复用；对剩余对象给出证明/benchmark，说明重算更合理。

例如 OOF residual / fractional likelihood 可能最终仍需要部分重算；只要 profile 证明它不是主要瓶颈，就不应为了“形式上的全增量”继续膨胀代码。

---

## 71. GPU 条件

只有：

1. 数学正确性冻结；
2. 数学 identities 做尽；
3. CPU 工程做尽；
4. exact incremental 做尽；
5. profile 显示剩余热点是 dense LA/FFT 等 GPU 友好 kernel；

才允许开 GPU backend。

GPU 不得成为隐藏算法缺陷的逃生通道。

---

# 第二十三部分：Architecture Reset 规范

## 72. 为什么当前需要整理

当前快照性能/诊断代码已经显著膨胀：

- `incremental.jl` ≈ 867 行；
- `ceiling_probes.jl` ≈ 747 行；
- `residual_oracle.jl` ≈ 331 行；
- `response.jl` ≈ 1080 行；
- `predict.jl` ≈ 646 行。

问题不是 LOC 本身，而是“同一个数学概念有多个 owner”。

---

## 73. 目标目录

建议 1.0 收敛为：

```text
data.jl
geometry.jl
prepare.jl          # PreparedProblem + reference prepare
response.jl         # one posterior solver
predict.jl          # predictive law + scenarios
residuals.jl        # Dense/Lazy common interface
kelly.jl
accelerator.jl      # optional exact incremental prepare backend
backtest.jl         # reference + simple block runner
broker.jl
live.jl
KTrader.jl

dev/
    probes.jl
    profiling.jl
    benchmarks.jl
```

不要求物理文件名完全一致，但 ownership 必须如此清晰。

---

## 74. Production 不允许 instrumentation 改接口

[historical snapshot — 旧快照时点的陈述] 当时 `oof_shadow` /
`OOFCounters` / probe identity registry 是诊断脚手架。M1 施工后
这些设施已从 production 物理删除（factory/counters/身份异常废除，
probes 迁 `dev/probes.jl` 独立加载）；本条保留为历史记录，
当前实现见 §52.12/§53 的同步陈述。

最终 production solver 不应为了 profiler 增加复杂参数。

最终 production solver 不应为了 profiler 增加复杂参数。

监测应通过：

- wrapper；
- profiler；
- debug build；
- optional non-invasive event hooks。

不要“为了观察程序，改变程序的数学 API”。

---

# 第二十四部分：从空目录恢复代码的保姆级顺序

## 75. Step 1 — 建 Julia module 与依赖

创建 `KTrader.jl`：

```julia
module KTrader
using LinearAlgebra, Statistics, Random, Dates
using TimeZones
using PrecompileTools
using Convex, Clarabel
using HTTP, JSON3, CSV
using FFTW, Distributions
...
end
```

先不要写 incremental / GPU。

---

## 76. Step 2 — 写 data.jl

按顺序实现：

1. `Bars`；
2. `carry`；
3. raw→Bars constructor；
4. `signal_prices`；
5. save/load；
6. Yahoo fetch；
7. final-bar date filtering。

测试：

- 上市前 NaN；
- missing bar marking carry；
- signal missing 保持 NaN；
- split-adjusted series shape。

---

## 77. Step 3 — 写 geometry.jl

实现 constants：

```julia
TAUS
BANDS
BANDCOL
WARMUP
```

实现：

1. ruler direct；
2. prefix ruler stats；
3. ruler_from_stats；
4. center-of-mass utilities。

reference direct 与 prefix 必须逐 prefix 一致。

---

## 78. Step 4 — 写 numerics.jl

最少实现：

- `FitWorkspace`；
- `PriceHistoryCache`；
- `RidgeCovariance`；
- `relative_gauge`；
- covariance action；
- FFT workspace；
- principal PSD square root。

先不写复杂 profiler。

---

## 79. Step 5 — 写 path basis

实现：

- `path_basis_1d`；
- `compute_s_perp`；
- `build_X_rel_stacked`；
- `fill_design_matrix!`；
- `compute_B_rel_at_t`。

做 synthetic exact-value tests。

---

## 80. Step 6 — 写 response reference solver

先支持小 N 的 dense reference：

1. macro ridge posterior；
2. relative Matrix-Normal posterior；
3. exact trace conditioning；
4. predictive marginal。

用显式 \(\Omega=V\otimes\Sigma\) 在小系统验证结构化实现。

然后再加 primal/dual / spectral optimization。

---

## 81. Step 7 — 写 reference prepare

从 full history：

1. active universe；
2. log prices / returns；
3. ruler；
4. field embedding；
5. cumulative levels；
6. scalar rulers；
7. Q/P design；
8. full/fold sufficient stats；
9. decision-time basis。

输出 `PreparedProblem`。

---

## 82. Step 8 — 写 solve(PreparedProblem)

1. full response；
2. F folds response；
3. residual provider；
4. macro residual series；
5. fractional posterior；
6. predictive moments；
7. `V1Model`。

---

## 83. Step 9 — 写 scenarios

严格实现随机流、posterior draw、residual bootstrap、fractional scaling、gross return。

先用 dense residual matrix；验证后再替换 lazy residual provider。

---

## 84. Step 10 — 写 Kelly

先 Clarabel reference，再专用 fast solver，再 certificate。

fast solver 必须逐随机问题与 Clarabel objective / KKT 对齐。

---

## 85. Step 11 — 写 sequential backtest

不要一开始写并行。

每一天严格：

```text
prepare reference
solve
scenarios
kelly
advance holdings
```

这就是最终宪法 oracle。

---

## 86. Step 12 — 写 execution

最后接 broker / live，不允许 execution requirement 倒灌理论层。

---

## 87. Step 13 — 数学加速

按 profiling 与代数恒等式逐项替换 reference 内部，但 reference tests 常驻。

---

## 88. Step 14 — CPU 工程加速

只在数学等价测试全部通过后进行。

---

## 89. Step 15 — incremental accelerator

此时才实现 `ExactInferenceState`。

每一个 incremental milestone 都必须能被删除而不影响 reference model。

---

# 第二十五部分：1.0 发布验收清单

## 90. 理论验收

- [ ] price-only information primitive；
- [ ] fixed path law language；
- [ ] direct target weights；
- [ ] full-history theory retained；
- [ ] no score / ranking / Top-K；
- [ ] execution separate；
- [ ] exact Kelly objective。

## 91. 统计验收

- [ ] Matrix-Normal posterior coherent；
- [ ] constrained neutrality support；
- [ ] EB certificate；
- [ ] independent OOF fits；
- [ ] lazy residual = dense reference；
- [ ] fractional posterior grid includes quadrature measure；
- [ ] posterior uncertainty enters scenarios。

## 92. 数值验收

- [ ] primal/dual agreement；
- [ ] FFT/direct convolution agreement；
- [ ] fast Kelly/Clarabel agreement；
- [ ] scenario refinement convergence；
- [ ] covariance floor refinement convergence；
- [ ] fixed-seed reproducibility。

## 93. 不变量验收

- [ ] asset permutation；
- [ ] price scale；
- [ ] relative gauge；
- [ ] dummy inactive asset；
- [ ] trace-zero per band；
- [ ] backtest/reference deterministic equality。

## 94. 架构验收

- [ ] `PreparedProblem` 边界存在；
- [ ] posterior solver 只有一个 production owner；
- [ ] batch reference 可独立运行；
- [ ] incremental 可删除而不伤核心；
- [ ] probes 不污染 production solver signature；
- [ ] accelerator 逐字段对齐 reference。

---

# 第二十六部分：明确禁止事项

## 95. 不得为了回测好看做的事

禁止：

- 因某 rank Sharpe 高而选 rank；
- 因某 scenario count Sharpe 高而选 S；
- 因某 d grid Sharpe 高而选 grid；
- 手调 `ridge_alpha`；
- 后验不确定性额外乘 confidence score；
- 传统 factor mixing；
- Top-K；
- parameter sweep 以回测收益为 theory selection；
- numerical approximation 反向升格成“市场规律”。

---

# 第二十七部分：项目的最简精神模型

如果一个工程师只能记住十句话，记住下面这些：

1. **输入是价格完整历史，不是 factor 表。**
2. **真实规律固定，变化的是历史与 posterior。**
3. **理论直接输出目标 weights。**
4. **有限 Q/P/G 是路径泛函的可计算表示，不是宇宙本体。**
5. **自然板块来自价格算子的谱，不来自标签。**
6. **response operator 给 conditional mean，innovation law 给剩余风险。**
7. **finite evidence 的不确定性必须进入 posterior predictive。**
8. **Kelly 是唯一经济目标。**
9. **reference implementation 是宪法，accelerator 只能算得更快，不能算别的。**
10. **优化顺序永远是：数学正确性 → 数学加速 → 计算机科学加速 → 增量化 → GPU。**

最终统一链：

\[
\boxed{
\mathcal H_t
\rightarrow
\text{multiscale price path}
\rightarrow
\Pi(G,\Sigma,\text{innovation}\mid\mathcal H_t)
\rightarrow
P(r_{t+1}\mid\mathcal H_t)
\rightarrow
\arg\max_w E[\log(R^Tw)]
\rightarrow
w_t^*
}
\]

这条链之外的东西，除非能明确说明属于哪一层、解决什么不可替代的问题，否则默认不进入 KTrader 1.0。

---

# 附录 A：推荐的最终文件职责表

| 文件 | 唯一职责 | 不得拥有 |
|---|---|---|
| `data.jl` | 数据与 signal/marking 语义 | posterior / Kelly |
| `geometry.jl` | path scale 与基础几何 | EB / execution |
| `prepare.jl` | reference `History -> PreparedProblem` | portfolio |
| `response.jl` | posterior response solver | backtest scheduler |
| `residuals.jl` | OOF residual provider interface | theory selection |
| `predict.jl` | innovation + predictive scenarios | broker |
| `kelly.jl` | exact log-growth decision | alpha model |
| `accelerator.jl` | exact PreparedProblem accelerator | 独立 posterior 数学 |
| `backtest.jl` | causal simulation | model definition |
| `broker.jl` | account/orders | prediction |
| `live.jl` | glue | 新理论 |
| `dev/*` | probes/bench/profile | production exports |

---

# 附录 B：推荐最小 CI 套件

每个 PR 至少：

```text
unit/data
unit/ruler
unit/path_basis
unit/conditioned_gaussian
unit/eb_certificate
unit/residual_oracle
unit/fractional_fft
unit/kelly_certificate
property/price_scale
property/permutation
property/gauge
property/dummy_asset
property/trace_neutral
reference/primal_dual
reference/fft_direct
reference/lazy_dense_residual
reference/fast_reference_prepare
reference/kelly_clarabel
integration/backtest_determinism
```

性能 PR 额外提交：

```text
before wall time
after wall time
same-machine configuration
reference differences
allocation delta
hotspot profile
```

没有“数学相同”的证据，不合并性能 PR。

---

# 附录 C：当前状态的一句话判断

当前项目的理论拓扑已经基本稳定；最困难的剩余工作不是继续发明金融结构，而是：

> **把已经明确的数学对象压缩成一个可以长期维护的 reference architecture，再在其外部完成 exact acceleration，尤其是跨日增量化。**

1.0 的成功标准不是“代码最多”，而是：

\[
\boxed{
\text{一个陌生工程师能指出每个公式只有一个 owner，
每个 accelerator 都能被拔掉，
而理论结果仍完整。}
}
\]

---

# KTrader / Path Kelly — 0.0 → 1.0 → 2.0 保姆级路线图

**文档状态：Normative Roadmap**  
**目的：让完全陌生的工程师理解“项目从哪里来、为什么变成现在这样、下一步严格按什么顺序走、什么叫做完成”。**  
**基准：最新仓库快照 `repomix-output(20261007-113017).xml` + 已冻结的理论原则。**

---

# 0. 阅读指南

这份路线图不是“愿望清单”，而是一套**Gate 制度**。

项目统一推进顺序：

\[
\boxed{
\text{数学正确性}
\rightarrow
\text{数学加速}
\rightarrow
\text{计算机科学加速}
\rightarrow
\text{跨日增量化}
\rightarrow
\text{GPU}
}
\]

硬规则：

> **前一层没有做尽，不开后一层。**

“做尽”不等于“什么都优化到理论极限”，而是满足：

1. 已知 correctness 问题全部关闭；
2. 这一层能证明有效的主要优化全部完成；
3. 剩余热点已被 profile 证明属于下一层；
4. 有 reference tests 能确保下一层不改变本层结果；
5. 不再靠拍脑袋开新坑。

版本含义：

- **0.x**：理论与实现共同发现、共同修正的探索期；
- **1.0**：数学定义冻结、reference architecture 完成、可维护；
- **1.x**：在 1.0 数学不动的前提下，依次完成数学加速、CPU 工程、增量化；
- **2.0**：CPU evaluation engine 收口，数学、CPU 工程与增量路径按原门槛验收；GPU 不作为 2.0 发布阻塞（用户于 2026-10-09 明确调整范围）。
- **2.1**：承接 GPU backend、完整数值证书、设备资源生命周期和端到端收益验收；已有 GPU 原语不等于后端交付。

本文中的 0.1/0.5/0.7/0.9 等是**概念里程碑**，不要求与 git tag 一一对应。

---

# 第一篇：0.0 — 项目为什么出现

## 1. 0.0 的原始问题

传统量化系统很容易变成：

```text
momentum factor
+ reversal factor
+ volatility factor
+ sector factor
+ macro factor
+ quality factor
+ hand tuned weights
+ rank
+ top K
+ turnover constraint
```

问题是：

1. 理论对象越来越多；
2. 参数越来越多；
3. 每个现象被一个新 factor 命名，而不是从少数规律统一推出；
4. 最终模型输出一个“score”，仍需要第二套规则决定持仓；
5. backtest 很容易成为超参选择器；
6. 很难区分金融理论、统计 regularization 与纯工程近似。

KTrader 的出发点正相反：

> **像 Maxwell 一样，试图用尽量少的基本对象统一解释趋势、回归、左/右侧、板块结构、板块轮动、风险与仓位。**

---

## 2. 0.0 的第一条革命：不要 score，直接要权重

传统：

\[
\text{history}\to score_i\to rank\to topK\to weight.
\]

本项目改成：

\[
\boxed{
\mathcal H_t\to P(r_{t+1}\mid\mathcal H_t)\to w_t^*
}
\]

Kelly 使理论与仓位直接闭合：

\[
w_t^*=
\arg\max_w
E[\log(1+w^Tr_{t+1})\mid\mathcal H_t].
\]

这一步奠定了整个项目的方向：

- 没有 score ontology；
- 没有排名 ontology；
- 没有 Top-K ontology；
- 没有“alpha 模型”和“portfolio model”两个互不相干的世界。

---

## 3. 0.0 的第二条革命：价格历史是原语

市场对象：

\[
x_i(t)=\log P_i(t).
\]

信息集：

\[
\mathcal H_t=\{x_i(s):s\le t\}.
\]

行业、宏观、板块不作为输入标签；如果存在真正的行业/宏观共同运动，应从价格关系中自然显现。

---

# 第二篇：0.1–0.4 — 从因子思维转向路径思维

## 4. 发现“趋势”和“回归”不应是两个独立 factor

局部 paired path response：

\[
E[\Delta q_h\mid q,p]=a_h(-q)+b_hp.
\]

可以写：

\[
\rho=\sqrt{a^2+b^2},
\qquad
\cos\theta=a/\rho,
\qquad
\sin\theta=b/\rho.
\]

于是：

- mean reversion / trend；
- anticipatory / confirmation；

可以成为**同一个 response direction 的不同象限**，而不是四个手工 factor。

这一阶段确立：

> Q/P paired coordinates 是描述 path geometry 的局部有限 basis，而不是新的 factor bag。

---

## 5. 多尺度不是多因子

定义：

\[
\Delta_\tau x(t)=x(t)-x(t-\tau).
\]

经验 scaling：

\[
|\Delta_\tau x|\sim C\tau^H.
\]

\(H\) 的角色是 scaling / renormalization exponent，而不是另一条 alpha signal。

从此项目使用 dyadic scales：

```text
TAUS  = 1,2,4,...,256
BANDS = 2,4,...,128
```

但明确：

> dyadic grid 是当前数值离散，不是市场真实只存在这些周期。

---

# 第三篇：0.4–0.6 — 自然价格几何与板块

## 6. 板块从标签变成谱结构

曾考虑 path kernel：

\[
C_{ij}(\tau,\sigma)
\]

及作用量 / kernel eigenmodes：

\[
\sum_j\int K_{ij}(\tau,\sigma)\psi_{k,j}(\sigma)d\sigma
=
\lambda_k\psi_{k,i}(\tau).
\]

这给出一个重要思想：

> sector = coherent natural price mode/subspace，而不是 GICS label。

但后续发现：若保留完整 relative \(N-1\) 维空间，数据依赖 eigenvectors 作为计算坐标会带来：

- sign ambiguity；
- degenerate eigenspace rotation；
- fold 间 mode identity；
- bootstrap geometry matching；
- 不必要的 tracking state。

因此后来改成：

\[
\boxed{
\text{固定数值 gauge，算子的谱负责“自然结构”解释}
}
\]

这是项目第一次重要“删掉复杂性”的胜利。

---

# 第四篇：0.6–0.7 — 全历史与非 Markov 原则

## 7. 拒绝把有限 state 当真理

曾经 q/p 被代码称为 state。

理论纠正为：

\[
Q_k[\mathcal H],\qquad P_k[\mathcal H]
\]

是历史的有限 basis functionals。

精确对象仍然是：

\[
m_h[\mathcal H_t]=E[r_{t,t+h}\mid\mathcal H_t].
\]

因此：

> 有限 basis 可以改、可以加密、可以收敛；理论本体不能依赖某一有限 state 选择。

---

## 8. “规律不时变”被正式钉死

项目明确拒绝：

\[
\mathcal G_t
\]

作为真实规律本体。

正确语言：

\[
\mathcal G\text{ fixed},
\qquad
\Pi(d\mathcal G\mid\mathcal H_t)\text{ evolves}.
\]

这一点把 regime switching 从“市场真的换规则”降级为可能的有限模型表达，而不是本体。

---

# 第五篇：0.7 — 接口有形，推断仍是假货

## 9. 0.7 的典型状态

当时已经出现：

- `kelly_weights` exact log objective；
- `path_kelly` 直接 weights；
- live/backtest 共用理论；
- execution 分层。

但 response / posterior 仍存在主要问题：

1. ridge + in-sample \(R^2\) shrink 被称作 Bayesian；
2. scenario 只是 recent demeaned returns + point mean；
3. posterior parameter uncertainty 没真正进入 scenarios；
4. diagonal/scalar response 太弱；
5. basis levels/returns 混用；
6. PC1 被误称 macro；
7. backtest/live portfolio state 语义不完全同构；
8. Kelly solver 失败时曾退化 mean-variance。

这一阶段的核心判断：

> **Kelly是真的，posterior predictive 是假的。**

---

# 第六篇：0.8–0.9 — Response Operator 成形

## 10. Macro 改成 center of mass

取消：

> PC1 = macro

改成：

\[
e_0=\frac1{\sqrt N}(1,\ldots,1)^T,
\]

\[
m=e_0^Tu,
\qquad
u_\perp=(I-e_0e_0^T)u.
\]

这使 macro 定义不再依赖样本 covariance eigenvector。

---

## 11. Response 从 diagonal 升级为 full G

\[
\mu_l=
\sum_{k,b}
[A_{lk,b}Q_{k,b}+B_{lk,b}P_{k,b}].
\]

即：

\[
G_b=A_b+iB_b.
\]

off-diagonal 使 sector rotation 不再是另一个 factor，而是 response operator 本身。

---

## 12. Structural trace neutrality

早期双重 neutralization 被统一成 operator constraint。

最终 V1 选择：

\[
\operatorname{tr}A_b=0,
\qquad
\operatorname{tr}B_b=0,
\quad\forall b.
\]

它是理论 identification constraint，而非 regularization trick。

---

# 第七篇：0.9 — 架构接近，posterior 仍没“真的”

## 13. 0.9 的重要修复

已完成：

- center-of-mass macro；
- strict relative projection；
- complete \(A+iB\) 表示；
- path basis 使用 integrated levels；
- locked-risk Kelly；
- backtest/live 接近同构；
- 不再用 mean-variance fallback。

但核心缺陷：

- relative posterior draw 没进入 scenarios；
- sampled `mu_s` 被算后丢弃；
- coefficient posterior 只做一半；
- trace neutrality 只是 post-hoc mean correction；
- fixed ridge alpha 是策略 knob；
- residual risk zero-fill；
- in-sample residual 过窄；
- geometry uncertainty 未传播。

阶段评价：

> **理论拓扑基本定型；最值钱的工作是让“posterior”这个词真正成立。**

---

# 第八篇：0.95 — 固定 gauge 与数学清场

## 14. Helmert fixed gauge

发现 full \(N-1\) relative space 下 covariance eigenvectors 只是换坐标，不应成为程序追踪的“板块 ID”。

改成固定 Helmert gauge：

\[
Q^T1=0,
\qquad
Q^TQ=I.
\]

自然 sector 改由 \(G\) / operator 的谱解释。

这是重要原则：

> **能靠不变量消灭的问题，不靠更多状态去追踪。**

---

## 15. scalar relative ruler

发现逐 Helmert coordinate scaler 会破坏 gauge invariance。

改成每 band 一个：

\[
s_\perp(\tau).
\]

从而相对空间正交换基不改变模型。

---

# 第九篇：0.96–0.98 — 真正 posterior / OOF / innovation

## 16. Matrix-Normal posterior

明确：

\[
G\mid\Sigma,Y
\sim MN(\hat G,\Sigma,V),
\]

\[
V=(X^TX+\alpha I)^{-1}.
\]

逐步实现：

- primal/dual spectral representation；
- predictive marginal；
- parameter uncertainty；
- relative covariance support。

---

## 17. Trace neutrality 从 projection 升级到 Gaussian conditioning

错误方式：

\[
G\to G-\text{mean diagonal}.
\]

正确方式：

\[
Cg=0
\]

下做 exact Gaussian conditioning。

约束维数只有：

\[
2|BANDS|=14.
\]

所以可把大问题压缩成小 constraint solve。

---

## 18. OOF residual 变真

从训练 residual：

\[
r_{t+1}-\hat\mu_t
\]

升级为：

\[
r_{t+1}-\hat\mu_t^{(-fold)}.
\]

每 fold 必须独立 EB；held-out rows 不得通过 full alpha 泄漏回 train fit。

---

## 19. fractional innovation 恢复 non-Markov risk

对 OOF macro residual 估计 fractional-memory conditional variance posterior。

同时修正：

- absolute scale；
- non-uniform grid quadrature mass；
- FFT exact convolution。

---

# 第十篇：0.98–0.99 — 性能优化开始反噬架构

## 20. 为什么代码突然膨胀

当模型数学趋于正确后，单日 fit 成本变高：

- full response；
- 3 OOF fold response；
- joint EB；
- exact neutrality；
- residual history；
- fractional likelihood；
- scenarios；
- certified Kelly。

6 核笔记本曾达到约：

\[
7.4\text{ days/s}
\]

后来某些 correctness 加强导致回归，再恢复到约：

\[
6.62\text{ days/s}.
\]

1 分钟 10 年的 mission target 大约要求：

\[
42\text{ days/s}
\]

于是工程开始同时尝试：

- workspace；
- primal/dual；
- FFT pool；
- lazy residual；
- timing buckets；
- ceiling probes；
- contiguous scheduler；
- checkpoint；
- incremental regime engine。

结果：

> 性能优化本身逐渐成为比理论代码更复杂的系统。

---

# 第十一篇：Architecture Reset — 1.0 前必须做的事

## 21. 1.0 不是“达到 1 分钟”

1.0 的定义改为：

\[
\boxed{
\text{数学正确}
+
\text{reference architecture 冻结}
+
\text{可维护}
}
\]

10 年 1 分钟是性能 mission，不是理论 release gate。

原因：

如果为了：

\[
5\text{ min}\to1\text{ min}
\]

引入：

- 2000 行状态机；
- 多套 solver；
- probes 渗入核心；
- scheduler framework；

最终系统没人敢改，则项目失败。

---

## 22. 1.0 核心边界：PreparedProblem

必须引入：

\[
\boxed{History\to PreparedProblem\to Solver}
\]

所有 acceleration 只能优化前半段或 solver 的数学等价计算方式。

禁止复制第二套 posterior pipeline。

---

# 第十二篇：Gate M0 — 数学正确性做尽

## 23. 进入条件

当前 0.x 所有已知理论结构已有明确形式。

此 Gate 不允许以速度为理由跳测试。

---

## 24. M0 工作包 A：理论语言冻结

必须写进 SPEC / docstring：

- 固定路径规律；
- full history；
- price-only；
- direct weights；
- execution separate；
- per-band trace neutrality；
- zero embedding 的精确定义；
- Kelly objective。

### M0-A Exit

任何工程师都不会再把：

- `G_t` 当真实时变规律；
- Q/P 当有限 Markov ontology；
- numerical grid 当 theory parameter。

---

## 25. M0 工作包 B：Dense reference Gaussian tests

构造小系统：

\[
N=2,3,4,
\quad B=1,2.
\]

显式形成：

\[
\Omega=V\otimes\Sigma
\]

和完整 constraint matrix \(C\)。

比较结构化实现：

- conditioned mean；
- conditioned covariance；
- log evidence；
- predictive marginal。

### M0-B Exit

relative error \(\le 10^{-9}\sim10^{-8}\)（按条件数设合理 tolerance）。

---

## 26. M0 工作包 C：EB certificate

任何 EB solver 返回必须带：

- alpha KKT / derivative；
- Sigma stationarity；
- SPD/support 检查；
- bound status；
- line-search status。

不得：

> line search failed → return current point。

### M0-C Exit

所有 production fit 要么 certified，要么 loudly fail。

---

## 27. M0 工作包 D：OOF 隔离

每 fold：

\[
\alpha^{(-f)},\Sigma^{(-f)},G^{(-f)}
\]

只能使用 train rows。

测试：修改 held-out row 后 train posterior 不发生非法变化。

---

## 28. M0 工作包 E：Residual reference

保留 dense OOF residual 实现做 test oracle。

Lazy ResidualOracle 必须逐 cell / row / macro series 对齐。

---

## 29. M0 工作包 F：Fractional posterior

测试：

- direct convolution vs FFT；
- grid refinement；
- cell-width prior mass；
- absolute volatility scale；
- causal indexing。

---

## 30. M0 工作包 G：Kelly

随机 scenario problems：

- fast solver；
- Clarabel solver；
- certificate；

三者一致。

---

## 31. M0 工作包 H：宪法性质

必须全部 CI：

- price-scale invariance；
- permutation covariance；
- gauge covariance；
- dummy inactive asset；
- trace neutrality；
- deterministic seed；
- no-lookahead。

---

## 32. M0 禁止事项

M0 期间禁止：

- 开 GPU；
- 新增 incremental algebra；
- 再做复杂 scheduler；
- 新 feature；
- 新 factor；
- 调 Sharpe 选参数。

---

# 第十三篇：Gate M1 — Architecture Freeze

## 33. 目标

把当前多 owner 收敛成：

```text
History
  -> reference prepare
  -> PreparedProblem
  -> one solve
  -> V1Model
  -> scenario
  -> Kelly
```

---

## 34. M1 工作包 A：PreparedProblem

定义 typed struct。

字段按数学语义分组：

- geometry；
- full sufficient stats；
- fold sufficient stats；
- decision features；
- residual metadata。

不要塞 profiling state。

---

## 35. M1 工作包 B：一个 solver owner

`response.jl` / `predict.jl` 之间确定唯一边界。

`incremental.jl` 不允许独立拟合 posterior。

---

## 36. M1 工作包 C：probes 迁出 production

当前 `ceiling_probes.jl` 已成为大型诊断框架。

迁移：

```text
dev/probes.jl
bench/
```

删除 production：

- `oof_shadow`；
- probe-only counter plumbing；
- identity registries；
- export probes。

---

## 37. M1 工作包 D：Residual provider 接口

统一：

```text
AbstractResidualProvider
  DenseResiduals
  LazyResiduals
```

消费者不关心底层实现。

---

## 38. M1 Exit

1. 删除 accelerator，reference 仍完整；
2. 删除 probes，模型结果不变；
3. `solve(PreparedProblem)` 唯一；
4. 一个陌生工程师能在 30 分钟内画出完整数据流。

完成后打：

\[
\boxed{1.0}
\]

---

# 第十四篇：1.0 的定义

## 39. Path Kelly 1.0

1.0 不承诺极限速度。

它承诺：

### 理论

\[
\mathcal H_t
\to
P(r_{t+1}|\mathcal H_t)
\to
w_t^*.
\]

### Bayesian

posterior uncertainty 真实进入 predictive。

### OOF

创新分布不是 in-sample optimism。

### Kelly

exact log objective。

### Architecture

reference 可维护；accelerator 可拔掉。

### Tests

数学不变量可执行验证。

---

# 第十五篇：Gate A1 — 数学加速做尽

## 40. 定义

数学加速：

> 用恒等式/边缘化/充分统计把同一个数学对象算得更便宜。

输出应在数值 tolerance 内与 reference 一致。

---

## 41. A1.1 Decision-time marginalization

不要抽整个：

\[
G\in\mathbb R^{N\times14N}.
\]

只计算当前：

\[
Gx_t.
\]

这是已完成/必须保留的关键优化。

---

## 42. A1.2 Constraint low-dimensionalization

trace constraints 只有 14 个。

所有：

\[
C\Omega C^T
\]

必须通过 Matrix-Normal structure 计算，而不是 materialize full covariance。

---

## 43. A1.3 Primal / dual

自动在：

\[
\min(n,P)
\]

空间工作。

完整验证两 branch 一致。

---

## 44. A1.4 Fold sufficient statistics

full stats = fold stats sum。

train = full - eval fold。

禁止重复扫描 train history。

---

## 45. A1.5 Fractional FFT

保持 direct reference，仅 production 用 FFT。

---

## 46. A1.6 Residual lazy evaluation

Fractional likelihood 只需要 macro scalar residual series；scenario 只需要采样 rows。

不要为了接口方便物化全部 \(T\times N\)。

---

## 47. A1.7 证书驱动 early stop

允许 warm start / fewer iterations，条件是达到同一个 mathematical certificate。

禁止固定 iteration 缩水。

---

## 48. A1 Exit

建立一张清单：所有高阶代数操作都问过：

- 能否解析边缘化？
- 能否用 sufficient stats？
- 能否用 Kronecker？
- 能否用低维 constraint？
- 能否 primal/dual？

确认主要答案都已落实。

---

# 第十六篇：Gate C1 — 计算机科学加速做尽

## 49. 进入条件

数学表达冻结。

任何 performance diff 都能在 reference 层验证。

---

## 50. C1.1 Allocation elimination

目标 steady state：

- 大矩阵反复 reuse；
- 无每日巨型 GC；
- no hidden copies；
- BLAS leading dimensions 正确。

当前 `FitWorkspace` 已是雏形。

---

## 51. C1.2 BLAS 结构

优先 Level-3：

- GEMM；
- SYRK；
- Cholesky；
- symmetric eigen。

避免 Julia 标量循环包围大 dense algebra。

---

## 52. C1.3 Thread topology

必须实测：

```text
6 tasks × BLAS1
3 tasks × BLAS2
2 tasks × BLAS3
1 task × BLAS6
```

笔记本共享 cache/memory bandwidth 常使“核数=任务数”不是最优。

---

## 53. C1.4 Warm start

允许：

- alpha；
- Sigma；
- Kelly；
- iterative eigensolver（若最终使用）。

但必须收敛到同 certificate。

---

## 54. C1.5 Scheduler 简化

先简单 contiguous blocks。

避免为了榨 5% 性能建立复杂任务框架。

若 RAM 可承受，优先内存结果；若必须 spill，应使用简单大块文件/mmap，而不是大量 per-date serialization files。

---

## 55. C1.6 Precompile / FFT plans / caches

这些都属于 implementation detail。

必须保证缓存不影响 deterministic outputs。

---

## 56. C1 Exit

Profile 上：

- allocation 不再主导；
- scheduler 不再主导；
- obvious duplicate computations 清零；
- 热点已经是模型本身必须做的矩阵求解/历史重算。

此时才允许进入增量化。

---

# 第十七篇：Gate I0 — 增量化研究的正确姿势

## 57. 为什么增量化是最难的计算问题

每天只增加一行 price，不代表内部只增加一行：

1. ruler 改变；
2. 所有历史 standardized field 改变；
3. ragged projector 随历史日不同；
4. Q/P basis 使用窗口和 cumulative path；
5. fold boundary 会移动；
6. OOF models 改变；
7. OOF residual 历史会改变；
8. fractional posterior 因 residual 改变；
9. active universe 会扩维。

所以普通：

\[
S_{t+1}=S_t+x_tx_t^T
\]

是不正确的。

---

## 58. 增量化的真正目标

寻找：

\[
\boxed{
\text{metric-independent raw state}
+
\text{today-dependent exact contraction}
}
\]

使：

\[
Prepared_t=C(State_t,\theta_t),
\]

\[
State_{t+1}=U(State_t,new\ day).
\]

---

# 第十八篇：Gate I1 — 完整 panel 增量化

## 59. I1 scope

先禁止 ragged：所有 active assets 每天都有 return。

目的：验证核心 metric factorization，不同时解决所有问题。

---

## 60. I1 数学

今日 metric：

\[
D_t=\operatorname{diag}(1/s_1(t)).
\]

relative：

\[
e_s=P_0D_t r_s.
\]

由于 temporal filters 线性，可维护 raw filtered sufficient stats，再用今日 D contraction。

---

## 61. I1 Exit

逐日：

\[
Prepared^{inc}=Prepared^{ref}.
\]

测真实 speedup。

若完整 panel 都无法获得明显收益，先修增量算法，不进入 ragged。

---

# 第十九篇：Gate I2 — 固定 active universe + arbitrary ragged masks

## 62. I2 数学核心

每历史日 observation mask：

\[
m_s.
\]

\[
P_s=M_s-\frac{m_sm_s^T}{n_s}.
\]

\[
e_s(d)=P_s\operatorname{diag}(z_s)d=A_sd.
\]

这证明 ragged 仍然严格线性于今日 metric。

---

## 63. I2 两种 exact 实现候选

### 方法 A：mask-run / regime

相同 mask 连续区间聚合。

window 跨 regime 的 rows exact materialize。

优点：常见稳定 panel 快。

缺点：mask 高频变化时复杂。

### 方法 B：operator-row aggregation

直接存 metric-independent \(A_s\) / filtered operator sufficient statistics。

更一般，但内存结构和 contraction 设计更难。

---

## 64. I2 决策原则

先用真实数据测：

- unique masks；
- stable run median / p90；
- transition coverage；
- boundary row ratio；
- memory cost。

不是凭直觉决定。

---

## 65. I2 Exit

无 permanent ragged fallback。

任何 route 到 batch 只能是：

- numerical failure；
- explicit resource budget；

并且 route 不改变输出。

---

# 第二十篇：Gate I3 — Active universe 扩维

## 66. 问题

IPO / first valid return：

\[
N\to N+1.
\]

旧实现曾全历史 replay，这是不必要的 O(T) 成本。

---

## 67. Exact embedding

新资产上市前，在 zero-embedded field 中该坐标历史贡献为 0。

因此：

- vectors 插零；
- matrices 插零 row/col；
- masks 扩维；
- old statistics permutation/embed；

即可 exact 保留历史。

---

## 68. I3 Exit

IPO day 不触发 full-history replay。

reference equality 通过。

---

# 第二十一篇：Gate I4 — Fold state 增量化

## 69. 问题

fold ranges 随：

\[
q=\lfloor n/F\rfloor
\]

变化。

旧 row 偶尔会移动 fold。

---

## 70. Exact update

维护：

- full raw stats；
- F fold raw stats；
- row membership。

新 row append；边界移动 row 做：

\[
-fold_{old}+fold_{new}.
\]

---

## 71. I4 Exit

正常日期不再 full recompute fold Grams。

boundary move 成本与移动 row 数成正比。

---

# 第二十二篇：Gate I5 — OOF / Residual / Innovation 增量化

## 72. 这是最后做的一层

原因：OOF model 今天变化时，旧 residual：

\[
\epsilon_s^{(t)}
\]

也会变化。

所以不能简单 append residual。

---

## 73. I5 先问：值得吗？

如果 lazy residual + FFT 已经只占 5% wall time：

\[
\boxed{
不要继续增量化它。
}
\]

“做尽”意味着做 cost-benefit 证明，而不是形式主义。

---

## 74. 可选 exact 思路

- fold response change 的 low-rank sensitivity；
- lazy residual only for required rows；
- macro residual scalar contraction；
- incremental fractional likelihood only if residual law admits exact update。

每一个都必须 reference equality。

---

# 第二十三篇：I-Gate 总 Exit — 什么叫“增量化做尽”

## 75. 最终判据

对每个当前主要 wall-time bucket，必须归类：

1. 已 exact incremental；
2. 无法 exact reuse，有数学证明；
3. exact reuse 可做但收益小于复杂性，有 benchmark；
4. 属于 per-day unavoidable solve。

不能留下“也许还能缓存”的巨大未知。

---

## 76. 代码收敛条件

最终只允许两个 prepare backend：

```text
ReferencePrepare
IncrementalPrepare
```

不得保留四代 experimental engine。

每完成一个阶段，删除旧 prototype。

---

# 第二十四篇：Gate G — GPU（最后）

## 77. 进入条件

只有 I-Gate 完成以后。

必须先 profile 证明剩余热点主要是：

- GEMM/SYRK；
- eigen；
- Cholesky；
- FFT；
- batched transforms。

若剩余热点仍是 branch-heavy state machine，GPU 不会救。

---

## 78. GPU 原则

GPU 只改变设备：

\[
\text{same matrices}
\to
\text{same algebra}
\to
\text{different backend}.
\]

不能借 GPU：

- 改 Float64 为低精度但不做收敛；
- 截 rank；
- 减 folds；
- 减 posterior；
- 降 scenario accuracy。

---

## 79. Vulkan 的位置

不要一开始 raw Vulkan。

先抽象 backend interface。

根据硬件选择成熟 Julia GPU stack；若必须 cross-vendor 且其它 backend 不满足，再评估 Vulkan/Lava。

但这属于最后阶段，不影响 1.x 工作。

---

# 第二十五篇：2.0 的定义

## 80. Path Kelly 2.0

2.0 不是“加更多金融 feature”。

2.0 是：

\[
\boxed{
\text{1.0 数学完全不动，
但 evaluation engine 已完成所有值得做的 exact acceleration。}
}
\]

其特征：

### 80.1 数学

与 1.0 reference 一致。

### 80.2 架构

reference / accelerator 可替换。

### 80.3 CPU

数学 identities、内存、线程、incremental 已做尽。

### 80.4 GPU

按用户于 2026-10-09 明确的版本范围：GPU 不阻塞 2.0，GPU backend 放到 2.1。

2.0 仍须满足 CPU、数学一致性和维护性验收；不能用这次范围调整放宽证书、
删掉失败测试或把未执行的 CPU 对照记为通过。2.1 再验收 GPU 的原始精度、
完整后验/场景/Kelly 证书、设备生命周期与端到端收益。现有 GPU 原语和日志
保留为 2.1 的开发证据，不由 CPU 发布命令加载或执行。

### 80.5 维护性

任何 performance backend 坏掉时，可切回 reference，而不是整个项目瘫痪。

---

# 第二十六篇：性能 mission 与现实判断

## 81. 历史基准

6 核笔记本曾测：

\[
7.4\text{ days/s}
\]

后续约：

\[
6.62\text{ days/s}.
\]

10 年约 2500 trading days。

1 分钟 mission：

\[
\sim42\text{ days/s}.
\]

---

## 82. Mission 不是 release gate

性能目标用于推动：

- profile；
- algebra；
- system design。

但不能使项目违背：

\[
\text{正确性 > 速度}.
\]

若最终 exact CPU/incremental 是 20 days/s，而 42 需要 GPU，则在 CPU 2.0
验收后按顺序于 2.1 做 GPU；该 mission 不反向成为 2.0 的 GPU 发布门槛。

若 42 需要破坏数学，则 mission 必须失败，而不是数学失败。

---

# 第二十七篇：工程师每天怎么工作

## 83. 每个 PR 必须先声明层级

PR description 第一行写：

```text
Layer: Math Correctness
```

或：

```text
Layer: Math Acceleration
Layer: CS Acceleration
Layer: Incremental
Layer: GPU
```

禁止一个 PR 同时横跨三层。

---

## 84. 性能 PR 模板

必须报告：

### Before

```text
machine
Julia version
BLAS
threads
dataset
days/s
wall time
allocations
hot buckets
```

### Correctness

```text
PreparedProblem max diff
posterior diff
scenario diff
weight diff
invariants
```

### After

同样指标。

### Complexity

新增 LOC / 删除 LOC。

### Rollback

如何一键切回 reference。

---

## 85. 禁止“先写很多再测”

每个性能想法必须最小 proof-of-concept：

1. 证明数学恒等；
2. benchmark 单一 hotspot；
3. 确认至少 20–30% 局部收益；
4. 再产品化。

否则不允许新增 500 行 subsystem。

---

# 第二十八篇：代码膨胀控制政策

## 86. 净复杂度预算

性能功能进入 production 时：

> 新增一个 backend，必须删除相应旧 prototype。

建议软规则：

- 新增 300 LOC performance code，目标删除/合并至少 200 LOC 旧实验路径；
- probes 不进入 production exports；
- 每个数学概念一个 owner。

---

## 87. 什么值得复杂

值得：

- exact conditioned posterior；
- independent OOF；
- certified Kelly；
- reference vs accelerator dual implementation。

因为这些维护的是数学真实性。

不值得：

- 为 3% speedup 写 400 行 scheduler；
- 为了一个 benchmark 写第二套 instrumentation protocol；
- 多个性能 backend 同时长期存在。

---

# 第二十九篇：阶段性“不要做”清单

## 88. 1.0 前不要做

- GPU；
- 新 alpha features；
- geometry posterior 新模型；
- 高频化；
- transaction-cost utility；
- 新资产类别；
- deep neural net；
- RL；
- 分布式集群。

---

## 89. 1.0 后、数学加速没做尽前不要做

- incremental rewrite；
- device backend。

---

## 90. CPU 工程没做尽前不要做

- symbolic ragged compiler；
- GPU。

---

## 91. incremental 没做尽前不要做

- Vulkan；
- multi-GPU；
- cluster。

---

# 第三十篇：从当前仓库出发的具体 30 步施工单

## 92. 第 1–5 步：收敛 1.0 architecture

1. 从最新快照打稳定 branch；
2. 固定 reference numerical outputs；
3. 定义 `PreparedProblem`；
4. 让 `_prepare_v1` 转成 reference constructor；
5. 让 `_fit_prepared_v1` 只接 PreparedProblem。

---

## 93. 第 6–10 步：清理诊断污染

6. 将 `ceiling_probes.jl` 移到 dev；
7. 删除 production `oof_shadow` 参数；
8. 删除 probe-only `OOFCounters` plumbing；
9. 保留普通 timing buckets 即可；
10. 验证输出完全不变。

---

## 94. 第 11–15 步：Residual 接口化

11. 定义 residual provider abstract contract；
12. dense provider；
13. lazy provider；
14. all consumers 只用 interface；
15. dense/lazy property tests。

---

## 95. 第 16–20 步：数学 correctness freeze

16. conditioned Gaussian dense tests；
17. EB certificate tests；
18. OOF isolation；
19. fractional refinement；
20. invariance suite 全绿。

此时：**1.0 RC**。

---

## 96. 第 21–24 步：数学加速审计

21. 列出所有大矩阵 expression；
22. 对每个做 algebraic cost review；
23. primal/dual / marginalization / low-dimensional constraints 做尽；
24. 打 profile。

---

## 97. 第 25–27 步：CS 加速

25. allocation / BLAS / thread topology；
26. scheduler 简化；
27. warm starts / precompile / FFT plan。

---

## 98. 第 28–30 步：incremental 重启

28. 将当前 `incremental.jl` 视为研究 prototype，而不是不可修改遗产；
29. 按 I1→I5 逐阶段，每阶段 reference equality；
30. 每阶段完成后删掉上一代 prototype。

之后再讨论 GPU。

---

# 第三十一篇：风险登记表

## 99. 风险 1：性能目标反客为主

症状：

- 代码翻倍；
- 理论测试没增加；
- days/s 只提高 10%。

处理：回滚性能分支，重新过 Gate。

---

## 100. 风险 2：reference 也被优化到不可读

reference 必须故意简单。

Fast code 可以复杂；oracle 不能复杂。

---

## 101. 风险 3：instrumentation 侵入核心

当前已发生过。

处理：probes 外置，核心函数签名保持数学语义。

---

## 102. 风险 4：incremental 与 batch 漂移

处理：PreparedProblem 逐字段 CI comparison。

---

## 103. 风险 5：工程 fallback 偷偷变模型

任何 fallback 必须有：

\[
\text{same objective / same posterior / same rows}.
\]

否则禁止。

---

## 104. 风险 6：EB optimization 重新成为性能黑洞

必须 profile：

- alpha eval count；
- Sigma iteration；
- line search；
- certificate。

warm start 只允许减少求解工作，不改变 optimum。

---

# 第三十二篇：最终哲学

## 105. 0.0 的问题

“怎样找到更多 alpha？”

## 106. 1.0 的回答

不是更多 alpha，而是：

\[
\boxed{
\text{price path}
\to
\text{response + innovation posterior}
\to
\text{Kelly}
}
\]

## 107. 2.0 的问题

“怎样让同一个理论更快，而不是让理论变成工程近似？”

## 108. 2.0 的回答

\[
\boxed{
\text{reference truth}
+
\text{exact algebra}
+
\text{exact systems optimization}
+
\text{exact incremental reuse}
+
\text{必要时 device acceleration}
}
\]

---

# 附录 A：版本含义速查

| 阶段 | 主要问题 | 核心成果 | 退出条件 |
|---|---|---|---|
| 0.0 | 因子袋/score 不统一 | price history → Kelly | direct weights 理念成立 |
| 0.3 | 趋势/回归分裂 | paired path response | Q/P framework |
| 0.5 | 板块标签 | natural price geometry | sector 作为谱结构 |
| 0.7 | posterior 名不副实 | exact Kelly + interfaces | 发现 posterior fake |
| 0.9 | operator 结构 | full G, COM macro | response topology 稳定 |
| 0.95 | eigenvector tracking | fixed gauge | gauge complexity 删除 |
| 0.97 | Bayesian/OOF/risk | conditioned posterior + OOF + fractional | 数学接近闭合 |
| 0.99 | performance pressure | lazy residual / incremental research | 发现架构膨胀 |
| 1.0 | 可维护真系统 | reference + PreparedProblem + one solver | 数学/架构冻结 |
| 1.x | exact acceleration | math→CS→incremental | 每 Gate 做尽 |
| 2.0 | CPU evaluation engine 收口 | exact CPU engine、可替换增量路径 | CPU/reference 验收及维护性闭环；GPU 不阻塞 |
| 2.1 | GPU 端到端加速 | 完整 GPU backend，不只是原语 | 原数学证书、设备生命周期及实际收益验收 |

---

# 附录 B：每个 Gate 的一句话

### M0 数学正确性

> “它到底算的是不是我们声称的 posterior / Kelly？”

### M1 架构冻结

> “一个数学概念是不是只有一个 owner？”

### A1 数学加速

> “同一个式子能不能通过恒等式少算几个数量级？”

### C1 计算机科学加速

> “同样的 FLOPs 能不能让 CPU 更高效地执行？”

### I1–I5 增量化

> “昨天已经算过的历史，今天哪些可以严格复用？”

### GPU

> “所有算法问题解决后，剩余 dense compute 是否值得换设备？”

---

# 附录 C：工程师交接时必须能回答的 20 个问题

1. 为什么真实 law 不是 time-varying？
2. 为什么 full history 是理论原语？
3. Q/P 是什么，不是什么？
4. `s1` 的角色是什么？
5. missing return 和 zero-embedded field 有什么区别？
6. 为什么 macro 不是 PC1？
7. 为什么 fixed gauge 不损失自然 sector？
8. full \(G=A+iB\) 怎么表示轮动？
9. trace neutrality 为什么是理论 constraint？
10. 为什么必须 condition posterior 而不是减 trace mean？
11. 为什么 OOF residual 必须独立 EB？
12. fractional grid 为什么要乘 \(\Delta d\)？
13. predictive scenarios 里 epistemic uncertainty 在哪里？
14. 为什么不能 sample recent returns + point mean 就叫 posterior？
15. Kelly fallback 为什么不能用 mean-variance？
16. `PreparedProblem` 的边界是什么？
17. reference 与 accelerator 如何比较？
18. 为什么增量化不能简单 rank-one update？
19. 为什么 ragged 仍可写成 \(e_t(d)=A_td\)？
20. 为什么 GPU 必须最后做？

如果新工程师能正确回答这 20 个问题，并能让 reference CI 全绿，他就已经真正理解项目，而不是只会运行代码。

---

# 附录 D：路线图的最终底线

无论未来性能目标如何变化，以下东西永远不能为了速度牺牲：

\[
\boxed{
\begin{aligned}
&\text{完整因果信息集}\cr
&\text{真实 posterior uncertainty}\cr
&\text{独立 OOF}\cr
&\text{structural neutrality}\cr
&\text{exact log-Kelly objective}\cr
&\text{reference 可验证性}
\end{aligned}
}
\]

真正可以变化的是：

- 表示；
- 算法；
- 缓存；
- 数值积分；
- 线程；
- 增量 state；
- 计算设备。

项目最终目标不是“写一个很快的回测程序”，而是：

> **让一套少数原则统一的价格路径理论，在统计上诚实、数值上可证、工程上可维护，并最终达到足够快的研究迭代速度。**

这就是从 0.0 到 2.0 的完整方向。

---

# KTrader 开发守则（Development Constitution）

> 本文件是 KTrader 项目的强制开发纪律。  
> 它不是“建议”、不是“风格指南”、不是“最佳实践清单”，而是所有设计、修改、优化、测试、基准、重构、合并和版本推进都必须遵守的工程宪法。
>
> 任何实现若违反本守则，即使：
>
> - 能跑；
> - 回测更快；
> - Sharpe 更高；
> - 测试暂时通过；
> - 代码已经写了很多；
> - 工期很紧；
> - “第一版以后再整理”；
>
> 也不得视为合格实现。
>
> 本项目的核心原则：
>
> \[
> \boxed{
> \text{先理解}
> \rightarrow
> \text{先证明}
> \rightarrow
> \text{先静态穷尽}
> \rightarrow
> \text{再微基准}
> \rightarrow
> \text{再有限测试}
> \rightarrow
> \text{再大规模运行}
> }
> \]
>
> 以及：
>
> \[
> \boxed{
> \text{后 Gate 破坏前 Gate}
> \Rightarrow
> \text{立即回退阶段}
> }
> \]
>
> 最后：
>
> \[
> \boxed{
> \text{优雅重构优先于赶工补丁}
> }
> \]

---

# 0. 本守则的优先级

本守则对以下行为具有约束力：

- 阅读代码；
- 修改代码；
- 新增模块；
- 删除模块；
- 重构；
- 数学推导；
- 数值实现；
- 性能优化；
- 增量化；
- 并行化；
- GPU 化；
- 测试；
- benchmark；
- profiler；
- backtest；
- live glue；
- debug；
- CI；
- code review；
- PR 合并；
- release gate；
- 版本号升级。

若本守则与某个临时 issue、TODO、实验脚本、旧注释、旧 benchmark 结论冲突：

\[
\boxed{\text{以本守则为准}}
\]

若本守则与 `SPEC.md` 在数学定义上冲突：

\[
\boxed{\text{以最新 SPEC 的数学定义为准}}
\]

但工程执行顺序、测试准入、性能实验纪律和 Gate 回退规则以本文件为准。

---

# 1. 第一原则：优先阅读和修改代码，禁止“先跑再说”

## 1.1 默认行为

任何工程任务开始时，默认顺序必须是：

1. 阅读相关源代码；
2. 阅读调用链；
3. 阅读数据结构；
4. 阅读数学注释与实际实现；
5. 静态推导控制流；
6. 静态推导数据流；
7. 静态推导复杂度；
8. 检查不变量；
9. 检查是否已有重复实现；
10. 检查是否有更简单的重构；
11. 修改代码；
12. 再考虑是否需要运行。

禁止把“跑一下看看”作为理解代码的替代品。

---

## 1.2 静态手段未穷尽，严禁执行测试

强制规则：

\[
\boxed{
\text{如果尚未穷尽静态手段，严禁执行测试}
}
\]

“静态手段”包括但不限于：

### 代码级

- 阅读完整函数；
- 阅读所有 caller；
- 阅读所有 callee；
- 阅读类型定义；
- 阅读默认参数；
- 阅读分支条件；
- 阅读 fallback；
- 阅读异常路径；
- 阅读缓存生命周期；
- 阅读 mutation；
- 阅读线程所有权；
- 阅读 workspace 复用；
- 阅读随机数来源；
- 阅读 seed 规则；
- 阅读序列化/反序列化路径；
- 阅读 batch/reference path；
- 阅读 fast/incremental path。

### 数学级

- 手推公式；
- 检查矩阵维度；
- 检查约束是否真在 posterior support；
- 检查概率分布是否和注释一致；
- 检查是否发生隐式近似；
- 检查 numerical shortcut 是否改变统计模型；
- 检查有限样本操作是否改变 posterior predictive；
- 检查 OOF 是否真的 held-out；
- 检查 ragged/missing 是否被偷偷变成 zero；
- 检查 scale/gauge/permutation invariance；
- 检查边界条件；
- 检查极端维数；
- 检查退化空间。

### 复杂度级

在跑 benchmark 之前，至少先写出：

\[
T,\ N,\ P,\ B,\ F,\ S
\]

对应的主要复杂度：

- \(O(T)\)
- \(O(TN)\)
- \(O(TP)\)
- \(O(P^2)\)
- \(O(P^3)\)
- \(O(NP^2)\)
- 内存字节数
- 每日重复次数
- full + fold 倍数
- scenario 倍数
- thread/BLAS 嵌套倍数

如果从静态复杂度就能证明某方案不可能达到目标，不得浪费时间跑大测试确认。

---

# 2. 静态优先原则的目的

本规则不是为了“少测试”。

恰恰相反，它是为了让测试成为：

\[
\boxed{
\text{验证推导}
}
\]

而不是：

\[
\boxed{
\text{替代推导}
}
\]

坏流程：

```text
改代码
→ 跑全回测
→ 看炸没炸
→ 猜原因
→ 再改
```

正确流程：

```text
读代码
→ 建模调用链
→ 静态定位
→ 推导不变量
→ 最小修改
→ 微验证
→ 局部测试
→ 必要时才全回测
```

---

# 3. 修改代码优先于“绕着错误测”

如果通过阅读已经确认 bug：

\[
\boxed{\text{先修 bug，不准先构造大测试去证明它确实是 bug}}
\]

例如：

- 参数传错；
- fold 泄漏；
- NaN 被替换成 0；
- active index 映射错误；
- trace constraint 只投影均值；
- posterior draw 没进入 scenario；
- fallback 更换 objective；
- OOF 使用 full-data hyperparameter；
- inactive asset 进入 Kelly；
- checkpoint 每日 deepcopy 整个 Gram；
- permanent fallback flag 明显锁死 fast path。

这类问题如果静态已经确定，不需要先跑十年回测。

---

# 4. 测试准入 Gate

执行任何测试之前，工程师必须能回答：

### 4.1 测什么？

必须具体到：

> “验证函数 A 在条件 B 下满足不变量 C。”

禁止：

> “跑一下看看。”

---

### 4.2 为什么静态分析不足？

必须说明：

- 哪个行为依赖 runtime；
- 哪个数值误差无法静态界定；
- 哪个性能常数只能实测；
- 哪个线程调度只有 runtime 可见；
- 哪个 solver convergence 需要数值验证。

如果说不出来：

\[
\boxed{\text{不准运行}}
\]

---

### 4.3 最小测试是什么？

必须选择能回答问题的最小输入。

例如：

- 3 assets；
- 20 rows；
- 2 folds；
- 2 bands；
- 固定 RNG；
- 单线程；
- 小矩阵 dense reference。

禁止一上来：

- 10 年；
- 全 universe；
- 300 scenarios；
- 多线程；
- adaptive quadrature；
- live API；
- 全部 feature。

---

# 5. 性能实验第一原则：优先微基准

强制规则：

\[
\boxed{
\text{如果没有穷尽微基准方法，严禁运行超过 1 分钟的大体量任务}
}
\]

“1 分钟”是工程纪律阈值，不是性能目标。

只要单次任务预计可能超过：

\[
60\text{ seconds}
\]

就必须先证明：

> 已经无法通过更小 benchmark 得到所需结论。

---

# 6. 什么叫“穷尽微基准”

至少按以下层级推进。

## 6.1 函数级

单独 benchmark：

- ruler；
- path basis；
- Gram；
- eigensolver；
- EB；
- conditioning；
- OOF fold fit；
- OOF prediction；
- fractional likelihood；
- scenario generation；
- Kelly；
- incremental contraction；
- checkpoint；
- residual oracle。

目标：

\[
t_f
\]

而不是整个 backtest 时间。

---

## 6.2 单日 pipeline

固定一个后期 decision day。

分别测：

\[
t_{\rm prepare},
t_{\rm solve},
t_{\rm scenario},
t_{\rm Kelly}.
\]

不要把所有东西混在一起。

---

## 6.3 固定 PreparedProblem

如果在优化 solver：

不要每次重新 prepare。

固定：

\[
PreparedProblem
\]

重复 solver。

---

## 6.4 固定 solver output

如果在优化 scenario：

固定 model，不重复 inference。

---

## 6.5 固定 scenario

如果在优化 Kelly：

固定 scenario matrix，不重复 model。

---

## 6.6 缩短时间轴

如果要验证跨日行为：

先跑：

- 5 天；
- 20 天；
- 50 天；
- 100 天。

而不是直接 2500 天。

---

# 7. 大体量运行准入条件

只有同时满足以下条件，才允许执行预计超过 1 分钟的任务：

1. 静态分析已经完成；
2. 微基准已经完成；
3. 热点已经明确；
4. 修改目标明确；
5. 小规模 correctness 已通过；
6. 数学 reference 已通过；
7. 已知此次大运行要回答的唯一问题；
8. 结果会改变下一步工程决策；
9. 没有更小实验能回答；
10. 有明确停止条件。

否则：

\[
\boxed{\text{禁止大跑}}
\]

---

# 8. 禁止“烧机器换理解”

以下行为一律禁止：

- 为了找 bug 反复跑十年；
- 为了看哪个函数慢直接跑完整回测；
- 每改一行就 full backtest；
- 未知复杂度先跑；
- 用多线程掩盖单线程算法缺陷；
- 用 GPU 掩盖 CPU 算法缺陷；
- 用更多 RAM 掩盖状态复制问题；
- 用缓存掩盖错误的数据依赖；
- 用更少 scenarios 假装提速；
- 用更少 folds 假装提速；
- 用更低 rank 假装提速；
- 用更短 history 假装提速。

---

# 9. Gate 制度

KTrader 的开发顺序固定为：

\[
\boxed{
G_0:\text{数学正确性}
\rightarrow
G_1:\text{数学加速}
\rightarrow
G_2:\text{计算机科学加速}
\rightarrow
G_3:\text{跨日增量化}
\rightarrow
G_4:\text{GPU}
}
\]

必须按顺序。

禁止并行打开后续 Gate。

---

# 10. Gate 0：数学正确性

目标：

\[
\mathcal H_t
\to
\Pi(\cdot\mid\mathcal H_t)
\to
P(r_{t+1}\mid\mathcal H_t)
\to
w_t^*
\]

数学定义完整且实现一致。

必须确认：

- fixed law ontology；
- full-history semantics；
- posterior uncertainty；
- trace neutrality；
- constrained evidence；
- OOF isolation；
- ragged semantics；
- innovation law；
- quadrature prior；
- exact Kelly；
- invariances；
- no hidden knobs；
- no objective-changing fallback。

Gate 0 没过：

\[
\boxed{\text{禁止讨论性能}}
\]

---

# 11. Gate 1：数学加速

只允许：

\[
\boxed{\text{计算相同数学对象的代数重写}}
\]

例如：

- analytic marginalization；
- Kronecker identity；
- Woodbury；
- primal/dual；
- sufficient statistics；
- constraint low-rank identity；
- FFT exact convolution；
- exact elimination；
- exact block algebra。

禁止：

- truncation；
- fewer folds；
- approximate posterior；
- approximate evidence；
- heuristic shrinkage；
- changing likelihood；
- changing prior；
- changing scenario law。

---

# 12. Gate 2：计算机科学加速

数学表达固定后，才能优化：

- allocation；
- cache locality；
- BLAS；
- LAPACK；
- workspace；
- SIMD；
- threading；
- scheduling；
- memory layout；
- mmap；
- serialization；
- warm start；
- precompilation；
- branch reduction。

所有结果必须数值等价。

---

# 13. Gate 3：增量化

跨日增量属于独立 exact algorithm。

目标：

\[
PreparedProblem_t^{incremental}
=
PreparedProblem_t^{reference}.
\]

增量化只能改变：

\[
\text{如何获得相同的 PreparedProblem}
\]

不能改变：

- posterior；
- OOF；
- residual law；
- scenario；
- Kelly。

---

# 14. Gate 4：GPU

只有在：

- Gate 0 完成；
- Gate 1 穷尽；
- Gate 2 穷尽；
- Gate 3 穷尽；

之后才允许 GPU。

GPU 是执行设备优化，不是算法逃生通道。

---

# 15. Gate 回退原则

这是本守则最重要的规则之一。

\[
\boxed{
\text{后面的 Gate 一旦破坏前面的 Gate，立即回退}
}
\]

例：

### 增量化破坏数学正确性

比如：

- OOF 泄漏；
- ragged semantics 改变；
- trace neutrality 变成投影；
- posterior covariance 不一致。

则：

\[
G_3\to G_0.
\]

不准：

> “先把增量化做完再修。”

---

### 计算机优化破坏数学加速所得结构

比如：

- 为了并行重新 materialize 巨矩阵；
- 为了 SIMD 改变 NaN 处理；
- 为了 cache 简化 posterior。

则：

\[
G_2\to G_1/G_0.
\]

---

### GPU 结果不等价

则：

\[
G_4\to G_2.
\]

不准边修 GPU 边继续扩 GPU 范围。

---

# 16. “带伤前进”绝对禁止

禁止以下语言作为工程决策：

- “先这样，以后再修。”
- “第一版先跑起来。”
- “这个地方先 hardcode。”
- “暂时先用 full alpha。”
- “先做一个简化版。”
- “先把数学降级一点测性能。”
- “先 approximate，等快了再恢复 exact。”
- “先 copy 一份代码。”
- “先加 fallback。”
- “先多一个 flag。”
- “先多一个 route。”
- “先多一个 special case。”

如果当前设计不优雅：

\[
\boxed{\text{先重构，再继续}}
\]

---

# 17. 优雅重构拥有最高局部优先级

只要发现：

- 同一数学概念有两个 owner；
- 同一个公式有两个实现；
- 同一个数据结构有两个表示；
- 同一个 solver 有 fast/reference 两份；
- 同一 invariance 在不同模块重复修补；
- probe 开始修改 production API；
- performance route 进入数学代码；
- fallback 数量持续增加；
- flag 组合爆炸；
- function signature 越来越长；
- comment 比代码复杂；
- “要解释为什么这个分支存在”需要一页文字；

则必须优先考虑重构。

规则：

\[
\boxed{
\text{复杂度失控}
\Rightarrow
\text{暂停功能}
\Rightarrow
\text{重构}
}
\]

---

# 18. 禁止“第一版破烂”

本项目明确禁止：

> “先写一个烂版本，再慢慢重构。”

允许 prototype 的唯一条件：

- 在 `dev/` 或独立实验分支；
- 不进入 production；
- 不成为 API；
- 不被其他 production 模块依赖；
- 不写成未来兼容包袱；
- 实验完成后删除。

生产代码第一次进入主路径时，就必须：

- owner 清晰；
- API 清晰；
- invariants 清晰；
- fallback 清晰；
- tests 清晰；
- docs 清晰。

---

# 19. 一次只解决一个层次的问题

禁止一个 PR 同时：

- 修数学；
- 改 solver；
- 改 scheduler；
- 改 incremental state；
- 改 GPU；
- 改 scenario law。

正确做法：

### PR A

只修数学。

### PR B

只做数学等价优化。

### PR C

只做执行优化。

这样任何回归都能定位。

---

# 20. 单一真源原则

每个概念只能有一个 owner。

例如：

| 概念 | 唯一 owner |
|---|---|
| price semantics | data |
| geometry/ruler | geometry |
| PreparedProblem | prepare |
| posterior response | response |
| OOF residual interface | residuals |
| predictive law | predict |
| Kelly | kelly |
| exact incremental preparation | accelerator/incremental |
| backtest sequencing | backtest |
| execution | broker/live |

禁止跨 owner 重复公式。

---

# 21. Reference path 永久保留

任何高性能实现都必须有：

\[
\boxed{\text{simple exact reference}}
\]

Reference 要：

- 简单；
- 直接；
- 无 cleverness；
- 无 caching；
- 无 threading；
- 无 incremental；
- 无 GPU；
- 少 fallback；
- 可人工审计。

Fast path 永远和它对比。

禁止为了维护方便删除 reference。

---

# 22. 性能优化必须提交“等价性证据”

任何 performance PR 必须回答：

### 数学对象是否完全相同？

必须给：

\[
\Delta s,
\Delta S_{xx},
\Delta S_{xy},
\Delta S_{yy},
\Delta \alpha,
\Delta\Sigma,
\Delta\mu,
\Delta w.
\]

### 快在哪里？

必须给 microbenchmark：

\[
t_{before}
\rightarrow
t_{after}.
\]

### 为什么快？

必须说明：

- FLOPs 减少；
- memory traffic 减少；
- allocation 减少；
- factorization 次数减少；
- cache locality；
- parallelism；
- solver iterations。

禁止只说：

> “整体回测快了 12%。”

---

# 23. 优化必须有理论复杂度说明

每个性能 PR 至少写：

\[
O_{\rm before}(\cdot)
\]

和：

\[
O_{\rm after}(\cdot).
\]

如果只是常数优化，也要明确：

> asymptotic unchanged; reduces X allocations / Y bytes / Z GEMMs。

---

# 24. 不以 Sharpe 验证工程正确性

严禁：

> “优化后 Sharpe 差不多，所以数学没变。”

正确验证：

- deterministic matrix equality；
- posterior moment equality；
- predictive law equality；
- numerical tolerance；
- invariance；
- KKT；
- likelihood/evidence。

Sharpe 只能作为金融结果。

---

# 25. 不以 backtest speed 验证局部优化

比如优化 `condition_trace_neutrality`：

必须 benchmark 它本身。

不能只比较：

\[
5.8\text{ min}
\to
5.6\text{ min}
\]

因为噪声太大，也不能说明局部效果。

---

# 26. benchmark 必须从小到大

顺序：

\[
\boxed{
\text{kernel}
\rightarrow
\text{stage}
\rightarrow
\text{single day}
\rightarrow
\text{short range}
\rightarrow
\text{full backtest}
}
\]

不能跳级。

---

# 27. 超过 1 分钟的命令必须显式说明理由

执行前必须写清：

```text
Purpose:
Why static analysis is insufficient:
Microbenchmarks already completed:
Why <60s run cannot answer:
Expected runtime:
Abort condition:
Decision that will be made from result:
```

缺一项：

\[
\boxed{\text{不准执行}}
\]

---

# 28. 失败任务不得“顺手再跑一次”

如果一个大任务：

- timeout；
- OOM；
- hang；
- 极慢；
- 无日志；
- 不知道卡哪里；

禁止马上重跑。

必须先：

1. 看日志；
2. 看 profiler；
3. 看 stage boundary；
4. 缩小 workload；
5. 添加最小 instrumentation；
6. 重新做微基准。

---

# 29. instrumentation 不得污染核心

诊断代码：

- 不进入 model；
- 不改变 solver signature；
- 不改变 data structure；
- 不进入 serialization；
- 不参与 branch；
- 默认零开销。

优先：

- external profiler；
- wrapper；
- dev scripts；
- benchmark harness。

避免 production API 出现：

- `shadow_timing`
- `probe`
- `debug_counter`
- `diagnostic_state`

除非该信息本身属于稳定产品语义。

---

# 30. 注释必须描述事实，不描述愿望

禁止：

> “exact matrix-free posterior”

但代码实际上 materialize \(P\times P\)。

禁止：

> “true OOF”

但 alpha 用了 full data。

禁止：

> “not wired”

但 production 已经调用。

注释必须和当前实现同步。

---

# 31. TODO 不是免责条款

发现数学问题：

不能写：

```text
TODO fix later
```

然后继续后续 Gate。

如果影响前置 Gate：

\[
\boxed{\text{立即回退}}
\]

---

# 32. fallback 纪律

Fallback 只能存在两类。

## 32.1 同数学目标 fallback

例如：

- custom Kelly solver失败；
- Clarabel 解相同 exact Kelly objective。

允许。

---

## 32.2 reference evaluator fallback

例如 incremental backend 无法处理某状态：

\[
prepare_{\rm incremental}
\to
prepare_{\rm reference}.
\]

允许。

---

禁止：

- Kelly → mean variance；
- posterior → ridge point estimate；
- exact likelihood → heuristic score；
- ragged → zero imputation；
- full \(G\) → diagonal \(G\)。

---

# 33. 参数纪律

任何参数都要分类：

### 理论参数

属于模型。

### 先验参数

属于 epistemology。

### 数值容差

属于 computation。

### 资源阈值

属于 routing。

绝不允许混用。

例如：

`materialize_row_limit=4096`

只能决定：

> fast backend 还是 reference backend。

不得改变输出。

---

# 34. 数值容差必须做收敛测试

例如：

\[
\epsilon=10^{-6}
\]

必须能验证：

\[
\epsilon\to \epsilon/10
\]

时：

\[
w_\epsilon\to w^*.
\]

不能因为某个 epsilon backtest 好看就选择它。

---

# 35. 随机性纪律

所有 stochastic 流程必须：

- seed 显式；
- date seed deterministic；
- parallel scheduling 不改变 RNG law；
- reference/fast 使用相同随机输入；
- benchmark 区分 RNG cost 和 kernel cost。

---

# 36. 并发纪律

并行前先单线程正确。

禁止：

> “多线程下才测试。”

顺序：

1. deterministic single thread；
2. reference equality；
3. race audit；
4. thread ownership；
5. parallel benchmark。

---

# 37. 每个 mutable state 必须有 owner

任何 mutable object 必须回答：

- 谁创建？
- 谁修改？
- 谁读取？
- 是否跨 thread？
- 生命周期？
- checkpoint 是否 deep/shallow？
- 是否允许 alias？

答不出来不得进入 production。

---

# 38. 增量化的特殊纪律

增量化必须把：

\[
\boxed{\text{state}}
\]

视为数学缓存，而不是“方便的 mutable bag”。

每个 state field 必须对应某个明确的 raw sufficient statistic 或数值缓存。

禁止保存：

- 无法解释的中间量；
- 重复信息；
- 派生但未验证同步的信息；
- “也许以后会用”的缓存。

---

# 39. 增量 state 必须可重建

任意日期 \(t\)：

\[
State_t
\]

必须可以从 reference history：

\[
\mathcal H_t
\]

重新构造。

且：

\[
Prepared(State_t)
=
PreparedReference(\mathcal H_t).
\]

---

# 40. 增量化不追求“所有东西都增量”

正确目标：

> 所有值得递推、能够 exact 递推的对象都递推。

如果某块：

- 重算便宜；
- 递推状态复杂；
- proof 很困难；
- mutation risk 高；

则允许重算。

“做尽增量化”不等于“零重算”。

---

# 41. GPU 前置禁止事项

进入 GPU Gate 之前不得：

- 写 shader；
- 写 Vulkan command buffer；
- 建 device-specific model；
- 改 Float64 为 Float32；
- 为 GPU 重写数学；
- 为 GPU 改 layout 导致 CPU reference 消失。

GPU backend 必须消费同一个数学对象。

---

# 42. 性能目标不是理论公理

例如：

\[
10\text{ years}<60s
\]

是 engineering target。

不是 theory requirement。

因此不得为了性能目标：

- 减少信息；
- 改 prior；
- 改 posterior；
- 减 folds；
- 改 horizon；
- 改 Kelly。

---

# 43. 如果性能目标和代码质量冲突

优先级：

\[
\boxed{
\text{正确}
>
\text{清晰}
>
\text{可维护}
>
\text{可验证}
>
\text{快}
}
\]

但：

“清晰”不是慢代码的借口。

正确做法：

> 找到既保持结构又更快的算法。

---

# 44. 发现更优雅结构时允许推翻旧实现

禁止 sunk-cost fallacy。

即使已经写了：

- 500 行；
- 1000 行；
- 一周；
- 很多测试；

如果发现一个更统一的数学结构：

\[
\boxed{\text{重写}}
\]

不要因为旧代码成本高就继续背债。

---

# 45. 代码膨胀报警条件

出现任一条件，必须暂停：

- 单文件 > 1000 行且职责不单一；
- 同概念 > 2 个实现；
- fallback > 3 层；
- 同函数 > 10 个 keyword 参数；
- debug/probe 参数进入核心 solver；
- state struct > 20 fields 且缺少数学分类；
- 一个优化需要 > 3 个新类型；
- 一个 bug fix 需要同时改 > 5 个模块；
- README 无法解释调用链；
- 新工程师无法在 30 分钟内找到公式 owner。

报警后：

\[
\boxed{\text{先重构}}
\]

---

# 46. 重构不是“以后再做”

重构属于当前任务的一部分。

如果当前任务导致结构恶化：

任务未完成。

---

# 47. PR 必须有“删除项”

所有大型 PR 都要回答：

> 这次删除了什么复杂度？

可以是：

- 删除旧 path；
- 删除 flag；
- 删除 duplicate solver；
- 删除 materialization；
- 删除 fallback；
- 删除 special case；
- 删除 diagnostic hook。

如果只增加不删除，必须特别说明为什么。

---

# 48. Dev / Production 分离

实验性内容只能放：

```text
dev/
bench/
experiments/
```

production 模块不得依赖它们。

包括：

- profiler；
- ceiling probe；
- benchmark harness；
- one-off scripts；
- alternative solver prototype；
- GPU experiment。

---

# 49. 测试分层

## Level A：纯数学单元

毫秒级。

## Level B：small dense reference

秒级。

## Level C：single-day realistic

秒级。

## Level D：short-range integration

<60 秒。

## Level E：full backtest

>60 秒，仅在前四层穷尽后。

---

# 50. Full backtest 的定位

Full backtest 只回答：

- end-to-end regression；
- financial behavior；
- final throughput；
- memory stability；
- long-run determinism。

它不负责：

- 找局部 bug；
- 找单函数性能问题；
- 验证矩阵公式；
- 验证 fold isolation。

---

# 51. 任何数学优化必须先证明，再实现

顺序：

1. 写公式；
2. 写维度；
3. 写等价证明；
4. 写 reference identity；
5. 写代码；
6. 微测试。

禁止：

> “代码看起来等价。”

---

# 52. 任何复杂优化必须有退出条件

例如 incremental regime compiler：

若发现：

\[
\text{complexity cost} > \text{measured benefit}
\]

应停。

不得因为已经投入很多而无限扩展。

---

# 53. 设计评审要问的五个问题

每次设计前必须问：

1. 这是哪个 Gate？
2. 它会不会破坏更早 Gate？
3. 能不能通过数学恒等式消灭问题？
4. 能不能通过重构减少状态？
5. 是否有更简单的 exact reference？

---

# 54. Debug 顺序

固定：

\[
\boxed{
\text{static}
\rightarrow
\text{assertion}
\rightarrow
\text{tiny reproduction}
\rightarrow
\text{unit test}
\rightarrow
\text{microbenchmark}
\rightarrow
\text{integration}
}
\]

禁止逆序。

---

# 55. 性能回归处理

发现性能回归时：

1. 不准立刻“再优化”；
2. 对比 commit；
3. 静态检查复杂度变化；
4. microbenchmark；
5. attribution；
6. 找出新增成本；
7. 决定 revert / redesign。

如果原因不明：

\[
\boxed{\text{不准继续叠优化}}
\]

---

# 56. Correctness 回归优先于性能回归

如果同时发现：

- 快了 2×；
- 但数学等价性失败；

结论是：

\[
\boxed{\text{失败}}
\]

没有折中。

---

# 57. 文档必须跟着代码

修改：

- theory；
- API；
- state；
- fallback；
- Gate；
- benchmark procedure；

必须同步更新：

- SPEC；
- ROADMAP；
- 本守则（若开发纪律变化）；
- module docstring。

---

# 58. 陌生工程师测试

任何稳定模块必须满足：

> 一个没参加历史开发的人，仅凭 SPEC + 本守则，可以解释：
>
> - 这个模块为什么存在；
> - 输入；
> - 输出；
> - 数学对象；
> - invariant；
> - owner；
> - reference；
> - fast path；
> - failure behavior。

如果解释不了，文档或架构未完成。

---

# 59. 版本推进纪律

## 0.x

允许理论变化。

## 1.0

理论闭合。

## 1.x

允许：

- exact mathematical acceleration；
- computer-science optimization；
- exact incrementalization；
- device backend。

不允许偷偷改变理论。

## 2.0

只有明确理论升级才升。

---

# 60. 一项优化“做尽”的定义

“做尽”不是：

> 再也想不到任何优化。

而是：

1. hotspot 已知；
2. 该层所有主要结构性机会已检查；
3. 余下收益只有小常数；
4. 再继续会明显增加复杂度；
5. 下一个 Gate 的收益预期更高；
6. correctness 有证书；
7. 文档完成。

满足后才能开下一 Gate。

---

# 61. 数学加速做尽的清单

在进入 CS 加速前，确认检查过：

- sufficient statistics；
- marginalization；
- conditional Gaussian identities；
- Kronecker；
- Woodbury；
- Schur complement；
- low-rank constraints；
- symmetry；
- gauge；
- exact FFT；
- analytic gradients；
- common subexpressions；
- primal/dual；
- block matrix identities。

---

# 62. CS 加速做尽的清单

进入 incremental 前确认：

- zero avoidable allocation；
- BLAS3 化；
- no repeated factorization；
- workspace；
- cache-friendly layout；
- no accidental copies；
- no thread oversubscription；
- no repeated serialization；
- no huge deepcopy；
- warm start；
- profiler hotspots已解决；
- compiler specialization；
- FFT plan reuse。

---

# 63. 增量化做尽的清单

进入 GPU 前确认：

- ruler state；
- raw sufficient statistics；
- mask/operator factorization；
- fold movements；
- active expansion；
- OOF state；
- residual strategy；
- innovation strategy；
- checkpoint strategy；
- block parallelism；
- exact reference equality。

并明确哪些对象选择重算及原因。

---

# 64. GPU 做尽以后才考虑更多设备/分布式

禁止：

- CPU没做完 → GPU；
- GPU没做完 → multi-GPU；
- multi-GPU没做完 → cluster。

每次只开一个复杂度维度。

---

# 65. 每日工程工作流模板

开始任务：

```text
1. Read SPEC section
2. Identify Gate
3. Read owner module
4. Read callers
5. Read callees
6. Static derivation
7. Complexity derivation
8. Decide refactor vs patch
9. Implement minimal coherent change
10. Static review
11. Tiny deterministic test
12. Microbenchmark if performance-related
13. Escalate test size only if necessary
14. Update docs
15. Remove obsolete path
```

---

# 66. Commit 前 checklist

- [ ] 我是否先读代码而不是先跑？
- [ ] 静态方法是否穷尽？
- [ ] 是否知道自己在哪个 Gate？
- [ ] 是否可能破坏前 Gate？
- [ ] 是否引入第二份数学实现？
- [ ] 是否有更优雅重构？
- [ ] 是否引入临时破烂？
- [ ] 测试是否从最小规模开始？
- [ ] 是否执行了 >1min 任务？若是，理由是否充分？
- [ ] performance PR 是否有 microbenchmark？
- [ ] 是否有 reference equality？
- [ ] 是否删除旧复杂度？
- [ ] 文档是否同步？

任一关键项为“否”：

\[
\boxed{\text{不提交}}
\]

---

# 67. Code Review 必问

Reviewer 必须问：

### Correctness

- 公式和代码一致吗？
- layer 分类对吗？
- 是否引入隐式假设？
- 是否改变 prior/likelihood？
- OOF 是否隔离？
- invariance 是否保持？

### Architecture

- owner 是否唯一？
- 是否复制逻辑？
- 是否该重构？
- 是否是临时破烂？

### Performance

- 为什么快？
- microbenchmark？
- complexity？
- memory traffic？
- 是否只是换硬件掩盖算法问题？

### Process

- 是否违规提前跑大测试？
- 是否越 Gate？
- 是否带伤前进？

---

# 68. 禁止以工期为理由降低标准

“时间不够”时正确行动：

- 缩 scope；
- 延后功能；
- 保留 reference；
- 不开下一 Gate。

错误行动：

- hardcode；
- copy；
- heuristic；
- approximate；
- undocumented fallback；
- TODO debt。

---

# 69. 本项目宁可少做，也不做烂

优先：

\[
\boxed{
\text{少而统一}
>
\text{多而拼接}
}
\]

一个正确的 response operator 胜过十个 feature。

一个正确的 solver 胜过五个 fallback。

一个正确的 incremental state 胜过三种 partial accelerator。

---

# 70. 最终工程哲学

KTrader 的目标不是：

> 写出最多的聪明优化。

而是：

> 用尽可能少的原则，把理论精确、快速、可验证地实现出来。

因此工程上的 Maxwell 原则是：

\[
\boxed{
\text{少数不变量}
\rightarrow
\text{少数核心对象}
\rightarrow
\text{统一实现}
\rightarrow
\text{可证明优化}
}
\]

---

# 71. 最终强制条款

以下条款没有例外：

## 条款 A

\[
\boxed{\text{静态手段未穷尽，严禁执行测试}}
\]

## 条款 B

\[
\boxed{\text{微基准未穷尽，严禁执行预计超过 1 分钟的大任务}}
\]

## 条款 C

\[
\boxed{\text{后 Gate 破坏前 Gate，立即回退}}
\]

## 条款 D

\[
\boxed{\text{禁止带伤前进}}
\]

## 条款 E

\[
\boxed{\text{优雅重构优先于赶工}}
\]

## 条款 F

\[
\boxed{\text{禁止用“第一版破烂”进入 production}}
\]

## 条款 G

\[
\boxed{\text{任何性能提升不得缩水数学性质}}
\]

## 条款 H

\[
\boxed{\text{一次只打开一个复杂度层次}}
\]

---

# 72. 一句话版本

若工程师只记住一句话：

> **先把代码和数学读透；能静态解决绝不跑，能微基准绝不大跑；每个 Gate 必须完全站稳再进下一层，一旦后层破坏前层立即回退；任何时候都宁可优雅重构，也不准拿临时破烂顶上。**

这就是 KTrader 的开发纪律。
