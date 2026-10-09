# KTrader / Path Kelly — 所有需要的保姆级裁决

**文档状态：Normative Decision Book / Gate-0 裁决书**  
**目的：一次性裁决 KTrader 2.0 回测失败后暴露的全部规范性歧义。**  
**适用范围：当前 2.0 CPU 代码线及其后续纠偏；本文件优先于旧 SPEC 中与本裁决冲突的条款，直到这些裁决被正式并入新的唯一权威 SPEC。**  
**开发纪律：严格服从《KTrader 开发守则》——静态手段未穷尽不得测试，微基准未穷尽不得执行预计超过 1 分钟的大任务，后 Gate 破坏前 Gate 必须回退，不准带伤前进。**

---

# 0. 为什么需要这份裁决书

当前项目并不是因为“回测亏钱”而回退 Gate。

真正触发 Gate-0 重开的原因是：

\[
\boxed{
\text{项目声称的数学对象}
\neq
\text{当前生产实现中若干关键对象}
}
\]

目前已经静态确认的主要差距包括：

1. 理论 Kelly 是
   \[
   \arg\max_w E[\log R^\top w\mid\mathcal H_t],
   \]
   但正式两年回测使用的是固定 \(S=300\) 的 sample-Kelly，且 `adaptive=false`。
2. 当前 response posterior 只积分
   \[
   G\mid\hat\alpha,\hat\Sigma,
   \]
   而 \(\alpha,\Sigma,\sigma^2\) 是 point-estimate plug-in。
3. 当前 innovation law 是
   \[
   \text{历史 OOF residual row}
   \times
   \text{一个 common scalar fractional volatility multiplier},
   \]
   不是完整 vector conditional innovation law。
4. 每 band / 每 Q-P channel 的 14 条 trace-zero constraint 已被实现得非常精确，但没有从更高层理论推出。
5. 当前 macro 与 relative response 被分别拟合，等价于默认删除 macro↔relative cross-response block；这一点没有被理论证明。
6. 早期 `shrink_drift` 的缺陷被清除是正确的，但随后“任何 DC / unconditional drift 都不存在”也被悄悄固化，二者不是同一件事。
7. 当前 fully-invested simplex 把 cash 排除在 Kelly feasible set 外；这与一般形式
   \[
   \log(1+w^\top r)
   \]
   所隐含的 cash numeraire 并不等价。
8. 回测脚本通过篡改 `Bars.bar` 表达 252 日可交易规则，把“是否有真实观测”和“是否允许交易”混为一谈。
9. 等权回测的可交易集合依赖 Path Kelly 的 `active_indices`，benchmark 不是完全外生。
10. 回测报告对 “Volatility Pump”“完全脱离 beta”“低 SNR 导致 Kelly 集中”作了当前证据不支持的因果解释。

因此，本文件不讨论“怎样调到赚钱”。

本文件只做一件事：

\[
\boxed{
\text{重新钉死 Path Kelly 到底是什么。}
}
\]

---

# 1. 裁决优先级与生效方式

## D-001 — 本文件是临时最高规范裁决

在新 SPEC 合并完成前：

\[
\boxed{
\text{本裁决书}
>
\text{旧 SPEC 中冲突条款}
>
\text{当前实现}
>
\text{历史注释}
}
\]

不冲突的旧 SPEC 条款继续有效。

---

## D-002 — 不改写历史证据

当前已经发布、打 tag、保存 hash 的 “2.0.0 Final” 文件、日志、回测结果、artifact：

**全部保留原字节、原 hash、原时间线。**

禁止：

- 修改旧报告冒充“当时其实没发布”；
- 重写旧日志；
- 覆盖旧 artifact；
- 删除失败结果；
- 改 git tag 以隐藏失败。

正确做法是新增状态文件说明：

> “2.0.0 Final 作为历史发布物保留，但其数学闭合声明已被后续 Gate-0 审计撤销，不再是当前推荐规范。”

---

# 2. 当前版本地位裁决

## D-003 — 撤销“当前数学 Final”地位

当前代码不得继续被称为：

> KTrader 2.0 Final mathematical model

当前开发状态统一改为：

\[
\boxed{
\text{KTrader 2.0-RC / Gate 0 Reopened}
}
\]

这是规范状态，不要求删除历史 `2.0.0` tag。

---

## D-004 — 两年 -62.22% 不是理论裁决

当前两年结果只能被视为：

\[
\boxed{
\text{当前 RC 实现的 failure fixture}
}
\]

它证明：

> 当前生产 predictive law + 当前有限 scenario numerical integration + 当前 feasible set 在该历史上产生了极差结果。

它**不证明**：

- Kelly 理论错误；
- price-only 理论错误；
- path response 错误；
- 低信噪比必然导致集中；
- 等权一定优于 Path Kelly；
- trace neutrality 一定错误；
- 早期版本一定更正确。

---

## D-005 — 不回滚到早期“赢家”

V0 / 0.7 / 0.9 中已经静态确认的：

- selection leakage；
- in-sample residual；
- zero residual fallback；
- moment matching；
- ARD/BF gate；
- locked risk as cash；
- 几何 eigenvector identity 问题；

不得恢复。

\[
\boxed{
\text{纠偏不是回滚。}
}
\]

目标是从当前理论继续前进，而不是用早期回测收益挑旧机制。

---

# 3. 开发 Gate 裁决

## D-006 — 当前立即回到 Gate 0

即日起暂停：

- 性能优化；
- incremental 新功能；
- GPU；
- Vulkan；
- full 2-year backtest；
- 10-year benchmark；
- 参数搜索；
- Sharpe 对比实验；
- portfolio cap / fractional Kelly；
- 新 scheduler；
- 新 cache；
- 新 performance route。

只允许：

\[
\boxed{\text{Gate 0 数学定义、静态重构、reference 实现}}
\]

---

## D-007 — 后续顺序不变

项目顺序正式重申：

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

任何后层破坏前层：

\[
\boxed{\text{立即回退。}}
\]

---

# 4. 项目本体论继续保留的裁决

以下旧原则继续有效，不重开：

## D-008 — 规律固定

\[
\boxed{
\mathcal F:\mathcal H_t\mapsto P(r_{t+1}\mid\mathcal H_t)
}
\]

规律不时变，输入历史在演化。

---

## D-009 — 全历史是原语

不把有限 state 当市场本体。

有限 basis 只是当前模型表示。

---

## D-010 — 价格是唯一 alpha 原始输入

不引入：

- 财报；
- 新闻；
- 行业标签；
- 人工 regime；
- factor bag。

execution 所需账户/报价信息继续独立。

---

## D-011 — 日频语义不重开

当前 theoretical step = one trading day。

不再讨论 intraday live 语义。

---

# 5. 数据语义裁决：四种 mask 必须彻底拆开

这是本轮必须最先施工的代码边界之一。

---

## D-012 — `Bars.bar` 永远只表示物理观测事实

定义：

\[
O_{t,i}
=
\mathbf 1
\{\text{日期 }t\text{ 资产 }i\text{ 有真实市场 bar}\}.
\]

`Bars.bar[t,i]` 唯一语义就是 \(O_{t,i}\)。

绝对禁止：

- 因为不想交易而改成 false；
- 因为历史不足而改成 false；
- 因为模型不 admission 而改成 false；
- 因为风控而改成 false。

市场事实不是策略配置。

---

## D-013 — 新增独立 model-admission mask

定义：

\[
A^{model}_{t,i}.
\]

它回答：

> 该资产真实观测是否允许进入当日模型的信息空间？

默认核心模型裁决：

\[
\boxed{
A^{model}_{t,i}=1
\quad\text{当且仅当 prefix 中至少存在一条有效 daily return。}
}
\]

理由：

- 弱证据由 posterior 表达；
- 不用任意 MINROWS 门槛替代认识论不确定性；
- 全 NaN dummy asset 严格排除。

若未来提出更严格 admission rule：

必须作为**理论变更**单独审查，不得藏在 `bar` 中。

---

## D-014 — 新增独立 trade-eligibility mask

定义：

\[
E^{trade}_{t,i}.
\]

它回答：

> 即使模型看到了该资产，今天是否允许新建/增加风险仓位？

例如用户设定的 252 有效交易日规则属于这里。

---

## D-015 — 新增 physical-execution mask

定义：

\[
T^{exec}_{t,i}.
\]

它由：

- 当天真实 bar；
- broker quote；
- 市场可执行性；

决定。

理论 backtest 的简单形式至少需要：

\[
T^{exec}_{t,i}=O_{t,i}.
\]

---

## D-016 — free mask 的唯一合法定义

新建风险仓位的自由集合：

\[
\boxed{
free_{t,i}
=
E^{trade}_{t,i}
\land
T^{exec}_{t,i}
}
\]

模型 admission 不得偷偷混入 free。

当前模型是否预测该资产，是另一个问题。

---

## D-017 — held position 永远单独处理

若：

\[
w^{held}_i>0
\]

但：

\[
free_i=0,
\]

该资产成为 locked risk。

它的 scenario wealth contribution 必须保留。

绝不当 cash。

---

# 6. PONY / 252 日规则裁决

## D-018 — 当前报告中的 252 日规则属于 trade eligibility

因为报告原文称：

> “1 年数据回看与可交易性硬约束”

所以本次裁决把它解释为：

\[
\boxed{
E^{trade}_{t,PONY}=0
\quad\text{直到累计 252 个有效 bar。}
}
\]

但：

\[
O_{t,PONY}
\]

仍然保持真实市场事实。

因此 252 日以前：

- 模型可以看见真实 PONY 价格；
- PONY 可以作为预测信息；
- PONY 不允许持仓；
- 等权 benchmark 也不持仓 PONY。

---

## D-019 — 禁止再通过篡改 `Bars.bar` 表达 252 日规则

当前回测脚本：

```julia
b_bar[1:t_252-1, pony_j] .= false
```

这种写法正式判为：

\[
\boxed{\text{语义错误}}
\]

即使 PnL 恰好符合实验意图，也必须重写。

---

## D-020 — benchmark universe 不得依赖 Path Kelly active set

等权 benchmark 的候选集合只能来自：

\[
E^{trade}\land T^{exec}.
\]

不得写成：

\[
model.active\_indices \cap free.
\]

实验组不能决定对照组有哪些资产。

---

# 7. Macro / Relative 几何裁决

## D-021 — center-of-mass / relative decomposition 保留

对 observed normalized returns：

\[
u_{t,i}=r_{t,i}/s_{1,i}
\]

继续分解为：

- macro scalar；
- zero-sum relative field。

这个结构是核心价格几何。

---

## D-022 — zero embedding 继续保留，但只能是 field algebra

未观测坐标：

\[
e_{t,i}=0
\]

只表示：

> 在当前 active coordinate space 中 relative field 的嵌入值为零。

它**不表示**：

\[
r_{t,i}=0.
\]

observation mask 必须永久保留。

---

## D-023 — fixed Helmert gauge 继续是纯数值 gauge

Helmert \(Q\)：

\[
Q^TQ=I,\qquad Q^T\mathbf1=0
\]

继续使用。

它不是板块、不是 factor、不是市场模式。

自然模式来自 operator spectrum。

---

## D-024 — ruler / gauge 不作为 posterior 随机对象

本轮正式裁决：

\[
\boxed{
s_1,\ s_m,\ s_\perp,\ Q
}
\]

属于固定模型定义下的**确定性历史泛函 / 数值 gauge**。

因此：

- 不恢复 geometry bootstrap；
- 不恢复 `Phi_draws`；
- 不为 Helmert basis 加 posterior；
- 不把 gauge tracking 当 epistemic uncertainty。

若未来要让尺度律本身成为未知 law parameter：

那是新的理论版本。

---

# 8. Response Operator 的重大裁决：取消无证明的 block-diagonal 假设

当前代码实际上拆成：

\[
macro \rightarrow macro
\]

以及：

\[
relative \rightarrow relative.
\]

这意味着默认：

\[
G_{m\leftarrow\perp}=0,
\qquad
G_{\perp\leftarrow m}=0.
\]

没有理论推导支持这两个 block 恒为零。

---

## D-025 — 新 reference response 必须允许 macro↔relative cross-response

统一 mode output：

\[
y_t
=
\begin{bmatrix}
m_t\\
q_t
\end{bmatrix},
\qquad
q_t=Q^T e_t.
\]

统一 mode input：

\[
x_t
=
\begin{bmatrix}
B_m(t)\\
B_\perp(t)
\end{bmatrix}.
\]

其中：

\[
B_m\in\mathbb R^{14},
\qquad
B_\perp\in\mathbb R^{14(N-1)}.
\]

总输入维度：

\[
14N.
\]

统一 response：

\[
\boxed{
y_{t+1}
=
G x_t+\epsilon_{t+1}
}
\]

其中：

\[
G\in\mathbb R^{N\times 14N}.
\]

block 展开：

\[
G=
\begin{bmatrix}
G_{mm} & G_{m\perp}\\
G_{\perp m} & G_{\perp\perp}
\end{bmatrix}.
\]

四个 block 全部允许由数据/posterior决定。

---

## D-026 — “sector rotation / mode transfer” 必须允许跨 macro-relative

尤其不得预先禁止：

- relative path 预测整体市场；
- macro path 预测横截面 rotation。

这才符合原来的 full response operator 理念。

---

# 9. Trace Neutrality 最终裁决

这是本轮最重要裁决之一。

---

## D-027 — 当前 14 条 per-band trace neutrality 不再属于核心公理

当前：

\[
\operatorname{tr}A_b=0,
\qquad
\operatorname{tr}B_b=0
\]

逐 band、逐 channel 的 14 条约束：

正式分类为：

\[
\boxed{\text{candidate modeling hypothesis}}
\]

不是：

- theorem；
- identification theorem；
- price-only symmetry 的必然结果。

---

## D-028 — 修正版 2.x core 默认关闭 14 条 trace constraint

新 reference core：

\[
\boxed{
\text{不施加 }
\operatorname{tr}A_b=
\operatorname{tr}B_b=0.
}
\]

理由：

1. 没有高层推导；
2. likelihood 一般对这些方向有信息；
3. 它可能改变 diagonal/off-diagonal 经济归因；
4. 当前 evidence 只能证明“约束实现正确”，不能证明“约束理论正确”。

---

## D-029 — 旧 trace-conditioned solver 不删除

保留为：

```text
experiments/trace_neutral_hypothesis/
```

或等价 dev 路径。

用途：

- 理论反例；
- ablation；
- 未来若获得推导可复活。

但：

\[
\boxed{\text{不得成为默认 production core}}
\]

---

## D-030 — 真正必须 hard enforce 的只有可达 support

relative input 本身位于：

\[
\mathbf1^\perp.
\]

relative output 也位于：

\[
\mathbf1^\perp.
\]

因此新实现直接在 gauge 坐标：

\[
Q^T e
\]

中工作。

这构造性删除：

- input common unreachable direction；
- output common/relative 重复表示；

无需 14 条 trace constraint。

---

# 10. Drift / DC 分量裁决

早期 `shrink_drift` 有缺陷，并不推出：

\[
\text{真实条件均值的 DC component 必须为 0}.
\]

Q/P path basis 本质上主要描述动态 path variation。

没有 DC channel 等价于给模型强加：

> 如果所有 dynamic path feature 不解释收益，则 unconditional expected field 必须为零。

该假设没有被证明。

---

## D-031 — 恢复一个统一 DC / intercept channel

新 response 设计矩阵增加常数通道：

\[
x_{0,t}=1.
\]

于是：

\[
\boxed{
y_{t+1}
=
b_0+Gx_t+\epsilon_{t+1}.
}
\]

其中：

\[
b_0\in\mathbb R^N
\]

位于统一 mode output 空间。

它可以包含：

- macro unconditional drift；
- zero-sum relative long-run drift。

---

## D-032 — DC prior 必须零中心

禁止：

- 人工正收益 prior；
- `shrink_drift` 式“默认股票长期上涨”的注入；
- 根据历史赢家指定正先验。

DC prior 均值：

\[
\boxed{0}
\]

其非零 posterior 只能来自价格证据。

---

## D-033 — DC 与 dynamic path 使用两个自然 group precision

定义两个未知正精度：

\[
\alpha_0
\]

用于 DC，

以及：

\[
\alpha_p
\]

用于全部 path coefficients。

禁止：

- 每 band 一个 alpha；
- 每资产一个 alpha；
- ARD/BF 关闭通道；
- 根据 Sharpe 选 group。

两个 group 的理由仅是：

\[
\boxed{
\text{zero-frequency DC}
\neq
\text{dynamic path response}
}
\]

---

# 11. Posterior 最终裁决：生产不再使用 plug-in EB

这是当前过度确信问题必须纠正的核心。

---

## D-034 — \(\alpha\)、response covariance 不再是 production point estimate

当前：

\[
\hat\alpha,\hat\Sigma
\]

作为确定值代入下游的做法：

正式降级为：

\[
\boxed{\text{diagnostic / approximation reference only}}
\]

生产 corrected core 必须积分它们的不确定性。

---

## D-035 — 采用 reference prior，而不是人为 proper prior

统一 mode-space response 模型：

\[
Y=XB+E.
\]

给定 row/output covariance \(\Sigma_R\)：

\[
B\mid\Sigma_R,\alpha_0,\alpha_p
\]

采用 Gaussian / Matrix-Normal prior，feature precision matrix：

\[
\Lambda_\alpha
=
\operatorname{diag}
(
\alpha_0,
\alpha_p,\ldots,\alpha_p
).
\]

超参先验：

\[
\boxed{
p(\alpha_0)\propto1/\alpha_0,
\qquad
p(\alpha_p)\propto1/\alpha_p
}
\]

即 log-scale reference prior。

covariance prior：

\[
\boxed{
p(\Sigma_R)
\propto
|\Sigma_R|^{-(N+1)/2}
}
\]

即 covariance 的 Jeffreys/reference prior。

这些不是通过回测选出来的。

---

## D-036 — improper prior 必须有 posterior-propriety gate

使用 reference prior 的前提：

\[
\boxed{\text{posterior proper}}
\]

工程师必须静态推导 propriety 条件，并用最小解析/数值 reference 测试验证。

若某日不满足：

- 不得偷偷 clamp；
- 不得改成 EB；
- 不得套 inverse-Wishart 超参数。

必须：

\[
\boxed{\text{fail loudly}}
\]

并回 SPEC 审查 proper prior。

---

## D-037 — \(\Sigma_R\) 优先解析积分

当前 Matrix-Normal 结构允许：

给定 \((\alpha_0,\alpha_p)\) 后，

尽可能解析积分：

\[
B,\Sigma_R.
\]

高维 \(\Sigma\) 的 Laplace/quadrature 不作为第一方案。

必须先推导共轭/reference-prior 下的：

- marginal evidence；
- coefficient predictive；
- decision-time \(Bx_t\) predictive；
- Student-t / matrix-t 形式。

只有证明解析路线失败后，才允许数值积分 \(\Sigma\)。

---

## D-038 — alpha 在 log-space 做 deterministic adaptive quadrature

新的 production posterior 要积分：

\[
(\log\alpha_0,\log\alpha_p)\in\mathbb R^2.
\]

不得：

- profile argmax 后当已知；
- 固定 `EB_ALPHA_MIN/MAX` 作为 prior support；
- 让 grid node 数成为策略参数。

数值 integration：

- adaptive；
- tail mass 有证书；
- refinement 有证书；
- 不收敛 fail loudly。

旧 EB optimizer 可以提供：

- mode；
- initial bracket；
- diagnostic comparison。

不能再定义 posterior。

---

## D-039 — macro \(\sigma^2\) 独立 point estimate 被取消

因为 macro 不再独立成一个 regression。

统一 mode-space covariance \(\Sigma_R\) 负责 response working likelihood 的 output covariance。

因此删除：

```text
sig2 = SSE/(n-gamma)
```

作为生产 posterior 的特殊 macro 点估计。

---

# 12. 这不是“纯 generative Bayesian”——正式采用 Modular Predictive 语言

response posterior 与 innovation law 的角色必须分开。

---

## D-040 — Response 模块负责 epistemic uncertainty

它回答：

> 给定有限历史，我们对条件均值算子 \(G\) 有多不确定？

其输出是：

\[
P(\mu_{t+1}\mid\mathcal H_t)
\]

或等价参数 posterior。

---

## D-041 — Innovation 模块负责 out-of-fit aleatoric predictive law

它回答：

> 在 OOF 意义下，真实下一日残差可能是什么样？

它使用 cross-fitted residual。

---

## D-042 — 顶层名称

在 innovation 仍使用经验 / quasi-likelihood 结构时：

禁止称：

> fully Bayesian generative posterior predictive

正式名称使用：

\[
\boxed{
\text{modular posterior predictive}
}
\]

或：

> Bayesian response posterior + cross-fitted semiparametric innovation predictive.

只有未来把 innovation 也做成完整 generative Bayesian law 后，才恢复“fully Bayesian”。

---

# 13. OOF 裁决

## D-043 — OOF 保留，且扩展到新统一 response

每个 fold：

- 独立 fit；
- 独立 \((\alpha_0,\alpha_p)\) posterior integration；
- 不读取 held-out rows；
- 不复用 full-data posterior；
- 昨日同 fold state 只能作为 numerical warm-start。

---

## D-044 — OOF 不是模型选择工具

OOF 只用于：

\[
\epsilon_t^{OOF}
=
y_{t+1}
-
E[y_{t+1}\mid H_t,\text{train excluding fold}]
\]

构造 innovation calibration。

不根据 OOF Sharpe 选：

- bands；
- priors；
- constraints；
- d；
- alpha；
- universe。

---

# 14. Innovation Law 最终裁决

当前 scalar macro scaling 正式退出最终 core。

---

## D-045 — innovation memory 必须是 vector / mode-space 对象

对统一 mode residual：

\[
\epsilon_s\in\mathbb R^N
\]

定义：

\[
\boxed{
V_t(d)
=
\frac{
\sum_{\tau\ge1}
k_d(\tau)
\epsilon_{t+1-\tau}\epsilon_{t+1-\tau}^T
}{
\sum_{\tau\ge1}k_d(\tau)
}.
}
\]

\(V_t(d)\) 在：

\[
[m,\ Q^Te]
\]

mode 坐标中定义。

因此天然包含：

- macro variance；
- relative covariance；
- macro-relative cross covariance；
- sector/mode risk rotation。

---

## D-046 — fractional kernel 继续统一

继续使用同一个 fractional family：

\[
k_d(\tau).
\]

不得：

- macro 一套；
- relative 一套；
- 每资产一套；
- 每 sector 手工一套。

这保持 Maxwell-like 统一。

---

## D-047 — \(d\) 的先验正式定为连续 Uniform(0,1)

\[
\boxed{
d\sim Uniform(0,1)
}
\]

\(d=1\) 可作为闭区间端点极限包含。

当前 `DGRID_V1`：

不再是理论对象。

它只能作为数值 quadrature 节点。

任何非均匀网格都必须带 cell mass。

---

## D-048 — kernel 使用全部可用因果历史

不设固定 rolling window。

对 decision day \(t\)：

使用所有满足 causal 条件的历史 residual。

kernel 自动决定远历史权重。

禁止根据回测设置：

- 63 日；
- 252 日；
- 3 年；
- 5 年。

---

## D-049 — `burn=30` 不再是理论参数

早期没有足够过去数据来形成 \(V_{s-1}\) 的 rows：

直接不进入 likelihood / score。

有效起点由数学可定义性决定。

不再硬编码 “burn 30” 作为模型定义。

---

# 15. Innovation 的 PSD / singular support 裁决

## D-050 — 不允许固定 covariance floor 成为理论

理论对象允许：

\[
V_t(d)\succeq0.
\]

若某方向历史数据不足而 rank deficient：

该方向 variance 可以为 0 / 未识别 support。

不得通过：

\[
V+\delta I
\]

偷偷创造理论风险。

---

## D-051 — inverse square root 使用 support pseudoinverse

标准化时：

\[
V^{-1/2}
\]

理解为：

\[
\boxed{
\text{Moore-Penrose inverse square root on observed positive support}.
}
\]

零特征方向：

- 不逆；
- 不注入随机噪声；
- 保留为 null support。

任何 numerical eigen floor 只能用于浮点分类，并必须做 floor→0 refinement。

---

# 16. Innovation shape 裁决

我们保留经验 tail shape，而不强行 Gaussian 化。

---

## D-052 — standardized residual shape

对每个 \(d\)：

\[
\boxed{
z_s(d)
=
V_{s-1}(d)^{+1/2}
\epsilon_s.
}
\]

这里 \(+\) 表示 support pseudoinverse。

预测时：

\[
\boxed{
\epsilon_{t+1}
=
V_t(d)^{1/2}z_{s'}(d).
}
\]

---

## D-053 — empirical standardized shape 是 2.x 的规范

\(z\) 的 predictive shape 使用历史 OOF standardized rows 的经验测度。

这明确是：

\[
\boxed{\text{semiparametric empirical predictive}}
\]

不是声称已知真实 tail distribution。

优点：

- 保留 tail；
- 保留 skew；
- 保留 cross-mode shock shape；
- 不新增 Student-t df 等手工参数。

---

## D-054 — 禁止 own-row cell stitching

当前：

> shared row 某 asset 缺失 → 单独从该资产 own-row 抽一个 cell

会破坏联合 residual vector 的 cross-sectional dependence。

新规范：

\[
\boxed{
\text{一个 scenario innovation 必须来自一个联合合法 shape row。}
}
\]

不得把不同日期的 asset cells 拼成一个“历史向量”。

---

## D-055 — ragged shape 按 mask/support 处理

历史 standardized row 只能用于与当前 decision risk set **兼容**的 support。

默认严格规则：

> 被用于一个联合 scenario 的历史 residual row，必须覆盖当前需要生成风险的全部 risky assets。

若不满足：

该 row 不进入当前 joint empirical shape pool。

---

## D-056 — 无足够 joint rows 时 fail / 保留 cash，不拼残差

若当前 risky set 没有足够联合历史：

- 不 own-row stitch；
- 不 zero-fill；
- 不假装 covariance full-rank。

优先：

1. 该资产不进入 free risky set；
2. 已持有则 locked；
3. 资金可以留在 cash。

这正是 cash 被纳入 Kelly 的重要原因之一。

---

# 17. \(d\) 的更新身份裁决

经验 shape 不是高斯时，Gaussian covariance likelihood 不再是完整 likelihood。

因此：

## D-057 — \(d\) 使用 Gaussian covariance quasi-likelihood

定义：

\[
\ell_d
=
-\frac12
\sum_s
\left[
\log\det^+ V_{s-1}(d)
+
\epsilon_s^T V_{s-1}(d)^+\epsilon_s
\right]
\]

只在该 row 的有效 support 上计算。

然后：

\[
q(d\mid H)
\propto
\exp(\ell_d)\cdot 1_{(0,1)}(d).
\]

正式名称：

\[
\boxed{\text{quasi-posterior over }d}
\]

不是普通 Bayes posterior。

---

## D-058 — 不引入 generalized-Bayes temperature

learning-rate / temperature：

\[
\eta
\]

固定为 1。

不允许根据回测调。

若未来理论要求 calibration temperature：

必须另立理论版本。

---

# 18. Innovation absolute scale 裁决

## D-059 — 删除 `v_bootstrap` scalar anchor

vector law 已直接包含绝对二阶矩：

\[
V_t(d).
\]

因此不再需要：

\[
\sqrt{v_{forecast}/v_{bootstrap}}
\]

的 scalar anchor。

这不是丢失 absolute scale。

恰恰相反：

\[
\boxed{
V_t(d)
}
\]

自身就是绝对 scale。

---

# 19. 总 predictive law 裁决

最终 corrected 2.x one-step law：

1. 从 response posterior 抽/积分 conditional mean：
   \[
   \mu_{t+1}.
   \]
2. 从 \(q(d\mid H)\) 抽/integrate \(d\)。
3. 从 compatible standardized residual empirical measure 抽 \(z(d)\)。
4. 构造：
   \[
   \epsilon_{t+1}=V_t(d)^{1/2}z(d).
   \]
5. mode field：
   \[
   y_{t+1}=\mu_{t+1}+\epsilon_{t+1}.
   \]
6. 映射回 asset normalized return field。
7. 乘 \(s_1\) 恢复 asset log returns。
8. exponentiate 成 gross returns。

---

## D-060 — response uncertainty 与 innovation noise 不得重复计算

报告必须区分：

\[
\boxed{
\underbrace{
Var(\mu\mid H)
}_{epistemic}
+
\underbrace{
E[V_{\epsilon}\mid H]
}_{aleatoric}
}
\]

scenario generator：

- response posterior draw 负责 mean uncertainty；
- innovation draw 负责 residual uncertainty。

不得把 regression \(\Sigma_R\) 又作为额外 future residual shock重复加一次。

\(\Sigma_R\) 在 response 模块里是 working-likelihood covariance / posterior scaling nuisance。

真正未来 aleatoric risk 由 vector innovation law 提供。

---

# 20. Numerical Integration 最终裁决

当前固定 \(S=300\) 正式退出 production definition。

---

## D-061 — “Full Kelly” 禁止再指固定 sample Kelly

只有同时满足：

\[
I_M(w)\to I(w)
\]

和：

\[
w_M\to w^*
\]

的数值收敛证书后，

才允许称：

> numerical approximation to full Kelly.

---

## D-062 — 生产必须 adaptive

生产决策禁止：

```text
adaptive=false, S=300
```

固定 \(S\) 只允许：

- unit tests；
- artifact replay；
- microbenchmark；
- historical fixture。

---

## D-063 — Halton 不作为最终 production 默认

高维：

\[
D\approx O(N)
\]

下 production numerical integral：

默认目标 backend：

\[
\boxed{\text{nested Owen-scrambled Sobol}}
\]

Halton：

保留为 reference / legacy comparison。

原因是数值积分结构，不是回测表现。

---

## D-064 — 至少两个独立 numerical certificate

### Certificate A — weight convergence

\[
\boxed{
\|w_{2M}-w_M\|_1
\le
\epsilon_w.
}
\]

### Certificate B — utility regret

在更细/独立 audit rule 上：

\[
\boxed{
I_{audit}(w_{2M})
-
I_{audit}(w_M)
\le
\epsilon_U.
}
\]

Kelly solver 自己的 KKT：

是第三种 certificate。

三者不能互相替代。

---

## D-065 — integration convergence 必须使用独立 audit replicate

因为：

\[
w_M
\]

是在 optimization scenarios 上选择出来的。

只用同一 scenario set 检查 objective：

会存在 sample optimization optimism。

因此 production 至少维护：

- optimization nested rule；
- independent audit scrambled rule。

两者 seed 可确定性派生。

---

## D-066 — 数值容差不在本裁决书里拍脑袋

\[
\epsilon_w,\epsilon_U,M_{max}
\]

必须通过 numerical refinement 定稿：

1. synthetic laws 有解析答案；
2. realistic single-day fixtures；
3. tolerance 减半；
4. weight / utility 稳定；
5. 与 Sharpe/PnL 完全无关。

一旦定稿：

记录为 numerical configuration。

---

## D-067 — integration budget 耗尽必须 fail loudly

错误统一：

```text
Numerical integration did not converge
```

不得返回最后一层权重。

---

# 21. Kelly feasible set 最终裁决：cash 回归

这是本轮第二个非常重要的理论纠正。

---

## D-068 — cash 是 Kelly 的显式 numeraire asset

无利息 cash gross return：

\[
R_{cash}=1.
\]

risky weights：

\[
w_i\ge0.
\]

cash：

\[
w_c\ge0.
\]

总预算：

\[
\boxed{
w_c+\sum_iw_i=1.
}
\]

等价：

\[
\boxed{
\sum_iw_i\le1.
}
\]

剩余自动留 cash。

---

## D-069 — fully-invested risky simplex 降级为特殊实验约束

当前：

\[
\sum_i w_i=1
\]

且没有 cash：

不再称“canonical exact Kelly”。

它只是一种：

> fully-invested risky-only Kelly constraint.

如果以后要测试：

可以作为显式 experiment。

不能作为默认理论。

---

## D-070 — cash 不是 fractional Kelly

允许留 cash 是 exact feasible set。

它不是：

- 人工缩仓；
- 置信度乘数；
- 0.5 Kelly；
- volatility targeting。

如果 posterior 没有足够 edge：

exact Kelly 自己选择 cash。

---

## D-071 — 不允许仓位上限修 concentrated output

禁止：

- `max_weight=0.1`；
- entropy penalty；
- diversification penalty；
- risk parity blend。

除非未来 execution / mandate 明确要求。

这些都不是当前纠偏工具。

---

# 22. Benchmark 裁决

## D-072 — 至少保留三个 benchmark

未来正式报告：

1. **Cash**
   \[
   R=1.
   \]
2. **Daily equal-weight risky**
3. **Initial equal-weight buy-and-hold**

其中 2 vs 3 才能讨论 rebalancing premium。

---

## D-073 — 禁止未经识别称“Volatility Pump”

只有：

\[
EW_{daily}
-
EW_{buyhold}
\]

的 counterfactual 分解完成后，

才能讨论 rebalancing bonus。

---

## D-074 — correlation 不能替代 beta decomposition

收益相关：

\[
corr(r_{PK},r_{EW})=0.4184
\]

只能描述相关性。

不能写：

> 完全脱离大盘 beta。

若要讲 beta：

必须定义 benchmark factor 并做明确回归/投影。

---

# 23. Concentration 诊断裁决

## D-075 — 高集中不是独立 bug

未来：

\[
w_{max}>95\%
\]

只有在以下均通过后才可解释为理论答案：

1. posterior object 正确；
2. innovation law 正确；
3. scenario integration 收敛；
4. Kelly KKT 通过；
5. cash feasible；
6. decision utility margin 显著。

---

## D-076 — 必须输出 concentration cause report

对每个高集中日，记录：

- top asset；
- risky weight；
- cash weight；
- posterior expected log-growth；
- response epistemic variance；
- innovation variance；
- utility regret if 1% weight shifted to next alternatives；
- integration \(M\)；
- convergence certificate；
- top posterior mean directions；
- top innovation covariance eigenmodes。

不能只打印：

> “Kelly concentrated.”

---

# 24. Geometry uncertainty 最终裁决

## D-077 — 不恢复旧 geometry bootstrap

旧 eigenmodes 的 bootstrap uncertainty：

主要包含：

- eigenvector sign；
- degenerate rotation；
- coordinate tracking；

这些已经由 fixed gauge 消灭。

因此：

\[
\boxed{
\text{不把旧 }\Phi\text{ bootstrap 恢复进 posterior。}
}
\]

---

## D-078 — 价格几何若未来成为未知 law，必须另立理论

例如想让：

- ruler exponent H；
- scale functional；
- kernel family；

本身随机：

必须在未来版本显式定义 prior。

不能把 sampling variability 随手叫 posterior uncertainty。

---

# 25. BANDS / TAUS 裁决

## D-079 — 当前 BANDS/TAUS 是 2.x 模型类定义

它们不是：

- 市场本体；
- 可用 Sharpe 调的超参数；
- 当前需要做 scenario refinement 的 numerical quadrature。

改变：

```text
TAUS
BANDS
```

视为：

\[
\boxed{\text{模型类 / 理论版本变更}}
\]

需要 separate theory review。

---

## D-080 — 不在本轮同时研究 continuum bands

Gate 0 当前目标是恢复 predictive law 的一致性。

不要同时：

- 连续化 band；
- 加 wavelet family；
- 增加更多 temporal basis。

一次只解决一层问题。

---

# 26. API / 类型裁决

纠偏后建议的最小核心类型：

```julia
struct MarketFacts
    dates
    symbols
    close
    adj
    observed
end
```

```julia
struct Eligibility
    model_admitted
    trade_eligible
    executable
end
```

```julia
struct PreparedProblem
    # geometry
    ...
    # unified mode design/targets
    X
    Y
    # fold stats
    ...
    # observation/mask provenance
    ...
end
```

```julia
struct ResponsePosterior
    # posterior quadrature / analytic representation
    ...
end
```

```julia
struct InnovationState
    # OOF vector residuals / V_t(d) / d quasi-posterior
    ...
end
```

```julia
struct PredictiveLaw
    ...
end
```

---

## D-081 — `V1Model` 大杂烩结构必须拆

当前把：

- response；
- mean；
- residual provider；
- d posterior；
- bootstrap scale；

塞在一个 model struct。

纠偏后每个数学概念有一个 owner。

---

# 27. Reference 实现裁决

## D-082 — 先写新的 slow reference，不在当前 fast core 上继续补丁

旧 2.0 production：

保留为 historical fixture。

新 Gate-0 reference：

- 单线程；
- 无 incremental；
- 无 workspace cleverness；
- 无 scheduler；
- 无 GPU；
- 无 special performance path。

先把新数学写清楚。

---

## D-083 — Reference 不追求 1 分钟

Gate 0 reference 即使：

- 单日慢；
- 两年不能跑；

也可以接受。

数学正确优先。

---

# 28. 旧 conditioned-EB 代码裁决

## D-084 — 不继续优化它

`optimize_conditioned_eb`：

已经证明工程上做得非常认真。

但由于：

- trace neutrality 已降级；
- plug-in EB 已退出 production posterior；

它不再是 corrected core 的中心。

处理方式：

1. 保留 source/history；
2. 保留 tests；
3. 移入 legacy/hypothesis；
4. 不再继续性能优化。

不要因为 sunk cost 强行保住理论地位。

---

# 29. 数值 integration of response hyperparameters 的实现顺序

## D-085 — 能解析绝不采样

给定 \((\alpha_0,\alpha_p)\)：

优先解析积分：

- coefficient matrix；
- response covariance nuisance；
- decision-time linear form。

然后只对低维 hyperparameters 做 deterministic quadrature。

---

## D-086 — 不允许 MCMC 作为第一版

两维 alpha：

先 adaptive deterministic quadrature。

只有证明：

- posterior 多峰；
- deterministic integration 不可承受；

才重开 MCMC。

---

# 30. 结果报告必须新增的不确定性分解

未来单日 model audit 必须给：

\[
\boxed{
Var(r\mid H)
=
Var_{response}
+
Var_{innovation}
}
\]

并进一步把 response 拆：

- coefficient conditional；
- \(\alpha_0,\alpha_p\) hyperposterior；
- covariance nuisance integration。

任何无法计算部分：

必须写：

> NOT COMPUTED

不能显示 0。

---

# 31. 旧 `S=300` 回测怎么处理

## D-087 — 保留为 regression fixture

两年 -62.22%：

永久保留。

它用于未来回答：

> 哪个数学纠偏改变了什么。

但禁止拿它作为：

- performance target；
- tuning objective；
- acceptance threshold。

---

# 32. 纠偏后的实验纪律

## D-088 — 第一阶段完全禁止两年重跑

直到下面全部完成：

1. new SPEC；
2. static owner review；
3. tiny analytic tests；
4. single-day integration convergence；
5. OOF isolation；
6. innovation PSD/support tests；
7. cash Kelly tests；
8. reference vs formulas。

之后才允许短窗口。

---

## D-089 — 测试升级阶梯

固定：

\[
\boxed{
\text{static}
\rightarrow
\text{tiny analytic}
\rightarrow
\text{single day}
\rightarrow
5\text{ days}
\rightarrow
20\text{ days}
\rightarrow
60\text{ days}
\rightarrow
501\text{ days}
}
\]

任何一级红：

立即停止升级。

---

# 33. 需要新增的最小解析测试

## D-090 — Symmetric world

构造 \(N=3\)：

三个资产 predictive law 完全 exchangeable。

有 cash 且 risky expected log-growth > cash 时：

risky 权重必须对称。

若 risky law 没有正优势：

允许全部 cash。

---

## D-091 — No-signal world

response posterior：

\[
E[G]=0.
\]

innovation 对称。

Kelly 不得由于 finite scenario 偶然赢家稳定地产生 95% 单票。

integration refinement 后应收敛到对称 solution / cash。

---

## D-092 — Known cross-mode world

人工：

\[
G_{\perp m}\neq0
\]

或：

\[
G_{m\perp}\neq0.
\]

新 full response 必须 recover。

旧 block-diagonal 实现应故意失败，用于证明新模型确实修复了被删 block。

---

## D-093 — Known DC world

人工 constant drift：

新 DC channel recover。

dynamic Q/P coefficients 保持 0。

---

## D-094 — Trace-hypothesis counterexample

构造真实 diagonal common response：

验证：

- unconstrained new core 能 recover；
- old trace-neutral branch 会系统性删除。

这是 trace constraint 从 core 降级的重要反例测试。

---

## D-095 — Vector volatility world

人为让某一 relative direction volatility 最近显著上升。

新 \(V_t(d)\) 必须只放大该方向。

旧 scalar law 会错误放大全场。

---

# 34. OOF tests

必须证明：

- held-out row 不进 response posterior；
- held-out row 不进 alpha quadrature evidence；
- fold residual 与 dense reference 一致；
- unified macro-relative cross blocks也严格 OOF；
- row with no observations 不得当 residual=0 进入 likelihood。

---

# 35. Ragged tests

必须覆盖：

1. dummy all-NaN asset；
2. IPO；
3. 单日 missing；
4. held but unobservable；
5. current free set 是 historical subset；
6. no compatible joint residual shape rows；
7. asset permutation；
8. gauge rotation。

---

# 36. Numerical integration tests

至少：

- known Gaussian 2-asset analytic Kelly；
- nested \(M\to2M\)；
- optimization/audit independent replicate；
- same seed replay；
- different seed convergence to same weight；
- budget exhaustion fail-loud；
- no fixed-S production shortcut。

---

# 37. Cash Kelly tests

必须覆盖：

### Case A

所有 risky gross deterministically < 1：

\[
w_{cash}=1.
\]

### Case B

一资产 deterministically > 1：

\[
w_{asset}=1.
\]

### Case C

高均值高风险 risky：

根据 exact log utility 选择内部解。

### Case D

locked position：

cash + free + locked wealth 正确相加。

---

# 38. Naming 裁决

## D-096 — 以下名称立即禁用

直到条件满足：

- “Exact Bayesian Kelly”
- “Full Bayesian posterior”
- “Full Kelly” 用于固定 300 scenario
- “Trace neutrality is an identification theorem”
- “Low SNR causes Kelly concentration”
- “Volatility Pump explains EW outperformance”
- “Completely detached from beta”

---

## D-097 — Gate-0 期间推荐名称

当前旧实现：

> KTrader 2.0 RC legacy implementation.

新纠偏模型：

> Path Kelly modular predictive reference.

当 numerical integration 收敛：

> converged log-Kelly decision.

---

# 39. 回测报告最终模板裁决

未来正式报告必须至少包含：

### Data

- hashes；
- observed mask policy；
- model admission；
- trade eligibility；
- execution mask。

### Model

- response posterior definition；
- DC prior；
- alpha prior；
- innovation definition；
- trace hypothesis on/off；
- bands/version。

### Numerical

- quadrature backend；
- M per day distribution；
- fail count；
- weight convergence；
- utility convergence；
- KKT certificate。

### Portfolio

- risky exposure；
- cash weight；
- max single risky weight；
- locked weight。

### Predictive diagnostics

- epistemic variance；
- innovation variance；
- \(d\) quasi-posterior；
- \(V_t\) spectrum；
- concentration utility margin。

### Benchmarks

- cash；
- EW daily；
- EW buy-hold。

### PnL

最后才给。

---

# 40. 下一阶段保姆级施工顺序

必须严格按以下顺序。

---

## Step 0 — 只改文档

先把本裁决正式并入：

- SPEC；
- ROADMAP；
- README release status。

不改 runtime。

静态审查无冲突后再开 Step 1。

---

## Step 1 — 拆 observation / eligibility

实现：

- `observed`
- `model_admitted`
- `trade_eligible`
- `executable`

删掉所有“改 bar 表达政策”的代码。

此步只改数据语义。

不碰 posterior。

---

## Step 2 — Kelly cash feasible set

把：

\[
\sum risky=1
\]

改成：

\[
\sum risky\le1
\]

或显式 cash column gross=1。

保留 exact same objective。

先完成 tiny analytic tests。

---

## Step 3 — 统一 mode coordinate producer

只做：

\[
[m,\ Q^Te]
\]

以及：

\[
[B_m,\ B_\perp].
\]

暂不改 posterior。

验证：

- reconstruct；
- permutation；
- gauge。

---

## Step 4 — Full block response reference

允许四块：

\[
G_{mm},
G_{m\perp},
G_{\perp m},
G_{\perp\perp}.
\]

先用固定 ridge / simple Gaussian reference，

只为验证 design/target algebra。

不要立刻上 full posterior。

---

## Step 5 — 加 DC channel

在 full response 中加 constant feature。

验证 known-DC synthetic world。

---

## Step 6 — 移除 production trace neutrality

默认 core 不调用 14 constraints。

旧 conditioned branch 留 dev。

验证 trace counterexample。

---

## Step 7 — 推导 full response posterior

在纸面/Markdown先完成：

- reference prior；
- posterior propriety；
- analytic integration；
- alpha marginal evidence；
- decision-time predictive。

静态 review 完成以前：

\[
\boxed{\text{严禁写复杂 optimizer}}
\]

---

## Step 8 — 写 slow full-posterior reference

先单日。

不优化。

---

## Step 9 — 重建 OOF full-mode folds

每 fold 全 posterior。

dense residual reference 先实现。

lazy oracle 以后再说。

---

## Step 10 — 写 vector innovation reference

直接 \(O(T^2N^2)\) 都可以。

这是 reference。

先证明数学。

禁止先 FFT / incremental。

---

## Step 11 — standardized empirical shape

实现同 mask joint row。

删除 own-cell stitching。

---

## Step 12 — continuous d quasi-posterior

先 direct sums。

不要 FFT。

---

## Step 13 — 组合 PredictiveLaw

只做 one-day scenario/reference draws。

---

## Step 14 — adaptive RQMC

实现 nested Sobol + audit replicate。

先 tiny/single day。

---

## Step 15 — 端到端 Kelly

full posterior + vector innovation + converged integration + cash。

至此才重新拥有：

\[
\mathcal H_t
\to
P(r_{t+1}\mid H_t)
\to
w_t^*.
\]

---

## Step 16 — Gate-0 constitutional suite

全部通过后，

才宣布 Gate 0 closed。

---

# 41. Gate 0 Exit 条件

以下全部为真：

- [ ] `Bars.bar` 只表示 observation；
- [ ] model/trade/execution mask 分离；
- [ ] benchmark universe 外生；
- [ ] full mode response 含 cross blocks；
- [ ] DC channel 存在且 zero-centered；
- [ ] 14 trace constraints 不在 default core；
- [ ] response hyperparameters 不再 point plug-in；
- [ ] posterior propriety 有证明；
- [ ] vector innovation law 实现；
- [ ] no own-cell stitching；
- [ ] d prior 显式；
- [ ] fixed S production path 删除；
- [ ] integration 有 independent audit certificate；
- [ ] cash 在 feasible set；
- [ ] OOF full isolation；
- [ ] permutation / gauge / dummy invariance；
- [ ] synthetic worlds 全部通过；
- [ ] 所有 fail condition fail loudly。

少一项：

\[
\boxed{\text{Gate 0 不得关闭。}}
\]

---

# 42. Gate 0 后的数学加速清单

Gate 0 closed 后才允许：

1. 解析积分 \(B,\Sigma_R\)；
2. alpha quadrature cache；
3. sufficient statistics；
4. vector fractional convolution FFT；
5. direct \(Gx\) marginalization；
6. mode-space Kronecker identities；
7. joint residual lazy evaluation；
8. exact low-rank covariance actions。

---

# 43. 数学加速完成后才能恢复 CS 优化

之后才：

- workspace；
- BLAS；
- threading；
- allocations；
- precompile；
- scheduler；
- file spool；
- cache。

---

# 44. CS 加速完成后才能恢复 incremental

新的 incremental 必须针对**新 PreparedProblem**。

旧 incremental engine：

不得因为投入太大而绑架新数学。

若新 model 结构变化让旧 engine 无法复用：

\[
\boxed{\text{允许删除重写。}}
\]

---

# 45. GPU 仍然最后

不变。

只有：

- 数学正确；
- 数学加速做尽；
- CPU CS 做尽；
- incremental 做尽；

才打开 GPU。

---

# 46. 哪些旧资产必须保留

保留：

- 原 2.0 source snapshot；
- two-year failure artifact；
- hashes；
- conditioned EB tests；
- trace hypothesis branch；
- old response semantic diff；
- residual dense reference；
- old incremental equality fixtures。

它们是历史/反例资产。

不是 production owner。

---

# 47. 哪些概念必须删除或降级

默认 core 删除/降级：

- per-band trace hard constraint；
- separate macro-only response；
- separate relative-only response；
- plug-in EB predictive；
- scalar fractional scale；
- `v_bootstrap`；
- own-row cell stitching；
- fixed-S production；
- risky-only forced full investment；
- `Bars.bar` policy mutation；
- benchmark dependency on model.active；
- causal story “low SNR → concentration”。

---

# 48. 什么绝对不能作为“救回测”的工具

禁止：

- 10% 单票上限；
- 20% 单票上限；
- fractional Kelly；
- half Kelly；
- entropy；
- risk parity；
- volatility targeting；
- sector caps；
- manual blacklist；
- “ZIM 太危险”特判；
- 改 seed；
- 根据 2024-2026 结果选 prior；
- 根据 2024-2026 结果开/关 trace；
- 根据 2024-2026 结果选 d；
- 根据 2024-2026 结果选 bands；
- 根据 2024-2026 结果选 S。

---

# 49. 如果纠偏后仍然亏损怎么办

如果：

- Gate 0 完整通过；
- numerical integration 收敛；
- cash 正确；
- posterior uncertainty 完整；
- innovation vector risk 完整；

然后两年仍然显著落后：

接受结果。

那时结论可以是：

\[
\boxed{
\text{当前 price-only finite path response model 没有足够预测力。}
}
\]

然后才进入新的理论研究。

不能因为亏损再偷偷回到旧缺陷。

---

# 50. 如果纠偏后突然大幅赚钱怎么办

也不能宣布成功。

必须首先检查：

- 是否引入未来信息；
- 是否 benchmark universe 不同；
- 是否 trade eligibility 泄漏；
- 是否 integration tolerance 太松；
- 是否 prior 不proper；
- 是否 OOF 泄漏；
- 是否 cash accounting 错；
- 是否 test period 被用于选择设计。

收益不是 correctness certificate。

---

# 51. 本轮最关键的七项裁决摘要

若工程师只能记住七件事：

### 1

\[
\boxed{\text{2.0 Final 数学地位撤销，Gate 0 重开}}
\]

### 2

\[
\boxed{\text{14 条 trace neutrality 不再进入默认 core}}
\]

### 3

\[
\boxed{\text{macro / relative 必须允许完整 cross-response}}
\]

### 4

\[
\boxed{\text{恢复零中心 DC channel，不恢复正漂移作弊}}
\]

### 5

\[
\boxed{\text{response hyperparameters 必须积分，不再 plug-in EB}}
\]

### 6

\[
\boxed{\text{innovation 从 scalar scale 升级为 vector fractional law}}
\]

### 7

\[
\boxed{\text{cash + converged integration 才是 canonical Kelly}}
\]

---

# 52. 最终理论链

纠偏后的 2.x reference target：

\[
\boxed{
\mathcal H_t
\rightarrow
\text{deterministic path geometry}
\rightarrow
\text{unified macro-relative path coordinates}
\rightarrow
\Pi(b_0,G,\alpha_0,\alpha_p,\Sigma_R\mid\mathcal H_t)
}
\]

\[
\boxed{
\rightarrow
\text{OOF vector innovations}
\rightarrow
q(d\mid\mathcal H_t)
\rightarrow
V_t(d)
\rightarrow
P(\epsilon_{t+1}\mid\mathcal H_t)
}
\]

\[
\boxed{
\rightarrow
P(R_{t+1}\mid\mathcal H_t)
\rightarrow
\arg\max_{\substack{w_i\ge0\\\sum_iw_i\le1}}
E[\log(1+w^\top r_{t+1})\mid\mathcal H_t]
}
\]

其中：

\[
w_{cash}=1-\sum_iw_i.
\]

这才是下一阶段所有工程工作的唯一目标。

---

# 53. 工程经理验收话术

今后任何工程师声称：

> “完成了”

经理只问：

1. 你完成的是哪个 Gate？
2. 数学对象写出来是什么？
3. 和 reference 的差异是什么？
4. 哪些不确定性被积分？
5. 哪些还是 plug-in？
6. numerical integral 收敛证书在哪里？
7. cash 是否在 feasible set？
8. OOF 是否完全 held-out？
9. innovation 是 scalar 还是 vector？
10. 有没有用回测表现做任何选择？

任何一个答不清：

\[
\boxed{\text{不验收。}}
\]

---

# 54. 本裁决的最终精神

这次失败最有价值的地方，不是证明“Kelly 太激进”。

它证明：

> 一个局部每一步都能写出漂亮公式、都有 certificate 的系统，仍然可能在对象层级上逐渐偏离最初想要的理论。

因此后续只有一条纪律：

\[
\boxed{
\text{先钉死对象，再证明算法，再优化计算。}
}
\]

以及：

\[
\boxed{
\text{如果一个限制没有从理论推出，就不要因为它曾经写进 SPEC 很久而把它当定理。}
}
\]

还有：

\[
\boxed{
\text{如果一种不确定性真实存在，就不要因为积分很贵而把它变成点估计，再把条件 posterior 称为完整 posterior。}
}
\]

最后：

\[
\boxed{
\text{如果 numerical approximation 尚未收敛，它就不是策略答案，只是计算中的中间数。}
}
\]

---

# 55. 立即执行清单

工程师收到本文件后，第一阶段只做：

- 更新规范；
- 标记 release 状态；
- 画出新 mode-space response 的维度图；
- 推导 reference posterior；
- 推导 vector innovation；
- 设计新数据 mask 类型；
- 写 static implementation plan。

**不要运行任何测试。**

直到静态设计审查完成。

下一步才进入最小 reference 编码。

---

# 56. 明确废止的旧表述

以下句子在新 SPEC 中必须删除或改写：

> “Strict Trace Neutrality 是 identification requirement。”

改成：

> “历史 2.0 分支曾采用 per-band trace-neutral hypothesis；Gate-0 审计未找到其 identification theorem，因此已从默认 core 移出。”

---

> “当前模型是 exact Bayesian Kelly。”

改成：

> “历史 2.0 RC 使用 conditioned plug-in response posterior、scalar fractional innovation 与 fixed-scenario sample Kelly；新 Gate-0 reference 正在恢复完整 uncertainty 与 numerical convergence。”

---

> “低信噪比导致 Kelly 极端集中。”

改成：

> “历史 RC predictive law 经常给 Kelly 足以产生极端集中仓位的信念；该信念是否正确尚未通过完整 uncertainty 与 integration convergence 审核。”

---

> “Equal weight 的胜利来自 volatility pump。”

改成：

> “Daily equal-weight 在该历史窗口胜出；rebalancing contribution 尚未通过 buy-and-hold counterfactual 分解。”

---

> “相关 0.4184 说明完全脱离 beta。”

改成：

> “Path Kelly 与 daily equal-weight 的日收益相关为 0.4184；该统计量不能单独识别 beta exposure。”

---

# 57. 版本命名建议

在 Gate 0 完成前：

```text
2.0.0       historical released artifact
2.0-RC-G0   current corrective development line
```

Gate 0 通过但性能尚未恢复：

```text
2.0.1-reference
```

数学加速 / CS / incremental 全部重新通过后：

```text
2.1.0 CPU
```

GPU 若未来完成：

```text
2.2.0 GPU backend
```

版本号最终由 owner 决定，但不得再用 “Final” 掩盖 Gate 状态。

---

# 58. 最后一条裁决

本项目现在不缺“再多一个聪明机制”。

它缺的是：

\[
\boxed{
\text{从最少、最统一、真正被证明的数学对象重新长出代码。}
}
\]

因此：

> **所有不能证明属于理论的约束，降级。**  
> **所有真实存在但未传播的不确定性，恢复。**  
> **所有数值离散，必须收敛。**  
> **所有执行/eligibility 事实，不能污染市场 observation。**  
> **所有 portfolio 风险，都交给正确的 predictive law + exact Kelly + cash，而不是事后加帽子。**

这就是本轮全部裁决。

---

# KTrader / Path Kelly — 旧的权威 SPEC

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
==> docs/INNOVATION_LAW.md <==
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

==> docs/MODEL_LEDGER.md <==
# KTrader 理论对象与生产实现对账账本（Model Ledger）

状态：纯静态事实账本。生成方式为逐文件静态阅读，**未运行任何命令、未执行任何测试、未修改任何源码**。所有行号来自本次阅读时的工作树快照；证据标记：[事实] = 代码直接支持；[推断] = 由代码语义纯静态推导；[未知] = 本次阅读无法确定。

对账对象：AGENTS.md 声明的理论对象 与 当前生产代码实际计算的对象。差距分类标签：identical / plug-in approximation / scalar degeneration / finite-sample approximation / unknown。

---

## 0. 证据方法与边界

- [事实] 本次已读：src/data.jl、src/prepare.jl、src/predict.jl、src/response.jl、src/residual_oracle.jl、src/kelly.jl、src/backtest.jl 全文；src/geometry.jl（常量与 ruler 段）、src/numerics.jl（PriceHistoryCache 段）、src/broker.jl（tradable/liquidation/rebalance 段）、universe.txt 全文；若干 dev 脚本与测试的相关段。
- [事实] 账本唯一新建文件即本文件；未触碰 src/、test/、README.md、AGENTS.md。
- [事实] 全仓检索 PONY：唯一命中 universe.txt:46；未找到把标的"前 251 日 bar 置 false"的 wrapper 脚本（检索模式含 PONY、251、.bar[...]=false、.bar[1: 等）。详见第 11 节。
- 理论声明引用 AGENTS.md 的章节号（§编号），不对 AGENTS.md 编造行号。

---

## 1. History H_t：进入模型的信息

理论（AGENTS.md §3/§4/§8）：H_t 是完整价格历史；x_i(t)=log P_i(t)；missing 不得被 carried marking 冒充为 r=0。

[事实] 数据结构：Bars 含 dates/symbols/close/adj/bar（src/data.jl:7-13）。close/adj 是 marking series：上市前 NaN，之后 carry 前值（src/data.jl:15-25；docstring src/data.jl:2-4）。

[事实] 理论输入由 signal_prices(b)=ifelse.(b.bar, b.adj, NaN) 恢复（src/data.jl:37；docstring src/data.jl:33-36）：bar=false 之日严格 NaN，绝不把 carried marking 注入为 0 收益。

[事实] 模型入口消费 signal：_prepare_v1 内 log_adj=log.(adj_act)（src/predict.jl:216）；r=diff(log_adj; dims=1)（src/predict.jl:247）——只有相邻两日均 finite 才有 return。

[事实] backtest 的模型输入是 signal=signal_prices(b)（src/backtest.jl:149）；账户 marking 另用 b.adj[t+1,:]./b.adj[t,:]（src/backtest.jl:296）。两条路径互不冒充。

[事实] zero-embedding 只发生在 relative field 内部（src/predict.jl:43-70；注释 38-42），且 observed mask 单独保留（src/predict.jl:68）。这不是把 missing return 当 0。

[推断] 模型可见的 H_t 即 signal 的 finite 格；carried adj 不进入 r（r 基于 signal 的 log）。

[未知] 本地 panel CSV 的原始 provenance 沿用 AGENTS.md 的 unknown 记录，不在本账本重判。

分类：identical（§8/§8.1/§8.2 语义与 signal_prices、diff 实现一致）。

---

## 2. Observation mask O_{t,i}：真实市场观测在哪里表达

[事实] 市场真实观测的权威字段是 Bars.bar[t,j]（src/data.jl:12；注释 3-4）。4 参数构造：bar=isfinite.(rawclose).&isfinite.(rawadj)（src/data.jl:28）。

[事实] 保存时 bar 决定落盘缺失：ifelse.(b.bar, b.close, NaN) / ifelse.(b.bar, b.adj, NaN)（src/data.jl:80-82）。

[事实] return 级观测 mask：observed=isfinite.(r)（src/predict.jl:68）。

[事实] 决策日 support：alive_now 默认取 adj_act 末行 isfinite（src/prepare.jl:131），构造时强制 owned copy（src/prepare.jl:160-161）；solve 用 findall(prep.alive_now) 计算 e0（src/predict.jl:473-478）。

[事实] ResidualOracle 每行 own mask：O={j: isfinite(r[t+1,j])}，c=|O|（src/residual_oracle.jl:19-20；evaluation 160、198、283、290-291）。

[事实] missing-cell fallback 只使用该资产自身 observed 残差行（src/predict.jl:596-602；src/residual_oracle.jl:262-274）。

[推断] 三处表征同源且互不矛盾：bar → signal NaN → 无 return → 不进 own rows。alive_now 是决策日（末行）快照，observed 是 return 行级。

分类：identical。

---

## 3. Model admission A^model：active universe 链

[事实] active_universe_indices(adj)：资产当且仅当存在 t in 2:T 使 adj[t-1,j] 与 adj[t,j] 均 finite（src/predict.jl:125-135；判定在 130 行）。即"至少一条有效 daily return"。

[事实] cache 路径：active_idx=findall(<(T_raw), history_cache.first_return)（src/predict.jl:207-208）；first_return 定义为第一个 finite returns 行（src/numerics.jl:150）。

[推断] 两路等价：第一个 finite return 行 t0 对应首个相邻 finite 对 (t0, t0+1)；first_return < T_raw 等价于该对落在 prefix 内。

[事实] warmup：WARMUP=2*max(BANDS)=256（src/geometry.jl:14；BANDS=2..128，src/geometry.jl:12）；训练行 ts_total=WARMUP:T-2（src/predict.jl:286）；F_folds 校验 2<=F_folds<=n_res（src/predict.jl:288）。

[事实] 消费者：ruler（src/predict.jl:222-243）、X_rel/s_perp/B_m（src/predict.jl:256-281）、full/fold Gram（src/predict.jl:300-305）、solve 的 e0（src/predict.jl:473-478）、Kelly 的 active 掩码（src/kelly.jl:152-153）、backtest free（src/backtest.jl:241-245）。

[事实] 全 NaN dummy 资产严格排除（src/predict.jl:122-123 注释）；e0 只在 alive_now 上归一 e0_now[alive_now].=1/sqrt(N_alive)（src/predict.jl:475-478）；mu_asset_full 只写 active_idx（src/predict.jl:485-486），inactive 留 0 而不进入坐标空间。

[事实] 退化特例：N==1 时相对空间维为零、响应精确为零（src/response.jl:1156-1163）；N==2 有构造性 witness 解析候选（src/predict.jl:371；src/response.jl:832-851）。

[推断] "进入 active"（坐标空间）与"可训练"是两道门：active 需要 ≥1 条有效 return；训练还需要 ts_total 非空（T_raw>=258 才有 >=1 行，>=259 才有 >=2 行）；ruler 另有 4*tau<=T-f+1 门槛（src/geometry.jl:38）。

分类：identical（§11 的实现一致）。

---

## 4. Trading eligibility E^trade：是否存在独立概念

[事实] 当前代码中没有名为 E^trade 的独立对象（本账本阅读范围内未见）。承担者是 backtest 的 free 掩码：free=falses(N)；for j in active; free[j]=b.bar[t,j]（src/backtest.jl:242-245）——即 free = active ∩ bar(决策日 t)。

[事实] Kelly 层：free=tradable .& active；locked=current .* .!free；budget=1-sum(locked)（src/kelly.jl:152-156）。参数名 tradable 在 backtest 调用处传入 decision.free（src/backtest.jl:80,291-292）。

[事实] path_kelly_v1 的 tr=tradable===nothing ? trues(N) : tradable（src/kelly.jl:177）。

[推断] bar 同时承担两个语义：观测 mask（§2）与"当日可交易"（src/data.jl:3-4 注释明确 bar "was tradable"）。

分类：unknown（理论未见独立 E^trade 声明；实现是调度级掩码，且与观测 mask 共用 bar 字段）。

---

## 5. Physical tradability T：broker/tradable 语义

[事实] broker.tradable(q)=q.bid>0 && q.ask>0 && q.last>0（src/broker.jl:136），语义为"有活跃双边市场"；halted/unquoted → false。

[事实] rebalance! 对 tradable=false 的标的跳过操作、保留仓位（src/broker.jl:192）；target_shares 由权重与 last 价取整（src/broker.jl:143）；liquidation 对外部持仓渐进卖出（rate in (0,1]，src/broker.jl:155-168）。

[事实] 该层属于 execution（AGENTS.md §5/§41）；backtest 的 bar 掩码不经过 broker.tradable。

分类：identical（物理可交易性由 broker 语义承担，理论层不消费）。与第 4 条的区别：T 是报价级、E^trade（free）是 bar 级，二者当前不互相校验。

---

## 6. Response posterior：solve→fit_response_operator→optimize_conditioned_eb 实际链

[事实] 链：solve(prep)（src/predict.jl:349-505）→ fit_response_operator（src/response.jl:1124-1209）。

[事实] macro 分支：optimize_matrix_normal_eb 求 alpha_m 点估计（src/response.jl:1146-1148；实现与证书 351-379）；gm=vec(ev.vectors*(B.*dm))（1151）；sig2=max(sse/max(n-gamma_m,1),1e-8)（1154）；covm=sig2.*((ev.vectors .* dm')*ev.vectors')（1155）。

[事实] relative 分支：ridge_spectrum（src/response.jl:1174-1176）→ optimize_conditioned_eb 求 (alpha_rel, Sigma) 点估计（src/response.jl:1178-1185；实现 934-1121，含 witness 分支 943-953 与逐迭代证书 994-1000）。

[事实] G：B_scaled_G=spectrum.B .* dr（1188）；G=B_scaled_G'*spectrum.basis'（1191）；V=ridge_covariance(spectrum,alpha_rel)（1199）；cond=condition_trace_neutrality(G,V,Sigma,N,len(BANDS))（1200-1202；实现 131-143）。

[事实] OOF folds：need_uncertainty=false（src/predict.jl:443），V_out 变为空 RidgeCovariance（src/response.jl:1205）——fold 只消费条件均值。

[事实] predictive_moments（src/response.jl:1211-1229）：mu_m=dot(G_macro,Bm)（1218）；var_m=max(dot(Bm,post_cov_m*Bm),0)（1219）；mu_rel=G_c_mean*x（1220）；cov=Symmetric(dot(x,v).*Sigma_rel − H*inv_M_constraint*H')（1226）；L_rel=ev.vectors.*sqrt.(max.(ev.values,0))'（1228）。

[事实] 场景采样：relative=principal_sqrt_root(L_rel)*z + mu_rel（src/predict.jl:570-572）；macro_draw=mu_m+sqrt(var_m)*z（src/predict.jl:612）。

[推断] 实际对象是 Pi(dG | alpha_bar, Sigma_hat, H)：G 在 EB 点估计 (alpha_bar, Sigma_hat) 条件下为 Gaussian（均值 G_c，协方差为 V⊗Sigma 经 trace 约束条件化后的预测边缘）；**不是**对 (alpha, Sigma) 的联合后验积分——alpha、Sigma 是 EB 最优点（response.jl:374-379、1178-1185），其自身不确定性不作为 predictive 方差项进入。

[推断] G 是条件采样（scenarios 从 L_rel 与 var_m 抽），不是只用 G_c_mean；macro 的 var_m 与 relative 的 L_rel 都携带条件后的参数不确定性。

分类：plug-in approximation（α、Σ 点估计插件；G 条件 Gaussian 采样）。

---

## 7. Innovation law：fractional posterior、场景缩放、均匀行 bootstrap

[事实] fractional posterior 只作用于 scalar macro OOF 残差：e_res_m=macro_residual_series(res_history)（src/predict.jl:493-495；实现 src/residual_oracle.jl:216-260）。

[事实] 标量公式：e[idx]=sumr/sqrt(c) − mu_macro*sw/c − dot(h,x)/sqrt(c)（src/residual_oracle.jl:257）；c==0 → 0.0（239）。h 只在单次调用内按 (fold, mask) 缓存（246-254）。

[事实] 似然与后验：causal_fractional_posterior（src/predict.jl:76-99）；FFT 路径 fractional_likelihood!（101-119）；log_weights=ll .+ log.(DELTA_D_V1) 后 softmax（94-96）；DGRID_V1 / DELTA_D_V1（10、14）。

[事实] 绝对缩放：v_bootstrap=max(var(e_res_m),1e-8)（src/predict.jl:499）；每场景 scale=sqrt(v_forecasts[d]/v_bootstrap)（615）。

[事实] 均匀行 bootstrap：row=min(floor(ur*T)+1,T)，ur=uniforms[s,2]（src/predict.jl:591、578）；每场景抽一行、全资产共享该行（除 fallback）；缺格 own-row fallback（596-602）。

[事实] 场景公式：loggross=(mu_m*e0[j]+relative[j,s]−shift)*s1[j]+residual*scale（src/predict.jl:620）。

[事实] residual 行来自 OOF oracle（src/residual_oracle.jl:150-211），非 in-sample；dense 参考公式保留为测试用 dense_oof_residuals（301-337）。

[推断] 实际 innovation law =（单一 scalar macro 残差的分数阶缩放）×（OOF 残差行的均匀 bootstrap）；横截面风险没有逐资产 fractional、没有残差协方差重建，跨资产相关性只来自共享行与 L_rel 的结构项。

分类：scalar degeneration（宏观维退化到标量序列）+ finite-sample approximation（经验行 bootstrap）。

---

## 8. Structural constraint：C 的构造与 14 行

[事实] get_constraint_columns(N,n_bands) 生成 14 个列块：for b in 1:n_bands, channel in 1:2，块为 ((b-1)*2+channel-1)*N+1 : ((b-1)*2+channel)*N（src/response.jl:106-107）——7 bands × 2 channels = 14 行（约束数）。

[事实] 列布局：channel1 = Q 块（feature 偏移 (b-1)*2N+j），channel2 = P 块（(b-1)*2N+N+j）（src/predict.jl:1168-1169；src/residual_oracle.jl:141-142）。

[事实] h[c]=tr(G[:,cols[c]])（src/response.jl:128）；M=CΩC' 的 14×14（constraint_moments，109-129）。

[事实] 条件化均值：λ=M^{-1}h；K=Σ λ_r * Σe_cols 外积；G_c=G−K V（src/response.jl:131-143）。这是 exact Gaussian conditioning，不是事后均值减 trace。

[事实] fit 内 conditioning 在 full N 空间（src/response.jl:1199-1202）；EB 内部把约束用 gauge=relative_gauge(N) 压缩到 N−1（420-424、1177）。

[事实] 约束后 trA/trB 统计（src/response.jl:1206-1207）；dense V 版本（测试）144-148。

分类：identical（约束=SPEC §21 的 trA_b=0 与 trB_b=0，对全部 7 个 band；条件化在 posterior support 内精确执行）。

---

## 9. Numerical integral：S 与 adaptive

[事实] generate_scenarios_v1 默认 S=500（src/predict.jl:546）。

[事实] backtest_v1 默认 S=300（src/backtest.jl:117）；path_kelly_v1 默认 S=300（src/kelly.jl:173）。

[事实] 正式回测默认 adaptive=false（src/backtest.jl:119）；固定分支：generate_scenarios_v1(model;S,rng=MersenneTwister(seed+t))（250），S 来自参数（默认 300）。scenario_counts 记录实际 S（293）。

[事实] adaptive 分支存在：adaptive_scenario_weights（src/predict.jl:627-653；min_scenarios=64、max_scenarios=512 默认，628）；doubling（634-635）；收敛双条件 norm(weights-previous,1)<=weight_tol 且 certificate.objective_gap<=tol（646-648，默认 1e-3 / 1e-5）；不收敛 error（652）。backtest 的 adaptive 路径（src/backtest.jl:71-77）。

[事实] 固定 S=300 的显式仓库用法：dev/local_panel_stages.jl:118-119、dev/earlier_window_replay.jl:18 与 61（scenario_counts==fill(300,8)）、dev/release_freeze.jl:54。

[推断] 证书范围：kelly_certificate 只证明"给定这套 S 个场景样本内"的 KKT/可行性/gap（src/kelly.jl:21-36）；adaptive 的双条件是 S 翻倍过程中的经验门限收敛；**固定 S=300 的非 adaptive 路径没有任何 S 收敛证书**（SPEC §65 的 ‖w_2S−w_S‖→0 只在 adaptive 模式里以门限形式体现，而它默认关闭）。

分类：finite-sample approximation（固定 S 的蒙特卡洛/拟蒙特卡洛样本；adaptive 存在但非默认，且证书是门限而非极限证明）。

---

## 10. Decision：kelly_weights_v1 目标（含 base_s）

[事实] kelly_weights_v1(X;budget=1.0,base=nothing)（src/kelly.jl:132-134）→ fast_kelly_solver：max (1/S)Σ_s log(X_s·w + base_s)，s.t. w>=0，Σw=budget（src/kelly.jl:60-130；目标构造在 102、107-108）。

[事实] base_s locked holdings：locked_wealth(X,locked) 对每 held 列乘 gross，held 非 finite 抛错（src/kelly.jl:137-148）。

[事实] scenario_weights：locked=current .* .!free；budget=1−sum(locked)；out=copy(locked)；free 列解 Kelly（src/kelly.jl:150-170）。

[事实] 证书：kelly_certificate 的 feasibility/kkt/objective_gap（src/kelly.jl:21-36）；fast 失败回退 clarabel（129）；clarabel 解同一 log-Kelly 目标（39-58）。

[事实] backtest 持仓结转：h=(w.*gross_clean)./(1+ret)（src/backtest.jl:301）。

分类：identical（exact log-growth 目标；locked 风险以 base_s 计入，非当 cash）。

---

## 11. PONY 静态推演：(a) 置 false 后果、(b) 进入 active 日期

定位事实：
- [事实] universe.txt:46 为 PONY（universe 共 65 个标的，PONY 是第 45 个 symbol；行 2..66 为代码，行 1 为注释）。
- [事实] 全仓检索未定位到名为 PONY wrapper 的脚本；PONY 字符串唯一出现在 universe.txt:46。仓库中存在的同类操作是"复制前缀"而非"置 false"：dev/earlier_window_replay.jl:37、50；dev/release_failure_probe.jl:15；dev/release_freeze.jl:103-104（Bars(... b.bar[1:t,:])）。
- 因此 (a)(b) 为按当前代码语义的纯静态推演（无 wrapper 文件可引）。

(a) 若把 PONY 前 251 行 bar 置 false（假设性）：
1. signal_prices：前 251 行 PONY 列全部 NaN（src/data.jl:37）。
2. active_universe_indices(signal)：PONY 需要至少一条相邻 finite return（src/predict.jl:130）；前 251 行全 NaN 时最早合法对是 (252,253)。故决策 prefix T_raw=252 时 PONY 仍非 active；T_raw>=253 时 active（cache 路径同结论：first_return=252，findall(<(T_raw),·) 需 T_raw>=253）。
3. r 层面：r[252]=logs[253]−logs[252]（diff 语义，src/predict.jl:247）——必须有第 253 行 finite 才产生这条 return。
4. 若 wrapper 只改 b.bar 而不改 b.adj：账户 marking 不受影响（src/backtest.jl:296 用 adj），但 free（244）与 signal 都受影响；bar 也是保存落盘的判据（src/data.jl:80-82）。
5. 训练门槛：active 仅是坐标空间准入；ts_total=WARMUP:T-2=256:T-2（src/predict.jl:286）需 n_res>=2 且 F_folds>=2（288）；ruler 另有 4*tau<=T−f+1（src/geometry.jl:38），f=252 时 tau=1 需 T>=255。

(b) 若 2025-12-01 是（前 251 行被清后）首个有效 bar（即行 252）：
- 主解读（与 (a) 语境一致，[推断]）：首日当天不足以判定 active；PONY 要到**行 253（2025-12-01 之后的下一个有 bar 交易日）**才进入 active。off-by-one 根因：active 需要 return，而 return 属于相邻 finite 对的第二日。
- 若行 253 无 bar，则顺延到下一个与前一个有效 bar 相邻的有效 bar 日。
- 精确日历日期：[未知]（未读 panel CSV 的日期列；静态只能给出"下一个有效 bar 日"）。
- 另一种解读（若 PONY 此前已累积 251 个有效 bar、2025-12-01 是第 252 个）：则当天即 active（第 251、252 个 bar 相邻）。该解读与 (a) 的"前 251 日置 false"矛盾，故不作为主结论。

分类：本节的 (a)(b) 为静态推演，非运行证据；wrapper 文件本身标记为"未定位"。

---

## 12. 映射总表

| # | 理论声明（AGENTS.md/SPEC） | 当前实现 | 证据（file:line） | 差距分类 |
|---|---|---|---|---|
| 1 | H_t 全价格历史；missing≠0（§3/§4/§8） | signal_prices；r=diff(log signal) | src/data.jl:37；src/predict.jl:216,247 | identical |
| 2 | 观测 mask O（§8.1） | bar / observed / alive_now 三表征同源 | src/data.jl:12,28；src/predict.jl:68；src/prepare.jl:131,160-161 | identical |
| 3 | active universe（§11） | active_universe_indices / first_return | src/predict.jl:125-135,207-208；src/numerics.jl:150 | identical |
| 4 | （无独立 E^trade 声明） | backtest free=active∩bar_t | src/backtest.jl:242-245；src/kelly.jl:152-153 | unknown |
| 5 | 物理可交易性（§41） | broker.tradable（bid/ask/last） | src/broker.jl:136 | identical |
| 6 | Bayesian response posterior（§17/§19/§25/§26） | α、Σ EB 点估计 + 条件 Gaussian G；场景从 L_rel/var_m 采样 | src/response.jl:934-1209；src/predict.jl:570-572,612 | plug-in approximation |
| 7 | OOF innovation law（§30/§31/§32） | scalar macro fractional × 均匀行 bootstrap | src/predict.jl:493-499,570-620；src/residual_oracle.jl:216-260 | scalar degeneration + finite-sample approximation |
| 8 | trA_b=0、trB_b=0（§21） | 14 列块 exact conditioning | src/response.jl:106-143,1199-1202 | identical |
| 9 | 场景/积分收敛（§37/§65） | 默认固定 S=300；adaptive 可选（64→512 门限） | src/backtest.jl:117,119,250；src/predict.jl:627-653 | finite-sample approximation |
| 10 | exact log-Kelly（§6/§38） | kelly_weights_v1 + base_s locked | src/kelly.jl:60-170 | identical |

---

## 13. 未能确定 / 未决

- [未知] PONY wrapper 脚本不在本仓库可见位置（全仓检索仅 universe.txt:46 命中 PONY）。
- [未知] (b) 的确切日历日期：需要 panel 的实际交易日历（CSV 日期列未读）。
- [未知] α、Σ 点估计对 predictive 覆盖与权重分布的数值影响（需受控运行，非静态可判）。
- [未知] live 层（src/live.jl）未逐行阅读；broker/live 与 backtest 的交易资格同构性未在本次核对（AGENTS.md 的 live 描述仅作背景）。
- [推断] 本地 panel 的列序与 universe.txt 一致、PONY 列号=45（基于 local_panel_stages.jl:86/97 的 N==65 与 universe 顺序；未读 CSV 头）。

---

## 14. 结论摘要（一句话）

理论链 H_t→Pi(G,Sigma|H)→P(r|H)→w* 在"数据/signal 语义、观测 mask、active 准入、trace 约束、Kelly 目标"五处是 identical；集中差距在统计认识论与数值积分三层：
1. response 的 (α,Σ) 是 EB 点估计插件（G 仅条件采样）——plug-in approximation；
2. innovation 是 macro 标量 fractional × 经验行 bootstrap（横截面退化）——scalar degeneration + finite-sample approximation；
3. 默认回测固定 S=300、无 S 收敛证书（adaptive 存在但默认关闭）——finite-sample approximation。

==> docs/NUMERICAL_INTEGRATION_SPEC.md <==
# KTrader 数值积分收敛规范（Numerical Integration Spec）

**文档状态：SPEC 级定义草案（Normative Draft）——不是实现描述，不是性能承诺。**
**适用对象：`generate_scenarios_v1` / `adaptive_scenario_weights` / `kelly_*` 决策链。**
**权威关系：本文档细化 `AGENTS.md` §37（AGENTS.md:1176-1191）与 §65（AGENTS.md:2133-2141）的执行语义。与 `AGENTS.md` 冲突时以 `AGENTS.md`（Common Law / SPEC）为准。本文档不扩大任何角色职权，不授权任何命令执行，不修改源码、测试与既有文档。**

---

## 0. 目标对象与两类误差

决策目标（AGENTS.md:1201-1203）：

- I(w) = E[log(base + Rᵀw) | H]，连续（无限样本）目标；
- I_M(w) = (1/M) Σₛ log(base_s + (X_M w)_s)，给定 rule 下 M 个 scenario 的有限采样近似；
- w_M = argmax_{w∈W_t} I_M(w)，W_t = { w ≥ 0, Σw = budget }（预算与 locked base 语义见 AGENTS.md:1208-1218）。

目标：

- 积分收敛：沿嵌套 rule 细化 M 时 I_M(w) → I(w)（对每个固定 w）；
- 决策收敛：w_M → w* = argmax I(w)。

必须区分两类独立误差：

1. **求解误差**：给定 X_M，solver 相对 w_M 的误差。由 `kelly_certificate` 界定（src/kelly.jl:21-37），是「凸问题解好了没有」的证书。
2. **积分误差**：X_M 的采样律相对真实 posterior predictive 的离散化误差。由本规范 §3 的 (a)(b) 界定，是「M 够不够大」的证书。

现有 Kelly 证书只覆盖第 1 类。把它当成积分收敛证明，是把样本内的最优性当成样本外的逼近，属于层级错误。

---

## 1. 当前实现映射

### 1.1 有限采样近似在哪里发生

`generate_scenarios_v1`（src/predict.jl:546-625）按 §35 的六要素（AGENTS.md:1124-1131）生成 X_M ∈ R^{M×N_universe}：

- posterior draws 与 residual bootstrap 在 src/predict.jl:564-572 与 576-624；
- 目标函数求值发生在 kelly 层：wealth = X*w .+ b（src/kelly.jl:26），g = Xᵀ(1./wealth)/S（src/kelly.jl:30），objective = sum(log, wealth)/S（src/kelly.jl:102、110）；
- I_M(w) 就是 `sum(log, X*w .+ base)/S`（src/kelly.jl:30、102）。

### 1.2 S（scenario 数）的出现位置与默认

| 位置 | 默认 | 语义 |
|---|---|---|
| src/predict.jl:546 | `S=500` | `generate_scenarios_v1` 的 IID 默认 scenario 数 |
| src/kelly.jl:173 | `S=300` | `path_kelly_v1` 默认；`adaptive=false` 时直接使用 |
| src/backtest.jl:117 | `S=300` | `backtest_v1` 默认；`adaptive=false` 时逐日使用 |
| src/backtest.jl:249-251 | `S`（调用方） | 正式回测逐日 `generate_scenarios_v1(model; S, rng=MersenneTwister(seed+t))` |
| README.md:36、44 | `300` | 文档示例与 `SCENARIOS=300` 环境变量 |
| dev/m1_artifact_replay.jl:626-627 等 | `300` | M1 历史对照基准，非生产路径 |

### 1.3 adaptive 机制真实存在

不是占位，是已接线的实现：

- `ScenarioQuadrature`（src/predict.jl:507-521）：构造时取一组素数基与**一份共享随机 shift**（`rand(rng,dim)`，src/predict.jl:520）；嵌套性来自 `quadrature_uniform`（src/predict.jl:522-531）按**全局索引 s** 生成点：任意 M' > M 的序列前 M 个点与 M 序列逐位相同，不重新洗牌。
- quadrature 模式下 `generate_scenarios_v1` **不消耗调用方 rng**（src/predict.jl:552-563：`uniforms=nothing`，用 `quantile(Normal(), quadrature_uniform(...))` 逐点确定性变换；契约由 test/posterior_contract_tests.jl:34-45、241+ 覆盖）。
- `adaptive_scenario_weights`（src/predict.jl:627-653）：默认 `min_scenarios=64, max_scenarios=512`（src/predict.jl:628），从 S=64 起按 `S=min(2S,max_scenarios)` 倍翻（src/predict.jl:635），每轮用**同一个 rule 对象**重生成样本（src/predict.jl:636），因此样本是嵌套前缀。
- 收敛判据（src/predict.jl:647）：`‖w_new − w_prev‖₁ ≤ weight_tol(=1e-3)` 且 `kelly_certificate(X_new, w_prev).objective_gap ≤ tol(=1e-5)`。
- 预算耗尽：`error`（src/predict.jl:652）——fail-loud 已是现状。
- 测试证据：test/numerical_tests.jl:160-182（`small ≈ large[1:64,:] atol=1e-14` 证前缀嵌套；零响应 world 收敛到 S=128；`max_scenarios=128, tol=0, weight_tol=0` 时确定性抛错）；test/backtest_target_tests.jl:25-37（adaptive 接线、seed 保持、结果计数）；test/posterior_contract_tests.jl（quadrature 下 lazy/dense 一致）。

### 1.4 正式回测实际使用的参数

`backtest_v1` 默认 `adaptive=false`（src/backtest.jl:117-121）。非 adaptive 路径：worker 每日生成 `X = generate_scenarios_v1(model; S=300, rng=MersenneTwister(seed+t))`（src/backtest.jl:249-251），consumer 调 `scenario_weights`（src/backtest.jl:79-82、291-292）。**正式回测的默认不是嵌套 quadrature，而是固定 S=300 的 IID 采样**。`adaptive=true` 时经 `_backtest_target` 调 `adaptive_scenario_weights(...; rng=MersenneTwister(seed+t), tol=quadrature_tol, max_scenarios)`（src/backtest.jl:72-78）。默认参数下 §37 的细化判据根本不执行。

### 1.5 现有 Kelly 证书证明什么、不证明什么

`kelly_certificate`（src/kelly.jl:21-37）在给定矩阵 X_M 上验证：

- feasibility（simplex 越界，src/kelly.jl:25）；
- kkt_residual（互补性 `w .* (dual .- g)`，src/kelly.jl:32）；
- objective_gap = budget·max(g) − gᵀw ≥ F_M(w*) − F_M(w)（由 concavity，src/kelly.jl:17-19、33）。

它证明的是：**在固定的这一批 X_M 上，w 离 argmax I_M 有多远**。它不证明 I_M → I，也不证明 w_M → w*。若 X_M 有采样偏差，证书全绿的同时最优决策仍可能有系统误差。`fast_kelly_solver` 以 tol=1e-8 内部自证（src/kelly.jl:125-126），失败则 `clarabel_kelly_solver` 解同一目标并再次自证（src/kelly.jl:129、55-56）——两者都是**同一 X_M 上的求解误差**，样本层面的收敛不在其职责内。

---

## 2. 目标

### 2.1 收敛目标

沿同一嵌套 rule 细化：M → 2M → 4M → …

- I_M(w) → I(w)；
- 决策收敛：w_M → w*。

### 2.2 嵌套确定性 rule 的硬性要求

1. 一次 refinement 全程同一 rule 身份：primes 与 shifts 固定（src/predict.jl:512-521 的共享 shift）；2M 序列的前 M 个点必须逐位等于 M 序列的点；禁止重新洗牌、重抽 shift、或换 rule 后再比较。
2. quadrature 模式下不消耗调用方 rng（现状满足；见 §1.3）。
3. 调度无关：同一决策日、同一 seed 下，worker/线程调度不得改变 rule 或点序（AGENTS.md:1151-1172）。
4. rule 替换（如 Sobol/Owen，AGENTS.md:1191）是数值 backend 变更，不改变 posterior law；替换后必须重新走 §3 证书。

---

## 3. 收敛证书

对一次 refinement 的相邻两级 M、2M：

- (a) **权重稳定性**：‖w_{2M} − w_M‖₁ ≤ ε_w。
- (b) **目标间隙**：I_{2M}(w_{2M}) − I_{2M}(w_M) ≤ ε_U，用更细的同一 rule 求值。
- (c) **可选 KKT**：在 X_{2M} 上复核 kelly_certificate(X_{2M}, w_{2M}) 的全部指标 ≤ solver tol。

### 3.1 各自在哪一层计算、谁是 owner

| 判据 | 计算层 | owner | 理由 |
|---|---|---|---|
| (a) | scenario refinement 循环（持有 w_M 与 w_{2M}） | `predict.jl`（`adaptive_scenario_weights`） | kelly 层看不到跨 M 的另一半 |
| (b) | scenario generator 层求值（样本在 generator 层，目标语义在 kelly 层） | `predict.jl` 计算差；目标求值语义归 `kelly.jl` | 必须在同一更细样本上比较两个权重 |
| (c) | 给定 X 的 KKT/最优性 | `kelly.jl`（`kelly_certificate`） | 已有唯一实现，禁止复制 |

(a)(b) 是积分收敛证据；(c) 是求解误差证据。二者必须分别成立、分别记录，不得互相顶替。任何实现都不得把「certificate 绿了」说成「积分收敛了」。

### 3.2 通过条件

一次 adaptive 返回必须满足：fine 解 w_{2M} 自身有有效 solver 证书（现状：fast 内部 tol=1e-8 或 Clarabel 自证，src/kelly.jl:61、126、55-56），且 (a) ≤ ε_w、且 (b) ≤ ε_U。若启用 (c)，在 fine 样本上复核。

### 3.3 为什么 (b) 不能由 (a) 替代

仅有 (a)：w_M 与 w_{2M} 可能同步漂移（同一采样偏差下近似相等），‖·‖₁ 小但都远离 w*；仅有 (c)：I_M 可以因 M 不足而整体偏离 I，每步都「解得好」但解的是错的积分。所以需要 (b) 在更细同一 rule 上直接比较目标值。以 Sharpe/回测收益作判据违反 AGENTS.md:1183-1189 与 2133-2141。

### 3.4 现状与证书定义的差距

- 现状（src/predict.jl:646-647）：`certificate = kelly_certificate(X_new, w_prev)` 是 **w_M 在 X_{2M} 上的次优间隙**（(b) 的一种上界变体），不是直接的 I_{2M}(w_{2M}) − I_{2M}(w_M)；返回的是 w_{2M}（`weights`），证书却属于 w_M——语义不对称。
- `tol=1e-5`（integration 层）与 solver 内部 `tol=1e-8`（src/kelly.jl:61）是两个不同层次的阈值，必须分开命名、分开记录。
- 非 adaptive 默认路径（src/backtest.jl:117、249-251）完全不做跨 M 检查。

---

## 4. Fail loudly

- 预算内不收敛必须抛错，错误文本必须包含 `"Numerical integration did not converge"`（现状 src/predict.jl:652 为 `"posterior quadrature did not converge by $max_scenarios scenarios"`，语义一致、措辞待统一到规范文本）。
- 严禁「到 512 就算了」：不得在耗尽 `max_scenarios` 时返回 `converged=false` 的默认权重、静默回退 IID、或把最后一次权重当成结果。
- 严禁：靠调大 `max_scenarios` 让某次运行通过而不记录证书；靠删除 (b) 让循环提前退出；靠重跑直到碰巧通过。
- 与 AGENTS.md:2012（§56 fail-loud）一致。

---

## 5. S=300 降级

- `S=300` 只允许作为：数值初始规模、测试/诊断规模、历史对照（如 dev/m1_artifact_replay.jl:626-627 的 M1 基准）、或 adaptive 细化的中间起点。
- 生产决策（backtest/live 最终权重）不得使用未经 §3 证书的固定 S。当前「`S=300` + `adaptive=false`」仍是默认（src/kelly.jl:173、src/backtest.jl:117），属于与本文档目标的已知差距（§8），其修复必须走独立评审与实现流程，不得在本草案阶段声称已达成。
- 一旦 adaptive 成为生产默认，生产签名里的 `S=300` 必须被重新审查或删除。

---

## 6. 与 SPEC §37 的关系

- §37（AGENTS.md:1176-1191）：细化判据是 ‖w_{2S}−w_S‖₁ 与 Kelly objective certificate，不是 Sharpe。本文档把该句细化为：(a) 即 L1 判据；(b)(c) 把「Kelly objective certificate」拆成积分目标间隙与求解 KKT 两个层次。
- §65（AGENTS.md:2133-2141）：‖w_{2S}−w_S‖₁ → 0；不得用回测收益选 S。与本文档一致。
- 张力（必须标注）：
  1. §37 表述「Kelly objective certificate」是单数，现实现中它同时被当成 (b) 的代理与 (c) 的阈值，而 solver 内部另有 1e-8——层次混淆。
  2. §37 同时支持 IID 与 quadrature，但默认回测走 IID 且固定 S（src/backtest.jl:117）；生产若声明依赖 §37 的收敛保证，默认路径必须显式走证书路径。
  3. §37 允许未来替换 Sobol/Owen（数值 backend，不改变 posterior law）——本文档要求替换后重新走 §3 证书，并保留 rule 身份用于重放。

---

## 7. 未决决定（判据均不依赖 Sharpe）

| # | 未决项 | 决定所需证据 | 判据 |
|---|---|---|---|
| 1 | `max_scenarios` 默认（现状 512） | 不同 N 与响应强度下 M→2M→4M 的 (a)(b) 序列；撞顶比例 | 证书通过所需 M 的分布；撞顶即暴露给重新审查，不静默放过 |
| 2 | ε_w 数值（现状 weight_tol=1e-3） | 同上；量纲（full N_universe 空间 L1） | 缩小 ε_w 时 w 变化符合收敛趋势 |
| 3 | ε_U 数值与定义（绝对/相对） | (b) 的直接实现与实验 | ε_U 减半时返回 M 单调不减/证书仍可复现 |
| 4 | 嵌套下 RNG 确定性细节 | rule shift 抽取与返回对象的身份记录 | 同 seed、同 rule 逐位可重放；跨日期 `MersenneTwister(seed+t)` 与调度无关 |
| 5 | adaptive 返回对象是否携带 rule 身份与证书数值 | 审计、重放与复现要求 | 返回 primes+shifts+M+证书，或明确记录可重放来源 |
| 6 | adaptive 在 backtest 中的序列化/并行成本 | src/backtest.jl:105-111 记录的现状约束 | 只序列化数学所需的最小对象；不得改变任何数学 |

全部未决项的裁决必须回答：证书在预算内是否通过。禁止以「回测更好/更快」裁决数值参数。

---

## 8. 现状差距清单

| # | 项目 | 现状 | 规范目标 | 引用 |
|---|---|---|---|---|
| 1 | 嵌套 rule | 已实现（Halton + 共享 shift） | 保持，受证书约束 | src/predict.jl:507-531、627-653 |
| 2 | 生产默认路径 | `adaptive=false`, S=300 IID | 生产须走证书路径，或显式声明降级 | src/backtest.jl:117、249-251 |
| 3 | (a) 权重 L1 | 已实现 | ε_w 经实验确定 | src/predict.jl:647 |
| 4 | (b) 目标间隙 | 代理：w_prev 在 X_new 上的 KKT gap ≤ 1e-5 | 显式 I_{2M}(w_{2M}) − I_{2M}(w_M) ≤ ε_U | src/predict.jl:646-647 |
| 5 | (c) KKT | fast/Clarabel 内部 1e-8 | 可选 fine 样本复核 | src/kelly.jl:61、126 |
| 6 | 错误文本 | `posterior quadrature did not converge by ...` | 含 `"Numerical integration did not converge"` | src/predict.jl:652 |
| 7 | rule 身份/重放 | 返回对象不含 primes/shifts | 记录或可重放 | src/predict.jl:643-648 |
| 8 | 高维 rule 有效性 | dim = 2N+3；N 大时未证收敛速率 | 收敛实验覆盖真实 N | src/predict.jl:630 |

---

## 9. 一句话

证书必须回答两个不同的问题——「这批 scenario 上的凸问题解好了吗」(c) 与「这批 scenario 够代表真实积分了吗」(a)(b)；前者已有唯一 owner（kelly.jl），后者必须由 scenario refinement 层在预算内证明，否则 fail loudly，而不是「到 512 就算了」。

==> docs/OLD_TO_CURRENT_SEMANTIC_DIFF.md <==
# OLD → CURRENT 语义差异报告（Gate 0 重开 · Deliverable 2）

**状态：分析文档（非规范、非发布凭据）**
**基准：旧侧 = `1d9b7ad`（Path Kelly V0 baseline，2026-10-06 12:57:22 +0800）；新侧 = 当前工作树（release 基线 `fc54ce48`，2026-10-09 12:14:17 +0800）**
**性质：纯静态语义比对。本文不含任何收益 / PnL / Sharpe 数字；不运行任何命令；不修改源码、测试与任何既有证据文件。**
**证据来源：`dev/evidence/manager13/` 侦查材料（`defect_markers.txt`、`src_evolution.txt`、`old_src_snapshots.txt`）与三份快照目录；行号引用均指各文件的静态文本。**

---

## 0. 范围声明与基准选择

### 0.1 为什么以 `1d9b7ad` 作为旧侧

【事实】`dev/evidence/manager13/defect_markers.txt`（§A–§F）以 pickaxe 与逐提交存在性矩阵证明：答辩点名的判别性缺陷标志——`shrink_drift`、ARD（MacKay 分组 ARD）、BF 家族（`BF_MIN` / `bayes_ard` 稀疏门控）、selection 自选择缺陷承认注释——的**唯一宿主**是 `1d9b7ad`；从 `21017b9`（V1.0）起全部为零命中（`defect_markers.txt:309-316, 518-527, 562-564`）。`old_src_snapshots.txt:108-121` 的判定 1 与之一致：它同时是时间序上最早的实质提交，也是这些标志的最后（唯一）宿主。

因此，`1d9b7ad` 是"缺陷标志意义上"可被证据锚定的旧侧端点，而不是凭印象挑选的版本。

### 0.2 为什么不是字面的「最后胜出版本」

【事实】`old_src_snapshots.txt:108-131` 明确记录：本仓库 git 历史中**不存在**具名为「早期胜出版本 / winner」的单一 commit；答辩报告原文不在本机可读范围（`old_src_snapshots.txt:124-128` 判定 3）。`defect_markers.txt:552-559`（F4）进一步说明：'BF' 的字面 token 在所有提交的源码中零命中（仅二进制 fixture `t14294_conditioned_fold3.jls` 字节巧合），若答辩所指为 BF 稀疏门控语义，则其最后宿主同为 `1d9b7ad`；两种解释都已记录、不作裁决。

因此本文严格使用"证据最强的旧侧基准"这一口径。中间对照 `21017b9`（V1.0，修复发生点）与 `602b897`（性能线终点/发布父提交）在必要处引用为**对照面**，但它们不是旧侧的替代品。

### 0.3 不确定性（必须随结论携带）

1. 答辩报告原文不可读；关键词清单来自任务转述，未逐字核对。若报告点名了其他标志，需以其证据重做对照（`old_src_snapshots.txt:125-127`）。
2. `shrink_drift` 在 V0 中是一个具体函数（模型内收缩估计），其是否等同于报告所称"缺陷"需报告原文确认；本文只给"该标志在此提交存在/不存在"的事实（`old_src_snapshots.txt:128-129`）。
3. 所有行号来自转储快照的静态文本；`old_src_snapshots.txt:65-74` 的 `diff -r` 保真验证与逐文件 SHA-256（`:76-106`）证明转储内容与 git 对象逐内容一致，但 git blob 名是 SHA-1、不可与 SHA-256 直接比较（`:70-72`）。
4. 本报告是**语义分析**：分类（修 bug / 理论变化 / 数值变化 / 结构重构）为工程判断，凡推断均显式标注【推断】；凡规范裁决均回引 SPEC 章节。本文不宣布任何收益结论。

---

## 1. 十项数学对象的逐项 diff

分类词约定：
- **修 bug**：旧实现与项目自身声明的数学/契约不一致，且移除/修复在规范或注释中有据；
- **理论变化**：模型对象、先验、法律或信息集语义改变（需要 SPEC 审查确认归属）；
- **数值变化**：同一数学对象的离散化/算法/常数改变；
- **结构重构**：数学对象不变或近不变，但 ownership / 表示 / 调用路径改变；
- **【需 SPEC 审查】**：无法在上述四类内唯一归类，或分类依赖于尚不可读的规范判断。

### 1.1 Conditional mean

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:503-507`：`conditional_mean = drift.mean .+ response_mean`；`response_mean` 对 geometry draws 取平均（`model.jl:504`）；`predict` 中每场景抽 drift 层级（`model.jl:535-536`） | `predict.jl:480-486`：`mu_rel_proj = mu_rel .- mean(mu_rel)`；`mu_norm = mu_m .* e0_now .+ mu_rel_proj`；`mu_asset_act = mu_norm .* s1`；嵌入完整 universe（`L485-486`）；`predict.jl:463-466` 由 `predictive_moments` 给出 `mu_m, mu_rel` |
| 语义 | 条件均值 = 跨资产层级漂移后验均值（`shrink_drift`，`model.jl:317-339`，含 Newey–West 长期方差 SE，`model.jl:288-305`）+ 对 8 个 bootstrap 几何的响应均值 | 条件均值 = macro ridge 均值（`response.jl:1146-1151`）+ trace 条件化后的 relative 均值（`response.jl:1200-1201`）；无 drift 项；relative 分量在活资产上零和 |
| 分类 | **理论变化**（层级 drift 被移除；`drift` 于 V1.0 即已消失：`old_src_21017b9/src/predict.jl:165-166` 只余 `mu_m e0 + Phi_perp mu_rel`）。**修 bug 成分**（V0 的 in-sample 拟合均值参与 `resid`，见 1.6/3.3） |

【推断】drift 项的存废是 V0→当前对**低信噪比横截面**影响最大的单点：V0 把每资产均值向共同 `μ0` 收缩（`model.jl:329-334`），当前完全不表达该先验；同时当前 relative 投影显式去均值（`predict.jl:480`），而 V0 的 `response_mean` 是 mode 空间平均，其零和性依赖 neutrality 约束。此项改变预测的横截面形状，最可能解释策略行为变化。是否属理论推进需 SPEC 对 SPEC §34 的对照（当前定义与 SPEC §34 一致）。

### 1.2 Response operator

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:374-470`（`fit_response`）：几何窗口内每 band 的协方差特征向量 `Φ=natural_modes(...)`（`model.jl:164, 192-196`）；每条通道是 `φ_k ⊗ c_k(t)`（`model.jl:374-390`）；系数块 `Cb`（`model.jl:433-438`）；`bayes_ard` 求解 | `response.jl:1124-1209`（`fit_response_operator`）：macro 为 ridge/EB；relative 为 `P=14N` 维 design 上的谱表示（`response.jl:150-174`）+ trace 条件化（`L1200-1208`）；design 由 `fill_design_matrix!`（`response.jl:55-75`）在**固定 Helmert gauge** 相对空间上构造（`numerics.jl:368-380` `relative_gauge`） |
| 语义 | 数据依赖的每 band 模式（sign / 简并旋转不可控），每个 mode 一对 `(a_k,b_k)`；`θ_posterior` 报告 `ρ,θ,R1,R2`（`model.jl:472-500`） | 固定数值 gauge；"自然结构"由算子谱解释（SPEC §14 的 0.95 决策）；`ResponseOperator` 持有 `G_c_mean, covariance, Sigma_rel, inv_M_constraint`（`response.jl:92-104`） |
| 分类 | **结构重构**（表示从 eigenmode 坐标到固定 gauge 坐标）+ **理论变化**（mode identity / sign / 简并问题被消除；V0 的 `MODE_STABILITY` 过滤 `model.jl:207-218` 与 `GEOM_DRAWS` 在 HEAD 已不存在）。V1.0 是过渡：`old_src_21017b9/src/predict.jl:100-105` 仍用 `relative_modes` + `Phi_draws` |

### 1.3 Structural neutrality

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:408-412`：`Cn` 只在非 macro mode 上置 1，`Z=nullspace(Cn)`；`bayes_ard` 以 `Z` 参数化 `β=Zγ`，neutrality **在先验子空间上**（`model.jl:16-17, 222-232`） | `response.jl:131-148`（`condition_trace_neutrality`）：由 `constraint_moments`（`L109-130`）在 `CΩCᵀ`（14×14）上求解，`G_c = G - Ω Cᵀ (CΩCᵀ)⁻¹ C G`；均值与协方差同时条件化（`response.jl:1192-1208`） |
| 中间对照 V1.0 | `old_src_21017b9/src/response.jl:172-196`：对 `G_raw` 的每个 band **减去对角均值**（mean-only）；`predict.jl:250-265` 对每个样本再减均值。静态代码中**没有**对 covariance 的约束条件化 | — |
| 分类 | **【需 SPEC 审查】**：V0 的"先验支持上约束"在数学上合法（SPEC §22 允许的路径之一的前身），V1.0 的 mean-only 投影与 SPEC §22 明确禁止列表中的"只把 mean 的 trace 减掉"字面相符【推断，静态行号支撑】；当前实现即 SPEC §22/§25 描述的 constrained mean/covariance。V0→当前是**理论深化**（先验约束 → 精确条件化），但 V1.0 中间态相对 V0 是一次**局部退步**，值得在 SPEC 历史审计中单列 |

### 1.4 Parameter prior

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:222-232`：每 mode 一个 ARD 精度 `α_k`，`(a_k,b_k)~N(0, α_k⁻¹ I₂)`；`ALPHA_MIN/ALPHA_MAX = 1e-6/1e7`（`model.jl:233`） | `response.jl:1160-1185` + SPEC §19：`G|Σ,α ~ MN(0, Σ, α⁻¹I)`；`EB_ALPHA_MIN/MAX = 1e-4/1e6`（`response.jl:207-208`） |
| 语义 | 每通道独立精度（分层稀疏先验）；`α_k→∞` 表示模式关闭 | 全局标量 `α_rel`（macro 另有 `α_m`）；不确定性由 `Σ`（relative innovation covariance）与 `V=(XᵀX+αI)⁻¹` 表达 |
| 分类 | **理论变化**（每模 ARD 分层 → 全局标量 Matrix-Normal 先验；稀疏性从"BF 门控开关"变为连续证据） + **数值变化**（域界常数 1e-6/1e7 → 1e-4/1e6，SPEC §24 限定为数值域界） |

### 1.5 Hyperparameter treatment

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `bayes_ard` 内循环（`model.jl:248-286`，500 iters / tol 1e-4）→ 对每个 live mode 计算 `logev` 增益 → `weak = gain .< log(BF_MIN*M)` → 关闭并重拟合（`L274-281`） | `maximize_logalpha`（`response.jl:212-273`）：33 节点对数网格 + bracket 二分 + secant + 确定性 tie-break；`bounded_alpha_certificate`（`L276-282`）与 `covariance_certificate`（`L297-347`）双证书；`optimize_matrix_normal_eb` / `optimize_conditioned_eb` 以 `certificate.valid || error` fail-loud（`response.jl:378, 996-999, 1119-1121`） |
| 中间对照 V1.0 | `old_src_21017b9/src/response.jl:72-113`：`optimize_evidence_alpha` 固定点 15 步、clamp `1e-4..1e6`、**无 stationarity 证书** | — |
| 分类 | **理论深化**（证书驱动，SPEC §25：stationarity certificate 必须存在） + **数值变化**（迭代策略、域界） + **修 bug 候选**（V1.0 的"line search 失败即返回"类做法被 SPEC §25 禁止；当前在无证书时抛错）。V0 的 BF 稀疏门控整体移除：**理论变化**（先验结构改变） |

### 1.6 Innovation law

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `modecov.jl:1-230`：13 点网格 `DGRID`（`L1`）；每模自己的 `d_k`（`L3-7, L67-92`）；Horn 平行分析 + tail 规则选模（`L121-160`）；idiosyncratic 块对角投影（`L164-193`）；`model.jl:527-548` 每场景抽 `mode_vol_model` 或 iid 残差行 | `predict.jl:76-119`：7 点 `DGRID_V1` + `DELTA_D_V1` 权重（`L10, L14, L94`）；FFT 精确卷积；残差来源为 OOF residual（下述 1.7）；`generate_scenarios_v1` 用单标量 `scale=sqrt(v_forecasts[d]/v_bootstrap)`（`L615`）+ 行 bootstrap/own-row（`L577-604`） |
| 语义 | 多模 + 异质 + 每模 FIGARCH 型长期记忆 + 尾部规则；尺度是模自身的 `σ` | macro 标量分数后验 + 绝对尺度归一（SPEC §31–§32）；relative 分量的风险由 predictive covariance 表达（`response.jl:1221-1229`） |
| 分类 | **理论变化**（innovation 法律从多模异质降为 macro 标量 + covariance；是否等价于 SPEC 的 KISS 版需要 SPEC §30 对照——SPEC 明言这是 V1 的 KISS 版非 Markov 风险，不是最终无限维 law） + **数值变化**（FFT vs `DSP.conv`；13→7 点网格且引入 `Δd` 权重，SPEC §31 明确要求含测度） |

【事实】V1.0 的 `mode_vol_factors`（`old_src_21017b9/src/predict.jl:205`）在 `generate_scenarios_v1` 中构造后**未被消费**：`predict.jl:237-250` 的 `eps_shock` 直接取 `res_history` 行、无缩放，最终 `scenarios_log = mu_s .+ eps_shock`（`L250`）。这是中间对照面的静态观察（未接线字段），当前实现有显式绝对尺度（`predict.jl:615`）。

### 1.7 Posterior uncertainty

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | 几何 bootstrap：`model.jl:402-403`（`bs`）→ 每个 draw 有自己 response posterior（`model.jl:465-469`，`Draw`）→ `predict` 均匀混合（`model.jl:523, 537`）；drift 层级后验（`mu0sd/sd`，`model.jl:536`） | 决策时精确边缘化：`predictive_moments`（`response.jl:1211-1229`）返回 `mu_m/var_m/mu_rel/L_rel`；`L_rel` 为 predictive covariance 的 PSD 主平方根（`numerics.jl:452-460`；`predict.jl:570-572`）；scenario 只抽 relative 因子 + macro 标量（`predict.jl:555-563, 610-622`） |
| 语义 | 抽整个 `G`（每 bootstrap 几何一个）；几何子空间不确定性显式传播 | 不抽整张 `G`；`Gx` 的边缘 + constrained covariance 精确表达（SPEC §33：这是精确边缘化，不是降阶近似）；**geometry draws 不再存在**（V1.0 的 `Phi_draws` 未出现在当前 `V1Model`，`predict.jl:16-36`） |
| 分类 | **理论深化**（SPEC §33 精确边缘化） + **理论变化/【需 SPEC 审查】**：geometry subspace uncertainty 从"显式 bootstrap 混合"退场，由固定 gauge + scalar ruler 的确定性几何替代。SPEC §14 论证了固定 gauge 消除了 sign/简并等跟踪问题，但"移除几何不确定性传播"是否在 SPEC §26 的 epistemic uncertainty 语义下被完全覆盖，属规范判断，本文不裁决 |

### 1.8 Scenario integration

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:521-548`：纯 IID MC（默认 S=1000）；**moment matching**（`L542-546`：`out .+= conditional_mean' .- mean(out, dims=1)`） | `predict.jl:546-625`：IID 或嵌套 randomized Halton（`ScenarioQuadrature`，`L507-531`）；`adaptive_scenario_weights` 用 `kelly_certificate` 的 objective gap + 权重 L1 收敛（`L627-653`）；**无 moment matching**（`predict.jl:533` docstring "No post-hoc mean matching."；`defect_markers.txt:457, 521` 记录该否定声明由 `c109d9f` 引入） |
| 语义 | 强制场景样本一阶矩等于解析预测均值（V0 注释自称 numerics-only） | 由 quadrature refinement + Kelly certificate 控制采样误差（SPEC §37：数值 refinement 依据 `‖w_{2S}-w_S‖₁` 与目标证书，不得用回测 Sharpe） |
| 分类 | **理论变化/认识论**（moment matching 移除；收敛改由证书证明） + **数值变化**（Halton） |

【事实】V1.0 亦有 moment matching（`old_src_21017b9/src/predict.jl:253-254`）。移除时间点为 `c109d9f`（2026-10-06 19:00，`defect_markers.txt:19-20, 448-457, 521`）。

### 1.9 Kelly feasible set

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:553-568`：Clarabel `max sum(log(X f + base))/S`，`f≥0, sum f = budget`；结果 `clamp.(..., 0, budget)` 后 `out .* (budget/sum(out))`；`allocate`（`L600-621`）中 locked 为 `held .* .!free`，未建模 locked 以常数财富 `fill(sum(locked[.!mask]), S)` 参与 base（`L614`） | `kelly.jl:6-134`：`kelly_inputs` 校验（`L6-15`：gross return 必须正有限；base 非负有限）；`fast_kelly_solver` + `kelly_certificate`（`L21-37`，concavity gap / KKT / feasibility），失败回退 `clarabel_kelly_solver`（`L39-58`，同目标同证书）；`locked_wealth`（`L137-148`）对每个 held 资产要求有预测律，缺则 error（`L143`） |
| 语义 | 已持有且不在 scenarios 中的资产 = 常数财富（等价于把其风险当 cash）；无 fast solver、无证书 | free 资产 = tradable ∧ active；locked 只乘真实持有的列；`0*NaN` 不再进入 base；同一 log-Kelly 目标的两个 solver 都须通过证书 |
| 分类 | **修 bug**（SPEC §38.1：locked 风险不得当 cash；`0*NaN` 防护与 fail-loud）+ **结构重构**（certificate 化；SPEC §39-§40）。归一化除零/空集边界为数值修复 |

### 1.10 Universe semantics

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:570-589`：`firstrows`；`eligible(f,T)=(T-f+1)>=MINROWS`（`L581-582`）；`MINROWS=WARMUP+GEOM_MIN+REG_MIN=256+256+64=576`（`L56-57`）；`scenarios` 只在 eligible 列上生成；`tradable` 默认 `isfinite.(adj[end,:])`（`L630`） | `predict.jl:125-135`：`active_universe_indices` = 至少一条有效日收益；`_prepare_v1` 对全部 active 列建模；`alive_now` = 决策日观测 mask（`prepare.jl:131`）；`mu` 嵌回全 universe（`predict.jl:485-486`）；`kelly.jl:150-160` free = tradable ∧ active |
| 语义 | 资产须有 ≥576 行历史才进入模型空间（硬门槛）；弱证据以排除表达 | 无 per-asset 硬门槛；弱证据由 posterior 表达（`defect_markers.txt:330` 引 V0 自述"weak evidence is expressed by the posterior ... not by exclusion"是 V0 的目标，但其 `eligible` 仍是硬门槛）；全 NaN dummy 严格排除（SPEC §11/§60） |
| 分类 | **理论变化**（信息集扩大：所有有收益历史的资产进入建模） + **结构重构**（active/eligible/alive_now 语义分层）。两条路径都满足 SPEC §11（至少一条有效 return）与 §60（dummy 排除），差异在 V0 额外要求 576 行 |

---

## 2. 答辩点名缺陷的逐条核实

### 2.1 `shrink_drift`

- **V0 存在**：【事实】函数定义 `old_src_1d9b7ad/src/model.jl:325-339`（`DriftPost` 结构 `L317-323`，网格 `TAUGRID = [0.0,0.1,0.2,0.4,0.8,1.6,3.2]` `L315`）；拟合调用 `model.jl:464`；`predict` 中消费 `model.jl:535-536`；测试命中 `1d9b7ad:test/runtests.jl:285, 289`（`defect_markers.txt:322-327`）。
- **移除版本**：【事实】`defect_markers.txt:328, 519, 534-535, 562-564`：`21017b9` 起零命中；`shrink` 家族（含空格变体）唯一存活于 `1d9b7ad`。
- **当前残留**：无。【事实】对 `src/` 全目录分别检索 `shrink_drift` 与 `shrink`（含空格/连字符变体）均零命中；`defect_markers.txt:534-535` 亦确认 V0 之后全历史零命中（此前一次宽匹配中的 `discard`/`forward` 命中来自同时检索的 `ARD` 子串，与 `shrink` 无关）。当前条件均值不含 drift 项（`predict.jl:480-486`）。
- 【推断】该缺陷的语义后果（横截面收缩先验消失）与 1.1 的行为差异直接相关；但"缺陷"定性以报告原文为准（见 0.3 第 2 条）。

### 2.2 ARD / BF 门控

- **V0 存在**：【事实】`bayes_ard`（`model.jl:248-286`）：`BF_MIN=10.0`（`L246`）；`weak = gain .< log(BF_MIN*M)`（`L279`）；`α[weak] .= ALPHA_MAX`（`L281`）；调用 `model.jl:453`；注释自述"26% of macro modes on white noise"的过发现（`L238-239`）。存在性矩阵：`defect_markers.txt:329-339, 538-548`。
- **移除版本**：【事实】`defect_markers.txt:330-333, 539, 553-559`：ARD 与 BF家族的唯一宿主为 `1d9b7ad`；`21017b9` 起零命中；字面 token 'BF' 全历史零源码命中（仅二进制 fixture）。
- **当前残留**：无。【事实】对 `src/` 全目录搜索 `bayes_ard` / `BF_MIN` / `ARD`（含大小写变体）无命中；`defect_markers.txt:538-539` 确认唯一宿主为 V0。当前先验为 Matrix-Normal + 全局 α（`response.jl:1160-1185`），EB 由证书把关（`response.jl:351-380`）。

### 2.3 Selection 泄漏承认注释

- **V0 存在**：【事实】`model.jl:18-21`：自述几何窗口与响应证据"disjoint"，随后承认"Estimating the eigenvectors on the very rows they are then regressed on is selection on the response: on pure noise P(|t|>1.96)=0.68 instead of 0.05"——即缺陷自认。存在性矩阵 `defect_markers.txt:343, 480-483, 522`。
- **修复版本**：【事实】`21017b9` 头部声明"Blocked Cross-Fitting: geometry is estimated out-of-fold to strictly eliminate selection leakage"（`old_src_21017b9/src/predict.jl:1-8`；`defect_markers.txt:492-501, 510-513` 引 `predict.jl:107-137` 的分块交叉拟合）。
- **当前残留**：无缺陷面。【事实】`src/` 无 `selection` 命中；`bin/bench.jl:47-53` 的 `selection` 是基准引擎选择变量名（`defect_markers.txt:363-367, 522`）。当前 OOF 为真正的 per-fold 独立 EB（`predict.jl:405-447`，train Gram = full − fold，`L407-410`）。
- 【事实补充，非报告点名但同族】V0 的残差 `resid` 是 **in-sample** 拟合残差（`model.jl:458-462`，同一 `ref` 几何的 `β`）；V1.0 的残差预测同样使用**全数据** `resp`（`old_src_21017b9/src/predict.jl:173`，`pred_t=predict_modes(resp,...)`），其"cross-fit"只发生在坐标 `z_rel_cf`（`L112-136`），不在模型参数上。当前 SPEC §27-§29 的 OOF 隔离（每 fold 独立 EB，`predict.jl:429-447`）是真正的 out-of-fit innovation。
- 【推断】"selection 泄漏"在两代旧版中形态不同：V0 是几何/响应重叠（已自认），V1.0 是坐标交叉但模型未交叉。二者都被当前实现取代；该演进与 residual oracle 的 9 行样本保真证据（AGENTS.md 第6任段）一致。

### 2.4 可能的 zero residual / mean matching

**zero residual**：
- 【事实】V0 无 zero-fill：`defect_markers.txt:66, 524`（`zero_fill` 在 V0 零命中）。V0 的缺失残差回退是 own-row 随机行（`model.jl:509-516` `innovation`；`modecov.jl:173-178`），若某资产无 own 行会直接 `rand(1:0)` 抛错——无显式零回退。
- 【事实】V1.0 存在 `: 0.0` 零回退：`old_src_21017b9/src/predict.jl:246`（`!isempty(own_r) ? model.res_history[rand(rng, own_r), j] : 0.0`）。
- 【事实】当前为 fail-loud：`predict.jl:598`（`isempty(own) && error("active asset has no observed OOF residual")`）。
- 【事实】`zero embedding`（当前）是规范语义而非缺陷：`predict.jl:38-42` 明确"absent components are defined to be zero in the field, not in the return data"，`incremental.jl:6, 32` 说明新资产列插入精确零；`defect_markers.txt:66, 524-527`（HEAD 命中为规范 zero embedding + 测试断言）。
- 分类：**修 bug**（把 V1.0 的静默零回退替换为显式失败；SPEC §56 fail-loudly）。

**mean matching**：
- 【事实】V0 存在 moment matching（`model.jl:542-546`）；V1.0 存在（`old_src_21017b9/src/predict.jl:253-254`）；当前不存在，且 docstring 明确否定（`predict.jl:533`）；`defect_markers.txt:19-20, 448-457, 521` 记录 `c109d9f`（2026-10-06 19:00）引入"No post-hoc mean matching"。
- 分类：**理论变化/认识论**（V0/V1.0 把它标为 numerics-only，当前以 quadrature + certificate 承担同一误差控制职责，SPEC §37）。

---

## 3. 语义漂移总结

### 3.1 缺陷清除（修 bug）

1. `shrink_drift` / ARD / BF 稀疏门控整体拆除（唯一宿主 `1d9b7ad`；`21017b9` 起零命中）——先验结构与选择门控被 Matrix-Normal + 证书化 EB 取代。
2. in-sample residual → 真正 per-fold OOF（SPEC §27-§29；V0 `model.jl:458-462`，V1.0 坐标交叉而非模型交叉）。
3. locked 资产风险不得当 cash：V0 `allocate` 的常数财富（`model.jl:614`）→ 当前 `locked_wealth` fail-loud（`kelly.jl:137-148`，SPEC §38.1）；`0*NaN` 防护。
4. V1.0 的静默 `0.0` 缺失残差回退 → 显式 error（`predict.jl:598`）。
5. Kelly 无证书 → 双证书（`kelly.jl:21-37`）与同目标 Clarabel 回退（SPEC §39-§40）。

### 3.2 理论深化（认识论升级）

1. 每模 ARD 先验 → 全局 Matrix-Normal + evidence-maximised α，且 stationarity / covariance 双证书（SPEC §25）。
2. neutrality：先验子空间（V0）→ 精确 Gaussian conditioning 于均值与协方差（当前；SPEC §22）。V1.0 的 mean-only 投影作为中间态记录。
3. posterior uncertainty：抽整张几何/响应 → `Gx` 决策时边缘化 + PSD 主平方根唯一化（SPEC §26, §33；`numerics.jl:452-460`）。
4. 场景误差控制：moment matching → Halton quadrature + `kelly_certificate` 收敛判据（SPEC §37）。
5. universe：576 行硬门槛 → 全部 active 资产 + posterior 表达弱证据（SPEC §11）。

### 3.3 可能改变策略行为但无理论判决（建议 SPEC 审查的候选）

1. **drift 层级先验的存废**（1.1）：当前 SPEC §34 不包含 drift 项，故形式合规；但它是一次实质先验删除，且是 V0→当前差异中横截面行为最敏感的一项。建议 SPEC 明确记录"层级 drift 是否被有意废弃、其统计职能由谁承接"。
2. **geometry subspace uncertainty 的退场**（1.7）：V0/V1.0 显式传播 bootstrap 几何不确定性；当前 `V1Model` 无 `Phi_draws`。固定 gauge 论证（SPEC §14）解释了几何**跟踪**问题，但未显式裁决"几何不确定性是否仍应进入 predictive"。建议 SPEC 给出结论。
3. **innovation 法律的多模 → macro 标量**（1.6）：SPEC §30 自述这是 KISS 版、非最终 law；但 V0 的每模异质 d 与 idio 投影被整体替换，其相对/横截面风险由 `L_rel` 承接。需确认在 SPEC §26 总不确定性公式下无风险丢失。
4. **V1.0 的 mean-only neutrality**（1.3）：若 SPEC 历史审计需要，此条应作为"修复提交中的局部退步"单列；当前实现已超越它。
5. **`MINROWS` 硬门槛移除**（1.10）：对极年轻资产，当前让其进入模型（由 posterior/EB 收缩），V0 排除。是否为有意的信息集扩大，建议 SPEC 一句确认。

### 3.4 相对行为的可能主因（排序，均为【推断】）

1. drift 层级移除（横截面收缩消失，低信噪比下预测形状改变）；
2. innovation 尺度与法律（多模异质 → macro 标量 + predictive covariance；scenario 尾部/分散度改变，Kelly 规模改变）；
3. OOF residual 化（残差宽度校准改变，`scale=sqrt(v_forecasts/v_bootstrap)` 的 `v_bootstrap` 取自 OOF）；
4. moment matching 移除（有限 S 下样本均值噪声不再被逐资产对齐）；
5. neutrality 处理差异（对 band 间投影的均值/协方差形状影响）。

---

## 4. 未能确定的项

1. 答辩报告原文不可读（`old_src_snapshots.txt:125-127`）；"shrink_drift / ARD / BF 门控" 是否就是报告全部点名项，未逐字核对（`old_src_snapshots.txt:128-129`）。
2. 旧版本之间（V0→V1.0→`85f6e97`→`d5e97dd`→`c109d9f`→…→HEAD）存在多个中间提交，本文只逐项给出"V0 / V1.0 / 当前"三点证据；精确的移除提交点除已引用者（shrink/ARD/BF=`21017b9`、mean matching=`c109d9f`、zero fill 规范=`HEAD`）以外未逐提交穷举。
3. 未运行任何命令，未做数值对照；"策略行为改变"的排序是静态推断，不是收益证据。
4. geometry uncertainty 退场的规范性归属（3.3-2）与 drift 存废（3.3-1）依赖不可读的规范史，本文仅提供可核对的源码事实。
5. `602b897` 快照（性能线终点）未在本文逐项展开；如需第四点对照，需另立任务。

---

## 附：证据索引（路径与行号）

- 旧快照（V0，8 文件）：`old_src_1d9b7ad/src/model.jl`（含 Kelly `L553-568`）、`old_src_1d9b7ad/src/modecov.jl`、`old_src_1d9b7ad/src/backtest.jl`、`old_src_1d9b7ad/src/broker.jl`、`old_src_1d9b7ad/src/data.jl`、`old_src_1d9b7ad/src/KTrader.jl`、`old_src_1d9b7ad/src/live.jl`、`old_src_1d9b7ad/src/repomix-output.xml`（V0 无独立 kelly.jl / geometry.jl 文件；`old_src_snapshots.txt:12-20`）。
- 中间对照（V1.0）：`old_src_21017b9/src/predict.jl`、`old_src_21017b9/src/response.jl` 等 9 文件（`old_src_snapshots.txt:29-38`）。
- 当前源码：`src/predict.jl`、`src/response.jl`、`src/prepare.jl`、`src/kelly.jl`、`src/numerics.jl`、`src/geometry.jl`、`src/residual_oracle.jl`、`src/incremental.jl` 等 13 文件。
- 侦查证据：`dev/evidence/manager13/defect_markers.txt`（A–F 节）、`src_evolution.txt`、`old_src_snapshots.txt`（含聚合与逐文件 SHA-256）。
- 规范对照：SPEC §11、§14、§19、§22、§24-§26、§27-§29、§30-§32、§33-§34、§37、§38-§40、§56（AGENTS.md 内）。

---

*本报告为纯静态语义比对；所有分类均为该层级内的工程判断，推断处已标【推断】。不宣布任何发布、验收或收益结论。*

==> docs/POSTERIOR_DEFINITION.md <==
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

==> docs/TRACE_NEUTRALITY_DERIVATION.md <==
# Trace Neutrality 推导与裁决

**文档状态：静态分析记录（非规范性文档，不改写 SPEC）**
**任务范围：Gate 0 重开后的 Deliverable 3。本轮未运行任何命令、未执行任何数值实验；全部结论来自源码、测试、文档与 dev 注记的逐字阅读。若本文件与 SPEC/AGENTS 冲突，以后者为准。**
**结论先行：当前仓库中不存在从 price-only 对称性出发生成的 trace neutrality 推导。该约束的诚实裁决是 candidate hypothesis（V1 显式建模假设），尚不能升级为 theorem 或 identification requirement。**

---

## A. Forbidden direction 的精确语义

### A.1 机制事实（逐字核对）

`src/response.jl:106-108`：

```julia
function get_constraint_columns(N::Int, n_bands::Int)
    [(((b-1)*2+channel-1)*N+1):(((b-1)*2+channel)*N) for b in 1:n_bands for channel in 1:2]
end
```

- 外层遍历 `b = 1:n_bands`，内层 `channel = 1:2`；每个元素是一个长度 `N` 的连续列区间。
- `n_bands=7`（`BANDS = 2 .^ (1:7)`）时共 **14** 个区间，顺序为：`(b1,Q),(b1,P),(b2,Q),(b2,P),…,(b7,Q),(b7,P)`。

设计矩阵列布局（`src/response.jl:55-75`，`fill_design_matrix!`）：

```julia
offset = (b-1)*2N
col_q = offset + j
col_p = offset + N + j
```

- 每个 band `b` 占 `2N` 列，前半 `N` 列为 Q 通道（逐资产），后半 `N` 列为 P 通道（逐资产）。
- 因此 `cols[c]` 展开后第 `j` 个列索引恰对应资产 `j` 在该 band/channel 的自己的 feature 列。

约束矩阵 `C` 的显式构造见于 `test/conditioned_eb_tests.jl:155-158`：

```julia
cols=KTrader.get_constraint_columns(N,length(KTrader.BANDS)); C=zeros(length(cols),N*P)
for c in eachindex(cols),j in 1:N
    C[c,j+(cols[c][j]-1)*N]=1.0
end
```

按 Julia 列主序，`vec(G)` 中 `G[i,p]` 位于下标 `i+(p-1)*N`。故对第 `c` 行：

\[
(C\,\mathrm{vec}(G))_c \;=\; \sum_{i=1}^{N} G\!\left[i,\;\mathrm{cols}[c][i]\right].
\]

因为 `cols[c]` 是连续区间且区间内第 `i` 列对应资产 `i`，该式化为所在块的 **对角和**：

\[
c=2b-1:\quad \operatorname{tr} A_b \;=\; \sum_{i=1}^{N} G[i,\,\text{asset }i\text{ 的 Q 列}],
\qquad
c=2b:\quad \operatorname{tr} B_b \;=\; \sum_{i=1}^{N} G[i,\,\text{asset }i\text{ 的 P 列}].
\]

这与 `src/response.jl:128` 的实现 `h[c] = tr(view(G, :, cols[r]))` 以及 `ridge_constraint_traces`（`src/response.jl:412-418`，直接收缩各块对角）逐字一致；`dev/response_deadwork_20261008.md:7-19` 的等价改写记录同样确认该语义。

### A.2 语义陈述

`C` 的每一行是一个作用在系数空间上的线性泛函，语义为：

> **在固定的时间尺度 band \(b\) 与固定的通道（Q 或 P）上，全部资产"对自身输入"的系数之和为零。**

- Q 与 P 在此处是 AGENTS §16 定义的 paired local path coordinates：Q 对应"位置偏离"（`-(X-c0)/s`），P 对应"方向/确认"（`(c0-c1)/s`）。`A_b`/`B_b` 只是 G 在该块上的记号（AGENTS §17：\(G_b=A_b+iB_b\)），**不是"实部/虚部"意义上的频域对象**。
- 约束只涉及**块内对角配对** \((i,\text{asset }i\text{ 自己的列})\) 的和，**不涉及**块内 \(i\neq j\) 的 off-diagonal 项，也不涉及跨 band/跨通道的任意组合。
- 约束禁止的是一族方向：任何满足 \(\operatorname{tr}A_b\neq0\) 或 \(\operatorname{tr}B_b\neq0\) 的系数矩阵。AGENTS §21 给这族方向的解释是"每个时间尺度的 universal common timing response 被移除"；含义文本为"所有资产在某 band 上同样的趋势或同样的 anticipatory bias"。**注意语义精度**：C 实际杀死的是**对角和**；例如块内 \(G[i,j]=g\)（全常数矩阵）满足 \(\operatorname{tr}=Ng\neq0\) 会被移除，但块内 \(G[i,i]=0,\ G[i,j]=g\ (i\neq j)\) 的纯交换分量 \(\operatorname{tr}=0\) 不被该约束禁止。也就是说，"universal common response 被移除"这一表述只在"共同响应指对角自响应分量"时才与 C 严格等价；若共同响应还包括资产间的对称传导分量，则 C 并不移除它。这一语义缺口在本文件 D 与未决问题中继续处理。

---

## B. 为什么它应当是 hard removed（identification），现有论证是否成立

### B.1 仓库现有论证的原文与位置

- AGENTS §21（line 704-723）："它是 structural / identification constraint，不是防过拟合的数值 regularizer。"
- AGENTS §22（line 727-765）："禁止：先拟合无约束 posterior；只把 mean 的 trace 减掉；covariance 仍允许离开 neutral subspace。"并给出精确条件化公式 \(m_c=m-\Omega C^T(C\Omega C^T)^{-1}Cm\)、\(\Omega_c=\Omega-\Omega C^T(C\Omega C^T)^{-1}C\Omega\)。
- AGENTS §61（line 2083-2093）：宪法测试要求 posterior mean / sample support 均满足 \(\operatorname{tr}A_b=\operatorname{tr}B_b=0\)，threshold 应接近 machine precision。
- `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217`："Posterior uncertainty over the path-response law and trace neutrality are theoretical constraints that would remain under unlimited compute."
- AGENTS 历史段 §13（line 3087）把"trace neutrality 只是 post-hoc mean correction"列为 0.9 时代的核心缺陷；§12（line 3051-3064）称"早期双重 neutralization 被统一成 operator constraint"。

### B.2 评估

把现有材料拆成可检验的主张：

1. **"必须存在于 posterior support 而非只修 mean"——成立且证据充分。** 后验协方差若不条件化，scenario 抽样（`generate_scenarios_v1` 消费 `L_rel`）仍会离开 neutral 子空间；`condition_trace_neutrality` 的 dense oracle 对照（`test/conditioned_eb_tests.jl:190-198`，`exact.G_c ≈ dense`、`exact.inv_M ≈ inv(Symmetric(M))`，atol 1e-10）证明实现与标准条件化公式一致。这一条是"hard conditioning 的正确实现方式"，不是识别性证明。
2. **"它是 identification constraint 而非数值 regularizer"——作为识别性主张，仓库材料不足以支持。** 识别性（identification）的正式标准是：约束方向在观测分布上不可识别，或与已建模通道严格冗余/共线。仓库中没有任何一处证明 \(\operatorname{tr}A_b\)、\(\operatorname{tr}B_b\) 方向在 data-law 下不可区分。相反，`dev/theory_incremental_20261009/THEORY.md:243-251` 明确记录：约束后的修正被分配到"未识别方向"，且"未识别系数的均值不一定是 0"、"不同支撑块会经 14 个 trace 约束产生条件相关"。这段文字实际上把 forbidden 方向当作**可以被条件化机制触碰的方向**，与"这些方向已被识别掉、不存在"的强识别语义不符。
3. **"无限算力下仍存在"——不蕴含 identification。** 一个硬先验/支撑选择同样在无限算力下存在；该句只能排除"这是数值截断/近似"的误读，不能排除"这是建模选择"。
4. **一个尚未被仓库提出的潜在识别论证（本文档分析，非既有依据）**：relative 目标 \(Y\) 与 innovation 位于严格 zero-sum relative support（AGENTS §12；`fit_response_operator` 对 `Sigma` 做 `supported_covariance(..., gauge)`；`conditioned_eb` 以 `gauge` 参数投影），若模型把"\(E\) 严格支撑在 \(\mathbf{1}^\perp\)"当作精确约束，则每行预测 \(XG^T\) 也需行和为零；而对角均匀方向 \(cI\) 对行和的贡献是 \(c\cdot\sum_{p\in\text{block}}X[t,p]\)，一般不为零。这条路线**若**能被形式化，可以论证该方向被 target support 的似然结构识别为 0，硬约束只是把有限样本下的估计噪声提前截掉。但仓库没有做这个推导，且它需要与 §34 的 \(\mu_{rel}^{zero-sum}\) 使用端投影交互核对，因此当前只能列为"升级所需的可能推导形态"，不能算作既有证据。

### B.3 小结

现状是：**hard removal 在工程上被完整实现且自洽（精确条件化、进入协方差、有 dense oracle 测试）；但"它是 identification requirement"这一规范性主张在仓库中没有推导支撑，属于被制度化的声明。** 因此对 B 项的诚实回答是：为什么它"应当是"hard removed，在现有材料中找不到可引用的证明；能找到的只是"V1 选择这样做"的记录。把 mean 修掉、covariance 不管的做法有明确反例意义（scenario 会违反 §61），这一半成立；"因此必须 hard remove 而非留给 posterior"另一半的识别性前提未证。

---

## C. 为什么是逐 band、逐 Q/P 的 14 条约束

### C.1 事实

- 14 = \(2\times|BANDS|=2\times7\)（AGENTS §22 line 765："把核心约束 solve 压到 14×14"）。
- 粒度是 per-band、per-channel（A 项已证）。`test/relative_support_tests.jl` 的 fixture（N=2, gauge 维 1）与 `dev/theory_incremental_20261009/THEORY.md:251` 的"14 个 trace 约束"均按此粒度引用。

### C.2 为什么不能更弱——变体差异分析（本文档推导；仓库未给直接论证）

仓库中**未找到**"为什么是这一特定粒度、而不是这些变体"的论证。以下比较是本文档基于线性代数与语义的静态分析：

1. **只保留一条全局约束 \(\sum_b(\operatorname{tr}A_b+\operatorname{tr}B_b)=0\)**：
   - 个别 band 的共同分量可相互抵消而整体通过约束。例如 band1 对角 \(+c\)、band2 对角 \(-c\)，全局和为零，但每个尺度内部都存在真实的 common response，与 §21 的文字含义（"每个时间尺度的……被移除"）不符。
   - 对后验几何的影响：单条约束的零空间远大于 14 条，forbidden 方向只是被"平均"掉，per-scale 识别方向保留在 support 中。
2. **只保留 \(\operatorname{tr}A_b+\operatorname{tr}B_b=0\)（合并 Q/P）**：
   - Q 与 P 是语义不同的通道（位置偏离 vs 方向/确认）。合并后共同的"位置偏差响应"与共同的"确认/预期偏差"可以互相抵消：\(+\delta\) 在 Q、\(-\delta\) 在 P 时通过约束，但两条经济通道各自的共同分量都未被移除。
   - per-channel 分开保证每个语义方向独立地净共同响应为零。
3. **只约束 posterior mean（post-hoc trace correction），covariance 不条件化**：
   - 这是 0.9 历史缺陷（AGENTS §13）。scenario 抽样经 `L_rel` 产生，若支撑未条件化，samples 会离开 neutral 子空间，§61 的"posterior mean / sample support 均满足"无法成立。
   - 因此该变体的差别不是"少一条约束"，而是"约束没有进入模型的预测分布"——AGENTS §22 已明确禁止。
4. **一般投影 \(P_{\text{forbidden}}G=0\)**：
   - 若 \(P_{\text{forbidden}}\) 的零空间恰等于 14 条 C 行的零空间，则与当前约束**数学等价**；当前 `condition_trace_neutrality` 正是该投影在 \(\Omega\) 度量下的精确条件化实现（B.2 第 1 条）。
   - 若"更一般"指不同的投影（如只把输出方向 e0 上的整体载荷清零），则与 C 不同：C 是按"输入资产自己的列 × 输出行求和"配对的 14 个泛函 \(\left(\sum_i G[i,\text{asset }i\text{ 的列}]\right)\)，而 e0 输出通道是 \(\sum_i G[i,p]\) 对每个固定输入列 \(p\) 的载荷。两者在块内全常数矩阵 \(G[i,j]=g\) 上会给出不同结果（C 得到 \(Ng\)，e0 通道每列得到 \(Ng\) 同样非零，但结构不同；在 \(G[i,j]=\delta_{ij}c\) 上 C 得到 \(Nc\) 非零，而每个固定列 \(p\) 的 e0 通道载荷为 \(c\) 也只有当 \(p\) 是该资产列时非零）。
   - 频域变体 \(\operatorname{tr}G(\omega)=0\ \forall\omega\)：项目没有频域算子表示；finite basis 下任何有限条约束都不能逐点表达该无限维条件，除非对 G 的跨频率结构另行假定。且 AGENTS §16 明确禁止把任意 Fourier 相位无条件映射到当前 paired basis 的金融语义。因此该变体在 1.0/2.0 的 finite-basis 架构中不可直接实现；当前 14 条是对每个离散尺度、每个通道的自然有限表述。
5. **为什么不更强**（例如同时约束 off-diagonal 之和、或整块均值）：
   - 仓库未讨论。可静态观察：C 只删对角和意味着模型仍能用 off-diagonal 系数表达任何跨资产传导结构；若连 off-diagonal 总和也约束为零，将禁止"全体资产同向传导"这类可能真实的结构，把约束从 identification 风格推向更强的经济先验。当前选择保留了最大灵活性，同时只移除"资产对自身输入的共同响应"。

### C.3 小结

"14 条"与实现结构（2 通道 × 7 尺度 × N 资产块的 per-block 对角和）严格对应，工程上是自洽的最小完备集：**每个 band、每个通道各一条，互不抵消、互不遮蔽**。仓库对"为什么不能更弱/更强"没有成文论证；上述差异比较是本文档的补充分析。

---

## D. 不变量审计

### D.1 permutation（§58）

- C 的每一行是"资产索引 i 同时选择输出行 i 与输入列（资产 i 自己的列）"的求和，求和哑标 \(i\) 对指标重命名不变。资产置换 \(\Pi\)（输出行与输入列块同步重排）下 \(C\cdot\mathrm{vec}(G)\to C\cdot\mathrm{vec}(\Pi^\top G\Pi_{\text{block}})\)，数值仍为各块对角和——**约束集合与置换兼容**。
- 测试覆盖：`test/v1_constitutional_tests.jl:13-18` 断言 `mu_pred`、`res_history`、`L_rel*L_rel'` 的置换协变，但**没有直接断言 `G_c_mean` 的约束行在置换下的数值不变**。分析支持兼容，直接测试缺失。

### D.2 price-scale（§57）

- 设计矩阵经 `inv_scales`（由 ruler 给出）标准化，系数空间与价格单位无关；C 是纯系数线性泛函。约束与价格缩放解耦。
- 测试覆盖：`v1_constitutional_tests.jl:9-12` 断言缩放不变性于 `mu_pred`/`L_rel` 层，**不含针对 C 的直接断言**。

### D.3 relative gauge 旋转（§59）

- 关键事实：`constraint_moments`（`src/response.jl:109-130`）**完全不使用 gauge**；它只用 `V.basis`、`V.weights`、`Sigma` 与 `G` 的块对角。`condition_trace_neutrality` 同样不含 gauge。
- C 的行在 asset 空间定义（对输出行求和即对 e0 方向的载荷），而 gauge 旋转只作用在 \(\mathbf{1}^\perp\) 内部（Helmert basis，AGENTS §13）。因此**约束本体在 gauge 旋转下表示无关**：\(\mathbf{1}^\top Q=0\) 使 e0 方向在 gauge 下不动。
- gauge 影响的是 conditioning 中 \(\Sigma\) 的 relative support 与证据计算（`_conditioned_block_geometry` 对块做 `gauge'*view(...)` 投影，`src/response.jl:420-424`）。因此端到端 gauge covariance 需要经由 \(\Sigma\) support 传播后仍成立——现有测试是间接的：`test/relative_support_tests.jl` 用 N=2（gauge 维 1，旋转群平凡）的解析消元 fixture；`test/conditioned_eb_tests.jl:148-199` 在 `gauge=Q` 下对照 dense oracle，但不旋转 Q 本身。**直接的 gauge-rotation-invariance 测试（Q→QR）针对 trace 约束未见。**

### D.4 band 表示

- C 的 14 个块与 `BANDS` 的顺序、数量硬绑定（`(b-1)*2N` offsets）。改变 bands 集合或顺序即改变约束集合。这是**表示依赖，不是不变量缺陷**——bands 是设计选择而非对称性；但没有 band 置换等价性的测试（也不应有，不同 band 是不同语义对象）。

### D.5 是否强迫 diagonal self-response 转移到 off-diagonal（sink 风险）

**数学形式（本文档分析）**。条件化后的均值修正为：

\[
m_c-m=-\Omega C^\top(C\Omega C^\top)^{-1}Cm.
\]

- 修正位于 \(\mathrm{span}(\Omega C^\top)\)，即 14 个 forbidden 行在 \(\Omega\) 度量下的像所张成的子空间。\(\Omega\) 的结构决定这 14 个方向如何分布到具体系数上。
- 若数据（或先验）在无约束后验里支持一个共同对角响应（所有资产的 own-response 相同，\(Cm\neq0\)），条件化会把它删除并沿 \(\Omega C^\top\) 的方向重新分配。若 \(\Omega\) 中 forbidden 方向与"同块 \(i\neq j\)"项相关较强，修正就会出现在 off-diagonal 系数上；预测层面等价于把"共同自响应"重写为"资产间互相响应"。
- `dev/theory_incremental_20261009/THEORY.md:243-251` 记录了相关现象的两面：约束修正被分配到"未识别方向"（\(P_A G_c P_A=-(δ/α)\lambda_c P_A\)），并警告"不同支撑块会经 14 个 trace 约束产生条件相关，不能只留对角方差"。这证明项目已知修正会跨方向传播，**但没有量化其是否集中到少数 cross-asset 方向**。
- **仓库中没有**任何实验、日志或测试断言测量修正 \(\|m_c-m\|\) 在块内各资产上的分布，因此"人为制造少数 cross-asset signal sink"无法从现有材料证实或证伪。静态上可以确定的是：修正的集中性完全由 \(\Omega\) 的低秩/块结构决定，而不是 C 本身单独决定；当 \(\Omega\) 近各向同性时修正分散，当 \(\Omega\) 低秩时可能集中。
- 另一点可以静态确定：无论修正如何分布，**它不改变条件后验的自洽性**（dense oracle 已证），改变的只是模型把数据信号表达为 diagonal 还是 off-diagonal 的**经济归因**。这是模型选择偏差问题，不是数值正确性问题。

### D.6 审计汇总

| 不变量 | C 本体是否满足 | 直接测试 |
|---|---|---|
| permutation | 分析成立（哑标求和） | 无直接断言（仅 mu/cov 层间接） |
| price-scale | 成立（系数空间无单位） | 无直接断言 |
| relative gauge 旋转 | 约束本体表示无关（不含 gauge；e0 不动） | 无 Q→QR 直接测试；间接 fixture |
| band 表示 | 与 BANDS 硬绑定（设计选择） | 不适用 |
| diagonal→off-diagonal 转移 | 风险路径存在；程度未量化 | 无任何测量 |

---

## E. 是否存在从 price-only 对称性出发的真正推导

**未找到。**

已检索的位置与结果：

- `AGENTS.md`：§21/§22、§57-§61、§12/§13 历史段、附录 C 第 9 问（"trace neutrality 为什么是理论 constraint？"）。全部为**声明式/历史式**文本，没有任何"从价格序列的某个对称性或可观测等价性推出 C"的推导。
- `README.md`：未检索到 trace neutrality 的推导段。
- `CHANGELOG.md` 与 `dev/**/*.md`：仅有 `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217` 的"theoretical constraints that would remain under unlimited compute"一句（声明），与 `dev/theory_incremental_20261009/THEORY.md:243-251,353`（讨论约束**之后**的后验几何与数值保留，不是约束的证成）。
- 源码注释：`src/response.jl` 相关函数注释只解释计算次序与等价改写（如 `ridge_constraint_traces` 的收缩），不含约束来源的推导。
- 提交注记：本机 `.git` 存在，但本轮不允许运行任何命令；**无法查阅提交历史**。就仓库可读文本而言，AGENTS §13 的历史叙述（"早期双重 neutralization 被统一成 operator constraint"）表明其来源是设计演进中的收敛选择，而非从数据对称性演绎。若提交历史中存在推导，本文件无权声称已覆盖——按委托口径，此类内容记为"本机可读文本中未找到"。
- 搜索关键词覆盖：`trace neutrality`、`neutrality`、`universal common`、`common timing`、`anticipatory bias`、`identification`、`structural constraint`、`gauge freedom`、`uniform`、`per-band` 等（大小写与中英文混用）。

---

## 裁决

### 分类

**candidate hypothesis（V1 显式建模假设），当前不满足升级为 theorem 或 identification requirement 的证明标准。**

- **theorem**：不可行。没有任何从更高层公理（price-only、固定规律、价格几何）到 C 的演绎链。C 是关于响应算子**系数**的结构约束，price-only 输入原语本身不产生它。
- **identification requirement**：未证明。缺少"该方向在观测分布上不可识别，或与已建模通道（macro 通道、relative 支撑）严格冗余"的论证。反而有理由认为该方向在有限样本中是**可拟合、可识别的**（似然对它的敏感度一般非零）；被移除是建模偏好。
- **candidate hypothesis**：与现有文本最一致。README.pre-final.md 的"would remain under unlimited compute"只支持"非数值近似"，不支持更强级别。THEORY.md:243-251 的"未识别方向"措辞同样与"已识别掉"矛盾。

### 升级所需的推导或证据形态

任一即可构成升级路径，三者都要求形式化证明与可复核的测试/反例：

1. **target-support 识别论证**：证明在"innovation 严格支撑于 \(\mathbf{1}^\perp\)、relative 目标行和为零"的模型下，forbidden 方向 \(cI\) 对观测似然的 Fisher 信息为零或被 target support 精确排除（需与 §34 的 \(\mu_{rel}^{zero-sum}\) 使用端操作交互核对，因为使用端投影本身会改变 forbidden 方向的可观测性）。
2. **参数冗余/重参数化不变性**：证明对任意 \(G\)，存在 \(G'=G+\Delta\)（\(\Delta\) 落在 forbidden 方向且非零）使所有可达预测 \(Gx_t\) 与 \(G'x_t\) 在观测等价意义下相同，从而约束是"去掉冗余参数化"而非"删除可识别自由度"。
3. **对称性公理**：给出一条独立于 C 的价格世界假设，并证明 C 是其必要条件；同时该假设不能被"真实世界共同响应存在"这类反例反驳。按当前证据，路径 3 最难成立。

配套证据形态：一个能红/绿的最小测试（例如构造数据含真实共同对角响应，验证移除前后预测差异与识别论证的预测一致），并在 \(\Omega\) 的结构下量化 D.5 的修正分布。

### "降级为 explicit hypothesis、不进入 2.0 core" 的操作含义

- **不是**删除约束：`get_constraint_columns`/`condition_trace_neutrality`/`ridge_constraint_traces` 机制、AGENTS §61 的测试、14×14 条件化全部保留不动；不得无声删除、不得放宽阈值。
- **是**措辞与地位修正：在所有对外叙述（SPEC/README/发布说明）中把该约束标注为"V1 声明的建模假设（candidate hypothesis），未证明为 identification"；不得再以"理论约束/识别性要求"的确定口吻对外宣称；与 macro/relative 分解、Q/P basis 等其它已声明的建模选择并列管理。
- **重审触发器**：出现 (a) 上述升级证明，或 (b) 反例证据表明该约束造成系统性预测扭曲（如 D.5 的修正集中在少数 cross-asset 方向并被数据证伪），则重新裁决。
- 本文件不改动任何规范文本；是否执行降级措辞由规范性文档（AGENTS/SPEC）的 owner 决定。

---

## 未决问题清单

1. **升级证明缺失**：B/E 所述识别性论证不存在；target-support 路线尚未与 §34 使用端投影的交互核对。
2. **D.5 修正分布未量化**：conditioning 修正 \(\Omega C^\top(C\Omega C^\top)^{-1}Cm\) 在 off-diagonal 上的集中程度无任何实验/日志；本轮禁止数值实验，故保持未决。
3. **语义缺口**：C 只移除对角和；"universal common timing response 被移除"的 AGENTS §21 措辞在"共同响应含资产间对称传导"的解释下过宽。是否需要在规范中收紧措辞，未决。
4. **gauge 旋转直接测试缺失**：约束本体分析为表示无关，但 Q→QR 的端到端协变性无直接测试。
5. **提交历史未阅**：本机 `.git` 不可访问（无命令权限）；若提交注记中存在推导，本文件的"未找到"仅覆盖可读文本。

---

## 关键证据位置索引

- `src/response.jl:106-108` — `get_constraint_columns` 的 14 块构造。
- `src/response.jl:109-130` — `constraint_moments`：\(h_c=\operatorname{tr}(G[:,\text{cols}[c]])\)。
- `src/response.jl:131-148` — `condition_trace_neutrality`：精确条件化实现。
- `src/response.jl:412-418` — `ridge_constraint_traces`：对角收缩等价改写。
- `src/response.jl:55-75` — 设计矩阵列布局（Q/P 逐资产）。
- `test/conditioned_eb_tests.jl:155-158` — C 的显式构造（语义锚点）。
- `test/conditioned_eb_tests.jl:190-198` — dense oracle 条件化对照（实现正确性证据）。
- `test/v1_constitutional_tests.jl:21-27` — §61 约束的宪法测试。
- `test/relative_support_tests.jl` — gauge 支撑（N=2）分析性测试。
- `AGENTS.md:704-765` — §21/§22 规范文本。
- `AGENTS.md:2083-2093` — §61 宪法测试要求。
- `AGENTS.md:3051-3064, 3087` — 历史段（约束统一与 0.9 缺陷）。
- `dev/theory_incremental_20261009/THEORY.md:243-251` — 约束后验几何与"未识别方向"记录。
- `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217` — "theoretical constraints / unlimited compute"声明。
- `dev/response_deadwork_20261008.md:7-19` — trace 收缩的等价性记录。
