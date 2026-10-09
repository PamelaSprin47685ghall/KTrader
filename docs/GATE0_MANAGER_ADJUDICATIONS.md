# KTrader Gate-0 Manager 终审裁决记录

| 项 | 内容 |
|---|---|
| 文档状态 | Manager 终审裁决记录（代录落盘），自落盘起生效（裁决 I1） |
| 裁决日期 | 2026-10-09 |
| 裁决者 | Gate-0 Manager |
| 优先级 | AGENTS.md 裁决书 > 本裁决记录 > 三份 Gate-0 文档修订版 > 旧 docs/ 分析文档（裁决 I1） |
| 代录身份 | Engineer 代录：裁决内容与权柄在 Manager，代录不增删裁决实质内容；自由度仅限文档结构组织（标题层级、编号）、格式（Markdown 表格/列表），以及「依据」「可推翻条件」字段的呈现——只能使用裁决文本中已给出的依据，不得自行发明新依据 |

## 0. 头部声明

本文件是 Gate-0 静态设计交叉审查（CrossAuditorAlpha 接口对齐报告 + CrossAuditorBeta 施工图一致性报告 + 四份 Wave-1 交付物的待裁项清单）的 Manager 终审裁决记录，代录落盘。

- **裁决日期**：2026-10-09
- **被裁决对象**：
  - `docs/GATE0_RESPONSE_POSTERIOR.md`
  - `docs/GATE0_VECTOR_INNOVATION.md`
  - `docs/GATE0_IMPLEMENTATION_PLAN.md`
  - 及其待裁项：
    - innovation 文档 T1-T7
    - response 文档 §12 七项
    - 施工图第 10 节 9 项开放决策
    - 两审查报告的全部阻断级与非阻断级发现

## 裁决索引

| 裁决 | 主题 | 条目 |
|---|---|---|
| A | 域实例化（D-045a 澄清条款） | A1-A5 |
| B | 超参先验修正（D-035a 修正条款） | B1-B4 |
| C | innovation 语义默认（T1/T2/T4/T7） | C1-C4 |
| D | 接口锚统一（吸收 Alpha-B1/B2 与 M2/M4/M5） | D1-D4 |
| E | ragged 归一语义（M1） | E1 |
| F | RNG/抽样契约（M3） | F1 |
| G | 施工图更新指令（吸收 Beta-B1'/B2'/B3' 与 N 系） | G1-G8 |
| H | Gate0Planner 九项开放决策 | H1-H9 |
| I | 生效与边界 | I1-I3 |

---

## 裁决 A｜域实例化（D-045a 澄清条款）

**A1.** V_t(d)、标准化 shape z、joint row pool 全部实例化在决策日 risk 域 R_t = free ∪ locked 的 mode 坐标 E_{R_t} = [e₀^{(R_t)}, Q^{(R_t)}] 上（N_R 维）。

**A2.** ε 的源对象是 asset 空间 OOF 残差行（含 b₀ 扣除）。response 层新增输出契约：asset 空间 OOF 残差行 ε̃_s = E_active · [y_s − (b₀^{(−fold(s))} + G^{(−fold(s))} · x_{s−1})]，其中 ŷ = E[y_s | train^{(−fold(s))}] 为 D-041 期望语义（fold posterior 的完整后验均值，含 α 积分）。

**A3.** innovation 层负责：从 asset 空间残差行提取 R_t 子向量 → 经 E_{R_t} 的 mode 变换 → V_t/z/pool 构造。禁止从 active 域 mode 残差直接取「子向量」（数学上无定义）。

**A4.** μ 契约：response 层规范输出 asset 空间 μ_asset = E_active · μ（active 域 N 维 asset 坐标）；scenario 合成在 R 子集上执行：μ_R = E_{R_t}ᵀ · μ_asset[R_t]；y_R = μ_R + ε_R；u_R = E_{R_t} · y_R（R 域 asset normalized return）；乘 s₁[R_t] 恢复 log return；exp 成 gross。Kelly scenario 矩阵列集 = R_t（free 列参与优化，locked 列进 base_s）。

**A5.** 方差分解（裁决书 §30 报告义务）在 asset 空间 R 子集上报告：Var(u_R) = E_{R_t}·Var(μ_R)·E_{R_t}ᵀ + E_{R_t}·E[V_R]·E_{R_t}ᵀ（同域可加）；active 域全量 epistemic 分量可作诊断附加项（标注 NOT REQUIRED）。

**依据**：D-055/D-056 的「risky assets」字面语义；active 域设计下 D-056 cash 出路数学上结构性失效（mode 坐标分量混合全部资产，不可按资产取子向量——CrossAuditorAlpha 核实的数学事实）；裁决书 §19 八步链条未指定域实例化，属推导自由度。

**可推翻条件**：若 owner/SPEC 正式审查裁定 active 域字面为准，则 innovation 层需重新设计行集收缩机制，本裁决 A2-A5 全部重议。

## 裁决 B｜超参先验修正（D-035a 修正条款）

**B1.** α₀ 与 α_p 的先验从 p(α) ∝ 1/α 修正为 p(α) ∝ 1/[α(1+α)]（log 坐标 u = log α 下密度 ∝ 1/(1+e^u)：u→−∞ 趋于常数即 log-uniform 局部行为，u→+∞ 指数衰减截断无穷远平台）。

**B2.** 修正依据：纯数学 propriety 论证——α→∞ 时 evidence 饱和于零模型平台 K(n,N)·|S_yy|^{−n/2}（与 α 无关的正常数），log-uniform 尾测度发散，后验恒不 proper（PosteriorDeriver 推导 §5.3-§5.4 的数学事实）；修正保持 reference 先验局部行为；本修正不是通过回测选出来的（D-035 纪律）。

**B3.** D-036 propriety gate 语义更新：修正后 gate 应绿；若红是实现 bug（不再是「预期红」）。

**B4.** D-038 的「禁止固定 EB_ALPHA_MIN/MAX 作为 prior support」不受本修正影响（修正的是先验本身，不是截断域）。

**可推翻条件**：若 owner/SPEC 正式审查否决本修正，回退 p(α) ∝ 1/α 并恢复 D-036 恒红状态（此时 Step 8 路线整体阻塞，需重新裁决先验形式）。

## 裁决 C｜innovation 语义默认（T1/T2/T4/T7）

**C1.（T1）** V_t(d) 的分母行集收缩：分子分母同步收缩到 joint 合法行集（行集 J_t 的行才进统计）。

**依据**：V_t 的统计只由实际观测的行定义；与 D-050 的 support 语义一致。

**C2.（T2）** ℓ_d 的求和行集 = V_{s−1} 可定义行集；单一行集概念：joint pool 行集 = V 统计行集 = ℓ 求和行集（三者同一定义）。

**C3.（T4）** locked 资产 joint 覆盖不足时 fail loudly（错误文本含 innovation coverage failure 语义）；回测驱动器捕获该错误、记录诊断、当日维持持仓（无新权重、不拼残差、不 zero-fill）。

**依据**：locked 资产的 scenario wealth 贡献（base_s）必须有合法 joint innovation，覆盖不足时拼残差/零填充均被 D-054/D-056 禁止；fail-loudly 是唯一诚实选项（与 D-067 同构）。

**C4.（T7）** d quasi-posterior 的数值 quadrature 节点留实现阶段定稿，必须走 D-066 流程（synthetic 解析对照、tolerance 减半收敛、与 Sharpe/PnL 无关）。

（T5/T6 由裁决 D 吸收：T5 kernel 归一采纳 VI 的部分升级条款（保持与现有 frac_weights 同族归一）；T6 接口锚对齐见裁决 D。）

## 裁决 D｜接口锚统一（吸收 Alpha-B1/B2 与 M2/M4/M5）

**D1.** 两文档逐字一致的接口锚文字：「ε 为统一 mode 残差：源对象是 asset 空间 OOF 残差行 ε̃_s = E_active·[y_s − (b₀^{(−fold(s))} + G^{(−fold(s))}·x_{s−1})]（含 b₀ 扣除，ŷ = E[y_s|train^{(−fold)}] 为 D-041 期望语义）；innovation 层按决策日 risk 域 R_t 消费其 R 子向量经 E_{R_t} 的 mode 变换。」

**D2.** 行 mask 契约（两文档逐字一致）：O_s = {j : r[u_s, j] finite}（ResidualOracle 行身份语义，ts_total 与目标日 u_s）。

**D3.** GATE0_RESPONSE_POSTERIOR.md 的全部旧 INNOVATION_LAW.md 引用改为 GATE0_VECTOR_INNOVATION.md，且语义同步（行身份引用 VI §1.2 契约，不得只换文档名不换语义）。

**D4.** response 文档的记号区分：working likelihood 噪声（分布对象）与 OOF 残差（实现值）使用可区分记号或显式语境标注，防止实现者把 Σ_R 抽样当 innovation（D-060 红线）。

## 裁决 E｜ragged 归一语义（M1）

**E1.** m_s 按行级观察集 O_s 归一（1/√|O_s|，SPEC §12 现状）；决策日 mode 基 E_{R_t} 用 R_t 内等权；E_active 用 active 集等权。行级 field 定义与域基定义是两个概念，两文档引用同一本裁决。

## 裁决 F｜RNG/抽样契约（M3）

**F1.** μ 通道 scenario 用 matrix-t predictive 后验抽样（消耗独立随机流位置）；(α₀, α_p) 的 quadrature 与 d quasi-posterior 的构造是确定性计算（不消耗 RNG）；d 的 scenario 抽样、行抽样各占独立流位置。SPEC §36 确定性契约（同 seed 同输入可重放）不变。若未来 μ 改数值积分矩（不抽样），须修订本契约并记录。

## 裁决 G｜施工图更新指令（吸收 Beta-B1'/B2'/B3' 与 N 系）

**G1.** 开放决策清单并入全部推导级待裁项；本次裁决已定夺 T1-T6 与 response §12 第 1 项（A4 先验修正）；response §12 其余 6 项（path group 划分、ragged 权重细节、quadrature 审计字段、RNG 接线、D3 措辞收敛、ε_quad 定稿）按推导文档的默认建议执行并在施工图标注「实现期可复审」。

**G2.** InnovationState 类型按 R 域重设计：V_t 为 N_R 维、承载 R_t/E_{R_t}/J_t 行集判定、z_pool 表达 per-d 结构（或 d-conditioned accessor）、propriety 断言（|J_t|≥1、|L|≥1）入字段或函数层。

**G3.** Step 10/11 前置依赖补 Step 1（Eligibility）与 Step 2（locked 语义）。

**G4.** 矩阵行更新：D-035/D-036 行加注「D-035a 修正已裁决，Step 8 应绿」；D-045 行加 R 域注记；D-055 行加单一行集概念注记；D-056 行加 T4 判定次序注记。

**G5.** Step 8 测试补：A2 数据层 fixture（n<N/退化 Y）、S(α) 双构造一致性断言（推导明示高危陷阱：S(α) 不是 ridge 残差平方和）、matrix-t vs dense 小系统数值积分对照、μ t 边缘矩 vs dense 对照、quadrature 同 seed 重放、预算耗尽统一错误文本。

**G6.** Step 10/11 测试补：shape 语义类断言（M_z 诊断报告、tail/skew 保留、cross-mode shock 分量间经验相关 = 历史相关）、|J_t|=0 与 rank deficient fixture、空观察行不进统计、stitching 拼接向量反例（不在任何 pool）、未来行注入构造性反例、N_R=1 标量对照（kernel 部分一致 vs 锚定部分已删除必须分离报告）、innovation 层 dummy 资产 V_t 逐字节不变。

**G7.** ResponsePosterior 类型补：每节点 evidence 值 p(Y|α⁽ᵏ⁾)、t 分布自由度 ν = n+1−N 的精确绑定。

**G8.** 施工图证据基准更新为八份文档（六份分析 + 两份 Gate-0 推导）；Step 7 文档名修正为 GATE0_RESPONSE_POSTERIOR.md；README 状态声明补 D-042 顶层名称（modular posterior predictive——旧线不得称 fully Bayesian generative，新线目标名称）；verify_release.jl 遗留问题登记（已 defer，Wave 2 后 DevOps 受控查验）。

## 裁决 H｜Gate0Planner 九项开放决策

**H1.** 模块形态：独立 module KTraderGate0，src/gate0/ 目录线；旧 src/ 冻结为 historical fixture 不动；KTrader 主 module 与新线互不 include；新线测试独立入口。

**依据**：D-082（先写新 slow reference）+ D-046（旧 snapshot 保留）。

**H2.** 文件切分：按施工图建议。

**H3.** 路线乙（新建独立线、旧线冻结）：确认。

**H4.** 252 口径：累计第 252 个有效 bar 当日解除（当日 E^trade=1）。「直到累计 252 个有效 bar」读作第 252 个 bar 当日即满足。

**H5.** Step 7 推导文档名：GATE0_RESPONSE_POSTERIOR.md（现状即正确）。

**H6.** D-084 旧 conditioned solver：不复制；旧线冻结即保留，新线不 include。

**H7.** 几何原语复用：纯函数（ruler、relative_gauge、principal_sqrt_root 等）复制到 src/gate0/ 并标注来源；不跨 module 依赖（旧线冻结是快照不是活跃 owner，复制不构成双 owner）。

**H8.** buy-and-hold benchmark 首日：首个决策日等权组合（候选集 = E^trade ∧ T^exec），之后不再 rebalance（D-072/D-073）。

**H9.** 501 天后两年重跑放行条件：Gate-0 Exit 十八项全过 + D-088 清单完成。

## 裁决 I｜生效与边界

**I1.** 本裁决自落盘起生效，优先级：AGENTS.md 裁决书 > 本裁决记录 > 三份 Gate-0 文档修订版 > 旧 docs/ 分析文档。

**I2.** 本裁决不改变任何 runtime 行为；Wave 2 编码（Step 1 起）在本裁决与三份文档修订落盘后方可启动。

**I3.** 三份文档的修订（按裁决 A-H）由原 Engineer 并行执行；修订完成后 Wave 2 开工。

---

*（代录尾注：本文件为本次代录唯一新建的文件；落盘过程未修改任何其他文件、未运行任何命令。裁决条目编号与原文一致；「依据」「可推翻条件」字段均取自裁决文本原文。）*
