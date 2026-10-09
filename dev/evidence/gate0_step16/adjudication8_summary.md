# Gate-0 裁决 8 执行轮 — testset 5 fixture 重设计 + 20-day 评估

日期：2026-10-09。裁决 8（fixture 解耦）+ backtest 复跑 + 20-day。

## 1. 裁决 8 执行与实证结论

**静态可行性推导先行**（adjudication8_feasibility.md，源码行号核对）：
「J 覆盖行」与「fold train 的 rank 保障行」共用同一「资产观测行」
集合（row_ids = rows 目标日；O_u ⊇ R ⟺ 资产观测行；Syy rank 需
e_j 自由度 ⟺ 资产观测行）。**约束不等式**：T4 触发需资产 4 观测
行 ≤ 1；contiguous F_folds ≥ 2 的 propriety 需 ≥ F_folds ≥ 2（每
fold train 至少一行、分属不同 fold）——**矛盾**。

**实证检验**（裁决字面构造）：testset 5 fixture 重设计为裁决 8 的
「分散观测」形态（资产 4 观测行 = 目标日 258/262（fold 1/2 各一——
propriety 保障）+ 279 起恢复；NaN 化在 MarketFacts 构造前——不绕过
D-012 校验）。**实测（t4_branch_check.log）**：t=279/280/281 全部
mode=reference、R_universe=[1,2,3,4]——**J = {258, 262} = 2 ≥ 2 →
T4 不触发**——**推导成立：解耦在当前行集语义下不可构造**（实证
证据）。

**testset 5 的最终形态**（backtest_tests.jl L156-207 区域）：双分支
断言设计——mode ∈ (:held_maintained, :reference)；held_maintained
分支保留原 T4 断言全套（维持/漂移/恢复）；reference 分支断言全域
R（解耦不可构造的实证锚点）。**T4 多日跨语义的覆盖代价如实记录**：
该语义在多日 backtest 层暂无正向覆盖（单日 driver_tests 的 T4 仍
73/73 覆盖——held_maintained 的转译语义本身有验证）；多日 T4 跨日
的正向覆盖需要语义层变更（见 feasibility 文档的四个选项，处置权
在 Manager）。

## 2. 运行结果

| 命令 | log | rc | elapsed | RSS | 结果 |
|------|-----|----|---------|-----|------|
| backtest_tests | backtest_tests.log | **0** | 39s | 1439MiB | **49/49 全绿** |
| t4 分支确认（-e） | t4_branch_check.log | 0 | 27s | 1467MiB | mode=reference ×3（实证） |
| 20-day 合成运行 | backtest_20day.log | 124 | 40s | 1474MiB | **超时**（护栏边界——见下） |

**backtest_tests 49/49 全绿**（rc=0、39s/1439MiB、killed=0）——
testset 1-7 无回归；testset 5 的新形态全过（含 propriety——分散
观测行的 fold rank 保障生效）。

**20-day**（backtest_20day.jl：N=4、T=300、t=277..296、reference
S=32）：**rc=124 超时**（50s deadline 于 40s 处杀、killed=1、栈在
fold posterior 的 quadrature 细分）——20 决策日 × ~2.5s/日 + 编译
≈ 55-60s > 50s 护栏。**定性：护栏边界，非数学失败**（任务书预见
「40-60s 边缘」；单日成本与 backtest_tests 的 3 日/39s 观测一致
外推）。**60-day 不执行**（任务书：预期超 60s——护栏边界）。

**20-day 的未观察状态**：净值曲线/财富守恒/benchmark 的 20 日形态
未取得绿证据（超时截断）——D-089 阶梯的 20-day 级**未闭合**；
解除路径：(a) 分批执行（10+10 日两段——每段 <50s）；(b) 性能层
（posterior quadrature 的加速——Gate-1 数学加速的范畴）；(c) 护栏
放宽（owner 决策）。

## 3. 修复清单（file:line 最终状态）

**test/gate0/backtest_tests.jl**（testset 5，~L156-207）：
- before（IPO 形态）：IPO 资产于行 278——T4 触发 ✓ 但恢复日 fold
  propriety 亏（上轮红）。
- after（裁决 8 分散观测形态）：资产 4 观测 mask = 目标日 258/262
  （fold 1/2 各一）+ 279 起；NaN 化构造前（D-012）；双分支断言；
  注释钉死推导与实证结论（adjudication8_feasibility.md 引用）。

**无源码修改**（裁决 8 是 fixture 侧修复；posterior.jl/oof.jl 本轮
未动——上轮的弱信息分支与 10× tail_rel 保持）。

## 4. 边界

未修改：旧 src/、旧 test/、docs/、git。诊断/运行脚本（feasibility
文档、20-day 脚本、t4 分支检查）留证据目录。全部绿证据单机当前
时点一次运行事实。
