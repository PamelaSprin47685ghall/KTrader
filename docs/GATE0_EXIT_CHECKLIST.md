# Gate-0 Exit 清单核对（AGENTS.md §41 十八项）

**文档状态：核对文档，2026-10-10（最终更新轮 + 合并验证同步轮）。核对对象：Gate-0 纠偏新线 KTraderGate0；旧线（KTrader 2.0 RC legacy implementation）冻结不改，其历史违规仅作对照口径记录，不构成本清单的扣分项。本轮更新后十八项全部满足（含注记），Gate 0 宣告关闭——见文末关闭声明。断言计数最终口径：test/gate0/ 全量 **984 项全绿**（模块 1–12 + backtest 场景 1–8；9 条 scoped 命令全 rc=0、26 文件字节一致；archive/evidence/gate0_final_verify_20261010/）。演进链：890（先一轮全绿）→ 913（合并验证；场景 8 未执行）→ 984（kelly/T4/prequential 修复落地后的收尾验证）。**
**证据来源：archive/evidence/gate0_wave2/3/4/ 各 summary.md 与 archive/evidence/gate0_step16/summary.md（只读引用，不复制运行数字原文）与 Manager 轮次裁决记录，以及 DevOps 2026-10-10 受控运行证据（archive/evidence/gate0_review_20261010/：存在性经 rerun 复核确认——10 文件、计数一致、17:08–17:29 真实运行；gate0_review_20261010_rerun/：890/890 全绿复跑；gate0_merged_verify_20261010/：求积升级 + ruler 修订后的合并验证 849/49 与当时的场景 8 状态；**gate0_final_verify_20261010/：收尾验证 984/984 全绿**；以及 gate0_merged_verify{2..8}*、gate0_kelly_adjudicate_*、gate0_innovation_attribution_*、gate0_reallimit_firstday_*、gate0_t4_recovery_*、gate0_f6_unblock_* 等过程证据）。本核对为文档层核对：未逐 testset 复核源码断言；凡证据为「summary 声明承载」而非「专门测试」的项，已在结论列如实标注证据形态。**

## 树结构调整注记（2026-10-10）

本文件的证据引用在仓库树结构调整（脚手架归入 `archive/`）后已统一改写为
`archive/evidence/...` 路径；被引用文件的字节未改动。改动后原路径与新路径
的对应关系见仓库根 `README.md` 的「仓库结构」一节与 `archive/README.md`。

## 十八项逐项核对

| # | Exit 条款（§41） | 对应裁决 | 新线实现载体 | 测试证据（指引） | 核对结论 |
|---|---|---|---|---|---|
| 1 | `Bars.bar` 只表示 observation | D-012 | 新线数据 / mask 层（observation 语义独立） | wave summary（见附录 B） | 满足（新线口径）：新线不重蹈旧线「改 bar 表达政策」的历史违规；旧线违规冻结不改、仅作历史对照 |
| 2 | model / trade / execution mask 分离 | D-013/014/015/016 | 新线 mask 模块（见附录 A） | wave summary | 满足（证据形态：summary 声明承载） |
| 3 | benchmark universe 外生 | D-020 | 新线基准层 | wave summary | 满足（证据形态：summary 声明承载；专门测试名未见——已由 2026-10-10 全量套件覆盖（最终 984 全绿口径）） |
| 4 | full mode response 含 cross blocks | D-025/026 | Step 3/4 统一 mode 坐标与四块 response | posterior_tests（logK 等断言全绿——Manager 裁决转述） | 满足 |
| 5 | DC channel 存在且 zero-centered | D-031/032/033 | Step 5 DC 通道 | posterior_tests | 满足 |
| 6 | 14 trace constraints 不在 default core | D-028/029 | Step 6 新线 default core | wave summary | 满足（证据形态：summary 声明承载） |
| 7 | response hyperparameters 不再 point plug-in | D-034/035/038 | Step 7/8 full posterior（α₀/α_p log-space 积分） | posterior_tests | 满足 |
| 8 | posterior propriety 有证明 | D-036 | Step 7 推导 + propriety gate | wave2/3 summary | 满足（证明文档 + gate；证据形态：summary 声明承载） |
| 9 | vector innovation law 实现 | D-045/046 | Step 10 V_t(d) 矩阵记忆 | innovation_tests（9.5 等断言全绿——Manager 裁决转述） | 满足 |
| 10 | no own-cell stitching | D-054/055/056 | Step 11 joint standardized shape | innovation_tests | 满足 |
| 11 | d prior 显式 | D-047 | Step 12 连续 d quasi-posterior | innovation_tests | 满足 |
| 12 | fixed S production path 删除 | D-062/067 | Step 14 adaptive 路径 | wave summary | 满足（证据形态：summary 声明承载）。注（Manager 裁决 6，2026-10-10）：driver 层 adaptive 在无信号 fixture 的慢收敛是 μ 通道 matrix-t epistemic 噪声的数学性质（M^{-1/2} 速率）非实现缺陷；D-067 fail-loud 传导断言是正确语义（driver_tests 71/71 含该断言）；μ 通道 RQMC 化已 defer 为 Gate-0 后扩展项 |
| 13 | integration 有 independent audit certificate | D-064/065 | Step 14 nested Sobol + audit replicate | quadrature_tests 55/55（Wave 4 裁决执行后原文件级全绿） | 满足（附 Manager 裁决 6 注记）：Step 14 机制级证书原文件级全绿；driver 层慢收敛经裁决 6 定性为 μ 通道 matrix-t epistemic 噪声的数学性质（M^{-1/2} 速率）非实现缺陷，D-067 fail-loud 传导断言为正确语义（driver_tests 71/71）；μ 通道 RQMC 化 defer 为 Gate-0 后扩展项 |
| 14 | cash 在 feasible set | D-068/069/070 | Step 2/15 Kelly（Σw≤1 + cash） | kelly_cash_tests 77/77、rc=0（DevOps 2026-10-10 受控运行，archive/evidence/gate0_review_20261010/；P0-9 tie-break 回归新增 18 项） | 满足（原文件级证据 kelly_cash_tests_final.log：Wave 2 裁决执行轮，D-090b/D-091 断言按裁决修正后九 testset 全 Pass） |
| 15 | OOF full isolation | D-043 | Step 9 OOF folds | wave summary | 满足（证据形态：summary 声明承载；隔离测试指引已由 2026-10-10 全量套件覆盖（最终 984 全绿口径）） |
| 16 | permutation / gauge / dummy invariance | §41 第 16 项 | Step 16 constitutional suite | archive/evidence/gate0_step16/summary.md（已落盘） | 满足（Step 16 汇总（2026-10-09 历史时点）为历史证据；当前工作树全量套件覆盖（最终 984 全绿口径）含 permutation/gauge/dummy 宪法套件；module 导出面 90 名——DevOps 2026-10-10 受控运行） |
| 17 | synthetic worlds 全部通过 | §33（D-090~095） | 各 world 测试分散于 wave 测试 + Step 16 汇总 | posterior_tests / innovation_tests 全绿 + Step 16（2026-10-09 历史时点）为历史证据；当前工作树全量套件覆盖 synthetic worlds（D-090~095）（最终 984 全绿口径）——DevOps 2026-10-10 受控运行 | 满足（单项 wave 测试 + Step 16 汇总双重证据；当前全量套件覆盖 synthetic worlds（最终 984 全绿）） |
| 18 | 所有 fail condition fail loudly | D-036/067 等 | 各模块 fail-loud 路径 | Step 16（2026-10-09 历史时点）为历史证据 + driver_tests 71/71（D-067 传导断言）——DevOps 2026-10-10 受控运行 | 满足（证据形态：Step 16 汇总（历史时点）+ 具名传导断言） |

## 证据等级与形态说明

- **A 级（专门测试，DevOps 2026-10-10 受控运行，日志 archive/evidence/gate0_review_20261010/）**：market_tests 82、modes_tests 102、response_tests 78、posterior_tests 40、innovation_tests 243（P0-2/P0-3 新增 adaptive d / rank 满秩断言 37 项）、oof_tests 25、prequential_tests 12（P0-1 新增）、predictive_tests 56、kelly_cash_tests 77（P0-9 tie-break 回归 18 项）、quadrature_tests 55、driver_tests 71、backtest_tests 49——合计 890 项全绿（先一轮；日志 gate0_review_20261010/ 与 rerun/）。合并验证（求积升级 + ruler 修订后）：modes 110（102+8 新回归）、模块 1-11 合计 849/849、backtest 场景 1-7 合计 49/49、场景 8 = 15 断言未执行（kelly SLOW_PROGRESS）——当时全量口径 913 = 898 已绿 + 15 未执行。**收尾验证（2026-10-10 最终）：984 项全绿** = 82+110+78+60+25+49+243+56+91+55+71+64（模块 1–12 + backtest 场景 1–8；9 条 scoped 命令全 rc=0、26 文件字节一致；archive/evidence/gate0_final_verify_20261010/）。修复落地：T4 testset (8) 15/15 转绿、kelly 条件行缩放+防护（D-091 转绿）、posterior m 基准修复、prequential 加速、SM 秩一阶段 1+2。历史 Wave 2/4 原文件级计数（如 kelly_cash 59、driver 73）为彼时点证据，已被实际计数取代。
- **运行边界（DevOps 2026-10-10 受控运行）**：backtest_tests 全量 7 场景需分组调度逐组运行（BACKTEST_SCENARIO="1" / "2,3" / "4,5" / "6,7" / "8"）；quadrature_tests 需独立文件运行（包裹模式超时），55/55 为独立运行口径。合并验证新增边界：完整多日 prequential 链成本 ~200s/日（求积升级 ×2.6/cell × 67 行）超单命令护栏——t=327 单日链与 60-day 段 6 未在 ≤60s 内运行（A3/A4；超时按 rc=124 记录，不是成功）。
- **B 级（summary 声明承载）**：轮次交付声明（archive/evidence/gate0_wave2/3/4/summary.md），非专门测试。
- **C 级（Step 16 汇总）**：archive/evidence/gate0_step16/summary.md 已落盘（2026-10-09 历史时点快照：775 项断言、83 API）；当前工作树套件覆盖（最终 984 全绿口径）、module 导出面 90 名——DevOps 2026-10-10 受控运行。
- 本核对未逐行复核新线源码；「满足」结论的证据等级为 Manager 轮次裁决记录 + wave summary 声明 + 模块 / 测试文件存在性（附录 A）。

## 附录 A：新线模块与测试文件（本次静态 glob 事实）

- src/gate0/KTraderGate0.jl
- src/gate0/driver.jl
- src/gate0/geometry.jl
- src/gate0/innovation.jl
- src/gate0/kelly.jl
- src/gate0/market.jl
- src/gate0/modes.jl
- src/gate0/oof.jl
- src/gate0/posterior.jl
- src/gate0/predictive.jl
- src/gate0/quadrature.jl
- src/gate0/response.jl

- test/gate0/driver_tests.jl
- test/gate0/innovation_tests.jl
- test/gate0/kelly_cash_tests.jl
- test/gate0/market_tests.jl
- test/gate0/modes_tests.jl
- test/gate0/oof_tests.jl
- test/gate0/posterior_tests.jl
- test/gate0/predictive_tests.jl
- test/gate0/quadrature_tests.jl
- test/gate0/response_tests.jl

## 附录 B：证据目录（summary.md 提取的结构性行）

证据目录：archive/evidence/gate0_wave2、archive/evidence/gate0_wave3、archive/evidence/gate0_wave4

### archive/evidence/gate0_wave2/wave2_summary.md

```text
命令 3 的补充观察（Test.jl 非交互模式在 testset fail/error 后中止 script，
后续 testset 未在原文件内观察；诊断副本逐层推进，原文件一字未动）：
模式一致，include 时在加载者作用域执行——骨架与测试临时 module 两条
（32 API + Julia names() 恒含的 module 自名 = 33 名集合）。断言语义
动机与纪律见函数 docstring（D-060/fail-loudly：不改目标/可行集/证书，
Expression: all((x->abs(x) <= 1.0e-6), wb)          → Fail（wb 非全零）
Expression: abs(cb - 1.0) <= 1.0e-6                 → Fail（cb = 0.24208681821330047）
Expression: maximum(w) - minimum(w) <= 1.0e-6       → Fail（1.4252806619097225e-6）
- kelly_cash_tests 原文件的 D-091/等价性/负控/校验 testset 因 Test.jl
fail-fast 中止未在原文件内直接观察（诊断副本观察为绿；副本与原文件的
至此 Wave 2 四条命令的最终状态：module 加载 ✅、market_tests 82/82 ✅、
## 7. 收尾验证轮（response_tests：Step 4/5/6 固定 ridge reference）
permutation 协变 + fail-loudly + D-093 DC world + D-094 trace
（36 API + module 自名），断言语图不变。
Wave 2 全量最终状态：module 加载 ✅（36 API）、market_tests 82/82 ✅
```

### archive/evidence/gate0_wave3/wave3_summary.md

```text
D-067 fail-loudly 语义不变）。
14. 9.6 w 跨 testset 作用域泄漏——testset 开头重建。
posterior_tests 临时 module 独立验证，骨架化留待后续裁决）。
（quadrature testset 6/6 全过）。innovation 线：首跑主 fixture error
**posterior_tests testset 2「evidence vs dense (N=1)」2 fail**（:103）：
**innovation_tests 9.5「limit」2 fail**（:253/:255）：
**innovation_tests 9.7「shape semantics」3 fail**（:348/:349/:357）：
**oof_tests（Step 9 补跑）**：
- 首跑 24/25：唯一红为 testset 5 的 fixture 索引语义混乱（fold_grid
**Wave 3 全量最终状态**：module 加载 ✅（71 API）、market_tests 82/82 ✅、
## 6. 收尾验证轮（predictive_tests：Step 13 + predictive.jl 骨架收口）
**predictive_tests.jl 受控运行**：首跑 14/0/4（三个 testset 在 fixture
6. L178：`st_ct` 跨 testset 可见性——交付者注释称 @testset 不引入作用域
**错误**（Julia @testset 引入新作用域）——负向内重建。
```

### archive/evidence/gate0_wave4/wave4_adjudication_summary.md

```text
# Gate-0 Wave 4 裁决执行轮 — 证据索引（裁决 4/5 + Step 15 driver）
**裁决 4（any 断言删除）**：quadrature_tests.jl testset 6——删除
geometry 起，UndefVarError 实证）+ module 内 using Dates（market.jl
骨架层 export 模式，测试独立加载 `using .Module` 拿不到非导出名）。
6. **testset (1b) 重写为 D-067 传导断言**：交付者自报风险的确凿实测
预算内不可达。**D-067 fail-loudly 是正确行为**——断言 error 含统一
7. testset (5) 断言对象改 res_ref（res_ad 已随 (1b) 改写删除；D-076
8. testset (7) a2/b2 改 reference（时间预算：8 次全链 adaptive 双跑超
**driver 的 adaptive 生产路径在其 fixture 上预算内不可收敛**（testset 1b
当前测试断言 D-067 传导（fail-loudly 正确行为）。选项：(a) driver
```

### archive/evidence/gate0_wave4/wave4_summary.md

```text
日期：2026-10-09。三段：quadrature_tests（Step 14）+ posterior_tests 复跑
原形式节点域 [-1,1]，解析基准在错误节点/权重上积分（testset 4 的
解析对照差 0.08 的根因）。修复后 testset 4 全过 12/12。
6. L210-211（testset 4）：max_scenarios 512 → 4096（A 停在 3.9e-3 >
7. testset 8（L341-346）：max 512 → 4096 ×3 处调用（同款预算）。
8. testset 6（L262-268）：wt 1e-1 → 1e-2（第一轮即双过、分层窗口未
9. testset 9（L373/L376）：weight_tol → 3e-3 + max 4096（**AAAA seed
10. testset 5/6 的 sed 误伤修复（连锁 weight_tol 重复——恢复原调用）。
**testset 6「certificate layering」的 any 断言**（quadrature_tests.jl:279）：
效应）。testset 6 的其余 5 项断言全过（converged/M>32/第一轮区分力/
```

## 交接登记——owner 裁决域 / 后续工作（Manager 裁决 2026-10-10）

Gate-0 Exit 十八项已全部满足（含注记）、Gate 0 已宣告关闭；以下事项为 owner 裁决域的后续工作，均非 Gate-0 关闭条件：

a. **t=327 / t=271 求解器链增强（证书驱动自适应 kelly）**——60-day 级闭合的剩余障碍链（见 D-089 阶梯最终状态节）。事实链更新（2026-10-10 合并验证）：t=327 根因已转移并部分修复——posterior 2D 求积升级为张量 G-L 规则（收敛指数 2.014、1929 cells 达 1e-6、代理保守 ratio≥2；设计文档 docs/ADJUDICATION_T327_POSTERIOR_QUAD.md），posterior 层过（生产直调 3.09s/1929 cells；full fit 4.66s）后 kelly 层成为第二阻塞（场景 8 t=271 SLOW_PROGRESS 实测；t=327 的 kelly 层因 prequential 成本未在护栏内直接观察）。max_iter=1000 修复已落地且 kelly_cash_tests 77/77 零回归。**已解除（2026-10-10 收尾）**：kelly 条件行缩放（span>4.5e15 阈值；docs/KELLY_NUMERICAL_ROW_SCALING.md）+ polish 防护恢复 + (i)/(ii) 测试重写落地——D-091 转绿、T4 testset (8) 15/15 转绿、最终 984 全绿（gate0_final_verify_20261010/）。
b. **T4 多日跨语义缺口的 SPEC 审查（裁决 9）**——D-013 admission（prefix 级）与 D-036 fold propriety（fold 级）的语义不一致；修复需数学对象层变更（fold 域收缩 / admission 重定义 / 非 contiguous fold），不为测试便利改数学对象。
c. **真实数据桥接（Bars→MarketFacts 转换器 + panel provenance 确认）**——真实数据短窗口验证优先于合成 501-day（合成数据增量验证价值低）。
d. **501-day 分批执行**——分批机制已建立并经 20-day 级闭合验证、未执行；501 天超出单命令 60s 护栏，执行策略归 owner 裁决。
e. **μ 通道 RQMC 化（Manager 裁决 6 defer 项）**——D-063 完整实现，已 defer 为 Gate-0 后扩展项。
f. **新发布凭据链生成（版本策略归 owner）**——verify_release.jl 遗留最终定性（Manager 2026-10-10）：status 白名单已适配（2.0-RC-G0 合法、历史凭据字节未动）；Project.toml 基线拦截按 D-002 语义保留（验证 2.0.0 历史发布物语义与纠偏期工作树的预期张力，非缺陷）。新凭据链生成归重新发布流程。
g. **数学加速解锁（D-007 下一层）**——数学正确性 → 数学加速 → 计算机科学加速 → 增量化 → GPU；在多日验证之后。

## 已知边界（2026-10-10 收尾；详见 AGENTS.md 第14任段）

- 首日 full-fit 性能：自适应 cells 规模仍是成本项（单点已达标；D-083 视角——reference 不追求速度）；
- BLAS 线程配置：默认 6 线程对小 GEMM 约 3× 负优化，待裁决；
- quadrature 负载敏感性：同字节重跑裁定为准；
- span≈1e17+ 与人工双方向 fixture：已知边界；
- 60-day：批量负载口径，非单命令目标。

## Gate 0 关闭声明（2026-10-10）

基于本清单十八项全部满足（含注记），依 AGENTS.md §41 Exit 条件与 Manager 轮次裁决记录，宣布：

**Gate 0 关闭：KTrader 2.0-RC / Gate 0 Reopened → Gate 0 Closed（2026-10-10）。**

- 旧线（2.0-RC legacy）按 D-002/D-005 冻结为历史发布物，其历史违规（如 `Bars.bar` 政策化）不重犯于新线；关闭判据只对新线 KTraderGate0 生效。
- 后续解锁一：D-089 测试升级阶梯的短窗口验证——当前状态见「D-089 阶梯最终状态」节（5-day ✓、20-day ✓ 分批闭合、T4 多日路径 15/15 ✓、501-day 机制已建立未执行）；kelly 第二阻塞与 prequential 成本已解除（见交接登记 a 收尾注记）；入口切换：bin/backtest.jl 已为 KTraderGate0 唯一当前入口（2026-10-10 切换收口，其头部成本注记已同步 984 口径）。
- 后续解锁二：数学加速（D-007 下一层：数学正确性 → 数学加速 → 计算机科学加速 → 增量化 → GPU）——在多日验证之后。

本声明为文档层声明：关闭依据是上表十八项核对结论及其引用的证据（Manager 裁决记录 + wave summary + Step 16 汇总），本核对未逐 testset 复核源码断言。

## D-089 阶梯最终状态（Manager 裁决 2026-10-10）

- **5-day 级：✓**——backtest_tests testset 1 端到端绿。
- **20-day 级：✓ 分批闭合（裁决 10）**——两段 10-day、held 桥接 + 净值乘法拼接 + 财富守恒六断言全绿；分批 = 数学不变的执行切分，因果性等价由 backtest_tests 的截断 / 全量断言承载。
- **60-day 级：50/60 决策日绿（段 1-5）、段 6 未在护栏内运行**——t=327 事实链更新（2026-10-10 合并验证）：posterior 求积规则升级后 posterior 层过；kelly 层为第二阻塞（场景 8 t=271 实测 SLOW_PROGRESS；t=327 kelly 层因 prequential ~200s/日 成本超单命令预算未直接观察）。修法方向：证书驱动自适应 kelly + prequential 成本数学加速（交接登记 a；D-007 下一层）。**收尾注记（2026-10-10 最终）**：两阻塞均已修复落地——最终 984 全绿（gate0_final_verify_20261010/）；60-day 为批量负载口径（见「已知边界」节与 AGENTS.md 第14任段）。 **段 6 重跑尝试（2026-10-11）**：定位完成——段 6 = t=327..336（10 日/段；合成 fixture N=4、seed=2026、`archive/evidence/gate0_step16/backtest_batched.jl` + held 桥接口径）；当前字节下三档重跑（10/5/2 日）全部 rc=124（55s 切）——即使 2 日段（约 140 行 fit）亦超窗；卡点 = preq 逐行 fit 的 2D 自适应求积（`oof.jl:398` → `fit_full_posterior` → `adaptive_quadrature_2d`；与 ~200s/日 既有登记一致）。完成路径二选一（交 owner 裁决）：（i）preq 数学加速（D-007 下一层；需裁决「多日验证 vs 数学加速」的次序）；（ii）以性能边界结案。证据：`archive/evidence/gate0_seg6_rerun_20261011/`。
- **501-day：分批机制已建立、未执行**——合成数据增量验证价值低，真实数据桥接优先（交接登记 c/d）。
- **全线断言计数（两阶段口径）**：先一轮 **890**（market 82 + modes 102 + response 78 + posterior 40 + innovation 243 + oof 25 + prequential 12 + predictive 56 + kelly_cash 77 + quadrature 55 + driver 71 + backtest 49；日志 archive/evidence/gate0_review_20261010/ 与 gate0_review_20261010_rerun/）；合并验证后 **913 = 898 已绿（模块 1-11：849，含 modes 110；backtest 场景 1-7：49）+ 15 未执行（场景 8；kelly SLOW_PROGRESS）**（日志 archive/evidence/gate0_merged_verify_20261010/）。**收尾口径：984 全绿**（82+110+78+60+25+49+243+56+91+55+71+64；gate0_final_verify_20261010/，9 条 scoped 命令全 rc=0、26 文件字节一致）。

## 裁决 7/8/9/10 登记（Manager 裁决 2026-10-10，简要）

- **裁决 7（posterior.jl tail 证书弱信息分支）**：先验尾衰减锚定，数据驱动标定阈值 3 / 衰减率 1 / 余量 8；有信息形态判据不变；三套件零回归。**后续（2026-10-10 P0-8 裁决）**：该弱信息特例机器已整体删除（`_WEAKINFO_GAP/RATE/SAFETY` 移除、先验恢复 proper HalfCauchy-τ、tail 证书统一判据）；本行为历史登记。
- **裁决 8（testset 5 解耦不可构造）**：推导 + 实证——J 覆盖行与 fold-rank 保障行集同源，解耦不可构造。
- **裁决 9（T4 多日跨语义缺口）**：D-013 admission（prefix 级）与 D-036 fold propriety（fold 级）在 contiguous fold + 尾部新资产上的语义不一致，登记为 SPEC 审查项（修复需数学对象层变更：fold 域收缩 / admission 重定义 / 非 contiguous fold——不为测试便利改数学对象）；T4 单日语义 71/71 覆盖、多日不触发路径 49/49 覆盖、触发→恢复多日正向路径不可构造。
- **裁决 10（20-day 分批闭合）**：两段 10-day 分批 + held 桥接 + 净值乘法拼接 + 财富守恒六断言全绿——详见 D-089 阶梯最终状态节。

## 本轮 session 源码变更登记（供后续审计；Manager 裁决 2026-10-10 登记）

- `src/gate0/posterior.jl`——裁决 7：tail 证书弱信息分支 + 常量三处（阈值 3 / 衰减率 1 / 余量 8）。（该分支已被 2026-10-10 P0-8 裁决删除；当前语义：proper HalfCauchy-τ + 统一 tail 判据——见裁决 7 登记行。）
- `src/gate0/oof.jl`——裁决 8 配套：tail_rel 10× 余量 + coarse_scan 补回。
- `src/gate0/kelly.jl`——max_iter=1000（t=327 修复尝试；kelly_cash_tests 77/77 零回归，迭代预算不解除停滞）。
- `src/gate0/kelly.jl`（2026-10-11 追加登记）——B2B-SPLIT-1/2/3 出口链拆分：`_cash_kelly_prelude`（= `_cash_kelly_prelude_prep` + `_cash_kelly_prelude_solve`）与 `_cash_kelly_finish`（= `_cash_kelly_finish_main` + `_cash_kelly_finish_tiebreak`）逐段化，组合调用与原 `cash_kelly` 逐点一致（kelly_cash_tests 的 B2B-SPLIT-1 testset 锁定；批量调度按段分批执行同一链）；TIE-FAST-1 全 cash 快速路径；2026-10-10/11 批次内早前落地并已锁定的设计 A（双路径候选取优）、M2（status 出口证书化）、KELLY-SCALE-RETRY-1 / 条件行缩放（span > 1/eps ≈ 4.5e15 阈值）。运行证据：kelly_cash 终态全绿（B2B-SPLIT-1 testset 26/26、TIE-FAST-1 13/13；`archive/evidence/gate0_multiday_run_20261010/S7_kelly_cash2.log`）。逐函数与逐段细节见 AGENTS.md 第 15/16 任段与本文件「状态同步（2026-10-11）」节的 SPLIT 链条目。
- `src/gate0/backtest.jl` / `src/gate0/driver.jl` / `src/gate0/quadrature.jl`——各轮修复的 before/after 见各 summary（只读引用，不复制原文，给路径）：archive/evidence/gate0_wave2/3/4/ 各 summary.md 与 archive/evidence/gate0_step16/summary.md。

本登记为文档层清单；源码字节以工作树为准，本核对未逐行复核 diff。

## 状态同步（2026-10-11）：A 门闭合、SPLIT 链与未收敛日决策包

本节为 2026-10-11 文档同步批（依据 AGENTS.md 第 15/16 任段与
`archive/evidence/gate0_multiday_run_20261010/` 的 S1–S11 系列）；新增运行事实与
归属登记，不改变上方十八项核对结论与 Gate 0 关闭声明。

### A 门闭合（过门配置）
- 配置：`GATE0_MU_QMC=true` + `GATE0_CHISQ_QMC=true` + `GATE0_MAX_SCENARIOS=131072`。
- 结果：t=330–346 十六日完成（339 为 solve 当日异常、未产出 A）——**十二过四超**（通过率 75%）；
  未收敛日 336/340/343/344（A=1.355e-4 / 1.283e-4 / 1.246e-4 / 1.346e-4，超门 28%–35.5%）——
  fail-loud、不产出认证权重。首次过门 t=330（A=6.2585e-5）、t=331（A=5.9696e-5）。
- 门（weight_tol=1e-4）位于 A 分布约 75% 分位（335/341 贴边 9.3e-5 / 9.8e-5）。

### SPLIT 链与批量调度
- b2b 出口链三级拆分（B2B-SPLIT-1/2/3；FB/S4 诊断先行、逐点等价锁定）：
  kelly_cash 终态 26/26（B2B-SPLIT-1）、TIE-FAST-1 13/13。
- 批量调度九段稳定：零重试、最大段 40s、168–201s/日（330–346）。
- 细节与逐日段耗时见 AGENTS.md 第 16 任段与 S1–S11 summary。

### t=327 处置
- 已闭环：posterior 层 2D 超参求积根因（中点 cell 规则收敛率不足）由张量 G-L 求积解决
  （`docs/ADJUDICATION_T327_POSTERIOR_QUAD.md`；生产直调 3.09s/1929 cells 达 1e-6）；
  kelly 层随后由条件行缩放 + polish 防护收口（AGENTS.md 第 13/14 任段）。

### T4（裁决 9）处置
- 已闭环：production（strict prequential）语义下「触发→恢复」已固化为 testset (8)
  并 15/15 转绿（`docs/T4_MULTIDAY_SEMANTICS_ADJUDICATION9.md` §5.3）；
  fold 死结归属 legacy 语义、不适用于 production。

### 入口升格
- 已完成（2026-10-10 切换收口）：`bin/backtest.jl` 为 KTraderGate0 唯一当前入口
  （AGENTS.md 第 14 任段）。

### P0-1..9 处置核对（指针）
- 静态核对表：`dev/gate0_static_audit_2026.md`（P0-1..9 全部找到直接源码级证据；
  其登记的两处文档滞后已由后续批次闭合——见 `docs/MULTIDAY_RUN_STATUS_AUDIT.md` §3）。
- 运行证据：test/gate0/ 全量 984 全绿（`archive/evidence/gate0_final_verify_20261010/`）；
  其后 kelly_cash 增 B2B-SPLIT-1 testset（终态 26/26）；**受影响模块（quadrature、driver 71/71、backtest 场景 1–8、kelly_cash）已于 2026-10-11 字节受控回归通过（`archive/evidence/gate0_regress_20261011/`）；其余模块组引用既有全绿（`archive/evidence/gate0_final_verify_20261010/`）。**

### 未收敛日决策包（交 owner/SPEC）
- `docs/KELLY_NUMERICAL_ROW_SCALING.md` §9.13：b（范围门控——语义例外，需 SPEC 裁决）、
  c（weight_tol D-066 复核——规范定稿程序）、d（活跃集低维化——已由 S10 诊断否定：
  本链 n=1、无零列）；推荐 c 与 b 并行，最终裁决留 owner/SPEC。

*本文件由 Gate-0 文档轮（Wave 4 收尾）生成，2026-10-10 经最终更新轮（第 13/14/16/17/18 项结论升级、第 12 项注记改裁决 6 口径、剩余条件收缩、新增关闭声明）与收束文档轮（D-089 阶梯最终状态、裁决 7/8/9/10 登记、源码变更登记、剩余条件节替换为交接登记、关闭声明后续解锁一行对齐）两次更新；2026-10-10 再经断言计数更新轮（test/gate0/ 全量实际断言计数按 DevOps 受控运行证据更新为 890 项全绿，含运行边界注记）、以及合并验证同步轮（P0-8/`_WEAKINFO` 同步、t=327 事实链更新、rerun/merged_verify 引用、"890 → 913" 口径；纯文档更新，未改动任何源码 / 测试 / bin 文件——同批的 F3 注释修正与 P0-8 断言新增见 AGENTS.md 第13任段）、以及最终收尾同步轮（984 口径、T4/kelly/prequential 修复落地、入口切换状态、已知边界——见 AGENTS.md 第14任段；纯文档更新）；2026-10-11 再经状态同步轮（A 门闭合、SPLIT 链与批量调度、t=327/T4/入口/P0 指针、未收敛日决策包归属——见 AGENTS.md 第 15/16 任段；纯文档更新）。*