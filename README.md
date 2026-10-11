# KTrader 2.0.0 Final — CPU

> **Gate-0 状态声明（2026-10-09）**
>
> 2.0.0 作为历史发布物保留，原字节、原 hash、原时间线不变（裁决书 D-002）；
> 但其「数学闭合」声明已被 Gate-0 审计撤销，不再是当前推荐规范（D-003）。
> 当前开发状态为 **KTrader 2.0-RC / Gate 0 Reopened**（版本线记号 2.0-RC-G0，
> 见裁决书 D-097 与 §57）；本仓库当前实现按裁决书推荐名称称为
> KTrader 2.0 RC legacy implementation。
>
> 规范依据：[AGENTS.md](AGENTS.md) 的 Gate-0 裁决书；逐项分析见 docs/ 下六份文档：
> MODEL_LEDGER.md、OLD_TO_CURRENT_SEMANTIC_DIFF.md、
> TRACE_NEUTRALITY_DERIVATION.md、POSTERIOR_DEFINITION.md、
> INNOVATION_LAW.md、NUMERICAL_INTEGRATION_SPEC.md。
>
> 在 Gate 0 关闭前，本 README 其余以发布时点口径写成的表述（含「正式版本」
> 等）一律按历史时间线理解；裁决书 D-096 的禁用名称清单同时生效。

> **Gate-0 纠偏新线名称与现状（2026-10-10，裁决书 D-042）**：纠偏新线 KTraderGate0
> 已落地十一个模块、Step 1-15；测试证据目录为 archive/evidence/gate0_wave2/3/4/ 各
> summary.md。新线的正式名称是「modular posterior predictive」——即 Bayesian
> response posterior + cross-fitted semiparametric innovation predictive。在
> innovation 仍使用经验 / quasi-likelihood 结构期间，禁止称 fully Bayesian
> generative posterior predictive（D-042）。Gate-0 Exit 十八项清单的逐项核对见
> [docs/GATE0_EXIT_CHECKLIST.md](docs/GATE0_EXIT_CHECKLIST.md)。
>
> **当前入口切换（2026-10-10，Gate-0 验收结论第 7 步）**：KTraderGate0 升格为
> 唯一当前入口——`bin/backtest.jl` 是当前生产回测入口，参数面统一为
> `GATE0_*`（不保留旧线回测参数）。旧 2.0 线（`src/KTrader`）冻结为历史
> release：其对比/报告/执行工具（`bin/bench.jl`、`bin/report.jl`、
> `bin/live.jl`）保留为历史线工具，不再是当前入口。Gate-0 是 slow
> reference：单线程、无 incremental；多日窗口是批量负载，默认末 20 天，
> 60/501 天阶梯归 D-088/D-089 分批调度；kelly 收敛鲁棒性缺口已修复
> （2026-10-10：条件行缩放 + polish 防护），最终受控验证 984/984 全绿
> （archive/evidence/gate0_final_verify_20261010/）；其后运行状态见下段。

> **Gate-0 现状与 A 门闭合（2026-10-11）**：Gate 0 已于 2026-10-10 宣告关闭
> （十八项核对见 [docs/GATE0_EXIT_CHECKLIST.md](docs/GATE0_EXIT_CHECKLIST.md)；
> 全量测试 984/984 全绿，archive/evidence/gate0_final_verify_20261010/）。
> 当前线 = KTraderGate0 + 过门配置：`GATE0_MU_QMC=true`、
> `GATE0_CHISQ_QMC=true`、`GATE0_MAX_SCENARIOS=131072`。该配置下
> t=330–346 十六日十二过四超（未收敛日 336/340/343/344，fail-loud 不产出
> 认证权重；通过率 75%，未收敛日约 25%）。未收敛日处理决策包（范围门控 /
> weight_tol D-066 复核 / 活跃集低维化）见
> [docs/KELLY_NUMERICAL_ROW_SCALING.md](docs/KELLY_NUMERICAL_ROW_SCALING.md)
> §9.13——归属 owner/SPEC。批量调度九段稳定；证据
> archive/evidence/gate0_multiday_run_20261010/（AGENTS.md 第 15/16 任段）。

价格历史 → 因果多尺度特征 → 独立 OOF 后验 → 收益场景 → 原始对数 Kelly → 执行。

## 仓库结构（2026-10-10 树结构调整后）

| 目录 | 角色 |
|---|---|
| `src/` | `KTraderGate0`（`src/gate0/`，当前 Gate-0 线，由唯一当前入口 `bin/backtest.jl` 驱动）+ `KTrader`（2.0-RC 历史 release 线，冻结） |
| `test/` | 标准测试入口 `test/runtests.jl` + 契约注册表 + `test/fixtures/` |
| `bin/` | 当前入口：`backtest`（KTraderGate0）；历史线工具：`bench / report / live`；数据层：`fetch`；工具：`scoped_run.sh / verify_release`；`ceiling_probes.jl` 为受门禁的 dev 命名空间入口 |
| `docs/` | 规范与审计文档（六份分析 + Gate-0 四份） |
| `release/` | 发布打包与校验工具（`package_source.py` / `verify_tests.jl` / `smoke.jl`） |
| `dev/` | 开发侧活动资产：探针命名空间 `probes.jl`、M1 重放链路、验收驱动、GPU 原语，以及发布凭据链 `dev/evidence/{cpu20_acceptance, earlier_closure, final_2_0_0, release_freeze}_20261009/` |
| `archive/` | 已离场的轮次日志、一次性脚本、研究线与注记（被 git 忽略；见 [archive/README.md](archive/README.md)） |
| `data/` | 本地行情与账户数据（不随源码发布） |

入口清晰性规则：`src/` 不 include `dev/`（由 `test/architecture_contract_tests.jl`
的 AST 门禁强制）；发布/测试/工具链仍需消费的 dev 资产留在 `dev/`；
其余一律入 `archive/` 并退出 git 索引（历史与 tag `v2.0.0` 原样保留）。
归档文件取回方式：`git show 8d54f0d:<原路径>`。

2.0.0 是 CPU 正式版本。保留原 1.0 数学定义和已有 `*_v1` API；GPU 属于
[2.1](ROADMAP_2_1.md)，不是本版本依赖或发布门槛。
（时间线标注：本两行为 2.0.0 发布时点口径；其「正式版本」地位已被
Gate-0 裁决撤销，当前状态见文首声明。）

## 安装与版本校验

验证环境为 Julia 1.12.7、OpenBLAS、Linux。使用随包提供的 Manifest 锁定依赖。
以下安装命令会按需要下载依赖，不会下载行情或连接交易账户：

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. -e 'using KTrader; println(Base.pkgversion(KTrader))'
```

`using KTrader` 的版本号 `2.0.0` 属于历史 release 线包身份；当前 Gate-0
工作线经 `bin/backtest.jl` 与 `test/gate0/runtests.jl` 使用（见「使用」）。

`bin/verify_release.jl` 校验的是 2.0.0 历史发布物身份（acceptance /
qualification 凭据与基线工程字节）。在 Gate-0 工作树上它预期会在逐文件
hash 处拦截（工作树已含 `src/gate0/` 等新增/变更文件）——这是历史快照
校验器的预期行为，不是当前开发线的验收工具，也不把哈希校验冒充新的
数值测试。

## 使用（当前入口：Gate-0）

`data/close.csv`、`data/adj.csv` 使用 `date` 列加资产列；未观察价格用 `NaN`。
行情、账户信息和已拟合模型不随源码包发布。

当前生产回测入口是 `bin/backtest.jl`，驱动独立模块 `KTraderGate0`
（`src/gate0/`；两线互不 include，数据桥只经旧线数据层取裸矩阵）。默认
末 20 天短窗口、`GATE0_MODE=adaptive`（生产路径）、严格 posterior 口径。
参数面全部为 `GATE0_*`，完整清单见脚本头注释：

```sh
# 默认：末 20 天窗口，adaptive 生产路径
julia --startup-file=no --project=. bin/backtest.jl
# 最小 smoke（5 天窗口；t_start 必须 ≥ WARMUP+1 = 257，示例按数据调整）
GATE0_T_START=14001 GATE0_T_END=14006 julia --startup-file=no --project=. bin/backtest.jl
# reference 固定 S 对照（D-062 合法用途，非生产默认）
GATE0_MODE=reference GATE0_S_REFERENCE=64 julia --startup-file=no --project=. bin/backtest.jl
```

Gate-0 是 slow reference（D-082/D-083）：单线程、无 incremental、无缓存；
每个决策日重跑完整链条，因此**多日窗口是批量负载**。入口将 BLAS 设为单
线程（`BLAS.set_num_threads(1)`——gate0 负载实测最优配置：单点
8.36ms→2.52ms，见 archive/evidence/gate0_merged_verify8_20261010/）。
60/501 天阶梯与完整两年在单命令 60s 护栏之外，归 owner 分批调度
（D-088/D-089）；不得把短窗口计时外推为长回测耗时。Gate-0 测试线最终
受控验证 984/984 全绿（kelly 缺口已修复）；其后 kelly_cash 增 B2B-SPLIT-1
testset（26/26），受影响模块（quadrature/driver/backtest 1–8/kelly_cash）
已受控回归通过（archive/evidence/gate0_regress_20261011/）；其余组引用
既有全绿（archive/evidence/gate0_final_verify_20261010/）。过门配置多日批量
运行（t=330–346）见文首声明。

### 历史 release（2.0.0 / 2.0-RC 旧线）

旧线 API（历史 release 语义；旧测试与验收凭据仍冻结守护该线）：

```julia
using KTrader, Random, LinearAlgebra

BLAS.set_num_threads(6)
bars = load_bars("data")
prices = signal_prices(bars)   # 缺失交易日保持 NaN，不用前向填充价格推断
prepared = prepare_reference(prices; F_folds=3)
model = solve(prepared)
scenarios = generate_scenarios_v1(model; S=300, rng=MersenneTwister(1))
weights = KTrader.scenario_weights(
    scenarios, model.active_indices, bars.bar[end, :], nothing)
```

旧线工具入口：`bin/bench.jl`（引擎对比）、`bin/report.jl`（模型报告）、
`bin/live.jl`（执行）——均为历史线工具，不属于当前 Gate-0 入口；
`bin/fetch.jl` 为双线共用的数据层工具。历史发布物校验见
`bin/verify_release.jl`。

## 数学与实现边界

以下为 2.0-RC 历史线的边界描述（当前 Gate-0 线的主路径是
`MarketFacts/Eligibility → single_day_decision → run_gate0_backtest`，
见「使用」）。

`prepare_reference → PreparedProblem → solve` 是历史线主路径。增量准备
最终也进入同一个求解器；资源预算只选择准备路径，不改变结果。full 和
每个 OOF 折独立拟合。所有返回仍须原 alpha、协方差和 Kelly 证书通过，
失败即报错。

2.0 优化了拟合内几何和工作数组复用、按需梯度、同 alpha 证书计算、稀疏
观察掩码遍历、增量列拷贝和 Kelly 矩阵布局。没有缩历史、截秩、减折或
降精度。`*_v1` 名称描述数学/API 兼容性，不是残留的包版本号。

核心层次：`data / geometry / numerics / response / residual_oracle / prepare /
predict / incremental / kelly / backtest / broker / live`。生产模块不加载
`dev/` 中的诊断或 GPU 原型。执行入口需要账户授权，本次发布不会调用它们。

## 验证与已知限制

**当前 Gate-0 线**：测试入口为 `test/gate0/runtests.jl`（12 模块，按
`GATE0_MODULES` 分组调度；默认 backtest 场景 1 为 5-day end-to-end）。
最近一次全量受控运行报告 984/984 全绿（kelly 收敛鲁棒性缺口已修复；
archive/evidence/gate0_final_verify_20261010/）；其后 kelly_cash 增 B2B-SPLIT-1
testset（终态 26/26）；受影响模块（quadrature/driver/backtest 1–8/kelly_cash）已受控
回归通过（archive/evidence/gate0_regress_20261011/）；其余组引用既有全绿
（archive/evidence/gate0_final_verify_20261010/）。完整多日阶梯（60/501 天）
与全量 backtest 场景超出单命令 60s 护栏，需单独分批调度（D-088/D-089）；
过门配置多日批量（330–346）与未收敛日（约 25%）现状见文首声明。

**历史线（2.0-RC）**：已验收 CPU 源码的 41 个标准文件、22 个分组全部
通过；两个不重叠的 N65 八日窗口已验证。较早窗口的独立逐日模型和两次
进程输出比较，仓位 L1 和收益差异均为 0。详细证据与边界见
[CPU 2.0 交付](RELEASE_CPU_2_0.md)。

测试入口和有界执行示例：

```sh
julia --project=. test/runtests.jl
bash bin/scoped_run.sh 45 /tmp/ktrader-architecture.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/runtests.jl --architecture-only
```

历史完整覆盖采用分组运行，不是单进程全套；数值／数据／执行模拟组使用
`-O1` 控制测试编译成本，其余组与正式性能测量使用默认优化级别。局部验证
不保证任意历史都收敛，也不证明 CPU 已无优化空间。42 days/s 的性能目标
尚未达到。源码校验和测试不替代部署前对自有数据与执行环境的验证。

## 发布内容

[RELEASE.toml](RELEASE.toml) 固定 Final 版本与验收来源（时间线标注：此为
发布时点口径；status 已按 Gate-0 裁决改为 2.0-RC-G0，[acceptance] 历史
验收凭据按 D-002 原样保留；另记 `current_entry` 指向当前 Gate-0 入口）；
[CHANGELOG.md](CHANGELOG.md) 汇总变化；[ROADMAP_2_1.md](ROADMAP_2_1.md) 记录 GPU 范围。
源码归档包含完整测试和必要开发测试库，不包含行情、拟合快照、GPU 二进制
或逐轮开发历史。仓库中的原始证据保留，旧 README 位于
`dev/evidence/final_2_0_0_20261009/README.pre-final.md`。

本仓库未配置远端；此版本的交付是本地发布提交、`v2.0.0` 标签和源码归档，
不表示已上传 GitHub 或 Julia 注册表。许可范围不因本次封版而另行扩大。
