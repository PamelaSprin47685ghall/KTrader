# 2026-10-09：完整文件覆盖与两日生产基准

## 本轮交付

生产改动仅在 `src/backtest.jl` 和 `src/response.jl`。

回测将数值目标计算收敛到 `_backtest_target`，在读盘/Channel 的动态类型
边界之后按实际输入类型调用数学代码。固定场景与 adaptive 共用原来的
scenario/Kelly 函数、种子和计时；持仓账本、spool、调度和经济目标不变。
该边界的独立 19 项测试及 timeblock 原 604 项均通过；没有声称它单独造成
下文的冷启动或吞吐变化。

EB 证书现在复用本次拟合、同一 alpha 的 R/h/谱权重/Jacobian cores。
公开接口不增加缓存注入参数；内部缓存仅由其拟合 owner 传入，并拒绝不同
alpha/n。Sigma 重新投影、scalar evidence/导数、covariance certificate
和每项验收仍重算，不能用历史 valid 顶替当前判断。

首版严格字段测试确实红过：已有 cores 会让 M 从原 fresh-cache GEMM
改走 core-dot，产生最后几位差异。修复是提取并复用原
`_conditioned_constraint_direct`，证书显式保持原 M 的求和顺序，**不改
isequal 断言或任何容差**。修订后新增 456 项通过；旧红保留在
`qualification/response_b.log`，最终五文件 response_b 共 6091 项通过。

## 当前源码的标准文件覆盖

唯一最终快照：`qualification/final/qualification_snapshot.toml`，SHA256：

`b130327b5147a16ff1f1b325bd03f19f1a16ad9f05f904223fc0ef5bcf1301b0`

`dev/release_batch.jl` 将实际标准入口的文件清单与计划比对；覆盖 31 个
REQUIRED 文件、6 个标准 legacy 数值文件、1 个架构文件，共 38 个。
长文件按原有入口分相，所有 4 个增量相和 16 个进程护栏相均运行，没有复制
数学测试或删除断言。每组执行前后检查源码/测试/脚本/依赖清单的完整哈希，
只有运行正常完成且哈希不变才产生该快照下的 pass receipt。

最终同一 consumer：`status_complete.log` RC0，`groups=20/20`、
`standard_files=38`、`source_unchanged=true`。在尚缺最后两组时，
`status_missing.log` 实际 RC1 并明确列出 guard_c/guard_d，未先报全过。
老快照和老 receipt 不参与本次最终判定。

| 普通文件组 | 通过断言 |
|---|---:|
| architecture | 25 |
| constitution | 34 |
| numerics（数值、数据、执行模拟、Kelly） | 446 |
| response_a | 1576 |
| response_b | 6091 |
| eb（含冻结 N65 fold 的 BLAS1/实际 BLAS6 回归） | 160 |
| prepared | 1598 |
| posterior | 125 |
| residuals | 71 |
| timeblocks + numerical boundary | 623 |
| interfaces | 438 |
| reference isolation | 57 |

增量与进程护栏的断言计数在各 child 日志；不把重跑、历史快照或基准中的
断言重复加进上述覆盖。完整文件覆盖不等于单进程下的模块交互验证。

配置例外必须保留：numerics 组使用 `-O1` 控制测试编译成本；标准全入口的
import preamble 必须显式加载，因为 `numerical_tests.jl` 本身依赖它。
第一轮 runner 遗漏 Random 导致 UndefVarError，第二轮 preamble 遗漏 Dates
且随后超时，均是 harness 错误，日志保留，不归为模型缺陷。成功命令是：

```sh
timeout 55s bash bin/scoped_run.sh 45 <NEW_LOG> --rss-guard=2048 \
  env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=6 \
  julia -O1 --startup-file=no --compiled-modules=existing --project=. \
  -e 'using Test, Random, LinearAlgebra, Statistics, Dates, Convex, Clarabel; using KTrader; include("dev/release_batch.jl")' numerics
```

已成功的 group 不允许覆盖 receipt；上述是复现配置，不是要求现在原样重跑。
其余最终组使用默认优化级别。所有性能数据均默认优化级别，未拿 O1 吞吐
与 O2 比快。`test/runtests.jl` 未改，标准入口仍拒绝局部分相参数。

## 编译资源问题已分开处理

普通 `using KTrader` 首次预编译在 native image 生成阶段触及 RSS：
`precompile.log` RC124，peak2055MiB，guard2048，elapsed16s。这不是成功，
也不是“所有慢都是编译”的证据。

改用本进程 `JULIA_IMAGE_THREADS=1`、`JULIA_HEAP_SIZE_HINT=768M`，并保留
Julia1/BLAS1、预编译 task1、offline 与原 45s/55s/RSS2048 护栏后，
`precompile_bounded.log` RC0，elapsed35s，peak1657MiB。两个资源设置一起
改变，不单独归因给其中一个。heap hint 是 GC 提示，不是 RSS 硬限制；原
采样护栏依然启用。没有安装新依赖、改 Manifest 或提高资源限制。

这让同源码的 timeblock 组从此前超时变为最终 623/623、elapsed23s。
新缓存未被写进 repo 或模型；测试源码哈希没有因此改变。

## 本地完整两日生产回测

输入为当前本地 CSV，经生产 `load_bars -> signal_prices`，不是截短历史
或合成价格。N65，决策日 2026-09-30、2026-10-01，对应完整前缀
1:14308、1:14309；三折独立 EB、S300、seed1、engine=batch。
本地输入字节与消费语义已确认，不替上游供应商历史数据质量背书。

`dev/batch_window.jl` 先显式执行小合成 startup，再执行相同真实窗口 warmup，
最后一次 measured replay；每次 backtest 都重新开始账户与 alpha 链。
下面计入准备、后验、OOF、场景、spool I/O、Kelly 与持仓结算，全部正式
测量的 native compile/recompile 都是 0，未从耗时中减编译。

| date_tasks × BLAS线程 | 两日 wall 秒 | 两日窗口 days/s | 累计分配 GB | GC 秒 | 进程采样峰值 MiB |
|---|---:|---:|---:|---:|---:|
| 1 × 1 | 4.714015286 | 0.424266762 | 3.7914 | 0.898304283 | 1653 |
| 1 × 6 | 3.872951531 | 0.516402022 | 3.8487 | 0.897935970 | 1774 |
| 2 × 1 | 2.442362947 | 0.818879112 | 3.8199 | 0.394742254 | 1978 |

日志为 `window_b1.log`、`window_b6_compiled.log`、`window_p2.log`；
各配置相同窗口重放通过。跨配置每日期仓位 L1 最大值：1×1 对 2×1 为 0；
1×6 对二者为 1.3086346284605208e-8，小于原 1e-7 预算。不是跨机器或
任意线程的字节恒等保证。并发 worker bucket 可以重叠，不能相加冒充 wall。

每配置只有一个 measured 窗口；warmup 非第二个独立稳态样本。未证明长期
吞吐或线程最优，未改生产默认 topology。2×1 只余约70MiB护栏空间，不扩大
任务数。较早未编译的 `window_b6.log` 在真实 warmup 的 OOF 证书计算中
超时；日志保留，不能全归因 JIT，也不能当正式吞吐。

## 证书局部微基准

`certificate_cache_micro.log` 使用认证本地 N65/P910 充分统计和已存模型，
一遍谱分解、零 posterior fit。比较时 fit-owned geometry 两边已有，新增
路线还复用优化器已经计算的 alpha cache/cores；其一次性成本没有宣称为零。

| 顺序 | 中位秒 | 分配字节 | 编译秒 |
|---|---:|---:|---:|
| before A | 0.015733455 | 20883592 | 0 |
| after B | 0.005591820 | 10213304 | 0 |
| after B2 | 0.007510276 | 10213304 | 0 |
| before A2 | 0.012328683 | 20883592 | 0 |

证书所有字段精确相同，valid=true。只据此认定该调用的分配约减51%、观察
到局部 latency 收益，不将其百分比套到完整 backtest。

## 仍未完成的 2.0 范围

当前已完成同源码标准文件范围的分组验证，以及完整两日三配置生产比较。
尚未取得单进程全套运行、较长区间和跨样本稳定吞吐、更多 topology 的内存
成本结论或 GPU 设备试验。小窗口的 OOF 与累计分配仍是主要剩余工作；不能
据此宣称 CPU 做尽、GPU 无路或 mission 达标。没有改版本号或打 2.0 tag。

末次只读 `ps -C julia` 返回空进程表（rc1 为无匹配进程），Git 仍为有用户
未提交改动的 master。没有 commit/reset/clean。下列 final key hashes 是
末次实际观察；完整 inventory 仍以唯一 qualification snapshot 为准。
