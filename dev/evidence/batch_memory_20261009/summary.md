# 回测常驻缓存清理与八日验证（2026-10-09）

## 生产改动与数学边界

本轮生产文件只改 `src/backtest.jl`；response、geometry、predict、numerics、
incremental、Kelly 的哈希与进入本轮时一致。

此前回测总会构造 T×N×9 的 PrefixRulerStats。每次 batch 拟合消费它之前，
`verify_ruler_stats_prefix` 又完整扫描同一前缀，因此这里的缓存不节省该扫描。
现在回测调用原 `fit_v1` 的无 ruler-cache 分支，使用原 `ruler`；未引入
新数值函数、信任标记、绕过校验的通道或公共参数。公开调用显式传入缓存时，
所有原来源/计数/有限性校验仍执行。其独立回归在本轮实际通过。

两个分支的数学对象都是全部有效端点差平方的和、有效计数及原 power-law
拟合；不删历史、资产、频带、fold 或缺失掩码。直接 ruler 允许原 SIMD
求和，prefix builder 原为顺序累加，不能对任意数据承诺字节恒等。
本地 N65 前缀的 ruler、full/fold统计在本轮恰好逐值一致，后验/仓位由
完整四日旧输出对照另行验证，而不是用 Sharpe 相似替代数值验证。

增量回测也不再预建仅供初始化核验、之后不消费的可选 PriceHistoryCache；
其状态仍从同一完整 signal 自行构造。batch保留现有 PriceHistoryCache，
避免把每个日期的 log/return 数组重复分配。未改两个公共缓存接口。

复杂度：删除一次 O(TNH) 的全表构造和常驻存储；每个 batch 前缀的 ruler
仍为 O(TNH) 扫描（H=9）。N65/T14310 的原表 payload 为
T×N×H×(sizeof(Float64)+sizeof(Int32)) =100456200字节，约95.8MiB。
这不是整个回测变成零分配；每次准备的设计矩阵与其他数组仍分配。

## 原始局部测量

`backtest_cache_micro.jl` 读取认证本地CSV，经生产load_bars/signal_prices，
对照完整1:14309前缀，零posterior fit。首次版本重复准备时触及RSS护栏：
RC124，2116MiB；已打印17项比较通过，但**整条命令不是成功基准**。

修订测量每次仅保存时间/字节元数据，丢弃已完成PreparedProblem，并在各
原语计时外显式GC。这样不能用于自然GC吞吐，却能在资源范围内比较相同
原语。`cache_micro_bounded.log`：17/17、RC0、17s、1919MiB。

| 原语 | 中位耗时秒 | 每次分配字节 |
|---|---:|---:|
| cached prepare A | 0.090564381 | 163006208 |
| plain prepare B | 0.085380066 | 163006208 |
| plain prepare B2 | 0.082207349 | 163006208 |
| cached prepare A2 | 0.081521571 | 163006208 |

原表单独构造分配100462608字节、观测0.00982003秒；该成本和常驻表不在
逐次prepare计时内。两条prepare路径耗时重叠，不声称稳定提速。核心收益
是删掉常驻表，而不是减少每次prepare的163MB分配。
三种ruler及三个full统计最大差/相对Frobenius差均为0，三折统计也通过
原atol=1e-9/rtol=1e-11门槛。

## 完整生产窗口

`batch_cache_window.jl` 调用原 `backtest_v1`，N65/F3/S300/default EB/
seed1/batch，价格为原本地CSV。每一天都用完整历史，不把八日窗口当成
八行训练数据。先显式小合成API startup，再两日真实specialization
warmup，最后一次测量；所有正式调用compile/recompile均为0。
每次backtest重建账户与alpha链，未将warmup的答案当作当前结果。

| 窗口/配置，均默认GC | 正式wall秒 | 累计分配字节 | GC秒 | 整条命令RSS采样峰值MiB |
|---|---:|---:|---:|---:|
| 4日，1 task/BLAS6 | 8.072236463 | 6109389416 | 1.579947658 | 1611 |
| 4日，2 tasks/BLAS1 | 5.022984575 | 5829558632 | 0.682773710 | 2042 |
| 8日，1 task/BLAS6 | 15.767236451 | 12062085488 | 3.167018443 | 1684 |

四日为2026-09-28至2026-10-01；八日为2026-09-22至2026-10-01。
通过完整日期、场景数、持仓合法性和财富账本检查。四日还与前轮认证输出
`oof_memory_20261009/after_b6.jls` 对照：固定SHA256
3b7968ec5aec533e985f48dc88fb12cbac9085d3abcd591c19bddf3f0fc43f92，
且所有生产文件除了backtest之外必须相同。单任务仓位/收益差异均为0。
双任务与前轮单任务、与当前单任务的每日最大仓位L1为1.3086346284605208e-8，
收益最大差6.343758851556913e-11，通过原跨拓扑门槛。

前轮同四日serial正式调用7.886837966秒/6209850472累计字节，本轮8.072秒/
6109389416字节。分配正好少约100.46MB，耗时反而略大；不能据此宣称单任务
更快。旧脚本使用四日真实warmup、新脚本使用两日warmup，各命令的采样峰值
不构成严格配对的常驻内存差，不能把1842->1611MiB全归因于删缓存。

双任务默认GC这次成功，但只余6MiB护栏余量，未继续8日parallel。以前默认GC
失败、改heap hint后速度波动的事实不能被本次成功抹掉，也不把5.02秒当作
稳定上限。八日只有一个正式窗口，无旧八日基线、同配置重复或跨拓扑八日
对照，不宣称长区间稳定性已经证明。逐日OOF桶合计12.602218632秒，约占
八日wall的79.9%；GC约20.1%，两者重叠，不能相加。

## 当前源码回归：既有范围通过，新文件未运行

最终快照：qualification/qualification_snapshot.toml，SHA256
680f1494d2d435847af2b4972d8be440e9f4f519fb2af0889789e302dbabeeba。
它覆盖当前runtime/test/dev脚本/依赖清单；每组前后核验相同字节。

新增 `test/backtest_cache_tests.jl` 已登记REQUIRED，但其数值启动命令被工具
以无法确定安全状态为由拦截，未取得执行结果。未改wrapper或通过另一group
强行执行同一被拒操作。只做了Meta.parseall静态语法检查，记录在
syntax_only.log；这不执行该文件，也不充当数值测试通过证据。

其余39个既有标准文件在当前源码下重新验证，20组均成功，包括全部增量分相、
进程护栏、公开缓存守卫、PreparedProblem生命周期、EB难例、后验、残差、
执行模拟及回测调度。数值/数据/执行模拟组仍用-O1控制测试编译成本，其余
测试及所有性能测量为默认优化级别。未改单进程标准入口。

最终同一汇总消费者实际RC1，正确保持不完整：

```
MISSING cache_elision
COVERAGE groups=20/21 standard_files=40 source_unchanged=true
```

40=33 REQUIRED+6 legacy+1架构；其中新增1文件无执行记录。不能说当前40
文件全过，更不能用旧39文件的快照冒充这次40文件发布。各通过计数和child
日志都在qualification/，没有把历史重跑重复累计成新断言。

## 资源与交付范围

所有命令沿用45s内层/55s外层及2048MiB采样级RSS护栏。最大成功命令33s；
首次micro的RSS终止、precompile父进程超时（RC124,35s）均保留为非成功。
预编译虽生成了新模块映像，仍不把整条命令改判通过。后续加载与验证实际成功。
生产没有新增堆提示、线程默认值、资源豁免或内存硬限制；没有模型目标改变。

已交付：删除不省扫描的常驻缓存、四日旧输出对照、双任务默认GC观测、八日
完整流水线记录及当前39文件复验。尚缺新测试执行、稳定的并发内存余量、
更多长区间样本与单进程发布回归。没有宣布CPU做尽、GPU无路或发布2.0。
原checkout保留，未commit/reset/clean；最终关键源码哈希另见final_key_hashes.txt。
