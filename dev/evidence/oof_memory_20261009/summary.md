# OOF 内存与四日回测：2026-10-09

## 已落地的生产改动

仅修改 `src/response.jl`；backtest、Kelly、incremental 与上一轮一致。
新增测试文件 `test/conditioned_storage_tests.jl`，登记为 REQUIRED。

一次 `optimize_conditioned_eb` 内，为加权块、直接收缩乘积和 Jacobian 核
复用存储。每次 alpha 变化后重新计算全部加权数据和核心矩阵，绝不拿旧数值
当新答案。缓冲仅由本次拟合所有，不跨 full/fold、日期或任务共享。
带缓冲的私有 alpha-cache 有代际检查：下一次准备开始时旧借用即失效，
包含失败准备；错误 owner 在写入之前被拒绝。公开缓存仍独立分配。

直接 M 收缩仍是同形状 GEMM，核心矩阵仍用原来的 `(a+b)/2` 对称化。
返回的 M/J/方向矩阵不引用可覆盖的 scratch；返回模型不携带该存储。
另外只融合自然方向的逐元素运算，保留 `S*J*S`、`S*S` 及原运算结合顺序。
冷起点、独立 OOF、rank、协方差 floor、alpha 搜索、迭代预算及证书不变。

复杂度：FLOPs 与渐近时间不变。若 m 个约束、d 个自由坐标、r 个谱方向，
每 alpha 原本为加权块/直接乘积/核心重复申请 O(mdr + m²d²) 存储；现在
每次拟合申请一次并按需重写。N65/P910 的三组 payload 共约15.72MiB，
不是整个拟合的内存上限。其他临时数组仍分配，不能声称零分配。

## 局部基准（含首次缓冲构造）

`conditioned_storage_micro.jl` 用同一合成 N65/P910/rank910，在固定 Sigma
下执行八个 alpha 的 cache/state/certificate 工作循环；不是完整 EB 拟合。
两条路线已有相同固定几何，新的存储构造包含在每个 after 周期内。
每个条目显式预热后测5次，ABBA顺序；所有正式样本编译时间为0。

| 配置 | 独立分配的两轮中位秒 | 复用存储的两轮中位秒 | 每周期分配 before/after |
|---|---|---|---|
| BLAS1 | 0.416637 / 0.414793 | 0.232177 / 0.227316 | 220123584 / 52462712 字节 |
| BLAS6 | 0.307805 / 0.343059 | 0.167568 / 0.146080 | 220123584 / 52462712 字节 |

分配减少76.17%；两条路线 checksum 精确相同。局部耗时包括各自产生的
GC，不能直接把这里的倍数套到回测。原始记录：`storage_micro_b1.log`、
`storage_micro_b6.log`。

## 真正完整四日对照

生产入口 `backtest_v1`；本地 CSV 经原 `load_bars -> signal_prices`，
N65/F3/S300/default independent EB/seed1。决策日2026-09-28至2026-10-01，
分别使用完整历史1:14306至1:14309，没有缩历史、减资产、删折或固定 alpha。
每次回测重建账户和 alpha 链；先显式小型 API startup，再真实四日 warmup，
最后 measured replay。下面两次都是一任务/BLAS6、默认 GC、默认优化级别。

| 四日生产调用 | before | after |
|---|---:|---:|
| wall 秒 | 8.232587813 | 7.886837966 |
| 累计分配字节 | 8112685976 | 6209850472 |
| GC 秒 | 1.815692219 | 1.368643298 |
| native compile/recompile | 0 / 0 | 0 / 0 |
| 整条命令采样 RSS 峰值 MiB | 1880 | 1842 |

累计分配减少23.455%，约1.903GB；GC观测降低24.62%，wall观测降低4.20%。
这是一份输入的一次配对观察，不是统计显著性或长期稳态保证。
same-config replay 8项和before/after比较5项通过，所有仓位与收益差异为0。
原始记录：`before_b6.log`、`after_b6.log`。

模型来源在本轮基准中只允许 `src/response.jl` 改变，其余运行文件完整哈希
逐项核对；CSV和前后输出均记录完整SHA256，读取前验证同一批字节。
after_b6 运行时尚使用第一版比较脚本；后续脚本修正仅影响 p2 的跨配置
比较规则与保存顺序，不反向改写这份同配置日志。

## 并发不是一个稳态倍数

相同四日、两任务/BLAS1探索（始终同一45s内层/55s外层和2048MiB采样护栏）：

| 运行配置 | 观察与判定 |
|---|---|
| before，heap hint1024M | warmup超时，RC124，峰值1789MiB；无正式速率 |
| after，hint1280M | warmup16.590s，其中GC12.035s；正式回放超时，RC124 |
| after，默认GC | warmup5.337s；正式回放RSS到2062MiB，被护栏终止，RC124 |
| after，hint1408M首轮 | 正式10.286s、GC5.940s、峰值1895MiB；重放8项过，比较脚本错用阈值导致RC1 |
| after，hint1408M修订验证 | 正式6.811615067s、累计5930021928字节、GC2.465188131s、编译0，峰值1987MiB；13项通过，RC0 |

1408M首轮把“不同拓扑”的收益差错用“同拓扑前后”的1e-12门槛，差异约
6.34e-11。仓位本已通过原 L1<=1e-7。新脚本区分这两个口径，跨拓扑恢复
早已存在于 `dev/batch_window.jl` 的1e-9/1e-9门槛，同拓扑保留1e-12/1e-10。
不是修改数学或原有测试容差。原失败日志保留，未伪造成功返回。
输出也改成先保存已通过自身重放的观测，再执行跨运行比较，避免失败丢数据。

修订验证与serial baseline的每日最大仓位L1为1.3086346284605208e-8，
收益最大差6.343758851556913e-11，均通过对应原跨拓扑门槛。它不是与同配置
before配对的加速证据，因为旧p2没有正式输出。两个1408M观测的明显波动
必须保留，不挑6.81s当长期能力。没有切换生产默认线程、engine或GC设置。

Julia官方内存文档说明 heap-size hint 使GC更积极，不是RSS硬限制：
`https://docs.julialang.org/en/v1/manual/memory-management/`。
本轮资源提示仅在相应进程设置；没有把采样护栏提高到2GiB以上。当前p2只余
约61MiB观测余量，不扩任务数。两日上的加速不能直接外推到更多日期。

## 当前版本完整标准文件范围

最终快照：`qualification/final/qualification_snapshot.toml`
SHA256 `965c00871def6932cdc297f889d97a22171cdfba92c8849c591f5486dc5f1b75`。
最终消费者 `qualification/final/status_complete.log` 实际RC0：

```
COVERAGE groups=20/20 standard_files=39 source_unchanged=true
```

39个文件=32 REQUIRED+6标准legacy+1架构；所有增量和进程护栏分相均运行。
每组执行前后核对同一快照。不是拿旧版本receipt凑当前覆盖，也不是单进程
全套交互运行。数值/数据/执行模拟组延续上一轮 `-O1` 控制测试编译开销，
其余组和全部性能测量使用默认优化级别。标准入口仍未改。
runner这次内置了标准import preamble，避免上轮遗漏Random/Dates的脚本错误。

新增存储612项包含 primal/dual、有无gauge、不同alpha、原始core公式对照、
NaN污染后重写、过期借用拒绝、不同owner拒绝、输出不被下一轮覆盖、
公开独立缓存仍保留旧值。原梯度960项与全部EB难例回归也通过。
最终 response_b 组6703项；其余各组精确计数在status及child日志中。
初版新增隔离断言错误地用冷M对比暖M，32项红已保留；修订用同路径污染前后
的快照作isequal比较，已有测试和容差没有改。生产状态的前后原公式比较始终过。

预编译命令的子阶段记录KTrader完成，但父进程随后超时，整体RC124，未计绿。
后续以现有缓存加载模块和运行验证均实际成功。最大成功命令为33s（numerics）；
超时/超内存日志仍为失败，不算进通过数。累计分配GB不是同时驻留GB。

## 当前边界

已交付：OOF存储复用与逐元素临时数组消除、同源完整四日对照、并发资源
实测、当前39文件全范围分组覆盖。尚不能宣称长区间稳定吞吐、并发内存余量
充足、单进程全套通过、CPU做尽或GPU无路。没有修改版本号或发布2.0标签。
工作仍在用户原checkout，未commit/reset/clean。最终关键哈希另见
`final_key_hashes.txt`；完整inventory以最终qualification snapshot为准。
