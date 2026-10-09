# 发布收口检查：发现并修复真实失败，仍不发布2.0

## 关于“还需要几次交互”

没有可靠的次数估计。本轮原计划不再加小优化，冻结现有实现、补齐最后一项
测试并重复验证。结果证明缺口不只是几份日志：换到不重叠日期后，生产回测
确实失败。发布判断应据具体阻断是否关闭，而不是完成多少轮聊天。

原AGENTS.md§80–82仍是标准：数学不变、reference/accelerator可替换、
CPU/增量方面完成有价值的工作，以及条件需要时完成GPU最后加速层。约42
days/s是mission，不是可以牺牲数学的硬门槛；短窗口通过也不能自动等于2.0。
本轮没有改版本号、打标签、提交、reset或clean。

## 修复前的冻结结果

上轮唯一缺失的`backtest_cache_tests.jl`经原有
`dev/release_batch.jl cache_elision`实际执行，110/110、RC0、13s、963MiB。
原快照`680f1494d2d435847af2b4972d8be440e9f4f519fb2af0889789e302dbabeeba`
保持不变，原汇总实际达到21/21组、40标准文件：
`../batch_memory_20261009/qualification/status_complete.log`。
这是修复前源码的分组全覆盖，不是单进程全套，也不是当前修复版本的通过记录。

新增冻结消费者`dev/release_freeze.jl`单独登记自身哈希，逐项核验原快照
文件和所有通过凭据。它没有把自己的新代码追认为旧测试覆盖的一部分。
冻结记录`freeze.toml`的SHA256为
`b41e70fc18cb6c05b2ae2811b53f19932da233925b1de531c85aaad55d5c929c`。

同一真实八日窗口（2026-09-22至2026-10-01）在独立进程复跑：

| 观察 | 完整八日秒数 | 正式编译时间 | 采样RSS峰值MiB | 与原结果差异 |
|---|---:|---:|---:|---|
| 上轮基线 | 15.767236451 | 0 | 1684 | 基线 |
| latest1 | 15.807302412 | 0 | 1660 | 仓位L1/收益均0 |
| latest2 | 15.741374539 | 0 | 1689 | 仓位L1/收益均0 |

均为N65、全部历史、F3/S300、默认EB、一个日期任务/BLAS6、默认GC。
每次独立启动，显式小型API startup和两日真实warmup在正式测量之外，
正式回测重置持仓及alpha链。只有同一窗口，不能据此估计所有历史的耗时分布。

## 不重叠日期发现两个问题

`earlier1.log`使用2026-09-10至2026-09-21，但在两日真实预热阶段就失败，
尚未执行正式八日测量。对外错误是`InvalidStateException: Channel is closed`。
代码原来仅捕捉EOFError，因此没有抛出生产者已经保存的真正异常。
原冻结汇总随后正确返回RC1、2/4重放完成，见`status_blocked.log`。
这不是数值通过，不是超时，也不能归因于编译。

`earlier_direct_root.log`直接调用原fit入口、保留前一日各折alpha链：
2026-09-18、完整前缀1:14300完成；2026-09-21、1:14301在OOF中报：

```
conditioned EB has no certified ascent direction
free_rms = 1.4960563744338608e-6
active_directions = 19
```

原门槛是1e-6，所以这是应当保留的数学拒绝，不能把值约等于容差视为通过。
直接复现命令RC1、31s、1740MiB；没有新参数、缩历史或改tol。

## 两处正确性修复

`src/backtest.jl`提取`_backtest_take`。先正常取完缓冲中的有效结果；遇到
已关闭且已耗尽的Channel时，识别真实的InvalidStateException（并保留旧
EOFError兼容），抛出该block原先保存的异常对象。没有根因却提前结束时
报协议错误。没有message解析、全局异常吞并或把失败转成空结果。
该改动保存异常类型/对象，不声称恢复生产者原始backtrace；原栈由直接复现取得。

`src/response.jl`已有一个Riccati终止候选，只有通过原完整alpha/协方差
证书才可返回。但非正方向导数分支在失败后直接抛错，绕过了同一个兜底。
现在只有这个“原本必失败”的分支也进入既有候选；成功步仍走原continue，
正导数分支、冷起点、rank、alpha搜索、迭代预算和所有门槛均不改。
候选未通过仍抛原失败，不能返回未认证结果。没有新求解器或放宽容差。

相同真实前缀直接复现随后完成两日，见`earlier_candidate_trial.log`，
RC0、30s、1722MiB。既有conditioned-EB测试145/145通过，包含BLAS1及
实际BLAS6冻结难例；既有timeblock测试604/604通过。
原拒绝路径被修复，不证明所有日期都能收敛。

`test/backtest_failure_tests.jl`新增了原异常身份、缓冲排空、异步关闭唤醒、
无根因提前结束、真实worker失败、spool清理和BLAS恢复检查，已登记REQUIRED。
其启动被工具拦截，未执行；没有改wrapper或换runner绕行。因此这些新增
测试没有绿证据。修复后的架构检查另行实际执行25/25通过。

## 修复后的端到端证据与失败

第一次最近八日复验在真实warmup中超时，`repaired_latest.log` RC124。
768M heap hint的包加载也超时，不能算成功或把全部耗时解释成JIT。
1024M提示下的后续离线包加载实际2s完成、640MiB；它不跑拟合。
后续所有生产测量仍是默认GC，没有把这项提示用于回测。

`repaired_latest_loaded.log`完整最近八日完成，15.893177345秒，
累计12062085872字节、GC3.139859037秒、native compile/recompile均0，
整条命令33s/1687MiB。原日期、账本、场景数检查6项通过，与认证旧输出
比较4项通过，仓位L1和收益差异均0。这不是新加速比例，而是原成功窗口
没有被正确性修复改变的证据。

`repaired_earlier.log`中的两日warmup现在完成，证明不再立即卡在已知拒绝。
随后正式八日仍在数值不动点/Jacobian路径中触及时限，RC124、35s、1684MiB，
没有完整输出。栈快照只说明停止的位置，不能分解整段总时间。没有重试
加长deadline、扩大迭代预算或降低门槛。这一窗口仍未验收，不发布2.0。

## GPU环境核对

只读lspci显示Navi21；实际`/opt/rocm/bin/rocminfo`成功退出，详见
`rocm_inventory.log`：GPU agent `gfx1030`，Marketing Name为AMD Radeon
RX6800。amdgpu内核驱动、/dev/kfd、/opt/rocm及HIP运行库存在。
Project.toml当前未声明AMDGPU/CUDA后端依赖。
这说明设备与运行时可被枚举，不证明KTrader内核能加速或可以直接使用某个
库。未装依赖、未跑GPU计算，也没有依据宣称“GPU做不了”。

## 当前版本的明确状态

新快照：`post_repair_qualification/qualification_snapshot.toml`，SHA256
`5ba9407e8b27eee1982008da8f5d277fa0be3e586a9766093f4d5fcdd3e51600`。
它登记修复后的41个标准文件和22个组，不继承旧40文件的通过凭据。
本轮修复后实际证据是145项EB、604项timeblock、25项架构和最新窗口重放；
不能据此写成41文件全绿。新增异常测试仍未运行。

冻结尝试没有扩大功能范围；发现真实缺陷才解冻并做了上述两处小修复。
剩余交付固定为：修复版本的独立/发布回归、更早窗口的完整资源内验收，
以及按原2.0定义对CPU剩余工作和GPU最后加速层作出有证据的结论。
不把交互次数或又一轮局部优化当作发布门槛。
旧通过、原数值失败、所有超时、工具拦截都保留，状态始终NOT RELEASED。
