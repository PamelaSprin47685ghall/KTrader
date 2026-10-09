# 单次拟合与残差遍历的复用（2026-10-08）

本轮只改 `response.jl` 的 scalar evidence cache 和
`residual_oracle.jl` 的掩码临时分配。其他人的 trace/J 去冗余改动保留。
不改证书、冷起点、迭代预算、OOF、随机流或模型结构；不声明 M0/M1
通过。上一轮 `inference_prices` 的面板拼接改动也补了值与所有权测试。

## Scalar 几何只构造一次

同一次 full/fold 拟合内，谱、targets 和 gauge 不变，所以 `B*gauge`、
`gauge'*YtY*gauge`、各约束块的 gauge 变换以及 hcoef 不随 alpha/Sigma
变化。把它们移出 EB 外层循环，不改变任何乘法和求和次序。

每个 Sigma 仍重新构造 Cholesky、quadratic、tensors、constant；每个
alpha 仍计算原 evidence/导数。证书仍走原独立入口。直接调用
`conditioned_evidence_cache` 会从该次输入重建；只有 optimizer 的
局部变量复用，不引入全局缓存、可注入 keyword 或跨折状态。

若 EB 需要 I 轮，nc 个 d×N 与 N×r 的约束块乘法从 I 次降为 1 次；
该块的主乘加计数约节省 `(I-1)*2*nc*d*N*r`。N65/d64/r910/nc14 时，
每少做一轮约 1.06 亿次浮点运算。这不是整个 fit 的加速倍数。

## 掩码只在 cache miss 时复制

原 scalar residual 遍历每行分配 N 个 Bool；现在复用一份 lookup
scratch，只在第一次遇见某个 `(fold, mask)` 时复制字典键。scratch
绝不作为存储键，避免后续改写破坏哈希表。原数值公式和求和顺序不变。
R 行、U 个非空 `(fold, mask)` 时，掩码载荷分配从 O(RN) 降为
O((U+1)N)；扫描成本仍 O(RN)，没有声称消除全历史计算。

## 验证

`test/fit_local_reuse_tests.jl` 已登记 REQUIRED（现 23 个文件；登记不等于
整个 registry 都运行通过）。
保留 test-only 旧 scalar-cache/残差公式，对照 primal/dual、gauge/no
gauge、多个 Sigma/alpha、导数差分、非 SPD 拒绝、闭包互不污染、复现
掩码和跨折、空行、重复调用，以及面板 NaN/Inf/负零和所有权。
本轮已实际运行；无完整回测、N65 fit 或超过 60 秒的任务。四条命令均为
Julia 1 线程 / OpenBLAS 1 线程，scoped deadline=50s、采样 RSS guard=2048
MiB，外层 timeout 55s、TERM 后最多 5s。没有放宽原证书或测试容差。

日志均在 `dev/evidence/`：

| 日志 | 结果 | scoped elapsed | peak RSS |
| --- | --- | --- | --- |
| `fit_local_reuse_arch_0318.log` | 架构 25/25，RC0 | 4s | 697 MiB |
| `fit_local_reuse_tests_0318.log` | 新增 244+7+6=257/257，RC0 | 5s | 723 MiB |
| `fit_local_reuse_neighbors_0318.log` | jcore 118 + residual-flow 45 + deadwork 228，全部通过，RC0 | 13s | 884 MiB |
| `fit_local_reuse_micro_0318.log` | 先重跑新增 257 项，再做局部测量，RC0 | 5s | 741 MiB |

四条日志尾行都是 killed=0/cancelled=none。相邻测试在独立 Julia module
中加载，使用既有断言；micro 重复执行的 257 项不重复计入覆盖数量。

## 局部测量，不是整机吞吐

可重放脚本 `dev/fit_local_reuse_micro.jl`：纯合成输入，不跑 prepare、
EB 优化循环、scenario 或 backtest。每个函数显式预热一次，分配取
3 次最小值，时间取 5 次中位数；两个版本在同一进程使用同一输入，
先断言结果一致。缓存对比针对第 2 轮起的复用，明确不含一次性 geometry
构造成本。不是整段 fit 的 before/after，也不推算真实 N65 吞吐。

| 局部对象 | 旧分配 | 新分配 | 旧中位时间 | 新中位时间 |
| --- | --- | --- | --- | --- |
| 每 Sigma cache + 一次 evidence；N8/n160/P112 | 390112 B | 279968 B | 158.26 μs | 129.79 μs |
| scalar residual；1800 行/N8/F3/重复掩码 | 564424 B | 28520 B | 470.161 μs | 218.521 μs |

分配分别少约 28% 和 95%。时间只是本机该次微观测，不是泛化提速承诺。

测试前已记录源码 SHA256；末次 hash/进程/Git 组合核对被工具安全层
拦截、未执行，因此不声明最终 hash 绑定或当前无其它进程。前面的
定向 source diff whitespace 检查 RC0；全仓检查指出 AGENTS.md 原有
Markdown 尾空格，未改动他人的文档。未 commit/reset/clean。
