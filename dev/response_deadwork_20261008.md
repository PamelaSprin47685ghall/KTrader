# 后验求解去掉无用计算（2026-10-08）

范围：只改 `src/response.jl` 的两个内部计算步骤。此次按用户要求与
正确性收尾并行推进；不宣称 M0/M1 已通过，不改证书、迭代预算、
冷 covariance 起点、OOF 隔离、数据、场景或 Kelly。原工作树保留，未提交。

## Trace 直接收缩

设 `Bd = B .* d` 为 `rank × N`，`U` 为 `P × rank`，每个约束的
列索引是 `cols[c]`。原来计算整张 `G = Bd' * U'` 后仅取各块 trace。
现在直接计算 `h[c] = sum_i dot(Bd[:,i], U[cols[c][i],:])`。
保留先对谱维求和、再对对角线求和的次序；BLAS/GEMM 和 dot 的舍入
允许不同，测试与显式 G 对照。共享同一次 `Bd` 给 R，不引入新缓存。

trace 部分从 O(N P rank) 降为 O(N nc rank)，`P=N nc`。
N65/P910/rank910/nc14 时主乘加计数约从 107653000 降为 1656200；
这是该步骤的运算量之比 65，不是整个缓存或回测的加速倍数。
内部 alpha-cache 的无人使用字段 G 删除；最终 posterior mean 的 G
仍由 `fit_response_operator` 正常计算。公开模型结构与函数签名不变。

## 不动点只计算 J

将 J 的原有 core 收缩提取为 `conditioned_constraint_jacobian`，完整
state 求值和不动点共用，不复制数学公式。每个不动点试探点保留 M/X
的 Cholesky（含原 SPD 拒绝），但不再计算丢弃的 evidence、谱梯度及
direction。残差、Riccati 候选、回溯、外层 evidence/free_rms 接受条件
和最终完整证书均不改。剩余 Riccati 特征分解不属于被消掉的工作。

## 验证范围

新增 `test/conditioned_deadwork_tests.jl`，已登记 REQUIRED（22 项；
登记不等于全部运行）。覆盖 primal/dual、有无 relative gauge、多个
alpha、显式 dense trace、冷 core 装配、floor 起点、短不动点轨迹逐
步对照和非 SPD 拒绝。旧 full-state 不动点循环仅作 test oracle。
既有测试及其容差未修改。运行结果以本次后续日志为准，未预填通过。

不运行完整回测、不重跑全套、不做 N65 拟合。共享机器上另有工程师
运行测试时，不把受干扰的计时包装成性能证据。
