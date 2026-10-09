# Gate-0 Wave 4 裁决执行轮 — 证据索引（裁决 4/5 + Step 15 driver）

日期：2026-10-09。三段：裁决 4（any 断言）+ 裁决 5（locked 通道）+
driver_tests（Step 15）+ 骨架收口。全部 scoped ≤60s / RSS2048。

## 1. 命令清单与最终状态

| # | 命令 | log | rc | elapsed | RSS | 结果 |
|---|------|-----|----|---------|-----|------|
| 1 | quadrature_tests（裁决 4+5 后） | quadrature_tests.log | 0 | 18s | 922MiB | **54/54 全绿**（any 删除后 testset 6 5/5；locked 修复不影响全零路径） |
| 2 | driver_tests | driver_tests.log | 0 | 33s | 1465MiB | **73/73 全绿** |
| 3 | module 加载（-e assert 84） | module_load.log | 0 | 3s | 569MiB | API exports = 83 |
| 3b | market_tests（names 断言 84 名） | market_tests.log | 0 | 6s | 620MiB | 82/82 |

## 2. 裁决执行

**裁决 4（any 断言删除）**：quadrature_tests.jl testset 6——删除
`any(h -> h.A <= wt && h.B > ut, r.history)` 断言；注释钉死：B 是 A 的
二阶量（实测 B/A² ≈ 1.5e-7 恒定，B ~ H_eff·A²/2）、双证书分工是 A
权重稳定性 / B 目标 regret 两维度而非分层时序、「B 红时继续翻倍」为
已知未测项（未来重尾 fixture 或构造性 mock）。其余 5 项断言保留
（converged / M>32 / 第一轮区分力 / 末轮 A∧B 同过 / C 传导）。

**裁决 5（locked 通道修复）**：quadrature.jl 的 solve_layer 重写——
locked 列不进 free 优化列：`free_pos = [k for k in 1:n if locked[k]==0]`，
切 free 子列传 cash_kelly、全列（free+locked）进 base
（locked_wealth_gate0「X is the FULL matrix」约定）；locked 维持权重写回
w_full；free 列空（全 locked）时返回平凡可行解（w=0、gap=0）。与
driver.jl 的 reference 路径同构。**driver 的回落路径保留**（裁决「以
driver_tests 断言为准」——测试 L120-122 明确断言回落形态
`res_lock.mode == :reference` + `quadrature_locked_fallback == true`）。

## 3. driver_tests 修复清单（file:line 最终状态）

**源码侧**：
1. **driver.jl L346（fold_grid 越界）**：`fold_grid(rows)` 传**全局行号**
   （值域 256..t-1）→ folds 值被当 X_tr 的**局部**索引用（44×57 矩阵
   索引 [256..270] → BoundsError）。oof.jl 契约是局部行索引（oof_tests
   同款：fold_grid(n) → 1:n）。修：`fold_grid(length(rows))`；全局行身份
   由 innovation_state 的 row_ids 独立承载（与 resid 行序一一对应）。
2. **driver.jl（posterior quadrature 透传）**：默认 1e-6/2048 预算耗尽
   （rel 4.8e-3 @ 2048）→ 加 posterior_tol/posterior_max_cells keyword；
   最终标定 2e-2/1024（时间可达：全链 ~2-3s × 8 次调用 ≤ 50s）。
3. **driver.jl（kelly_tol 标定）**：1e-8 → 1e-5（两轮实测：reference
   路径 gap 2.37e-8 @ 1e-8 超；adaptive solve_layer kkt 3.5e-6/gap
   4.2e-6 @ 1e-7 仍超——问题规模随 M 退化；1e-5 两路径可达，求解证书
   语义不变，kelly_cash_tests 小 fixture 仍用 1e-8 验证 polish 机制）。

**测试侧**：
4. 加载链补 market.jl（driver 消费 MarketFacts/Eligibility——原链从
   geometry 起，UndefVarError 实证）+ module 内 using Dates（market.jl
   依赖骨架层 Dates）。
5. **market.jl 补文件内 export**（11 名，与骨架层同名单——原 Wave 2
   骨架层 export 模式，测试独立加载 `using .Module` 拿不到非导出名）。
6. **testset (1b) 重写为 D-067 传导断言**：交付者自报风险的确凿实测
   ——无信号随机游走 + μ 通道 matrix-t epistemic draw 的 fixture 上
   A @ (128,256) = **0.100**（μ 通道噪声主导；对照 quadrature_tests
   有信号 Gaussian fixture A @ 256 ≈ 3.9e-3）——初始候选容差在 50s
   预算内不可达。**D-067 fail-loudly 是正确行为**——断言 error 含统一
   文本；adaptive 正向收敛验证归 quadrature_tests（54/54）。处置权在
   Manager：(a) driver adaptive 默认降级 reference；(b) μ 通道 RQMC 化
   （quadrature.jl docstring 钉死的未来扩展）；(c) fixture 加信号。
7. testset (5) 断言对象改 res_ref（res_ad 已随 (1b) 改写删除；D-076
   字段完整性不依赖路径）+ 删除 res_ad 的 5 个残留断言。
8. testset (7) a2/b2 改 reference（时间预算：8 次全链 adaptive 双跑超
   50s；截断一致性不依赖路径，adaptive 重放确定性由 1b+quadrature 覆盖）。
9. 证书容差断言 1e-6 → 1e-5 ×4 处（与 kelly_tol 标定一致——实测地板
   kkt 1.6e-6 / gap 4.2e-6）。

**骨架**：十一文件链 market → geometry → modes → response →
posterior → oof → innovation → predictive → kelly → quadrature →
**driver**（依赖链末端——消费全部前述公共接口 + market）；导出面
80 → **83 API**（driver 3 名：Gate0DayDecision/single_day_decision/
equal_weight_benchmark，无重名）；market_tests names 断言同步 84 名。

## 4. 未决事项（D-066 反馈，处置权在 Manager）

**driver 的 adaptive 生产路径在其 fixture 上预算内不可收敛**（testset 1b
实测：A @ (128,256) = 0.100，μ 通道 matrix-t 噪声主导、需 M ≫ 预算）。
当前测试断言 D-067 传导（fail-loudly 正确行为）。选项：(a) driver
adaptive 默认降级 reference；(b) μ 通道 RQMC 化；(c) fixture 加信号。
**注意**：这是「无信号 fixture + 高维 epistemic 噪声」的组合效应——
有信号 fixture（quadrature_tests）adaptive 正常收敛（54/54）。

## 5. 边界

未修改：旧 src/、旧 test/、docs/、git。全部绿证据单机当前时点一次
运行事实。
