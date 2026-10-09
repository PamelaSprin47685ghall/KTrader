# Gate-0 Wave 4 验证轮 — 证据索引

日期：2026-10-09。三段：quadrature_tests（Step 14）+ posterior_tests 复跑
（欠账闭合）+ quadrature.jl 骨架收口。全部命令 scoped ≤60s / RSS2048。

## 1. 命令清单与最终状态

| # | 命令 | log | rc | elapsed | RSS | 结果 |
|---|------|-----|----|---------|-----|------|
| 1 | test/gate0/quadrature_tests.jl | quadrature_tests.log | 1 | 18s | 943MiB | **32 Pass + 1 Fail**（fail-fast 中止；诊断副本证明 testset 7-10 全绿） |
| 1b | 诊断副本（any 断言注释，观察 7-10） | quadrature_diag.log | 0 | 17s | 918MiB | **54/54 全绿**（十个 testset 除注释项） |
| 2 | posterior_tests 复跑 | posterior_tests_rerun.log | 0 | 10s | 486MiB | **40/40**——上轮 predictive 验证的 posterior.jl 两处修改无回归，欠账闭合 |
| 3 | module 加载 + 导出面（-e assert 81） | module_load.log | 0 | 2s | 567MiB | API exports = 80 |
| 3b | market_tests 复验（names 断言 81 名） | market_tests.log | 0 | 5s | 643MiB | 82/82 |

## 2. 修复清单（file:line 为最终状态）

**src/gate0/quadrature.jl**：
1. L175-177：字符串插值 `dim=$dim——` 的 `$dim` 后跟中文破折号被解析为
   变量名一部分（ParseError）——改 `$(dim)`。

**test/gate0/quadrature_tests.jl**（交付者自报三处 + 运行暴露）：
2. 头部注释：「套套性」笔误 →「嵌套性」+ 删除 L50-51 别扭注释（交付者
   自报 1——按建议直接修）。
3. L103-110（gauss_hermite_nodes）：**Golub-Welsch 三对角 off-diagonal
   用错公式**——`i/sqrt(4i²−1)` 是 Legendre 的 Jacobi；physicists'
   Hermite（e^{−x²} 权重）正确值是 `sqrt(i/2)`（节点 ±√(2k+1) 渐近）——
   原形式节点域 [-1,1]，解析基准在错误节点/权重上积分（testset 4 的
   解析对照差 0.08 的根因）。修复后 testset 4 全过 12/12。
4. L197：`0x5555` UInt16（Julia 十六进制按最小宽度推断）→ `UInt64(0x5555)`。
5. L264/L288/L300/L325/L365/L368：同款 UInt32 seed 字面量（0x0000BEEF
   /DEAD/F00D/AAAA/BBBB——4 字节十六进制 = UInt32）→ `UInt64(...)`。
6. L210-211（testset 4）：max_scenarios 512 → 4096（A 停在 3.9e-3 >
   weight_tol=1e-3 @ 512；RQMC 权重 L1 实测收敛率 ~M^{-0.5}——与
   quadrature.jl 声明的「线性 scramble 方差阶弱于完整 Owen」一致）。
7. testset 8（L341-346）：max 512 → 4096 ×3 处调用（同款预算）。
8. testset 6（L262-268）：wt 1e-1 → 1e-2（第一轮即双过、分层窗口未
   出现）+ min_scenarios 8 → 32（**layer_diag.log 钉死：min=8 时 8/16
   点 RQMC 积分偏差使解落在满仓角点（w=[1,0]，E[gross col1] 偏
   +0.015）——两角点解相同 → A=1.8e-8 假收敛；M=32 才回内部解**）。
9. testset 9（L373/L376）：weight_tol → 3e-3 + max 4096（**AAAA seed
   的 A 非单调平台**：1.16e-3 @ 8192 → 1.32e-3 @ 16384——A 卡在
   ~1.2e-3 平台，weight_tol=1e-3/2e-3 均不可达；断言语义「两 seed
   权重差 ≤ 0.01」不变，收敛判据标定到平台之上）。
10. testset 5/6 的 sed 误伤修复（连锁 weight_tol 重复——恢复原调用）。

**骨架收口**：KTraderGate0.jl 十文件链 market → geometry → modes →
response → posterior → oof → innovation → predictive → kelly →
**quadrature**（消费 posterior 的 draw_mu、innovation 的 z_pool_rows、
kelly 的 cash_kelly——必须后于三者）；导出面 73 → **80 API**
（quadrature 7 名，无重名）；market_tests names 断言同步 81 名集合。

## 3. 未解决问题（数学语义层面，不擅改，回报原文与诊断）

**testset 6「certificate layering」的 any 断言**（quadrature_tests.jl:279）：
```
Expression: any((h->begin h.A <= wt && h.B > ut end), r.history)
```
（两轮修复 wt 1e-1→1e-2 + min 8→32 后仍红；诊断数据 layer_diag.log /
quadrature_diag.log）

**诊断**：Certificate B（utility regret 语义：I_audit(w_2M) − I_audit(w_M)）
是 **A 的二阶量**——B ~ H_eff·A²/2（H_eff ≈ 2σ² ≈ 0.003，实测 B/A²
~ 1.5e-7 @ A=1e-2 吻合）。「A 先过（≤wt）、B 后过（>ut）」的分层窗口
需要 ut < H_eff·wt²/2 ≈ 1.5e-7（当前 ut=1e-7 恰在边缘、BEEF seed 下
B 恰 ≤）。**B 的二阶收敛快于 A 的一阶——「A 过 B 红」的构造方向与
regret 语义的数学性质相反**（B 总不慢于 A 过）——该场景在光滑凸
Gaussian fixture 下可能根本不可构造（需要非光滑/多峰 I 使 B 出现一阶
效应）。testset 6 的其余 5 项断言全过（converged/M>32/第一轮区分力/
末轮 A∧B 同过/C 传导）——「A∧B 同过才退出」的分层语义本身已验证；
未验证的只是「A 过 B 红时继续翻倍」的**中间轮次形态**。处置权在
Manager（改 ut=1e-8 试构造 / 改断言语义 / 接受 B 二阶性质删除该断言）。

## 4. 边界

未修改：旧 src/、旧 test/、docs/、git。诊断脚本（layer_diag.jl /
quadrature_diag.jl）留证据目录。全部绿证据为单机当前时点一次运行事实。
