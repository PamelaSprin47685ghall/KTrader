# Gate-0 裁决 7 执行轮 — 弱信息 tail 证书判据忠实化 + 回归确认

日期：2026-10-09。裁决 7（fold posterior 弱信息 tail 证书）+ backtest
复跑 + 三套件回归 + 20-day 评估。

## 1. 裁决 7 修复（posterior.jl，两处证书段同步）

**二维版（adaptive_quadrature_2d 证书段，~L279-321）**：
- before：单一判据 `边界 logf ≤ m_ref + log(tail_rel) + log(area)`（相对
  m_ref 锚定——弱信息时峰平台差小、相对语义失真）。
- after：**弱信息分支**——`gap = m_ref - plateau < _WEAKINFO_GAP(3.0)`
  时改用先验尾衰减锚定：`边界 logf ≤ plateau − _WEAKINFO_RATE·d +
  _WEAKINFO_SAFETY`（d = 边界到域中心的右尾距离和 max(0,u₀−r₀) +
  max(0,u_p−r_p)；RATE=1 为 D-035a 右尾每维 1/unit 的解析值；SAFETY=8
  log 单位）。**有信息形态（gap ≥ 3）判据完全不变**；**真的不衰减
  （边界 ≥ 平台）仍然红**、错误文本不变。

**一维版（_adaptive_quadrature_1d 证书段，~L418-435）**：同构同步。

**常量（posterior.jl ~L120-127）**：`_WEAKINFO_GAP = 3.0`、
`_WEAKINFO_RATE = 1.0`、`_WEAKINFO_SAFETY = 8.0`——实测标定（注释
钉死）：真弱信息 fold gap=1.09（backtest t=271 fold 2，走弱信息分支
全过）、中等信息 gap 4.1-7.4（走原判据全过）、有信号 > 10。**首轮
阈值 8 误伤中等形态（gap=7.13 的 testset 1 回归）——诊断数据驱动
修正为 3**（adjud7_diag.log / bt_tail_diag.log 的两份形态数据）。

**oof.jl 无独立判据**（消费 posterior.jl 的 quadrature 函数——确认）。

## 2. 运行结果

| 命令 | log | rc | elapsed | RSS | 结果 |
|------|-----|----|---------|-----|------|
| backtest_tests | backtest_tests.log | 1 | 36s | 1382MiB | **27 Pass / 1 Error**（testset 1-4 全绿——**tail 修复生效**；testset 5 新红见 §3） |
| posterior_tests（回归） | posterior_tests_rerun.log | 0 | 10s | 489MiB | **40/40** ✓ |
| oof_tests（回归） | oof_tests_rerun.log | 0 | 7s | 542MiB | **25/25** ✓ |
| predictive_tests（回归） | predictive_tests_rerun.log | 0 | 24s | 1539MiB | **56/56** ✓ |

**裁决 7 的验收**：tail 证书弱信息假阳性已消除（testset 4 的 fold 2
gap=1.09 形态走先验锚定全过）；三套件回归零回归。**20-day 未执行**：
backtest_tests 未全绿（testset 5 新红）——前置不满足。

## 3. 未解决事项（数学语义红，回报原文）

**backtest_tests testset (5)「T4 跨日」的 error**（L179 栈）：
```
fit_fold_posteriors: posterior improper (fold f=3) — train-layer
condition failed (n⁽⁻ᶠ⁾=16, N=3, rank(Syy⁽⁻ᶠ⁾)=2; D-043/D-036)
```

**定性（结构性死结，静态推导）**：testset 5 的 fixture（IPO 资产于行
278 起 observed、T4 触发需要 IPO 覆盖不足、恢复语义需要次日正常决策）。
恢复日 t=280 的 rows = 257..279——**IPO 资产的非零 Y 行（s ≥ 277）
全部位于 rows 尾部**；contiguous fold 的 fold 3 恒含尾部 ⇒ **fold 3
的 train（前部行）结构性缺少 IPO 信号 ⇒ rank(Syy⁽⁻ᶠ⁾) < N ⇒
propriety gate 正确拒绝**（D-036 fail-loudly 是正确行为）。**IPO 提早
则 train 有信号但 T4 不再触发（覆盖充足）——两个语义在该 fixture
形态上互斥**。修法选项（均数学语义层，处置权在 Manager）：
(a) testset 5 放弃「恢复日正常决策」断言（T4 日后即结束——恢复语义
    另用非 IPO fixture 覆盖）；
(b) 换非 IPO 的 T4 触发设计（如 mask 间歇缺失——需保证恢复日 train
    信号完整）；
(c) fold propriety 对 IPO 列的降级规则（数学对象变更）。

**注意**：driver_tests 的 T4 单日测试（73/73）不受影响——单日决策的
rows 更长（IPO 行占比高）、fold train 有信号；本红是多日短窗口
（rows 24 行）+ IPO 尾部 + contiguous fold 的组合。

## 4. 边界

未修改：旧 src/、旧 test/、docs/、git。诊断脚本（adjud7_diag.jl /
bt_tail_diag.jl）留证据目录。全部绿证据单机当前时点一次运行事实。
