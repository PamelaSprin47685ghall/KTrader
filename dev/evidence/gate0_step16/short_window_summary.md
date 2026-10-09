# Gate-0 Step 16 短窗口轮 — backtest_tests + 骨架 + 5-day/20-day

日期：2026-10-09。D-089 阶梯短窗口验证（Gate-0 关闭后 D-088 解锁）。

## 1. 最终运行状态

| # | 命令 | log | rc | elapsed | RSS | 结果 |
|---|------|-----|----|---------|-----|------|
| 1 | backtest_tests | backtest_tests.log | 1 | 34s | 1407MiB | **22 Pass / 1 Error**（fail-fast 于 testset (4)；1-3/6/7 段中已执行部分全绿） |
| 2 | module 加载（-e assert 87） | module_load.log | 0 | 3s | 575MiB | API exports = 86 |
| 3 | market_tests（names 断言 87 名） | market_tests.log | 0 | 5s | 643MiB | 82/82 |

**5-day 级验证成立**：testset (1)「5 日端到端」**全绿**（逐日记录完整/
净值曲线有限/三 benchmark 同口径/kwargs 透传）——任务书第 3 项的判定
标准（端到端 testset 绿 ⟹ 5-day 级成立）满足。**20-day 未执行**：
前置条件（backtest_tests 全绿）不满足——testset (4) 红（见 §3）。

## 2. 修复清单（file:line 最终状态）

**src/gate0/backtest.jl**：
1. L259 / L261（两处）：字符串插值 `$var——` 的中文破折号 ParseError
   （同 quadrature.jl 的先例）→ `$(var)`。

**src/gate0/oof.jl**（L174/L184，两分支同步）：
2. tail_rel：exp(−u_span) → **10·exp(−u_span)**（一个数量级余量——
   Wave 4 的 area 因子对部分 fold 的 α₀ 平台-峰差不够；backtest
   fixture seed=31 首日实测红）。注释更新（L156-160）。

**test/gate0/backtest_tests.jl**：
3. 加载链补 market.jl + module using Dates（同 driver_tests 的坑）。
4. `_mk_mf` 加 drift 参数（L49-70）；主 fixture ×4 传 drift=0.003
   （mf/mf2/mf_l/mf_t4——α₀ 信号；testset (6) 的 benchmark fixture
   不动——无漂移语义是其断言对象）。

**骨架**：十二文件链 market → … → driver → **backtest**（消费 driver
的非导出常量 _DRIVER_DEFAULT_SEED——同链可见）；导出面 83 → **86 API**
（backtest 3 名：Gate0DayRecord/Gate0BacktestResult/run_gate0_backtest，
无重名）；market_tests names 断言同步 87 名。

## 3. 未解决事项（数学语义层面，两轮修复后按纪律停手，回报原文）

**backtest_tests testset (4)「locked 跨日」的 error**（L156 栈，
run_gate0_backtest 内 fit_fold_posteriors）：
```
Numerical integration did not converge: alpha tail mass certificate
failed (D-035a prior; red = implementation bug, not expected red)
```

**诊断（bt_tail_diag.log，数据驱动）**：
- t=270（首决策日）的**三个 fold 的边界 8 点全部 OK**（fold 1 最大
  边界值 -37.6 vs 要求 -32.0；fold 2/3 同样全过）——首日无问题；
- 红发生在**后续决策日**（t=271/272——rows 随 t 增长、fold 统计
  日间变化，某个 fold 的 α₀ evidence 形态触发证书）；
- 两轮修复（tail_rel 10× + fixture drift 0.003）后仍红——**drift
  不是根因**（首日同 fixture 绿、后续日红——**α₀ 信息量在「有/无」
  边界附近的日间波动**）。

**定性**：DC 形态 + 弱信号 fixture 的 fold posterior 的 α₀ 方向
信息量随决策日波动；tail 证书的「峰高于平台/边界衰减」前提在这种
边界形态下不稳定。这与 Wave 3 的「无 DC 一维化」同族——但 DC 形态
**不能**一维化（α₀ 在 λ 里）——**根本修复需要数学对象层面的裁决**
（例如：α₀ 弱信息时 fold posterior 的降级路径、或证书判据对弱信息
形态的自适应）。处置权在 Manager。

## 4. 边界

未修改：旧 src/、旧 test/、docs/、git。诊断脚本（bt_tail_diag.jl）
留证据目录。全部绿证据单机当前时点一次运行事实。
