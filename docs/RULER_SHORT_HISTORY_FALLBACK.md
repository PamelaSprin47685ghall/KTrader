# ruler 短历史回退修订（n_use < 2）——设计文档

**状态：实现 + 回归已落盘（2026-10-10）；登记供 SPEC 审查。**
**改动面：`src/gate0/geometry.jl` 的 `ruler` 回退分支（`n_use < 2`）+ `test/gate0/modes_tests.jl` 新增回归 testset。不改其它既有行为；`n_use ≥ 2` 的拟合路径字节不变。**
**证据基准：`archive/evidence/gate0_t4_recovery_20261010/`（DevOps 首跑与变体实验；本文按 Manager 转述引用该事故链，未逐一复核日志正文）。**

---

## 0. 结论摘要

ruler 的 `n_use < 2` 回退从 **s(τ) ≡ 1.0**（日波动 100% 的荒谬尺度）修订为
**该资产自身无门槛 τ=1 增量 RMS 的平坦外推（H = 0）**——最小修订、自足、不引入横截面先验；
该修订消解 testset (8) 恢复日（t=271）的预测律病态，且可证明覆盖全部 active 资产场景。

---

## 1. 事故与证据链

- **首跑（testset (8)，`01_first_run_s8.log`）**：t=270 触发日按设计工作（`held_maintained`、`R_universe=[1,2]`、`w=[0.1,0.3]` ✓）；
  t=271 恢复日红——资产 2（观测自 268 起、t=271 时 `T−f+1=4`）在 ruler 的 `n_use<2` 分支获得 **s1 = 1.0**（正常 ~0.05），
  场景 gross 爆炸（col2 上界 2.86e289、4 处 NaN）→ `locked_wealth_gate0` fail loudly（"held asset 2 has no predictive return law"）。
- **变体实验闭环（`05_diag_s8_history.*` 等）**：仅把资产 2 起点提前到 260（s1=0.05）→ t=271 正常返回（`:reference`、`R_universe=[1,2]`）。
  爆炸与「极短历史触发 ruler 回退」强相关。
- **根因**：D-013 允许的极短历史资产（首观测后 ≤3 行、τ=1 未过 `4τ ≤ T−f+1` 门槛 ⇒ `n_use=0`；或仅 τ=1 过门槛 ⇒ `n_use=1`）在 ruler 层获得
  「无信息回退 1.0」——量级错误 20 倍（0.05 → 1.0），标准化/场景映射随之病态；held（locked）机制假设「资产有可用预测律」，对极短历史资产不成立。

---

## 2. ruler 全链（计算与消费）

- **计算**（`src/gate0/geometry.jl` 的 `ruler`）：逐资产扫 τ ∈ TAUS；对满足 `4τ ≤ T−f+1` 且存在 finite 增量的 τ 计 `val = log(sqrt(max(acc/ws, 1e-12)))`；
  `n_use ≥ 2` 时在 log-log 上拟合 H 并外推到全部 TAUS；`n_use < 2` 走回退分支（本次修订对象）。
- **消费**：driver 步骤 2 `s1_act = vec(ruler(x_log, f_first)[:, 1])`（只用 τ=1 列）→ `build_mode_problem(r, obs_ret, s1_act, E_active; dc=true)`（field 标准化 `u = r/s1`）；
  决策层 `predictive_law(...; s1 = s1_act, ...)` 与 `adaptive_scenario_kelly(...; s1 = s1_act, ...)` 把 mode 空间预测映射回 asset 空间（μ_i = s1_i·μ_norm，innovation 同乘）→
  locked 资产的 scenario wealth 由其 gross 承担（`locked_wealth_gate0`，资产缺预测律即 fail loudly）。
- **量级语义**：s1 是「一日 log 收益尺度」——0.05 量级对日频资产正常；1.0 = 100% 日波动，物理荒谬。

---

## 3. 候选比较与裁决

### (a) τ=1 RMS 平坦外推（采纳）

- **定义**：回退值 = 无门槛 τ=1 增量 RMS（`sqrt(max(acc/ws, 1e-12))`），对所有 τ 输出同一值（H=0）。
- **量级论证**：τ=1 的 RMS 增量就是「一日尺度」的直接估计（正常拟合下 s(1) ≈ 同量）；
  「平坦」是对「单点无法识别 scaling 斜率」的诚实表达（不发明趋势），且消费点只用 s(τ=1)，其余列仅作一致填充。
- **不变量**：纯自身计算 ⇒ permutation（逐资产独立）、price-scale（log 平移相消）、dummy（全 NaN 列零贡献）全部构造性成立。
- **可达性**：active 资产在 prefix 内必有 ≥1 个相邻观测对（model_admitted 的定义），且 f 为首观测行 ⇒ 无门槛扫描至少命中 1 个增量对 ⇒ `ws ≥ 1`；
  `ws == 0` 保留 1.0 兜底（不可达防御，见 §4）。

### (b) 横截面中位尺度（不采纳）

- 量级与不变量：中位在 permutation 下不变、dummy 不参与、log 平移不变——理论上均成立。
- **不采纳理由**：(1) 对可达场景（n_use=1，自身 τ=1 可算）无增益；
  (2) 对「无任何自身增量」的不可达情形只是理论装饰（active 语义已排除）；
  (3) 引入跨资产耦合与「同类资产尺度相近」的先验——在 price-only 纪律下这笔先验需要单独论证，超出最小修订边界。

### (c) 组合（a + 收缩/下限）（不采纳当前形式）

- 「收缩/下限」的动机是小样本低估（单个极小增量 ⇒ s1 过小 ⇒ 反向放大）真实存在，但下限数值无法从数据推导（不得拍脑袋常数；不得用回测选择）；
- 该风险记为未决点（§8），留给 SPEC 审查与后续 refinement；当前保持纯 τ=1 RMS。

---

## 4. 修订定义（实现）

`n_use < 2` 分支（仅此分支）：

```julia
acc = 0.0; ws = 0.0
@inbounds @simd for t in (fj + 1):T
    x1 = x[t, j]; x0 = x[t - 1, j]
    if isfinite(x1) && isfinite(x0)
        d = x1 - x0
        wt = w === nothing ? 1.0 : w[t]
        acc += wt * d * d
        ws += wt
    end
end
out[j, :] .= ws > 0 ? sqrt(max(acc / max(ws, 1e-12), 1e-12)) : 1.0
```

- 确定性：无随机、同输入同输出；`n_use ≥ 2` 分支一行未动（字节不变）；
- `w`（可选权重）语义与主循环一致；`1e-12` 下限沿既有数值保护口径。

---

## 5. 影响面盘点（n_use < 2 场景）

| 场景 | 现状 | 修订后 | 断言影响 |
|---|---|---|---|
| `driver_tests` (3) 单日 T4（IPO 于 269、t=270；n_use=0） | s1=1.0 | s1=真实 τ=1 RMS | held_maintained 在步骤 6 触发、不消费预测律；其断言不依赖 s1 数值（行集判定只看 mask）|
| `backtest_tests` testset (8)（t=270 n_use=0 / t=271 n_use=1） | t=271 爆炸（红） | s1≈0.05，恢复日正常 | 修复目标：testset (8) 应转绿 |
| `backtest_tests` testset (5)（资产 2 fj=257、t=279；n_use=3） | 拟合分支 | 不变 | 无影响 |
| `modes_tests` L59-63（T=300、f=1；n_use≥2） | 拟合分支 | 不变 | 无影响（且只断言形状/有限/正，未锁 1.0）|
| `market_tests` L156（导出面） | — | 不变 | 无影响 |

---

## 6. 回归设计（`modes_tests.jl` 新 testset，8 条断言）

`@testset "ruler short-history fallback (n_use < 2)"`：
1. n_use==1：`s1 ≈ 0.05`（已知增量 RMS；**旧 1.0 必红**）；
2. 平坦外推：全部 τ ≈ 0.05；
3. 尺度合理性：`all(s < 1.0)`（**旧 1.0 必红**）；
4. n_use==0：`s1 ≈ 0.02`（无门槛 τ=1；**旧 1.0 必红**）；
5. 无增量：保留 1.0 兜底（行为沿旧）；
6-8. 不变量：permutation 逐位协变、dummy 零贡献、price-scale 不变。

预测律有限性的端到端验证 = **testset (8) 重跑**（修复后应绿；其断言语义不改）。

---

## 7. 重跑清单（由 DevOps 受控执行；本任务不运行）

1. `test/gate0/modes_tests.jl`（含新回归 testset）——预期 102 + 8 = 110 条全绿；
2. `test/gate0/backtest_tests.jl` 场景 8（`BACKTEST_SCENARIO="8"`）——预期 testset (8) 15/15 转绿；
3. `test/gate0/backtest_tests.jl` 既有场景（至少 5，验证无回归）——预期 49 条无回归；
4. `test/gate0/driver_tests.jl`（含单日 T4 (3)）——预期 71/71 无回归。

全部 scoped ≤60s / RSS 护栏；串行执行。

---

## 8. 未决点与风险

1. **小样本低估（无下限）**：单个极小增量会使回退 s1 过小（反向放大）；下限数值不可拍脑袋——留 SPEC 审查/refinement（§3c）。
2. **SPEC 登记**：本修订为「模型定义的最小修订」，需并入 SPEC 的 ruler 语义段（含 `4τ` 门槛与回退语义的完整陈述）。
3. **证据复核边界**：本文按 Manager 转述引用事故链，未逐一复核 `01/02/03/05` 日志正文。
4. **n_use==0 的无门槛语义**：以「active ⇒ ≥1 相邻对」论证可达性；若未来 admission 语义变化，需重审。

---

## 9. 边界与纪律

- 不得用回测收益选择：本修订的取值全部由数据自身定义（自身 τ=1 RMS），无自由参数；
- 改动仅回退分支 + 测试 + 本文档；`n_use ≥ 2`、D-013、T4 语义、testset (8) 断言均不动。
