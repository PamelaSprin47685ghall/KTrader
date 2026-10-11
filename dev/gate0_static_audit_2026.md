# Gate-0 P0-1..9 与 Exit 十八项：只读静态核对表

**状态：纯静态审计记录（新文件，2026-10-10）。**
**方法：逐文件静态阅读 `src/gate0/`（13 文件相关段）与 `test/gate0/`（测试断言段），并以只读 grep 复核符号残留。未运行任何命令、未执行任何测试；未修改 `src/`、`test/`、`docs/` 或任何既有文件。**
**证据基准：当前工作树（HEAD 602b897 + dirty）。凡引用的行号均为本次阅读时点的静态文本行号。**
**口径：本文只记录「实现/测试/fail-loud 是否真实存在」的静态证据；不宣布任何运行验证结果，不替代 DevOps 的运行证据。**

---

## 0. 一句话结论

P0-1..9 在 `src/gate0/` 与 `test/gate0/` 中**全部找到直接源码级证据**；三项重点补核全部确认：

1. **P0-5**：driver.jl 与 quadrature.jl 中全部 6 处 locked wealth 组装均为方案 2（`X_free·w_free + w_cash + base_locked`；全 locked 为 `w_cash·1 + base_locked`），无 `X_full·w_full + base` 双重计入残留；
2. **P0-4**：driver 的 `:adaptive` 分支无任何 locked 回落条件，构造性直连 `adaptive_scenario_kelly`；`quadrature_locked_fallback` 符号仅存于"已删除"注释；
3. **P0-6**：Certificate A/B_opt/B_audit/C 求值代码齐全，`M == max_scenarios` 同层比较由 `M2 == M → break` 结构性排除并 fail loudly。

发现 **2 处注释/文档滞后**（backtest.jl 的 locked 回落旧文案；GATE0_EXIT_CHECKLIST.md 的裁决 7 旧登记）+ **若干测试覆盖注记**（P0-8 prior 无直接公式断言；P0-6 分层时序为已知未测项）。详见第 4 节。

---

## 1. P0-1..9 逐项核对

### P0-1 — prequential residual history（F_folds 退出 production theory）

**裁决要求**（AGENTS.md 开头 P0-1）：`H_t → predict r_{t+1} → observe → ε → append`；`F_folds` 退出 production；`oof.jl` 保留为 legacy/reference diagnostic。

**实现位置：**

- 核心实现：`src/gate0/oof.jl:366-418` — `prequential_residual_rows(X, Y, E; tol, max_cells, u_span)`：
  - strictly causal（构造性）：`oof.jl:380-388` — train = 行 `1..s-1`（`X[1:(s-1), :]`），行 s 自身绝不进训练集；
  - 早期行/求积失败 → NaN：`oof.jl:391-409`（catch 经 `_row_undefined_error` 判定后置 NaN）；
  - 全退化 fail loudly：`oof.jl:415-416`（文本含 `"prequential residual"`）。
- 行级不可定义错误判定：`src/gate0/oof.jl:435-439`（只吞 `"posterior improper"` 与 `"Numerical integration did not converge"` 两类，其余 rethrow）。
- production 接线：`src/gate0/driver.jl:380-402` — L387 调用 `prequential_residual_rows`；L397-402 NaN 行过滤（`keep_res`）+ 空集 fail（L398-399）。
- legacy 标注（oof.jl 保留为 diagnostic）：
  - 文件头：`src/gate0/oof.jl:13-24`；
  - `fold_grid`：`oof.jl:56-60`（"【LEGACY / REFERENCE DIAGNOSTIC — P0-1 后 production 不再消费】"）；
  - `fold_stats`：`oof.jl:94-100`；`train_stats`：`oof.jl:108-118`；`fit_fold_posteriors`：`oof.jl:131-137`；`oof_residual_rows_full`：`oof.jl:240-245`。
- 不 export 声明：`src/gate0/oof.jl:47-49`（保持 5 名导出面不变）。

**测试位置：**

- `test/gate0/prequential_tests.jl:45-118`（专项 12 断言）：
  - 输出形态+早期 NaN：L56-69；
  - strictly causal：L74-90（L83 修改行 s target 不影响 s 自身；L87 修改 s+1.. 不影响 s；L89 正向控制）；
  - vs legacy OOF 语义区分：L95-104（L103 断言二者不逐位相同）；
  - fail loudly：L109-118（L110/L112 维度、L116-117 全退化）。
- `test/gate0/driver_tests.jl:78-79`（fixture 注释引用 prequential 逐行成本）。

**fail-loud 行为：** `oof.jl:374-377`（维度不符 DimensionMismatch）、`oof.jl:415-416`（全行不可定义 error）。

**结论：达标。** `F_folds` 参数仍存在于 legacy `fold_grid` 默认（`oof.jl:73-76`），但 production 不再消费（driver.jl 无 `F_folds` 参数，已读全文确认）；driver 链已切换为 prequential。

---

### P0-2 — adaptive continuous d（删除 production 固定 7 点网格）

**裁决要求**：`default_d_grid()` 退出 production；生产为 adaptive 1D deterministic quadrature；旧网格保留为 initial mesh。

**实现位置：**

- 初始网格：`src/gate0/innovation.jl:350-376` — `initial_mesh()`（7 节点 + midpoint cell mass，ΣΔ=1）；兼容别名 `default_d_grid() = initial_mesh()`：`innovation.jl:378-380`（docstring："保留供测试/历史对照；生产 adaptive 路径用 initial_mesh 作为起点"）。
- adaptive 实现：`src/gate0/innovation.jl:382-498` — `adaptive_d_quadrature`：
  - refinement certificate（相邻两级 M→2M−1）：`innovation.jl:488-490`（`|E2_d−E1_d|≤tol_d`、`|E2_d2−E1_d2|≤tol_d2`、`opnorm(E2_V−E1_V)≤tol_V·max(1,‖E1_V‖)`）；
  - 先验质量不漂移：`innovation.jl:453-468`（cell 细分只把旧 cell mass 拆给子 cell）；
  - fail loudly：`innovation.jl:497`（文本含 `"Numerical integration did not converge"`，禁止返回最后一层）。
- 生产路径选择：`src/gate0/innovation.jl:731-752` — `use_adaptive_d = d_nodes === nothing || d_cells === nothing`；缺省走 adaptive（`innovation.jl:776-781` 调 `adaptive_d_quadrature`）；固定网格需显式传入（校验 L741-751）。
- 固定网格消费方（非生产）：`innovation.jl:782-789`（显式 `d_nodes/d_cells` 时才走）。

**测试位置：**

- `test/gate0/innovation_tests.jl:535-603`（9.12 专项）：
  - `initial_mesh`/`default_d_grid` 等价：L536-541；
  - adaptive 路径（不传网格）：L543-560（节点≥初始、Σcells=1、与固定网格公共节点逐位一致 L557-560）；
  - refinement certificate 直接调用：L562-578；
  - 预算耗尽 fail loudly：L580-591；
  - 非法初始网格：L599-602。
- 导出面：`test/gate0/market_tests.jl:144-145`（initial_mesh/adaptive_d_quadrature）、`market_tests.jl:174-175`（default_d_grid/residual_rank/rank_sufficient）。

**fail-loud 行为：** `innovation.jl:497`；初始网格校验 `innovation.jl:431-439`（空/域/质量不符 error）。

**备注：** `default_d_grid` 仍在 export 列表（`innovation.jl:55`）与定义（L380）——符合 P0-2"保留为 initial mesh"的许可；grep 复核 `src/gate0` 中除 export/定义外无生产调用点。

**结论：达标。**

---

### P0-3 — innovation 满秩 gate（sample null space ≠ zero-risk）

**裁决要求**：joint residual span 必须满秩 `rank{ε_s^(R) : s∈J} = N_R`；不满秩时收缩 free / 全 cash / locked 不可收缩则 fail closed；禁止把 sample null space 宣布为 physical zero-risk space；Moore-Penrose 保留诊断。

**实现位置：**

- 秩工具：`src/gate0/innovation.jl:500-524`（`residual_rank`，SVD + 与 `mp_sqrt_factors` 同口径浮点分类）、`innovation.jl:526-534`（`rank_sufficient`）。
- 构造层 gate（`require_full_rank`）：keyword 文档 `innovation.jl:659-663`；实现 `innovation.jl:767-772` — 不满秩 error，文本含 `"innovation coverage failure"` 与 `"sample null space"`（L771）。
- 决策层 gate：`src/gate0/innovation.jl:940-992` — `resolve_risk_domain`：
  - 满秩判据：L966-974（`rank_ok`，`residual_rows` 提供时启用）；
  - free 逐个剔除：L976-985（剔"因果覆盖行数最少"的 free，tie 按索引升序）；
  - locked 不可收缩 → fail loudly：L986-990（文本含 `"innovation coverage failure"`，L988 引 P0-3 语义）。
- driver 生产闭环：`src/gate0/driver.jl:404-427`（二次 resolve 传 `residual_rows = resid`；T4 错误转译 held_maintained 或 rethrow）；`driver.jl:429-435`（`innovation_state(...; require_full_rank = true)`）。
- Moore-Penrose 保留诊断：`innovation.jl:280-303`（`mp_sqrt_factors`；null 不逆不注噪）。

**测试位置：**

- `test/gate0/innovation_tests.jl:608-659+`（9.13 专项）：
  - `rank_sufficient` 满秩/亏秩 fixture：L610、L617-618；
  - `require_full_rank=true` 不满秩 → error 且文本含 `"innovation coverage failure"`/`"sample null space"`：L620-630；
  - 满秩 fixture 不误杀：L632-635；
  - `resolve_risk_domain` 剔除亏秩 free：L637-651（`R_res == [1,2]`、`dropped == [5]`）；
  - locked 不可收缩 fail：L653-659。
- 导出面：`test/gate0/market_tests.jl:145`、`market_tests.jl:175`。

**fail-loud 行为：** `innovation.jl:771`（构造层）、`innovation.jl:989`（resolve 层）；driver 对 T4 错误转 held_maintained（`driver.jl:358-365`、L420-427），非 coverage 类 rethrow。

**备注（语义边界，非违规）：** `require_full_rank` 默认 `false`（`innovation.jl:701`）——默认保留 Moore-Penrose 诊断构造语义；**生产 driver 显式传 `true`**（`driver.jl:435`），且 resolve 满秩判据由 driver 二次调用强制（`driver.jl:413-416`）。生产闭环成立；直接调用 `innovation_state` 而不传该 keyword 的路径为诊断语义。

**结论：达标。**

---

### P0-4 — locked + adaptive 无 fixed-S fallback

**裁决要求**：删除 `quadrature_locked_fallback` 整条 production fallback；adaptive 直接处理 locked。

**实现位置：**

- quadrature 修复：`src/gate0/quadrature.jl:583-597`（注释：locked 列不进 free 优化列；切 free 子列传 `cash_kelly`、全列进 `base`）；`quadrature.jl:598` — `free_pos = [k for k in 1:src.n_assets if locked[k] == 0]`。
- driver 直连：`src/gate0/driver.jl:462-481` — `if mode == :adaptive` 分支内直接调用 `adaptive_scenario_kelly(post, st, x_now; ...; locked = locked_w, ...)`（L466-474）；**无任何 `locked` 条件分支/回落判定**（构造性证据：已读全文，L462-481 无第二分支）。
- driver 声明：`driver.jl:44-51`（文件头"P0-4 收口"）、`driver.jl:209-212`（docstring）、`driver.jl:452-456`（"不再存在…production fallback"）、`driver.jl:102-103`（`DriverDiag` 注释"不再有 quadrature_locked_fallback 字段"）。
- 符号复核：全 `**/*.jl` grep `quadrature_locked_fallback` → **唯一命中 `src/gate0/driver.jl:102` 的"不再有"注释**，零活跃代码。
- `:reference` 固定 S 仅保留为 D-062 合法用途：`driver.jl:482-521`（含 L455-456 声明）。

**测试位置：**

- `test/gate0/driver_tests.jl:142-166`：locked 语义（L156 `res_lock.diagnostics.reason == :ok`（无回落）；L157-166 locked 维持/budget/财富守恒/双重计数构造性反证注释 L164-166）。
- `test/gate0/driver_tests.jl:168-188`：T4 held_maintained（L182-188）。
- `test/gate0/quadrature_tests.jl`：locked 直接处理的生产路径未在 quadrature_tests 单独构造（其 fixture 无 locked）；kelly 侧 locked 输入校验见 `test/gate0/kelly_cash_tests.jl:131-164`、L306-308。

**fail-loud 行为：** `quadrature.jl:567-573`（locked 长度/非负/预算校验）；locked 资产缺预测律由 `kelly.jl:464-478`（`locked_wealth_gate0`，L472-473 error）。

**矛盾（未达标项 #1）：** `src/gate0/backtest.jl:244-245` docstring 仍写：

> "`mode` 默认 `:adaptive`（生产路径；locked 非零日自动回落 `:reference` —— driver 的接口摩擦语义原样延续，逐日 diagnostics 可见）。"

与 driver 收口后的现实不符；backtest.jl 代码本身（L273-277）直接透传 `mode`，**无回落逻辑**。这是陈旧注释（开发守则 §30"注释必须描述事实"），非行为缺陷。

**结论：实现达标；backtest.jl docstring 滞后（见第 4 节 #1）。**

---

### P0-5 — locked wealth 组装统一方案 2（无双重计入）

**裁决要求**：wealth 只能二选一：`X_full·w_full + w_c`（locked 填回）或 `X_free·w_free + w_c + base_locked`；禁止 `X_full·w_full + base` 双重计入；全 locked 分支与 utility-margin 诊断同规则。

**全量清单（driver.jl + quadrature.jl，逐处核对）：**

| # | 位置 | 形态 | 判定 |
|---|---|---|---|
| 1 | `src/gate0/quadrature.jl:602-613`（solve_layer 全 locked） | `objective = mean(log.(budget .+ base))`——无 `X·w` 项，`w_cash=budget` | ✅ 方案 2（L606-608 注释自证：旧写法 `vec(X*locked) .+ budget .+ base` 中 `X*locked == base` → 双重计入） |
| 2 | `src/gate0/quadrature.jl:653-660`（B_opt） | 全 locked：`wc_2M .+ base_2M`；否则 `X_free_2M * w_2M[free_pos] .+ wc_2M .+ base_2M` | ✅ 方案 2（L649-651 注释：绝不在 `X_full·w_full` 上再加 base） |
| 3 | `src/gate0/quadrature.jl:673-679`（B_audit） | 同型：`X_a_free * w_2M[free_pos] .+ wc_2M .+ base_a` | ✅ 方案 2 |
| 4 | `src/gate0/driver.jl:492-504`（reference 全 locked） | `obj = mean(log.(budget .+ base))`；L494-496 注释明确旧写法 `pl.gross*locked_w .+ budget .+ base` 中 `pl.gross*locked_w == base` → 双重计入 | ✅ 方案 2 |
| 5 | `src/gate0/driver.jl:505-519`（reference free 非空） | `X_free = pl.gross[:, free_pos]`；`cash_kelly(X_free; base=base, budget=budget)`；locked 填回 `w_R` | ✅ 方案 2 |
| 6 | `src/gate0/driver.jl:570-599`（utility-margin 诊断） | `I_of(w_free, wc) = mean(log.(vec(X_free_d * w_free) .+ wc .+ base_d))`（L587-588） | ✅ 方案 2 |

**grep 复核调用点**（`locked_wealth_gate0` 全 `src/gate0`）：`driver.jl:491`、`driver.jl:585`、`quadrature.jl:601`、`quadrature.jl:652`、`quadrature.jl:672`；定义 `kelly.jl:464`；其余命中为注释/export。**无其它 locked wealth 组装点**。

**旁证：** `kelly.jl:102-126`（`cash_kelly_certificate` 的 `wealth = X * w .+ w_cash .+ b`，L109）同为方案 2 语义（w 仅 free 列、b 为 base）；`kelly.jl:464-478`（`locked_wealth_gate0` 全列×locked 权重）。

**非组装点澄清：** `src/gate0/backtest.jl:278-281` 的 `wealth = dot(dec.w_universe, gross) + dec.w_cash` 是**账户财富结转**（`w_universe` 已含 locked 维持），不属于 Kelly scenario 财富组装，无双重计入。

**测试位置：**

- `test/gate0/driver_tests.jl:157-166`：`locked_exposure==0.25`、`budget==0.75`、`w_universe[2]==0.25`、财富守恒、L164-166 双重计数构造性反证注释；
- `test/gate0/kelly_cash_tests.jl:306-308`：`locked_wealth_gate0` 的 0*NaN 防护（`@test_throws ErrorException`）；
- quadrature 侧 B 求值路径由 `test/gate0/quadrature_tests.jl:212-238` 间接覆盖（无 locked fixture，覆盖 B_opt/B_audit 公式分支的 free 形态）。

**结论：达标；无 `X_full·w_full + base` 残留。**

---

### P0-6 — Certificate A + B_opt + B_audit + C（同层假绿结构性排除）

**裁决要求**：`A=‖w_2M−w_M‖₁`；`B_opt=I_2M(w_2M)−I_2M(w_M) ≤ ε_U`；`B_audit=|I_audit(w_2M)−I_audit(w_M)| ≤ ε_U,audit`；`C=fine KKT`；四者不可互替；`M==max_scenarios` 同层比较不得假绿。

**实际求值代码：**

- **A**：`src/gate0/quadrature.jl:644` — `A = norm(vcat(w_2M, wc_2M) .- vcat(w_M, wc_M), 1)`（含 cash 分量的增广 L1）。
- **B_opt**：`quadrature.jl:645-665` — `base_2M = locked_wealth_gate0(X_2M, locked)`（L652）；free 形态 L658-660；`B_opt = sum(log, wealth_f_opt) / M2 - sum(log, wealth_c_opt) / M2`（L665）；非正 wealth fail L662-664。
- **B_audit**：`quadrature.jl:666-686` — `X_a = src.gen(M2, audit_rule, MersenneTwister(am_seed))`（L671）；`B_audit = abs(I_f - I_c)`（L686）；非正 wealth fail L681-683。
- **C**：`quadrature.jl:639-642`（`cert_2M` 由 `cash_kelly` fail-loudly 保证；数值携带于返回值）。
- **联合判据**：`quadrature.jl:689` — `A <= weight_tol && B_opt <= utility_tol && B_audit <= utility_tol_audit`。
- **同层假绿结构性排除**：`quadrature.jl:632-638` — `M2 = min(2M, max_scenarios)`；`if M2 == M → break`（注释 L634-636："不生成同层 (max,max) 比较（A=B=0 假绿），直接 fail"）；预算顶 fail：`quadrature.jl:697-700`（文本含 `"Numerical integration did not converge"`，L698-699 报末轮 A/B_opt/B_audit）。
- 语义自证：`quadrature.jl:443-465`（docstring 循环伪码）、L472-493（三证书语义，含 L482-483 "旧实现的单向 B ≤ ε_U 允许大幅负 regret 通过——已修复"）、L504-508（控制流声明）。

**测试位置：**

- `test/gate0/quadrature_tests.jl:227-237`：C 传导 L228-230；A L232；B（B_opt）L233；B_audit L234；rule 身份 L235-236。
- `test/gate0/quadrature_tests.jl:263-298`（分层测试）：
  - 区分力断言 L277-280（第一轮未全过）；
  - **L281-290：明确记录"「A 过 B 红时继续翻倍」断言已按 Manager 裁决（Wave 4 执行轮）删除"，"「B 红时循环继续翻倍」的循环行为验证为已知未测项"**；
  - 末轮 A∧B_opt∧B_audit 同过：L291-294；
  - C 全程绿但循环继续：L295-297。
- `test/gate0/quadrature_tests.jl:303-327`：预算耗尽 fail loudly（L313-315 文本断言；**L316-326 `min==max` 同层排除的直接测试**——`err2` 必须 error）。
- `test/gate0/quadrature_tests.jl:243-254`：tolerance 减半 M 单调不减、权重稳定。

**注记（非违规）：**（a）实现无显式 `B_opt >= 0` 断言——非负性由 `w_2M` 为 fine 问题最优解（C 证书 objective_gap ≤ kelly_tol）支撑，判据只检查 `<= utility_tol`；（b）A 的口径为含 cash 的增广权重 L1（比"risky-only"更完整）。

**结论：达标（附上述注记与测试覆盖注记）。**

---

### P0-7 — 恢复严格 reference numerical contract

**裁决要求**：恢复严格 reference 口径；运行超时 fail / 缩小 fixture，禁止放宽 tolerance。

**生产入口默认值（driver.jl）：**

- `src/gate0/driver.jl:252-272`：
  - `weight_tol = 1e-4`（L257）；
  - `utility_tol = 1e-6`（L258）；
  - `kelly_tol = 1e-6`（L259）；
  - `posterior_tol = 1e-6`（L264）；
  - `posterior_max_cells = 2048`（L265）；
  - L252-263 注释："恢复裁决起始 reference 口径（1e-4/1e-6/1e-6 为严格起点…禁止由运行预算决定）"。
- 透传：`driver.jl:377-378`（posterior）、L387-390（prequential）、L472-474（quadrature）。

**对照违规值（AGENTS.md P0-7 所列旧实现）**：`posterior_tol=2e-2`、`ε_w=1e-3`、`ε_U=1e-5`、`kelly_tol=1e-5` → 现值分别为 `1e-6 / 1e-4 / 1e-6 / 1e-6`，**全部更严格**，恢复成立。

**其余层默认：**

- `src/gate0/posterior.jl:539-540`（`tol=1e-6`、`max_cells=2048`）、L514-517（docstring 纪律）；
- `src/gate0/oof.jl:369-371`（`tol=1e-6`、`max_cells=2048`）。

**注记（供审查）：**

- (a) `src/gate0/quadrature.jl:554-557` 函数级默认仍为 `weight_tol=1e-3 / utility_tol=1e-5 / utility_tol_audit=1e-5 / kelly_tol=1e-8`——**生产入口 driver 显式传参覆盖**（1e-4/1e-6/1e-6）；直接调用 `adaptive_scenario_kelly` 而不传 tol 的路径使用历史默认值（weight/utility 较松、kelly 较严）。若未来出现第二个生产调用方，需审查其传参。
- (b) `test/gate0/driver_tests.jl:76-84`：fixture 级 `POST_TOL_FIXTURE = 1e-2`（L84），注释明确"生产默认保持 1e-6（P0-7 严格口径）…此降级仅限测试 fixture"——属"缩小 fixture"形态，生产合同未动。
- (c) `u_span = 5.0`（`driver.jl:266-272`、`posterior.jl:524-534` docstring）：域半宽从 10 缩小到 5，注释自证为数值域参数（与 `tail_rel=exp(-u_span)` 联动、tail certificate 仍保护尾部），非 tolerance 放宽；该论证的强度依赖 docstring 推导，属 SPEC 复审素材。
- (d) **注释不一致（未达标项 #3）**：`test/gate0/driver_tests.jl:102` 注释写"容差与 driver kelly_tol=1e-5 标定一致"，而 `driver.jl:259` 实际为 `1e-6`——测试断言 `<= 1e-5`（L101-103）仍然成立（更严格解必然满足），但注释引用的是旧值。

**结论：达标（附注记；一处注释滞后）。**

---

### P0-8 — 恢复 proper HalfCauchy-τ prior；删除 _WEAKINFO 特例

**裁决要求**：`τ ~ HalfCauchy(0,1)`、`p(α) ∝ 1/(√α(1+α))`、log 坐标密度 `(1/2)u − log(1+e^u)`；删除 `_WEAKINFO_GAP`、`_WEAKINFO_SAFETY` 及特殊 weak-information tail 分支。

**实现位置：**

- 文件头声明：`src/gate0/posterior.jl:14-17`（"D-035a …已被 Gate-0 P0-8 裁决替换为 proper hierarchical prior τ~HalfCauchy(0,1)…tail 证书按统一判据签发，无弱信息特例分支"）。
- 常量删除注释：`src/gate0/posterior.jl:127-133`（"P0-8 裁决（2026-10-10）：删除 _WEAKINFO_GAP / _WEAKINFO_RATE / _WEAKINFO_SAFETY 弱信息特例机器——其阈值由回测 fixture 标定…违反开发守则"）。
- prior 实现：`src/gate0/posterior.jl:158-176` — `log_prior_d035a(u0, up) = (0.5*u0 - _log1pexp(u0)) + (0.5*up - _log1pexp(up))`（L175-176，与裁决公式逐字一致；`_log1pexp` L127）。
- tail 证书统一判据：`posterior.jl:297-319`（2D；错误文本 L319 "HalfCauchy-tau prior; red = implementation bug"）、`posterior.jl:417-423`（1D；L422-423）。
- 消费：`posterior.jl:575-576`（生产 logf = evidence + prior）、L604-605（1D 路径）。

**符号复核（grep 全仓 `**/*.jl`）：** `_WEAKINFO` 仅 `posterior.jl:129、130、173` 出现——三处均为"已删除"注释，**无活跃常量或分支**。

**测试位置：**

- **直接断言缺口（注记）**：全 `test/gate0/` grep `HalfCauchy` 零命中；`d035a` 仅 `posterior_tests.jl:6`（文件头引用）与 `market_tests.jl:166`（导出名单）。**未发现对 `log_prior_d035a` 公式的直接数值断言**；quadrature 收敛/尾证书的间接覆盖以 `posterior_tests.jl` 头注释（L10-11："#2（D-035a 下 quadrature 应绿）"）与 #11 收敛测试（L181+，本核对未逐行展读）承载。
- 弱信息分支删除的专项 red/green 测试未见（删除后应为"尾证书统一判据恒绿"语义，测试以 quadrature 收敛间接覆盖）。

**文档矛盾（未达标项 #2）：** `docs/GATE0_EXIT_CHECKLIST.md:176`（"裁决 7（posterior.jl tail 证书弱信息分支）：先验尾衰减锚定，数据驱动标定阈值 3 / 衰减率 1 / 余量 8"）与 `:183`（"posterior.jl——裁决 7：tail 证书弱信息分支 + 常量三处"）仍登记**已被 P0-8 删除**的机制。两处为清单的历史时点登记，与当前源码不符（文档滞后，源码侧 P0-8 已落实）。

**结论：源码达标；测试直接断言缺失（间接覆盖）；Exit 清单登记滞后。**

---

### P0-9 — cash Kelly canonical tie-break

**裁决要求**：`U(w)≈U*` 时：二阶段 `max cash`；三阶段同 cash 下 `min ‖w_risky‖₂²`；deterministic lexicographic。

**实现位置：**

- 文件头：`src/gate0/kelly.jl:39-49`；docstring：`kelly.jl:236-263`。
- 主调用：`kelly.jl:338-352` — `if tie_eps > 0` → `_cash_kelly_canonical_tiebreak(...)`（L344-346）；成功才替换（L347-351）；失败保留主解（L341-343 注释语义）。
- 二阶段（cash-max）：`kelly.jl:403-421` — `maximize(w2[m], [w2 >= 0, sum(w2) == budget, U2 >= U_lower])`（L407-410）。
- 三阶段（risky L2-min）：`kelly.jl:423-440` — `minimize(sumsquares(w3[1:n]), [..., w3[m] == wc_max])`（L427-431）。
- re-certify（原目标）：`kelly.jl:446-449`（`cash_kelly_certified(cert3)` 不过即返回 nothing）。
- `tie_eps` 校验：`kelly.jl:268`（负值 ArgumentError）。
- 主求解 fail-loud：`kelly.jl:304-305`（solver 状态）、L312（degenerate）、L335-336（证书失败 error）。
- 数值精修（非 tie-break）：`kelly.jl:147-211`（Newton polish，同一 KKT 系统；L317-331 采纳条件）。

**测试位置：**

- `test/gate0/kelly_cash_tests.jl:321-379`（P0-9 专项 18 断言）：
  - D-090(b) 对称射线平局 → 全 cash 端点（canonical）：L322-335；
  - D-091 无信号世界 → 全 cash：L337-350；
  - 唯一最优不受影响（Case C）：L352-364；
  - `tie_eps=0` 关闭 canonicalization：L366-370；
  - 负 `tie_eps` fail loudly：L372-373；
  - 目标不变（objective 差 ≤ 1e-7）：L375-378。

**结论：达标。**

---

## 2. 补核细节汇总（任务指定的三项）

### 2.1 P0-5 全量 locked 组装 — 见 §1 P0-5 清单（6 处全部方案 2；零残留）

### 2.2 P0-4 构造性证据

- driver `:adaptive` 分支（`driver.jl:462-481`）无 `locked` 参与的任何条件判断；调用参数 `locked = locked_w`（L469）；
- `quadrature.jl:598-626` 的 `solve_layer` 为唯一 locked 处理点（free 切列 + 全列 base）；
- 全仓 grep `quadrature_locked_fallback` 唯一命中为"不再有"注释（`driver.jl:102`）；
- `DriverDiag` 字段面 `driver.jl:104-105` 不含 fallback 字段；docstring L102-103 声明。

### 2.3 P0-6 求值与假绿排除 — 见 §1 P0-6（`quadrature.jl:632-638`、L644-700；测试 `quadrature_tests.jl:303-327`）

---

## 3. Exit 十八项重点项（1/2/3/6/15）直接证据定位

### 项 1 — `Bars.bar` 只表示 observation（D-012）

- 源码：`src/gate0/market.jl:33-54`（`observed` 唯一语义 + 构造不变量 L43-54："绝不静默改写 observed"）；`signal_prices` L104-115（`ifelse.(observed, adj, NaN)`，L115）。
- 测试：`test/gate0/market_tests.jl:118-127`（D-012 语义与 NaN 恢复；L125-126 缺 bar 处 NaN 而 marking 有限）；L129-201（只读防御：L191-193 aliasing、L196-200 政策层不污染 observed）；L113-115（carry 合法性）。
- 结论：**直接证据存在**（强于清单所标"summary 声明承载"）。

### 项 2 — model / trade / execution mask 分离（D-013/014/015/016）

- 源码：`src/gate0/market.jl:136-148`（三 mask 独立构造 + 防御 copy）；`model_admitted` L189-208（prefix 至少一条 return；L202-204 首个 admitted 日）；`trade_eligible` L211-217（默认全 true）；`effective_bars_252` L234-245（252 按 O 累计）；`executable` L247-258（= observed 独立副本）；`free` L260-273（= trade ∧ exec，**不含 admission**）。
- 测试：`test/gate0/market_tests.jl:129-136`（字段类型钉死）；L138-151（导出面 91 名断言，含三 mask 名）；L195-200（分离防御）；L203-231（D-013 判定含 dummy 排除 L226-230）；L233-256（252 边界 L240-252）；L258-292（free/locked，L273-277 admission 不混入 free）。
- 结论：**直接证据存在**。

### 项 3 — benchmark universe 外生（D-020）

- 源码：`src/gate0/market.jl:296-315`（`benchmark_universe` 只接收 `Eligibility`；L300-309 外生性声明）；`src/gate0/driver.jl:625-648`（`equal_weight_benchmark`；L634-635 注释"不接收 model/posterior/active 概念——外生性是签名层面的构造保证"；L643 调 `benchmark_universe`）。
- 测试：`test/gate0/market_tests.jl:294-309`（L299-303 W 未 admitted 仍进候选集；**L308 `@test_throws MethodError benchmark_universe(mfw)` 签名负控**；L304-305 与 free 一致性）；`test/gate0/driver_tests.jl:32`；`test/gate0/backtest_tests.jl:31`。
- 结论：**直接证据存在**（含负控）。

### 项 6 — 14 trace constraints 不在 default core（D-028/029）

- 源码：新线无任何 trace 约束实现——全 `src/gate0` grep `/neutrality|trA|trB/i` 唯一命中 `src/gate0/geometry.jl:11-16`（H7 注释："带旧数学耦合者（fit_response_operator、condition_trace_neutrality、optimize_conditioned_eb 等）**一律不进新线**"）；`posterior.jl` 的 response 后验无 trace conditioning（全文件已读）。
- 测试：`test/gate0/response_tests.jl:336-399`（**D-094 trace-hypothesis counterexample 闭环**：L359 构造健全断言 `trQ(G,1)=N·c≠0`；L361-366 unconstrained 恢复（全矩阵 + 点名对角 L365-366）；L368-388 trace-neutral 对照系统性删除（L386 对角和≈0、L388 删除误差 >100×）；L390-394 残差语义；L396-399 小噪声鲁棒）。L313-335 测试头注释钉死"新线永不施加 trace 约束；对照是测试内构造，不进 src/"。
- 结论：**直接证据存在且强**（反例测试比"不在 core"本身更强）。

### 项 15 — OOF full isolation（D-043）

- 源码：`src/gate0/oof.jl:156-233`（`fit_fold_posteriors`：train 差分 = full − fold L164-167；fold 独立 quadrature L173-221；fold 独立 propriety gate L168-171）。
- 测试：`test/gate0/oof_tests.jl:69-89`（**held-out 隔离直接断言**：L76-82 直接路径——修改 held-out 行后 train posterior 节点/权重逐位不变；L83-88 差分路径——held-out 修改后 fold posterior 不变）；L52-64（差分 vs 直接恒等式）；L94-105（残差行对照）。
- 补充：`test/gate0/prequential_tests.jl:74-90`（production 残差源的 strictly causal 断言）。
- 结论：**直接证据存在**（强于清单所标 B 级）。

**其余项（4/5/7/8/9..14/16..18）** 未在本次重点列表，但其源码载体已在 §1 对应 P0 项中定位（如项 4=response 四块、项 5=DC、项 7/8=posterior、项 9/10/11=innovation、项 12/13=quadrature、项 14=kelly）。

---

## 4. 未达标 / 矛盾 / 张力清单（按严重度排序）

| # | 类别 | 位置 | 事实 | 严重度 |
|---|---|---|---|---|
| 1 | 注释与代码不一致 | `src/gate0/backtest.jl:244-245` | docstring 称"locked 非零日自动回落 `:reference`——driver 的接口摩擦语义原样延续"；driver 已 P0-4 收口（`driver.jl:44-51/452-456`），backtest 代码（L273-277）亦无回落。陈旧注释，违反守则 §30 | 中（文档） |
| 2 | 文档登记滞后 | `docs/GATE0_EXIT_CHECKLIST.md:176,183` | 仍登记"裁决 7：tail 证书弱信息分支 + 阈值 3 / 衰减率 1 / 余量 8"；P0-8 已删除该机制（`posterior.jl:127-133`），grep 零活跃 `_WEAKINFO`。清单为历史时点文档 | 中（文档） |
| 3 | 注释引用旧值 | `test/gate0/driver_tests.jl:102` | 注释"driver kelly_tol=1e-5"；`driver.jl:259` 实际 1e-6。断言 `<=1e-5` 仍成立，仅注释滞后 | 低 |
| 4 | 测试覆盖注记 | `test/gate0/posterior_tests.jl` | P0-8 的 `log_prior_d035a` 无直接公式断言；全 test/gate0 grep `HalfCauchy` 零命中。间接覆盖为 quadrature 收敛 | 低-中（测试） |
| 5 | 测试覆盖注记 | `test/gate0/quadrature_tests.jl:281-290` | "A 过 B 红继续翻倍"分层时序断言已删除、自标"已知未测项"；末轮三条件同过断言（L291-294）与 `min==max` 排除（L316-326）保留 | 低（测试，已自标） |
| 6 | 语义边界 | `src/gate0/quadrature.jl:554-557` | 函数级默认（1e-3/1e-5/1e-5/1e-8）与 driver 生产传参（1e-4/1e-6/1e-6）不同；生产由 driver 覆盖，未来第二调用方需审查 | 低（供审查） |
| 7 | 语义边界 | `src/gate0/innovation.jl:701` | `require_full_rank` 默认 false（诊断语义）；生产 driver 显式 true。默认路径若被新调用方消费会静默保留 MP 诊断语义 | 低（供审查） |
| 8 | 数值选择 | `src/gate0/posterior.jl:524-534` | `u_span=5`（含 tail_rel 联动）为数值域参数；docstring 自证非 tolerance 放宽，属 SPEC 复审素材 | 低（供审查） |
| 9 | 范围边界 | `src/gate0/predictive.jl`、`src/gate0/backtest.jl` 全量 | 本次未全文核对（driver/quadrature 为任务指定重点，已全量）；backtest 逐日财富语义仅局部核对（L273-289） | 边界声明 |

**未发现**任何与 P0-1..9 实现语义直接冲突的活跃代码缺陷（如双重计入、fallback、固定 S 返回、_WEAKINFO 活跃分支、F_folds production 消费等）。

---

## 5. 核对方法与边界声明

- 本次为**纯静态**：读取 `src/gate0/` 13 文件（driver/quadrature/posterior/innovation/kelly/oof/market/geometry/backtest 相关段）、`test/gate0/` 12 文件相关段、`docs/GATE0_EXIT_CHECKLIST.md` 全文；grep 复核 6 组符号。
- **未执行**任何命令、测试、编译；未修改任何既有文件；本文件为唯一新增。
- 行号可能因后续编辑漂移；一切以源码字节为准。
- 本文不宣布任何运行结论：所有"达标"均为静态证据层面的判定；数值行为与全量测试状态以 DevOps 受控运行证据为准。
