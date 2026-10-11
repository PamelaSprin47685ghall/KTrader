# T4 多日语义裁决（裁决 9）——production prequential 正向覆盖与语义边界

**文档状态：Manager 裁决记录 + 设计文档（2026-10-10）。**
**适用对象：`src/gate0/`（KTraderGate0）多日 T4 语义；新测试 `test/gate0/backtest_tests.jl` testset (8)。**
**边界：本文档不改任何裁决原文（D-013/D-036 保持 `AGENTS.md` 原样）；不改 `src/`；不动现有 testset 断言。运行验证已落盘（2026-10-10 三段：首跑 `archive/evidence/gate0_t4_recovery_20261010/`；ruler 修订后复跑 `archive/evidence/gate0_merged_verify_20261010/` B2；kelly 修复后 `archive/evidence/gate0_final_verify_20261010/`——testset (8) 15/15 转绿）——见第 5 节。**

---

## 0. 结论摘要

T4 多日「触发→恢复」路径在 **production（strict prequential）语义**下可构造，并已固化为 testset (8)；
裁决 8 登记的「解耦不可构造」成立于 **legacy fold 语义**——该语义下 J 覆盖行与 fold train 的 rank 保障行集同源互斥；
fold 已退役为 reference/diagnostic，该死结不适用于 production。三候选修法均经 Manager 裁决不执行（详见第 3 节）。

---

## 1. 冲突重构：D-013 与 D-036 在尾部新资产上的语义不一致

### 1.1 两个规范

- **D-013**（`AGENTS.md:1246`，prefix 级 admission）：A^model_{t,i} = 1 ⟺ prefix 中至少存在一条有效 daily return；
  弱证据由 posterior 表达；不用 MINROWS 门槛。实现：`src/gate0/market.jl` 的 `model_admitted`（约 L118–157，return-pair 判定与 `admitted[first_return:T, j] .= true` 在约 L133–153）；
  driver 落地：`src/gate0/driver.jl:307-310` 步骤 1 `act = findall(el_t.model_admitted[t, :])`。
- **D-036**（`AGENTS.md:1914`，posterior-propriety gate）：reference prior 的前提是 posterior proper；不满足必须 fail loudly，不得 clamp/EB/inverse-Wishart，回 SPEC 审查。
  fold 级实现：`src/gate0/oof.jl` 的 `fit_fold_posteriors`（约 L125–183），gate 在约 L147–150：`(st_train.n >= N && rank_syy == N) || error("fit_fold_posteriors: posterior improper (fold f=$f) — ...")`；
  production 对应物：`fit_full_posterior` 的 A2 数据层 gate（`src/gate0/posterior.jl:548-551`：`n ≥ N && rank(Syy) = N`，文本含 "posterior improper"）。

### 1.2 行集同源死结（fold 语义；裁决 8 推导原文见 `archive/evidence/gate0_step16/adjudication8_feasibility.md:27-45`）

三个行集共用同一「资产观测/覆盖行」集合 S_j ⊆ rows：
1. driver 的 `rows = [s ∈ WARMUP:(t-1) : any(obs_ret[s+1, :])]`（`driver.jl:330-331`），行 s 目标日 u = s+1；
2. innovation 的 joint 行 J = {u ≤ t : O_u ⊇ R}（行集判定，`innovation.jl` 的 `joint_row_indices`）；
3. fold 的 train = rows − fold f（contiguous 等分，`oof.jl` 的 `fold_grid`）。

矛盾不等式（F_folds ≥ 2）：
- T4 触发（locked 资产 j 覆盖不足）要求 |J| ≤ 1 ⟹ |S_j| ≤ 1；
- fold propriety（每 fold 的 train 都要有 ≥1 个 S_j 行保 rank）要求 |S_j| ≥ 2 且不集中在单 fold；
- 二者互斥：凑到 2 行则 J = 2（T4 不触发）；只有 1 行则某 fold 的 train 缺该行 → `posterior improper`。

### 1.3 最小时间线（fold 语义下的「恢复日红」）

旧 IPO 形态（资产 j 于 v0 起观测、locked）：
- t = v0+1：J = {v0+1} 单行 → 首次 resolve 即 T4 触发（held_maintained）；
- t+1 = v0+2：mask 覆盖行 {v0+1, v0+2} → 首次 resolve 通过 → 步骤 7 通过 → 步骤 8 fold：contiguous 划分下尾部行之前的 fold train 不含覆盖行 → `fit_fold_posteriors: posterior improper (fold f=…)` → 恢复日红。

即 `archive/evidence/gate0_step16/adjudication8_summary.md:59-60` 记录的「T4 触发 ✓ 但恢复日 fold propriety 亏」。

---

## 2. 关键转折：production 残差源已切换为 strict prequential

- P0-1 裁决后：`driver.jl` 步骤 8 = `prequential_residual_rows`（行级不可定义 → NaN；全 NaN 才 fail loudly）；3-fold OOF（`oof_residual_rows_full`）降级为 legacy/reference diagnostic，**不再被 production 决策链消费**（文件头与 driver 注释钉死）。
- prequential 没有「每 fold train 需覆盖行」的要求：行 s 的残差可定义 ⟺ 行 s 的 train（严格过去行）里已有资产 j 的覆盖行——**与 J 的过滤同源且相容**（覆盖行出现后，其后各行自然可定义），不构成死结。
- **最小构造**（静态推演，后经受控运行验证，见第 5 节）：资产 j 观测从行 t-2 起连续、locked：
  - t 日：首次 resolve（mask 行）J = {t-1, t} = 2 → 通过；步骤 7 通过；步骤 8 仅行 s = t-1 可定义 → keep_res = {t-1}、row_ids_ok = {t}；**二次 resolve：J = {t} 单行 → 剔除 free 后仍不足 → T4 触发**；
  - t+1 日：keep_res = {t-1, t} → row_ids_ok = {t, t+1}；**二次 resolve：J = {t, t+1} = 2 行、rank(2×2) = 2 → 恢复**（正常决策）。
- 同一构造在 fold 语义下仍死（最后段的 train 不含覆盖行）——**跨语义缺口**的精确含义：同一 T4 多日场景，legacy fold 判死、production prequential 放行。

---

## 3. Manager 裁决（2026-10-10）

1. **裁决对象 = production（prequential）语义**；fold 为 legacy/reference。
2. **不采三修法**：
   - *admission 重定义*：否决——与 D-013 文本直接冲突（须走理论变更审查门），且把 OOF/fold 实现细节写进模型准入，职责倒置；
   - *非 contiguous fold*：不充分——不消灭一般形态（|S_j| ≤ 1 时任何划分都死），只移动阈值；且动机不纯（不为测试便利改数学对象）；
   - *fold 域收缩*：记为「fold 复活为正式对象时的预案」，当前不执行（与 innovation 联合行域对齐是唯一正当性来源，须带数学论证）。
3. **行动**：(1) 验证 prequential 最小构造（触发→恢复）；(2) 成功后固化为正式新测试（production 语义，不声称 fold 可行性）；(3) 语义边界成文（本文档）。

---

## 4. 新测试设计：testset (8)「T4 recovery (prequential)」

### 4.1 落点论证

落点 = `test/gate0/backtest_tests.jl`（而非 driver_tests.jl）：本路径是**多日循环**语义（触发日 + 恢复日 + 跨日持仓漂移），归 `run_gate0_backtest` 领域；与 testset (5)（T4 不触发路径）构成正反对照；复用该文件的 `_BT_SCENARIOS` 分组调度协议（默认只跑场景 1，不拖慢默认运行；全量分组需在 `BACKTEST_SCENARIO` 中加入 "8"）。

### 4.2 fixture（自含、确定性）

- `T = 273`、`N = 2`；`t_start = 270`、`t_end = 272`（决策日 270、271；marking 至 272）；
- 资产 1：全观测，平缓漂移 + 固定 seed 小噪声（提供 macro/DC 信号与训练样本）；
- 资产 2（尾部新资产、locked）：仅行 268..272 观测；log 价格窗口内交替 ±0.05（268→269 +、269→270 −、270→271 +、271→272 −）；`trade_eligible[:,2] = false`；`held = [0.1, 0.3]`；
- KW 与 testset (5) 同款 fixture 级降级：`mode = :reference`、`S_reference = 32`、`diag_S = 16`、`posterior_tol = 1e-2`（数值配置语义，D-066；生产默认不变）。

### 4.3 预期路径（静态推演）

决策日 270：首次 resolve J = {269, 270} 通过 → prequential keep_res = {269}（仅 train 含覆盖行 269 的行可定义）→ 二次 resolve J = {270} 单行 → 剔除 free 后仍不足（覆盖缺口由 locked 引起）→ **held_maintained / innovation_coverage_locked**。
决策日 271：keep_res = {269, 270} → 二次 resolve J = {270, 271} 两行、rank = 2（构造性，见下）→ **恢复正常决策，R_universe = [1, 2]**。

### 4.4 rank 确定化（不赌随机一般位置）

资产 2 的收益交替 ±：恢复日两条残差行（s=269 目标 270 为 −；s=270 目标 271 为 +）的资产 2 分量异号；且 prequential 逐行拟合会把前一冲击部分吸收、进一步放大方向差 → rank(2×2) = 2 由构造保证。固定 seed 使全部数值确定。

### 4.5 断言清单（15 条；以 testset (8) 实际源码为准）

触发日：`mode == :held_maintained`；`reason == :innovation_coverage_locked`；error 文本含 `innovation coverage failure`；`w_universe == held`；`held_before == held`；wealth 因子有限为正；`length(days) == 2`、`days[1].t == 270`。
恢复日：`days[2].t == 271`；`mode == :reference`；`R_universe == [1, 2]`（free 未剔除）；`locked_exposure ≈ held_before[2]`；预算守恒 `Σw + w_cash ≈ 1`；跨日漂移桥接 `held_before ≈ w·gross/wealth`。
全程：净值有限为正（无 error 传播）。

### 4.6 语义声明（写入测试注释）

本测试验证 **production prequential** 路径；fold 语义下该构造不可行（fold train 需覆盖行）；**不以此测试声称 fold 可行性**。

---

## 5. 运行验证

### 5.1 首跑（ruler 修订前）——语义红，根因闭环

证据：`archive/evidence/gate0_t4_recovery_20261010/`（01_first_run_s8.log 等 + summary.md）。

- 首跑为**语义红**：0/15 断言执行（断言前的运行时异常，非语法/机械错误；Test Summary 1 Error / 1 Total）。
- 触发日 t=270 **完全按设计工作**：mode=:held_maintained、R_universe=[1,2]、w=[0.1,0.3]、reason=:innovation_coverage_locked。
- t=271 恢复路径：预测律数值爆炸 → `locked_wealth_gate0` fail loudly（`held asset 2 has no predictive return law`）。
- 根因：资产 2 历史自 268 行起，t=271 时 ruler 仅 τ=1 可用（n_use<2）→ `out[j,:] .= 1.0` 回退（s1=1.0，正常约 0.05）→ gross 爆炸（col1 max≈1.75e21；col2 max≈2.86e289、4 处 NaN）。
- 诊断变体（资产 2 起点提前至 260）：s1=0.05、t=271 正常返回（`:reference`、R_universe=[1,2]；无 NaN）——爆炸与极短历史强相关，闭环。
- 处置：修订 ruler 短历史回退（设计文档 `docs/RULER_SHORT_HISTORY_FALLBACK.md`，τ=1 RMS 平坦外推），而非改测试构造（后者会改变 T4 触发构造与断言语义）。

### 5.2 复跑（ruler 修订后）——预测律修复生效，第二阻塞（kelly）

证据：`archive/evidence/gate0_merged_verify_20261010/` B2（B2_bt8.log / B2_diag_s1.log + summary.md）。

- **预测律修复生效**：恢复日 s1(资产2)=0.05（原回退值 1.0）；gross 两列全 finite（无 NaN；宽尾 1.57e-9~4.73e9 / 2.26e-55~3.46e56，属新资产预测律固有不确性）。
- **当时未转绿**：链路推进后 t=271 撞 kelly `SLOW_PROGRESS`（1 Error；15 断言未执行）——kelly 第二阻塞（后由 §5.3 的 kelly 修复解除）。
- 断言计数确认：场景 8 = 15；全量 49 → 64（与 §4.5/§4.6 预期一致）。
- 原风险点排查：风险 2（pre 单行 2D quadrature）与风险 1（rank 构造）未再构成红点——链路越过 resolve 抵达 kelly 层（正式构造下 resolve/rank 未再报红）。

### 5.3 kelly 修复后：testset (8) 15/15 转绿

证据：`archive/evidence/gate0_final_verify_20261010/`（最终验证批次；场景 8 与 1–7 合计 64/64 全绿）。

- kelly 条件行缩放（span>4.5e15 阈值；设计文档 `docs/KELLY_NUMERICAL_ROW_SCALING.md`）与 polish 防护恢复落地后，t=271 的 SLOW_PROGRESS 解除。
- testset (8) 15/15 全绿；backtest 场景 1–7（49）+ 场景 8（15）= **64 全绿**。
- 最终口径：test/gate0/ 全量 **984 断言全绿**（模块 1–12 + backtest 场景 1–8；见 AGENTS.md 第14任段与 `docs/GATE0_EXIT_CHECKLIST.md`）。

### 5.4 边界

- 各阶段均为单机、受控 scoped 运行（≤58s/RSS2048）；运行期间未改测试/source（SHA 前后一致，见各目录 pre/post hashes）。
- 本文件历史时点的「不声称场景 8 全绿」声明已被 §5.3 的收尾验证取代（15/15 转绿）。

---

## 6. 未决点

1. **裁决 8 时点的 driver 残差源**（fold vs prequential）与「上轮红」原始 log 缺失——是否请 DevOps 复现旧形态，归 owner 裁决；本裁决只对**当前 HEAD 语义**成立。
2. **修法 (a) 的精确收缩定义**（fold 复活预案）：收缩到哪个集合、域变化时如何保持可重放——需要数学论证与 refinement。
3. **非 contiguous fold 对 innovation 校准统计性质的影响**未评估（本裁决不执行该修法，记录为背景）。
4. **计数连锁（2026-10-10 收尾）**：旧 890 → 913（modes +8、场景 8 纳入）→ **最终 984 全绿**（testset (8) 15/15 转绿后，backtest 49→64；`archive/evidence/gate0_final_verify_20261010/`）。全量分组建议 "1" / "2,3" / "4,5" / "6,7" / "8"。

---

## 7. 参考

- `AGENTS.md:1246`（D-013）、`AGENTS.md:1914`（D-036）；
- `docs/GATE0_EXIT_CHECKLIST.md:147、178`（裁决 9 登记）；`archive/evidence/gate0_step16/adjudication8_feasibility.md`、`adjudication8_summary.md`；
- 源码：`src/gate0/market.jl`、`oof.jl`、`driver.jl`、`posterior.jl`、`innovation.jl`（file:line 见正文）；
- 测试：`test/gate0/backtest_tests.jl` testset (5)（不触发路径对照）、testset (8)（本正向路径）。
