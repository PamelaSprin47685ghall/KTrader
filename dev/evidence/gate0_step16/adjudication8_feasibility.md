# 裁决 8 可行性分析 — 解耦构造的行集同源性推导（静态，未运行）

## 目标
按裁决 8 构造 fixture：资产 4 历史「长且分散」（fold train 有 Syy 贡献
→ propriety 过）+ joint 行在 T4 日 ≤ 1（触发）+ 次日 ≥ 2（恢复）。

## 行集语义核对（源码事实）

1. **driver 的 rows**（driver.jl L284）：`rows = [s for s in WARMUP:(t-1)
   if any(view(obs_ret, s+1, :))]`——行 s 的资格 = 目标日 u=s+1 的
   **任一**资产观测。
2. **row_masks**（L288-289）：`row_masks[u] = obs_ret[u, :]`（目标日
   mask）。
3. **J 的定义**（innovation.jl L113-143）：`J = {i : row_ids[i] ≤ t ∧
   row_masks[i] ⊇ R}`——**O_u ⊇ R ⟺ 目标日 u 的 mask 覆盖 R 全部
   资产**。
4. **fold 的行集**（oof.jl L60-73）：contiguous——fold f 的 train =
   rows − fold f 的行。
5. **Syy 的 rank 来源**（posterior propriety）：Y_tr 的秩——mode
   坐标 y = E'·[m; e] 的自由度——**资产 j 未观测的行缺 e_j 自由度**
   （e_j = 0）；fold train 的全部行都缺资产 4 的 e 自由度 ⟹ Y_train
   的自由度塌缩 ⟹ rank(Syy⁽⁻ᶠ⁾) < N（实测：IPO fixture 的 fold 3
   train rank=2 < N=3——上轮 log）。
   ⇒ **fold f 的 train 需要 ≥ 1 个「资产 4 观测行」（e₄ ≠ 0 的行）
   才能保 rank**（随机数据下足够）。

## 同源性推导（死结证明）

**「资产 4 观测行」在三个行集中的身份完全同一**：
- J 的成员条件：`O_u ⊇ R`（R ∋ 4 ⟹ O_u ∋ 4 ⟹ **资产 4 观测行**）；
- fold train 的 rank 保障：需要 **资产 4 观测行**（e₄ 自由度）；
- 两者的行集都是 rows 的子集（row_ids = rows 的目标日）。

**约束不等式**：
- T4 触发（locked 资产 4 引起的覆盖不足）：`|J(R)| < 2`——R ∋ 4
  ⟹ **资产 4 观测行 ≤ 1**（J ⊆ 资产 4 观测行）；
- propriety（每个 fold 的 train）：**每 fold 的 train ≥ 1 个资产 4
  观测行**——contiguous F_folds 下，单行只属一个 fold ⟹ **资产 4
  观测行 ≥ F_folds ≥ 2**（每 fold 至少一行，且分属不同 fold）。

**矛盾**：`≤ 1`（T4）与 `≥ 2`（propriety，F=2 时已需 2 行）不可同时
成立。**F_folds 任意 ≥ 2 时死结不变**（2 行 → J ≥ 2 → T4 不触发）。

**裁决 8 的「前部就有行」选项**：WARMUP 之前的观测行不进 rows（L284
从 WARMUP 起）⟹ 不进 J 也不进 Syy——无法为任一侧提供行。

## 结论

**fixture 侧解耦在当前行集语义下不可构造**——这不是 market.jl 的
observed 摩擦（mask 本身可任意构造——IPO fixture 即证明），而是
**J 与 fold-train rank 共用同一「资产观测行」集合的结构性同源**。
修复需要语义层变更（超出 fixture 层，处置权在 Manager）：
(a) J 的覆盖判定改为「资产级而非联合级」（数学对象变更——与
    VI §1.3 的 joint 行语义冲突）；
(b) fold propriety 对「train 缺某资产观测行」的降级/豁免规则
    （D-036 的适用范围变更）;
(c) testset 5 放弃「恢复日正常决策」断言（T4 日后即结束——恢复
    语义的验证归非 T4 fixture）；
(d) 非 contiguous fold（如 checkerboard——SPEC §28 的 contiguous
    语义变更）。

**本分析为静态推导**（源码行号核对，未运行）；如 Manager 认为推导
有误，可指定具体构造形态由 DevOps 实测验证。
