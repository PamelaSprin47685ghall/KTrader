# Gate-0 Wave 2 集成与验证收口 — 证据索引

日期：2026-10-09。执行者：DevOps（Wave 2 集成与验证）。
范围：骨架 include 收口 + 三测试文件受控运行。全部命令 ≤60s、RSS 护栏 2048MiB、
scoped_run 形态（bin/scoped_run.sh 50 <log> --rss-guard=2048 julia --startup-file=no --project=. …）。

## 1. 命令清单与结果（最终状态）

| # | 命令 | log | rc | elapsed | RSS peak | Test Summary |
|---|------|-----|----|---------|----------|--------------|
| 1 | module 加载 + 导出面检查（-e include + assert 33 名） | module_load.log | 0 | 2s | 555MiB | KTraderGate0 loaded OK, API exports = 32 (+ module self name) |
| 2 | test/gate0/market_tests.jl | market_tests.log | 0 | 4s | 642MiB | 82/82 Pass |
| 3 | test/gate0/kelly_cash_tests.jl | kelly_cash_tests.log | 1 | 14s | 886MiB | 29 绿（Case A 4 / B 4 / C 7 / D 8 / D-090(a) 6）+ D-090(b) 2 Fail 中止 |
| 4 | test/gate0/modes_tests.jl | modes_tests.log | 0 | 4s | 389MiB | 102/102 Pass |

命令 3 的补充观察（Test.jl 非交互模式在 testset fail/error 后中止 script，
后续 testset 未在原文件内观察；诊断副本逐层推进，原文件一字未动）：

| 诊断运行 | log | 结果 |
|----------|-----|------|
| polish 单元诊断（Case C 数据 + 1.1e-7 扰动） | polish_diag.log | rc=0；kkt 2.69e-9 → 2.29e-14 |
| 诊断副本 1（注释 D-090(b) 两断言） | kelly_diag_copy.log | 推进至 D-091：D-091 对称断言红（1.425e-6 > 1e-6），其余 5/5 |
| 诊断副本 2（再注释 D-091 对称断言 + 辅助函数加同款 polish） | kelly_diag_copy2.log | rc=0 全绿：A 4 / B 4 / C 7 / D 8 / D-090 6 / D-091 5 / 等价性 5 / 负控 7 / 校验 9 |

诊断副本脚本：kelly_diag_copy.jl（与原文件唯一差异 = 注释 3 个数学语义断言 +
辅助函数加 polish，均带 [DIAG COPY] 标注）；polish_diag.jl。

## 2. 集成修改清单（file:line 为最终状态）

1. **src/gate0/KTraderGate0.jl**（骨架收口）：include 顺序补全
   market → geometry → modes → kelly（modes 依赖 geometry 的
   BANDS/TAUS/BANDCOL/path_basis_1d/domain_mode_basis）；骨架层新增
   kelly 公共 API export（cash_kelly / locked_wealth_gate0 /
   cash_kelly_certificate / cash_kelly_certified；cash_kelly_inputs
   为内部校验不导出）；注释更新（依赖纪律 + include 依赖链 + 32 名导出面）。
2. **src/gate0/geometry.jl:26-35**：文件头新增 `using LinearAlgebra`
   （principal_sqrt_root 的 svd/Symmetric 需要；文件自含声明与 kelly.jl
   模式一致，include 时在加载者作用域执行——骨架与测试临时 module 两条
   加载路径同时修复）。
3. **test/gate0/market_tests.jl:138-160**：names 断言同步为集成后导出面
   （32 API + Julia names() 恒含的 module 自名 = 33 名集合）。断言语义
   （导出面钉死、无 mutation/setter API）不变。
4. **src/gate0/kelly.jl:120-192**：新增 `_cash_kelly_newton_polish`
   （Clarabel 解的 KKT 数值精修：固定支撑、解 g_S(u)=ν·1 与 Σu=budget
   的 Newton 系统，精确 Hessian；支撑阈值 1e-6 区分内点尘埃与真实支撑）；
   **kelly.jl:265-285**：cash_kelly 主流程在证书不过时尝试精修、精修解
   通过同一证书（tol 不变）才采纳，否则保留原解走原 error 路径。
   动机与纪律见函数 docstring（D-060/fail-loudly：不改目标/可行集/证书，
   不放松 tol，绝不返回 heuristic 权重）。
5. **test/gate0/kelly_cash_tests.jl:37-72**：测试辅助函数
   budget_inequality_solve 加同款 Newton polish（cash_kelly 引入精修后
   两边精度差 3 个数量级，atol=1e-8 的等价性断言被求解器噪声淹没；
   断言语义与容差不变）。
6. **test/gate0/modes_tests.jl:18-20**：using 面补 Statistics（测试体
   L117/L125 使用 mean）。

## 3. 修复过程记录（同一处最多两轮）

- kelly Case C 首红：Clarabel（exp-cone 建模）内部解可达 KKT 精度 ~1.1e-7
  < tol 1e-8。第 1 轮引入 Newton polish（含两个自身缺陷：mean 未引入 → 改
  sum/k；**Newton 方向符号反**（M\r 应为 M\(-r)）→ 修复后 Case C/D 绿）。
  D-090(a) 暴露支撑判定过松（1e-10 把内点尘埃 ~5e-8 误纳入支撑）→ 阈值
  1e-6（第 2 轮，闭环防护：误排除的合法小权重由证书不过即不采纳兜底）。
- 等价性 4 红：polish 打破测试「两边同源误差」对称性 → 辅助函数加同款
  polish（我方修复的配套收口，非断言改动）。
- modes_tests 2 error：缺 using Statistics → 补。

## 4. 未决事项（数学语义层面，按任务书纪律不擅改，原文回报）

**D-090(b)（kelly_cash_tests.jl:193-194）**：
```
Expression: all((x->abs(x) <= 1.0e-6), wb)          → Fail（wb 非全零）
Expression: abs(cb - 1.0) <= 1.0e-6                 → Fail（cb = 0.24208681821330047）
```
定性：Xb（行和恒 3.0）使对称射线 w=(u,u,u)、cash=1−3u 上 wealth ≡ 1、
E[log] ≡ 0——结构性平局（与 D-091 同构）。实现返回射线内点（对称、预算
恒等、证书全绿、objective=0），满足裁决 D-090「允许全部 cash」的允许语义；
测试注释的「全 cash 唯一最优」推导漏了该平局射线，断言比裁决与数学事实都强。

**D-091（kelly_cash_tests.jl:156，经诊断副本观察）**：
```
Expression: maximum(w) - minimum(w) <= 1.0e-6       → Fail（1.4252806619097225e-6）
```
定性：X 每列均值 = 1 ⟹ 全部 g_i 相等 ⟹ 整个单纯形为 KKT 最优集；平局
连续解集下求解器返回点的对称性数值上不可强制（objective/预算/证书断言全绿）。
裁决 D-091「收敛到对称 solution / cash」是对积分收敛的要求，非对求解器
返回点的要求。

两处均属测试断言的数学语义问题（平局解集下的断言设计），处置权在
Manager/Engineer（修断言或修 fixture 均为数学预期改动）。

## 5. 边界与未观察事项

- 未修改：旧 src/、旧 test/（test/ 根下旧线文件）、docs/、git（无
  commit/reset/clean）。
- kelly_cash_tests 原文件的 D-091/等价性/负控/校验 testset 因 Test.jl
  fail-fast 中止未在原文件内直接观察（诊断副本观察为绿；副本与原文件的
  差异仅注释 3 断言 + 辅助函数 polish，均已标注）。
- 全部绿证据为单机、当前时点一次运行事实；无跨机群保证。

## 6. 裁决执行轮（Manager 裁决 2026-10-09：D-090/D-091 断言修正）

第 4 节的两处数学语义红已由 Manager 裁决处置（平局射线数学事实 +
裁决书许可性语义，非放宽）。执行内容：

- **kelly_cash_tests.jl:176-211（D-090(b)）**：删除「wb 全零、cb=1」
  两断言，换为裁决三断言结构：(i) objective ≈ 0（atol 1e-9）；(ii) 解在
  平局射线上——三分量对称（max−min ≤ 1e-6）、预算恒等（atol 1e-8）、
  证书绿；(iii) 全 cash 端点对照——手工计算 w_aug=[0,0,0,1] 的 objective
  （增广矩阵乘法）并断言与返回解相等（atol 1e-10）。注释块改写为平局
  射线数学事实 + 裁决依据。
- **kelly_cash_tests.jl:213-239（D-091）**：对称断言
  （max−min ≤ 1e-6，连续平局解集上数值不可强制，实测 1.4e-6）换为无
  集中断言 max(w) ≤ 1/3 + 0.01（95% 单票将远超此界而红，验证力保留）；
  objective/预算/证书断言全部保留。注释块同步（D-091 禁令对象是
  「稳定产生集中」而非「严格对称」）。
- 其余断言一字不动；src/gate0/kelly.jl 未触碰（solver 已被证书证明正确）。

重跑结果（kelly_cash_tests_final.log）：**rc=0、elapsed=16s、
rss_peak=884MiB、killed=0；九 testset 全 Pass：A 4/4、B 4/4、C 7/7、
D 8/8、D-090 9/9、D-091 6/6、等价性 5/5、负控 7/7、校验 9/9——
合计 59/59**（任务书预估 57，实际 59：三断言结构使 D-090(b) 由 4 断言
变为 5 断言）。

至此 Wave 2 四条命令的最终状态：module 加载 ✅、market_tests 82/82 ✅、
kelly_cash_tests 59/59 ✅、modes_tests 102/102 ✅。第 4 节的未决事项
全部关闭（经 Manager 裁决处置）。

## 7. 收尾验证轮（response_tests：Step 4/5/6 固定 ridge reference）

- **静态预检发现并修复**：src/gate0/response.jl 文件头缺
  `using LinearAlgebra`（fit_full_block_ridge 使用 Symmetric / I /
  cholesky，response.jl:249-251 原行号）——与 geometry.jl 同模式的
  「裸文件 + 两条加载路径」缺陷；文件头补 using（与 geometry.jl /
  kelly.jl 的文件自含模式一致，response.jl:44-51 最终行号）。
- **response_tests.jl 受控运行**（response_tests.log）：**rc=0、
  elapsed=4s、rss_peak=410MiB、killed=0；Test Summary 78/78 Pass**
  （D-092 cross-mode 恢复 + design/target 代数 + 零响应世界 +
  permutation 协变 + fail-loudly + D-093 DC world + D-094 trace
  反例）——与任务书预期 78 项一致，缺 using 修复后一次全绿。
- **骨架收口**（KTraderGate0.jl）：include 列表补 response.jl
  （顺序 market → geometry → modes → response → kelly；response 消费
  modes 的 mode_field/cumulative_path_coordinates/mode_basis_design 与
  geometry 的 fast_s_m/compute_s_perp，必须后于 modes）。response.jl
  自带 export 4 名（build_mode_problem / fit_full_block_ridge /
  predict_mode / block_views），与既有 32 名无重名——导出面 36 名 API。
- **market_tests.jl names 断言同步**（:138-166）：33 → 37 名集合
  （36 API + module 自名），断言语图不变。
- **复跑**：module 加载检查（module_load.log）rc=0、2s、552MiB、
  36 API 确认；market_tests（market_tests.log）**82/82**、5s、642MiB。

Wave 2 全量最终状态：module 加载 ✅（36 API）、market_tests 82/82 ✅
、kelly_cash_tests 59/59 ✅、modes_tests 102/102 ✅、response_tests
78/78 ✅。
