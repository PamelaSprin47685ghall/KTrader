# 多日运行现状核查报告（MULTIDAY-RUN-STATUS-1）

**状态：只读调查报告（2026-10-10）。未运行任何命令、未修改 `src/`、`test/` 或任何既有文件；本文为新增文档。**
**任务对应：A. 现状核查——（1）「评审/评审确认的 GAP」文本痕迹检索；（2）上轮未尽项的权威清单。**
**配套文档：`docs/MULTIDAY_COST_OPTIMIZATION.md`（B 部分：多日运行成本优化设计）。**

---

## 0. 方法与边界

- 检索范围：`AGENTS.md`（最新段=第14任；无第15任及以后）、`docs/`、`dev/`、`archive/`（含被 git 忽略的 `archive/evidence/**`）。
- 检索关键词：`GAP`、`评审`、`review`、`findings`、`下一层`、`未尽`、`待裁决`，以及组合词「评审确认」「评审遗留」「gap 清单」「缺口清单」。
- 手段：只读 grep（含带词边界的 `\bGAP\b`）+ 目录/文件阅读（`dev/gate0_static_audit_2026.md`、`docs/GATE0_EXIT_CHECKLIST.md`、`archive/evidence/` 各 summary、`src/gate0` 与 `test/gate0` 相关段）。
- 本报告只记录「文本是否存在、内容是什么」；不宣布任何运行结论，不改写任何既有登记。

---

## 1. 「评审/评审确认的 GAP」文本痕迹检索结果

### 1.1 结论

**未发现**以「评审确认的 GAP」或类似名目为主题的专题落盘文本。检索证据：

1. `\bGAP\b` 在 `archive/evidence/` 全目录零命中（排除 `_WEAKINFO_GAP` 等代码标识符后）；「评审确认」「gap 清单」「缺口清单」零命中。
2. `AGENTS.md` 最新段（第14任）不含 `GAP`/`评审`/`findings` 字样；其未尽事项以「已知边界」与「修复清单」形式记载。
3. `docs/` 中关键词命中均为设计文档的「未决/待裁决」节或历史引用（如 `GATE0_RESPONSE_POSTERIOR.md` §6「二选一待裁决」、`POSTERIOR_DEFINITION.md` §7、`NUMERICAL_INTEGRATION_SPEC.md` §7），不是一份独立的评审 GAP 清单。
4. `AGENTS.md` 第11任段已明确记载历史事实：评审问题清单文本不在本机可读位置（`git refs/localspace/review/ws_*` 五个快照无清单文本；全仓关键词与外围路径双路检索零命中）。该历史结论在本轮复核中仍然成立。
5. `dev/` 目录检索命中均为历史评审注记的具体引用（`dev/m1_artifact_replay.jl:37` 的 "4th-review accepted"、`dev/m1_typed_replay.md:587-589` 的第6任评审注记、`dev/evidence/final_2_0_0_20261009/README.pre-final.md:629` 的 "three defects are static findings"——指 2.0-RC 历史线）；无独立 GAP 清单文本。

### 1.2 最接近的等价载体（按接近程度排序）

| 载体 | 内容 | 性质 |
|---|---|---|
| `dev/gate0_static_audit_2026.md` §4 | 「未达标 / 矛盾 / 张力清单」9 条（2 条注释滞后、1 条文档滞后、若干测试覆盖注记与语义边界） | 静态审计；最接近「评审确认的 GAP」的落盘文本 |
| `docs/GATE0_EXIT_CHECKLIST.md` §交接登记 a–g + §已知边界 | owner 裁决域待办 7 项 + 已知边界 5 条 | 权威登记（Gate-0 关闭时点） |
| 各证据 summary 的「未决/待裁决」节 | `gate0_reallimit_firstday_20261010`（2 条）、`gate0_merged_verify8_20261010`（3 条）、`gate0_f6_unblock_20261010`（结论节） | 轮次运行观察 |
| `AGENTS.md` 第13/14任段 | 下一层已登记工作、修复清单、已知边界 | 规范历史段 |

### 1.3 本轮发现的两处文档滞后（均为「事实登记」级别，非行为缺陷）

1. `README.md` 曾引「最近一次受控运行报告 898/913 绿」（原 L34/L103/L150）；当前口径为 **984/984 全绿**（`archive/evidence/gate0_final_verify_20261010/`、AGENTS.md 第14任段）。第14任登记 `bin/backtest.jl` 头部注记已同步 984；README 三处已于 2026-10-10 本轮同步（本条保留为修复前时点记录）。
2. `dev/gate0_static_audit_2026.md` §4 的 #1/#2/#3 三条随后已闭合（见第 3 节），该审计文件本身为时点快照、不再更新属预期。

---

## 2. 未闭合义务的权威清单

以 `docs/GATE0_EXIT_CHECKLIST.md`（交接登记 + 已知边界）、`AGENTS.md` 第13/14任段、各 summary 未决节为权威来源；与任务点名的五项逐项对照：

### 2.1 首日 fit 性能 —— **未闭合**

- 事实：真数据晚期窗口（t=14305）首日 full-fit 在 55s 被切（rc=124、elapsed=45s、RSS 1703MiB；`gate0_merged_verify8_20261010/C1_a2_true.log`）；BLAS=1 重跑同样 45s 被切。单点成本已达标（2.52ms@BLAS1 vs 15.6ms 优化前），剩余份额=自适应 cells 规模（同 summary §C）。
- **本轮新发现（结构性）**：`prequential` 对全部历史行逐行 fit（`src/gate0/oof.jl:386`，行数 ≈ t−WARMUP−1）；晚期窗口 ≈1.4 万行，仅粗扫描一项即 ≈4.3 小时（按 2.5ms/点保守外推）。因此 F6 晚期窗口「单命令 1 日 ≤55s」在现行数学下不可达——须 owner 裁决。详见配套设计文档 §1.4/§7-U1。
- 归属：成本设计与实验（本批）→ 实现（受控施工）→ F6 范围裁决（owner）。

### 2.2 60-day 批量 —— **未闭合**

- 事实：段 1–5 绿（50/60 决策日）；段 6（t=327–336）「未在护栏内运行」（`GATE0_EXIT_CHECKLIST.md` D-089 阶梯节）。后续收尾注记称「两阻塞已修复落地」，但**段 6 与 60-day 全链在修复后没有实际运行证据**；「60-day 为批量负载口径，非单命令目标」。
- 归属：配套设计文档的 E7 端到端实验（DevOps）→ 60-day 分批调度（owner）。

### 2.3 F6 完整运行 —— **未闭合**

- 事实：F6 冒烟（`gate0_f6_unblock_20261010`）取得启动路径观察；首日 posterior 2D 求积在 17 cells 出现 NaN（后由「posterior m 基准修复」按第14任段落地）；20 天窗口未执行。
- 现状：真数据首日 full-fit 仍超时（2.1）；prequential 1.4 万行（2.1 新发现）。**完整链从未跑通**。
- 归属：2.1 的裁决链。

### 2.4 BLAS 线程配置 —— **未裁决**

- 事实：默认 6 线程对小 GEMM 是 ~3× 负优化（1 线程 2.5ms）；`gate0_merged_verify8` 明确为「运行时数值工程配置决策（不改 src），需裁决生产在何处设定」。
- 归属：owner 裁决（U2）；零数学风险。

### 2.5 span≈1e17+ 与人工双方向 fixture —— **已知边界（未消除）**

- 事实：`GATE0_EXIT_CHECKLIST.md` 已知边界节；`test/gate0/kelly_cash_tests.jl:390` 附近登记行缩放后仍 SLOW_PROGRESS 的极值算例。
- 归属：数值工程边界；如需处理另立。

### 2.6 其余登记项（同权威来源）

| # | 事项 | 状态/归属 |
|---|---|---|
| 1 | T4 多日跨语义缺口（裁决 9）——D-013 admission 与 D-036 fold propriety 语义不一致 | 未闭合；SPEC 审查项（修复需数学对象层变更） |
| 2 | 真实数据桥接（Bars→MarketFacts 转换器 + panel provenance 确认） | 未闭合；真实数据短窗口优先 |
| 3 | 501-day 分批执行 | 机制已建立、未执行 |
| 4 | μ 通道 RQMC 化（裁决 6 defer） | 未执行 |
| 5 | 新发布凭据链生成 | 归重新发布流程 |
| 6 | quadrature 负载敏感性（45/45/34s 同字节三裁） | 已知边界；记录 load |
| 7 | 跨进程位级抖动疑点（A2/A2b 行为差异未复现） | 未复现、未结案 |
| 8 | quadrature 分层时序（「A 过 B 红继续翻倍」） | 已知未测项（`quadrature_tests.jl` 自标） |
| 9 | 函数级默认（1e-3/1e-5）vs 生产传参（1e-4/1e-6） | 供审查（第二调用方需复核） |
| 10 | `require_full_rank` 默认 false（诊断语义） | 供审查（生产 driver 显式 true） |
| 11 | `u_span=5` | SPEC 复审素材（docstring 自证非容差放宽） |
| 12 | README 898/913 口径滞后 | 已修复（2026-10-10 同步 984；本报告写作时为修复前时点） |

---

## 3. 静态审计 9 条的闭合情况对照（`dev/gate0_static_audit_2026.md` §4）

| # | 审计条目 | 现状（本轮静态复核） |
|---|---|---|
| 1 | backtest.jl docstring 陈旧回落文案 | **已闭合**：`src/gate0/backtest.jl:244-246` 现为「不存在……production fallback」的修正表述 |
| 2 | GATE0_EXIT_CHECKLIST 裁决 7 滞后登记 | **已闭合**：清单 L184 已补「该弱信息特例机器已整体删除」后续注记 |
| 3 | driver_tests.jl:102 注释引用旧值 | **已闭合**：现注释为「实际 1e-6（P0-7 收口）」 |
| 4 | P0-8 prior 无直接公式断言 | **部分闭合**：第13任新增 11 条表达式级断言（`posterior_tests.jl`）；`gate0_final_verify` 口径 posterior 60 项含该批（运行证据） |
| 5 | quadrature 分层时序未测项 | 开放（测试自标；见 2.6-#8） |
| 6 | 函数级默认 vs 生产传参 | 开放（供审查；2.6-#9） |
| 7 | `require_full_rank` 默认 false | 开放（供审查；2.6-#10） |
| 8 | `u_span=5` | 开放（复审素材；2.6-#11） |
| 9 | predictive/backtest 未全文核对 | 边界声明（本轮仍未全文核对） |

> 注：#1–#3 的「已闭合」依据为当前工作树静态文本；其修正动作在第13任段登记为「文档与注释修正（静态同步；未运行）」——闭合的是文本一致性，不涉及数值行为。

---

## 4. 与配套设计文档的衔接

- 2.1/2.4/2.7 的成本事实与候选优化（高阶规则、warm start、粗扫描缩减、BLAS 配置）见 `docs/MULTIDAY_COST_OPTIMIZATION.md`。
- 2.2/2.3 的端到端闭合路径（E7）与 F6 范围裁决（U1）同见该文档第 6/7 节。

---

## 5. 边界

- 本文为只读调查；所有「已闭合/未闭合」均为静态文本层面的判定，不替代任何运行证据。
- 行号以本次阅读时点的工作树为准，可能随后续编辑漂移。
- 未运行任何命令；未修改 `src/`、`test/`、`archive/` 既有文件；本文为唯一新增（配套设计文档亦为新增）。
