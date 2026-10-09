# OLD → CURRENT 语义差异报告（Gate 0 重开 · Deliverable 2）

**状态：分析文档（非规范、非发布凭据）**
**基准：旧侧 = `1d9b7ad`（Path Kelly V0 baseline，2026-10-06 12:57:22 +0800）；新侧 = 当前工作树（release 基线 `fc54ce48`，2026-10-09 12:14:17 +0800）**
**性质：纯静态语义比对。本文不含任何收益 / PnL / Sharpe 数字；不运行任何命令；不修改源码、测试与任何既有证据文件。**
**证据来源：`dev/evidence/manager13/` 侦查材料（`defect_markers.txt`、`src_evolution.txt`、`old_src_snapshots.txt`）与三份快照目录；行号引用均指各文件的静态文本。**

---

## 0. 范围声明与基准选择

### 0.1 为什么以 `1d9b7ad` 作为旧侧

【事实】`dev/evidence/manager13/defect_markers.txt`（§A–§F）以 pickaxe 与逐提交存在性矩阵证明：答辩点名的判别性缺陷标志——`shrink_drift`、ARD（MacKay 分组 ARD）、BF 家族（`BF_MIN` / `bayes_ard` 稀疏门控）、selection 自选择缺陷承认注释——的**唯一宿主**是 `1d9b7ad`；从 `21017b9`（V1.0）起全部为零命中（`defect_markers.txt:309-316, 518-527, 562-564`）。`old_src_snapshots.txt:108-121` 的判定 1 与之一致：它同时是时间序上最早的实质提交，也是这些标志的最后（唯一）宿主。

因此，`1d9b7ad` 是"缺陷标志意义上"可被证据锚定的旧侧端点，而不是凭印象挑选的版本。

### 0.2 为什么不是字面的「最后胜出版本」

【事实】`old_src_snapshots.txt:108-131` 明确记录：本仓库 git 历史中**不存在**具名为「早期胜出版本 / winner」的单一 commit；答辩报告原文不在本机可读范围（`old_src_snapshots.txt:124-128` 判定 3）。`defect_markers.txt:552-559`（F4）进一步说明：'BF' 的字面 token 在所有提交的源码中零命中（仅二进制 fixture `t14294_conditioned_fold3.jls` 字节巧合），若答辩所指为 BF 稀疏门控语义，则其最后宿主同为 `1d9b7ad`；两种解释都已记录、不作裁决。

因此本文严格使用"证据最强的旧侧基准"这一口径。中间对照 `21017b9`（V1.0，修复发生点）与 `602b897`（性能线终点/发布父提交）在必要处引用为**对照面**，但它们不是旧侧的替代品。

### 0.3 不确定性（必须随结论携带）

1. 答辩报告原文不可读；关键词清单来自任务转述，未逐字核对。若报告点名了其他标志，需以其证据重做对照（`old_src_snapshots.txt:125-127`）。
2. `shrink_drift` 在 V0 中是一个具体函数（模型内收缩估计），其是否等同于报告所称"缺陷"需报告原文确认；本文只给"该标志在此提交存在/不存在"的事实（`old_src_snapshots.txt:128-129`）。
3. 所有行号来自转储快照的静态文本；`old_src_snapshots.txt:65-74` 的 `diff -r` 保真验证与逐文件 SHA-256（`:76-106`）证明转储内容与 git 对象逐内容一致，但 git blob 名是 SHA-1、不可与 SHA-256 直接比较（`:70-72`）。
4. 本报告是**语义分析**：分类（修 bug / 理论变化 / 数值变化 / 结构重构）为工程判断，凡推断均显式标注【推断】；凡规范裁决均回引 SPEC 章节。本文不宣布任何收益结论。

---

## 1. 十项数学对象的逐项 diff

分类词约定：
- **修 bug**：旧实现与项目自身声明的数学/契约不一致，且移除/修复在规范或注释中有据；
- **理论变化**：模型对象、先验、法律或信息集语义改变（需要 SPEC 审查确认归属）；
- **数值变化**：同一数学对象的离散化/算法/常数改变；
- **结构重构**：数学对象不变或近不变，但 ownership / 表示 / 调用路径改变；
- **【需 SPEC 审查】**：无法在上述四类内唯一归类，或分类依赖于尚不可读的规范判断。

### 1.1 Conditional mean

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:503-507`：`conditional_mean = drift.mean .+ response_mean`；`response_mean` 对 geometry draws 取平均（`model.jl:504`）；`predict` 中每场景抽 drift 层级（`model.jl:535-536`） | `predict.jl:480-486`：`mu_rel_proj = mu_rel .- mean(mu_rel)`；`mu_norm = mu_m .* e0_now .+ mu_rel_proj`；`mu_asset_act = mu_norm .* s1`；嵌入完整 universe（`L485-486`）；`predict.jl:463-466` 由 `predictive_moments` 给出 `mu_m, mu_rel` |
| 语义 | 条件均值 = 跨资产层级漂移后验均值（`shrink_drift`，`model.jl:317-339`，含 Newey–West 长期方差 SE，`model.jl:288-305`）+ 对 8 个 bootstrap 几何的响应均值 | 条件均值 = macro ridge 均值（`response.jl:1146-1151`）+ trace 条件化后的 relative 均值（`response.jl:1200-1201`）；无 drift 项；relative 分量在活资产上零和 |
| 分类 | **理论变化**（层级 drift 被移除；`drift` 于 V1.0 即已消失：`old_src_21017b9/src/predict.jl:165-166` 只余 `mu_m e0 + Phi_perp mu_rel`）。**修 bug 成分**（V0 的 in-sample 拟合均值参与 `resid`，见 1.6/3.3） |

【推断】drift 项的存废是 V0→当前对**低信噪比横截面**影响最大的单点：V0 把每资产均值向共同 `μ0` 收缩（`model.jl:329-334`），当前完全不表达该先验；同时当前 relative 投影显式去均值（`predict.jl:480`），而 V0 的 `response_mean` 是 mode 空间平均，其零和性依赖 neutrality 约束。此项改变预测的横截面形状，最可能解释策略行为变化。是否属理论推进需 SPEC 对 SPEC §34 的对照（当前定义与 SPEC §34 一致）。

### 1.2 Response operator

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:374-470`（`fit_response`）：几何窗口内每 band 的协方差特征向量 `Φ=natural_modes(...)`（`model.jl:164, 192-196`）；每条通道是 `φ_k ⊗ c_k(t)`（`model.jl:374-390`）；系数块 `Cb`（`model.jl:433-438`）；`bayes_ard` 求解 | `response.jl:1124-1209`（`fit_response_operator`）：macro 为 ridge/EB；relative 为 `P=14N` 维 design 上的谱表示（`response.jl:150-174`）+ trace 条件化（`L1200-1208`）；design 由 `fill_design_matrix!`（`response.jl:55-75`）在**固定 Helmert gauge** 相对空间上构造（`numerics.jl:368-380` `relative_gauge`） |
| 语义 | 数据依赖的每 band 模式（sign / 简并旋转不可控），每个 mode 一对 `(a_k,b_k)`；`θ_posterior` 报告 `ρ,θ,R1,R2`（`model.jl:472-500`） | 固定数值 gauge；"自然结构"由算子谱解释（SPEC §14 的 0.95 决策）；`ResponseOperator` 持有 `G_c_mean, covariance, Sigma_rel, inv_M_constraint`（`response.jl:92-104`） |
| 分类 | **结构重构**（表示从 eigenmode 坐标到固定 gauge 坐标）+ **理论变化**（mode identity / sign / 简并问题被消除；V0 的 `MODE_STABILITY` 过滤 `model.jl:207-218` 与 `GEOM_DRAWS` 在 HEAD 已不存在）。V1.0 是过渡：`old_src_21017b9/src/predict.jl:100-105` 仍用 `relative_modes` + `Phi_draws` |

### 1.3 Structural neutrality

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:408-412`：`Cn` 只在非 macro mode 上置 1，`Z=nullspace(Cn)`；`bayes_ard` 以 `Z` 参数化 `β=Zγ`，neutrality **在先验子空间上**（`model.jl:16-17, 222-232`） | `response.jl:131-148`（`condition_trace_neutrality`）：由 `constraint_moments`（`L109-130`）在 `CΩCᵀ`（14×14）上求解，`G_c = G - Ω Cᵀ (CΩCᵀ)⁻¹ C G`；均值与协方差同时条件化（`response.jl:1192-1208`） |
| 中间对照 V1.0 | `old_src_21017b9/src/response.jl:172-196`：对 `G_raw` 的每个 band **减去对角均值**（mean-only）；`predict.jl:250-265` 对每个样本再减均值。静态代码中**没有**对 covariance 的约束条件化 | — |
| 分类 | **【需 SPEC 审查】**：V0 的"先验支持上约束"在数学上合法（SPEC §22 允许的路径之一的前身），V1.0 的 mean-only 投影与 SPEC §22 明确禁止列表中的"只把 mean 的 trace 减掉"字面相符【推断，静态行号支撑】；当前实现即 SPEC §22/§25 描述的 constrained mean/covariance。V0→当前是**理论深化**（先验约束 → 精确条件化），但 V1.0 中间态相对 V0 是一次**局部退步**，值得在 SPEC 历史审计中单列 |

### 1.4 Parameter prior

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:222-232`：每 mode 一个 ARD 精度 `α_k`，`(a_k,b_k)~N(0, α_k⁻¹ I₂)`；`ALPHA_MIN/ALPHA_MAX = 1e-6/1e7`（`model.jl:233`） | `response.jl:1160-1185` + SPEC §19：`G|Σ,α ~ MN(0, Σ, α⁻¹I)`；`EB_ALPHA_MIN/MAX = 1e-4/1e6`（`response.jl:207-208`） |
| 语义 | 每通道独立精度（分层稀疏先验）；`α_k→∞` 表示模式关闭 | 全局标量 `α_rel`（macro 另有 `α_m`）；不确定性由 `Σ`（relative innovation covariance）与 `V=(XᵀX+αI)⁻¹` 表达 |
| 分类 | **理论变化**（每模 ARD 分层 → 全局标量 Matrix-Normal 先验；稀疏性从"BF 门控开关"变为连续证据） + **数值变化**（域界常数 1e-6/1e7 → 1e-4/1e6，SPEC §24 限定为数值域界） |

### 1.5 Hyperparameter treatment

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `bayes_ard` 内循环（`model.jl:248-286`，500 iters / tol 1e-4）→ 对每个 live mode 计算 `logev` 增益 → `weak = gain .< log(BF_MIN*M)` → 关闭并重拟合（`L274-281`） | `maximize_logalpha`（`response.jl:212-273`）：33 节点对数网格 + bracket 二分 + secant + 确定性 tie-break；`bounded_alpha_certificate`（`L276-282`）与 `covariance_certificate`（`L297-347`）双证书；`optimize_matrix_normal_eb` / `optimize_conditioned_eb` 以 `certificate.valid || error` fail-loud（`response.jl:378, 996-999, 1119-1121`） |
| 中间对照 V1.0 | `old_src_21017b9/src/response.jl:72-113`：`optimize_evidence_alpha` 固定点 15 步、clamp `1e-4..1e6`、**无 stationarity 证书** | — |
| 分类 | **理论深化**（证书驱动，SPEC §25：stationarity certificate 必须存在） + **数值变化**（迭代策略、域界） + **修 bug 候选**（V1.0 的"line search 失败即返回"类做法被 SPEC §25 禁止；当前在无证书时抛错）。V0 的 BF 稀疏门控整体移除：**理论变化**（先验结构改变） |

### 1.6 Innovation law

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `modecov.jl:1-230`：13 点网格 `DGRID`（`L1`）；每模自己的 `d_k`（`L3-7, L67-92`）；Horn 平行分析 + tail 规则选模（`L121-160`）；idiosyncratic 块对角投影（`L164-193`）；`model.jl:527-548` 每场景抽 `mode_vol_model` 或 iid 残差行 | `predict.jl:76-119`：7 点 `DGRID_V1` + `DELTA_D_V1` 权重（`L10, L14, L94`）；FFT 精确卷积；残差来源为 OOF residual（下述 1.7）；`generate_scenarios_v1` 用单标量 `scale=sqrt(v_forecasts[d]/v_bootstrap)`（`L615`）+ 行 bootstrap/own-row（`L577-604`） |
| 语义 | 多模 + 异质 + 每模 FIGARCH 型长期记忆 + 尾部规则；尺度是模自身的 `σ` | macro 标量分数后验 + 绝对尺度归一（SPEC §31–§32）；relative 分量的风险由 predictive covariance 表达（`response.jl:1221-1229`） |
| 分类 | **理论变化**（innovation 法律从多模异质降为 macro 标量 + covariance；是否等价于 SPEC 的 KISS 版需要 SPEC §30 对照——SPEC 明言这是 V1 的 KISS 版非 Markov 风险，不是最终无限维 law） + **数值变化**（FFT vs `DSP.conv`；13→7 点网格且引入 `Δd` 权重，SPEC §31 明确要求含测度） |

【事实】V1.0 的 `mode_vol_factors`（`old_src_21017b9/src/predict.jl:205`）在 `generate_scenarios_v1` 中构造后**未被消费**：`predict.jl:237-250` 的 `eps_shock` 直接取 `res_history` 行、无缩放，最终 `scenarios_log = mu_s .+ eps_shock`（`L250`）。这是中间对照面的静态观察（未接线字段），当前实现有显式绝对尺度（`predict.jl:615`）。

### 1.7 Posterior uncertainty

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | 几何 bootstrap：`model.jl:402-403`（`bs`）→ 每个 draw 有自己 response posterior（`model.jl:465-469`，`Draw`）→ `predict` 均匀混合（`model.jl:523, 537`）；drift 层级后验（`mu0sd/sd`，`model.jl:536`） | 决策时精确边缘化：`predictive_moments`（`response.jl:1211-1229`）返回 `mu_m/var_m/mu_rel/L_rel`；`L_rel` 为 predictive covariance 的 PSD 主平方根（`numerics.jl:452-460`；`predict.jl:570-572`）；scenario 只抽 relative 因子 + macro 标量（`predict.jl:555-563, 610-622`） |
| 语义 | 抽整个 `G`（每 bootstrap 几何一个）；几何子空间不确定性显式传播 | 不抽整张 `G`；`Gx` 的边缘 + constrained covariance 精确表达（SPEC §33：这是精确边缘化，不是降阶近似）；**geometry draws 不再存在**（V1.0 的 `Phi_draws` 未出现在当前 `V1Model`，`predict.jl:16-36`） |
| 分类 | **理论深化**（SPEC §33 精确边缘化） + **理论变化/【需 SPEC 审查】**：geometry subspace uncertainty 从"显式 bootstrap 混合"退场，由固定 gauge + scalar ruler 的确定性几何替代。SPEC §14 论证了固定 gauge 消除了 sign/简并等跟踪问题，但"移除几何不确定性传播"是否在 SPEC §26 的 epistemic uncertainty 语义下被完全覆盖，属规范判断，本文不裁决 |

### 1.8 Scenario integration

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:521-548`：纯 IID MC（默认 S=1000）；**moment matching**（`L542-546`：`out .+= conditional_mean' .- mean(out, dims=1)`） | `predict.jl:546-625`：IID 或嵌套 randomized Halton（`ScenarioQuadrature`，`L507-531`）；`adaptive_scenario_weights` 用 `kelly_certificate` 的 objective gap + 权重 L1 收敛（`L627-653`）；**无 moment matching**（`predict.jl:533` docstring "No post-hoc mean matching."；`defect_markers.txt:457, 521` 记录该否定声明由 `c109d9f` 引入） |
| 语义 | 强制场景样本一阶矩等于解析预测均值（V0 注释自称 numerics-only） | 由 quadrature refinement + Kelly certificate 控制采样误差（SPEC §37：数值 refinement 依据 `‖w_{2S}-w_S‖₁` 与目标证书，不得用回测 Sharpe） |
| 分类 | **理论变化/认识论**（moment matching 移除；收敛改由证书证明） + **数值变化**（Halton） |

【事实】V1.0 亦有 moment matching（`old_src_21017b9/src/predict.jl:253-254`）。移除时间点为 `c109d9f`（2026-10-06 19:00，`defect_markers.txt:19-20, 448-457, 521`）。

### 1.9 Kelly feasible set

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:553-568`：Clarabel `max sum(log(X f + base))/S`，`f≥0, sum f = budget`；结果 `clamp.(..., 0, budget)` 后 `out .* (budget/sum(out))`；`allocate`（`L600-621`）中 locked 为 `held .* .!free`，未建模 locked 以常数财富 `fill(sum(locked[.!mask]), S)` 参与 base（`L614`） | `kelly.jl:6-134`：`kelly_inputs` 校验（`L6-15`：gross return 必须正有限；base 非负有限）；`fast_kelly_solver` + `kelly_certificate`（`L21-37`，concavity gap / KKT / feasibility），失败回退 `clarabel_kelly_solver`（`L39-58`，同目标同证书）；`locked_wealth`（`L137-148`）对每个 held 资产要求有预测律，缺则 error（`L143`） |
| 语义 | 已持有且不在 scenarios 中的资产 = 常数财富（等价于把其风险当 cash）；无 fast solver、无证书 | free 资产 = tradable ∧ active；locked 只乘真实持有的列；`0*NaN` 不再进入 base；同一 log-Kelly 目标的两个 solver 都须通过证书 |
| 分类 | **修 bug**（SPEC §38.1：locked 风险不得当 cash；`0*NaN` 防护与 fail-loud）+ **结构重构**（certificate 化；SPEC §39-§40）。归一化除零/空集边界为数值修复 |

### 1.10 Universe semantics

| | 旧版（V0） | 当前 |
|---|---|---|
| 实现 | `model.jl:570-589`：`firstrows`；`eligible(f,T)=(T-f+1)>=MINROWS`（`L581-582`）；`MINROWS=WARMUP+GEOM_MIN+REG_MIN=256+256+64=576`（`L56-57`）；`scenarios` 只在 eligible 列上生成；`tradable` 默认 `isfinite.(adj[end,:])`（`L630`） | `predict.jl:125-135`：`active_universe_indices` = 至少一条有效日收益；`_prepare_v1` 对全部 active 列建模；`alive_now` = 决策日观测 mask（`prepare.jl:131`）；`mu` 嵌回全 universe（`predict.jl:485-486`）；`kelly.jl:150-160` free = tradable ∧ active |
| 语义 | 资产须有 ≥576 行历史才进入模型空间（硬门槛）；弱证据以排除表达 | 无 per-asset 硬门槛；弱证据由 posterior 表达（`defect_markers.txt:330` 引 V0 自述"weak evidence is expressed by the posterior ... not by exclusion"是 V0 的目标，但其 `eligible` 仍是硬门槛）；全 NaN dummy 严格排除（SPEC §11/§60） |
| 分类 | **理论变化**（信息集扩大：所有有收益历史的资产进入建模） + **结构重构**（active/eligible/alive_now 语义分层）。两条路径都满足 SPEC §11（至少一条有效 return）与 §60（dummy 排除），差异在 V0 额外要求 576 行 |

---

## 2. 答辩点名缺陷的逐条核实

### 2.1 `shrink_drift`

- **V0 存在**：【事实】函数定义 `old_src_1d9b7ad/src/model.jl:325-339`（`DriftPost` 结构 `L317-323`，网格 `TAUGRID = [0.0,0.1,0.2,0.4,0.8,1.6,3.2]` `L315`）；拟合调用 `model.jl:464`；`predict` 中消费 `model.jl:535-536`；测试命中 `1d9b7ad:test/runtests.jl:285, 289`（`defect_markers.txt:322-327`）。
- **移除版本**：【事实】`defect_markers.txt:328, 519, 534-535, 562-564`：`21017b9` 起零命中；`shrink` 家族（含空格变体）唯一存活于 `1d9b7ad`。
- **当前残留**：无。【事实】对 `src/` 全目录分别检索 `shrink_drift` 与 `shrink`（含空格/连字符变体）均零命中；`defect_markers.txt:534-535` 亦确认 V0 之后全历史零命中（此前一次宽匹配中的 `discard`/`forward` 命中来自同时检索的 `ARD` 子串，与 `shrink` 无关）。当前条件均值不含 drift 项（`predict.jl:480-486`）。
- 【推断】该缺陷的语义后果（横截面收缩先验消失）与 1.1 的行为差异直接相关；但"缺陷"定性以报告原文为准（见 0.3 第 2 条）。

### 2.2 ARD / BF 门控

- **V0 存在**：【事实】`bayes_ard`（`model.jl:248-286`）：`BF_MIN=10.0`（`L246`）；`weak = gain .< log(BF_MIN*M)`（`L279`）；`α[weak] .= ALPHA_MAX`（`L281`）；调用 `model.jl:453`；注释自述"26% of macro modes on white noise"的过发现（`L238-239`）。存在性矩阵：`defect_markers.txt:329-339, 538-548`。
- **移除版本**：【事实】`defect_markers.txt:330-333, 539, 553-559`：ARD 与 BF家族的唯一宿主为 `1d9b7ad`；`21017b9` 起零命中；字面 token 'BF' 全历史零源码命中（仅二进制 fixture）。
- **当前残留**：无。【事实】对 `src/` 全目录搜索 `bayes_ard` / `BF_MIN` / `ARD`（含大小写变体）无命中；`defect_markers.txt:538-539` 确认唯一宿主为 V0。当前先验为 Matrix-Normal + 全局 α（`response.jl:1160-1185`），EB 由证书把关（`response.jl:351-380`）。

### 2.3 Selection 泄漏承认注释

- **V0 存在**：【事实】`model.jl:18-21`：自述几何窗口与响应证据"disjoint"，随后承认"Estimating the eigenvectors on the very rows they are then regressed on is selection on the response: on pure noise P(|t|>1.96)=0.68 instead of 0.05"——即缺陷自认。存在性矩阵 `defect_markers.txt:343, 480-483, 522`。
- **修复版本**：【事实】`21017b9` 头部声明"Blocked Cross-Fitting: geometry is estimated out-of-fold to strictly eliminate selection leakage"（`old_src_21017b9/src/predict.jl:1-8`；`defect_markers.txt:492-501, 510-513` 引 `predict.jl:107-137` 的分块交叉拟合）。
- **当前残留**：无缺陷面。【事实】`src/` 无 `selection` 命中；`bin/bench.jl:47-53` 的 `selection` 是基准引擎选择变量名（`defect_markers.txt:363-367, 522`）。当前 OOF 为真正的 per-fold 独立 EB（`predict.jl:405-447`，train Gram = full − fold，`L407-410`）。
- 【事实补充，非报告点名但同族】V0 的残差 `resid` 是 **in-sample** 拟合残差（`model.jl:458-462`，同一 `ref` 几何的 `β`）；V1.0 的残差预测同样使用**全数据** `resp`（`old_src_21017b9/src/predict.jl:173`，`pred_t=predict_modes(resp,...)`），其"cross-fit"只发生在坐标 `z_rel_cf`（`L112-136`），不在模型参数上。当前 SPEC §27-§29 的 OOF 隔离（每 fold 独立 EB，`predict.jl:429-447`）是真正的 out-of-fit innovation。
- 【推断】"selection 泄漏"在两代旧版中形态不同：V0 是几何/响应重叠（已自认），V1.0 是坐标交叉但模型未交叉。二者都被当前实现取代；该演进与 residual oracle 的 9 行样本保真证据（AGENTS.md 第6任段）一致。

### 2.4 可能的 zero residual / mean matching

**zero residual**：
- 【事实】V0 无 zero-fill：`defect_markers.txt:66, 524`（`zero_fill` 在 V0 零命中）。V0 的缺失残差回退是 own-row 随机行（`model.jl:509-516` `innovation`；`modecov.jl:173-178`），若某资产无 own 行会直接 `rand(1:0)` 抛错——无显式零回退。
- 【事实】V1.0 存在 `: 0.0` 零回退：`old_src_21017b9/src/predict.jl:246`（`!isempty(own_r) ? model.res_history[rand(rng, own_r), j] : 0.0`）。
- 【事实】当前为 fail-loud：`predict.jl:598`（`isempty(own) && error("active asset has no observed OOF residual")`）。
- 【事实】`zero embedding`（当前）是规范语义而非缺陷：`predict.jl:38-42` 明确"absent components are defined to be zero in the field, not in the return data"，`incremental.jl:6, 32` 说明新资产列插入精确零；`defect_markers.txt:66, 524-527`（HEAD 命中为规范 zero embedding + 测试断言）。
- 分类：**修 bug**（把 V1.0 的静默零回退替换为显式失败；SPEC §56 fail-loudly）。

**mean matching**：
- 【事实】V0 存在 moment matching（`model.jl:542-546`）；V1.0 存在（`old_src_21017b9/src/predict.jl:253-254`）；当前不存在，且 docstring 明确否定（`predict.jl:533`）；`defect_markers.txt:19-20, 448-457, 521` 记录 `c109d9f`（2026-10-06 19:00）引入"No post-hoc mean matching"。
- 分类：**理论变化/认识论**（V0/V1.0 把它标为 numerics-only，当前以 quadrature + certificate 承担同一误差控制职责，SPEC §37）。

---

## 3. 语义漂移总结

### 3.1 缺陷清除（修 bug）

1. `shrink_drift` / ARD / BF 稀疏门控整体拆除（唯一宿主 `1d9b7ad`；`21017b9` 起零命中）——先验结构与选择门控被 Matrix-Normal + 证书化 EB 取代。
2. in-sample residual → 真正 per-fold OOF（SPEC §27-§29；V0 `model.jl:458-462`，V1.0 坐标交叉而非模型交叉）。
3. locked 资产风险不得当 cash：V0 `allocate` 的常数财富（`model.jl:614`）→ 当前 `locked_wealth` fail-loud（`kelly.jl:137-148`，SPEC §38.1）；`0*NaN` 防护。
4. V1.0 的静默 `0.0` 缺失残差回退 → 显式 error（`predict.jl:598`）。
5. Kelly 无证书 → 双证书（`kelly.jl:21-37`）与同目标 Clarabel 回退（SPEC §39-§40）。

### 3.2 理论深化（认识论升级）

1. 每模 ARD 先验 → 全局 Matrix-Normal + evidence-maximised α，且 stationarity / covariance 双证书（SPEC §25）。
2. neutrality：先验子空间（V0）→ 精确 Gaussian conditioning 于均值与协方差（当前；SPEC §22）。V1.0 的 mean-only 投影作为中间态记录。
3. posterior uncertainty：抽整张几何/响应 → `Gx` 决策时边缘化 + PSD 主平方根唯一化（SPEC §26, §33；`numerics.jl:452-460`）。
4. 场景误差控制：moment matching → Halton quadrature + `kelly_certificate` 收敛判据（SPEC §37）。
5. universe：576 行硬门槛 → 全部 active 资产 + posterior 表达弱证据（SPEC §11）。

### 3.3 可能改变策略行为但无理论判决（建议 SPEC 审查的候选）

1. **drift 层级先验的存废**（1.1）：当前 SPEC §34 不包含 drift 项，故形式合规；但它是一次实质先验删除，且是 V0→当前差异中横截面行为最敏感的一项。建议 SPEC 明确记录"层级 drift 是否被有意废弃、其统计职能由谁承接"。
2. **geometry subspace uncertainty 的退场**（1.7）：V0/V1.0 显式传播 bootstrap 几何不确定性；当前 `V1Model` 无 `Phi_draws`。固定 gauge 论证（SPEC §14）解释了几何**跟踪**问题，但未显式裁决"几何不确定性是否仍应进入 predictive"。建议 SPEC 给出结论。
3. **innovation 法律的多模 → macro 标量**（1.6）：SPEC §30 自述这是 KISS 版、非最终 law；但 V0 的每模异质 d 与 idio 投影被整体替换，其相对/横截面风险由 `L_rel` 承接。需确认在 SPEC §26 总不确定性公式下无风险丢失。
4. **V1.0 的 mean-only neutrality**（1.3）：若 SPEC 历史审计需要，此条应作为"修复提交中的局部退步"单列；当前实现已超越它。
5. **`MINROWS` 硬门槛移除**（1.10）：对极年轻资产，当前让其进入模型（由 posterior/EB 收缩），V0 排除。是否为有意的信息集扩大，建议 SPEC 一句确认。

### 3.4 相对行为的可能主因（排序，均为【推断】）

1. drift 层级移除（横截面收缩消失，低信噪比下预测形状改变）；
2. innovation 尺度与法律（多模异质 → macro 标量 + predictive covariance；scenario 尾部/分散度改变，Kelly 规模改变）；
3. OOF residual 化（残差宽度校准改变，`scale=sqrt(v_forecasts/v_bootstrap)` 的 `v_bootstrap` 取自 OOF）；
4. moment matching 移除（有限 S 下样本均值噪声不再被逐资产对齐）；
5. neutrality 处理差异（对 band 间投影的均值/协方差形状影响）。

---

## 4. 未能确定的项

1. 答辩报告原文不可读（`old_src_snapshots.txt:125-127`）；"shrink_drift / ARD / BF 门控" 是否就是报告全部点名项，未逐字核对（`old_src_snapshots.txt:128-129`）。
2. 旧版本之间（V0→V1.0→`85f6e97`→`d5e97dd`→`c109d9f`→…→HEAD）存在多个中间提交，本文只逐项给出"V0 / V1.0 / 当前"三点证据；精确的移除提交点除已引用者（shrink/ARD/BF=`21017b9`、mean matching=`c109d9f`、zero fill 规范=`HEAD`）以外未逐提交穷举。
3. 未运行任何命令，未做数值对照；"策略行为改变"的排序是静态推断，不是收益证据。
4. geometry uncertainty 退场的规范性归属（3.3-2）与 drift 存废（3.3-1）依赖不可读的规范史，本文仅提供可核对的源码事实。
5. `602b897` 快照（性能线终点）未在本文逐项展开；如需第四点对照，需另立任务。

---

## 附：证据索引（路径与行号）

- 旧快照（V0，8 文件）：`old_src_1d9b7ad/src/model.jl`（含 Kelly `L553-568`）、`old_src_1d9b7ad/src/modecov.jl`、`old_src_1d9b7ad/src/backtest.jl`、`old_src_1d9b7ad/src/broker.jl`、`old_src_1d9b7ad/src/data.jl`、`old_src_1d9b7ad/src/KTrader.jl`、`old_src_1d9b7ad/src/live.jl`、`old_src_1d9b7ad/src/repomix-output.xml`（V0 无独立 kelly.jl / geometry.jl 文件；`old_src_snapshots.txt:12-20`）。
- 中间对照（V1.0）：`old_src_21017b9/src/predict.jl`、`old_src_21017b9/src/response.jl` 等 9 文件（`old_src_snapshots.txt:29-38`）。
- 当前源码：`src/predict.jl`、`src/response.jl`、`src/prepare.jl`、`src/kelly.jl`、`src/numerics.jl`、`src/geometry.jl`、`src/residual_oracle.jl`、`src/incremental.jl` 等 13 文件。
- 侦查证据：`dev/evidence/manager13/defect_markers.txt`（A–F 节）、`src_evolution.txt`、`old_src_snapshots.txt`（含聚合与逐文件 SHA-256）。
- 规范对照：SPEC §11、§14、§19、§22、§24-§26、§27-§29、§30-§32、§33-§34、§37、§38-§40、§56（AGENTS.md 内）。

---

*本报告为纯静态语义比对；所有分类均为该层级内的工程判断，推断处已标【推断】。不宣布任何发布、验收或收益结论。*
