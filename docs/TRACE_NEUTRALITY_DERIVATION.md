# Trace Neutrality 推导与裁决

**文档状态：静态分析记录（非规范性文档，不改写 SPEC）**
**任务范围：Gate 0 重开后的 Deliverable 3。本轮未运行任何命令、未执行任何数值实验；全部结论来自源码、测试、文档与 dev 注记的逐字阅读。若本文件与 SPEC/AGENTS 冲突，以后者为准。**
**结论先行：当前仓库中不存在从 price-only 对称性出发生成的 trace neutrality 推导。该约束的诚实裁决是 candidate hypothesis（V1 显式建模假设），尚不能升级为 theorem 或 identification requirement。**

---

## A. Forbidden direction 的精确语义

### A.1 机制事实（逐字核对）

`src/response.jl:106-108`：

```julia
function get_constraint_columns(N::Int, n_bands::Int)
    [(((b-1)*2+channel-1)*N+1):(((b-1)*2+channel)*N) for b in 1:n_bands for channel in 1:2]
end
```

- 外层遍历 `b = 1:n_bands`，内层 `channel = 1:2`；每个元素是一个长度 `N` 的连续列区间。
- `n_bands=7`（`BANDS = 2 .^ (1:7)`）时共 **14** 个区间，顺序为：`(b1,Q),(b1,P),(b2,Q),(b2,P),…,(b7,Q),(b7,P)`。

设计矩阵列布局（`src/response.jl:55-75`，`fill_design_matrix!`）：

```julia
offset = (b-1)*2N
col_q = offset + j
col_p = offset + N + j
```

- 每个 band `b` 占 `2N` 列，前半 `N` 列为 Q 通道（逐资产），后半 `N` 列为 P 通道（逐资产）。
- 因此 `cols[c]` 展开后第 `j` 个列索引恰对应资产 `j` 在该 band/channel 的自己的 feature 列。

约束矩阵 `C` 的显式构造见于 `test/conditioned_eb_tests.jl:155-158`：

```julia
cols=KTrader.get_constraint_columns(N,length(KTrader.BANDS)); C=zeros(length(cols),N*P)
for c in eachindex(cols),j in 1:N
    C[c,j+(cols[c][j]-1)*N]=1.0
end
```

按 Julia 列主序，`vec(G)` 中 `G[i,p]` 位于下标 `i+(p-1)*N`。故对第 `c` 行：

\[
(C\,\mathrm{vec}(G))_c \;=\; \sum_{i=1}^{N} G\!\left[i,\;\mathrm{cols}[c][i]\right].
\]

因为 `cols[c]` 是连续区间且区间内第 `i` 列对应资产 `i`，该式化为所在块的 **对角和**：

\[
c=2b-1:\quad \operatorname{tr} A_b \;=\; \sum_{i=1}^{N} G[i,\,\text{asset }i\text{ 的 Q 列}],
\qquad
c=2b:\quad \operatorname{tr} B_b \;=\; \sum_{i=1}^{N} G[i,\,\text{asset }i\text{ 的 P 列}].
\]

这与 `src/response.jl:128` 的实现 `h[c] = tr(view(G, :, cols[r]))` 以及 `ridge_constraint_traces`（`src/response.jl:412-418`，直接收缩各块对角）逐字一致；`archive/notes/response_deadwork_20261008.md:7-19` 的等价改写记录同样确认该语义。

### A.2 语义陈述

`C` 的每一行是一个作用在系数空间上的线性泛函，语义为：

> **在固定的时间尺度 band \(b\) 与固定的通道（Q 或 P）上，全部资产"对自身输入"的系数之和为零。**

- Q 与 P 在此处是 AGENTS §16 定义的 paired local path coordinates：Q 对应"位置偏离"（`-(X-c0)/s`），P 对应"方向/确认"（`(c0-c1)/s`）。`A_b`/`B_b` 只是 G 在该块上的记号（AGENTS §17：\(G_b=A_b+iB_b\)），**不是"实部/虚部"意义上的频域对象**。
- 约束只涉及**块内对角配对** \((i,\text{asset }i\text{ 自己的列})\) 的和，**不涉及**块内 \(i\neq j\) 的 off-diagonal 项，也不涉及跨 band/跨通道的任意组合。
- 约束禁止的是一族方向：任何满足 \(\operatorname{tr}A_b\neq0\) 或 \(\operatorname{tr}B_b\neq0\) 的系数矩阵。AGENTS §21 给这族方向的解释是"每个时间尺度的 universal common timing response 被移除"；含义文本为"所有资产在某 band 上同样的趋势或同样的 anticipatory bias"。**注意语义精度**：C 实际杀死的是**对角和**；例如块内 \(G[i,j]=g\)（全常数矩阵）满足 \(\operatorname{tr}=Ng\neq0\) 会被移除，但块内 \(G[i,i]=0,\ G[i,j]=g\ (i\neq j)\) 的纯交换分量 \(\operatorname{tr}=0\) 不被该约束禁止。也就是说，"universal common response 被移除"这一表述只在"共同响应指对角自响应分量"时才与 C 严格等价；若共同响应还包括资产间的对称传导分量，则 C 并不移除它。这一语义缺口在本文件 D 与未决问题中继续处理。

---

## B. 为什么它应当是 hard removed（identification），现有论证是否成立

### B.1 仓库现有论证的原文与位置

- AGENTS §21（line 704-723）："它是 structural / identification constraint，不是防过拟合的数值 regularizer。"
- AGENTS §22（line 727-765）："禁止：先拟合无约束 posterior；只把 mean 的 trace 减掉；covariance 仍允许离开 neutral subspace。"并给出精确条件化公式 \(m_c=m-\Omega C^T(C\Omega C^T)^{-1}Cm\)、\(\Omega_c=\Omega-\Omega C^T(C\Omega C^T)^{-1}C\Omega\)。
- AGENTS §61（line 2083-2093）：宪法测试要求 posterior mean / sample support 均满足 \(\operatorname{tr}A_b=\operatorname{tr}B_b=0\)，threshold 应接近 machine precision。
- `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217`："Posterior uncertainty over the path-response law and trace neutrality are theoretical constraints that would remain under unlimited compute."
- AGENTS 历史段 §13（line 3087）把"trace neutrality 只是 post-hoc mean correction"列为 0.9 时代的核心缺陷；§12（line 3051-3064）称"早期双重 neutralization 被统一成 operator constraint"。

### B.2 评估

把现有材料拆成可检验的主张：

1. **"必须存在于 posterior support 而非只修 mean"——成立且证据充分。** 后验协方差若不条件化，scenario 抽样（`generate_scenarios_v1` 消费 `L_rel`）仍会离开 neutral 子空间；`condition_trace_neutrality` 的 dense oracle 对照（`test/conditioned_eb_tests.jl:190-198`，`exact.G_c ≈ dense`、`exact.inv_M ≈ inv(Symmetric(M))`，atol 1e-10）证明实现与标准条件化公式一致。这一条是"hard conditioning 的正确实现方式"，不是识别性证明。
2. **"它是 identification constraint 而非数值 regularizer"——作为识别性主张，仓库材料不足以支持。** 识别性（identification）的正式标准是：约束方向在观测分布上不可识别，或与已建模通道严格冗余/共线。仓库中没有任何一处证明 \(\operatorname{tr}A_b\)、\(\operatorname{tr}B_b\) 方向在 data-law 下不可区分。相反，`archive/research/theory_incremental_20261009/THEORY.md:243-251` 明确记录：约束后的修正被分配到"未识别方向"，且"未识别系数的均值不一定是 0"、"不同支撑块会经 14 个 trace 约束产生条件相关"。这段文字实际上把 forbidden 方向当作**可以被条件化机制触碰的方向**，与"这些方向已被识别掉、不存在"的强识别语义不符。
3. **"无限算力下仍存在"——不蕴含 identification。** 一个硬先验/支撑选择同样在无限算力下存在；该句只能排除"这是数值截断/近似"的误读，不能排除"这是建模选择"。
4. **一个尚未被仓库提出的潜在识别论证（本文档分析，非既有依据）**：relative 目标 \(Y\) 与 innovation 位于严格 zero-sum relative support（AGENTS §12；`fit_response_operator` 对 `Sigma` 做 `supported_covariance(..., gauge)`；`conditioned_eb` 以 `gauge` 参数投影），若模型把"\(E\) 严格支撑在 \(\mathbf{1}^\perp\)"当作精确约束，则每行预测 \(XG^T\) 也需行和为零；而对角均匀方向 \(cI\) 对行和的贡献是 \(c\cdot\sum_{p\in\text{block}}X[t,p]\)，一般不为零。这条路线**若**能被形式化，可以论证该方向被 target support 的似然结构识别为 0，硬约束只是把有限样本下的估计噪声提前截掉。但仓库没有做这个推导，且它需要与 §34 的 \(\mu_{rel}^{zero-sum}\) 使用端投影交互核对，因此当前只能列为"升级所需的可能推导形态"，不能算作既有证据。

### B.3 小结

现状是：**hard removal 在工程上被完整实现且自洽（精确条件化、进入协方差、有 dense oracle 测试）；但"它是 identification requirement"这一规范性主张在仓库中没有推导支撑，属于被制度化的声明。** 因此对 B 项的诚实回答是：为什么它"应当是"hard removed，在现有材料中找不到可引用的证明；能找到的只是"V1 选择这样做"的记录。把 mean 修掉、covariance 不管的做法有明确反例意义（scenario 会违反 §61），这一半成立；"因此必须 hard remove 而非留给 posterior"另一半的识别性前提未证。

---

## C. 为什么是逐 band、逐 Q/P 的 14 条约束

### C.1 事实

- 14 = \(2\times|BANDS|=2\times7\)（AGENTS §22 line 765："把核心约束 solve 压到 14×14"）。
- 粒度是 per-band、per-channel（A 项已证）。`test/relative_support_tests.jl` 的 fixture（N=2, gauge 维 1）与 `archive/research/theory_incremental_20261009/THEORY.md:251` 的"14 个 trace 约束"均按此粒度引用。

### C.2 为什么不能更弱——变体差异分析（本文档推导；仓库未给直接论证）

仓库中**未找到**"为什么是这一特定粒度、而不是这些变体"的论证。以下比较是本文档基于线性代数与语义的静态分析：

1. **只保留一条全局约束 \(\sum_b(\operatorname{tr}A_b+\operatorname{tr}B_b)=0\)**：
   - 个别 band 的共同分量可相互抵消而整体通过约束。例如 band1 对角 \(+c\)、band2 对角 \(-c\)，全局和为零，但每个尺度内部都存在真实的 common response，与 §21 的文字含义（"每个时间尺度的……被移除"）不符。
   - 对后验几何的影响：单条约束的零空间远大于 14 条，forbidden 方向只是被"平均"掉，per-scale 识别方向保留在 support 中。
2. **只保留 \(\operatorname{tr}A_b+\operatorname{tr}B_b=0\)（合并 Q/P）**：
   - Q 与 P 是语义不同的通道（位置偏离 vs 方向/确认）。合并后共同的"位置偏差响应"与共同的"确认/预期偏差"可以互相抵消：\(+\delta\) 在 Q、\(-\delta\) 在 P 时通过约束，但两条经济通道各自的共同分量都未被移除。
   - per-channel 分开保证每个语义方向独立地净共同响应为零。
3. **只约束 posterior mean（post-hoc trace correction），covariance 不条件化**：
   - 这是 0.9 历史缺陷（AGENTS §13）。scenario 抽样经 `L_rel` 产生，若支撑未条件化，samples 会离开 neutral 子空间，§61 的"posterior mean / sample support 均满足"无法成立。
   - 因此该变体的差别不是"少一条约束"，而是"约束没有进入模型的预测分布"——AGENTS §22 已明确禁止。
4. **一般投影 \(P_{\text{forbidden}}G=0\)**：
   - 若 \(P_{\text{forbidden}}\) 的零空间恰等于 14 条 C 行的零空间，则与当前约束**数学等价**；当前 `condition_trace_neutrality` 正是该投影在 \(\Omega\) 度量下的精确条件化实现（B.2 第 1 条）。
   - 若"更一般"指不同的投影（如只把输出方向 e0 上的整体载荷清零），则与 C 不同：C 是按"输入资产自己的列 × 输出行求和"配对的 14 个泛函 \(\left(\sum_i G[i,\text{asset }i\text{ 的列}]\right)\)，而 e0 输出通道是 \(\sum_i G[i,p]\) 对每个固定输入列 \(p\) 的载荷。两者在块内全常数矩阵 \(G[i,j]=g\) 上会给出不同结果（C 得到 \(Ng\)，e0 通道每列得到 \(Ng\) 同样非零，但结构不同；在 \(G[i,j]=\delta_{ij}c\) 上 C 得到 \(Nc\) 非零，而每个固定列 \(p\) 的 e0 通道载荷为 \(c\) 也只有当 \(p\) 是该资产列时非零）。
   - 频域变体 \(\operatorname{tr}G(\omega)=0\ \forall\omega\)：项目没有频域算子表示；finite basis 下任何有限条约束都不能逐点表达该无限维条件，除非对 G 的跨频率结构另行假定。且 AGENTS §16 明确禁止把任意 Fourier 相位无条件映射到当前 paired basis 的金融语义。因此该变体在 1.0/2.0 的 finite-basis 架构中不可直接实现；当前 14 条是对每个离散尺度、每个通道的自然有限表述。
5. **为什么不更强**（例如同时约束 off-diagonal 之和、或整块均值）：
   - 仓库未讨论。可静态观察：C 只删对角和意味着模型仍能用 off-diagonal 系数表达任何跨资产传导结构；若连 off-diagonal 总和也约束为零，将禁止"全体资产同向传导"这类可能真实的结构，把约束从 identification 风格推向更强的经济先验。当前选择保留了最大灵活性，同时只移除"资产对自身输入的共同响应"。

### C.3 小结

"14 条"与实现结构（2 通道 × 7 尺度 × N 资产块的 per-block 对角和）严格对应，工程上是自洽的最小完备集：**每个 band、每个通道各一条，互不抵消、互不遮蔽**。仓库对"为什么不能更弱/更强"没有成文论证；上述差异比较是本文档的补充分析。

---

## D. 不变量审计

### D.1 permutation（§58）

- C 的每一行是"资产索引 i 同时选择输出行 i 与输入列（资产 i 自己的列）"的求和，求和哑标 \(i\) 对指标重命名不变。资产置换 \(\Pi\)（输出行与输入列块同步重排）下 \(C\cdot\mathrm{vec}(G)\to C\cdot\mathrm{vec}(\Pi^\top G\Pi_{\text{block}})\)，数值仍为各块对角和——**约束集合与置换兼容**。
- 测试覆盖：`test/v1_constitutional_tests.jl:13-18` 断言 `mu_pred`、`res_history`、`L_rel*L_rel'` 的置换协变，但**没有直接断言 `G_c_mean` 的约束行在置换下的数值不变**。分析支持兼容，直接测试缺失。

### D.2 price-scale（§57）

- 设计矩阵经 `inv_scales`（由 ruler 给出）标准化，系数空间与价格单位无关；C 是纯系数线性泛函。约束与价格缩放解耦。
- 测试覆盖：`v1_constitutional_tests.jl:9-12` 断言缩放不变性于 `mu_pred`/`L_rel` 层，**不含针对 C 的直接断言**。

### D.3 relative gauge 旋转（§59）

- 关键事实：`constraint_moments`（`src/response.jl:109-130`）**完全不使用 gauge**；它只用 `V.basis`、`V.weights`、`Sigma` 与 `G` 的块对角。`condition_trace_neutrality` 同样不含 gauge。
- C 的行在 asset 空间定义（对输出行求和即对 e0 方向的载荷），而 gauge 旋转只作用在 \(\mathbf{1}^\perp\) 内部（Helmert basis，AGENTS §13）。因此**约束本体在 gauge 旋转下表示无关**：\(\mathbf{1}^\top Q=0\) 使 e0 方向在 gauge 下不动。
- gauge 影响的是 conditioning 中 \(\Sigma\) 的 relative support 与证据计算（`_conditioned_block_geometry` 对块做 `gauge'*view(...)` 投影，`src/response.jl:420-424`）。因此端到端 gauge covariance 需要经由 \(\Sigma\) support 传播后仍成立——现有测试是间接的：`test/relative_support_tests.jl` 用 N=2（gauge 维 1，旋转群平凡）的解析消元 fixture；`test/conditioned_eb_tests.jl:148-199` 在 `gauge=Q` 下对照 dense oracle，但不旋转 Q 本身。**直接的 gauge-rotation-invariance 测试（Q→QR）针对 trace 约束未见。**

### D.4 band 表示

- C 的 14 个块与 `BANDS` 的顺序、数量硬绑定（`(b-1)*2N` offsets）。改变 bands 集合或顺序即改变约束集合。这是**表示依赖，不是不变量缺陷**——bands 是设计选择而非对称性；但没有 band 置换等价性的测试（也不应有，不同 band 是不同语义对象）。

### D.5 是否强迫 diagonal self-response 转移到 off-diagonal（sink 风险）

**数学形式（本文档分析）**。条件化后的均值修正为：

\[
m_c-m=-\Omega C^\top(C\Omega C^\top)^{-1}Cm.
\]

- 修正位于 \(\mathrm{span}(\Omega C^\top)\)，即 14 个 forbidden 行在 \(\Omega\) 度量下的像所张成的子空间。\(\Omega\) 的结构决定这 14 个方向如何分布到具体系数上。
- 若数据（或先验）在无约束后验里支持一个共同对角响应（所有资产的 own-response 相同，\(Cm\neq0\)），条件化会把它删除并沿 \(\Omega C^\top\) 的方向重新分配。若 \(\Omega\) 中 forbidden 方向与"同块 \(i\neq j\)"项相关较强，修正就会出现在 off-diagonal 系数上；预测层面等价于把"共同自响应"重写为"资产间互相响应"。
- `archive/research/theory_incremental_20261009/THEORY.md:243-251` 记录了相关现象的两面：约束修正被分配到"未识别方向"（\(P_A G_c P_A=-(δ/α)\lambda_c P_A\)），并警告"不同支撑块会经 14 个 trace 约束产生条件相关，不能只留对角方差"。这证明项目已知修正会跨方向传播，**但没有量化其是否集中到少数 cross-asset 方向**。
- **仓库中没有**任何实验、日志或测试断言测量修正 \(\|m_c-m\|\) 在块内各资产上的分布，因此"人为制造少数 cross-asset signal sink"无法从现有材料证实或证伪。静态上可以确定的是：修正的集中性完全由 \(\Omega\) 的低秩/块结构决定，而不是 C 本身单独决定；当 \(\Omega\) 近各向同性时修正分散，当 \(\Omega\) 低秩时可能集中。
- 另一点可以静态确定：无论修正如何分布，**它不改变条件后验的自洽性**（dense oracle 已证），改变的只是模型把数据信号表达为 diagonal 还是 off-diagonal 的**经济归因**。这是模型选择偏差问题，不是数值正确性问题。

### D.6 审计汇总

| 不变量 | C 本体是否满足 | 直接测试 |
|---|---|---|
| permutation | 分析成立（哑标求和） | 无直接断言（仅 mu/cov 层间接） |
| price-scale | 成立（系数空间无单位） | 无直接断言 |
| relative gauge 旋转 | 约束本体表示无关（不含 gauge；e0 不动） | 无 Q→QR 直接测试；间接 fixture |
| band 表示 | 与 BANDS 硬绑定（设计选择） | 不适用 |
| diagonal→off-diagonal 转移 | 风险路径存在；程度未量化 | 无任何测量 |

---

## E. 是否存在从 price-only 对称性出发的真正推导

**未找到。**

已检索的位置与结果：

- `AGENTS.md`：§21/§22、§57-§61、§12/§13 历史段、附录 C 第 9 问（"trace neutrality 为什么是理论 constraint？"）。全部为**声明式/历史式**文本，没有任何"从价格序列的某个对称性或可观测等价性推出 C"的推导。
- `README.md`：未检索到 trace neutrality 的推导段。
- `CHANGELOG.md` 与 `dev/**/*.md`：仅有 `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217` 的"theoretical constraints that would remain under unlimited compute"一句（声明），与 `archive/research/theory_incremental_20261009/THEORY.md:243-251,353`（讨论约束**之后**的后验几何与数值保留，不是约束的证成）。
- 源码注释：`src/response.jl` 相关函数注释只解释计算次序与等价改写（如 `ridge_constraint_traces` 的收缩），不含约束来源的推导。
- 提交注记：本机 `.git` 存在，但本轮不允许运行任何命令；**无法查阅提交历史**。就仓库可读文本而言，AGENTS §13 的历史叙述（"早期双重 neutralization 被统一成 operator constraint"）表明其来源是设计演进中的收敛选择，而非从数据对称性演绎。若提交历史中存在推导，本文件无权声称已覆盖——按委托口径，此类内容记为"本机可读文本中未找到"。
- 搜索关键词覆盖：`trace neutrality`、`neutrality`、`universal common`、`common timing`、`anticipatory bias`、`identification`、`structural constraint`、`gauge freedom`、`uniform`、`per-band` 等（大小写与中英文混用）。

---

## 裁决

### 分类

**candidate hypothesis（V1 显式建模假设），当前不满足升级为 theorem 或 identification requirement 的证明标准。**

- **theorem**：不可行。没有任何从更高层公理（price-only、固定规律、价格几何）到 C 的演绎链。C 是关于响应算子**系数**的结构约束，price-only 输入原语本身不产生它。
- **identification requirement**：未证明。缺少"该方向在观测分布上不可识别，或与已建模通道（macro 通道、relative 支撑）严格冗余"的论证。反而有理由认为该方向在有限样本中是**可拟合、可识别的**（似然对它的敏感度一般非零）；被移除是建模偏好。
- **candidate hypothesis**：与现有文本最一致。README.pre-final.md 的"would remain under unlimited compute"只支持"非数值近似"，不支持更强级别。THEORY.md:243-251 的"未识别方向"措辞同样与"已识别掉"矛盾。

### 升级所需的推导或证据形态

任一即可构成升级路径，三者都要求形式化证明与可复核的测试/反例：

1. **target-support 识别论证**：证明在"innovation 严格支撑于 \(\mathbf{1}^\perp\)、relative 目标行和为零"的模型下，forbidden 方向 \(cI\) 对观测似然的 Fisher 信息为零或被 target support 精确排除（需与 §34 的 \(\mu_{rel}^{zero-sum}\) 使用端操作交互核对，因为使用端投影本身会改变 forbidden 方向的可观测性）。
2. **参数冗余/重参数化不变性**：证明对任意 \(G\)，存在 \(G'=G+\Delta\)（\(\Delta\) 落在 forbidden 方向且非零）使所有可达预测 \(Gx_t\) 与 \(G'x_t\) 在观测等价意义下相同，从而约束是"去掉冗余参数化"而非"删除可识别自由度"。
3. **对称性公理**：给出一条独立于 C 的价格世界假设，并证明 C 是其必要条件；同时该假设不能被"真实世界共同响应存在"这类反例反驳。按当前证据，路径 3 最难成立。

配套证据形态：一个能红/绿的最小测试（例如构造数据含真实共同对角响应，验证移除前后预测差异与识别论证的预测一致），并在 \(\Omega\) 的结构下量化 D.5 的修正分布。

### "降级为 explicit hypothesis、不进入 2.0 core" 的操作含义

- **不是**删除约束：`get_constraint_columns`/`condition_trace_neutrality`/`ridge_constraint_traces` 机制、AGENTS §61 的测试、14×14 条件化全部保留不动；不得无声删除、不得放宽阈值。
- **是**措辞与地位修正：在所有对外叙述（SPEC/README/发布说明）中把该约束标注为"V1 声明的建模假设（candidate hypothesis），未证明为 identification"；不得再以"理论约束/识别性要求"的确定口吻对外宣称；与 macro/relative 分解、Q/P basis 等其它已声明的建模选择并列管理。
- **重审触发器**：出现 (a) 上述升级证明，或 (b) 反例证据表明该约束造成系统性预测扭曲（如 D.5 的修正集中在少数 cross-asset 方向并被数据证伪），则重新裁决。
- 本文件不改动任何规范文本；是否执行降级措辞由规范性文档（AGENTS/SPEC）的 owner 决定。

---

## 未决问题清单

1. **升级证明缺失**：B/E 所述识别性论证不存在；target-support 路线尚未与 §34 使用端投影的交互核对。
2. **D.5 修正分布未量化**：conditioning 修正 \(\Omega C^\top(C\Omega C^\top)^{-1}Cm\) 在 off-diagonal 上的集中程度无任何实验/日志；本轮禁止数值实验，故保持未决。
3. **语义缺口**：C 只移除对角和；"universal common timing response 被移除"的 AGENTS §21 措辞在"共同响应含资产间对称传导"的解释下过宽。是否需要在规范中收紧措辞，未决。
4. **gauge 旋转直接测试缺失**：约束本体分析为表示无关，但 Q→QR 的端到端协变性无直接测试。
5. **提交历史未阅**：本机 `.git` 不可访问（无命令权限）；若提交注记中存在推导，本文件的"未找到"仅覆盖可读文本。

---

## 关键证据位置索引

- `src/response.jl:106-108` — `get_constraint_columns` 的 14 块构造。
- `src/response.jl:109-130` — `constraint_moments`：\(h_c=\operatorname{tr}(G[:,\text{cols}[c]])\)。
- `src/response.jl:131-148` — `condition_trace_neutrality`：精确条件化实现。
- `src/response.jl:412-418` — `ridge_constraint_traces`：对角收缩等价改写。
- `src/response.jl:55-75` — 设计矩阵列布局（Q/P 逐资产）。
- `test/conditioned_eb_tests.jl:155-158` — C 的显式构造（语义锚点）。
- `test/conditioned_eb_tests.jl:190-198` — dense oracle 条件化对照（实现正确性证据）。
- `test/v1_constitutional_tests.jl:21-27` — §61 约束的宪法测试。
- `test/relative_support_tests.jl` — gauge 支撑（N=2）分析性测试。
- `AGENTS.md:704-765` — §21/§22 规范文本。
- `AGENTS.md:2083-2093` — §61 宪法测试要求。
- `AGENTS.md:3051-3064, 3087` — 历史段（约束统一与 0.9 缺陷）。
- `archive/research/theory_incremental_20261009/THEORY.md:243-251` — 约束后验几何与"未识别方向"记录。
- `dev/evidence/final_2_0_0_20261009/README.pre-final.md:213-217` — "theoretical constraints / unlimited compute"声明。
- `archive/notes/response_deadwork_20261008.md:7-19` — trace 收缩的等价性记录。
