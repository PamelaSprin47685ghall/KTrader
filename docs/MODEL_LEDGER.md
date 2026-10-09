# KTrader 理论对象与生产实现对账账本（Model Ledger）

状态：纯静态事实账本。生成方式为逐文件静态阅读，**未运行任何命令、未执行任何测试、未修改任何源码**。所有行号来自本次阅读时的工作树快照；证据标记：[事实] = 代码直接支持；[推断] = 由代码语义纯静态推导；[未知] = 本次阅读无法确定。

对账对象：AGENTS.md 声明的理论对象 与 当前生产代码实际计算的对象。差距分类标签：identical / plug-in approximation / scalar degeneration / finite-sample approximation / unknown。

---

## 0. 证据方法与边界

- [事实] 本次已读：src/data.jl、src/prepare.jl、src/predict.jl、src/response.jl、src/residual_oracle.jl、src/kelly.jl、src/backtest.jl 全文；src/geometry.jl（常量与 ruler 段）、src/numerics.jl（PriceHistoryCache 段）、src/broker.jl（tradable/liquidation/rebalance 段）、universe.txt 全文；若干 dev 脚本与测试的相关段。
- [事实] 账本唯一新建文件即本文件；未触碰 src/、test/、README.md、AGENTS.md。
- [事实] 全仓检索 PONY：唯一命中 universe.txt:46；未找到把标的"前 251 日 bar 置 false"的 wrapper 脚本（检索模式含 PONY、251、.bar[...]=false、.bar[1: 等）。详见第 11 节。
- 理论声明引用 AGENTS.md 的章节号（§编号），不对 AGENTS.md 编造行号。

---

## 1. History H_t：进入模型的信息

理论（AGENTS.md §3/§4/§8）：H_t 是完整价格历史；x_i(t)=log P_i(t)；missing 不得被 carried marking 冒充为 r=0。

[事实] 数据结构：Bars 含 dates/symbols/close/adj/bar（src/data.jl:7-13）。close/adj 是 marking series：上市前 NaN，之后 carry 前值（src/data.jl:15-25；docstring src/data.jl:2-4）。

[事实] 理论输入由 signal_prices(b)=ifelse.(b.bar, b.adj, NaN) 恢复（src/data.jl:37；docstring src/data.jl:33-36）：bar=false 之日严格 NaN，绝不把 carried marking 注入为 0 收益。

[事实] 模型入口消费 signal：_prepare_v1 内 log_adj=log.(adj_act)（src/predict.jl:216）；r=diff(log_adj; dims=1)（src/predict.jl:247）——只有相邻两日均 finite 才有 return。

[事实] backtest 的模型输入是 signal=signal_prices(b)（src/backtest.jl:149）；账户 marking 另用 b.adj[t+1,:]./b.adj[t,:]（src/backtest.jl:296）。两条路径互不冒充。

[事实] zero-embedding 只发生在 relative field 内部（src/predict.jl:43-70；注释 38-42），且 observed mask 单独保留（src/predict.jl:68）。这不是把 missing return 当 0。

[推断] 模型可见的 H_t 即 signal 的 finite 格；carried adj 不进入 r（r 基于 signal 的 log）。

[未知] 本地 panel CSV 的原始 provenance 沿用 AGENTS.md 的 unknown 记录，不在本账本重判。

分类：identical（§8/§8.1/§8.2 语义与 signal_prices、diff 实现一致）。

---

## 2. Observation mask O_{t,i}：真实市场观测在哪里表达

[事实] 市场真实观测的权威字段是 Bars.bar[t,j]（src/data.jl:12；注释 3-4）。4 参数构造：bar=isfinite.(rawclose).&isfinite.(rawadj)（src/data.jl:28）。

[事实] 保存时 bar 决定落盘缺失：ifelse.(b.bar, b.close, NaN) / ifelse.(b.bar, b.adj, NaN)（src/data.jl:80-82）。

[事实] return 级观测 mask：observed=isfinite.(r)（src/predict.jl:68）。

[事实] 决策日 support：alive_now 默认取 adj_act 末行 isfinite（src/prepare.jl:131），构造时强制 owned copy（src/prepare.jl:160-161）；solve 用 findall(prep.alive_now) 计算 e0（src/predict.jl:473-478）。

[事实] ResidualOracle 每行 own mask：O={j: isfinite(r[t+1,j])}，c=|O|（src/residual_oracle.jl:19-20；evaluation 160、198、283、290-291）。

[事实] missing-cell fallback 只使用该资产自身 observed 残差行（src/predict.jl:596-602；src/residual_oracle.jl:262-274）。

[推断] 三处表征同源且互不矛盾：bar → signal NaN → 无 return → 不进 own rows。alive_now 是决策日（末行）快照，observed 是 return 行级。

分类：identical。

---

## 3. Model admission A^model：active universe 链

[事实] active_universe_indices(adj)：资产当且仅当存在 t in 2:T 使 adj[t-1,j] 与 adj[t,j] 均 finite（src/predict.jl:125-135；判定在 130 行）。即"至少一条有效 daily return"。

[事实] cache 路径：active_idx=findall(<(T_raw), history_cache.first_return)（src/predict.jl:207-208）；first_return 定义为第一个 finite returns 行（src/numerics.jl:150）。

[推断] 两路等价：第一个 finite return 行 t0 对应首个相邻 finite 对 (t0, t0+1)；first_return < T_raw 等价于该对落在 prefix 内。

[事实] warmup：WARMUP=2*max(BANDS)=256（src/geometry.jl:14；BANDS=2..128，src/geometry.jl:12）；训练行 ts_total=WARMUP:T-2（src/predict.jl:286）；F_folds 校验 2<=F_folds<=n_res（src/predict.jl:288）。

[事实] 消费者：ruler（src/predict.jl:222-243）、X_rel/s_perp/B_m（src/predict.jl:256-281）、full/fold Gram（src/predict.jl:300-305）、solve 的 e0（src/predict.jl:473-478）、Kelly 的 active 掩码（src/kelly.jl:152-153）、backtest free（src/backtest.jl:241-245）。

[事实] 全 NaN dummy 资产严格排除（src/predict.jl:122-123 注释）；e0 只在 alive_now 上归一 e0_now[alive_now].=1/sqrt(N_alive)（src/predict.jl:475-478）；mu_asset_full 只写 active_idx（src/predict.jl:485-486），inactive 留 0 而不进入坐标空间。

[事实] 退化特例：N==1 时相对空间维为零、响应精确为零（src/response.jl:1156-1163）；N==2 有构造性 witness 解析候选（src/predict.jl:371；src/response.jl:832-851）。

[推断] "进入 active"（坐标空间）与"可训练"是两道门：active 需要 ≥1 条有效 return；训练还需要 ts_total 非空（T_raw>=258 才有 >=1 行，>=259 才有 >=2 行）；ruler 另有 4*tau<=T-f+1 门槛（src/geometry.jl:38）。

分类：identical（§11 的实现一致）。

---

## 4. Trading eligibility E^trade：是否存在独立概念

[事实] 当前代码中没有名为 E^trade 的独立对象（本账本阅读范围内未见）。承担者是 backtest 的 free 掩码：free=falses(N)；for j in active; free[j]=b.bar[t,j]（src/backtest.jl:242-245）——即 free = active ∩ bar(决策日 t)。

[事实] Kelly 层：free=tradable .& active；locked=current .* .!free；budget=1-sum(locked)（src/kelly.jl:152-156）。参数名 tradable 在 backtest 调用处传入 decision.free（src/backtest.jl:80,291-292）。

[事实] path_kelly_v1 的 tr=tradable===nothing ? trues(N) : tradable（src/kelly.jl:177）。

[推断] bar 同时承担两个语义：观测 mask（§2）与"当日可交易"（src/data.jl:3-4 注释明确 bar "was tradable"）。

分类：unknown（理论未见独立 E^trade 声明；实现是调度级掩码，且与观测 mask 共用 bar 字段）。

---

## 5. Physical tradability T：broker/tradable 语义

[事实] broker.tradable(q)=q.bid>0 && q.ask>0 && q.last>0（src/broker.jl:136），语义为"有活跃双边市场"；halted/unquoted → false。

[事实] rebalance! 对 tradable=false 的标的跳过操作、保留仓位（src/broker.jl:192）；target_shares 由权重与 last 价取整（src/broker.jl:143）；liquidation 对外部持仓渐进卖出（rate in (0,1]，src/broker.jl:155-168）。

[事实] 该层属于 execution（AGENTS.md §5/§41）；backtest 的 bar 掩码不经过 broker.tradable。

分类：identical（物理可交易性由 broker 语义承担，理论层不消费）。与第 4 条的区别：T 是报价级、E^trade（free）是 bar 级，二者当前不互相校验。

---

## 6. Response posterior：solve→fit_response_operator→optimize_conditioned_eb 实际链

[事实] 链：solve(prep)（src/predict.jl:349-505）→ fit_response_operator（src/response.jl:1124-1209）。

[事实] macro 分支：optimize_matrix_normal_eb 求 alpha_m 点估计（src/response.jl:1146-1148；实现与证书 351-379）；gm=vec(ev.vectors*(B.*dm))（1151）；sig2=max(sse/max(n-gamma_m,1),1e-8)（1154）；covm=sig2.*((ev.vectors .* dm')*ev.vectors')（1155）。

[事实] relative 分支：ridge_spectrum（src/response.jl:1174-1176）→ optimize_conditioned_eb 求 (alpha_rel, Sigma) 点估计（src/response.jl:1178-1185；实现 934-1121，含 witness 分支 943-953 与逐迭代证书 994-1000）。

[事实] G：B_scaled_G=spectrum.B .* dr（1188）；G=B_scaled_G'*spectrum.basis'（1191）；V=ridge_covariance(spectrum,alpha_rel)（1199）；cond=condition_trace_neutrality(G,V,Sigma,N,len(BANDS))（1200-1202；实现 131-143）。

[事实] OOF folds：need_uncertainty=false（src/predict.jl:443），V_out 变为空 RidgeCovariance（src/response.jl:1205）——fold 只消费条件均值。

[事实] predictive_moments（src/response.jl:1211-1229）：mu_m=dot(G_macro,Bm)（1218）；var_m=max(dot(Bm,post_cov_m*Bm),0)（1219）；mu_rel=G_c_mean*x（1220）；cov=Symmetric(dot(x,v).*Sigma_rel − H*inv_M_constraint*H')（1226）；L_rel=ev.vectors.*sqrt.(max.(ev.values,0))'（1228）。

[事实] 场景采样：relative=principal_sqrt_root(L_rel)*z + mu_rel（src/predict.jl:570-572）；macro_draw=mu_m+sqrt(var_m)*z（src/predict.jl:612）。

[推断] 实际对象是 Pi(dG | alpha_bar, Sigma_hat, H)：G 在 EB 点估计 (alpha_bar, Sigma_hat) 条件下为 Gaussian（均值 G_c，协方差为 V⊗Sigma 经 trace 约束条件化后的预测边缘）；**不是**对 (alpha, Sigma) 的联合后验积分——alpha、Sigma 是 EB 最优点（response.jl:374-379、1178-1185），其自身不确定性不作为 predictive 方差项进入。

[推断] G 是条件采样（scenarios 从 L_rel 与 var_m 抽），不是只用 G_c_mean；macro 的 var_m 与 relative 的 L_rel 都携带条件后的参数不确定性。

分类：plug-in approximation（α、Σ 点估计插件；G 条件 Gaussian 采样）。

---

## 7. Innovation law：fractional posterior、场景缩放、均匀行 bootstrap

[事实] fractional posterior 只作用于 scalar macro OOF 残差：e_res_m=macro_residual_series(res_history)（src/predict.jl:493-495；实现 src/residual_oracle.jl:216-260）。

[事实] 标量公式：e[idx]=sumr/sqrt(c) − mu_macro*sw/c − dot(h,x)/sqrt(c)（src/residual_oracle.jl:257）；c==0 → 0.0（239）。h 只在单次调用内按 (fold, mask) 缓存（246-254）。

[事实] 似然与后验：causal_fractional_posterior（src/predict.jl:76-99）；FFT 路径 fractional_likelihood!（101-119）；log_weights=ll .+ log.(DELTA_D_V1) 后 softmax（94-96）；DGRID_V1 / DELTA_D_V1（10、14）。

[事实] 绝对缩放：v_bootstrap=max(var(e_res_m),1e-8)（src/predict.jl:499）；每场景 scale=sqrt(v_forecasts[d]/v_bootstrap)（615）。

[事实] 均匀行 bootstrap：row=min(floor(ur*T)+1,T)，ur=uniforms[s,2]（src/predict.jl:591、578）；每场景抽一行、全资产共享该行（除 fallback）；缺格 own-row fallback（596-602）。

[事实] 场景公式：loggross=(mu_m*e0[j]+relative[j,s]−shift)*s1[j]+residual*scale（src/predict.jl:620）。

[事实] residual 行来自 OOF oracle（src/residual_oracle.jl:150-211），非 in-sample；dense 参考公式保留为测试用 dense_oof_residuals（301-337）。

[推断] 实际 innovation law =（单一 scalar macro 残差的分数阶缩放）×（OOF 残差行的均匀 bootstrap）；横截面风险没有逐资产 fractional、没有残差协方差重建，跨资产相关性只来自共享行与 L_rel 的结构项。

分类：scalar degeneration（宏观维退化到标量序列）+ finite-sample approximation（经验行 bootstrap）。

---

## 8. Structural constraint：C 的构造与 14 行

[事实] get_constraint_columns(N,n_bands) 生成 14 个列块：for b in 1:n_bands, channel in 1:2，块为 ((b-1)*2+channel-1)*N+1 : ((b-1)*2+channel)*N（src/response.jl:106-107）——7 bands × 2 channels = 14 行（约束数）。

[事实] 列布局：channel1 = Q 块（feature 偏移 (b-1)*2N+j），channel2 = P 块（(b-1)*2N+N+j）（src/predict.jl:1168-1169；src/residual_oracle.jl:141-142）。

[事实] h[c]=tr(G[:,cols[c]])（src/response.jl:128）；M=CΩC' 的 14×14（constraint_moments，109-129）。

[事实] 条件化均值：λ=M^{-1}h；K=Σ λ_r * Σe_cols 外积；G_c=G−K V（src/response.jl:131-143）。这是 exact Gaussian conditioning，不是事后均值减 trace。

[事实] fit 内 conditioning 在 full N 空间（src/response.jl:1199-1202）；EB 内部把约束用 gauge=relative_gauge(N) 压缩到 N−1（420-424、1177）。

[事实] 约束后 trA/trB 统计（src/response.jl:1206-1207）；dense V 版本（测试）144-148。

分类：identical（约束=SPEC §21 的 trA_b=0 与 trB_b=0，对全部 7 个 band；条件化在 posterior support 内精确执行）。

---

## 9. Numerical integral：S 与 adaptive

[事实] generate_scenarios_v1 默认 S=500（src/predict.jl:546）。

[事实] backtest_v1 默认 S=300（src/backtest.jl:117）；path_kelly_v1 默认 S=300（src/kelly.jl:173）。

[事实] 正式回测默认 adaptive=false（src/backtest.jl:119）；固定分支：generate_scenarios_v1(model;S,rng=MersenneTwister(seed+t))（250），S 来自参数（默认 300）。scenario_counts 记录实际 S（293）。

[事实] adaptive 分支存在：adaptive_scenario_weights（src/predict.jl:627-653；min_scenarios=64、max_scenarios=512 默认，628）；doubling（634-635）；收敛双条件 norm(weights-previous,1)<=weight_tol 且 certificate.objective_gap<=tol（646-648，默认 1e-3 / 1e-5）；不收敛 error（652）。backtest 的 adaptive 路径（src/backtest.jl:71-77）。

[事实] 固定 S=300 的显式仓库用法：archive/scripts/local_panel_stages.jl:118-119、archive/scripts/earlier_window_replay.jl:18 与 61（scenario_counts==fill(300,8)）、archive/scripts/release_freeze.jl:54。

[推断] 证书范围：kelly_certificate 只证明"给定这套 S 个场景样本内"的 KKT/可行性/gap（src/kelly.jl:21-36）；adaptive 的双条件是 S 翻倍过程中的经验门限收敛；**固定 S=300 的非 adaptive 路径没有任何 S 收敛证书**（SPEC §65 的 ‖w_2S−w_S‖→0 只在 adaptive 模式里以门限形式体现，而它默认关闭）。

分类：finite-sample approximation（固定 S 的蒙特卡洛/拟蒙特卡洛样本；adaptive 存在但非默认，且证书是门限而非极限证明）。

---

## 10. Decision：kelly_weights_v1 目标（含 base_s）

[事实] kelly_weights_v1(X;budget=1.0,base=nothing)（src/kelly.jl:132-134）→ fast_kelly_solver：max (1/S)Σ_s log(X_s·w + base_s)，s.t. w>=0，Σw=budget（src/kelly.jl:60-130；目标构造在 102、107-108）。

[事实] base_s locked holdings：locked_wealth(X,locked) 对每 held 列乘 gross，held 非 finite 抛错（src/kelly.jl:137-148）。

[事实] scenario_weights：locked=current .* .!free；budget=1−sum(locked)；out=copy(locked)；free 列解 Kelly（src/kelly.jl:150-170）。

[事实] 证书：kelly_certificate 的 feasibility/kkt/objective_gap（src/kelly.jl:21-36）；fast 失败回退 clarabel（129）；clarabel 解同一 log-Kelly 目标（39-58）。

[事实] backtest 持仓结转：h=(w.*gross_clean)./(1+ret)（src/backtest.jl:301）。

分类：identical（exact log-growth 目标；locked 风险以 base_s 计入，非当 cash）。

---

## 11. PONY 静态推演：(a) 置 false 后果、(b) 进入 active 日期

定位事实：
- [事实] universe.txt:46 为 PONY（universe 共 65 个标的，PONY 是第 45 个 symbol；行 2..66 为代码，行 1 为注释）。
- [事实] 全仓检索未定位到名为 PONY wrapper 的脚本；PONY 字符串唯一出现在 universe.txt:46。仓库中存在的同类操作是"复制前缀"而非"置 false"：archive/scripts/earlier_window_replay.jl:37、50；archive/scripts/release_failure_probe.jl:15；archive/scripts/release_freeze.jl:103-104（Bars(... b.bar[1:t,:])）。
- 因此 (a)(b) 为按当前代码语义的纯静态推演（无 wrapper 文件可引）。

(a) 若把 PONY 前 251 行 bar 置 false（假设性）：
1. signal_prices：前 251 行 PONY 列全部 NaN（src/data.jl:37）。
2. active_universe_indices(signal)：PONY 需要至少一条相邻 finite return（src/predict.jl:130）；前 251 行全 NaN 时最早合法对是 (252,253)。故决策 prefix T_raw=252 时 PONY 仍非 active；T_raw>=253 时 active（cache 路径同结论：first_return=252，findall(<(T_raw),·) 需 T_raw>=253）。
3. r 层面：r[252]=logs[253]−logs[252]（diff 语义，src/predict.jl:247）——必须有第 253 行 finite 才产生这条 return。
4. 若 wrapper 只改 b.bar 而不改 b.adj：账户 marking 不受影响（src/backtest.jl:296 用 adj），但 free（244）与 signal 都受影响；bar 也是保存落盘的判据（src/data.jl:80-82）。
5. 训练门槛：active 仅是坐标空间准入；ts_total=WARMUP:T-2=256:T-2（src/predict.jl:286）需 n_res>=2 且 F_folds>=2（288）；ruler 另有 4*tau<=T−f+1（src/geometry.jl:38），f=252 时 tau=1 需 T>=255。

(b) 若 2025-12-01 是（前 251 行被清后）首个有效 bar（即行 252）：
- 主解读（与 (a) 语境一致，[推断]）：首日当天不足以判定 active；PONY 要到**行 253（2025-12-01 之后的下一个有 bar 交易日）**才进入 active。off-by-one 根因：active 需要 return，而 return 属于相邻 finite 对的第二日。
- 若行 253 无 bar，则顺延到下一个与前一个有效 bar 相邻的有效 bar 日。
- 精确日历日期：[未知]（未读 panel CSV 的日期列；静态只能给出"下一个有效 bar 日"）。
- 另一种解读（若 PONY 此前已累积 251 个有效 bar、2025-12-01 是第 252 个）：则当天即 active（第 251、252 个 bar 相邻）。该解读与 (a) 的"前 251 日置 false"矛盾，故不作为主结论。

分类：本节的 (a)(b) 为静态推演，非运行证据；wrapper 文件本身标记为"未定位"。

---

## 12. 映射总表

| # | 理论声明（AGENTS.md/SPEC） | 当前实现 | 证据（file:line） | 差距分类 |
|---|---|---|---|---|
| 1 | H_t 全价格历史；missing≠0（§3/§4/§8） | signal_prices；r=diff(log signal) | src/data.jl:37；src/predict.jl:216,247 | identical |
| 2 | 观测 mask O（§8.1） | bar / observed / alive_now 三表征同源 | src/data.jl:12,28；src/predict.jl:68；src/prepare.jl:131,160-161 | identical |
| 3 | active universe（§11） | active_universe_indices / first_return | src/predict.jl:125-135,207-208；src/numerics.jl:150 | identical |
| 4 | （无独立 E^trade 声明） | backtest free=active∩bar_t | src/backtest.jl:242-245；src/kelly.jl:152-153 | unknown |
| 5 | 物理可交易性（§41） | broker.tradable（bid/ask/last） | src/broker.jl:136 | identical |
| 6 | Bayesian response posterior（§17/§19/§25/§26） | α、Σ EB 点估计 + 条件 Gaussian G；场景从 L_rel/var_m 采样 | src/response.jl:934-1209；src/predict.jl:570-572,612 | plug-in approximation |
| 7 | OOF innovation law（§30/§31/§32） | scalar macro fractional × 均匀行 bootstrap | src/predict.jl:493-499,570-620；src/residual_oracle.jl:216-260 | scalar degeneration + finite-sample approximation |
| 8 | trA_b=0、trB_b=0（§21） | 14 列块 exact conditioning | src/response.jl:106-143,1199-1202 | identical |
| 9 | 场景/积分收敛（§37/§65） | 默认固定 S=300；adaptive 可选（64→512 门限） | src/backtest.jl:117,119,250；src/predict.jl:627-653 | finite-sample approximation |
| 10 | exact log-Kelly（§6/§38） | kelly_weights_v1 + base_s locked | src/kelly.jl:60-170 | identical |

---

## 13. 未能确定 / 未决

- [未知] PONY wrapper 脚本不在本仓库可见位置（全仓检索仅 universe.txt:46 命中 PONY）。
- [未知] (b) 的确切日历日期：需要 panel 的实际交易日历（CSV 日期列未读）。
- [未知] α、Σ 点估计对 predictive 覆盖与权重分布的数值影响（需受控运行，非静态可判）。
- [未知] live 层（src/live.jl）未逐行阅读；broker/live 与 backtest 的交易资格同构性未在本次核对（AGENTS.md 的 live 描述仅作背景）。
- [推断] 本地 panel 的列序与 universe.txt 一致、PONY 列号=45（基于 local_panel_stages.jl:86/97 的 N==65 与 universe 顺序；未读 CSV 头）。

---

## 14. 结论摘要（一句话）

理论链 H_t→Pi(G,Sigma|H)→P(r|H)→w* 在"数据/signal 语义、观测 mask、active 准入、trace 约束、Kelly 目标"五处是 identical；集中差距在统计认识论与数值积分三层：
1. response 的 (α,Σ) 是 EB 点估计插件（G 仅条件采样）——plug-in approximation；
2. innovation 是 macro 标量 fractional × 经验行 bootstrap（横截面退化）——scalar degeneration + finite-sample approximation；
3. 默认回测固定 S=300、无 S 收敛证书（adaptive 存在但默认关闭）——finite-sample approximation。
