# KTrader / Path Kelly — Gate-0 复现 SPEC 指南

| 项 | 值 |
|---|---|
| 文档状态 | 复现指南（Reproduction SPEC Guide）。供未参与历史的工程师从零复现 Gate-0 当前线 |
| 基准 | 工作树 HEAD `602b897` + 未提交改动；全部行号与行为来自本指南写作时点的只读源码核对 |
| 规范优先级 | `AGENTS.md`（Gate-0 裁决书 D-001~D-097 与历任登记段）> `docs/GATE0_MANAGER_ADJUDICATIONS.md`（裁决 A~I）> 各设计文档 > 本指南。本指南与源码冲突时以源码为准 |
| 适用版本 | `KTrader 2.0-RC-G0`（`RELEASE.toml` 的 `status`）。旧 `KTrader`（2.0-RC legacy implementation）冻结为历史 fixture，不在本指南范围 |
| 当前入口 | `bin/backtest.jl` → `KTraderGate0`（`src/gate0/`）。唯一当前生产入口 |
| 写作纪律 | 本文只描述已核实的行为。未能核实处标注「未核实」。命令模板遵守仓库护栏：单命令 ≤60s、RSS ≤2048MiB、`bin/scoped_run.sh` 受控执行 |

阅读约定：

- 记号沿用裁决书：$\mathcal H_t$ 为价格历史，$r$ 为日对数收益，$s_1$ 为一日 ruler，$m$ 为 macro 标量，$q$ 为 relative field 的 gauge 坐标，$N$ 为 mode 维数（= active 资产数），$P$ 为 feature 维数。
- 标签含义：[规范] = 裁决书或 SPEC 明文；[实现] = 当前源码事实；[边界] = 已知限制或未决项。
- 「fail loudly」= 抛错并停止。禁止静默替换、静默降级、返回启发式结果（SPEC §56）。

---

## 0. Mise en place：三条不可违背的纪律

复现前先接受三条纪律。它们决定你如何调试、如何调参、如何读结果。

**纪律 1 — fail loudly。**
任何数学条件不满足时必须抛错。当前线中所有失败文本是验收锚点。不要吞错、不要补默认值。

**纪律 2 — 容差不由运行预算决定。**
运行超时的正确行为是 fail 或缩小 fixture。禁止为了跑完而放宽 `posterior_tol`、`weight_tol`、`utility_tol`、`kelly_tol`（P0-7 裁决；D-066 定稿流程）。

**纪律 3 — 结果与验收判据分离。**
回测净值不是验收依据。验收判据是证书：A、B_opt、B_audit、C（第 11.3 节）。禁止用收益/Sharpe 选择任何数值参数或模型开关（SPEC §95、开发守则 §24）。

另一条操作纪律：单命令 ≤60 秒、RSS ≤2048MiB，经 `bin/scoped_run.sh` 执行（第 2.4 节）。

---

## 1. 系统总览

### 1.1 理论链（规范）

$$
\boxed{
\mathcal H_t
\to \text{确定性路径几何}
\to \text{统一 macro/relative mode 坐标}
\to \Pi(b_0,G,\alpha_0,\alpha_p,\Sigma_R\mid\mathcal H_t)
\to \text{prequential 残差}
\to q(d\mid\mathcal H_t)\to V_t(d)\to P(\varepsilon_{t+1}\mid\mathcal H_t)
\to P(R_{t+1}\mid\mathcal H_t)
\to \arg\max_{w_i\ge 0,\ \sum_i w_i\le 1} E[\log(1+w^\top r_{t+1})\mid\mathcal H_t]
}
$$

现金 `w_cash = 1 − Σ w_i` 是显式 numeraire 资产（D-068）。

顶层名称（D-042）：**modular posterior predictive** = Bayesian response posterior + cross-fitted semiparametric innovation predictive。在 innovation 仍为经验测度期间，禁止称 fully Bayesian generative posterior predictive。

### 1.2 两条代码线

| 线 | 模块 | 状态 | 入口 |
|---|---|---|---|
| 当前线 | `KTraderGate0`（`src/gate0/`，模块 include 12 文件 + `market_io.jl` 由 CLI 额外加载） | Gate-0 关闭（2026-10-10）后的当前线 | `bin/backtest.jl`；测试 `test/gate0/runtests.jl` |
| 历史线 | `KTrader`（`src/`，旧 13 文件） | 冻结为 historical fixture（D-002/D-082） | 历史工具 `bin/bench.jl` / `bin/report.jl` / `bin/live.jl`；旧测试 `test/runtests.jl` |

两线**互不 include**（裁决 H1）。唯一数据桥：`bin/backtest.jl` 用 `KTrader.load_bars` 取裸矩阵，经 `KTraderGate0.load_market_facts` / `from_bars_arrays` 构造 `MarketFacts`。

### 1.3 目录结构

```text
src/gate0/            KTraderGate0（当前线，单模块 12 文件）
src/                  KTrader（历史线，冻结）
test/gate0/           当前线测试（16 文件；统一入口 runtests.jl）
test/                 历史线测试（冻结）
bin/backtest.jl       当前唯一生产入口
bin/scoped_run.sh     ≤60s / RSS 护栏（唯一受控执行工具）
bin/fetch.jl          数据下载（双线共用）
bin/verify_release.jl 2.0.0 历史发布物校验（历史语义）
docs/                 规范、设计与审计文档
dev/                  开发资产（probes、M1 重放、验收驱动）
archive/              已离场证据（git 忽略；`git show 8d54f0d:<原路径>` 取回）
data/                 本地行情（close.csv / adj.csv；不随源码发布）
```

### 1.4 模块 include 链（唯一顺序）

```mermaid
flowchart LR
    market --> geometry --> modes --> response --> posterior --> oof --> innovation --> predictive --> kelly --> quadrature --> driver --> backtest
```

约束：`src/gate0/KTraderGate0.jl` 是唯一 include owner。后文件消费前文件的公共接口；不复制数学。`src/` 不 include `dev/`（AST 门禁）。

### 1.5 KTraderGate0 导出面（90 名）

| 文件 | 导出（计数） |
|---|---|
| market | `MarketFacts, Eligibility, from_bars_arrays, signal_prices, model_admitted, trade_eligible, effective_bars_252, executable, free, locked, benchmark_universe`（11） |
| geometry | `TAUS, BANDS, BANDCOL, WARMUP, center_of_mass, ruler, relative_gauge, principal_sqrt_root, compute_s_perp, path_basis_1d, fast_s_m`（11） |
| modes | `domain_mode_basis, risk_domain_mode_basis, mode_field, asset_field_reconstruct, cumulative_path_coordinates, mode_basis_design`（6） |
| response | `build_mode_problem, fit_full_block_ridge, predict_mode, block_views`（4） |
| posterior | `SufficientStats, ConditionalFit, ProprietyCertificate, ResponsePosterior, sufficient_stats, s_alpha, log_evidence, log_prior_d035a, adaptive_quadrature_2d, fit_full_posterior, predict_mu, draw_mu, mu_asset, oof_residual_rows`（14） |
| oof | `fold_grid, fold_stats, train_stats, fit_fold_posteriors, oof_residual_rows_full`（5；`prequential_residual_rows` 不导出） |
| innovation | `InnovationState, innovation_state, SupportCertificate, DConditionedShapePool, draw_innovation, z_pool_rows, z_row_at, V_at, shape_moments, quasi_loglik, joint_row_indices, coverage_report, resolve_risk_domain, default_d_grid, initial_mesh, adaptive_d_quadrature, residual_rank, rank_sufficient, frac_weights_gate0, mp_sqrt_factors`（20） |
| predictive | `PredictiveLawResult, predictive_law`（2） |
| kelly | `cash_kelly, locked_wealth_gate0, cash_kelly_certificate, cash_kelly_certified`（4） |
| quadrature | `SobolOwenRule, rule_points, audit_seed, RQMCScenarioSource, predictive_rqmc_source, AdaptiveKellyResult, adaptive_scenario_kelly`（7） |
| driver | `Gate0DayDecision, single_day_decision, equal_weight_benchmark`（3） |
| backtest | `Gate0DayRecord, Gate0BacktestResult, run_gate0_backtest`（3） |

### 1.6 关键裁决点速查（复现时最容易踩错的地方）

| 主题 | 裁决 | 一句话 |
|---|---|---|
| 观测 vs 政策 | D-012/D-014/D-016 | `observed` 只表示真实观测；交易资格另有独立 mask；free 不含 model admission |
| locked 风险 | D-017 | 已持仓但不可交易 = locked risk，进 Kelly 的 base，绝不当 cash |
| benchmark 外生 | D-020 | 对照组候选集只来自 `E^trade ∧ T^exec`，不依赖模型 active set |
| trace 约束 | D-028 | 14 条 per-band trace 约束是 candidate hypothesis，不在默认 core |
| 统一响应 | D-025/D-026 | macro↔relative cross block 全开，禁止 block-diagonal |
| DC 通道 | D-031/D-032 | 统一输入首列 = 常数列；DC 先验零中心，非零后验只能来价格证据 |
| 超参积分 | D-034/D-038 | α₀/α_p 不再点估计，log 空间确定性求积 |
| 响应协方差 | D-037 | 给定 α 后解析积分 B 与 Σ_R；决策时只边缘化 μ = B·x̃ |
| 残差 | P0-1 | production 残差 = strict prequential，3-fold OOF 降级为 diagnostic |
| innovation | D-045/D-046 | 统一 mode 坐标，vector 记忆 $V_t(d)$，一个 kernel，无 scalar 缩放锚 |
| support | D-050/D-051 | Moore-Penrose；sample null space 不得当物理零风险；禁 $V+\delta I$ |
| d 先验 | D-047 | $d \sim U(0,1)$；网格只是数值 quadrature；cell mass 必须先验不漂移 |
| 拼接 | D-054/D-055 | 一个 scenario innovation 来自一个联合合法行；禁 own-cell stitching |
| 覆盖不足 | D-056/C3 | free 收缩 → 资金留 cash；locked 导致 → fail loudly → 当日维持持仓 |
| 积分收敛 | D-061~D-067 | 生产必须 adaptive；四证书（A/B_opt/B_audit/C）；耗尽 fail loudly；无固定 S 捷径 |
| cash Kelly | D-068/D-069 | cash 是显式 numeraire；完全投资 risky-only 不再称 canonical |
| 集中诊断 | D-075/D-076 | 高集中不是独立 bug；必须输出 cause report 字段 |
| 名称 | D-096/D-097 | 见第 16 节禁用清单 |
| T4 判定次序 | 裁决 C3 | 判定在 posterior 之前（不重排 = 零模型计算）；二次 resolve 带 residual_rows 做满秩 gate |
| P0-7 | 纪律 | 容差恢复严格 reference 口径；超时的正确行为 = fail / 缩小 fixture |

---

## 2. 复现前置条件

### 2.1 环境

| 项 | 要求 |
|---|---|
| Julia | 1.12.7（`RELEASE.toml` 的 `qualified_julia`） |
| BLAS | OpenBLAS（随 Julia 发行）。`bin/backtest.jl` 在启动时执行 `BLAS.set_num_threads(1)`（U2 裁决：gate0 load 下小 GEMM 在多线程 BLAS 下约 3× 慢） |
| OS | Linux（`/proc/uptime` 供 `scoped_run.sh` 单调时钟） |
| 依赖 | `Manifest.toml` 锁定。关键依赖：Convex、Clarabel（Kelly reference）、Distributions（matrix-t / Chisq 抽样）、CSV（数据适配器）、Test（测试） |
| depot | `~/.julia` 必须完整（历史事故：depot 缺失导致 `failed to find source of parent package`） |

### 2.2 安装步骤

前置：网络可用（或本地 depot 备份）。

```sh
cd <repo>
julia --project=. -e 'using Pkg; Pkg.instantiate()'          # 按 Manifest 安装
julia --startup-file=no --project=. -e 'using KTrader; println(Base.pkgversion(KTrader))'
```

验证：第二条命令输出 `2.0.0` 包身份（历史线包名；Gate-0 线经 `bin/backtest.jl` 使用，不改变包版本号）。

### 2.3 数据准备

`data/close.csv` 与 `data/adj.csv`，两文件列集和顺序必须一致：

- 首列名 `date`（ISO 日期，严格递增）。
- 其余列为资产代码，无重复。
- 缺失值留空（解析为 NaN）。有限值必须为正。

`load_market_facts` 的行为（`src/gate0/market_io.jl`）：

1. 校验两表列身份/顺序一致、日期一致、日期严格递增；违反即抛错。
2. `observed[t,j] = isfinite(close) && isfinite(adj)`。
3. `close`/`adj` 落盘为 carry-forward marking 序列（上市前 NaN，之后沿用最后观测值）。
4. 返回 `MarketFacts`。理论 signal 由 `signal_prices(mf) = ifelse.(observed, adj, NaN)` 恢复——缺观测处严格 NaN，绝不把 carry-forward 当收益。

也可用 `bin/fetch.jl` 下载（不随包发布行情）。

解析细则（fail loud）：CSV 用 `normalizenames=false`，列名即资产代码；有限值必须 > 0（0 或负值、Inf 抛错）；缺失（空）解析为 NaN 合法。**历史线的数据层**（`src/data.jl`/`load_bars`）语义兼容——`from_bars_arrays` 桥只搬运裸矩阵，不跨模块依赖数学层。

### 2.4 执行护栏（所有运行必须经它）

```sh
bash bin/scoped_run.sh <deadline_s> <logfile> [--rss-guard=<MiB>] <command> [args...]
```

- `deadline_s` ≤ 60（仓库硬约束；常用 45~50）。外层调用者 deadline 必须严格大于内层。
- `--rss-guard=<MiB>` ≤ 2048；采样级 RSS kill 护栏（≤100ms 间隔轮询进程组 VmRSS）。
- 退出码：`124` = 超时或 RSS 超限（永远不是成功）；`125` = 资源违约 fail-closed；`128+signum` = wrapper 被取消；其余透传子命令退出码。
- 日志尾行为统一格式：`scoped_run: rc=... elapsed=... deadline=... killed=... rss_guard=... rss_peak=...MiB`。验收时读这一行。

标准模板：

```sh
bash bin/scoped_run.sh 50 /tmp/gate0_smoke.log --rss-guard=2048 \
  julia --startup-file=no --project=. <target>
```

注意：Julia 首次运行含编译（约 25–30s）。计时前先预热（先运行一次 `julia --project=. -e 'using KTrader'` 或直接接受第一次的编译成本）。

---

## 3. 数据层：四 mask（`src/gate0/market.jl`）

### 3.1 类型

```julia
struct MarketFacts
    dates::Vector{Date}
    symbols::Vector{String}
    close::Matrix{Float64}    # marking 序列（仅账户估值）
    adj::Matrix{Float64}      # marking 序列（仅账户估值）
    observed::BitMatrix       # O_{t,i}：唯一语义「日期 t 资产 i 有真实市场 bar」
end
```

构造不变量（违反即抛错）：维度一致；`observed[t,i]=true` 蕴含 `close`、`adj` 均 finite。构造函数对全部字段做防御性 copy。**不存在任何 mutation 接口**——「因不想交易/历史不足/模型不 admission/风控而改 observed」在类型层不可表达（D-012 四禁）。

```julia
struct Eligibility
    model_admitted::BitMatrix    # A^model_{t,i}
    trade_eligible::BitMatrix    # E^trade_{t,i}
    executable::BitMatrix        # T^exec_{t,i}
end
```

### 3.2 四 mask 的判定规则

| mask | 规则 | 裁决 |
|---|---|---|
| `observed` | 物理观测事实。永不改写 | D-012 |
| `model_admitted[t,j]` | prefix 1:t 内至少一条有效 daily return（相邻两日均有观测且 adj finite）。t 单调不减；全 NaN dummy 全列 false；IPO 首个 admitted 日为「首条 return 的次日」 | D-013 |
| `trade_eligible[t,j]` | 当日是否允许新建/增加风险仓位。默认全 true。252 规则实例 `effective_bars_252`：按 O 计数，累计第 252 个有效 bar 当日解除（H4） | D-014/D-018 |
| `executable[t,j]` | 理论 backtest 简单形式 `= observed`（独立副本）。live 由 broker 报价供给，独立不校验 | D-015 |

派生量（函数拥有）：

- `free(el, t) = trade_eligible[t,:] .& executable[t,:]`——**model_admitted 不参与**（D-016）。
- `locked(held, el, t) = held .& .!free(el,t)`——held 但 free=0 的资产为 locked risk，**绝不当 cash**（D-017）。
- `benchmark_universe(el, t) = trade_eligible[t,:] .& executable[t,:]`——benchmark 候选集外生，与 model admission 无关（D-020）。

### 3.3 复现验证点

- `signal_prices`：`observed=false` 处必须 NaN。
- `model_admitted`：dummy 资产严格不影响其余资产；单日 missing 不回退。
- `effective_bars_252`：第 251 个 bar 当日仍 false，第 252 个 bar 当日 true。
- mask 只读：对返回 mask 做任何修改不污染 `MarketFacts`。

---

## 4. 几何层（`geometry.jl` / `modes.jl`）

### 4.1 常量

```julia
const TAUS   = 2 .^ (0:8)   # 1,2,4,...,256（ruler 拟合尺度）
const BANDS  = 2 .^ (1:7)   # 2,4,...,128（path basis 时间频带）
const BANDCOL = ...          # BANDS 在 TAUS 中的列索引
const WARMUP = 2 * maximum(BANDS)  # = 256
```

`WARMUP` 是所有训练行、决策日的前置下界：决策日 t 需 `t-1 >= WARMUP`（即 t ≥ 257）；backtest 需 `t_start-1 >= WARMUP`。

### 4.2 ruler（价格尺度）

`ruler(x, f, w=nothing) -> N×9`。对每资产在 log 价格上做 power-law 拟合的 τ-increment RMS 平滑，得 $s_j(\tau)$；`s1 = s[:,1]` 是一日 ruler。

- 有效 tau 需满足 `4τ ≤ T - f_j + 1`。
- 短历史回退（`n_use < 2`，2026-10-10 修订）：用该资产 **τ=1 增量 RMS** 作所有 τ 的平坦外推（H=0）。`ws==0 → 1.0` 为不可达兜底。
- ruler 是确定性历史泛函，不是 posterior 随机对象（D-024）。

### 4.3 固定 Helmert gauge 与域基

- `relative_gauge(N) -> N×(N-1)`：Helmert 构造，`QᵀQ=I`、`Qᵀ𝟏=0`。纯数值 gauge，不是板块也不是 factor（D-023）。
- `domain_mode_basis(n) = [e0  Q]`，`e0 = 𝟏/√n`（域级等权）。
- 语义分层（裁决 E1）：**域基**（active 域 $E_{active}$、risk 域 $E_{R_t}$）按域大小等权；**行级 field** 的 $m_s$ 按该行观察集归一（$1/\sqrt{|O_s|}$）。两者不同概念。

### 4.4 统一 mode 坐标（`modes.jl`）

**行级 field**（`mode_field`）：

$$
u_i = r_i/s_{1,i},\qquad
m_s = \frac{1}{\sqrt{|O_s|}}\sum_{i\in O_s} u_i,\qquad
e_{s,i} = \begin{cases} u_i - \overline{u}_{O_s}, & i\in O_s\\ 0, & i\notin O_s\end{cases}
$$

- 空观察行：`m_s=0`、`e=0`（零嵌入；该行是否入统计由消费层行集判定）。
- 零嵌入只表示 field 代数，不表示 r=0；observation mask 永久单独保留（D-022）。
- `q_s = Qᵀ e_s`；输出 `y = [m; q] ∈ R^N`。

**累积路径**（SPEC §14）：$X_m(t)=\sum_{s\le t} m_s$，$X_\perp(t)=\sum_{s\le t} e_s$。Q/P basis 在 level 上构造。

**Q/P path basis**（`path_basis_1d`，SPEC §16）：每个 band 两个通道：

$$
Q(t,\tau) = -\frac{X(t)-c_0}{s(\tau)},\qquad
P(t,\tau) = \frac{c_0 - c_1}{s(\tau)}
$$

$c_0$ = 近 τ 窗均值，$c_1$ = 前一 τ 窗均值。列序 `(band1 Q, band1 P, band2 Q, band2 P, …)`，每 band 2 列，共 14 列。

**标量 relative ruler**（`compute_s_perp`）：每 band 一个 gauge-invariant 标量。禁止逐坐标归一（破坏 Q→QR 协变性）。

**统一设计**（`mode_basis_design`）：

- `B_m = path_basis_1d(X_m, s_m)`：T×14。
- `X_q = X_rel * Q`（先累积再投影 = 先投影再累积）。
- `B_perp`：对 X_q 每 gauge 坐标做 `path_basis_1d(..., s_perp)`；T×14(N−1)。列布局：每 band 占 2(N−1) 列，Q 通道块在前、P 通道块在后。

**统一输入拼接**（含 DC，Step 5）：$x̃ = [1;\ B_m;\ B_\perp] ∈ R^{14N+1}$——列 1 = DC、列 2:15 = B_m、列 16:14N+1 = B_⊥。

### 4.5 mode→asset 重构

`asset_field_reconstruct(mode_vec, E) = E * mode_vec`。全观测行精确还原；ragged 行因域基与行级归一的差别不逐点还原（语义，不是 bug）。

### 4.6 必须测试的不变量

- 置换协变：资产列置换下预测/权重协变（SPEC §58）。
- gauge 旋转：$Q\to QR$ 下 asset 空间对象不变（SPEC §59）。
- dummy 全 NaN 资产：不影响其余资产。
- 线性交换性：`X_rel*Q` = 先投影再累积。

---

## 5. 响应层 A：固定 ridge reference（`response.jl`，Step 4/5）

用途：验证 design/target 代数、四块 cross-response 恢复（D-092）、DC 通道（D-093）。**不做** posterior、不做 trace conditioning、不做 EB。

### 5.1 组装与配对

`build_mode_problem(returns, observed, s1, E; s_m=nothing, s_perp=nothing, dc=false, compact=false)`：

- 行 t 的 feature 见 t 时刻信息；配对 `X = X_full[1:T-1,:]`、`Y = y_modes[2:T,:]`——Y 行 s 是 X 行 s 的下一日输出。
- `dc=true` 时 X_full 首列恒 1，共 1+14N 列；否则 14N 列。
- 返回 `(; X, Y, X_full, y_modes, B_m, B_perp, X_q, X_m, X_rel, m_series, e_embedded, E, Q, N, T, s_m, s_perp)`。

### 5.2 固定 ridge 拟合

`fit_full_block_ridge(X, Y; alpha=1.0)`：

$$
\min_B\ \|Y - XB^\top\|_F^2 + \alpha\|B\|_F^2,
\qquad
B^\top = (X^\top X + \alpha I)^{-1} X^\top Y\ \ (\text{Cholesky 解})
$$

- 四块全部由数据决定：`block_views(B)` 返回 `(b0, B_mm, B_mperp, B_perpm, B_perpperp)`——行块 = 输出 mode，列块 = 输入 mode。**不施加 block-diagonal、不施加 trace 约束**（D-025/D-026/D-028）。
- DC 形态首列必须恒 1，否则抛错（D-031）。
- `alpha` 是调用方固定正数（数值正则），不是策略参数。
- 零中心纪律（D-032）：DC 先验均值必须为 0；非零 posterior 只能来自价格证据。

### 5.3 预测

`predict_mode(B, x_t) = B * x_t` → N 维 mode 预测。

---

## 6. 响应层 B：完整后验（`posterior.jl`，Step 8）

### 6.1 模型与先验（RP §1-§2）

$$
y_{t+1} = b_0 + G\,x_t + \varepsilon^w_{t+1},\qquad
Y = X B^\top + E,\qquad
B \mid \Sigma_R,\alpha \sim \mathcal{MN}(0,\ \Sigma_R,\ \Lambda_\alpha^{-1})
$$

- $\Lambda_\alpha = \operatorname{diag}(\alpha_0,\ \alpha_p,\ldots,\alpha_p)$（D-033 双 group：零频 DC ≠ dynamic path；禁止 per-band/per-asset/ARD）。
- 先验（P0-8 裁决后的当前形式，`log_prior_d035a`）：$\tau \sim \mathrm{HalfCauchy}(0,1)$，即 $p(\alpha)\propto 1/[\sqrt{\alpha}(1+\alpha)]$；log 坐标 $u=\log\alpha$：
  $$
  \log p_u(u) = \tfrac12 u - \log(1+e^u)
  $$
  两端自然衰减，proper。$\Sigma_R$ 用 Jeffreys prior $|\Sigma_R|^{-(N+1)/2}$（RP §2.2）。先验形式不允许由回测选择。

### 6.2 充分统计与解析积分（RP §3-§4）

**A1**：后验只依赖 $(n, S_{xx}=X^\top X, S_{xy}=X^\top Y, S_{yy}=Y^\top Y)$。

给定 $\alpha$：

$$
S(\alpha) = Y^\top C^{-1} Y = S_{yy} - S_{xy}^\top (S_{xx}+\Lambda_\alpha)^{-1} S_{xy}
$$

**高危陷阱**：$S(\alpha)$ 不是 ridge 残差平方和（只在 α→0 极限重合）。实现见 `s_alpha`。

marginal evidence（`log_evidence`）：

$$
\log p(Y\mid\alpha) = \mathrm{const} + \tfrac{N}{2}\log|\Lambda_\alpha| - \tfrac{N}{2}\log|S_{xx}+\Lambda_\alpha| - \tfrac{n}{2}\log|S(\alpha)|
$$

常数与 α 无关、省略不影响形状。$B\mid Y,\alpha$ 为 matrix-t；决策时只边缘化 $\mu=B x̃_t$：

$$
\mu\mid Y,\alpha \sim t_\nu\!\left(\hat B x̃_t,\ \frac{c}{\nu}S(\alpha)\right),\qquad
\nu = n+1-N,\qquad c = x̃_t^\top (S_{xx}+\Lambda_\alpha)^{-1} x̃_t
$$

**ν 绑定（裁决 G7）**：`ν = n+1−N`，禁止 `n−N`。

### 6.3 2D 超参求积（`adaptive_quadrature_2d`）

在 $(\log\alpha_0,\log\alpha_p)$ 平面做确定性自适应 cell 积分：

- 主规则：张量积 **2 点 Gauss-Legendre**（每维节点 $\pm 1/\sqrt3$、权 1；二维权积归一 /4）。
- 误差代理：3 点 G-L 对照（节点 $0,\pm\sqrt{3/5}$、权 8/9,5/9）差 × 安全因子 s：$s\cdot|I_{3\times3}-I_{2\times2}|$。
- 细分：误差最大的 cell 沿曲率大的轴向二分（`split_axis=:curvature`，可选 `:longest`）。
- 收敛：$\sum$ cell 误差 ≤ tol·Z。
- 默认：`tol=1e-6`、`max_cells=8192`、`init_grid=4`、`tail_rel=1e-10`、`m_ref` 传粗扫描 mode。
- **尾质量证书**：域边界 8 点（4 角 + 4 边中点）必须满足 `logf ≤ refval + log(tail_rel) + log(area)`；违反即抛错（proper prior 下若红 = 实现 bug）。
- 预算耗尽：`error("Numerical integration did not converge: budget exhausted ...")`，禁止返回最后一层。
- 无 DC 形态（P=14N）走一维路径 `_adaptive_quadrature_1d`（α₀ 不在模型，先验积分是常数）。

### 6.4 主入口与返回

`fit_full_posterior(X, Y; prior=:d035a, tol=1e-6, max_cells=2048, alpha_fixed=nothing, u_span=5.0, rank1=true, mean_at=nothing, posdef_diag=nothing)`：

1. `sufficient_stats`。
2. **A2 数据层 gate**：`n ≥ N` 且 `rank(Syy) = N`，否则 `error("fit_full_posterior: posterior improper — data-layer condition failed ...")`。
3. `has_a0 = (P == 1+14N)`。
4. 粗扫描 mode（21×21 网格覆盖 [−12,12]²，仅作初始域中心与 `m_ref`，不定义 posterior）。
5. 求积（DC 形态 2D；否则 1D）。cell 预算：2D 用 `16*max_cells`（默认 2048 → 32768）；1D 用 `4*max_cells`（→8192）。
6. 每节点 `_fit_node` → `ConditionalFit`。

`ResponsePosterior` 字段：`n, N, P, has_a0, prior, alpha_nodes, alpha_weights, node_log_evidence, evidence, propriety, conditional, nu, stats, X, Y`。

`ConditionalFit` 字段：`B_hat`（N×P，可 dense 或惰性谱表示）、`V`、`V_fact`（precision 因子；`V_solve` 解的是 precision 系统）、`S_alpha`、`nu`、`weight`。

`ProprietyCertificate`：`a2_n_ge_N, a2_rank_syy_eq_N, a4_tail_ok, proper`。

### 6.5 消费接口

`predict_mu(post, x_t) -> (; mu, cov, within, between, within_computed)`：

$$
\mu = \sum_k p_k \hat B_k x̃,\qquad
\text{within} = \sum_k p_k \frac{c_k}{\nu-2} S(\alpha^{(k)}),\qquad
\text{between} = \sum_k p_k \hat\mu_k\hat\mu_k^\top - \mu\mu^\top
$$

- `within` 需要 $\nu>2$（n ≥ N+2）；否则 `within_computed=false`、`within=NaN`——**NOT COMPUTED，不显示 0**（D-030）。
- `cov = within + between`；`between` 是超参后验变异（旧 plug-in 实现恒 0 的对照面）。

`draw_mu(post, x_t, rng)`——matrix-t predictive 抽样（裁决 F；μ 通道唯一消耗 RNG 的入口）：

1. 按节点质量抽 k；2. $\mu = \hat B_k x̃ + \sqrt{c_k/\nu}\,L z / \sqrt{g/\nu}$，$z\sim N(0,I)$，$g\sim\mathrm{Chisq}(\nu)$，$L=\mathrm{chol}(S(\alpha^{(k)}))$。

**D-060 红线**：本抽样是 epistemic（mean uncertainty）；未来残差由 innovation 层独立提供；$\Sigma_R$ 不进入 scenario 残差通道。

### 6.6 R4 数值防御（如实登记）

S(α) 理论 PSD，但浮点下可出现 ~1e-12 量级负特征。防线：`_robust_logdet_psd`（eigen + floor，`_PSD_FLOOR_REL=1e-12` 相对 λmax；结构负值 `|λ|>1e-10·scale` 时 fail loudly）；求积路径 PosDef → 该点 −Inf + 记录，但中域失败或失败率 >0.2 时 fail loudly；最终节点构造 PosDef → fail loudly。

---

## 7. 残差源：strict prequential（`oof.jl`）

### 7.1 两条路径的分工（P0-1 裁决）

| 路径 | 函数 | 状态 |
|---|---|---|
| legacy OOF folds | `fold_grid / fold_stats / train_stats / fit_fold_posteriors / oof_residual_rows_full` | 保留为 reference/diagnostic；**不得被 production 决策链调用** |
| strict prequential | `prequential_residual_rows`（不导出） | **production 残差源** |

裁决语义：innovation 的残差必须是历史当时真正发生的 forecast surprise：

$$H_t \to \text{predict } r_{t+1} \to \text{observe } r_{t+1} \to \varepsilon_{t+1} \to \text{append}$$

3-fold OOF 对历史日期 s 的模型可以使用 s+1..t 的数据；对今天的 backtest 无 future leakage，但不是当时的 surprise——对 non-Markov innovation law 是本质区别。

### 7.2 driver 使用的生产路径

driver 不直接调 `prequential_residual_rows`；它调 `_market_prequential_residuals`（`driver.jl`）：对每个行身份 u：

1. 在 origin = u−1 上做因果预处理（`_response_prefix`，只读 ≤u−1 的价格、active 集、ruler——历史预测不得使用更晚决策日的 active 集/ruler）。
2. 在 origin 上拟合完整后验（`fit_full_posterior`，`mean_at=prep.xnow`）→ 预测均值。
3. 观察 u 日实现收益，残差：
   $$
   \varepsilon_u[j] = \big(\log adj[u,j] - \log adj[u-1,j] - \text{forecast}[j]\big) / s_1
   $$
   仅当 `observed[u-1,j] && observed[u,j]`。除 s₁ 只是单位换算，绝不回馈进历史拟合。

预测缓存（`_ForecastCache`）按 origin 冻结；「新上市资产不得改写旧预测」由缓存与因果 owner 共同保证。

### 7.3 row-undefined 语义（两类可接受失败 vs 真错误）

行级不可定义（→ 该行残差 NaN）仅限两类错误文本：

1. `"posterior improper"`（A2 propriety 未满足）；
2. `"Numerical integration did not converge"`（D-067 求积/收敛失败）。

其余错误（如 `PosDefException`、`ArgumentError`）是真正的实现/编程错误——**重抛**，不得吞成 NaN。全部行都不可定义时 `error("prequential_residual_rows: no row has a proper train posterior ...")`。

### 7.4 成本加速（2026-10-10 实现，登记）

- O(1) 增量充分统计（rank-one/外积精确更新），默认关闭时逐位不变。
- 行级并行（`Threads.@threads`）：每行只读共享输入、只写自己的输出行；**结果必须与串行逐位一致**（单线程会话自动退化为串行）。
- 误差槽：非行级错误在主线程按行序重抛（保持原错误语义）。

---

## 8. 创新层（`innovation.jl`，Step 10–12）

### 8.1 对象与域（裁决 A / D-045a）

- **risk 域** $R_t = \text{free} \cup \text{locked}$；mode 基 $E_{R_t} = [e_0^{(R_t)}; Q^{(R_t)}]$，维数 $N_R$。
- 行身份 $u_s$（目标日，严格递增）；行 mask $O_s$（VI 契约：$O_s = \{j : r[u_s,j]\ \text{finite}\}$）。
- **joint 合法行集**：$J_t = \{s : u_s\le t,\ O_s\supseteq R_t\}$。
- 残差提取路径：**asset 空间残差行 → R_t 子向量 → $E_{R_t}^{\top}$ 变换**。禁止从 active 域 mode 残差直接取子向量（mode 坐标无资产子集索引）。

### 8.2 kernel 与 $V_t(d)$（D-045～D-048）

统一 fractional family（与历史线同族；禁止 macro/relative/每资产各一套）：

$$
k_d(\tau)=\pi_{\tau-1},\qquad \pi_0=1,\qquad \pi_k=\pi_{k-1}\frac{k-1+d}{k},\qquad d\in(0,1]
$$

$$
V_t(d) = \frac{\sum_{s\in J_t} k_d(t+1-u_s)\,\varepsilon_s^{(R)}\varepsilon_s^{(R)\top}}{\sum_{s\in J_t} k_d(t+1-u_s)}
$$

- 分子分母同步收缩到 $J_t$（裁决 C1）——分母不是裸 $\sum k_d$。
- 全部因果历史，无 rolling window（D-048）；无 burn（D-049；有效起点 = 数学可定义性）。
- 构造性 PSD；秩与 d 无关（秩引理）：$\operatorname{rank}V_t(d)=\dim\operatorname{span}\{\varepsilon_s: s\in J_t\}$。
- $d=1$ 端点极限 = 等权二阶矩；$d\to0^+$ = 最近行独占；$N_R=1$ 退化为 scalar fractional 同族。
- **无任何 floor/δ 参数**：$V$ 的数学定义不含正则化（禁止 $V+\delta I$）；`V_direct` 防御性断言 $u_J$ 严格递增与因果性 $u_s \le \text{decision\_time}$（违反即抛错）。$V_{s-1}$ 以 `decision_time = u_s − 1`、前缀行集调用同一函数。

### 8.3 d 的 quasi-posterior（D-047/D-057/D-058）

先验 $d\sim U(0,1)$。Gaussian covariance quasi-likelihood：

$$
\ell_d = -\tfrac12\sum_{s\in L}\Big[\log\det{}^+ V_{s-1}(d) + \varepsilon_s^\top V_{s-1}(d)^+\varepsilon_s\Big],
\qquad
L = \{s\in J_t : V_{s-1}\ \text{可定义}\} = 2:|J_t|
$$

后验 $q(d\mid H)\propto \exp(\ell_d)\,\mathbf 1_{(0,1)}$；η≡1（禁止 temperature）。节点权重 $q_g\propto \exp(\ell(d_g))\Delta_g$（先验质量不漂移，SPEC §64）。

数值：`adaptive_d_quadrature(eps_R, u_J, t; tol_d=1e-4, tol_d2=1e-4, tol_V=1e-6, max_levels=12)` 从 `initial_mesh()`（节点 `[0.05,0.1,0.2,0.4,0.6,0.8,1.0]` + midpoint cell mass）出发做 cell 中点细分；证书 = 相邻两级 d 的 E[d]、E[d²]、E[V(d)] 稳定；耗尽 fail loudly。`default_d_grid()` 只是 `initial_mesh` 的兼容别名，不是生产固定网格。

细则：细分 M → 2M−1（新节点 = 旧节点 ∪ 各 cell 中点）；$E[V(d)]$ 的比较用 `opnorm`（谱范数，按源码实现；与文档描述的 Frobenius 不一致处按源码）；层内节点求值有精确 memo（同一 d 不重算——是同一不可变面板上的同一函数值，不是统计缓存）；耗尽错误文本含 `"Numerical integration did not converge"`。cell mass 总和（初始与细分后）恒 ≈ 1（atol=1e-9 断言，先验质量不漂移）。

### 8.4 Moore-Penrose support 协议（D-050/D-051）

`mp_sqrt_factors(V; cls_floor=nothing)`：eigen 分解，正特征 $\lambda > \text{cls\_floor}$ 判正；返回 `(half, inv_half, rank, eigenvalues, positive, cls_floor)`。

- null 方向**不逆、不注噪**、保留为 null support（variance=0 = 历史未提供证据，不是物理零风险）。
- 默认 `cls_floor = N·sqrt(eps)·max(λmax, floatmin)`；仅浮点分类，不进入 V 的数学定义；必须接受 floor→0 refinement 检验。
- 禁止 `V + δI`（偷偷创造理论风险）。

### 8.5 标准化 shape（D-052～D-056）

$$
z_s(d) = V_{s-1}(d)^{+1/2}\varepsilon_s
$$

- 经验测度（semiparametric empirical predictive）：一个 scenario innovation 必须来自**一个联合合法 shape row**（$L$ 行集）。禁止 own-cell stitching（跨日期拼向量）。
- 覆盖不足时不拼残差、不 zero-fill。
- `row_at` / `rows` / `z_pool_rows(st,d)` / `shape_moments`（诊断 $M_z$，不假设 = I）。

### 8.6 覆盖判定：`resolve_risk_domain`（裁决 C3）

输入 `(row_ids, row_masks, free_assets, locked_assets, t; residual_rows=nothing, cls_floor=nothing)`，返回 `(R_t, dropped_free)`：

1. 初始 $R^{(0)} = \text{free}\cup\text{locked}$。
2. 「足」= `|J| ≥ 2`；若传 `residual_rows` 还要求 `rank{ε_s^{(R)}: s∈J} = |R|`（P0-3：满秩⇒$V_t(d)\succ0$ ∀d；sample null space 不得宣布为零风险）。
3. 不足且仍有 free：每次剔除因果覆盖行数最少的 free（tie 按索引升序——确定性），资金留 cash（D-056 出路 4）。
4. 剔除后仍不足（locked 引起或 free 剔尽）：`error("innovation coverage failure: ...")`。驱动器捕获该错误 → 当日 `held_maintained`（维持持仓、记录诊断、不拼残差）。
5. free 剔尽且无 locked：返回空 $R_t$——调用方全留 cash。

判定次序（T4，裁决 C3）：driver 调用 resolve **两次**——

- 第一次在 posterior 之前（拿不到残差，只做基础判据 `|J| ≥ 2`）；此时 `held_maintained` 不需要任何模型计算（不重排 ⇒ 无后验消费）。
- 第二次在 prequential 残差就绪后，传 `residual_rows`，做满秩 gate；剔除 free 直至满秩；locked 导致不可收缩则维持持仓。

触发/恢复链（scenario 8，`test/gate0/backtest_tests.jl`）的最小构造：某 held 资产在行 t−2 起才有观测（locked）→ 第 t 日第一次 resolve 通过（J={t−1,t} 两行）、prequential 仅 s=t−1 可定义 → 二次 resolve 的 J={t} 单行 → T4 触发（`mode == :held_maintained`、`reason == :innovation_coverage_locked`、`w_universe == held`、错误文本含 `innovation coverage failure`）；第 t+1 日 J={t,t+1} 两行满秩 → 正常恢复决策（`mode == :reference`、`R_universe == [1,2]`、`locked_exposure ≈ held_before[2]`、跨日漂移桥 `held_before ≈ w·gross/wealth`）。fixture 用 `mode=:reference, S_reference=32, diag_S=16, posterior_tol=1e-2` 控制成本——fixture 级降级，生产默认（1e-6）不变。

### 8.7 状态与抽样

`innovation_state(residual_rows, row_ids, row_masks, R_t, E_active; t=nothing, d_nodes=nothing, d_cells=nothing, cls_floor=nothing, require_full_rank=false)`：

- 构造断言：`|J_t| ≥ 1`（V 可定义）且 `|J_t| ≥ 2`（ℓ 有信息），否则 `"innovation coverage failure"`；J 行残差必须 finite。
- `require_full_rank=true`（driver 生产设置）时 `residual_rank == N_R`，否则 fail loudly。
- 产出 `InnovationState`：`R_t, N_R, E_R, t, eps_R, row_ids, row_masks, J_t, L_t, d_nodes, d_cells, d_weights, V_t, support_cert, z_pool`。

`draw_innovation(st, rng)`——**流位置固定：先 d，后行**：

1. `u1 = rand(rng)` → d（cumsum 搜索）；
2. `u2 = rand(rng)` → 行 `idx = min(floor(u2·|L|)+1, |L|)`；
3. $\varepsilon = V_t(d)^{1/2}\,z_{s'}(d)$（对称 PSD 主根族）。
返回 `(; eps, d, z_row, z_row_id)`。禁止 $v_{bootstrap}$ 类标量锚——$V_t(d)$ 自携绝对尺度（D-059）。

---

## 9. 组装层：`predictive_law`（`predictive.jl`，Step 13）

`predictive_law(post, st, x_t; s1, E_active, rng, S=256) -> PredictiveLawResult`。

八步链条（裁决 A4）：

1. $\mu_{asset} = E_{active}\cdot E[\mu\mid H]$（含 DC；`predict_mu`）。
2. $\mu_R = E_{R_t}^{\top}\mu_{asset}[R_t]$。
3. 每 scenario（**流顺序固定，同 seed 逐位可重放**）：
   - 流 1：`draw_mu`（matrix-t）；
   - 流 2：`d ~ q(d|H)`；
   - 流 3：`z` 行 ~ Uniform(L)；
   - $\varepsilon_R = V_t(d)^{1/2} z$。
4. $y_R = \mu_R^{(s)} + \varepsilon_R^{(s)}$。
5. $u_R = E_{R_t} y_R$。
6. $r_R = u_R \odot s_1[R_t]$。
7. `gross = exp.(r_R)`。

方差分解（裁决 A5，asset 空间 R 子集）：

$$
\boxed{\operatorname{Var}(u_R) = E_R\operatorname{Var}(\mu_R)E_R^\top + E_R\,\mathbb E[V_t]\,E_R^\top}
$$

- `var_epistemic` 来自 `predict_mu` 的 cov；`var_aleatoric` = $\sum_g q_g V_t(d_g)$。
- ε 的完整二阶矩 = $\sum_g q_g V_g^{1/2}M_z(d_g)V_g^{1/2}$（$M_z$ 为经验二阶矩，不假设 = I）；报告口径是 $\mathbb E[V_t]$，差异如实标注、不掩盖。
- ν≤2 时 within 不存在：`not_computed = [:within]`、NaN 携带——**不显示 0**（D-030）。

`PredictiveLawResult` 字段：`R_t, gross, log_gross, mu_R, var_epistemic, var_aleatoric, var_within, var_between, within_computed, not_computed`。

D-060 红线（构造性）：本类型不含任何 $\Sigma_R$/未来残差冲击通道；scenario 不确定性只有 μ draw（epistemic）+ innovation draw（aleatoric）。

契约校验（fail loudly）：两侧 asset 空间必须同为 active 域（`N_a == post.N`）；`R_t ⊆ 1:N`；`s1` 全正有限；propriety 证书绿；`x_t` 长度 P 且有限；S ≥ 1。覆盖不足错误在 `innovation_state` 构造层已拦，本层不吞。

---

## 10. Kelly 层（`kelly.jl`，Step 2/15）

### 10.1 可行集与目标（D-068）

```
max_{w_aug ≥ 0, Σ w_aug = budget}  (1/S) Σ_s log( X_aug·w_aug + base )
```

`X_aug = [X  ones(S)]`（显式 cash 列，gross ≡ 1）。locked 语义（D-017/D-068）：调用方持 locked 时传 `budget = 1 − Σ w_locked` 与 `base = locked_wealth_gate0(X, locked)`；locked 列不进优化列。

`locked_wealth_gate0(X, locked)`：`base_s = Σ_{locked[j]>0} locked[j]·X[s,j]`；held 列缺预测 gross → `error("held asset j has no predictive return law")`——绝不静默当 0/当 cash。

### 10.2 证书（唯一验收口径）

`cash_kelly_certificate(X, w, w_cash; base=nothing, budget=1.0)` 在**原始 log 目标**上算三项：

- `feasibility = max(|Σw_aug − budget|, max(0, −min(w_aug)))`；
- `kkt_residual = max_i |w_i·(ν − g_i)|`，`g = [Xᵀ(1/wealth); Σ1/wealth]/S`，`ν = max(g)`（互补性：正权列 g=ν，零权列 g≤ν）；
- `objective_gap = max(0, budget·ν − gᵀw)`（concavity 上界）；
- `objective = mean(log(wealth))`；`wealth = X*w + w_cash + base`。

`cash_kelly_certified(cert; tol=1e-8)`：三项全 ≤ tol。非有限权重或非正 wealth 短路为全 Inf 证书（拒收）。

**契约**：求解器内部 status 不参与验收。没有第二套验收标准。

### 10.3 求解链（`cash_kelly`）

`cash_kelly(X; base=nothing, budget=1.0, tol=1e-8, tie_eps=1e-8, scale_retry_diag=nothing) -> (w_risky, w_cash, certificate)`。

结构（B2B-SPLIT，批量调度用）：

```
cash_kelly = _cash_kelly_prelude + _cash_kelly_finish
_cash_kelly_prelude = _cash_kelly_prelude_prep + _cash_kelly_prelude_solve
_cash_kelly_finish  = _cash_kelly_finish_main    + _cash_kelly_finish_tiebreak
```

内部求解：`:auto` 双 backend——先 IPM（LOGSUM-SIMPLEX-IPM-1，`_cash_kelly_ipm`），认证（证书通过且 μ ≤ μ_floor）即返回；未认证/异常回退 Clarabel 原路径。IPM 数值常数（D-066 候选，未定稿）：`μ0=1.0, σ=0.2, μ_floor_factor=1e-3, max_outer=60, max_newton=4, τ=0.99`。返回点绑定 `μ ≤ μ_floor = max(1e-3·tol, 1e-12)`（保护 TIE-FAST-1 的 1e-8 差值契约）。Clarabel 路径（reference）：Convex 建模 `maximize(sum(log)/S)`，MOI `tol_gap_abs/tol_gap_rel/tol_feas = min(tol/10, 1e-10)`、`max_iter=1000`；状态白名单 `{OPTIMAL, ALMOST_OPTIMAL}`；白名单外若可提取有限出口解（M2）则规范化后进入候选集，否则报错。

**行缩放（KELLY-SCALE-RETRY-1）**：每 scenario 行 `c_s = max(|b_s|, max_j|X_sj|)`；`X̃=X./c`、`b̃=b./c`、cash 列 `1.0./c`（不得保持 ones）。恒等式：`wealth̃=wealth/c`，目标差常数；argmax、证书在原始空间评估。触发条件：首选未缩放且证书 fail → 缩放重解一次；已缩放 fail 不重试。激活阈值 `span > 1/eps ≈ 4.5e15` 只是首选路径的装配规则；重试由**证书质量**驱动，阈值不动。

**设计 A（取优）**：候选 = raw（必进）+ scaled（仅当 raw 失败）；每候选 polish（原始空间，至多两次）；比较键：① certified 优先；② `q = max(feasibility, kkt, gap)` 低者；③ `q` 差 < 1e-15 时路径偏好 raw。`objective` 不进键。只有 certified 解可被采纳——取优从不放松验收。

**Newton polish**：`max_iters=200`（`stationarity=1e-13` 早退）；支撑判定阈值 `1e-6`；支撑列数 > S 时按最小权重剔除至 ≤ S；奇异 → 返回 `nothing`、保留原解。

### 10.4 P0-9 canonical tie-break

数值等价最优集 $W_\epsilon = \{w_{aug}\ge0,\ \sum w_{aug}=budget,\ U(w)\ge U^*-\epsilon_{tie}\}$：

1. Stage 2：`max w_c`；
2. Stage 3：在相同 `w_c` 下 `min ‖w_risky‖₂²`；
3. 结果在原空间重证书；不过则保留主解（tie-break 是规范化精修，不是新决策权）。

- `tie_eps` 默认 `1e-8`（D-066 候选值）。
- `m=2`（n==1）专用一维二分（80 次，W_ε 下端点）。
- `TIE-FAST-1`：主解 risky 分量全部 ≤ 1e-12 → 跳过 tie-break（输出与规范化逐位一致）。
- 一般 m>2：保留两阶段 Clarabel（扩展属未来工作）。

### 10.5 复现验证点（`test/gate0/kelly_cash_tests.jl`）

Case A 全 cash；Case B 全押；Case C 内部解解析一阶条件 $E[(X-1)/wealth]=0$；Case D locked 在 base；D-090 对称世界；D-091 无信号世界；显式 cash 列 vs 预算不等式等价；证书负控制；行缩放回归；设计 A 双路径；M2 status 出口；B2B-SPLIT；IPM 解析 fixture；主链 fail-loud 锚（全候选不过 → error、不返回权重）。

---

## 11. 积分层：adaptive RQMC（`quadrature.jl`，Step 14）

### 11.1 rule（D-063/D-065）

```julia
SobolOwenRule(dim, seed)   # dim ∈ {1,2}（Joe-Kuo 前 2 维方向数）；超出 fail loudly
rule_points(rule, M)       # M×dim，按全局索引 0..M−1 生成
```

- 方向数：Joe-Kuo 2008 前 2 维（d=1 = van der Corput；d=2 = s=1,a=0 递推）。增量 XOR（Antonov–Saleev）。
- Owen scramble = matrusca 线性（GF(2) affine）形式；矩阵与常数由 `MersenneTwister(seed)` 派生。
- **嵌套**：任意 M′ > M 的前 M 点逐位相同（硬保证）。
- `audit_seed(rule_seed) = rule_seed ⊻ 0x9E3779B97F4A7C15`；`_default_mu_seed(rule_seed) = rule_seed + 0x2545F4914F6CDD1D`。

### 11.2 scenario source

`predictive_rqmc_source(post, st, x_t; s1, E_active, mu_qmc=false, mu_chisq_qmc=false, mu_z1_vdc=false)` 与 `predictive_law` 八步链条逐行同构；差异在抽样驱动：

- d 通道：rule 点第 1 维 → 离散 inverse-CDF；
- z 行通道：rule 点第 2 维 → floor 变换；
- μ 通道：独立流（每层同 seed 重建 → 前缀嵌套）。默认 iid MC；`mu_qmc=true` 时前 2 主方向的 z 用 QMC（部分 QMC）、节点选择与卡方可选 QMC（`mu_chisq_qmc`）。

### 11.3 自适应循环与四证书

`adaptive_scenario_kelly(src; rule_seed=0x1D9B7A5C_E11A0F42, mu_seed=nothing, locked=zeros, min_scenarios=64, max_scenarios=512, weight_tol=1e-3, utility_tol=1e-5, utility_tol_audit=1e-5, kelly_tol=1e-8) -> AdaptiveKellyResult`。

```
M = min_scenarios
loop:
    M2 = min(2M, max_scenarios)
    M2 == M → error（预算顶，禁止同层假绿比较）
    在优化 rule（同一 rule 对象、嵌套）生成 M2 点 → cash_kelly 解 w_2M
    A       = ‖vcat(w_2M,wc_2M) − vcat(w_M,wc_M)‖₁        ≤ weight_tol
    B_opt   = I_2M(w_2M) − I_2M(w_M)                      ≤ utility_tol
    B_audit = |I_audit(w_2M) − I_audit(w_M)|              ≤ utility_tol_audit
    C       = cash_kelly_certificate(X_2M, w_2M) 全绿      （kelly_tol）
    全过 → 返回; 否则 M ← M2
```

语义分层（NUMERICAL_INTEGRATION_SPEC §3.1）：

- A、B_opt、B_audit 是**积分收敛证据**；C 是**求解误差证据**；四者不得互替。
- `B_opt` 是 fine-rule regret（fine 解不得显著落后 coarse 解）；旧单向 `B ≤ ε_U` 允许大幅负值通过——已是历史。
- `B_audit` 取绝对值；audit rule 与优化 rule 独立随机化（防 sample optimization optimism）。
- 预算耗尽：`error("Numerical integration did not converge by $max_scenarios scenarios ...")`——**不返回最后一层权重**。

`AdaptiveKellyResult`：`w_risky, w_cash, M, rule_seed, audit_seed, certificate_A, certificate_B, certificate_B_audit, certificate_C, converged, iterations, history`。

`locked` 语义：每层 `base = locked_wealth_gate0(X, locked)`；free 列进 `cash_kelly`；财富统一方案 2：`wealth = X_free·w_free + w_cash + base_locked`。禁止在 `X_full·w_full` 上再加 base（locked double-count，P0-5 已修复）。

### 11.4 复现验证点

- 嵌套：`rule_points(rule, M)` 与 `rule_points(rule, 2M)` 前 M 行逐位相同（atol=0）。
- 同 seed 重放；不同 seed 收敛到相同权重。
- known Gaussian 2-asset 解析 Kelly + KKT 传导。
- 容差减半 → M 单调不减、权重稳定。
- A 过 B 红必须继续翻倍（证书分层）。
- 预算耗尽 fail loudly；无固定 S 捷径（mock source 调用计数）。

---

## 12. 驱动器与回测（`driver.jl` / `backtest.jl`，Step 15）

### 12.1 `single_day_decision`

```julia
single_day_decision(mf, el, held;
    t::Int,
    seed::UInt64 = 0x00D05EED00000001,   # _DRIVER_DEFAULT_SEED
    mode::Symbol = :adaptive,            # :adaptive | :reference
    S_reference::Int = 256,
    diag_S::Int = 64,
    min_scenarios::Int = 64,
    max_scenarios::Int = 512,
    weight_tol::Float64 = 1e-4,
    utility_tol::Float64 = 1e-6,
    kelly_tol::Float64 = 1e-6,
    posterior_tol::Float64 = 1e-6,
    posterior_max_cells::Int = 2048,
    u_span::Float64 = 5.0,
    forecast_cache = nothing,
    mu_qmc::Bool = false,
    mu_chisq_qmc::Bool = false)
```

步骤（`driver.jl`）：

0. **因果截断**：物理构造 `mf_t = MarketFacts(...[1:t]...)`——截断输入不含 >t 行是构造事实，不是约定。
1. active 域：`act = findall(el_t.model_admitted[t,:])`；空 → `:all_cash`（reason `:no_active_assets`）。
2–3. `_response_prefix(mf_t, el_t, t)`：每资产 ruler、row-level field、mode design（`dc=true`）。
4. 训练行：`t-1 >= WARMUP`；`rows = [s in WARMUP:(t-1) if any(obs[s+1,:])]`；`length(rows) >= 2`。行身份 `row_ids = s+1`；行 mask `O_s = obs_ret[u,:]`。
5. free/locked → active 域映射。
6. 第一次 `resolve_risk_domain`（基础判据）；捕获 `"innovation coverage failure"` → `:held_maintained`（维持持仓 + 诊断）；其余错误重抛。
7. **完整后验**：`fit_full_posterior(X_tr, Y_tr; tol=posterior_tol, max_cells=posterior_max_cells, u_span=u_span)`；把今日 full-posterior 均值写入 forecast cache（供明日 surprise）。
8. **strict prequential 残差**（`_market_prequential_residuals`）；仅保留存在 finite 分量的行；`row_masks_ok = row_masks .& isfinite.(resid)`。
8b. 第二次 `resolve_risk_domain(...; residual_rows=resid)`（P0-3 满秩 gate）。
9. `innovation_state(resid, row_ids_ok, row_masks_ok, R_act, E_active; t=t, require_full_rank=true)`。
10. `x_now = mp.X_full[t,:]`（P = 1+14N）。
11. locked 权重（R 域列序）；`budget = 1 − Σ locked`。
12. 决策分支：
    - `:adaptive`：`adaptive_scenario_kelly(post, st, x_now; s1, E_active, rule_seed=seed, locked=locked_w, ...)`。locked 直接处理，无固定 S 回落（P0-4）。
    - `:reference`：`predictive_law(...; S=S_reference, rng=MersenneTwister(seed))` + `cash_kelly`。A/B/B_audit = NaN（固定 S 无积分证书）。
13. 写回 universe 坐标。
14. 方差分解 + D-076 concentration（含 `utility_margin_1pct`，在独立诊断样本 `seed ⊻ 0xD1A6571C5EED0000`、S=diag_S 上求值——诊断量，非证书）。

模式字段：`:adaptive` / `:reference` / `:all_cash` / `:held_maintained`。财富守恒：`Σw_universe + w_cash = 1`；`budget + Σlocked + w_cash = 1`。

### 12.2 `run_gate0_backtest`

```julia
run_gate0_backtest(mf, el;
    t_start::Int, t_end::Int,
    held0 = zeros(N),
    mode::Symbol = :adaptive,
    seed::UInt64 = _DRIVER_DEFAULT_SEED,
    mu_qmc=false, mu_chisq_qmc=false,
    reuse_forecasts=true, forecast_workers=1,
    checkpoint=nothing, resume=false,
    max_new_days=nothing, max_new_forecasts=nothing,
    progress=false, kwargs...)
```

- 决策日 `t ∈ t_start:(t_end-1)`；`t_end` 是最后 marking 日。要求 `t_start-1 >= WARMUP`。
- 每日：`single_day_decision` → `_marking_gross` → `wealth = dot(w, gross) + w_cash` → 权益乘子 → 持仓漂移 `held′ = w.*gross ./ wealth`（cash 隐含 `1−Σheld′`）。
- `_marking_gross`：持仓列缺合法 marking → fail loudly；未持仓列缺价格 → 中性 1.0（2026-10-11 修复：旧 NaN 经 `0·NaN` 污染财富）。
- T4 跨日语义：`held_maintained` 日后以维持并漂移后的持仓继续决策，不重置、不清仓。
- 三 benchmark（D-072/H8）：Cash（恒 1）、EW-daily（每日按 `benchmark_universe(el,t)` 等权再平衡）、EW-buyhold（首日等权后不再平衡）。`ew_rebalance_gap = equity_ew_daily − equity_ew_buyhold` 只报告，不做 volatility pump 因果解释（D-073）。
- checkpoint（version=1）：payload 含 `signature`（对 `(VERSION, mf, el, options)` + `src/gate0/*.jl` 内容 + `Manifest.toml` 的 SHA-256）。resume 时校验签名、日计数、权益计数；不一致抛错。切片：`max_new_days` / `max_new_forecasts` 触发 `BacktestPaused <: Exception`——刻意暂停，不是完成；`bin/backtest.jl` 捕获后以退出码 **75** 退出。
- 历史预测预填充 `_prime_historical_forecasts!`：只在 `first_t` 之前的 origin 上评估（执行顺序重排，不改变数学）；workers 并行只在完成后按日期序更新缓存与 checkpoint。

`Gate0BacktestResult`：`t_start, t_end, days, equity, equity_cash, equity_ew_daily, equity_ew_buyhold, ew_rebalance_gap`。`Gate0DayRecord`：`t, decision, held_before, gross_realized, wealth_factor, equity_after`。

### 12.3 `equal_weight_benchmark`

`equal_weight_benchmark(el, mf, t) -> (; t, mask, weights, n, symbols)`；候选集 = `benchmark_universe(el,t)`（外生，D-020）；空候选返回零权重。

---

## 13. 数值契约汇总

### 13.1 容差层级（唯一口径）

源码内（`quadrature.jl` 自适应循环）的唯一收敛判据：

$$
A \le \text{weight\_tol}\ \wedge\ B_{opt} \le \text{utility\_tol}\ \wedge\ B_{audit} \le \text{utility\_tol\_audit}\ \wedge\ C\ \text{全绿}
$$

批量调度链（`dev/batch_t_solve3.jl` 等工具，非库内函数）在四证书之外另加归一性检查 `norm_err = |Σw_risky + w_cash − 1| ≤ 1e-8`（即第 15/16 任段「过门判据式」中的 `norm≈0` 项）。复现批链时应一并执行该检查。

| 层 | 参数 | 值 | 出处 |
|---|---|---|---|
| 外部过门参考线 | A（weight_tol） | **1e-4** | 第 15/16 任 |
| batch 工具默认 | A | 1e-3 | quadrature 函数默认 |
| B 工具默认 | utility_tol / utility_tol_audit | 1e-5 | 函数默认 |
| driver 生产默认 | utility_tol | 1e-6 | `single_day_decision` 默认 |
| driver 生产默认 | utility_tol_audit | 1e-5（未显式传，走函数默认） | — |
| C 线 | kelly_tol | batch 链 1e-8；driver 默认 1e-6 | `driver.jl` |
| posterior | posterior_tol / max_cells | 1e-6 / 2048（2D 实际 32768 cells） | `fit_full_posterior` 默认 |
| d quasi-posterior | tol_d / tol_d2 / tol_V / max_levels | 1e-4 / 1e-4 / 1e-6 / 12 | `adaptive_d_quadrature` 默认 |
| cash_kelly | tol / tie_eps | 1e-8 / 1e-8 | `cash_kelly` 默认 |

**D-066 定稿流程**（所有数值容差必须走）：synthetic 解析对照 → realistic single-day fixture → 容差减半 → 权重/目标稳定 → 记录为 numerical configuration。全程与 Sharpe/PnL 完全无关。已知开放项：A 门限 1e-4 与深段噪声底同量级（~1e-4–7e-4）——是否加深/改口径/降方差属 owner 裁决；`μ_floor_factor` 1e-3 待定稿。

### 13.2 关键数值常数

| 常数 | 值 | 位置 |
|---|---|---|
| `_DRIVER_DEFAULT_SEED` | `0x00D05EED00000001` | driver |
| `_DIAG_SEED_XOR` | `0xD1A6571C5EED0000` | driver（诊断流） |
| `adaptive_scenario_kelly` 默认 `rule_seed` | `0x1D9B7A5C_E11A0F42` | quadrature |
| audit seed 派生 | `rule_seed ⊻ 0x9E3779B97F4A7C15` | quadrature |
| μ 流 seed 派生 | `rule_seed + 0x2545F4914F6CDD1D` | quadrature |
| 行缩放阈值 | span > `1/eps ≈ 4.5e15` | kelly |
| `_KELLY_CANDIDATE_TIE` | 1e-15 | kelly（设计 A 比较） |
| IPM 常数 | μ0=1.0, σ=0.2, μ_floor_factor=1e-3, max_outer=60, max_newton=4, τ=0.99 | kelly |
| polish | max_iters=200, stationarity=1e-13, 支撑阈值 1e-6 | kelly |
| PSD floor | `_PSD_FLOOR_REL=1e-12`, `_PSD_NEG_TOL_REL=1e-10` | posterior |
| `cls_floor`（innovation） | `N·sqrt(eps)·λmax` | innovation |
| 2D 求积 | 主 2 点 G-L + 3 点对照；init_grid=4；2D `16*max_cells` | posterior |

### 13.3 fail-loud 文本锚（验收/调试用）

| 场景 | 文本片段 |
|---|---|
| A2 propriety | `"posterior improper"` |
| 求积不收敛 | `"Numerical integration did not converge"` |
| innovation 覆盖不足 | `"innovation coverage failure"` |
| prequential 全不可定义 | `"prequential residual"` |
| locked 缺预测律 | `"held asset j has no predictive return law"` |
| 持仓缺 marking | `"_marking_gross: 持仓资产 ... 缺合法 marking 价格"` |
| checkpoint 不一致 | `"backtest checkpoint mismatch"` |
| Kelly 主链失败（Clarabel 状态非白名单且无出口解） | `"Clarabel cash log-Kelly failed"` |
| Kelly 退化配置 | `"cash Kelly solver returned a degenerate allocation"` |
| Kelly 出口解未过证书 | `"exit candidate not certified"` |
| Kelly 双路径候选全败 | `"双路径候选均未过证书；不返回未认证权重"` |

---

## 14. 复现操作手册

### 14.1 最小冒烟（≤60s）

前置：`data/` 就绪；Julia 环境可用。

```sh
bash bin/scoped_run.sh 50 /tmp/g0_smoke.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/gate0/market_tests.jl
```

预期：testset 全 Pass；log 尾行 `rc=0`。若 `rc=124`：先做预热再重跑；不要放宽任何容差。

### 14.2 单模块测试

每个测试文件自含（自建 `module XxxEnv` + include 链），可独立运行：

```sh
bash bin/scoped_run.sh 50 /tmp/g0_modes.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/gate0/modes_tests.jl
```

注意：`quadrature_tests.jl` 必须独立运行（包裹入口会超时）。

### 14.3 全量测试（分组）

统一入口：`test/gate0/runtests.jl`。12 模块累计超单命令预算；实测约两模块一组，单组超窗再拆（`quadrature_tests.jl` 必须独立运行——包裹入口会超时）。

```sh
# 模块分组示例（约两模块一组；下表为可用的分组起点）
for g in "1,2" "3,4" "5,6" "7,8" "11,12"; do
  GATE0_MODULES="$g" bash bin/scoped_run.sh 50 "/tmp/g0_m${g//,/}.log" --rss-guard=2048 \
    julia --startup-file=no --project=. test/gate0/runtests.jl
done
# kelly_cash（9）与 quadrature（10）独立运行
GATE0_MODULES="9"  bash bin/scoped_run.sh 50 /tmp/g0_m9.log  --rss-guard=2048 \
  julia --startup-file=no --project=. test/gate0/runtests.jl
GATE0_MODULES="10" bash bin/scoped_run.sh 50 /tmp/g0_m10.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/gate0/runtests.jl
```

注：`GATE0_MODULES` 是逗号分隔序号集合（如 `"1,2"`）；`"1..7"` 不是合法值。历史受控批次另用过 `"1,2,3,4"` / `"5,6,7,8"` / `"9"` 等切分——都只是调度切分，不改变数学。

backtest 场景（1–9）另经 `BACKTEST_SCENARIO` 选择，默认 1：

```sh
GATE0_MODULES="12" BACKTEST_SCENARIO="1,2,3,4" bash bin/scoped_run.sh 50 /tmp/g0_bt1.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/gate0/runtests.jl
# 其余批："5,6,7,8" / "9"
```

当前登记口径（2026-10-11 复核）：62 testset / 1204 断言全绿。分组仅为调度，不改变数学。

模块 ↔ 文件映射：

| 模块 | 测试文件 | 覆盖 |
|---|---|---|
| 1 | market_tests.jl（+market_io_tests.jl） | Step 1 四 mask |
| 2 | modes_tests.jl | Step 3 mode 坐标 |
| 3 | response_tests.jl | Step 4/5 固定 ridge full block |
| 4 | posterior_tests.jl（+covariance_action_tests.jl） | Step 8 完整后验 |
| 5 | oof_tests.jl | Step 9 legacy OOF |
| 6 | prequential_tests.jl | P0-1 strict prequential |
| 7 | innovation_tests.jl | Step 10–12 vector innovation |
| 8 | predictive_tests.jl | Step 13 组装 |
| 9 | kelly_cash_tests.jl | Step 2 cash Kelly |
| 10 | quadrature_tests.jl | Step 14 adaptive RQMC |
| 11 | driver_tests.jl（+causal_forecast_tests.jl） | Step 15 单日 |
| 12 | backtest_tests.jl（+checkpoint_tests.jl） | Step 15 多日 |

backtest 场景清单：1 = 5-day end-to-end；2 = holdings rollover conservation；3 = causality truncation；4 = locked multi-day；5 = T4 multi-day；6 = benchmark；7 = fail loudly；8 = T4 recovery (prequential)；9 = marking NaN propagation。

各模块覆盖要点（复现时按序验证）：market = 四 mask 语义/只读防御/252 边界/置换协变/数据桥；modes = H7 几何复制语义/reconstruct/置换/gauge/ragged/dummy/维度/线性交换性/ruler 短历史回退；response = D-092 cross-mode 恢复/design-target 代数/零响应/D-093 DC/D-094 trace 反例；posterior = S(α) 双构造/evidence 对照/matrix-t 矩/μ 边缘/点质量极限/A2 gate/求积收敛与重放/DC/抽样/（P0-8 表达式断言）/G-L 归一化/R4 鲁棒 logdet；covariance_action = Gauss 节点认证/方向细分/inverse precision/谱 evidence/快慢等价；oof = fold 差分恒等/隔离/残差行/跨块 OOF/空观察行剔除；prequential = 形态与因果/与 legacy OOF 语义区分/fail loud/成本加速不变量；innovation = 13 个 9.x 小节（support/refinement/拼接反例/因果与重放/极限/support 不变性/shape/quasi-likelihood/覆盖/D-095 向量波动世界/接口 fail-loud/P0-2/P0-3）；predictive = 正负契约/D-060 无 Σ_R 通道/ν≤2 NOT COMPUTED/错误传导；kelly_cash = 25 testset（Case A–D、D-090/D-091、等价性、负控制、输入校验、P0-9、行缩放、K-S-R-1、设计 A、polish、M2、TIE-FAST、B2B-SPLIT、IPM、后端与回退、m=2 tie 专用化、行缩放装配、主链 fail-loud 锚、IPM 失败语义）；quadrature = 嵌套逐位/重放与 audit 独立/Gaussian 解析 Kelly/容差减半/证书分层/预算耗尽/无固定 S/seed 间收敛/μ 通道 QMC/节点 QMC/NEST-SYS/VDC-CHISQ 开关；driver = 端到端双路径/locked 计财/T4 语义/all-cash/D-076 字段/benchmark 外生/因果/fail loud；backtest = 场景 1–9；checkpoint = 并行调度等价/预填充切片/resume 一致；causal_forecast = 历史预处理不见未来价格或上市。

### 14.4 回测运行

```sh
# 默认：末 20 天窗口、adaptive 生产路径
bash bin/scoped_run.sh 50 /tmp/g0_bt.log --rss-guard=2048 \
  julia --startup-file=no --project=. bin/backtest.jl

# 5 天最小窗口（t_start ≥ 257；t_end 必须 ≤ 数据行数 T，示例按数据调整）
GATE0_T_START=257 GATE0_T_END=262 bash bin/scoped_run.sh 50 /tmp/g0_bt5.log --rss-guard=2048 \
  julia --startup-file=no --project=. bin/backtest.jl

# reference 固定 S 对照（D-062 合法用途，非生产默认）
GATE0_MODE=reference GATE0_S_REFERENCE=64 bash bin/scoped_run.sh 50 /tmp/g0_ref.log --rss-guard=2048 \
  julia --startup-file=no --project=. bin/backtest.jl
```

环境变量（参数面 = `GATE0_*`；旧线参数已废止）：

| 变量 | 默认 | 语义 |
|---|---|---|
| `GATE0_DATA_DIR` | `<repo>/data` | 数据目录 |
| `GATE0_MODE` | `adaptive` | `adaptive`（生产，四证书收敛）或 `reference`（固定 S 对照） |
| `GATE0_S_REFERENCE` | 256 | reference 模式固定 S |
| `GATE0_T_START` / `GATE0_T_END` / `GATE0_START_DATE` / `GATE0_END_DATE` | 末 20 天 | 决策区间；日期与行索引二选一，不得同时设置 |
| `GATE0_SEED` | `0x00D05EED00000001` | 决策随机流 seed（十六进制） |
| `GATE0_TRADE_252` | false | 252 有效交易日政策 |
| `GATE0_POSTERIOR_TOL` / `GATE0_POSTERIOR_MAX_CELLS` | 1e-6 / 2048 | posterior 求积口径（P0-7：超时的正确行为是 fail/缩小 fixture，不是放宽） |
| `GATE0_MAX_SCENARIOS` | 512 | adaptive 积分预算（过门配置 131072） |
| `GATE0_WEIGHT_TOL` / `GATE0_UTILITY_TOL` / `GATE0_KELLY_TOL` | 1e-4 / 1e-6 / 1e-6 | 证书容差 |
| `GATE0_MU_QMC` / `GATE0_CHISQ_QMC` | false | μ 通道 QMC 开关（过门配置 true） |
| `GATE0_CHECKPOINT` / `GATE0_RESUME` / `GATE0_MAX_NEW_DAYS` / `GATE0_MAX_NEW_FORECASTS` / `GATE0_REUSE_FORECASTS` / `GATE0_FORECAST_WORKERS` | — / false / — / — / true / 1 | 检查点与切片 |

预期输出：控制台打印三 benchmark 净值、`ew_rebalance_gap`、逐日记录（`t= / mode= / M= / cash= / locked= / wealth= / equity=`）、末决策日集中度诊断（`top_asset / top_weight / risky_weight / cash_weight / integration_M / converged / utility_margin_1pct / epistemic_variance / innovation_variance`）。未计算项显示 `NOT COMPUTED`（不允许显示 0）。

`reference` 模式 `A/B/B_audit` 为 NaN 是**语义正确**（固定 S 不冒充积分收敛）。

### 14.5 复现完成检查清单

- [ ] 环境：`Pkg.instantiate()` 成功；`using KTrader` 可加载。
- [ ] 数据：`load_market_facts` 通过；`observed`/`signal_prices` 语义抽检。
- [ ] 冒烟：`market_tests.jl` rc=0。
- [ ] 模块测试：12 模块分组全部 rc=0；backtest 9 场景（分组）rc=0。
- [ ] 单日：`single_day_decision`（`:reference` 与 `:adaptive`）在短区间运行，财富守恒 `Σw + w_cash = 1`。
- [ ] 多日：5 天与 20 天窗口跑通；`ew_rebalance_gap` 与三 benchmark 曲线有输出。
- [ ] 证书：任一 adaptive 决策日 `A ≤ weight_tol`、`B_opt ≤ utility_tol`、`B_audit ≤ utility_tol_audit`、C 全绿；否则按 fail-loud 路径处理（先归因，不放宽容差）。
- [ ] 确定性：同 seed 同输入重跑，逐位一致。

---

## 15. 已知边界与开放项

如实登记（不粉饰、不隐藏）：

1. **入口规模边界**：`bin/backtest.jl` 在大面板上的单命令自适应超窗。60-day 是批量负载口径（分批调度），不是单命令目标。过门配置 60 日结果：37 标准过门 + 23 超限；23 个超限日全部证得加深可达（21 天 @262144、2 天 @354/385 @524288；1048576 未用于任何一天）。
2. **A 判据口径待裁**：超限日处置（1e-4 vs 1e-3 vs 逐层自适应）与深段振荡底性质属 SPEC 层裁决；D-066 复核素材已备。
3. **μ_floor factor**：1e-3（生产绑定）尚未走完 D-066 定稿；有回调保护。
4. **m>2 tie-break**：当前仅 m=2 有专用一维二分；一般 m 保留两阶段 Clarabel。
5. **μ 通道 RQMC 化**：`mu_qmc` 是部分 QMC（前 2 主方向 + 节点/卡方通道），默认关闭；全维 QMC 属 Gate-0 后扩展（D-063 完整实现被 defer）。
6. **z1-vdc 开关**：实现与测试在位，但 driver/backtest 生产入口未透传——判定不可用，仅设计记录。
7. **T4 跨语义缺口**（裁决 9）：D-013 admission（prefix 级）与 fold propriety（fold 级）的历史张力；production（prequential）语义下已闭合（scenario 8 15/15），fold 语义属 legacy。
8. **501-day 分批执行**：机制已建立（checkpoint/切片）、未执行；真实数据桥接优先于合成 501。
9. **首日 full-fit 成本**：自适应 cells 规模仍是成本项（单点已达标；reference 不追求速度，D-083）。
10. **BLAS 线程**：默认多线程对小 GEMM 约 3× 负优化；当前入口已设单线程（U2）。若自行改线程数，需自证不改变数值对象。
11. **quadrature 负载敏感性**：同字节重跑时间漂移存在；以裁定重跑为准。
12. **span≈1e17+ 与人工双方向 fixture**：已知数值边界。
13. **`solver_diag` / `cert` 字段**：设计文档 §4.3 提及但生产未实现（无消费者）；未来受控工作。

---

## 16. 命名与措辞纪律（D-096/D-097）

禁止（直到条件满足）：

- "Exact Bayesian Kelly"；"Full Bayesian posterior"（对旧线 plug-in 结构）；"Full Kelly" 用于固定 S；"Trace neutrality is an identification theorem"；"Low SNR causes Kelly concentration"；"Volatility Pump explains EW outperformance"；"Completely detached from beta"；fully Bayesian generative posterior predictive（对当前 modular predictive）。

推荐：

- 旧实现：`KTrader 2.0 RC legacy implementation`。
- 新线：`Path Kelly modular predictive reference`。
- 数值收敛后：`converged log-Kelly decision`。

---

## 附录 A：单日链条的逐步对照（审计用）

```text
H_t（因果截断 t）
  → signal_prices（observed 处 adj，否则 NaN）
  → log / 日收益（相邻 finite 对）
  → active 域（A^model prefix 语义）
  → ruler s1 → E_active（domain_mode_basis）
  → build_mode_problem（[m; Qᵀe] 与 [1|B_m|B_⊥] 含 DC）
  → rows / row_ids（u_s 严格递增）/ row_masks（O_s）
  → free / locked（universe→active 映射）
  → resolve_risk_domain 第一次（基础判据 |J|≥2；T4 判定次序）
  → fit_full_posterior（Π(b₀,G,α₀,α_p,Σ_R|H)；2D 张量 G-L 自适应求积）
  → _market_prequential_residuals（strict prequential）
  → resolve_risk_domain 第二次（P0-3 满秩 gate，传 residual_rows）
  → innovation_state（ε→q(d)→V_t→z pool，R 域 mode 坐标；require_full_rank）
  → x_now（决策行 feature）
  → adaptive_scenario_kelly 四证书 或 predictive_law + cash_kelly（reference）
  → w_t*（cash + locked base_s）
  → D-076 concentration cause report
  → marking 结转 → 下一日
```

## 附录 B：关键文件路径速查

| 内容 | 路径 |
|---|---|
| 模块骨架与 include 顺序 | `src/gate0/KTraderGate0.jl` |
| 四 mask | `src/gate0/market.jl` |
| CSV 适配器 | `src/gate0/market_io.jl` |
| 几何原语 | `src/gate0/geometry.jl` |
| mode 坐标 | `src/gate0/modes.jl` |
| 固定 ridge response | `src/gate0/response.jl` |
| 完整后验 | `src/gate0/posterior.jl` |
| 残差（legacy OOF + prequential） | `src/gate0/oof.jl` |
| vector innovation | `src/gate0/innovation.jl` |
| PredictiveLaw | `src/gate0/predictive.jl` |
| cash Kelly | `src/gate0/kelly.jl` |
| adaptive RQMC | `src/gate0/quadrature.jl` |
| 单日 + 多日 driver | `src/gate0/driver.jl` / `src/gate0/backtest.jl` |
| 当前入口 | `bin/backtest.jl` |
| 护栏 | `bin/scoped_run.sh` |
| 测试入口 | `test/gate0/runtests.jl` |
| 发布元数据 | `RELEASE.toml` |
| Exit 清单 | `docs/GATE0_EXIT_CHECKLIST.md` |

## 附录 C：证据与设计文档索引

- 裁决总纲：`AGENTS.md`（D-001~D-097 与历任登记段）。
- 终审裁决：`docs/GATE0_MANAGER_ADJUDICATIONS.md`。
- 响应后验推导：`docs/GATE0_RESPONSE_POSTERIOR.md`。
- innovation 推导：`docs/GATE0_VECTOR_INNOVATION.md`。
- 数值积分规范：`docs/NUMERICAL_INTEGRATION_SPEC.md`。
- 容差定稿：`docs/NUMERICAL_TOLERANCE_FINALIZATION.md`。
- Kelly 求解器设计：`docs/KELLY_LOGSUM_SOLVER_DESIGN.md`、`docs/KELLY_NUMERICAL_ROW_SCALING.md`、`docs/KELLY_ADAPTIVE_CERTIFICATE_DRIVEN.md`。
- Exit 清单：`docs/GATE0_EXIT_CHECKLIST.md`。
- 多日运行状态：`docs/MULTIDAY_RUN_STATUS_AUDIT.md`、`docs/MULTIDAY_COST_OPTIMIZATION.md`。
- t=327 求积裁决：`docs/ADJUDICATION_T327_POSTERIOR_QUAD.md`。
- T4 语义裁决：`docs/T4_MULTIDAY_SEMANTICS_ADJUDICATION9.md`。
- μ 通道 QMC：`docs/MU_CHANNEL_QMC_DESIGN.md`。
- prequential 加速：`docs/PREQUENTIAL_COST_ACCELERATION.md`；首日成本：`docs/FIRST_DAY_FIT_COST_ACCELERATION.md`；早期行失败：`docs/EARLY_ROW_FIT_FAILURE_DIAGNOSIS.md`；ruler 回退：`docs/RULER_SHORT_HISTORY_FALLBACK.md`。
- 运行证据：`archive/evidence/gate0_final_verify_20261010/`、`archive/evidence/gate0_kelly_rewrite_20261011/`、`archive/evidence/gate0_multiday_run_20261010/`、`archive/evidence/gate0_assertion_recount_20261011/` 等。

---

*本指南为 Gate-0 当前线的复现入口文档；未修改任何源码、测试与既有证据。所有「已核实」陈述来自写作时点的只读核对；运行结果口径以 `archive/evidence/` 与 `AGENTS.md` 历任段为准。*
