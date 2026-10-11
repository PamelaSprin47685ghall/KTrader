# =============================================================================
# KTraderGate0 backtest — Step 15（多日）：sequential 多日回测 driver
# =============================================================================
#
# 文件地位（裁决 H1/H2）：未来顶层 module `KTraderGate0` 的组成文件；当前
# 按裸文件交付（骨架 include 归 DevOps 集成验证轮）。include 顺序：
# …… → kelly → quadrature → driver → 本文件（消费 single_day_decision 的
# 公共接口与 _DRIVER_DEFAULT_SEED；driver.jl 必须先于本文件 include——
# 文件头依赖声明钉死）。本文件不修改 driver.jl。
#
# 职责（施工图 Step 15 多日段 / 旧 SPEC §42 循环语义 / D-089 短窗口）：
# 把 single_day_decision 串成 sequential 多日回测——**5→20 天短窗口受控
# 运行**（Gate-0 关闭后解锁的阶梯级；60/501 天在护栏外，归 owner 分批
# 策略，不在本文件）。
#
# 旧 SPEC §42 基本循环的逐步落点（docstring 钉死，供审计对照）：
#   for decision day t ∈ t_start:(t_end-1):
#     1. 只使用 ≤t 的信息——single_day_decision 内部因果截断（构造性
#        保证：截断 MarketFacts 物理不含 >t 行；本循环不另做任何截断，
#        因果性由 driver 层单点拥有）；
#     2. 构造模型 + 产生场景 + Kelly target——全部在 single_day_decision
#        内（含 T4 held_maintained / all_cash 分支的转译语义）；
#     3. 持仓结转（t→t+1 marking）：gross_j = adj[t+1,j]/adj[t,j]
#        （carried marking 序列——旧线 src/backtest.jl:296 的 marking 语义；
#        理论 signal 与账户 marking 两条路径互不冒充，D-012）；
#        wealth_{t+1} = Σ_j w_j·gross_j + w_cash（cash gross ≡ 1）；
#        漂移 w'_j = w_j·gross_j / wealth、cash' = w_cash / wealth
#        （wealth 归一 ⇒ Σw' + cash' = 1——下一日 Kelly 重新分配全部预算，
#        cash 是 residual，与旧线 h=(w.*gross)./(1+ret) 的结转语义同构）；
#     4. 禁止未来 bar 进入推断——marking 只用于账户更新（t+1 的价格在
#        t+1 决策时才进入 signal；本循环的 t+1 决策用截断后的 ≤t+1 信息）。
#
# D-017 多日版（locked 风险不当 cash）：locked 资产不可交易但价格照漂——
# 其 wealth 贡献 w_j·gross_j 保留在账户（绝不清仓、绝不当 cash）；下一
# 决策日以漂移后的持仓为 held 继续决策（locked 判定按当日 free 语义重判，
# 漂移权重自然延续）。
#
# T4 跨日语义（裁决 C3）：held_maintained 日之后，下一决策日以**维持并
# 漂移后的持仓**为 held 继续决策——不重置、不强制清仓、不重排（裁决
# C3「当日维持持仓」的多日延续：维持是当日决策，次日重新走完整链条，
# 覆盖恢复则正常决策）。
#
# 三 benchmark（D-072）：
#   1. cash：恒 gross 1（净值曲线全 1）；
#   2. EW daily：每个决策日再平衡到 benchmark_universe(el, t) 等权
#      （候选集外生——D-020：E^trade ∧ T^exec，与 model_admitted 无关；
#      空候选日 = 该日无 benchmark 持仓，净值不动——诚实语义，不注入
#      默认资产）；
#   3. EW buy-and-hold：首日（t_start）等权买入后不再平衡（持有漂移；
#      H8 首日定义：候选集 = E^trade ∧ T^exec）。
# D-073 纪律（钉死）：只报告三者曲线与 EW_daily − EW_buyhold 差
# （ew_rebalance_gap）；**不做「volatility pump」因果解释**——rebalancing
# premium 的 counterfactual 分解未完成前，该差只是报告量。
#
# reference 级（D-082/D-083）：单线程、无缓存、无 scheduler、无优化——
# 多日循环的每成本就是 N 天 × 单日成本；fail-loudly（SPEC §56）；固定
# seed（SPEC §36）；全部判据与回测收益/Sharpe 无关（SPEC §95）。
# =============================================================================

using Random, LinearAlgebra, Statistics

export Gate0DayRecord, Gate0BacktestResult, run_gate0_backtest

"""
    Gate0DayRecord

单决策日的回测记录。字段（构造后只读）：

- `t`：决策日；
- `decision`：`Gate0DayDecision`（driver 的完整决策记录——含 D-076 集中
  诊断、证书、方差分解、T4/all_cash 分支诊断）；
- `held_before`：当日决策前的持仓权重（universe 维；T4 日 = 维持对象）；
- `gross_realized`：t→t+1 的 marking gross（universe 维；
  `adj[t+1,j]/adj[t,j]`）；
- `wealth_factor`：当日实现财富因子 `Σ w_j·gross_j + w_cash`（净值乘子）；
- `equity_after`：t+1 时点净值（起点 1.0 的累计乘积）。
"""
struct Gate0DayRecord
    t::Int
    decision::Gate0DayDecision
    held_before::Vector{Float64}
    gross_realized::Vector{Float64}
    wealth_factor::Float64
    equity_after::Float64
end

"""
    Gate0BacktestResult

多日回测结果。字段（构造后只读）：

- `t_start` / `t_end`：决策区间（决策日 t ∈ t_start:(t_end-1)；t_end 是
  最后 marking 日——最后决策日的收益由 t_end 的 marking 实现）；
- `days`：逐日记录（`Vector{Gate0DayRecord}`，长度 = t_end − t_start）；
- `equity`：账户净值曲线（长度 = t_end − t_start + 1；起点 1.0）；
- `equity_cash` / `equity_ew_daily` / `equity_ew_buyhold`：三 benchmark
  曲线（同长度同 marking 日口径；D-072）；
- `ew_rebalance_gap`：`equity_ew_daily − equity_ew_buyhold`（D-073：只
  报告，不做 volatility pump 因果解释）。
"""
struct Gate0BacktestResult
    t_start::Int
    t_end::Int
    days::Vector{Gate0DayRecord}
    equity::Vector{Float64}
    equity_cash::Vector{Float64}
    equity_ew_daily::Vector{Float64}
    equity_ew_buyhold::Vector{Float64}
    ew_rebalance_gap::Vector{Float64}
end

# ---------------------------------------------------------------------------
# marking gross（旧 SPEC §42 第 5 步；D-012 marking/signal 分界）
# ---------------------------------------------------------------------------

"""
    _marking_gross(mf, t, w) -> Vector{Float64}    # N 维

t→t+1 的 marking gross：`gross_j = adj[t+1,j] / adj[t,j]`（carried marking
序列——缺 bar 处 carry forward 是 adj 的字段语义）。对 `w_j > 0` 的持仓
列要求两端价格有限且正（持仓必须有合法 marking——fail loudly，SPEC §56；
w_j = 0 的列不校验——0·NaN 永不进入财富，旧线 locked_wealth 同语义）。
"""
function _marking_gross(mf::MarketFacts, t::Int, w::AbstractVector{Float64})
    T, N = size(mf.observed)
    (1 <= t < T) ||
        throw(ArgumentError("_marking_gross: need 1 ≤ t < T (got t=$t, T=$T)"))
    length(w) == N ||
        throw(DimensionMismatch("_marking_gross: w length $(length(w)) ≠ N=$N"))
    gross = Vector{Float64}(undef, N)
    for j in 1:N
        p0 = mf.adj[t, j]
        p1 = mf.adj[t + 1, j]
        if w[j] > 0
            (isfinite(p0) && isfinite(p1) && p0 > 0 && p1 > 0) ||
                error("_marking_gross: 持仓资产 $j 在 t=$t 缺合法 marking 价格（adj[$t,$j]=$p0, adj[$(t+1),$j]=$p1）——D-017：locked/持仓风险必须有 marking，绝不当 cash")
            gross[j] = p1 / p0
        else
            gross[j] = (isfinite(p0) && isfinite(p1) && p0 > 0 && p1 > 0) ?
                       p1 / p0 : NaN   # 未持仓列的 gross 仅记录（不进财富）
        end
    end
    gross
end

# ---------------------------------------------------------------------------
# 三 benchmark（D-072/D-073/D-020/H8）
# ---------------------------------------------------------------------------

"""
    _benchmark_curves(mf, el, t_start, t_end)
        -> (; equity_cash, equity_ew_daily, equity_ew_buyhold, ew_rebalance_gap)

三 benchmark 净值曲线（内部函数；决策日 t ∈ t_start:(t_end-1)、marking
t+1——与主账户同口径）：

- **cash**（D-072 #1）：gross ≡ 1 ⇒ 曲线恒 1；
- **EW daily**（D-072 #2）：每个决策日再平衡到 `benchmark_universe(el, t)`
  等权（外生候选集——D-020：E^trade ∧ T^exec；与 model_admitted 无关）；
  wealth_t = 候选集上 gross 的等权平均；空候选日 wealth = 1（无持仓）；
- **EW buy-and-hold**（D-072 #3 / H8）：首日 t_start 等权买入候选集，
  之后不再平衡——权重按 gross 漂移（wealth 归一），cash 分量同漂移；
- `ew_rebalance_gap` = EW_daily − EW_buyhold（**D-073：只报告差异，不做
  volatility pump 因果解释**——rebalancing premium 的 counterfactual 分解
  未完成，该差不是任何机制的证明）。
"""
function _benchmark_curves(mf::MarketFacts, el::Eligibility,
                           t_start::Int, t_end::Int)
    T, N = size(mf.observed)
    n_days = t_end - t_start
    # --- cash（恒 1）---
    equity_cash = ones(n_days + 1)
    # --- EW daily（每日再平衡；候选集外生 D-020）---
    eq = 1.0
    equity_ewd = [1.0]
    for t in t_start:(t_end - 1)
        mask = benchmark_universe(el, t)
        n = count(mask)
        if n == 0
            eq *= 1.0                       # 空候选：无持仓，净值不动
        else
            gross = _marking_gross(mf, t, mask ./ n)
            eq *= sum(gross[j] for j in findall(mask)) / n
        end
        push!(equity_ewd, eq)
    end
    # --- EW buy-and-hold（首日等权 H8；持有漂移不再平衡）---
    mask0 = benchmark_universe(el, t_start)
    n0 = count(mask0)
    w_bnh = n0 == 0 ? zeros(N) : Float64[mask0[j] / n0 for j in 1:N]
    cash_bnh = n0 == 0 ? 1.0 : 0.0
    eq = 1.0
    equity_bnh = [1.0]
    for t in t_start:(t_end - 1)
        gross = _marking_gross(mf, t, w_bnh)
        wealth = dot(w_bnh, gross) + cash_bnh
        (isfinite(wealth) && wealth > 0) ||
            error("_benchmark_curves: buy-and-hold wealth 非正（t=$t）——marking 非法")
        eq *= wealth
        push!(equity_bnh, eq)
        w_bnh = [w_bnh[j] * gross[j] / wealth for j in 1:N]
        cash_bnh = cash_bnh / wealth
    end
    (; equity_cash = equity_cash, equity_ew_daily = equity_ewd,
       equity_ew_buyhold = equity_bnh,
       ew_rebalance_gap = equity_ewd .- equity_bnh)
end

# ---------------------------------------------------------------------------
# 主入口：run_gate0_backtest
# ---------------------------------------------------------------------------

"""
    run_gate0_backtest(mf, el; t_start, t_end, held0, mode, seed, kwargs...)
        -> Gate0BacktestResult

sequential 多日回测（D-089 短窗口级：5→20 天受控运行；60/501 天在护栏
外归 owner 分批，D-088 边界由调用方遵守）。

循环（旧 SPEC §42；逐步对照见文件头）：对每个决策日 t ∈
t_start:(t_end-1)——

1. `single_day_decision(mf, el, held; t = t, seed = seed, mode = mode,
   kwargs...)`（因果截断在 driver 内部——**构造性**保证 ≤t；本循环不
   另做截断，因果性单点拥有；kwargs 透传 driver 的数值参数
   `S_reference / diag_S / min_scenarios / max_scenarios /
   weight_tol / utility_tol / kelly_tol`——数值配置语义，D-066）；
2. marking gross `gross_j = adj[t+1,j]/adj[t,j]`（`_marking_gross`——持仓
   列必须有合法 marking，fail loudly）；
3. 财富因子 `wealth = Σ_j w_j·gross_j + w_cash`（cash gross ≡ 1；D-017
   多日版：locked 资产价格照漂、财富贡献保留、绝不当 cash）；
4. 持仓漂移 `held'_j = w_j·gross_j / wealth`（cash 隐含 1 − Σheld'——
   下一日 Kelly 重新分配全部预算，cash 是 residual）。

分支的跨日语义：
- **T4（裁决 C3）**：`held_maintained` 日 w = held（维持）→ 结转照常
  漂移；下一决策日以漂移后的维持持仓为 held 继续决策——**不重置、不
  强制清仓**（覆盖恢复即正常决策）；
- **全 cash**（R 空 / 无 active）：w = 0、cash = 1 → wealth = 1、净值
  不动，照常结转；
- driver 的其余 fail-loudly（propriety 红、数据非法、积分预算耗尽等）
  **原样传导**——本循环只消费 T4/all_cash 的决策语义，不吞任何错误。

`mode` 默认 `:adaptive`（生产路径；locked 非零日的 adaptive 请求由
driver/quadrature 直接经 locked 通道求解——P0-4 收口，不存在「有 locked
即回落 `:reference` 固定 S」的 production fallback，逐日 diagnostics 可见）。

返回 `Gate0BacktestResult`（逐日记录 + 账户净值 + 三 benchmark 曲线 +
rebalance gap——D-072/D-073）。
"""
function run_gate0_backtest(mf::MarketFacts, el::Eligibility;
                            t_start::Int, t_end::Int,
                            held0::Vector{Float64} =
                                zeros(size(mf.observed, 2)),
                            mode::Symbol = :adaptive,
                            seed::UInt64 = _DRIVER_DEFAULT_SEED,
                            # b′（μ 通道部分 QMC）：透传 driver；默认 false
                            # 逐位不变（E3 判据已成立；60-day 前置待验证）。
                            mu_qmc::Bool = false,
                            # CHISQ-QMC（A 门过门配置，CQ_*）：透传 driver；默认 false。
                            mu_chisq_qmc::Bool = false,
                            kwargs...)
    T, N = size(mf.observed)
    (1 <= t_start < t_end <= T) ||
        throw(ArgumentError("run_gate0_backtest: need 1 ≤ t_start < t_end ≤ T (got t_start=$(t_start), t_end=$(t_end), T=$(T)——空区间或越界)"))
    t_start - 1 >= WARMUP ||
        throw(ArgumentError("run_gate0_backtest: t_start=$(t_start) 不足（需 t_start−1 ≥ WARMUP=$(WARMUP)——driver 训练行前置）"))
    length(held0) == N ||
        throw(DimensionMismatch("run_gate0_backtest: held0 length $(length(held0)) ≠ N=$N"))
    all(x -> isfinite(x) && x >= 0, held0) ||
        throw(ArgumentError("run_gate0_backtest: held0 must be non-negative and finite"))
    sum(held0) <= 1.0 + 1e-12 ||
        throw(ArgumentError("run_gate0_backtest: Σ held0 = $(sum(held0)) > 1（总预算超限）"))

    held = Float64.(collect(held0))
    equity = 1.0
    equity_curve = [1.0]
    records = Gate0DayRecord[]
    for t in t_start:(t_end - 1)
        held_before = copy(held)
        # 步骤 1-2：决策（因果截断在 driver 内；T4/all_cash 语义转译于此）
        dec = single_day_decision(mf, el, held; t = t, seed = seed,
                                  mode = mode, mu_qmc = mu_qmc,
                                  mu_chisq_qmc = mu_chisq_qmc, kwargs...)
        # 步骤 3：marking gross + 财富因子（D-017：locked 照漂不当 cash）
        gross = _marking_gross(mf, t, dec.w_universe)
        wealth = dot(dec.w_universe, gross) + dec.w_cash
        (isfinite(wealth) && wealth > 0) ||
            error("run_gate0_backtest: t=$t 的财富因子非正（wealth=$wealth）——决策或 marking 非法")
        # 步骤 4：持仓漂移（cash 隐含 1−Σheld'——下一日 Kelly 重新分配；
        # T4 日 w=held 维持 → 漂移即维持的跨日延续，不重置不清仓）
        equity *= wealth
        push!(equity_curve, equity)
        held = [dec.w_universe[j] * gross[j] / wealth for j in 1:N]
        push!(records, Gate0DayRecord(t, dec, held_before, gross,
                                      wealth, equity))
    end

    bm = _benchmark_curves(mf, el, t_start, t_end)
    Gate0BacktestResult(t_start, t_end, records, equity_curve,
                        bm.equity_cash, bm.equity_ew_daily,
                        bm.equity_ew_buyhold, bm.ew_rebalance_gap)
end
