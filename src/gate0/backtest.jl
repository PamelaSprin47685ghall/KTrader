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
# 把 single_day_decision 串成 sequential 多日回测。原 Gate-0 reference
# 验证范围为 5→20 日；长区间可选择带因果历史预测复用与检查点的相同
# sequential 账户循环。缓存仅避免重复历史预测，不改变任何模型对象。
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
# reference 推断级（D-082/D-083）：单线程、无 scheduler、无降级——
# 外层可选缓存历史预测、原子写入检查点；fail-loudly（SPEC §56）；固定
# seed（SPEC §36）；全部判据与回测收益/Sharpe 无关（SPEC §95）。
# =============================================================================

using Random, LinearAlgebra, Statistics, Serialization, SHA

export Gate0DayRecord, Gate0BacktestResult, run_gate0_backtest

"""An intentional, checkpointed pause is not a completed backtest."""
struct BacktestPaused <: Exception
    next_t::Int
    checkpoint::String
end
Base.showerror(io::IO, e::BacktestPaused) =
    print(io, "Backtest paused before decision ", e.next_t,
          "; resume checkpoint ", e.checkpoint, " (not a completed backtest)")

function _backtest_signature(mf, el, options)
    io = IOBuffer()
    serialize(io, (VERSION, mf, el, options))
    for path in sort(filter(p -> endswith(p, ".jl"), readdir(@__DIR__; join=true)))
        write(io, basename(path), '\0', read(path))
    end
    manifest = joinpath(@__DIR__, "..", "..", "Manifest.toml")
    isfile(manifest) && write(io, read(manifest))
    bytes2hex(sha256(take!(io)))
end

function _write_backtest_checkpoint(path::String, payload)
    mkpath(dirname(abspath(path)))
    temp, io = mktemp(dirname(abspath(path)))
    try
        serialize(io, payload)
        flush(io)
        close(io)
        mv(temp, path; force=true) # completed file replaces the old checkpoint
    finally
        isopen(io) && close(io)
        isfile(temp) && rm(temp)
    end
    nothing
end

"""Fill the initial prequential forecast history before fitting today's full
posterior. A bounded checkpoint slice must not repeat the same expensive
today-posterior on every restart while historical forecasts are being built.
Only origins strictly before `first_t` are evaluated; their own causal
preprocessing and posterior are unchanged. This is an execution reordering,
not a rolling-window or an innovation-law approximation.
"""
function _prime_historical_forecasts!(cache::_ForecastCache, first_t::Int;
                                       progress::Bool=false,
                                       workers::Int=1,
                                       required_universe::Union{Nothing,Vector{Int}}=nothing,
                                       max_new_forecasts::Union{Nothing,Int}=nothing)
    mf, el = cache.market, cache.eligibility
    tol, max_cells, u_span = cache.options
    N = size(mf.observed, 2)
    if required_universe !== nothing
        isempty(required_universe) && throw(ArgumentError("required_universe is empty"))
        all(j -> 1 <= j <= N,required_universe) ||
            throw(ArgumentError("required_universe contains an invalid asset"))
    end
    1 <= workers <= Threads.nthreads() ||
        throw(ArgumentError("forecast workers must be between 1 and Julia thread count"))
    origins = Int[]
    for u in (WARMUP+1):first_t
        origin = u-1
        haskey(cache.entries, origin) && continue
        # This row is absent from the prequential design unless an adjacent
        # pair of actual (not carried marking) prices exists.
        if required_universe === nothing
            any(mf.observed[u-1,j] && mf.observed[u,j] for j in 1:N) || continue
        else
            all(j -> mf.observed[u-1,j] && mf.observed[u,j] &&
                     el.model_admitted[u-1,j],required_universe) || continue
        end
        push!(origins, origin)
    end
    k = 1
    while k <= length(origins)
        # Slice/checkpoint tests must never compute beyond their requested
        # bound. Keep scheduling deterministic, independent of worker count.
        budget = max_new_forecasts === nothing ? workers :
                 min(workers, max_new_forecasts - cache.additions)
        budget > 0 || error("priming exhausted forecast slice without pause")
        upto = min(length(origins), k+budget-1)
        dates = view(origins, k:upto)
        calculate(origin) = _historical_forecast(mf,el,origin;
                              tol=tol,max_cells=max_cells,u_span=u_span,
                              preprocessed=cache)
        forecasts = if length(dates) == 1
            [calculate(dates[1])]
        else
            # Each origin is causally self-contained; workers only read market
            # facts. The shared cache and durable checkpoint are updated in
            # increasing date order AFTER all worker tasks have finished.
            tasks = [Threads.@spawn calculate(d) for d in dates]
            fetch.(tasks)
        end
        for (origin,forecast) in zip(dates,forecasts)
            _cache_forecast!(cache, origin, forecast)
            if progress && cache.additions % 128 == 0
                println("historical_forecasts=$(length(cache.entries)) origin=$origin date=$(mf.dates[origin])")
                flush(stdout)
            end
        end
        k = upto+1
    end
    nothing
end

"""
    Gate0DayRecord

单决策日的回测记录。字段（构造后只读）：

- `t`：决策日；
- `decision`：`Gate0DayDecision`（driver 的完整决策记录——含 D-076 集中
  诊断、证书、方差分解、T4/all_cash 分支诊断）；
- `held_before`：当日决策前的持仓权重（universe 维；T4 日 = 维持对象）；
- `gross_realized`：t→t+1 的 marking gross（universe 维；
  `adj[t+1,j]/adj[t,j]`；未持仓且缺价格列记中性 1.0）；
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

"""
    _assert_lockfree_independent_window(mf, el, start, stop, held0)

Prove from immutable policy/market masks that no portfolio produced by the
driver can create a locked position in the stated window. With the original
zero initial account or initial holdings free at start, the driver's only
dependency on prior `held` is its locked set; consequently full Bayesian
single-day decisions can be evaluated independently and their exact marking
and account drift replayed in date order. If even one date violates the
certificate, parallel evaluation is forbidden; use sequential backtest.
"""
function _assert_lockfree_independent_window(mf::MarketFacts,
                                              el::Eligibility,
                                              t_start::Int,t_end::Int,
                                              held0::AbstractVector{Float64})
    T,N=size(mf.observed)
    1<=t_start<t_end<=T ||
        throw(ArgumentError("lockfree replay requires a valid nonempty interval"))
    length(held0)==N && all(x->isfinite(x)&&x>=0,held0) &&
        sum(held0)<=1.0+1e-12 ||
        throw(ArgumentError("lockfree replay requires a feasible initial account"))
    first_free=free(el,t_start)
    any(j->held0[j]>0 && !first_free[j],1:N) &&
        error("lockfree parallel decisions are forbidden: initial locked holding")
    for t in t_start:(t_end-1)
        f=free(el,t)
        for j in 1:N
            if el.model_admitted[t,j] && !f[j]
                error("lockfree parallel decisions are forbidden: " *
                      "date=$(mf.dates[t]) model asset=$(mf.symbols[j]) is not free")
            end
        end
    end
    nothing
end

"""Return the EXACT union of historical forecast origins that a lock-free
selective prequential driver may consume over this calendar interval.

Rows are the same observed adjacent-bar pairs, the same first risk resolver,
and the same structural-rank floor as in single_day_decision. An active-set
change rebuilds the date masks; on unchanged days a new row is appended.
This avoids re-creating ~14,000 bit masks on every one of 502 decisions.
No posterior fit or portfolio decision is made by this function.
"""
function _lockfree_required_origins(mf::MarketFacts,el::Eligibility,
                                     t_start::Int,t_end::Int)
    _assert_lockfree_independent_window(mf,el,t_start,t_end,
                                        zeros(size(mf.observed,2)))
    t_start>WARMUP+1 ||
        throw(ArgumentError("lockfree forecast preparation needs warmup"))
    act_prev=Int[]
    row_ids=Int[]
    row_masks=BitVector[]
    required=Set{Int}()
    for t in t_start:(t_end-1)
        act=findall(view(el.model_admitted,t,:))
        isempty(act) && error("lockfree preparation requires a nonempty active risk domain")
        if act!=act_prev
            empty!(row_ids)
            empty!(row_masks)
            act_prev=copy(act)
            last_u=WARMUP
        else
            last_u=isempty(row_ids) ? WARMUP : row_ids[end]
        end
        for u in (last_u+1):t
            mask=BitVector([mf.observed[u-1,j] && mf.observed[u,j] for j in act])
            any(mask) || continue
            push!(row_ids,u)
            push!(row_masks,mask)
        end
        free_act=Int[k for (k,j) in enumerate(act) if free(el,t)[j]]
        R,dropped=resolve_risk_domain(row_ids,row_masks,free_act,Int[],t)
        isempty(R) && error("lockfree preparation refuses empty risk domain")
        floor,peeled=_forecast_risk_floor(row_ids,row_masks,free_act,
                                          Int[],R,t,el,act)
        if !isempty(peeled)
            counts=[count(i->row_masks[i][j],eachindex(row_ids)) for j in R]
            after=[counts[findfirst(==(j),R)] for j in floor]
            before=[counts[findfirst(==(j),R)] for j in peeled]
            if isempty(after) || minimum(after)-maximum(before)<WARMUP
                floor=R
            end
        end
        risk=act[floor]
        for u in row_ids
            all(j->mf.observed[u-1,j] && mf.observed[u,j] &&
                   el.model_admitted[u-1,j],risk) || continue
            push!(required,u-1)
        end
    end
    sort!(collect(required))
end

"""
    replay_independent_gate0(mf, el, decisions; t_start, t_end, held0)

Account/benchmark reducer for independently calculated *complete* adaptive
decisions on a proved lock-free window. It uses the EXACT same marking,
wealth, drift and benchmark equations as run_gate0_backtest; no interpolation,
portfolio changes, per-day PnL assumptions or different Kelly solve.
All decisions must be supplied and numerically certified. No held-maintained
or uncategorised day is accepted. Source/data/numerics signature checking is
the responsibility of the precomputed-decision file reader, not this reducer.
"""
function replay_independent_gate0(mf::MarketFacts,el::Eligibility,
                                  decisions::Vector{Gate0DayDecision};
                                  t_start::Int,t_end::Int,
                                  held0::Vector{Float64}=zeros(size(mf.observed,2)))
    _assert_lockfree_independent_window(mf,el,t_start,t_end,held0)
    length(decisions)==t_end-t_start ||
        throw(DimensionMismatch("independent replay needs one decision per session"))
    N=size(mf.observed,2)
    held=copy(held0)
    equity=1.0
    equity_curve=[equity]
    records=Gate0DayRecord[]
    for (i,dec) in enumerate(decisions)
        t=t_start+i-1
        dec.t==t && dec.mode==:adaptive ||
            error("parallel replay refuses absent, out-of-order or unverified adaptive decision t=$t")
        length(dec.w_universe)==N && all(x->isfinite(x)&&x>=0,dec.w_universe) &&
            isfinite(dec.w_cash) && dec.w_cash>=0 &&
            abs(sum(dec.w_universe)+dec.w_cash-1.0)<=1e-8 ||
            error("parallel replay refuses infeasible decision weights at t=$t")
        c=dec.certificates
        c.converged && c.A<=1e-4 && abs(c.B)<=1e-6 &&
            c.B_audit<=1e-6 && cash_kelly_certified(c.C;tol=1e-6) ||
            error("parallel replay refuses uncertified numerical decision t=$t")
        # Independent decision must use ONLY model-admitted/free coordinates.
        all(j->dec.w_universe[j]==0 ||
            (el.model_admitted[t,j] && free(el,t)[j]),1:N) ||
            error("parallel replay found a position outside the proved free domain")
        before=copy(held)
        gross=_marking_gross(mf,t,dec.w_universe)
        wealth=dot(dec.w_universe,gross)+dec.w_cash
        (isfinite(wealth) && wealth>0) ||
            error("parallel replay encountered nonpositive account wealth at t=$t")
        equity*=wealth
        push!(equity_curve,equity)
        held=[dec.w_universe[j]*gross[j]/wealth for j in 1:N]
        push!(records,Gate0DayRecord(t,dec,before,gross,wealth,equity))
    end
    benchmarks=_benchmark_curves(mf,el,t_start,t_end)
    Gate0BacktestResult(t_start,t_end,records,equity_curve,
                        benchmarks.equity_cash,benchmarks.equity_ew_daily,
                        benchmarks.equity_ew_buyhold,benchmarks.ew_rebalance_gap)
end

# ---------------------------------------------------------------------------
# marking gross（旧 SPEC §42 第 5 步；D-012 marking/signal 分界）
# ---------------------------------------------------------------------------

"""
    _marking_gross(mf, t, w) -> Vector{Float64}    # N 维

t→t+1 的 marking gross：`gross_j = adj[t+1,j] / adj[t,j]`（carried marking
序列——缺 bar 处 carry forward 是 adj 的字段语义）。对 `w_j > 0` 的持仓
列要求两端价格有限且正（持仓必须有合法 marking——fail loudly，SPEC §56；
`w_j = 0` 的列不校验：两端价格合法时记录真实比值，缺失时记中性值
1.0——未持仓列在全部消费端以 `0 · gross = 0` 参与，不产生 NaN 污染
（此前记 NaN 曾使 `0 · NaN = NaN` 污染财富/benchmark 求和；2026-10-11 修复）。
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
                       p1 / p0 : 1.0   # 未持仓缺价格列记中性 1.0——不进财富（0·NaN=NaN 会污染 dot 求和）
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
                            reuse_forecasts::Bool = true,
                            selective_forecasts::Bool = true,
                            forecast_workers::Int = 1,
                            checkpoint::Union{Nothing,String} = nothing,
                            resume::Bool = false,
                            max_new_days::Union{Nothing,Int} = nothing,
                            max_new_forecasts::Union{Nothing,Int} = nothing,
                            progress::Bool = false,
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
    for limit in (max_new_days, max_new_forecasts)
        limit === nothing || limit >= 1 || throw(ArgumentError("backtest slice limits must be positive"))
    end
    (resume || max_new_days !== nothing || max_new_forecasts !== nothing) && checkpoint === nothing &&
        throw(ArgumentError("resume/slicing requires an explicit checkpoint path"))
    max_new_forecasts !== nothing && !reuse_forecasts &&
        throw(ArgumentError("forecast slicing requires reuse_forecasts=true"))
    1 <= forecast_workers <= Threads.nthreads() ||
        throw(ArgumentError("forecast_workers requires 1 ≤ workers ≤ Julia thread count"))
    posterior_options = (Float64(get(kwargs, :posterior_tol, 1e-6)),
                         Int(get(kwargs, :posterior_max_cells, 2048)),
                         Float64(get(kwargs, :u_span, 5.0)))
    cache = reuse_forecasts ? _ForecastCache(mf, el, posterior_options) : nothing
    if cache !== nothing
        cache.selective = selective_forecasts
        cache.workers = forecast_workers
        cache.max_new_forecasts = max_new_forecasts
    end
    signature = checkpoint === nothing ? "" : _backtest_signature(mf, el,
        (t_start, t_end, held0, mode, seed, mu_qmc, mu_chisq_qmc,
         selective_forecasts, (; kwargs...)))

    held = Float64.(collect(held0))
    equity = 1.0
    equity_curve = [1.0]
    records = Gate0DayRecord[]
    next_t = Ref(t_start)
    if checkpoint !== nothing && isfile(checkpoint)
        resume || throw(ArgumentError("checkpoint already exists; use resume=true or a new path"))
        # Only read checkpoints created locally by this program, never untrusted
        # uploaded Serialization files. Verify data, policies, numerics and code.
        saved = deserialize(checkpoint)
        (saved.version == 1 && saved.signature == signature) ||
            error("backtest checkpoint mismatch: data, policy, numerical settings or source code changed")
        saved.next_t == t_start + length(saved.records) || error("backtest checkpoint has inconsistent day count")
        length(saved.equity_curve) == length(saved.records)+1 || error("backtest checkpoint has inconsistent equity count")
        t_start <= saved.next_t <= t_end || error("backtest checkpoint day is outside requested interval")
        held, equity = saved.held, saved.equity
        equity_curve, records = saved.equity_curve, saved.records
        next_t[] = saved.next_t
        cache === nothing || merge!(cache.entries, saved.forecasts)
    elseif resume
        throw(ArgumentError("checkpoint not found: $checkpoint"))
    end
    save_checkpoint() = checkpoint === nothing ? nothing :
        _write_backtest_checkpoint(checkpoint,
            (; version=1, signature, next_t=next_t[], held, equity,
               equity_curve, records,
               forecasts=cache === nothing ? Dict{Int,_ForecastValue}() : cache.entries))
    if cache !== nothing
        cache.on_update = () -> begin
            cache.additions % 8 == 0 && save_checkpoint()
            if max_new_forecasts !== nothing && cache.additions >= max_new_forecasts
                save_checkpoint()
                throw(BacktestPaused(next_t[], checkpoint))
            end
        end
    end
    save_checkpoint()
    new_days = 0
    try
    if cache !== nothing && !cache.selective && next_t[] < t_end &&
       any(view(el.model_admitted, next_t[], :)) &&
       (any(free(el, next_t[])) || any(>(0.0), held))
        _prime_historical_forecasts!(cache, next_t[]; progress=progress,
            workers=forecast_workers,max_new_forecasts=max_new_forecasts)
    end
    for t in next_t[]:(t_end - 1)
        if max_new_days !== nothing && new_days >= max_new_days
            throw(BacktestPaused(t, checkpoint))
        end
        held_before = copy(held)
        # 步骤 1-2：决策（因果截断在 driver 内；T4/all_cash 语义转译于此）
        dec = single_day_decision(mf, el, held; t = t, seed = seed,
                                  mode = mode, mu_qmc = mu_qmc,
                                  mu_chisq_qmc = mu_chisq_qmc,
                                  forecast_cache = cache, kwargs...)
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
        next_t[] = t+1
        new_days += 1
        save_checkpoint()
        if progress
            println("completed=$(length(records))/$(t_end-t_start) date=$(mf.dates[t]) t=$t M=$(dec.M) equity=$equity forecasts=$(cache === nothing ? 0 : length(cache.entries))")
            flush(stdout)
        end
    end
    catch
        save_checkpoint()
        rethrow() # numerical failures never become success or a cash fallback
    end

    bm = _benchmark_curves(mf, el, t_start, t_end)
    Gate0BacktestResult(t_start, t_end, records, equity_curve,
                        bm.equity_cash, bm.equity_ew_daily,
                        bm.equity_ew_buyhold, bm.ew_rebalance_gap)
end
