# =============================================================================
# KTraderGate0 backtest — Step 15 多日测试（tiny 级，D-089 短窗口）
# =============================================================================
#
# 测试对象：src/gate0/backtest.jl（run_gate0_backtest / 结转 / 三 benchmark）。
#
# 覆盖（任务书七项）：
# 1. 5 日端到端（合成小 N——逐日记录完整、净值曲线有限无 NaN、benchmark
#    曲线同口径）；
# 2. 持仓结转守恒（w·gross 漂移公式的手工数值对照）；
# 3. 因果性（截断输入与全量输入在同一决策日逐位一致——复用 driver 的
#    因果断言语义，多日版）；
# 4. locked 跨日（价格漂移计入财富、不清仓、不当 cash——D-017 多日版）；
# 5. T4 跨日（覆盖不足日 held_maintained → 下一日以漂移持仓继续决策，
#    不 crash 不重置——裁决 C3）；
# 6. benchmark（无漂移数据上 EW daily ≡ buy-hold、确定性漂移数据上构造性
#    分离（手工精确值）、cash 恒 1——D-072/D-073/D-020）；
# 7. fail loudly（空区间 / t 越界 / WARMUP 前置 / held0 长度与预算）。
#
# 纪律：tiny 级合成数据（N=2-4、T≤282、固定 seed）、无 I/O、**不做
# 60/501 天**（护栏外，D-088 边界由调用方遵守）；本任务不运行测试
# （Engineer 纪律——运行验证由 Manager 安排 DevOps 受控执行，
# ≤60s/RSS 护栏）；全部断言与回测收益/Sharpe 无关（SPEC §95）。
#
# 加载链（依赖顺序）：geometry → modes → response → posterior → oof →
# innovation → predictive → kelly → quadrature → driver → backtest。

using Test, Random, LinearAlgebra, Statistics, Dates

# 加载链修复（Step 16 短窗口轮，同 driver_tests 的坑）：backtest.jl 消费
# market.jl 的 MarketFacts/Eligibility/benchmark_universe——原链从 geometry
# 起漏 market（UndefVarError 风险）；module 内需自含 using Dates
# （market.jl 依赖骨架层 Dates）。完整依赖顺序：market → geometry →
# modes → response → posterior → oof → innovation → predictive →
# kelly → quadrature → driver → backtest。
module Gate0BacktestEnv
    using Dates
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "market.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "predictive.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "kelly.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "quadrature.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "driver.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "backtest.jl"))
end
using .Gate0BacktestEnv

# ---------------------------------------------------------------------------
# 合成 fixture（与 driver_tests.jl 同构——文件自含）：随机游走 log 价格；
# 可选 IPO（前 ipo_first-1 行 observed=false、价格 NaN）。
# ---------------------------------------------------------------------------
function _mk_mf(; N::Int = 2, T::Int = 280, seed::Integer = 2026,
                ipo_asset::Int = 0, ipo_first::Int = 1,
                drift::Float64 = 0.0)
    rng = MersenneTwister(seed)
    # drift 参数（Step 16 短窗口轮）：常数日漂移——主 fixture 传非零值
    # 使 mode 坐标的 DC 分量有信号（α₀ 有信息）。无信号随机游走 + DC
    # 形态下 fold 的 α₀ 无信息 ⇒ u₀ 左尾被积趋于常数 ⇒ 二维积分数学
    # 上不收敛（tail 证书红是正确行为——同 Wave 4 predictive 的先例
    # 定性）；端到端断言（记录/守恒/因果性）与信号强度无关。
    logp = cumsum(drift .+ 0.01 .* randn(rng, T, N); dims = 1)
    close = exp.(logp)
    adj = copy(close)
    observed = trues(T, N)
    if ipo_asset > 0
        observed[1:(ipo_first - 1), ipo_asset] .= false
        close[1:(ipo_first - 1), ipo_asset] .= NaN
        adj[1:(ipo_first - 1), ipo_asset] .= NaN
    end
    dates = collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T - 1)))
    MarketFacts(dates, ["A$j" for j in 1:N], close, adj, observed)
end
# =============================================================================
# 单文件多场景入口（ENV 选择，避免拆文件重复冷启动 JIT 编译）。
# 单个场景运行：BACKTEST_SCENARIO=1 julia test/gate0/backtest_tests.jl
# 分组运行（同一进程编译一次，摊薄冷启动）：BACKTEST_SCENARIO="1,2" 一次
# 跑场景 1 和 2；不设 ENV 时默认跑场景 1。7 个场景累积超单命令预算，
# 故全量模式需分组调度（如 "1,2"、"3,4"、"5,6,7"）。
# =============================================================================
const _BT_SCENARIO = get(ENV, "BACKTEST_SCENARIO", "1")
const _BT_SCENARIOS = Set(split(_BT_SCENARIO, ','))

if "1" in _BT_SCENARIOS
@testset "backtest (1) 5-day end-to-end" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)

    # ===================================================================
    # (1) 5 日端到端：逐日记录完整、净值曲线有限无 NaN
    # ===================================================================
    mf = _mk_mf(N = 2, T = 280, seed = 31, drift = 0.003)
    el = Eligibility(mf)
    res5 = run_gate0_backtest(mf, el; t_start = 270, t_end = 273,
                              seed = SEED_DEC, KW...)
    @test res5.t_start == 270 && res5.t_end == 273
    @test length(res5.days) == 3                          # 决策日 270..272
    @test [d.t for d in res5.days] == collect(270:272)
    @test all(d -> d.decision isa Gate0DayDecision, res5.days)
    @test all(d -> all(isfinite, d.gross_realized), res5.days)
    @test all(d -> isfinite(d.wealth_factor) && d.wealth_factor > 0, res5.days)
    @test length(res5.equity) == 4                        # 起点 1.0 + 3 个 marking 日
    @test res5.equity[1] == 1.0
    @test all(x -> isfinite(x) && x > 0, res5.equity)     # 无 NaN、净值正
    @test res5.equity ≈ cumprod([1.0; [d.wealth_factor for d in res5.days]])
    @test all(x -> isfinite(x), res5.equity_ew_daily)
    @test all(x -> isfinite(x), res5.equity_ew_buyhold)
    @test length(res5.equity_cash) == length(res5.equity_ew_daily) ==
          length(res5.equity_ew_buyhold) == 4             # 三 benchmark 同口径
    # kwargs 透传验证（diag_S/S_reference 到达 driver）
    @test res5.days[1].decision.M == 32

    # ===================================================================
    end


end

if "2" in _BT_SCENARIOS
@testset "backtest (2) holdings rollover conservation" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
# (2) 持仓结转守恒：w·gross 漂移公式的手工数值对照（2 日）
    # ===================================================================
    mf2 = _mk_mf(N = 2, T = 280, seed = 32, drift = 0.003)
    el2 = Eligibility(mf2)
    held0 = [0.2, 0.1]
    res2 = run_gate0_backtest(mf2, el2; t_start = 270, t_end = 272,
                              held0 = held0, seed = SEED_DEC, KW...)
    dec1 = single_day_decision(mf2, el2, held0; t = 270, seed = SEED_DEC,
                               mode = :reference, S_reference = 32,
                               diag_S = 16, posterior_tol = 1e-2)
    gross1 = [mf2.adj[271, j] / mf2.adj[270, j] for j in 1:2]
    wealth1 = dot(dec1.w_universe, gross1) + dec1.w_cash
    @test res2.days[1].decision.w_universe == dec1.w_universe  # 首日决策一致
    @test res2.days[1].wealth_factor ≈ wealth1                 # 财富因子手工对照
    @test res2.equity[2] ≈ wealth1
    @test res2.days[1].held_before == held0
    # 漂移公式：held'_j = w_j·gross_j / wealth（cash 隐含 1−Σheld'）
    drift1 = [dec1.w_universe[j] * gross1[j] / wealth1 for j in 1:2]
    @test res2.days[2].held_before ≈ drift1
    @test sum(res2.days[2].held_before) <= 1.0 + 1e-12         # 预算不超

    # ===================================================================
    end


end

if "3" in _BT_SCENARIOS
@testset "backtest (3) causality truncation" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
    mf = _mk_mf(N = 2, T = 280, seed = 31, drift = 0.003)
    el = Eligibility(mf)
# (3) 因果性：截断输入与全量输入逐日逐位一致（未来行不存在的构造保证）
    # ===================================================================
    t_cut = 272
    mf_c = MarketFacts(mf.dates[1:t_cut], mf.symbols,
                       mf.close[1:t_cut, :], mf.adj[1:t_cut, :],
                       mf.observed[1:t_cut, :])
    el_c = Eligibility(el.model_admitted[1:t_cut, :],
                       el.trade_eligible[1:t_cut, :],
                       el.executable[1:t_cut, :])
    res_full = run_gate0_backtest(mf, el; t_start = 270, t_end = t_cut,
                                  seed = SEED_DEC, KW...)
    res_cut = run_gate0_backtest(mf_c, el_c; t_start = 270, t_end = t_cut,
                                 seed = SEED_DEC, KW...)
    @test [d.decision.w_universe for d in res_full.days] ==
          [d.decision.w_universe for d in res_cut.days]    # 逐日逐位一致
    @test res_full.equity == res_cut.equity

    # ===================================================================
    end


end

if "4" in _BT_SCENARIOS
@testset "backtest (4) locked multi-day" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
# (4) locked 跨日：价格漂移计入财富、不清仓、不当 cash（D-017 多日版）
    # ===================================================================
    mf_l = _mk_mf(N = 2, T = 280, seed = 34, drift = 0.003)
    trade_l = trues(280, 2)
    trade_l[:, 2] .= false                                 # asset 2 不可交易
    el_l = Eligibility(mf_l, trade_l)
    held_l = [0.2, 0.3]                                    # asset 2 held ∧ ¬free
    res_l = run_gate0_backtest(mf_l, el_l; t_start = 270, t_end = 273,
                               held0 = held_l, seed = SEED_DEC, KW...)
    @test all(d -> d.decision.w_universe[2] > 0, res_l.days)      # 不被清仓
    @test all(d -> d.decision.locked_exposure > 0, res_l.days)    # locked 敞口保留
    @test all(d -> d.decision.mode != :all_cash, res_l.days)      # 不当 cash 出路
    # 漂移计入财富：首日 wealth = Σ w·gross + cash（locked 列照漂）
    gross_l = [mf_l.adj[271, j] / mf_l.adj[270, j] for j in 1:2]
    w1 = res_l.days[1].decision.w_universe
    @test res_l.days[1].wealth_factor ≈
          dot(w1, gross_l) + res_l.days[1].decision.w_cash
    # locked 资产权重按价格漂移（不被重置为 held0）
    @test res_l.days[2].held_before[2] ≈ w1[2] * gross_l[2] / res_l.days[1].wealth_factor

    # ===================================================================
    end


end

if "5" in _BT_SCENARIOS
@testset "backtest (5) T4 multi-day" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
# (5) T4 跨日（裁决 8 fixture 重设计——实证形态）
    # ===================================================================
    # 裁决 8 构造：资产 2 历史「长且分散」（rows 内 fold 1/2 各一行——
    # fold train 的 e₂ 自由度保障 propriety）+ T4 日 joint 行稀少 +
    # 次日恢复（观测行 279+）。**静态推导（adjudication8_feasibility.md）
    # 预测此形态 J = 2 ≥ 2 → T4 不触发**——本 fixture 为该推导的实证
    # 检验：若 mode ≠ held_maintained，推导成立（解耦不可构造——回报
    # Manager 裁决语义层选项）；若意外触发，推导有漏洞再挖。
    #
    # 【裁决 9 注记（2026-10-10）】上述「解耦不可构造」成立于 legacy fold
    # 语义（fold 已退役为 reference/diagnostic——见 oof.jl 文件头与 driver
    # 步骤 8 注释）；production（prequential）语义下的「触发→恢复」正向
    # 路径覆盖见 testset (8) 与 docs/T4_MULTIDAY_SEMANTICS_ADJUDICATION9.md。
    # 本注记不改本 testset 的任何断言。
    mf_t4 = _mk_mf(N = 2, T = 282, seed = 35)
    # 资产 2 的观测 mask：仅目标日 258/262（rows 内、分散于 fold 1/2）
    # + 279 起（恢复）；其余 false（价格 NaN——MarketFacts 契约：
    # NaN 化在构造前，不绕过 D-012 校验）
    mask_t4 = begin
        m = falses(282, 2)
        m[:, 1] .= true
        m[[257, 258, 261, 262], 2] .= true   # 相邻对→目标日 258/262
        m[278:282, 2] .= true                 # 恢复（目标日 279+）
        m
    end
    close_t4 = [mask_t4[t, j] ? mf_t4.close[t, j] : NaN for t in 1:282, j in 1:2]
    adj_t4 = [mask_t4[t, j] ? mf_t4.adj[t, j] : NaN for t in 1:282, j in 1:2]
    mf_t4 = MarketFacts(mf_t4.dates, mf_t4.symbols, close_t4, adj_t4,
                        BitMatrix(mask_t4))
    trade_t4 = trues(282, 2)
    trade_t4[:, 2] .= false                                 # 资产 2 不可交易（locked）
    el_t4 = Eligibility(mf_t4, trade_t4)
    held_t4 = [0.1, 0.3]                                    # 资产 2 locked
    res_t4 = run_gate0_backtest(mf_t4, el_t4; t_start = 279, t_end = 281,
                                held0 = held_t4, seed = SEED_DEC, KW...)
    @test length(res_t4.days) == 2                          # 全程完成（不 crash）
    # 实证断言：T4 日（t=279）的 mode——推导预测 :reference（J=2 不触发）；
    # 若实测 :held_maintained 则推导被推翻（后续断言按 T4 路径走）
    @test res_t4.days[1].decision.mode in (:held_maintained, :reference)
    if res_t4.days[1].decision.mode == :held_maintained
        @test res_t4.days[1].decision.w_universe == held_t4 # 当日维持持仓
        @test res_t4.days[1].held_before == held_t4
        @test isfinite(res_t4.days[1].wealth_factor)        # 维持持仓照常漂移
        # 下一日以漂移后的维持持仓为 held（不重置、不清仓）
        g1 = [mf_t4.adj[280, j] / mf_t4.adj[279, j] for j in 1:2]
        wf1 = res_t4.days[1].wealth_factor
        @test res_t4.days[2].held_before ≈ [held_t4[j] * g1[j] / wf1 for j in 1:2]
        @test res_t4.days[2].decision.mode != :held_maintained  # 恢复
    else
        # 推导成立路径：J = {258, 262} ≥ 2 → 正常决策（T4 不触发）——
        # 解耦不可构造的实证证据（adjudication8_feasibility.md 的结论）
        @test res_t4.days[1].decision.mode == :reference
        @test res_t4.days[1].decision.R_universe == [1, 2]  # 全域 R
    end
    @test res_t4.days[end].decision.mode != :held_maintained
    @test all(x -> isfinite(x) && x > 0, res_t4.equity)     # 净值全程良定

    # ===================================================================
    end


end

if "6" in _BT_SCENARIOS
@testset "backtest (6) benchmark" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
# (6) benchmark（D-072/D-073/D-020）
    # ===================================================================
    # (6a) 无漂移数据（价格恒定）：EW daily ≡ buy-hold ≡ cash ≡ 1
    T_f, N_f = 6, 2
    flat = MarketFacts(collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T_f - 1))),
                       ["A1", "A2"], ones(T_f, N_f), ones(T_f, N_f),
                       trues(T_f, N_f))
    bmf = Gate0BacktestEnv._benchmark_curves(flat, Eligibility(flat), 2, 6)
    @test bmf.equity_cash ≈ ones(5)                        # cash 恒 1（D-072 #1）
    @test bmf.equity_ew_daily ≈ ones(5)                    # gross≡1 → 不动
    @test bmf.equity_ew_buyhold ≈ ones(5)
    @test bmf.ew_rebalance_gap ≈ zeros(5)                  # 无漂移 → 重合
    # (6b) 确定性漂移：手工精确值（构造性分离断言）
    #   价格：asset1 = [1,1,2,2,…]、asset2 = [1,1,1,2,…]
    #   t_start=2、t_end=4：gross1 = [2,1]、gross2 = [1,2]
    #   EW daily：  wealth = 1.5、1.5 → [1, 1.5, 2.25]
    #   buy-hold：  漂移 w = [2/3,1/3] → wealth2 = 4/3 → [1, 1.5, 2.0]
    px = ones(T_f, N_f)
    px[3, 1] = 2.0; px[4: end, 1] .= 2.0
    px[4, 2] = 2.0; px[5: end, 2] .= 2.0
    drift_mf = MarketFacts(collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T_f - 1))),
                           ["A1", "A2"], px, copy(px), trues(T_f, N_f))
    bmd = Gate0BacktestEnv._benchmark_curves(drift_mf, Eligibility(drift_mf), 2, 4)
    @test bmd.equity_cash ≈ [1.0, 1.0, 1.0]                # cash 与市场无关
    @test bmd.equity_ew_daily ≈ [1.0, 1.5, 2.25]           # 每日再平衡（D-072 #2）
    @test bmd.equity_ew_buyhold ≈ [1.0, 1.5, 2.0]          # 首日等权持有漂移（H8）
    @test bmd.ew_rebalance_gap ≈ [0.0, 0.0, 0.25]          # D-073：只报告差
    # (6c) benchmark 候选外生（D-020）：未 trade_eligible 资产不进 EW 候选
    trade_bm = trues(T_f, N_f)
    trade_bm[:, 1] .= false                                # asset 1 不可交易
    el_bm = Eligibility(drift_mf, trade_bm)
    bm_x = Gate0BacktestEnv._benchmark_curves(drift_mf, el_bm, 2, 4)
    @test bm_x.equity_ew_daily ≈ [1.0, 1.0, 2.0]           # 只有 asset 2（等权=单资产）
    @test bm_x.equity_ew_buyhold ≈ [1.0, 1.0, 2.0]

    # ===================================================================
    end


end

if "7" in _BT_SCENARIOS
@testset "backtest (7) fail loudly" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：无信号随机游走 fixture
    # 后验宽平，2D 求积在预算内无法达 1e-6（rel 停在 ~0.0005）；1e-2 让求积
    # 提前停止（~420 cells），multi-day 循环可行。生产默认保持 1e-6（P0-7
    # 严格口径）。此降级仅限测试 fixture，不改变生产数学合同。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
    mf = _mk_mf(N = 2, T = 280, seed = 31, drift = 0.003)
    el = Eligibility(mf)
# (7) fail loudly
    # ===================================================================
    @test_throws ArgumentError run_gate0_backtest(mf, el; t_start = 272,
                                                  t_end = 272, KW...)  # 空区间
    @test_throws ArgumentError run_gate0_backtest(mf, el; t_start = 273,
                                                  t_end = 272, KW...)  # 倒区间
    @test_throws ArgumentError run_gate0_backtest(mf, el; t_start = 270,
                                                  t_end = 281, KW...)  # t_end > T
    @test_throws ArgumentError run_gate0_backtest(mf, el; t_start = 100,
                                                  t_end = 105, KW...)  # WARMUP 前置
    @test_throws DimensionMismatch run_gate0_backtest(
        mf, el; t_start = 270, t_end = 272, held0 = zeros(1), KW...)
    @test_throws ArgumentError run_gate0_backtest(
        mf, el; t_start = 270, t_end = 272, held0 = [0.6, 0.6], KW...)  # Σ>1
end

end


if "8" in _BT_SCENARIOS
@testset "backtest (8) T4 recovery (prequential)" begin
    SEED_DEC = 0x0000000000005678
    # 短窗口受控参数（数值配置语义——D-066；tiny 级成本控制）
    # fixture 级降级 tolerance（posterior_tol=1e-2）：与 testset (1)-(7) 同款
    # 受控降级（无信号/小样本 fixture 的 2D 求积在预算内无法达 1e-6）；
    # 生产默认保持 1e-6（P0-7 严格口径）。此降级仅限测试 fixture。
    KW = (mode = :reference, S_reference = 32, diag_S = 16,
          posterior_tol = 1e-2)
    # (8) T4 触发→恢复正向路径（production prequential 语义；裁决 9）
    # =================================================================== 
    # 语义声明（裁决 9 / docs/T4_MULTIDAY_SEMANTICS_ADJUDICATION9.md）：
    # 本测试验证 **production（strict prequential）** 残差源下的
    # 「T4 触发 → 次日恢复」多日正向路径。fold（legacy/reference）语义
    # 下该构造不可行（fold train 的 rank 需每 fold 覆盖行——J 与 fold
    # rank 保障行同源互斥，见 adjudication8_feasibility.md）；本测试
    # **不声称 fold 可行性**，也不改动 fold 路径的任何语义。
    #
    # 构造（确定性；rank 不赌随机一般位置）：
    # - 资产 1：全观测（平缓漂移 + 固定 seed 小噪声——macro/DC 信号）；
    # - 资产 2：尾部新资产、locked（trade_eligible=false、held=0.3）；
    #   仅行 268..272 观测；log 价格窗口内交替 ±0.05（268→269 +、
    #   269→270 −、270→271 +、271→272 −）。交替冲击使恢复日两条残差
    #   （s=269 目标 270 为 −；s=270 目标 271 为 +）的资产 2 分量异号
    #   ⇒ 二次 resolve 的 rank(2×2)=2 由构造保证。
    # - 决策日 270：首次 resolve J={269,270}=2 通过；prequential 仅
    #   s=269 可定义（其 train 含覆盖行 269）→ keep_res={269}；二次
    #   resolve J={270} 单行 → free 剔尽仍不足（覆盖缺口由 locked
    #   引起）→ held_maintained（裁决 C3 转译）。
    # - 决策日 271：keep_res={269,270} → 二次 resolve J={270,271}
    #   两行、rank=2 → 恢复（正常决策，free 未剔除）。
    # =================================================================== 
    T8, N8 = 273, 2
    rng8 = MersenneTwister(20261010)
    x1 = 0.003 .* collect(1:T8) .+ 0.01 .* randn(rng8, T8)
    x2 = fill(NaN, T8)
    x2[268] = 0.0
    x2[269] = 0.05
    x2[270] = 0.0
    x2[271] = 0.05
    x2[272] = 0.0
    close8 = fill(NaN, T8, N8)
    adj8 = fill(NaN, T8, N8)
    close8[:, 1] = exp.(x1); adj8[:, 1] = exp.(x1)
    for t in 268:272
        close8[t, 2] = exp(x2[t]); adj8[t, 2] = exp(x2[t])
    end
    observed8 = falses(T8, N8)
    observed8[:, 1] .= true
    observed8[268:272, 2] .= true
    dates8 = collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T8 - 1)))
    mf8 = MarketFacts(dates8, ["A1", "A2"], close8, adj8, observed8)
    trade8 = trues(T8, N8)
    trade8[:, 2] .= false          # 资产 2 不可交易 → 当日 free=0 → locked
    el8 = Eligibility(mf8, trade8)
    held8 = [0.1, 0.3]
    res8 = run_gate0_backtest(mf8, el8; t_start = 270, t_end = 272,
                              held0 = held8, seed = SEED_DEC, KW...)

    # --- 触发日 t=270（二次 resolve：单行 J → free 剔尽仍不足 → T4） ---
    @test length(res8.days) == 2
    @test res8.days[1].t == 270
    @test res8.days[1].decision.mode == :held_maintained
    @test res8.days[1].decision.diagnostics.reason == :innovation_coverage_locked
    @test occursin("innovation coverage failure",
                   res8.days[1].decision.diagnostics.error_text)
    @test res8.days[1].decision.w_universe == held8
    @test res8.days[1].held_before == held8
    @test isfinite(res8.days[1].wealth_factor) && res8.days[1].wealth_factor > 0

    # --- 恢复日 t=271（正常决策；free 未剔除 → R_universe 含相关资产） ---
    @test res8.days[2].t == 271
    @test res8.days[2].decision.mode == :reference
    @test res8.days[2].decision.R_universe == [1, 2]
    @test res8.days[2].decision.locked_exposure ≈ res8.days[2].held_before[2]
    @test sum(res8.days[2].decision.w_universe) + res8.days[2].decision.w_cash ≈ 1.0 atol = 1e-9
    # 触发日维持 → 漂移 → 恢复日 held（不重置、不清仓——裁决 C3 多日延续）
    g8 = [mf8.adj[271, j] / mf8.adj[270, j] for j in 1:2]
    wf8 = res8.days[1].wealth_factor
    @test res8.days[2].held_before ≈
          [res8.days[1].decision.w_universe[j] * g8[j] / wf8 for j in 1:2]

    # --- 全程无 error、净值良定 ---
    @test all(x -> isfinite(x) && x > 0, res8.equity)

    # =================================================================== 
    end


end
