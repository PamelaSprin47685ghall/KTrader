# ============================================================================
# test/gate0/market_tests.jl —— Gate-0 Step 1 tiny analytic 测试
# ============================================================================
#
# 覆盖（施工图 Step 1 测试段 + §9.4 ragged 1-3 + 任务书测试清单）：
#   D-012 observed 语义与 signal 恢复；mask 分离与只读防御；
#   D-013 model_admitted（dummy / IPO / 单日 missing / prefix 单调）；
#   D-018/H4 252 规则边界（O 计数口径）；D-016 free；D-017 locked；
#   D-020 benchmark 外生；permutation 协变；数据桥 from_bars_arrays；
#   全部非法输入 fail loudly（NaN/Inf 价格、维度不匹配、空输入）。
#
# 独立入口（裁决 H1）：本文件不 include 旧线 src/、不 using 旧 KTrader、
# 不读真实 panel、无随机数（Step 1 的全部判定是确定性的）。
# 运行（归 DevOps，Engineer 不执行）：
#   julia --startup-file=no test/gate0/market_tests.jl
#
# 注：含 NaN 的 Float64 矩阵比较用 isequal（== 对 NaN 判假）。

using Test, Dates

include(joinpath(@__DIR__, "..", "..", "src", "gate0", "KTraderGate0.jl"))
using .KTraderGate0

# ---------------------------------------------------------------------------
# 合成 fixture（纯确定性，无 Random）
# ---------------------------------------------------------------------------
#
# base_arrays（T=8、N=3）：
#   AAA 全程观测；BBB 行 1-4 上市前（价格 NaN），行 5 起 IPO 观测；
#   CCC 全程观测但行 6 缺 bar（marking carry forward，observed=false）。
function base_arrays()
    T, N = 8, 3
    dates = [Date(2026, 1, 1) + Day(t - 1) for t in 1:T]
    symbols = ["AAA", "BBB", "CCC"]
    close = fill(NaN, T, N)
    adj = fill(NaN, T, N)
    observed = falses(T, N)
    for t in 1:T
        observed[t, 1] = true; close[t, 1] = 100.0 + t; adj[t, 1] = 100.0 + t
        observed[t, 3] = true; close[t, 3] = 30.0 + t; adj[t, 3] = 30.0 + t
    end
    for t in 5:T
        observed[t, 2] = true; close[t, 2] = 50.0 + t; adj[t, 2] = 50.0 + t
    end
    observed[6, 3] = false       # 单日缺 bar（ragged 3）
    close[6, 3] = close[5, 3]    # marking carry forward（合法：非观测处有限值）
    adj[6, 3] = adj[5, 3]
    return dates, symbols, close, adj, observed
end

base_panel() = MarketFacts(base_arrays()...)

# dummy_panel：base + 全 NaN dummy 资产 DDD（无任何观测，价格全 NaN）。
function dummy_panel()
    d, s, c, a, o = base_arrays()
    return MarketFacts(d, [s; "DDD"], hcat(c, fill(NaN, 8)),
                       hcat(a, fill(NaN, 8)), hcat(o, falses(8)))
end

# panel_252（T=260、N=2）：X252 全程观测（第 t 行累计 t 个 bar）；
# Y252 行 10 缺 bar（第 t 行累计 t-1 个 bar——第 252 个有效 bar 落在行 253）。
function panel_252()
    T, N = 260, 2
    dates = [Date(2026, 1, 1) + Day(t - 1) for t in 1:T]
    symbols = ["X252", "Y252"]
    close = fill(NaN, T, N)
    adj = fill(NaN, T, N)
    observed = trues(T, N)
    for t in 1:T
        close[t, 1] = 10.0 + t; adj[t, 1] = 10.0 + t
        close[t, 2] = 20.0 + t; adj[t, 2] = 20.0 + t
    end
    observed[10, 2] = false
    close[10, 2] = close[9, 2]   # marking carry forward
    adj[10, 2] = adj[9, 2]
    return MarketFacts(dates, symbols, close, adj, observed)
end

# panel_with_w：base + W 资产（只在最后一行观测——首日上市，无相邻观测对
# ⟹ model_admitted 全 false；当日 observed=true ⟹ exec=true；默认 trade
# 政策全 true）。W 是 D-020 benchmark 外生性的正例资产。
function panel_with_w()
    d, s, c, a, o = base_arrays()
    T = 8
    cw = fill(NaN, T); aw = fill(NaN, T); ow = falses(T)
    ow[T] = true; cw[T] = 7.0; aw[T] = 7.0
    return MarketFacts(d, [s; "W"], hcat(c, cw), hcat(a, aw), hcat(o, ow))
end

# ---------------------------------------------------------------------------
@testset "gate0 Step 1: market facts / eligibility (D-012~D-020)" begin

    @testset "构造校验与 fail-loudly" begin
        mf = base_panel()
        @test mf isa MarketFacts
        @test mf.observed isa BitMatrix

        d, s, c, a, o = base_arrays()
        a_bad = copy(a); a_bad[3, 1] = NaN          # observed=true 处 adj NaN
        @test_throws ErrorException MarketFacts(d, s, c, a_bad, o)
        c_bad = copy(c); c_bad[3, 1] = Inf          # observed=true 处 close Inf
        @test_throws ErrorException MarketFacts(d, s, c_bad, a, o)
        @test_throws ErrorException MarketFacts(d[1:7], s, c, a, o)      # dates 长度不符
        @test_throws ErrorException MarketFacts(d, s, c[1:7, :], a, o)   # close 行数不符
        @test_throws ErrorException MarketFacts(d, s, c, a, o[:, 1:2])   # observed 列数不符
        @test_throws ErrorException MarketFacts(
            Date[], String[], zeros(0, 0), zeros(0, 0), falses(0, 0))    # 空面板
        @test_throws ErrorException Eligibility(falses(8, 3), falses(8, 3), falses(7, 3))
        @test_throws ErrorException Eligibility(falses(8, 3), falses(7, 3), falses(8, 3))
        @test_throws ErrorException Eligibility(falses(0, 0), falses(0, 0), falses(0, 0))
        @test_throws ErrorException Eligibility(mf, falses(7, 3))        # 政策 mask 尺寸不符

        # observed=false 处：NaN（上市前）与有限（carry）均合法——base_panel 构造成功即证
        @test isnan(mf.adj[1, 2]) && !mf.observed[1, 2]
        @test isfinite(mf.adj[6, 3]) && !mf.observed[6, 3]
    end

    @testset "D-012: observed 语义与 signal 恢复" begin
        mf = base_panel()
        sig = signal_prices(mf)
        @test sig isa Matrix{Float64}
        @test isequal(sig, ifelse.(mf.observed, mf.adj, NaN))
        @test sig[1, 1] == mf.adj[1, 1]      # 观测处 = adj
        @test all(isnan, sig[1:4, 2])        # 上市前（BBB 行 1-4）= NaN
        @test isnan(sig[6, 3])               # 缺 bar 处 = NaN……
        @test isfinite(mf.adj[6, 3])         # ……尽管 marking carry 是有限值
    end

    @testset "mask 分离与只读防御（D-012 四禁的类型层落地）" begin
        mf = base_panel()

        # 表示钉死
        @test fieldtype(MarketFacts, :observed) == BitMatrix
        @test fieldtype(Eligibility, :model_admitted) == BitMatrix
        @test fieldtype(Eligibility, :trade_eligible) == BitMatrix
        @test fieldtype(Eligibility, :executable) == BitMatrix

        # 导出面钉死：无任何 mutation / setter API（构造函数是唯一入口）。
        # Step 16 短窗口轮收口后的完整导出面（86 名 API = market 11 +
        # geometry 11 + modes 6 + response 4 + posterior 14 + oof 5 +
        # innovation 16 + predictive 2 + quadrature 7 + driver 3 +
        # backtest 3 + kelly 4；除 market/kelly 外均由文件内自带
        # export——见 src/gate0/KTraderGate0.jl 注释）。
        # 注：Julia 的 names(M) 恒包含 module 自名 :KTraderGate0，故期望
        # 集合共 87 名。
        # 注：backtest 的 _DRIVER_DEFAULT_SEED/_marking_gross/
        # _benchmark_curves 为内部函数/常量（driver/backtest 链内可见），
        # 不导出。
        @test Set(names(KTraderGate0)) == Set([
            :KTraderGate0,
            :MarketFacts, :Eligibility, :from_bars_arrays, :signal_prices,
            :model_admitted, :trade_eligible, :effective_bars_252,
            :executable, :free, :locked, :benchmark_universe,
            :TAUS, :BANDS, :BANDCOL, :WARMUP, :center_of_mass, :ruler,
            :relative_gauge, :principal_sqrt_root, :compute_s_perp,
            :path_basis_1d, :fast_s_m,
            :domain_mode_basis, :risk_domain_mode_basis, :mode_field,
            :asset_field_reconstruct, :cumulative_path_coordinates,
            :mode_basis_design,
            :build_mode_problem, :fit_full_block_ridge, :predict_mode,
            :block_views,
            :SufficientStats, :ConditionalFit, :ProprietyCertificate,
            :ResponsePosterior, :sufficient_stats, :s_alpha, :log_evidence,
            :log_prior_d035a, :adaptive_quadrature_2d, :fit_full_posterior,
            :predict_mu, :draw_mu, :mu_asset, :oof_residual_rows,
            :fold_grid, :fold_stats, :train_stats, :fit_fold_posteriors,
            :oof_residual_rows_full,
            :InnovationState, :innovation_state, :SupportCertificate,
            :DConditionedShapePool, :draw_innovation, :z_pool_rows,
            :z_row_at, :V_at, :shape_moments, :quasi_loglik,
            :joint_row_indices, :coverage_report, :resolve_risk_domain,
            :default_d_grid, :frac_weights_gate0, :mp_sqrt_factors,
            :PredictiveLawResult, :predictive_law,
            :SobolOwenRule, :rule_points, :audit_seed,
            :RQMCScenarioSource, :predictive_rqmc_source,
            :AdaptiveKellyResult, :adaptive_scenario_kelly,
            :Gate0DayDecision, :single_day_decision,
            :equal_weight_benchmark,
            :Gate0DayRecord, :Gate0BacktestResult, :run_gate0_backtest,
            :cash_kelly, :locked_wealth_gate0, :cash_kelly_certificate,
            :cash_kelly_certified])

        # aliasing 防御：构造后修改调用方自己的数组，不影响 MarketFacts
        d, s, c, a, o = base_arrays()
        mf2 = MarketFacts(d, s, c, a, o)
        o[1, 1] = !o[1, 1]
        a[1, 1] = NaN
        @test mf2.observed[1, 1] == true
        @test mf2.adj[1, 1] == 101.0

        # 政策 mask 与 observed 分离：改 Eligibility 不影响 MarketFacts
        el = Eligibility(mf)                 # 默认政策
        el.executable[1, 1] = false          # 政策层持有自己的副本
        @test mf.observed[1, 1] == true      # 市场事实层不受影响
        el.model_admitted[2, 1] = false
        @test model_admitted(mf)[2, 1] == true   # 判定函数重算不受政策层污染
    end

    @testset "D-013: model_admitted 判定" begin
        mf = base_panel()
        adm = model_admitted(mf)
        @test adm isa BitMatrix

        # AAA：首个有效 return 属于行 2（相邻对 (1,2)）——行 1 不 admitted
        @test adm[1, 1] == false
        @test adm[2, 1] == true && adm[8, 1] == true

        # BBB（行 5 IPO，ragged 2）：首个相邻对 (5,6) ⟹ 首个 admitted 日是行 6
        @test !any(adm[1:5, 2])
        @test all(adm[6:8, 2])

        # CCC（行 6 缺 bar，ragged 3）：行 2 起 admitted，单日 missing 不回退
        @test adm[1, 3] == false && adm[2, 3] == true
        @test all(adm[2:8, 3])

        # prefix 单调不减
        for j in 1:3
            col = collect(adm[:, j])
            @test col == sort(col)
        end

        # 全 NaN dummy 严格排除（ragged 1）且不影响其余资产
        mfd = dummy_panel()
        adm_d = model_admitted(mfd)
        @test !any(adm_d[:, 4])
        @test adm_d[:, 1:3] == adm
    end

    @testset "D-018/H4: 252 有效 bar 规则边界" begin
        # 默认政策：无限制，全 true（对照）
        @test all(trade_eligible(base_panel()))

        mfp = panel_252()
        e252 = effective_bars_252(mfp)

        # X252 全程观测：第 251 个有效 bar 当日（行 251）仍受限，
        # 累计第 252 个有效 bar 的当日（行 252）解除——H4 口径
        @test e252[251, 1] == false
        @test e252[252, 1] == true
        @test !any(e252[1:251, 1])
        @test all(e252[252:260, 1])

        # Y252 行 10 缺 bar：计数按 O 累计（观测 bar 数，非日历行数）——
        # 行 252 累计 251 个 bar（仍受限），第 252 个有效 bar 落在行 253 当日解除
        @test e252[252, 2] == false
        @test e252[253, 2] == true
        @test !any(e252[1:252, 2])
        @test all(e252[254:260, 2])

        # 252 规则不触碰 observed（D-012：模型仍可看见真实价格，只是不许开仓）
        @test mfp.observed[100, 1] == true && e252[100, 1] == false
    end

    @testset "D-016/D-017: free 派生与 locked" begin
        mf = base_panel()

        # free = trade ∧ exec（默认政策下退化为 exec）
        el = Eligibility(mf)
        @test free(el) == mf.observed
        @test collect(free(el, 6)) == collect(mf.observed[6, :])

        # trade=false 压过 exec=true（合取语义）
        tm = trues(8, 3); tm[8, 1] = false
        el2 = Eligibility(mf, tm)
        @test free(el2)[8, 1] == false
        @test free(el2)[8, 2] == true     # observed[8,2]=true 且 trade=true
        @test free(el2)[7, 1] == true     # 行 7 资产 1 未受行 8 政策影响

        # admission 不混入 free（D-016 核心）：W 从未 admitted 但当日 free
        mfw = panel_with_w()
        elw = Eligibility(mfw)
        @test !any(model_admitted(mfw)[:, 4])
        @test free(elw)[8, 4] == true

        # locked = held ∧ ¬free（D-017）
        held = [true, false, false]
        @test collect(locked(held, el2, 8)) == [true, false, false]    # 资产 1：held ∧ ¬free
        @test collect(locked(held, el2, 7)) == [false, false, false]   # 行 7 资产 1 free ⟹ 非 locked
        @test collect(locked([true, false, false, false], elw, 8)) ==
            [false, false, false, false]   # W 当日 free ⟹ held 也不 locked

        # held 长度校验 fail loudly
        @test_throws ErrorException locked([true, false], el2, 8)

        # D-017 语义注记：locked 是 locked risk 而非 cash——其 scenario
        # wealth 贡献由 Step 2 的 Kelly base_s 保留（本步钉判定语义，
        # 财富语义归 Step 2 的 Case D 测试）。
    end

    @testset "D-020: benchmark universe 外生" begin
        mfw = panel_with_w()
        elw = Eligibility(mfw)

        # 前提复核：W 从未 admitted
        @test !any(model_admitted(mfw)[:, 4])

        # 候选集 = trade ∧ exec，与 admission 无关：W（admitted=false）进候选集
        bu = benchmark_universe(elw)
        @test bu[8, 4] == true
        @test bu == free(elw)                                  # 与 free 数学同式（一致性检查）
        @test collect(benchmark_universe(elw, 8)) == collect(free(elw, 8))

        # 签名外生性：不接受 MarketFacts（更不可能接受 model/active/posterior）
        @test_throws MethodError benchmark_universe(mfw)
    end

    @testset "permutation 协变" begin
        mf = base_panel()
        perm = [3, 1, 2]
        mfp = MarketFacts(mf.dates, mf.symbols[perm], mf.close[:, perm],
                          mf.adj[:, perm], mf.observed[:, perm])

        @test model_admitted(mfp) == model_admitted(mf)[:, perm]
        @test executable(mfp) == executable(mf)[:, perm]
        @test trade_eligible(mfp) == trade_eligible(mf)       # 全真：置换不变
        @test isequal(signal_prices(mfp), signal_prices(mf)[:, perm])

        el = Eligibility(mf)
        elp = Eligibility(mfp)
        @test elp.model_admitted == el.model_admitted[:, perm]
        @test elp.executable == el.executable[:, perm]
        @test elp.trade_eligible == el.trade_eligible

        # 252 面板上的置换（每列独立计数 ⟹ 置换协变）
        m252 = panel_252()
        p2 = [2, 1]
        m252p = MarketFacts(m252.dates, m252.symbols[p2], m252.close[:, p2],
                            m252.adj[:, p2], m252.observed[:, p2])
        @test effective_bars_252(m252p) == effective_bars_252(m252)[:, p2]
    end

    @testset "数据桥 from_bars_arrays（H1/H7）" begin
        d, s, c, a, o = base_arrays()
        mf = MarketFacts(d, s, c, a, o)
        mfb = from_bars_arrays(d, s, c, a, o)

        # 桥 ≡ 主构造（observed ≡ 传入的 bar 数组，逐元素一致）
        @test mfb.observed == o
        @test isequal(mfb.close, c)     # 含 NaN：用 isequal
        @test isequal(mfb.adj, a)
        @test mfb.dates == d && mfb.symbols == s
        @test mfb.observed == mf.observed && isequal(mfb.adj, mf.adj)

        # 桥的校验路径与主构造一致（fail loudly）
        a_bad = copy(a); a_bad[2, 1] = NaN
        @test_throws ErrorException from_bars_arrays(d, s, c, a_bad, o)

        # 桥路径的 aliasing 防御：改传入数组不污染桥对象
        o[2, 1] = !o[2, 1]
        @test mfb.observed[2, 1] == true
    end
end
