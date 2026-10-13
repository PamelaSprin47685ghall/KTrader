# test/gate0/runtests.jl —— Gate-0 兼容层验收
#
# Gate-0 正文和它的 25 个测试文件已经搬到 KTraderNew 各库；run.jl 同时跑 v1 线抽出的库（KBars 等）（版本钉在根
# Project.toml 的 [sources]）。这里只验两件事：
#   1. gate0/KTraderGate0.jl 把旧名字（含 bin/ 与 dev/ 用到的内部名）拼回来了；
#   2. 钉住版本的各库自己的测试照样通过：按 [sources] 里的 url+rev 取一份
#      KTraderNew，用它自带的 run.jl 跑（各库测试环境用相对路径互指，
#      装进 depot 后 Pkg.test 找不到兄弟库，所以不走 Pkg.test）。
#
# 运行：julia --startup-file=no --project=. test/gate0/runtests.jl
# ENV：GATE0_LIBS —— 逗号分隔的库名，默认 "all"；"none" 只跑第 1 项。
using Test, TOML

include(joinpath(@__DIR__, "..", "..", "gate0", "KTraderGate0.jl"))
using .KTraderGate0
const G = KTraderGate0

# bin/ 与 dev/ 里以 G.xxx / KTraderGate0.xxx 访问的名字（rg 扫出来的全集）。
const OLD_CALLERS_NEED = (
    :_default_mu_seed, :_write_backtest_checkpoint, :_backtest_signature,
    :_DRIVER_DEFAULT_SEED, :Eligibility, :load_market_facts, :SobolOwenRule,
    :single_day_decision, :cash_kelly_certified, :audit_seed, :_historical_forecast,
    :_assert_lockfree_independent_window, :_ForecastCache, :WARMUP,
    :resolve_risk_domain, :innovation_state, :cash_kelly_certificate,
    :_lockfree_shard_days, :_lockfree_required_origins, :_cash_kelly_solve,
    :Gate0DayDecision, :BacktestPaused, :rule_points, :replay_independent_gate0,
    :predictive_rqmc_source, :joint_row_indices, :free, :fit_full_posterior,
    :_response_prefix, :_mu_node_rule_seed, :_mu_node_cdf,
    :_market_prequential_residuals, :_forecast_risk_floor, :_decision_forecast,
    :MarketFacts, :run_gate0_backtest, :cash_kelly, :initial_mesh)

@testset "KTraderGate0 compat layer re-exports the libs" begin
    for n in OLD_CALLERS_NEED
        @test isdefined(G, n)
    end
    # 同一绑定：兼容层不另造对象。
    @test G.single_day_decision === G.KDriver.single_day_decision
    @test G._ForecastCache === G.KBacktest._ForecastCache
    @test G.MarketFacts === G.KMarket.MarketFacts
    @test G.cash_kelly === G.KKelly.cash_kelly
    # 旧 export 面里的公开名字直接可用。
    @test first(cash_kelly([1.1 0.9; 0.95 1.05])) isa AbstractVector
end

const _SPEC = get(ENV, "GATE0_LIBS", "all")
const _LIBS = _SPEC == "all" ? String[] : _SPEC == "none" ? nothing : split(_SPEC, ',')
if _LIBS !== nothing
    src = TOML.parsefile(joinpath(@__DIR__, "..", "..", "Project.toml"))["sources"]["KMarket"]
    dir = mktempdir()
    run(`git clone --quiet $(src["url"]) $dir`)
    run(`git -C $dir checkout --quiet $(src["rev"])`)
    @test success(pipeline(`$(Base.julia_cmd()) --startup-file=no $(joinpath(dir, "run.jl")) $_LIBS`;
                           stdout, stderr))
end
