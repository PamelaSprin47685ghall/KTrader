# test/gate0/runtests.jl —— Gate-0 兼容层与组装层验收
#
# 正文和测试都在 KTraderNew；本仓库只依赖两个组装层（经私有 registry）。这里验两件事：
#   1. gate0/KTraderGate0.jl 把旧名字（含 bin/ 与 dev/ 用到的内部名）接回来了；
#   2. 本仓库 Manifest 解析出的组装层版本，在 KTraderNew 里对应的那次提交上 run.jl 全过
#      （各库单测 + 组装层跨库测试）。不用 Pkg.test：组装层测试环境用相对路径指兄弟库，
#      装进 depot 后找不到。提交按 Manifest 里组装层的 git-tree-sha1 在 KTraderNew 历史里定位。
#
# 运行：JULIA_PKG_USE_CLI_GIT=true julia --startup-file=no --project=. test/gate0/runtests.jl
# ENV：GATE0_LIBS —— 传给 run.jl 的库名（逗号分隔），默认 "all"；"none" 只跑第 1 项。
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
    @test G.single_day_decision === G.Assembly.single_day_decision
    @test G._ForecastCache === G.Assembly._ForecastCache
    @test G.MarketFacts === G.KMarket.MarketFacts
    @test G.cash_kelly === G.KKelly.cash_kelly
    # 旧 export 面里的公开名字直接可用。
    @test first(cash_kelly([1.1 0.9; 0.95 1.05])) isa AbstractVector
end

const _SPEC = get(ENV, "GATE0_LIBS", "all")
if _SPEC != "none"
    root = joinpath(@__DIR__, "..", "..")
    man = TOML.parsefile(joinpath(root, "Manifest.toml"))["deps"]
    tree = only(man["KTraderGate0"])["git-tree-sha1"]
    dir = mktempdir()
    run(`git clone --quiet https://github.com/PamelaSprin47685ghall/KTraderNew $dir`)
    rev = something(findfirst(c -> readchomp(`git -C $dir rev-parse $c:apps/KTraderGate0`) == tree,
                              split(readchomp(`git -C $dir rev-list origin/master`))), 0)
    revs = split(readchomp(`git -C $dir rev-list origin/master`))
    @test rev != 0
    if rev != 0
        run(`git -C $dir checkout --quiet $(revs[rev])`)
        args = _SPEC == "all" ? String[] : split(_SPEC, ',')
        @test success(pipeline(`$(Base.julia_cmd()) --startup-file=no $(joinpath(dir, "run.jl")) $args`; stdout, stderr))
    end
end
