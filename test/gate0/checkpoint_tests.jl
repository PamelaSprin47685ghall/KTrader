using Test, Dates, Random, Serialization

module CheckpointEnv
    using Dates
    include(joinpath(@__DIR__, "../../src/gate0/market.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/response.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/driver.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/backtest.jl"))
end
using .CheckpointEnv

# Cash-only is an explicit accounting/checkpoint fixture, NOT a performance
# benchmark or evidence that the real 65-asset two-year model has completed.
@testset "checkpoint resumes the same sequential account" begin
    rng = MersenneTwister(72)
    T = 270
    p = exp.(cumsum(0.01 .* randn(rng, T, 1); dims=1))
    dates = [Date(2020, 1, 1)+Day(i-1) for i in 1:T]
    mf = MarketFacts(dates, ["TEST"], p, p, trues(T, 1))
    el = Eligibility(mf, falses(T, 1))
    plain = run_gate0_backtest(mf, el; t_start=260, t_end=263, reuse_forecasts=false)
    mktempdir() do dir
        path = joinpath(dir, "run.jls")
        @test_throws CheckpointEnv.BacktestPaused run_gate0_backtest(mf, el;
            t_start=260, t_end=263, checkpoint=path, max_new_days=1)
        saved = deserialize(path)
        @test saved.next_t == 261
        @test length(saved.records) == 1
        @test length(saved.equity_curve) == 2
        resumed = run_gate0_backtest(mf, el; t_start=260, t_end=263,
                                     checkpoint=path, resume=true)
        @test resumed.equity == plain.equity
        @test [d.t for d in resumed.days] == [260, 261, 262]
        @test resumed.equity_ew_daily == plain.equity_ew_daily
        @test deserialize(path).next_t == 263
        done = run_gate0_backtest(mf, el; t_start=260, t_end=263,
                                  checkpoint=path, resume=true)
        @test done.equity == resumed.equity
        @test length(done.days) == 3
        @test_throws ArgumentError run_gate0_backtest(mf, el;
            t_start=260, t_end=263, checkpoint=path)
        @test_throws ErrorException run_gate0_backtest(mf, el;
            t_start=260, t_end=263, checkpoint=path, resume=true, posterior_tol=2e-6)
        changed = copy(p); changed[end, 1] *= 2
        mf2 = MarketFacts(dates, ["TEST"], changed, changed, trues(T, 1))
        @test_throws ErrorException run_gate0_backtest(mf2, Eligibility(mf2, falses(T, 1));
            t_start=260, t_end=263, checkpoint=path, resume=true)
    end
    @test_throws ArgumentError run_gate0_backtest(mf, el;
        t_start=260, t_end=263, max_new_days=1)
end
