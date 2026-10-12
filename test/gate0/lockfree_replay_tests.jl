using Test,Dates,Random,LinearAlgebra
module LockFreeReplayEnv
    using Dates,LinearAlgebra,Random
    include(joinpath(@__DIR__,"../../src/gate0/market.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/quadrature.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/driver.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/backtest.jl"))
end
const G=LockFreeReplayEnv

@testset "independent full decisions replay the original account and benchmarks" begin
    rng=MersenneTwister(19)
    T=290
    p=exp.(cumsum(0.008 .* randn(rng,T,2);dims=1))
    dates=[Date(2020,1,1)+Day(t-1) for t in 1:T]
    mf=G.MarketFacts(dates,["A","B"],p,p,trues(T,2))
    el=G.Eligibility(mf)
    t_start,t_end=270,276
    G._assert_lockfree_independent_window(mf,el,t_start,t_end,zeros(2))
    v=G._no_varsplit()
    conc=G._no_concentration(2)
    c=(;A=0.0,B=0.0,B_audit=0.0,
        C=(;feasibility=0.0,kkt_residual=0.0,objective_gap=0.0,
            objective=0.0,dual=1.0),
        converged=true,kelly_objective=0.0)
    decs=G.Gate0DayDecision[]
    for t in t_start:t_end-1
        risky=[t%2==0 ? 0.4 : 0.3,0.45]
        cash=1-sum(risky)
        push!(decs,G.Gate0DayDecision(:adaptive,t,[1,2],[1,2],
              copy(risky),copy(risky),cash,0.0,1.0,c,1024,
              v,conc,Int[],
              (;reason=:computed,error_text="",dropped_free_universe=Int[])))
    end
    res=G.replay_independent_gate0(mf,el,decs;t_start,t_end)
    @test length(res.days)==6 && length(res.equity)==7
    held=zeros(2)
    expected_eq=1.0
    for i in eachindex(decs)
        t=t_start+i-1
        dec=decs[i]
        gross=G._marking_gross(mf,t,dec.w_universe)
        wealth=dot(dec.w_universe,gross)+dec.w_cash
        expected_eq*=wealth
        @test res.days[i].held_before == held
        @test res.days[i].gross_realized == gross
        @test res.days[i].wealth_factor == wealth
        @test res.equity[i+1] == expected_eq
        held=[dec.w_universe[j]*gross[j]/wealth for j in 1:2]
    end
    b=G._benchmark_curves(mf,el,t_start,t_end)
    @test res.equity_cash==b.equity_cash
    @test res.equity_ew_daily==b.equity_ew_daily
    @test res.equity_ew_buyhold==b.equity_ew_buyhold
    @test res.ew_rebalance_gap==b.ew_rebalance_gap

    @test_throws DimensionMismatch G.replay_independent_gate0(mf,el,decs[1:5];t_start,t_end)
    bad=G.Eligibility(el.model_admitted,el.trade_eligible,
       BitMatrix([i==t_start && j==1 ? false : el.executable[i,j]
                  for i in 1:T,j in 1:2]))
    @test_throws ErrorException G.replay_independent_gate0(mf,bad,decs;t_start,t_end)
    @test_throws ErrorException G._assert_lockfree_independent_window(
           mf,bad,t_start,t_end,[0.5,0.0])

    old=decs[1]
    invalid=G.Gate0DayDecision(:held_maintained,old.t,old.R_t,old.R_universe,
          old.w_universe,old.w_risky,old.w_cash,old.locked_exposure,old.budget,
          old.certificates,old.M,old.variance_split,old.concentration,
          old.dropped_free,old.diagnostics)
    @test_throws ErrorException G.replay_independent_gate0(mf,el,
          vcat([invalid],decs[2:end]);t_start,t_end)
end
