using KTrader, Test, Random, LinearAlgebra

@testset "Backtest numerical boundary: same fixed scenarios and locked risk" begin
    X=exp.(0.02randn(MersenneTwister(1009),64,5))
    active=[1,3,5]; X[:,[2,4]].=NaN
    free=Bool[true,false,false,false,true]
    for held in (nothing,[0.3,0.0,0.4,0.0,0.3])
        timing=KTrader.DecisionTiming()
        decision=(;X,model=nothing,active,free,timing)
        original=copy(X); original_held=held===nothing ? nothing : copy(held)
        expected=KTrader.scenario_weights(X,active,free,held)
        weights,S=KTrader._backtest_target(decision,held,false,91,64,1e-5,512)
        @test weights==expected
        @test S==64
        @test isequal(X,original)
        @test isequal(held,original_held)
        @test timing.seconds[findfirst(==(:Kelly),KTrader.TIMING_BUCKETS)]>0
        @test count(>(0),timing.seconds)==1
        @test weights[3]==(held===nothing ? 0.0 : held[3])
    end
    decision=(;X=fill(-1.0,3,2),model=nothing,active=[1,2],free=trues(2),timing=KTrader.DecisionTiming())
    @test_throws ArgumentError KTrader._backtest_target(decision,nothing,false,1,3,1e-5,512)
end

@testset "Backtest adaptive boundary preserves seed and result count" begin
    # Deterministic zero-response fixture; no change to refinement tolerances.
    model=KTrader.fit_v1(ones(340,2);ridge_alpha=2.0,F_folds=3)
    free=trues(2); timing=KTrader.DecisionTiming()
    decision=(;X=nothing,model,active=model.active_indices,free,timing)
    expected=KTrader.adaptive_scenario_weights(model,free,nothing;
        rng=MersenneTwister(91),tol=1e-5,max_scenarios=512)
    weights,S=KTrader._backtest_target(decision,nothing,true,91,300,1e-5,512)
    @test weights==expected.weights
    @test S==expected.S
    @test timing.seconds[findfirst(==(:scenario),KTrader.TIMING_BUCKETS)]>0
    @test count(>(0),timing.seconds)==1
end
