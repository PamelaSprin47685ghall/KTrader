# Small release-installation check; does not replace the qualified suite.
using KTrader, Test, Random, LinearAlgebra

@testset "2.0.0 Final loaded package and CPU pipeline" begin
    @test Base.pkgversion(KTrader)==v"2.0.0"
    previous=BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        rng=MersenneTwister(1080751)
        prices=100exp.(cumsum(0.01randn(rng,340,3);dims=1))
        prices[1:35,3].=NaN
        prices[280:281,2].=NaN
        prices[302,1]=NaN
        original=copy(prices)
        prep=prepare_reference(prices;F_folds=3)
        model=solve(prep)
        @test prep isa PreparedProblem
        @test prep.F_folds==3
        @test isequal(prices,original)
        @test model.active_indices==[1,2,3]
        @test all(isfinite,model.mu_pred)
        X=generate_scenarios_v1(model;S=64,rng=MersenneTwister(108))
        w=KTrader.scenario_weights(X,model.active_indices,trues(3),nothing)
        @test size(X)==(64,3)
        @test all(x->isfinite(x)&&x>0,X)
        @test all(isfinite,w) && minimum(w)>=0
        @test isapprox(sum(w),1.0;atol=1e-8,rtol=0)
        @test KTrader.certified(KTrader.kelly_certificate(X,w),1e-8)
        @test !isdefined(KTrader,:CUDA) && !isdefined(KTrader,:AMDGPU)
    finally
        BLAS.set_num_threads(previous)
    end
    @test BLAS.get_num_threads()==previous
end
