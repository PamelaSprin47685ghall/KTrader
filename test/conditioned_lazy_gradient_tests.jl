using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","conditioned_cpu_reference.jl"))
include(joinpath(@__DIR__,"fixtures","conditioned_eager_state_reference.jl"))

@testset "Lazy optimizer gradient: eager formula, floor, gauge and dual agreement" begin
    for N in (3,5), dual in (false,true), gauged in (false,true)
        f=cpu_synthetic_spectrum(N;dual)
        gauge=gauged ? f.gauge : nothing
        YtY=gauged ? f.YtY : f.YtY+f.n*I
        d=gauged ? N-1 : N
        for alpha in (0.01,0.7,1e4)
            for floor_case in (false,true)
                cache=KTrader.conditioned_alpha_cache(f.spectrum,YtY,f.n,alpha;gauge)
                S=Matrix{Float64}(I,d,d)
                floor_case && (S[1,1]=1e-8)
                # Explicitly exercise cold and already materialized core paths.
                for cold in (true,false)
                    cold && (cache.jcore[]=nothing)
                    expected=eager_state_reference(deepcopy(cache),S)
                    actual=KTrader.conditioned_cached_state(deepcopy(cache),S;gradient=true)
                    lazy=KTrader._conditioned_cached_state(deepcopy(cache),S;direction=true)
                    @test isequal(actual,expected)
                    for key in (:value,:direction,:M,:J)
                        @test isequal(getproperty(lazy,key),getproperty(expected,key))
                    end
                    @test !hasproperty(lazy,:grad)
                    @test isequal(KTrader._conditioned_covariance_gradient(cache,S,lazy.J),expected.grad)
                    @test isequal(KTrader._conditioned_covariance_gradient(cache,S,lazy.J,eigen(Symmetric(S))),expected.grad)
                    @test KTrader.conditioned_cached_state(deepcopy(cache),S)==expected.value
                    KTrader.conditioned_constraint_jacobian(cache,cholesky(Symmetric(expected.M)),zeros(length(cache.h)))
                end
                @test_throws PosDefException KTrader._conditioned_cached_state(cache,-S;direction=true)
                @test_throws PosDefException KTrader.conditioned_cached_state(cache,-S;gradient=true)
            end
        end
    end
end
