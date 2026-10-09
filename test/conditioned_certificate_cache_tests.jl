using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","conditioned_cpu_reference.jl"))

@testset "Certificate alpha-cache reuse never reuses Sigma acceptance" begin
    for N in (3,5), dual in (false,true), gauged in (false,true)
        f=cpu_synthetic_spectrum(N;dual)
        gauge=gauged ? f.gauge : nothing
        YtY=gauged ? f.YtY : f.YtY+f.n*I
        geometry=KTrader._conditioned_scalar_geometry(f.spectrum,YtY;gauge)
        ag=KTrader._conditioned_alpha_geometry(geometry)
        dim=gauged ? N-1 : N
        for alpha in (0.1,1.0,31.0)
            cache=KTrader._conditioned_alpha_cache(f.spectrum,YtY,f.n,alpha,ag;gauge)
            original=(;R=copy(cache.R),h=copy(cache.h),bwcat=copy(cache.bwcat))
            for scale in (0.1,1.0,7.0)
                S=Matrix(Diagonal(collect(range(scale,2scale;length=dim))))
                Sigma=gauged ? gauge*S*gauge' : S
                fresh=KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha,Sigma,geometry,ag;gauge)
                reused=KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha,Sigma,geometry,ag,cache;gauge)
                @test isequal(reused,fresh)
                cores=cache.jcore[]
                @test cores!==nothing
                again=KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha,Sigma,geometry,ag,cache;gauge)
                @test isequal(again,fresh)
                @test cache.jcore[]===cores
                @test cache.R==original.R && cache.h==original.h && cache.bwcat==original.bwcat
            end
            Sigma=gauged ? gauge*gauge' : Matrix{Float64}(I,N,N)
            @test_throws ArgumentError KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha+1,Sigma,geometry,ag,cache;gauge)
            @test_throws ArgumentError KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n+1,alpha,Sigma,geometry,ag,cache;gauge)
            other=KTrader._conditioned_alpha_cache(f.spectrum,2YtY,f.n,alpha,ag;gauge)
            @test other.jcore!==cache.jcore
            @test other.R!=cache.R
        end
    end
end
