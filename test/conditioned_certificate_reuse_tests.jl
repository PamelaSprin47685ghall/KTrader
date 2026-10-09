using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","conditioned_cpu_reference.jl"))
include(joinpath(@__DIR__,"fixtures","conditioned_certificate_reference.jl"))

@testset "Certificate fit-geometry reuse: all original fields, no weakened gate" begin
    for N in (3,5), dual in (false,true), gauged in (false,true)
        f=cpu_synthetic_spectrum(N;dual)
        (;spectrum,n)=f
        gauge=gauged ? f.gauge : nothing
        YtY=gauged ? f.YtY : f.YtY+n*I
        geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
        ag=KTrader._conditioned_alpha_geometry(geometry)
        frozen=deepcopy((geometry,ag))
        d=gauged ? N-1 : N
        for floor_active in (false,true), alpha in (1e-4,0.7,1.0,27.0,1e6)
            S=Matrix(Diagonal(collect(range(0.7,1.3;length=d))))
            floor_active && (S[1,1]=KTrader.EB_COVARIANCE_FLOOR)
            Sigma=gauged ? gauge*S*gauge' : S
            reference=certificate_eager_reference(spectrum,YtY,n,alpha,Sigma;gauge)
            fresh=KTrader.conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma;gauge)
            reused=KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,ag;gauge)
            @test isequal(fresh,reference)
            @test isequal(reused,reference)
            @test isequal((geometry,ag),frozen)
            # Interleaved Sigma values do not inherit a previous acceptance.
            other=KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,2Sigma,geometry,ag;gauge)
            @test isequal(other,certificate_eager_reference(spectrum,YtY,n,alpha,2Sigma;gauge))
            @test isequal(reused,KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,ag;gauge))
        end
        bad=gauged ? gauge*(-Matrix{Float64}(I,d,d))*gauge' : -Matrix{Float64}(I,d,d)
        @test_throws PosDefException certificate_eager_reference(spectrum,YtY,n,1.0,bad;gauge)
        @test_throws PosDefException KTrader.conditioned_eb_certificate(spectrum,YtY,n,1.0,bad;gauge)
        @test_throws PosDefException KTrader._conditioned_eb_certificate(spectrum,YtY,n,1.0,bad,geometry,ag;gauge)
    end
end
