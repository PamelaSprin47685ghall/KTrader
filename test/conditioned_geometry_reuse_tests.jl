using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","conditioned_cpu_reference.jl"))

@testset "Alpha geometry: fit-local reuse, exact values, independent mutable state" begin
    for N in (3,5), dual in (false,true), gauged in (false,true)
        f=cpu_synthetic_spectrum(N;dual)
        (; spectrum,n)=f
        gauge=gauged ? f.gauge : nothing
        YtY=gauged ? f.YtY : f.YtY+n*I
        scalar_geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
        @test !hasproperty(scalar_geometry,:bcat)
        geometry=KTrader._conditioned_alpha_geometry(scalar_geometry)
        saved=deepcopy(geometry)
        d=gauged ? N-1 : N
        S=Matrix{Float64}(I,d,d)
        previous=nothing
        for alpha in (1e-4,0.7,1.0,27.0,1e6)
            expected=cpu_reference_alpha(spectrum,YtY,n,alpha;gauge)
            actual=KTrader._conditioned_alpha_cache(spectrum,YtY,n,alpha,geometry;gauge)
            fresh=KTrader.conditioned_alpha_cache(spectrum,YtY,n,alpha;gauge)
            for key in (:R,:blocks,:h,:constant,:bcat,:bwcat)
                @test isequal(getproperty(actual,key),getproperty(expected,key))
                @test isequal(getproperty(fresh,key),getproperty(expected,key))
            end
            @test actual.V.weights==expected.V.weights
            @test actual.V.baseline==expected.V.baseline
            @test actual.bcat===geometry.bcat
            @test actual.blocks===geometry.blocks
            @test fresh.bcat!==geometry.bcat
            @test actual.jcore[]===nothing
            @test isequal(KTrader.build_jacobian_cores(actual),cpu_reference_cores(expected))
            # Building/evaluating cores cannot mutate the shared geometry.
            KTrader.conditioned_constraint_jacobian(actual, cholesky(Symmetric(
                KTrader.conditioned_constraint_contraction(actual,S))),zeros(length(actual.h)))
            @test isequal(geometry,saved)
            if previous!==nothing
                @test actual.bwcat!==previous.bwcat
                @test actual.jcore!==previous.jcore
                @test actual.jcore[]!==previous.jcore[]
                @test actual.h!==previous.h
            end
            previous=actual
        end
        # A new fold/target and a direct call cannot inherit cached state.
        g2=KTrader._conditioned_alpha_geometry(KTrader._conditioned_scalar_geometry(spectrum,2YtY;gauge))
        @test g2.blocks!==geometry.blocks
        @test g2.bcat!==geometry.bcat
        c2=KTrader._conditioned_alpha_cache(spectrum,2YtY,n,0.7,g2;gauge)
        @test c2.R!=KTrader._conditioned_alpha_cache(spectrum,YtY,n,0.7,geometry;gauge).R
        @test isequal(geometry,saved)
    end
end

@testset "Core symmetrization uses both triangles and keeps independent matrices" begin
    cache=(;blocks=[[1.0;2.0],[3.0;4.0]],bcat=[1.0 3.0;2.0 4.0],bwcat=[1.0 3.0;2.0 4.0])
    saved=deepcopy(cache)
    cores=KTrader.build_jacobian_cores(cache)
    @test cores[2]==[3.0 5.0;5.0 8.0]
    @test isequal(cores,cpu_reference_cores(cache))
    @test isequal(cache,saved)
    cores[1][1,1]=100.0
    @test cores[2]==[3.0 5.0;5.0 8.0]
    @test isequal(cache,saved)
end
