using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"fixtures","conditioned_cpu_reference.jl"))

@testset "Fit-owned alpha storage: rewritten values, stale rejection, no output alias" begin
    for N in (3,5), dual in (false,true), gauged in (false,true)
        f=cpu_synthetic_spectrum(N;dual)
        gauge=gauged ? f.gauge : nothing
        YtY=gauged ? f.YtY : f.YtY+f.n*I
        geometry=KTrader._conditioned_scalar_geometry(f.spectrum,YtY;gauge)
        ag=KTrader._conditioned_alpha_geometry(geometry)
        buffers=KTrader._conditioned_alpha_storage(ag)
        d=gauged ? N-1 : N; S=Matrix{Float64}(I,d,d)
        Sigma=gauged ? gauge*S*gauge' : S
        frozen=deepcopy(ag); previous=nothing; saved_state=nothing
        for alpha in (0.1,1.0,17.0,0.1)
            independent=KTrader._conditioned_alpha_cache(f.spectrum,YtY,f.n,alpha,ag;gauge)
            borrowed=KTrader._conditioned_alpha_cache(f.spectrum,YtY,f.n,alpha,ag,buffers;gauge)
            @test borrowed.bwcat===buffers.weighted
            @test borrowed.jcore[]===nothing
            @test !hasproperty(independent,:scratch)
            for key in (:R,:h,:bwcat,:bcat,:constant)
                @test isequal(getproperty(borrowed,key),getproperty(independent,key))
            end
            # Cold M must follow the original direct multiplication order.
            @test isequal(KTrader.conditioned_constraint_contraction(borrowed,S),
                          KTrader.conditioned_constraint_contraction(independent,S))
            actual=KTrader.conditioned_cached_state(borrowed,S;gradient=true)
            expected=KTrader.conditioned_cached_state(independent,S;gradient=true)
            @test isequal(actual,expected)
            @test isequal(borrowed.jcore[],cpu_reference_cores(independent))
            @test isequal(KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha,Sigma,geometry,ag,borrowed;gauge),
                          KTrader._conditioned_eb_certificate(f.spectrum,YtY,f.n,alpha,Sigma,geometry,ag,independent;gauge))
            if previous!==nothing
                @test_throws ErrorException KTrader.conditioned_constraint_contraction(previous,S)
                @test_throws ErrorException KTrader._conditioned_constraint_direct(previous,S)
                @test_throws ErrorException KTrader.conditioned_cached_state(previous,S)
                @test_throws ErrorException KTrader.build_jacobian_cores(previous)
                @test isequal(saved_state[1],saved_state[2])
            end
            # Poison all reusable storage; the next lease must rewrite it.
            # Independent/public cache and returned state are unaffected.
            # Compare warm-core to warm-core here. The first expected state
            # above deliberately exercised the original cold/direct M route.
            unchanged_expected=KTrader.conditioned_cached_state(independent,S;gradient=true)
            saved_state=(actual,deepcopy(actual)); previous=borrowed
            fill!(buffers.weighted,NaN)
            fill!(buffers.product,NaN)
            foreach(A->fill!(A,NaN),buffers.cores[])
            @test isequal(KTrader.conditioned_cached_state(independent,S;gradient=true),unchanged_expected)
            @test isequal(ag,frozen)
        end
        # No scratch sharing across fits; reject a foreign geometry before
        # retiring the current lease or touching its buffers.
        ag2=KTrader._conditioned_alpha_geometry(geometry)
        generation=buffers.generation[]
        @test_throws ArgumentError KTrader._conditioned_alpha_cache(f.spectrum,YtY,f.n,1.0,ag2,buffers;gauge)
        @test buffers.generation[]==generation
        other=KTrader._conditioned_alpha_storage(ag2)
        @test other.weighted!==buffers.weighted
        @test other.product!==buffers.product
        @test other.cores!==buffers.cores
    end
end

@testset "Independent caches keep their prior alpha after later preparations" begin
    f=cpu_synthetic_spectrum(3)
    c1=KTrader.conditioned_alpha_cache(f.spectrum,f.YtY,f.n,0.1;gauge=f.gauge)
    first=deepcopy(c1)
    c2=KTrader.conditioned_alpha_cache(f.spectrum,f.YtY,f.n,17.0;gauge=f.gauge)
    @test c1.R==first.R && c1.h==first.h && c1.bwcat==first.bwcat && c1.constant==first.constant
    @test c1.jcore[]===first.jcore[]===nothing
    @test c1.bwcat!==c2.bwcat
    @test c1.jcore!==c2.jcore
end
