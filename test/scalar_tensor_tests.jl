using KTrader, Test, Random, LinearAlgebra
include(joinpath(@__DIR__,"fixtures","scalar_tensor_reference.jl"))

@testset "Packed scalar tensors: original products and scalar search" begin
    rng=MersenneTwister(1081305)
    nodes=collect(range(log(KTrader.EB_ALPHA_MIN),log(KTrader.EB_ALPHA_MAX);length=33))
    for N in (3,5), dual in (false,true), gauged in (false,true)
        P=2length(KTrader.BANDS)*N; n=dual ? 19 : 90
        X=randn(rng,n,P); Y=randn(rng,n,N)
        gauge=gauged ? KTrader.relative_gauge(N) : nothing
        gauged && (Y=Y*gauge*gauge')
        spectrum=KTrader.ridge_spectrum(X,X'X,X'Y;dual)
        geometry=KTrader._conditioned_scalar_geometry(spectrum,Y'Y;gauge)
        frozen=deepcopy(geometry)
        d=gauged ? N-1 : N
        A=randn(rng,d,d); Q=Matrix(qr(A).Q)
        for condition in (1.0,1e8)
            S=Q*Diagonal(collect(range(1/condition,1.0;length=d)))*Q'
            Sigma=gauged ? gauge*S*gauge' : S
            # The producer consumes THIS projected S, not the unrounded input.
            compact=gauged ? gauge'*Sigma*gauge : Sigma
            old=scalar_tensor_reference(geometry.blocks,compact)
            packed=KTrader._conditioned_scalar_tensors(geometry.blocks,compact)
            nc=length(geometry.blocks); pair=0
            @test size(packed)==(length(spectrum.values),div(nc*(nc+1),2))
            for r in 1:nc,s in r:nc
                pair+=1
                @test isequal(packed[:,pair],old[:,r,s])
            end
            reference=scalar_cache_reference(spectrum,n,Sigma,geometry;gauge)
            actual=KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
            for a in vcat(nodes,log.([0.17,0.7,27.0]))
                @test isequal(actual.evidence(a),reference.evidence(a))
                @test isequal(actual.derivative(a),reference.derivative(a))
            end
            # New Sigma/call cannot overwrite the first cache's data.
            expected=actual.evidence(0.0)
            another=KTrader._conditioned_scalar_cache(spectrum,n,2Sigma,geometry;gauge)
            another.evidence(0.0)
            @test isequal(actual.evidence(0.0),expected)
            @test isequal(geometry,frozen)
            if condition==1.0
                @test isequal(KTrader.maximize_logalpha(actual.evidence,actual.derivative),
                    KTrader.maximize_logalpha(reference.evidence,reference.derivative))
            end
        end
    end
end

@testset "Packed tensor producer: empty spectral space and no input mutation" begin
    blocks=[zeros(2,0) for _ in 1:14]; S=Matrix{Float64}(I,2,2)
    @test size(KTrader._conditioned_scalar_tensors(blocks,S))==(0,105)
    b=[reshape([1.0,2.0],2,1),reshape([3.0,4.0],2,1)]
    saved=deepcopy(b)
    @test KTrader._conditioned_scalar_tensors(b,S)==[5.0 11.0 25.0]
    @test isequal(b,saved)
    @test S==Matrix{Float64}(I,2,2)
end
