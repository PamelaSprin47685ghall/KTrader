using Test, Random, LinearAlgebra
module FFTInnovationEnv
    using Random, LinearAlgebra
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
end
using .FFTInnovationEnv

@testset "fractional covariance FFT equals date-aware direct prefix" begin
    rng=MersenneTwister(20261011)
    for N in (1,3,12), d in (0.05,0.6,1.0)
        n=132
        eps=0.012 .* randn(rng,n,N)
        ids=cumsum(rand(rng,1:3,n))
        prefix=FFTInnovationEnv._fft_prefix_covariances(eps,ids,d)
        @test prefix !== nothing
        for i in (2,3,9,64,100,132)
            vref=FFTInnovationEnv.V_direct(view(eps,1:i-1,:),view(ids,1:i-1),d,ids[i]-1)
            @test isapprox(prefix(i),vref;atol=1e-13,rtol=1e-10)
        end
        function reference_quasi()
            ll=0.0
            for i in 2:n
                v=FFTInnovationEnv.V_direct(view(eps,1:i-1,:),view(ids,1:i-1),d,ids[i]-1)
                f=mp_sqrt_factors(v)
                ll -= 0.5 * (sum(log,f.positive) + norm(f.inv_half * view(eps,i,:))^2)
            end
            ll
        end
        @test isapprox(quasi_loglik(eps,ids,d),reference_quasi();
                       atol=1e-6,rtol=1e-9)
        pool=FFTInnovationEnv.DConditionedShapePool(eps,ids,collect(2:n),nothing)
        Z=FFTInnovationEnv.rows(pool,d)
        for k in (1,3,64,100,131)
            @test isapprox(vec(Z[k,:]),FFTInnovationEnv.row_at(pool,d,k);
                           atol=1e-10,rtol=1e-8)
        end
        if N >= 3
            plan=FFTInnovationEnv._prepare_fractional_fft(eps,ids)
            @test plan!==nothing
            private_prefix=FFTInnovationEnv._fft_prefix_covariances(eps,ids,d;
                                      prepared=plan)
            immutable_expected=private_prefix(64)
            qs=quasi_loglik(eps,ids,d;prepared_fft=plan,
                               reuse_fft_buffers=true)
            @test isapprox(qs,quasi_loglik(eps,ids,d;
                                      prepared_fft=plan);atol=1e-7,rtol=1e-11)
            @test immutable_expected ≈ private_prefix(64) atol=1e-12
            other_d=d == 1.0 ? 0.05 : d+0.07
            qo=quasi_loglik(eps,ids,other_d;prepared_fft=plan,
                                   reuse_fft_buffers=true)
            @test isapprox(qo,quasi_loglik(eps,ids,other_d;
                                      prepared_fft=plan);atol=1e-7,rtol=1e-11)
            @test immutable_expected ≈ private_prefix(64) atol=1e-12
        end
    end
end
