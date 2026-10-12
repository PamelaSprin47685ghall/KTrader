using Test, Random, LinearAlgebra, Serialization
module PrefixScenarioEnv
    using Random, LinearAlgebra, Serialization
    include(joinpath(@__DIR__,"../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/response.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/innovation.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
    include(joinpath(@__DIR__,"../../src/gate0/quadrature.jl"))
end
using .PrefixScenarioEnv

@testset "risk coordinate projection is a full-rank algebraic composition" begin
    rng=MersenneTwister(609)
    E=domain_mode_basis(6)
    for R in ([1,2,3,4,5,6],[1,3,6],[2,4,5])
        Er=risk_domain_mode_basis(collect(R))
        T=Er' * E[R,:]
        @test size(T)==(length(R),6)
        for _ in 1:12
            mu=randn(rng,6)
            old=Er' * (E*mu)[R]
            @test isapprox(T*mu,old;atol=2e-13,rtol=1e-12)
        end
    end
end

@testset "QMC draw scratch preserves the complete RNG stream" begin
    rng=MersenneTwister(760)
    X=randn(rng,48,4)
    Y=randn(rng,48,4)
    post=fit_full_posterior(X,Y;prior=:point_mass,alpha_fixed=0.6)
    xt=randn(rng,4)
    cache=PrefixScenarioEnv._MuDrawCache(post)
    z=zeros(4)
    mc=zeros(2)
    mu=zeros(4)
    for ug in (NaN,0.3)
        reference=MersenneTwister(112)
        scratch=MersenneTwister(112)
        for k in 1:20
            node=mod(0.137*k,1)
            a=PrefixScenarioEnv._draw_mu_qmc(post,xt,node,0.23,0.72,
                                             reference;u_g=ug)
            b=PrefixScenarioEnv._draw_mu_qmc(post,xt,node,0.23,0.72,
                         scratch;u_g=ug,node_cdf=cache.cdf,node_cache=cache,
                         z_scratch=z,mc_scratch=mc,mu_scratch=mu)
            @test isapprox(a,b;atol=2e-12,rtol=2e-12)
            @test rand(reference)==rand(scratch)
        end
    end
end

@testset "parallel conditional factors preserve seeded QMC exactly" begin
    if Threads.nthreads() >= 2
        rng=MersenneTwister(20261012)
        X=randn(rng,88,43) # DC + 14 per mode, N=3
        Y=randn(rng,88,3)
        post=fit_full_posterior(X,Y)
        E=domain_mode_basis(3)
        eps=0.015 .* randn(rng,81,3)
        st=innovation_state(eps,collect(1:81),[trues(3) for _ in 1:81],
                            [1,2,3],E;t=81,d_nodes=[0.2,0.7],
                            d_cells=[0.4,0.6],require_full_rank=true)
        x=randn(rng,43)
        src_a=predictive_rqmc_source(post,st,x;s1=ones(3),E_active=E,
                           mu_qmc=true,mu_chisq_qmc=true,
                           prefetch_mu_factors=false)
        src_b=predictive_rqmc_source(post,st,x;s1=ones(3),E_active=E,
                           mu_qmc=true,mu_chisq_qmc=true,
                           prefetch_mu_factors=true)
        rule=SobolOwenRule(2,UInt64(885))
        for M in (128,512)
            a=MersenneTwister(100)
            b=MersenneTwister(100)
            ga=src_a.gen(M,rule,a)
            gb=src_b.gen(M,rule,b)
            @test isequal(ga,gb)
            @test rand(a)==rand(b)
        end
        @test isequal(src_b.gen(128,rule,MersenneTwister(100)),
                     src_b.gen(512,rule,MersenneTwister(100))[1:128,:])
    else
        @test true
    end
end

@testset "lazy selected innovation shape equals complete empirical pool" begin
    rng=MersenneTwister(20261012)
    X=randn(rng,64,3)
    Y=randn(rng,64,3)
    post=fit_full_posterior(X,Y;prior=:point_mass,alpha_fixed=0.7)
    E=domain_mode_basis(3)
    eps=0.025 .* randn(rng,132,3)
    dates=cumsum(rand(rng,1:3,132))
    st=innovation_state(eps,dates,[trues(3) for _ in dates],
                        [1,2,3],E;t=dates[end],
                        d_nodes=[0.2,0.8],d_cells=[0.5,0.5],
                        require_full_rank=true)
    x=randn(rng,3)
    rule=SobolOwenRule(2,UInt64(70))
    for qmc in (false,true)
        cached=predictive_rqmc_source(post,st,x;
                     s1=ones(3),E_active=E,mu_qmc=qmc,lazy_z=true)
        reference=predictive_rqmc_source(post,st,x;
                     s1=ones(3),E_active=E,mu_qmc=qmc,lazy_z=false)
        for m in (32,128)
            got=cached.gen(m,rule,MersenneTwister(56))
            want=reference.gen(m,rule,MersenneTwister(56))
            @test isapprox(got,want;atol=1e-11,rtol=1e-9)
        end
    end
end

@testset "nested scenario source preserves independent RNG and full replay" begin
    rng = MersenneTwister(95)
    X = randn(rng,30,2)
    Y = randn(rng,30,2)
    post = fit_full_posterior(X,Y;prior=:point_mass,alpha_fixed=0.6)
    eps = 0.02 .* randn(rng,22,2)
    E = domain_mode_basis(2)
    st = innovation_state(eps,collect(1:22),[trues(2) for _ in 1:22],
                         [1,2],E;t=22,d_nodes=[0.5],d_cells=[1.0])
    x = randn(rng,2)
    for qmc in (false,true), chisq in (false,true)
        !qmc && chisq && continue
        src = predictive_rqmc_source(post,st,x;s1=ones(2),E_active=E,
                                     mu_qmc=qmc,mu_chisq_qmc=chisq)
        rule = SobolOwenRule(2,UInt64(7))
        src.gen(16,rule,MersenneTwister(87))
        advance = MersenneTwister(87)
        cached = src.gen(64,rule,advance)
        fresh = predictive_rqmc_source(post,st,x;s1=ones(2),E_active=E,
                                       mu_qmc=qmc,mu_chisq_qmc=chisq)
        direct_rng = MersenneTwister(87)
        expected = fresh.gen(64,rule,direct_rng)
        @test isequal(cached,expected)
        @test rand(advance) == rand(direct_rng)
        @test isequal(cached[1:16,:],
                     src.gen(16,rule,MersenneTwister(87)))
        cached[1,1] = -1.0
        @test src.gen(64,rule,MersenneTwister(87))[1,1] > 0
        extended = src.gen(128,rule,MersenneTwister(87))
        fresh128 = predictive_rqmc_source(post,st,x;s1=ones(2),E_active=E,
                                        mu_qmc=qmc,mu_chisq_qmc=chisq)
        @test isequal(extended,fresh128.gen(128,rule,MersenneTwister(87)))
        @test isequal(extended[1:64,:],expected)
    end
end
