using Test, Random, LinearAlgebra
module DeferredBackend
    using LinearAlgebra
    include(joinpath(@__DIR__,"../../src/gate0/kelly.jl"))
end
using .DeferredBackend
const cash_kelly=DeferredBackend.cash_kelly
const cash_kelly_certificate=DeferredBackend.cash_kelly_certificate
const cash_kelly_certified=DeferredBackend.cash_kelly_certified

@testset "dominant vertex canonical is exact and avoids conic startup" begin
    @test !isdefined(DeferredBackend,:Convex)
    X = hcat(fill(1.06,128),fill(1.04,128),fill(1.02,128))
    w,c,cert=cash_kelly(X;tol=1e-7,tie_eps=1e-8)
    threshold=log(1.06)-1e-8
    exact_cash=1-(exp(threshold)-1)/0.06
    @test abs(c-exact_cash) < 2e-7
    @test w[2]==0.0 && w[3]==0.0
    @test cash_kelly_certified(cert;tol=1e-7)
    # Ustar from the certified numerical primary is slightly below the true
    # log(1.06) optimum. The vertex solution satisfies its ORIGINAL tie set;
    # the analytic optimum is only an independent numerical comparison.
    @test abs(cert.objective-threshold) < 1e-9
    @test !isdefined(DeferredBackend,:Convex)
end

@testset "certified cash-only canonical solution does not load conic solver" begin
    @test !isdefined(DeferredBackend,:Convex)
    X=fill(0.99,64,4)
    w,c,c2=cash_kelly(X;tol=1e-8,tie_eps=1e-8)
    @test w==zeros(4)
    @test c==1.0
    @test cash_kelly_certified(c2;tol=1e-8)
    @test c2.objective_gap==0
    @test !isdefined(DeferredBackend,:Convex)
end

@testset "full Kelly native one-risk analytic case does not eagerly load conic" begin
    X=reshape([1.2,0.85],2,1)
    w,c,cert=cash_kelly(X;tol=1e-8,tie_eps=1e-8)
    @test isapprox(w[1],5/6;atol=1e-6)
    @test isapprox(c,1/6;atol=1e-6)
    @test cash_kelly_certified(cert;tol=1e-8)
    @test !isdefined(DeferredBackend,:Convex)
end

@testset "native full-Hessian BLAS matches scalar original objective" begin
    rng=MersenneTwister(77)
    X=exp.(0.03 .* randn(rng,128,16))
    b=zeros(128)
    cash=ones(128)
    f=DeferredBackend._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=true)
    s=DeferredBackend._cash_kelly_ipm(X,b,cash,1.0,1e-6;blas_scan=false)
    @test f.converged && s.converged
    @test isapprox(f.w_val,s.w_val;atol=1e-7,rtol=1e-6)
    cf=cash_kelly_certificate(X,f.w_val[1:16],f.w_val[17];base=b)
    @test cash_kelly_certified(cf;tol=1e-6)
    @test !isdefined(DeferredBackend,:Convex)
end

if get(ENV,"PROBE_VERIFY_CONIC","false")=="true"
    @testset "nontrivial cash canonicalisation loads the original conic backend" begin
        X=Matrix{Float64}([1.4 1.03; 0.75 0.97;
                            1.20 0.99; 0.88 1.08;
                            1.32 0.91; 0.92 1.15])
        w,c,cert=cash_kelly(X;tol=1e-6,tie_eps=1e-8)
        @test all(isfinite,w) && c>=0
        @test cash_kelly_certified(cert;tol=1e-6)
        @test isdefined(DeferredBackend,:Convex)
    end
end
