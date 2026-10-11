using Test, Random, LinearAlgebra

module CovarianceActionEnv
    include(joinpath(@__DIR__, "../../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/modes.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/response.jl"))
    include(joinpath(@__DIR__, "../../src/gate0/posterior.jl"))
end
using .CovarianceActionEnv

@testset "posterior nodes are the certified Gauss integration nodes" begin
    q = adaptive_quadrature_2d((x, y) -> 0.0, (-1.0, 1.0), (-1.0, 1.0);
                               tail_rel=1.0, max_cells=16)
    @test length(q.nodes) == 4*q.n_cells
    @test exp(q.logZ) ≈ 4.0 atol=1e-13
    @test sum(q.weights) ≈ 1.0 atol=1e-14
    @test sum(w*x^2 for ((x,y), w) in zip(q.nodes, q.weights)) ≈ 1/3 atol=1e-13
    @test sum(w*y^2 for ((x,y), w) in zip(q.nodes, q.weights)) ≈ 1/3 atol=1e-13
    @test sum(w*x^2*y^2 for ((x,y), w) in zip(q.nodes, q.weights)) ≈ 1/9 atol=1e-13
end

@testset "posterior covariance action is inverse precision" begin
    rng = MersenneTwister(20261011)
    X = randn(rng, 40, 15)
    X[:, 1] .= 1.0
    Y = randn(rng, 40, 1)
    st = sufficient_stats(X, Y)
    lam = vcat(2.0, fill(5.0, 14))
    A = X' * X + Diagonal(lam)
    x = randn(rng, 15)
    expected = A \ x
    spec = CovarianceActionEnv._rank1_spec(st)
    dense = CovarianceActionEnv._fit_node(st, lam, 1.0)
    lazy = CovarianceActionEnv._fit_node(st, lam, 1.0; rank1=true, spec=spec)
    for node in (dense, lazy)
        actual = CovarianceActionEnv.V_solve(node, x)
        @test actual ≈ expected rtol=1e-11 atol=1e-12
        @test A * actual ≈ x rtol=1e-11 atol=1e-12
        @test dot(x, actual) ≈ dot(x, expected) rtol=1e-11
        @test norm(actual - A*x) > 1.0 # rejects the former inverted action
    end
    @test Matrix(lazy.B_hat) ≈ dense.B_hat rtol=1e-11 atol=1e-12
    @test lazy.B_hat * x ≈ dense.B_hat * x rtol=1e-11 atol=1e-12
    @test lazy.B_hat[1, 3] ≈ dense.B_hat[1, 3] rtol=1e-11
    post = fit_full_posterior(X, Y; prior=:point_mass, alpha_fixed=(2.0, 5.0))
    moments = predict_mu(post, x)
    C = Matrix{Float64}(I, size(X, 1), size(X, 1)) + X * (Diagonal(lam) \ X')
    S = only(Y' * (C \ Y))
    @test only(moments.within) ≈ S * dot(x, expected) / (post.nu - 2) rtol=1e-10
    @test norm(moments.between) < 1e-12

    # Information can only reduce the coefficient covariance at fixed prior.
    st2 = sufficient_stats(vcat(X, X), vcat(Y, Y))
    node2 = CovarianceActionEnv._fit_node(st2, lam, 1.0)
    @test dot(x, CovarianceActionEnv.V_solve(node2, x)) < dot(x, expected)
end

@testset "spectral evidence contracts the same Gaussian model" begin
    rng = MersenneTwister(41)
    X, Y = randn(rng, 70, 29), randn(rng, 70, 2)
    X[:, 1] .= 1.0
    st = sufficient_stats(X, Y)
    spec = CovarianceActionEnv._rank1_spec(st)
    cache = Dict{Float64,Union{Nothing,CovarianceActionEnv.Rank1EvidenceSlice}}()
    for a0 in (0.1, 2.0, 50.0), ap in (0.2, 5.0, 100.0)
        lam = vcat(a0, fill(ap, size(X, 2)-1))
        direct = log_evidence(st, lam)
        spectral = CovarianceActionEnv._log_evidence_rank1(st, lam, spec)
        cached = CovarianceActionEnv._log_evidence_rank1_cached(st, lam, spec, cache)
        @test spectral ≈ direct atol=1e-9 rtol=1e-11
        @test cached ≈ spectral atol=1e-9 rtol=1e-11
    end
    @test length(cache) == 3 # one P×N contraction per αp, not per (α0,αp)
end

@testset "full hyperposterior spectral representation vs dense reference" begin
    rng = MersenneTwister(42)
    X, Y = randn(rng, 35, 15), randn(rng, 35, 1)
    X[:, 1] .= 1.0
    x = randn(rng, 15)
    slow = fit_full_posterior(X, Y; rank1=false)
    fast = fit_full_posterior(X, Y; rank1=true)
    ps, pf = predict_mu(slow, x), predict_mu(fast, x)
    @test fast.evidence ≈ slow.evidence atol=1e-8 rtol=1e-9
    @test pf.mu ≈ ps.mu atol=1e-8 rtol=1e-7
    @test pf.within ≈ ps.within atol=1e-8 rtol=1e-7
    @test pf.between ≈ ps.between atol=1e-8 rtol=1e-7
    @test all(c -> c.V === nothing, fast.conditional)
    @test Base.summarysize(fast) < Base.summarysize(slow)
end
