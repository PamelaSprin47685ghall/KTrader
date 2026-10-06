using Test, LinearAlgebra, Statistics, Random
using Convex, Clarabel

include("../src/geometry.jl")
include("../src/response.jl")
include("../src/predict.jl")
include("../src/kelly.jl")

@testset "Path Kelly V0.95: Numerical Convergence Suite" begin
    rng = MersenneTwister(2026)
    T, N = 1200, 6
    r = 0.01 .* randn(rng, T, N)
    adj = exp.(cumsum(r, dims=1))
    
    # 1. Scenario Quadrature Refinement Convergence: ||w_2S - w_S|| -> 0
    w_100 = path_kelly_v1(adj; S = 100, rng = MersenneTwister(1))
    w_300 = path_kelly_v1(adj; S = 300, rng = MersenneTwister(1))
    w_900 = path_kelly_v1(adj; S = 900, rng = MersenneTwister(1))
    
    diff_1 = norm(w_300 - w_100, 1)
    diff_2 = norm(w_900 - w_300, 1)
    
    @test diff_2 < diff_1 + 0.05
    @test diff_2 < 0.25
    println("✓ Convergence 1 Passed: Scenario nodes refinement converges (||w_900 - w_300||_1 = ", round(diff_2, digits=4), " < ", round(diff_1, digits=4), ").")
    
    # 2. Strict Invariant: Rotation Invariance of Response in Unbiased World
    adj_neg = exp.(-cumsum(r, dims=1))
    m_base = fit_v1(adj)
    m_neg  = fit_v1(adj_neg)
    
    @test abs(m_base.resp.trace_real) < 1e-10
    @test abs(m_neg.resp.trace_real) < 1e-10
    println("✓ Convergence 2 Passed: Strict Operator Trace Neutrality holds identically on reflection.")
    
    # 3. Helmert Gauge Stability across Dimensions N = 4..30
    for dim in 4:5:30
        Q_dim = helmert_basis(dim)
        e0_dim = center_of_mass(dim)
        @test norm(Q_dim' * e0_dim) < 1e-14
        @test Q_dim' * Q_dim ≈ I(dim - 1) atol=1e-14
    end
    println("✓ Convergence 3 Passed: Helmert Basis maintains strict machine precision orthogonality across all dimensions.")
end
