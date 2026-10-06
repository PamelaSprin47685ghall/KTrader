using Test, LinearAlgebra, Statistics, Random
using Convex, Clarabel

include("../src/geometry.jl")
include("../src/response.jl")
include("../src/predict.jl")
include("../src/kelly.jl")

@testset "Path Kelly V0.95: Mathematical Invariants & Gauge Architecture" begin
    rng = MersenneTwister(2026)
    T, N = 1200, 8
    
    # ----------------------------------------------------
    # Invariant 1: Helmert Gauge Orthonormality & Strict e0 Orthogonality
    # ----------------------------------------------------
    e0 = center_of_mass(N)
    Q = helmert_basis(N)
    
    @test size(Q) == (N, N - 1)
    @test norm(Q' * e0) < 1e-14
    @test Q' * Q ≈ I(N - 1) atol=1e-14
    println("✓ Invariant 1 Passed: Helmert Basis Q strictly satisfies Q' * e0 ≡ 0 (norm = ", norm(Q' * e0), ") and Q' * Q ≡ I.")

    # ----------------------------------------------------
    # Invariant 2: Strict PSD Repair in (N-1) Subspace
    # ----------------------------------------------------
    r_rand = 0.01 .* randn(rng, T, N)
    u_decomp = center_of_mass_decomposition(r_rand)
    helm = project_helmert_gauge(u_decomp.u_perp, Q)
    C_Q = helm.C_Q
    
    @test size(C_Q) == (N - 1, N - 1)
    @test isposdef(Symmetric(C_Q))
    @test minimum(eigvals(Symmetric(C_Q))) >= 1e-6
    println("✓ Invariant 2 Passed: C_Q in (N-1) space is strictly Positive Semi-Definite (min eigenvalue = ", round(minimum(eigvals(Symmetric(C_Q))), digits=6), ").")

    # ----------------------------------------------------
    # Invariant 3: Operator Trace Neutrality tr(A_b) = 0, tr(B_b) = 0
    # ----------------------------------------------------
    adj_base = exp.(cumsum(r_rand, dims=1))
    model_base = fit_v1(adj_base)
    
    @test abs(model_base.resp.trace_real) < 1e-10
    @test abs(model_base.resp.trace_imag) < 1e-10
    println("✓ Invariant 3 Passed: Exact Operator Trace Neutrality tr(A_b) ≡ 0, tr(B_b) ≡ 0.")

    # ----------------------------------------------------
    # Invariant 4: Causal Fractional Likelihood Mixture
    # ----------------------------------------------------
    @test isapprox(sum(model_base.d_posterior), 1.0, atol=1e-8)
    @test all(model_base.d_posterior .>= 0.0)
    @test all(model_base.v_forecasts .> 0.0)
    println("✓ Invariant 4 Passed: Causal Fractional Log-Likelihood produces valid posterior mixture over d (Σp(d) = 1.0).")

    # ----------------------------------------------------
    # Invariant 5: Price Scale Invariance (P_i -> c_i * P_i)
    # ----------------------------------------------------
    scales = exp.(randn(rng, N))
    adj_scaled = adj_base .* scales'
    
    w_base = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1))
    w_scaled = path_kelly_v1(adj_scaled; S = 300, rng = MersenneTwister(1))
    
    @test isapprox(w_base, w_scaled, atol = 1e-4)
    println("✓ Invariant 5 Passed: Price Scale Invariance holds.")

    # ----------------------------------------------------
    # Invariant 6: Exact Locked-Risk Kelly
    # ----------------------------------------------------
    held = fill(1.0 / N, N)
    tradable = trues(N); tradable[3] = false
    w_locked = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1), tradable, held)
    
    @test w_locked[3] ≈ held[3]
    @test isapprox(sum(w_locked), 1.0, atol = 1e-8)
    @test isapprox(sum(w_locked[tradable]), 1.0 - held[3], atol = 1e-8)
    println("✓ Invariant 6 Passed: Exact Locked-Risk Kelly preserves locked weights and correctly shocks risk.")
end
