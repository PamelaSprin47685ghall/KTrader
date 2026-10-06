using Test, LinearAlgebra, Statistics, Random
using Convex, Clarabel

include("../src/geometry.jl")
include("../src/response.jl")
include("../src/predict.jl")
include("../src/kelly.jl")

@testset "Path Kelly V1: Correctness & Theoretical Invariants" begin
    rng = MersenneTwister(2026)
    T, N = 1200, 8
    
    # Base synthetic price paths
    r_rand = 0.01 .* randn(rng, T, N)
    adj_base = exp.(cumsum(r_rand, dims=1))
    
    # ----------------------------------------------------
    # Invariant 1: Strict Orthogonality e_0' * Phi_perp ≡ 0
    # ----------------------------------------------------
    u_decomp = center_of_mass_decomposition(r_rand)
    rel_modes = relative_modes(u_decomp.u_perp, u_decomp.e0)
    Phi_perp = rel_modes.Phi_perp
    K_rel = size(Phi_perp, 2)
    
    @test K_rel == N - 1
    @test norm(u_decomp.e0' * Phi_perp) < 1e-14
    @test Phi_perp' * Phi_perp ≈ I(K_rel) atol=1e-12
    println("✓ Property 1 Passed: Strict Orthogonal Subspace e_0' * Phi_perp ≡ 0 (norm = ", norm(u_decomp.e0' * Phi_perp), ").")

    # ----------------------------------------------------
    # Invariant 2: Operator Trace Neutrality tr(A_b) = 0, tr(B_b) = 0
    # ----------------------------------------------------
    model_base = fit_v1(adj_base; ridge_alpha = 10.0)
    
    @test abs(model_base.resp.trace_real) < 1e-12
    @test abs(model_base.resp.trace_imag) < 1e-12
    println("✓ Property 2 Passed: Exact Operator Trace Neutrality tr(A_b) ≡ 0, tr(B_b) ≡ 0.")

    # ----------------------------------------------------
    # Invariant 3: Price Scale Invariance (P_i -> c_i * P_i)
    # ----------------------------------------------------
    scales = exp.(randn(rng, N))
    adj_scaled = adj_base .* scales'
    
    w_base = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1))
    w_scaled = path_kelly_v1(adj_scaled; S = 300, rng = MersenneTwister(1))
    
    @test isapprox(w_base, w_scaled, atol = 1e-4)
    println("✓ Property 3 Passed: Price Scale Invariance holds.")
    
    # ----------------------------------------------------
    # Invariant 4: Asset Permutation Invariance
    # ----------------------------------------------------
    # When columns are permuted as A_perm = A[:, perm], the asset in column k is original asset perm[k]
    # Thus the optimal weight on column k must be w_base[perm[k]]: w_perm == w_base[perm]
    perm = randperm(rng, N)
    adj_perm = adj_base[:, perm]
    
    w_perm = path_kelly_v1(adj_perm; S = 300, rng = MersenneTwister(1))
    @test isapprox(w_perm, w_base[perm], atol = 1e-4)
    println("✓ Property 4 Passed: Asset Permutation Invariance holds.")

    # ----------------------------------------------------
    # Invariant 5: Exact Locked-Risk Kelly
    # ----------------------------------------------------
    held = fill(1.0 / N, N)
    tradable = trues(N); tradable[2] = false
    w_locked = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1), tradable, held)
    
    @test w_locked[2] ≈ held[2]
    @test isapprox(sum(w_locked), 1.0, atol = 1e-8)
    @test isapprox(sum(w_locked[tradable]), 1.0 - held[2], atol = 1e-8)
    println("✓ Property 5 Passed: Exact Locked-Risk Kelly preserves locked weights and reallocates remainder.")
end
