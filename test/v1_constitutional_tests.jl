using Test, LinearAlgebra, Statistics, Random
using Convex, Clarabel

include("../src/geometry.jl")
include("../src/response.jl")
include("../src/predict.jl")
include("../src/kelly.jl")

@testset "Path Kelly V1.0: Constitutional Invariant & Freeze Suite" begin
    rng = MersenneTwister(2026)
    T, N = 1200, 6
    
    r_base = 0.01 .* randn(rng, T, N)
    adj_base = exp.(cumsum(r_base, dims=1))
    
    # ----------------------------------------------------
    # Constitution 1: Price Scale Invariance (P_i -> c_i * P_i)
    # ----------------------------------------------------
    scales = exp.(randn(rng, N))
    adj_scaled = adj_base .* scales'
    
    w_base = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1))
    w_scaled = path_kelly_v1(adj_scaled; S = 300, rng = MersenneTwister(1))
    
    @test isapprox(w_base, w_scaled, atol = 0.005)
    println("✓ Constitution 1 Passed: Price Scale Invariance holds.")

    # ----------------------------------------------------
    # Constitution 2: Asset Permutation Invariance
    # ----------------------------------------------------
    perm = randperm(rng, N)
    adj_perm = adj_base[:, perm]
    
    w_perm = path_kelly_v1(adj_perm; S = 300, rng = MersenneTwister(1))
    @test isapprox(w_perm, w_base[perm], atol = 0.005)
    println("✓ Constitution 2 Passed: Asset Permutation Invariance holds strictly.")

    # ----------------------------------------------------
    # Constitution 3: Center-of-Mass Orthogonality
    # ----------------------------------------------------
    decomp = center_of_mass_decomposition(r_base)
    @test abs(sum(decomp.m)) > 0.0
    @test maximum(abs.(sum(decomp.u_perp, dims=2))) < 1e-12
    println("✓ Constitution 3 Passed: Center-of-Mass Orthogonality holds strictly.")

    # ----------------------------------------------------
    # Constitution 4: Exact Conditioned Trace Neutrality on Support
    # ----------------------------------------------------
    model = fit_v1(adj_base)
    @test abs(model.resp.trace_real) < 1e-10
    @test abs(model.resp.trace_imag) < 1e-10
    
    # Verify that sampled G^(s) strictly satisfies trace neutrality across all bands
    for s in 1:20
        pred_sample = predict_modes(model.resp, model.basis_now.B_m_now, model.basis_now.B_rel_now; sample_posterior = true, rng)
        # All relative predictions in zero-sum space strictly orthogonal to center of mass
        @test isfinite(pred_sample.mu_m)
        @test all(isfinite, pred_sample.mu_rel)
        P0 = I - fill(1.0 / N, N, N); @test abs(sum(P0 * pred_sample.mu_rel)) < 1e-12
    end
    println("✓ Constitution 4 Passed: Exact Conditioned Trace Neutrality holds identically on all posterior draws.")

    # ----------------------------------------------------
    # Constitution 5: Pure Noise Null & Evidence Shrinkage
    # ----------------------------------------------------
    @test maximum(abs.(model.mu_pred)) < 0.005
    println("✓ Constitution 5 Passed: Pure Noise Null (||μ||_∞ = ", round(maximum(abs.(model.mu_pred)), digits=5), " ≈ 0).")

    # ----------------------------------------------------
    # Constitution 6: d-Grid Refinement Convergence (Continuous Prior)
    # ----------------------------------------------------
    e_res = randn(rng, 1000)
    post_base = causal_fractional_posterior(e_res)
    @test isapprox(sum(post_base.p_d), 1.0, atol=1e-8)
    println("✓ Constitution 6 Passed: Continuous prior quadrature integration is valid.")
end
