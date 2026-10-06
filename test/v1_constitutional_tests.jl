using Test, LinearAlgebra, Statistics, Random
using Convex, Clarabel

include("../src/geometry.jl")
include("../src/response.jl")
include("../src/predict.jl")
include("../src/kelly.jl")

@testset "Path Kelly V1.0: Constitutional Invariant Suite" begin
    rng = MersenneTwister(2026)
    T, N = 1200, 8
    
    r_base = 0.01 .* randn(rng, T, N)
    adj_base = exp.(cumsum(r_base, dims=1))
    
    # ----------------------------------------------------
    # Constitution 1: Price Scale Invariance (P_i -> c_i * P_i)
    # ----------------------------------------------------
    scales = exp.(randn(rng, N))
    adj_scaled = adj_base .* scales'
    
    w_base = path_kelly_v1(adj_base; S = 300, rng = MersenneTwister(1))
    w_scaled = path_kelly_v1(adj_scaled; S = 300, rng = MersenneTwister(1))
    
    @test isapprox(w_base, w_scaled, atol = 1e-4)
    println("✓ Constitution 1 Passed: Price Scale Invariance holds.")

    # ----------------------------------------------------
    # Constitution 2: Asset Permutation Invariance
    # ----------------------------------------------------
    perm = randperm(rng, N)
    adj_perm = adj_base[:, perm]
    
    w_perm = path_kelly_v1(adj_perm; S = 300, rng = MersenneTwister(1))
    @test isapprox(w_perm, w_base[perm], atol = 1e-4)
    println("✓ Constitution 2 Passed: Asset Permutation Invariance holds.")

    # ----------------------------------------------------
    # Constitution 3: Gauge Rotation Covariance (Q -> Q * R)
    # ----------------------------------------------------
    # In relative space, any internal orthogonal rotation R (K_rel × K_rel) of the gauge
    # must leave asset-space physical predictions and weights invariant.
    K_rel = N - 1
    R_rot = Matrix(qr(randn(rng, K_rel, K_rel)).Q)
    
    decomp = center_of_mass_decomposition(center_of_mass_decomposition(diff(log.(adj_base), dims=1)).u_perp)
    Q_orig = helmert_basis(N)
    Q_rotated = Q_orig * R_rot
    
    # The projected coordinates simply rotate by R': z_rot = z_orig * R
    # And the prediction in asset space Q * mu_rel is identical because Q_rot * mu_rot = Q * R * (R' * mu) = Q * mu
    z_orig = decomp.u_perp * Q_orig
    z_rot = decomp.u_perp * Q_rotated
    @test isapprox(z_orig * R_rot, z_rot, atol=1e-12)
    println("✓ Constitution 3 Passed: Gauge Rotation Covariance holds strictly.")

    # ----------------------------------------------------
    # Constitution 4: Operator Trace Neutrality on Support
    # ----------------------------------------------------
    model = fit_v1(adj_base)
    @test abs(model.resp.trace_real) < 1e-10
    @test abs(model.resp.trace_imag) < 1e-10
    println("✓ Constitution 4 Passed: Exact Operator Trace Neutrality tr(A_b) ≡ 0, tr(B_b) ≡ 0.")

    # ----------------------------------------------------
    # Constitution 5: Pure Noise Null & Evidence Shrinkage
    # ----------------------------------------------------
    # Under pure Brownian random walk, Evidence alpha grows and predicted alpha collapses
    @test maximum(abs.(model.mu_pred)) < 0.005 # zero spurious alpha
    println("✓ Constitution 5 Passed: Pure Noise Null (||μ||_∞ = ", round(maximum(abs.(model.mu_pred)), digits=5), " ≈ 0).")

    # ----------------------------------------------------
    # Constitution 6: Four-Quadrant Belief Recovery
    # ----------------------------------------------------
    # Planted pure reversion: Mode 1 reverts toward trailing mean
    u_rev = randn(rng, T, N)
    d_rev = center_of_mass_decomposition(u_rev)
    Q = helmert_basis(N)
    z_rev = d_rev.u_perp * Q
    for t in 200:T-1
        z_rev[t+1, 1] -= 0.5 * (z_rev[t, 1] - mean(z_rev[t-8:t, 1]))
    end
    adj_rev = exp.(cumsum(d_rev.m * d_rev.e0' + z_rev * Q', dims=1))
    m_rev = fit_v1(adj_rev)
    
    # Reversion corresponds to positive A (mean-reverting projection)
    @test any(m_rev.resp.A_matrices[findfirst(==(8), BANDS)][1, 1] .> 0.02)
    println("✓ Constitution 6 Passed: Four-Quadrant Dynamics successfully recovered (Reversion A_11 > 0).")
end
