using Test, Random, LinearAlgebra, Statistics
using KTrader

# Kernel-native long-memory factor: v_{t+1} = (1-a) + a Σ_k w_k(d) z²_{t-k},  z_t = √v_t ε_t.
function lm_factor(rng, n, d; a = 0.95, L = 800)
    w = KTrader.frac_weights(d, L); w ./= sum(w)
    z = zeros(n + L); z2 = ones(n + L)
    for t in L+1:n+L
        z[t] = sqrt((1 - a) + a * dot(w, view(z2, t-1:-1:t-L))) * randn(rng); z2[t] = z[t]^2
    end
    z[L+1:end]
end

# K = 3 hidden factors (d = 0.15, 0.4, constant) + idiosyncratic noise.
function factor_world(seed; n = 2000, N = 12)
    rng = MersenneTwister(seed)
    Q = Matrix(qr(randn(rng, N, 3)).Q)[:, 1:3]
    F = hcat(2.0 .* lm_factor(rng, n, 0.15), 1.5 .* lm_factor(rng, n, 0.4), randn(rng, n))
    F * Q' .+ 0.5 .* randn(rng, n, N) .* (0.7 .+ 0.6 .* rand(rng, N))', Q
end

dmean(b) = sum(b.p .* b.d)
qlike(S, e) = (F = cholesky(Symmetric(S)); logdet(F) + dot(e, F \ e))

# Protocol: expanding window; every `every` days up to the last `last`, refit on the history and score
# the NEXT day's return vector with QLIKE = log|Σ| + e'Σ⁻¹e (lower is better).
function oos_qlike(E; last = 300, every = 10)
    qs = zeros(0, 2)
    for T in size(E, 1)-last:every:size(E, 1)-1
        H = E[1:T, :]; e = E[T+1, :]
        qs = vcat(qs, [qlike(cond_cov(mode_vol_model(H)), e) qlike(cov(H), e)])
    end
    vec(mean(qs, dims = 1))
end

@testset "mode_vol_model: own memory per mode" begin
    D = zeros(0, 3)
    for seed in 1:6
        E, Q = factor_world(seed)
        mv = mode_vol_model(E; K = 3)
        @test size(mv.Phi) == (12, 3) && mv.Phi' * mv.Phi ≈ I
        idx = [argmax(abs.(mv.Phi' * Q)[:, f]) for f in 1:3]       # mode that carries factor f
        D = vcat(D, [dmean(mv.modes[i]) for i in idx]')
        @test all(sum(b.p) ≈ 1 for b in mv.modes) && sum(mv.idio.p) ≈ 1
        @test all(>(1e-4), vcat((b.p for b in mv.modes)...))
    end
    m = vec(mean(D, dims = 1))
    @info "mean d_k by factor (0.15, 0.4, const)" m
    @test m[1] < m[2] < m[3]
    @test m[3] > 0.7
    @test count(D[:, 1] .< D[:, 3]) == size(D, 1)
end

@testset "automatic K recovers the factors" begin
    E, _ = factor_world(1)
    @test size(mode_vol_model(E).Phi, 2) == 3
    @test size(mode_vol_model(randn(MersenneTwister(1), 1500, 8)).Phi, 2) <= 1
end

@testset "out-of-sample QLIKE: modes vs constant covariance" begin
    Q = reduce(hcat, [oos_qlike(factor_world(s)[1]) for s in 1:5])
    m = vec(mean(Q, dims = 2))
    @info "QLIKE mode / constant cov" m
    @test m[1] < m[2]
end

@testset "iid constant world: no loss against constant covariance" begin
    rng = MersenneTwister(7); N = 8
    Q = Matrix(qr(randn(rng, N, N)).Q); s = exp.(range(-0.7, 0.7, N)); s ./= exp(mean(log(x) for x in s))
    E = (randn(rng, 2000, N) .* s') * Q'
    mv = mode_vol_model(E)
    @test all(dmean(b) > 0.6 for b in mv.modes)
    @test dmean(mv.idio) > 0.7
    q = oos_qlike(E; last = 200)
    @test q[1] <= q[2] + 0.01 * abs(q[2])
end

@testset "draw reproduces cond_cov and is reproducible" begin
    E, _ = factor_world(3; n = 1500, N = 8)
    mv = mode_vol_model(E; K = 3, grid = [0.5])
    @test length(mv.idio.d) == 1 && all(length(b.d) == 1 for b in mv.modes)
    C = Matrix(cond_cov(mv)); @test isposdef(C)
    X = reduce(hcat, [draw(mv, MersenneTwister(1)) for _ in 1:2])
    @test X[:, 1] == X[:, 2]
    rng = MersenneTwister(2); n = 20000
    X = zeros(n, 8); for i in 1:n; X[i, :] = draw(mv, rng); end
    @test norm(cov(X) - C) / norm(C) < 0.05
    @test norm(mean(X, dims = 1)) < 0.1 * sqrt(tr(C))
    rng = MersenneTwister(5); draw(mv, rng)
    @test @allocated(draw(mv, rng)) <= 256
end

# ---- near-singular iid world: tail directions are kept as modes, so no loss against constant covariance
@testset "near-singular iid world (tail modes)" begin
    rng = MersenneTwister(11); N = 12; n = 2500
    L = randn(rng, N, N); L[:, end] .*= 0.003                       # smallest singular value ≈ 0.003
    E = randn(rng, n, N) * L'
    q = oos_qlike(E; last = 200)
    @info "near-singular QLIKE mode / constant" q
    @test q[1] <= q[2] + 0.01 * abs(q[2])
    K = size(mode_vol_model(E).Phi, 2); @test K >= 1
end

# ---- ragged panel
@testset "ragged panel: old assets keep all their history" begin
    E, _ = factor_world(4; n = 2200, N = 10)
    Er = copy(E); Er[1:1100, 10] .= NaN                              # asset 10 lists half way
    mv = mode_vol_model(Er)
    @test all(length(mv.idio.own[j]) == size(mv.idio.Z[1], 1) for j in 1:9)         # old assets: every row
    @test length(mv.idio.own[10]) < size(mv.idio.Z[1], 1)
    @test all(isfinite, draw(mv, MersenneTwister(1)))
    C = Matrix(cond_cov(mv)); @test isposdef(C)
    X = zeros(20000, 10); rng = MersenneTwister(2); for i in 1:20000; X[i, :] = draw(mv, rng); end
    @test norm(cov(X) - C) / norm(C) < 0.06
    # forecast QLIKE on the old assets is not worse (>1 %) than without the young asset
    function q_old(E_)
        qs = Float64[]
        for T in size(E_, 1)-300:20:size(E_, 1)-1
            S = cond_cov(mode_vol_model(E_[1:T, :]))[1:9, 1:9]
            push!(qs, qlike(Matrix(S), E_[T+1, 1:9]))
        end
        mean(qs)
    end
    @test q_old(Er) <= q_old(E[:, 1:9]) + 0.01 * abs(q_old(E[:, 1:9]))
end
