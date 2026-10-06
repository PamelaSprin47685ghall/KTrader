# Does the DECISION behave correctly because of the belief layer?  Price supplies evidence,
# (ρ,θ) are beliefs: θ is never collapsed, ρ→0 with no evidence is a valid outcome, neutrality is
# part of the prior, and uncertainty reaches Kelly only through the posterior predictive.
# Only the public API is used; every tolerance below is backed by a measurement printed in the report
# of the commit that introduced it (the numbers are repeated next to each threshold).
using Test, Random, LinearAlgebra, Statistics
using KTrader

const S_SCEN  = 300

# ---- synthetic worlds -----------------------------------------------------------------------
# One hidden factor f_t loads on N assets with an alternating-sign pattern (so the response matters
# cross-sectionally: it must pick the right *side* of the book, not just a level).
#   :trend   f_t = φ f_{t-1} + σ_f ε                       (AR(+): moves continue)
#   :revert  f_t = −κ (x_{t-1} − mean(x_{t-20:t-1})) + σ_f ε  (OU around the trailing centre)
# x = cumsum f. The last K days carry an extra drift `sgn·kick` so the decision-time state is far
# from equilibrium (otherwise the planted next-day mean is ≈ 0 and there is nothing to decide).
# Returns prices, loadings, and the planted next-day mean of each asset.
function factor_world(kind; T = 4000, N = 8, seed = 1, φ = 0.25, κ = 0.15, σf = 0.01, σe = 0.004,
                      kick = 0.015, K = 15, sgn = 1.0)
    rng = MersenneTwister(seed); f = zeros(T); x = zeros(T); r = zeros(T, N)
    load = [isodd(i) ? 1.0 : -1.0 for i in 1:N] .* (1 .+ 0.2 .* randn(rng, N))
    mean_f(t) = kind == :trend ? φ * f[t-1] : -κ * (x[t-1] - (t > 21 ? mean(x[t-20:t-1]) : 0.0))
    for t in 2:T
        f[t] = mean_f(t) + σf * randn(rng) + (t > T - K ? sgn * kick : 0.0)
        x[t] = x[t-1] + f[t]
        r[t, :] = load .* f[t] .+ σe .* randn(rng, N)
    end
    # next-day planted mean: mean_f at t = T+1
    mnext = kind == :trend ? φ * f[T] : -κ * (x[T] - mean(x[T-19:T]))
    exp.(cumsum(r, dims = 1)), load, load .* mnext
end

# Hidden factor with a smooth planted conditional mean m_t = sgn·A·sin(2πt/Pd + ψ) (ψ so that the
# last row sits at the crest): over every window shorter than Pd the position q and the momentum p
# of the macro mode are almost collinear, so history alone cannot say "reverts" from "trends".
# `kick` adds a transient over the last K days that breaks the collinearity at decision time
# (q flips sign, the slow p does not).
function smooth_world(; T = 3000, N = 8, seed = 1, A = 0.0015, Pd = 400, σe = 0.008, K = 0, kick = 0.0, sgn = 1.0)
    rng = MersenneTwister(seed)
    load = [isodd(i) ? 1.0 : -1.0 for i in 1:N] .* (1 .+ 0.2 .* randn(rng, N))
    m = sgn * A .* sin.(2π .* (1:T+1) ./ Pd .+ (π / 2 - 2π * (T + 1) / Pd))
    f = m[1:T]; f[T-K+1:T] .+= kick
    exp.(cumsum(load' .* f .+ σe .* randn(rng, T, N), dims = 1)), load, load .* m[T+1]
end

white_noise(T, N, seed) = exp.(cumsum(0.01 .* randn(MersenneTwister(seed), T, N), dims = 1))

l1(a, b) = sum(abs, a .- b)
pk(P; seed = 1, kw...) = path_kelly(P; S = S_SCEN, rng = MersenneTwister(seed), kw...)
tilt_w(P; kw...) = pk(P; response = true, kw...) .- pk(P; response = false, kw...)    # decision tilt due to the response
# response part of the posterior-predictive mean (exact: scenario means are moment-matched)
function tilt_mean(fit)
    m(r) = vec(mean(predict(fit; S = 20, rng = MersenneTwister(1), response = r, volmodel = false), dims = 1))
    m(true) .- m(false)
end
se(v) = std(v) / sqrt(length(v))
# best-evidence (smallest α) macro mode
best_macro(fit; draws = 2000) = (mm = [m for m in theta_posterior(fit; draws) if m.macro_]; mm[argmin([m.alpha for m in mm])])

@testset "Belief → decision" begin

@testset "(a) planted trend vs planted reversion: the tilt follows the law, with opposite signs" begin
    # same final price move (+kick on the factor) in both worlds: trend says 'keep going' (tilt ∝ +load),
    # reversion says 'come back' (tilt ∝ −load). sgn alternates so no sign convention is baked in.
    res = Dict(k => (sgn = Float64[], mean = Float64[]) for k in (:trend, :revert))
    for kind in (:trend, :revert), seed in 1:12, sgn in (1.0, -1.0)
        P, load, mn = factor_world(kind; seed, sgn)
        t = tilt_w(P)
        push!(res[kind].sgn, sgn * dot(t, load)); push!(res[kind].mean, dot(t, mn) / norm(mn))
    end
    for kind in (:trend, :revert)
        s, m = res[kind]
        println("  (a) $kind: mean sgn·⟨tilt,load⟩ = ", round(mean(s), digits = 3), " ± ", round(se(s), digits = 3),
                "; right side in ", count(x -> x * (kind == :trend ? 1 : -1) > 0, s), "/", length(s),
                "; mean ⟨tilt, planted mean⟩/|mean| = ", round(mean(m), digits = 3), " (", count(>(0), m), "/", length(m), " > 0)")
    end
    @test mean(res[:trend].sgn) > 0 && mean(res[:revert].sgn) < 0
    @test count(>(0), res[:trend].sgn) >= 18 && count(<(0), res[:revert].sgn) >= 18        # ≥ 75 % of the 24 worlds
    # the planted law is what decides the direction: the tilt lines up with the planted next-day mean
    @test all(>(0), (mean(res[:trend].mean), mean(res[:revert].mean)))
    @test count(>(0), res[:trend].mean) >= 18 && count(>(0), res[:revert].mean) >= 18
end

@testset "(b) white noise: no evidence ⇒ response and baseline decisions coincide up to the Monte-Carlo floor" begin
    N, T = 8, 3000
    d_resp, d_floor = Float64[], Float64[]
    for seed in 1:10
        P = white_noise(T, N, 100 + seed)
        w1, w0 = pk(P; seed = 1, response = true), pk(P; seed = 1, response = false)
        push!(d_resp, l1(w1, w0))
        push!(d_floor, l1(w0, pk(P; seed = 2, response = false)))          # same model, other RNG: pure Monte-Carlo
    end
    println("  (b) white noise L1(response vs baseline) = ", round(mean(d_resp), digits = 3), " ± ", round(se(d_resp), digits = 3),
            ";  Monte-Carlo floor L1(baseline seed 1 vs 2) = ", round(mean(d_floor), digits = 3), " ± ", round(se(d_floor), digits = 3))
    @test mean(d_resp) < 2 * mean(d_floor) + 0.2
    # …whereas the planted worlds of (a) sit far above that floor (L1 ≈ 1.0–1.4)
end

@testset "(c) evidence monotonicity: more rows ⇒ more conviction along the planted direction" begin
    ns = (1200, 2400, 4800)
    A = Dict(n => Float64[] for n in ns); L = Dict(n => Float64[] for n in ns); R = Dict(n => Float64[] for n in ns)
    for seed in 1:32
        P, load, mn = factor_world(:trend; T = last(ns), seed, φ = 0.1, σe = 0.004, kick = 0.01, sgn = isodd(seed) ? 1.0 : -1.0)
        for n in ns
            Q = P[end-n+1:end, :]
            t = tilt_w(Q)
            push!(A[n], dot(t, mn) / norm(mn)); push!(L[n], sum(abs, t))
            push!(R[n], dot(tilt_mean(fit_response(Q)), mn) / norm(mn) * 252)
        end
    end
    for n in ns
        println("  (c) n=$n: aligned weight tilt ", round(mean(A[n]), digits = 3), " ± ", round(se(A[n]), digits = 3),
                "; aligned response mean (ann.) ", round(mean(R[n]), digits = 3), " ± ", round(se(R[n]), digits = 3),
                "; L1 tilt ", round(mean(L[n]), digits = 3), " ± ", round(se(L[n]), digits = 3))
    end
    d = A[ns[end]] .- A[ns[1]]
    println("  (c) paired aligned tilt(4800) − tilt(1200) = ", round(mean(d), digits = 3), " ± ", round(se(d), digits = 3))
    # L1 saturates at the long-only corners (Σ|Δw| ∈ [1, 2] as soon as the argmax asset changes), so the
    # growth with n is tested on the part of the tilt that lies along the planted direction.
    @test mean(A[ns[1]]) < mean(A[ns[2]]) < mean(A[ns[3]])
    @test mean(d) > 2 * se(d)
end

@testset "(d) ambiguity (q≈p collinear history) reduces conviction when the decision state splits the hypotheses" begin
    # Smooth hidden factor: over every window shorter than its period, −q and p of the macro mode carry the same
    # information, so "reverts" and "trends" are not separable (θ stays spread over the two opposite peaks).
    # Control: AR(+) factor with the same planted signal variance (φ = A√2/σ_f), where they are separable.
    # Conviction = |posterior-mean response| / sd of the response over posterior draws (z), along the book
    # direction `load`; all numbers come from the posterior itself.
    A, σe, K, kick = 0.0015, 0.008, 20, 0.006
    φ = A * sqrt(2) / 0.01
    function z_along(f, load)
        U = cholesky(f.post.A).U; rng = MersenneTwister(1); u = load ./ norm(load)
        proj(β) = dot(f.s1 .* (f.D * (f.Z * β)), u) * 252
        pr = [proj(f.post.m .+ U \ randn(rng, size(U, 1))) for _ in 1:1000]
        abs(proj(f.post.m)) / std(pr)
    end
    evid(f) = [m for m in theta_posterior(f; draws = 2000) if m.macro_ && m.alpha < 100]      # modes the data speaks about
    R1a, R1u, za, zs, zu, ma, mu = (Float64[] for _ in 1:7)
    for seed in 1:16
        sg = isodd(seed) ? 1.0 : -1.0
        Pa, la, mna = smooth_world(; seed, A, σe, sgn = sg)
        Pk, lk, _ = smooth_world(; seed, A, σe, sgn = sg, K, kick = -kick * sg)
        Pu, lu, mnu = factor_world(:trend; T = 3000, seed, φ, σe, kick = 0.0)
        fa, fk, fu = fit_response(Pa), fit_response(Pk), fit_response(Pu)
        ea, eu = evid(fa), evid(fu)
        isempty(ea) || push!(R1a, minimum(m.R1 for m in ea)); isempty(eu) || push!(R1u, minimum(m.R1 for m in eu))
        push!(za, z_along(fa, la)); push!(zs, z_along(fk, lk)); push!(zu, z_along(fu, lu))
        push!(ma, norm(mna) * 252); push!(mu, norm(mnu) * 252)
    end
    println("  (d) planted |next-day mean| (ann.): ambiguous world ", round(mean(ma), digits = 2), ", unambiguous ", round(mean(mu), digits = 2))
    println("  (d) least-concentrated evidenced macro mode, R1: ambiguous ", round(mean(R1a), digits = 3), " (n=", length(R1a),
            "), unambiguous ", round(mean(R1u), digits = 3), " (n=", length(R1u), ")")
    println("  (d) conviction z: ambiguous/collinear state ", round(mean(za), digits = 2), ", ambiguous/state splits q,p ", round(mean(zs), digits = 2),
            ", unambiguous ", round(mean(zu), digits = 2))
    # KNOWN GAP (measured after the cross-fitting rewrite: R1 = 0.999 ambiguous vs 0.998 unambiguous; posterior
    # anisotropy of (a,b) is 2.9 vs 10.9, i.e. the other way round): the θ posterior does NOT widen where the data
    # cannot separate "reverts" from "trends". The theory asks for it; the decision-level properties below hold.
    @test_broken mean(R1a) < mean(R1u) - 0.05
    @test mean(zs) < 0.6 * mean(zu)                     # …so the response tilt at a state that needs the choice is less certain
    # at a state where −q and p agree, nothing needs choosing and the same fit is confident
    @test mean(za) > mean(zu)
end

@testset "(e) neutrality is a property of every posterior draw" begin
    rng = MersenneTwister(6)
    f = 0.01 .* randn(rng, 3000, 2); ld = randn(rng, 2, 8)
    fit = fit_response(exp.(cumsum(f * ld .+ 0.004 .* randn(rng, 3000, 8), dims = 1)))
    idx = [l.k >= 2 for l in fit.labels]
    @test any(idx)
    U = cholesky(fit.post.A).U
    B = fit.Z * (fit.post.m .+ U \ randn(MersenneTwister(1), size(U, 1), 400))          # 400 draws of β=(a,b)
    na, nb = sum(B[2 .* findall(idx) .- 1, :]; dims = 1), sum(B[2 .* findall(idx), :]; dims = 1)
    println("  (e) max |Σa|, |Σb| over 400 draws (non-macro modes) = ", maximum(abs, na), ", ", maximum(abs, nb))
    @test maximum(abs, na) < 1e-8 && maximum(abs, nb) < 1e-8
    nm = filter(m -> !m.macro_, theta_posterior(fit))
    @test isapprox(sum(m.a for m in nm), 0; atol = 1e-8) && isapprox(sum(m.b for m in nm), 0; atol = 1e-8)
end

@testset "(f) sign symmetry: reflecting log prices flips the tilt and keeps its size" begin
    rs, rm, w1s, w2s, dir = Float64[], Float64[], Float64[], Float64[], Float64[]
    for seed in 1:12
        P, load, _ = factor_world(:trend; seed)
        Q = 1 ./ P                                        # x → −x
        fp, fq = fit_response(P), fit_response(Q)
        tp, tq = tilt_mean(fp), tilt_mean(fq)
        push!(rs, dot(tp, tq) / (norm(tp) * norm(tq))); push!(rm, norm(tq) / norm(tp))
        wp, wq = tilt_w(P), tilt_w(Q)
        push!(w1s, sum(abs, wp)); push!(w2s, sum(abs, wq))
        any(!iszero, wp) && any(!iszero, wq) && push!(dir, sign(dot(wp, load)) * sign(dot(wq, load)))
    end
    println("  (f) response-mean cosine(P, reflected P) = ", round(minimum(rs), digits = 6), " … ", round(maximum(rs), digits = 6),
            "; |tilt| ratio ", round(minimum(rm), digits = 6), " … ", round(maximum(rm), digits = 6),
            "; weight tilt L1 ", round(mean(w1s), digits = 3), " vs reflected ", round(mean(w2s), digits = 3),
            "; weight tilts on opposite sides in ", count(<(0), dir), "/", length(dir))
    @test all(<(-0.999), rs) && all(r -> 0.999 < r < 1.001, rm)
    # Kelly on gross returns is not exactly reflection-symmetric (convexity of exp, long-only corners), so the
    # weight tilts only need to be of the same order; the belief layer itself is exact (cosine −1, ratio 1).
    @test 0.5 < mean(w1s) / mean(w2s) < 2
    @test count(<(0), dir) >= 0.9 * length(dir)
end

end
