# 诊断（裁决 7 轮）：testset 1 首日（mf seed=31 drift=0.003, t=270）
# 的 full posterior tail 证书形态——两分支的判据值对比。
module DA
    using Dates
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "market.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "..", "src", "gate0", "posterior.jl"))
end
using .DA: MarketFacts, signal_prices, domain_mode_basis, build_mode_problem,
    WARMUP, sufficient_stats, lambda_diag, log_evidence, log_prior_d035a,
    ruler
using Random, LinearAlgebra, Dates, Statistics

function mk_mf(N, T, seed; drift = 0.003)
    rng = MersenneTwister(seed)
    logp = cumsum(drift .+ 0.01 .* randn(rng, T, N); dims = 1)
    MarketFacts(collect(Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(T - 1))),
                ["A$j" for j in 1:N], exp.(logp), exp.(logp), trues(T, N))
end
mf = mk_mf(3, 280, 31)
t = 270
act = [1, 2, 3]
signal = signal_prices(mf)[:, act]
x_log = log.(signal)
f_first = [findfirst(view(mf.observed, 1:t, j)) for j in act]
r = Matrix{Float64}(undef, t, 3)
r[1, :] .= NaN
r[2:t, :] .= x_log[2:t, :] .- x_log[1:t-1, :]
obs_ret = isfinite.(r)
s1 = vec(ruler(x_log, Int[i for i in f_first])[:, 1])
E = domain_mode_basis(3)
mp = build_mode_problem(r, Matrix{Bool}(obs_ret), s1, E; dc = true)
rows = [s for s in WARMUP:(t-1) if any(view(obs_ret, s + 1, :))]
X_tr = mp.X[rows, :]
Y_tr = mp.Y[rows, :]
st = sufficient_stats(X_tr, Y_tr)
has_a0 = true
logf(u0, up) = log_evidence(st, lambda_diag(43, u0, up, has_a0)) +
               log_prior_d035a(u0, up)
function coarse_scan(logf)
    grid = range(-12.0, 12.0; length = 21)
    m0, mp2, mval = 0.0, 0.0, -Inf
    for u0 in grid, up in grid
        v = logf(u0, up)
        if v > mval
            mval = v; m0 = u0; mp2 = up
        end
    end
    (m0, mp2, mval)
end
m0, mp2, mval = coarse_scan(logf)
println("full-post mode = ($m0, $mp2), mval = ", mval)
u_span = 10.0
u0lo, u0hi = m0 - u_span, m0 + u_span
uplo, uphi = mp2 - u_span, mp2 + u_span
inner = [(u0lo + i * (u0hi - u0lo) / 5, uplo + j * (uphi - uplo) / 5)
         for i in 1:4, j in 1:4]
m_ref = maximum(logf(p[1], p[2]) for p in inner)
area = (u0hi - u0lo) * (uphi - uplo)
bpts = [(u0lo, uplo), (u0hi, uplo), (u0lo, uphi), (u0hi, uphi),
        ((u0lo+u0hi)/2, uplo), ((u0lo+u0hi)/2, uphi),
        (u0lo, (uplo+uphi)/2), (u0hi, (uplo+uphi)/2)]
plateau = maximum(logf(p[1], p[2]) for p in bpts)
println("m_ref = ", m_ref, "  plateau = ", plateau,
        "  gap = ", m_ref - plateau, "  (weak if < 8)")
r0 = (u0lo + u0hi) / 2
rp = (uplo + uphi) / 2
tail_rel = 10 * exp(-u_span)
for p in bpts
    v = logf(p[1], p[2])
    crit_old = m_ref + log(tail_rel) + log(area)
    d = max(0.0, p[1] - r0) + max(0.0, p[2] - rp)
    crit_weak = plateau - 2 * d + 5.0
    println("boundary ", p, " -> ", round(v, digits=3),
            "  old-crit=", round(crit_old, digits=3),
            v <= crit_old ? " [OK]" : " [RED]",
            "  weak-crit=", round(crit_weak, digits=3),
            v <= crit_weak ? " [OK]" : " [RED]")
end
