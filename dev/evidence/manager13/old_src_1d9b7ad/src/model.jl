# Path Kelly model. Pure, causal function of the price history `adj` (T×N total-return
# prices, NaN before an asset's listing; last row = decision-time price, in live trading the
# preview close).
#
# Market theory (fixed law)
#   ruler      per-asset fractal scale s_i(τ)=c_i τ^{H_i}  (power-law fit of RMS increments)
#   modes      per scale band τ: eigenvectors of the covariance of ruler-normalised increments
#              u_i(t,τ); mode 1 of each band is the macro mode
#   state      per mode, two OBSERVED coordinates: q = deviation of the price from the trailing-τ
#              centre, p = slope of that centre (low-frequency trend), both in ruler units
#   response   next-day return ẑ_{t+1} = Σ_modes φ_k (a_k(−q_k) + b_k p_k) + noise. (q,p) are
#              observation coordinates; (ρ_k,θ_k) are BELIEF parameters:  a_k=ρ_k cosθ_k is the
#              mean-reversion axis (the deviation q reverts), b_k=ρ_k sinθ_k the belief axis
#              "confirm the move (b>0) vs front-run it (b<0)" on the observed centre slope p.
# Epistemic layer (belongs to the theory, not to the numerics)
#   * response coefficients carry an evidence-maximised Gaussian posterior; the neutrality
#     Σ_{non-macro k} a_k = Σ_{non-macro k} b_k = 0 lives on the prior (constraint subspace).
#   * the geometry (modes, noise covariance) is estimated on the most recent window of G rows;
#     the response evidence uses only returns before that window (disjoint, see GEOM_FRAC).
#     Estimating the eigenvectors on the very rows they are then regressed on is selection on
#     the response: on pure noise P(|t|>1.96)=0.68 instead of 0.05.
#   * geometry uncertainty is propagated: GEOM_DRAWS stationary-block-bootstrap draws of the
#     geometry window, each with its own response posterior; the predictive mixes over draws.
#   * a mode is identifiable only if its subspace is stable across those draws (MODE_STABILITY);
#     no iid / Marchenko–Pastur assumption is used.
#   * sparsity: a mode's response is switched off unless its evidence beats "no response" by
#     BF_MIN·M (see `bayes_ard`); the evidence is a likelihood ratio, not a posterior odds.
#   * ragged panels: every asset keeps all its own history. Pairwise-available covariances,
#     exact Gaussian likelihood on the observed sub-panel of every row, and posteriors that widen
#     with less data replace a hard common window.
# Decision: exact Kelly on posterior-predictive scenarios, fully invested (Σw=1, w≥0,
# leverage 1, no cash asset).
# Numerics (S, solver, FFT) are engineering only.

const TAUS  = 2 .^ (0:8)      # ruler fit horizons
const BANDS = 2 .^ (1:7)      # mode bands
const BANDCOL = [findfirst(==(τ), TAUS) for τ in BANDS]
const WARMUP = 2 * maximum(BANDS)         # rows before an asset has every band state (c(t−τ) needs 2τ)

"""
Fit-geometry constants (theory-layer choices, stated):
  GEOM_FRAC   share of the rows given to the geometry window (the most recent
              returns). Disjointness from the response evidence is what removes the leakage;
              1/3 leaves the scarce resource (response evidence) the larger part, while the
              geometry (a few well-separated eigenvectors) needs far fewer rows than a regression.
  GEOM_MIN    geometry window floor: the largest band needs two windows (2·max(BANDS)).
  REG_MIN     evidence floor (rows).
  MINROWS     the numerical minimum fit_response needs from ONE asset: WARMUP rows for its states +
              GEOM_MIN geometry rows + REG_MIN evidence rows. It is not a statistical threshold:
              weak evidence is expressed by the posterior (ARD, drift hierarchy), not by exclusion.
  GEOM_DRAWS  stationary-bootstrap geometry draws mixed in the predictive.
  MODE_STABILITY  a band keeps mode k only if median over draws of cos(largest principal angle)
              between span(φ_1..φ_k) and its bootstrap counterpart ≥ this (angle ≲ 26°).
  EIGFLOOR    PSD repair of the pairwise-available noise covariance (relative eigenvalue floor).
"""
const GEOM_FRAC, GEOM_MIN, GEOM_MAX, REG_MIN = 1 / 3, 2 * maximum(BANDS), 1500, 64
const MINROWS = WARMUP + GEOM_MIN + REG_MIN
const GEOM_DRAWS, MODE_STABILITY, EIGFLOOR = 8, 0.9, 1e-3

"""
Politis–Romano stationary bootstrap: indices of a series of length n made of circular blocks with
geometric lengths of mean `L`, so the resample keeps the serial dependence inside blocks (an iid
row resample would destroy exactly the path memory the theory is about).
"""
function stationary_bootstrap(rng, n, L)
    idx = Vector{Int}(undef, n); i = rand(rng, 1:n)
    for k in 1:n
        idx[k] = i
        i = rand(rng) < 1 / L ? rand(rng, 1:n) : mod1(i + 1, n)
    end
    idx
end

"Multiplicity of every row in a stationary-bootstrap resample (moments of the resample are weighted moments)."
function boot_weights(rng, n, L)
    w = zeros(n)
    for i in stationary_bootstrap(rng, n, L); w[i] += 1; end
    w
end

blocklen(τ, n) = min(max(2τ, 16), n ÷ 4)      # a τ-increment is serially dependent over ≈τ rows

"PSD repair of a pairwise-available covariance: eigenvalues floored at EIGFLOOR × their mean."
function repair(Σ)
    λ, V = eigen(Symmetric(Σ))
    V * Diagonal(max.(λ, EIGFLOOR * mean(λ))) * V'
end

"""
Pairwise-available weighted covariance of the columns of X0 (zero where the entry is missing, M the
0/1 availability, `w` row weights): each pair (i,j) uses the rows where both are observed.
A pair with no weighted common row (a block-bootstrap resample can miss every row of a young asset)
is not identifiable in that resample: its entry is taken from `fallback` (the covariance of the
un-resampled data) instead of being the 0/0 = NaN, and without a fallback it is an error.
"""
function wcov(X0::AbstractMatrix{Float64}, M::AbstractMatrix{Float64}, w::AbstractVector{Float64}; fallback = nothing)
    n, N = size(X0)
    Mw = M .* w
    Xw = X0 .* w
    W = M' * Mw
    A = X0' * Mw
    C = X0' * Xw
    @inbounds for j in 1:N, i in 1:N
        Wij = W[i, j]
        if Wij <= 0.0
            fallback === nothing && error("a pair of columns has no common weighted observation and no fallback was given")
            C[i, j] = fallback[i, j]
        else
            C[i, j] = C[i, j] / Wij - (A[i, j] / Wij) * (A[j, i] / Wij)
        end
    end
    C
end

const RULER_BLOCK = 64          # block mean length of the ruler bootstrap (≈ the middle fit horizons)

"""
Per-asset fractal ruler s_i(τ) on TAUS: power-law fit of the weighted RMS τ-increments over ALL of the
asset's own rows and the horizons they support (4τ ≤ own rows). `w`: row weights (length T, index =
end row of the increment). Not a response-selected quantity (a scale, not a direction), so it uses the
whole history. N×length(TAUS).
"""
function ruler(x::AbstractMatrix{Float64}, f::AbstractVector{<:Integer}, w::AbstractVector{Float64})
    T, N = size(x)
    Ntaus = length(TAUS)
    out = Matrix{Float64}(undef, N, Ntaus)
    lτ = log.(TAUS)
    L_buf = zeros(Float64, Ntaus)
    for j in 1:N
        n_use = 0; sum_l = 0.0; sum_log_tau = 0.0; fj = f[j]
        for a in 1:Ntaus
            τ = TAUS[a]
            if 4τ <= T - fj + 1
                n_use += 1
                acc = 0.0; ws = 0.0
                @inbounds @simd for t in (fj + τ):T
                    d = x[t, j] - x[t - τ, j]
                    wt = w[t]
                    acc += wt * d * d
                    ws += wt
                end
                val = log(sqrt(acc / ws))
                L_buf[n_use] = val
                sum_l += val
                sum_log_tau += lτ[a]
            end
        end
        m_l = sum_l / n_use
        m_tau = sum_log_tau / n_use
        num = 0.0; den = 0.0
        for a in 1:n_use
            d_tau = lτ[a] - m_tau
            num += (L_buf[a] - m_l) * d_tau
            den += d_tau * d_tau
        end
        H = num / den
        for a in 1:Ntaus
            out[j, a] = exp(m_l + H * (lτ[a] - m_tau))
        end
    end
    out
end

natural_modes(C) = eigen(Symmetric(C)).vectors[:, end:-1:1]

"""
Geometry window: the last G rows. Raw (un-normalised) increments of every band and of the 1-day
return, with their availability masks (an asset contributes from its own listing on).
"""
function geodata(x, f, G)
    T, N = size(x); xg = ifelse.(isfinite.(view(x, T-G+1:T, :)), view(x, T-G+1:T, :), 0.0)
    ok = [t >= f[j] - (T - G) for t in 1:G, j in 1:N]
    inc(τ) = (t = τ+1:G; M = ok[t, :] .& ok[t .- τ, :]; (U0 = ifelse.(M, xg[t, :] .- xg[t .- τ, :], 0.0), M = Float64.(M)))
    (; G, bands = [inc(τ) for τ in BANDS], Y = inc(1))
end

"""
One geometry: ruler, modes of every band and noise covariance of the next-day returns (PSD-repaired),
for the window `gd` and the full history `x`. `boot=false`: the data themselves; `boot=true`: one
stationary block-bootstrap resample, the block length matched to each band's serial dependence
(2τ) and to the ruler's (RULER_BLOCK).
"""
function geometry(x, f, gd, rng, boot)
    T, G = size(x, 1), gd.G
    wt(n, L) = boot ? boot_weights(rng, n, L) : ones(n)
    # un-resampled covariances: the value of a pair the resample cannot identify
    fb(d, w) = boot ? wcov(d.U0, d.M, w) : nothing
    s = ruler(x, f, wt(T, RULER_BLOCK)); s1 = s[:, 1]
    w1 = wt(G, blocklen(1, G))
    Σ = wcov(gd.Y.U0, gd.Y.M, w1[2:G]; fallback = fb(gd.Y, ones(G - 1))) ./ (s1 * s1')
    Σ = repair(Σ)
    Φ = map(enumerate(BANDS)) do (b, τ)
        sb = s[:, BANDCOL[b]]
        d = gd.bands[b]
        natural_modes(wcov(d.U0, d.M, wt(G, blocklen(τ, G))[τ+1:G]; fallback = fb(d, ones(G - τ))) ./ (sb * sb'))
    end
    (; s, Σ, Φ)
end

"""
Identifiable modes per band: the largest K such that for every k ≤ K the subspace span(φ_1..φ_k)
of the reference geometry is reproduced by the block-bootstrap geometries (median over draws of the
cosine of the largest principal angle ≥ MODE_STABILITY). Unlike a Marchenko–Pastur edge (iid entries,
effective sample n/τ) this uses the dependence actually present in the data. The macro mode is
always kept (its response is then judged by the evidence).
"""
function stable_modes(ref, ds)
    N = size(ref.Φ[1], 1)
    map(eachindex(BANDS)) do b
        K = 1
        while K < N - 1
            c = median(minimum(svdvals(ref.Φ[b][:, 1:K+1]' * d.Φ[b][:, 1:K+1])) for d in ds)
            c >= MODE_STABILITY || break
            K += 1
        end
        K
    end
end


"""
Grouped ARD (MacKay) on the constraint subspace. Coefficients are pairs (a_k,b_k), one pair per
mode k. Belief about mode k is a rotation-invariant Gaussian  (a_k,b_k) ~ N(0, α_k⁻¹ I₂), i.e.
θ_k uniform a priori and ρ_k Rayleigh with scale α_k^{-1/2}; the neutrality constraints are
imposed on the prior itself (β = Zγ, Z orthonormal, so the conditional prior on γ has precision
P = Zᵀ diag(α) Z). α_k is chosen by evidence maximisation: a mode with no evidence gets
α_k → ∞, i.e. ρ_k → 0, while θ_k is never collapsed to a point.
Stationarity of the evidence in α_k:  tr[(P⁻¹ − A⁻¹) Z_kᵀZ_k] = ‖β_k‖²,  A = P + b·ZᵀGZ, so the
update is α_k ← γ_k/‖β_k‖² with γ_k = α_k tr[(P⁻¹ − A⁻¹) Z_kᵀZ_k] (effective number of
parameters). b (noise precision) is updated with the usual (n − Σγ_k)/RSS. Sufficient
statistics G=X'X, c=X'y, yy=y'y over n observations.
"""
const ALPHA_MIN, ALPHA_MAX = 1e-6, 1e7

"""
Sparsity (Occam) gate of the ARD (epistemic layer). Evidence maximisation alone keeps a mode alive as
soon as its two-coefficient Wald statistic exceeds ≈2, which invents an effect in a large share of
pure-noise modes (measured: 26 % of macro modes on white noise even with disjoint geometry), because
the maximised evidence is a likelihood-ratio, not a posterior. The prior is therefore sparse: a priori
a mode carries a response with odds 1/M (M modes searched). A mode stays alive only if its maximised
marginal likelihood beats the same model with that mode switched off (α_k→∞) by `BF_MIN · M`
(BF_MIN = 10: Jeffreys' "strong" evidence, after the search over M modes); a mode below that
(and every other unsupported mode) is switched off (α_k = ALPHA_MAX, ρ_k → 0) and the rest refitted. Planted responses with ρ ≳ 0.3 pass
by orders of magnitude.
"""
const BF_MIN = 10.0

function bayes_ard(G, c, yy, n, Z; iters = 500, tol = 1e-4)
    q = size(Z, 2); P = size(Z, 1); M = P ÷ 2
    Gz, cz = Symmetric(Z' * G * Z), Z' * c
    Zg = [Z[2g-1:2g, :] for g in 1:M]
    precision(α) = Symmetric(Z' * (Diagonal(repeat(α, inner = 2)) * Z))
    # log marginal likelihood (up to terms independent of α) at precision α, noise precision b
    function logev(α, b)
        Pm = precision(α); A = Symmetric(Pm + b * Gz); m = b * (A \ cz)
        0.5 * (logdet(Pm) - logdet(A)) - 0.5 * b * (yy - 2dot(m, cz) + dot(m, Gz * m)) - 0.5 * dot(m, Pm * m)
    end
    α = fill(1.0, M); b = n / yy
    dead = falses(M)
    while true
        for it in 1:iters
            Pm = precision(α); A = Symmetric(Pm + b * Gz)
            m = b * (A \ cz)
            Δ = inv(Pm) - inv(A)
            βb = Z * m
            γ = [α[g] * tr(Zg[g] * Δ * Zg[g]') for g in 1:M]
            αn = clamp.(γ ./ max.(vec(sum(abs2, reshape(βb, 2, M); dims = 1)), 1e-300), ALPHA_MIN, ALPHA_MAX)
            αn[dead] .= ALPHA_MAX
            b = max(n - sum(γ), 1.0) / max(yy - 2dot(m, cz) + dot(m, Gz * m), eps())
            done = maximum(abs.(log.(αn) .- log.(α))) < tol
            α = αn
            done && break
        end
        gain = fill(Inf, M)                                  # log Bayes factor of each live mode vs. switched off
        for g in findall(.!dead)
            α0 = copy(α); α0[g] = ALPHA_MAX
            gain[g] = logev(α, b) - logev(α0, b)
        end
        weak = gain .< log(BF_MIN * M)
        any(weak) || break
        dead .|= weak; α[weak] .= ALPHA_MAX                  # switch the unsupported modes off, refit the rest
    end
    Pm = precision(α)
    A = Symmetric(Pm + b * Gz); m = b * (A \ cz)
    (m = m, A = A, alpha = α, noise_precision = b)
end

"""
Newey–West (Bartlett) long-run variance of each row of R (N×n): Var(mean) ≈ lrv/n. Replaces the
iid σ²/n, which is wrong under the path memory the theory insists on.
"""
function lrv(R)
    N, n = size(R)
    L = floor(Int, 4 * (n / 100)^(2 / 9))
    out = zeros(N)
    for i in 1:N
        x = R[i, :] .- mean(R[i, :])
        v = dot(x, x) / n
        for k in 1:L
            v += 2 * (1 - k / (L + 1)) * dot(view(x, 1+k:n), view(x, 1:n-k)) / n
        end
        out[i] = max(v, 0.1 * dot(x, x) / n)           # keep positive under strong negative dependence
    end
    out
end

"""
Hierarchical drift  m_i ~ N(μ₀, τ²),  x̄_i | m_i ~ N(m_i, se_i²)  with τ integrated out:
τ = c·median(se) on the grid `TAUGRID` (c=0 is "no cross-sectional signal"), uniform prior on
the grid cells (scale-free: it is a prior on c, the signal-to-noise ratio), flat prior on μ₀.
The weak evidence a finite sample gives for τ>0 is therefore carried as posterior mass, not
collapsed to a point estimate. Fields: p (cell posterior), m/sd (G×N, cell-conditional posterior
mean and sd of every m_i), mu0sd (sd of the common level per cell), mean (overall posterior mean).
"""
const TAUGRID = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

struct DriftPost
    p::Vector{Float64}
    m::Matrix{Float64}
    sd::Matrix{Float64}
    mu0sd::Vector{Float64}
    mean::Vector{Float64}
end

function shrink_drift(xbar, se; grid = TAUGRID)
    N = length(xbar); G = length(grid); s0 = median(se)
    ll = zeros(G); m = zeros(G, N); sd = zeros(G, N); mu0sd = zeros(G)
    for (g, c) in enumerate(grid)
        τ2 = (c * s0)^2; v = se .^ 2 .+ τ2; w = 1 ./ v
        μ0 = sum(w .* xbar) / sum(w)
        ll[g] = -0.5 * sum(log.(v)) - 0.5 * sum(w .* (xbar .- μ0) .^ 2) - 0.5 * log(sum(w))
        B = τ2 ./ v
        m[g, :] = μ0 .+ B .* (xbar .- μ0)
        sd[g, :] = sqrt.(B) .* se
        mu0sd[g] = 1 / sqrt(sum(w))
    end
    p = exp.(ll .- maximum(ll)); p ./= sum(p)
    DriftPost(p, m, sd, mu0sd, vec(sum(p .* m, dims = 1)))
end


struct Draw
    post::@NamedTuple{m::Vector{Float64}, A::Symmetric{Float64, Matrix{Float64}}, alpha::Vector{Float64}, noise_precision::Float64}
    D::Matrix{Float64}      # N×P decision-time state (this geometry)
    s1::Vector{Float64}     # 1-day ruler (this geometry)
end

struct Fit
    labels::Vector{@NamedTuple{τ::Int, k::Int}}   # one per (a,b) channel pair
    Z::Matrix{Float64}      # coefficient space → constraint subspace (shared by all draws)
    post::@NamedTuple{m::Vector{Float64}, A::Symmetric{Float64, Matrix{Float64}}, alpha::Vector{Float64}, noise_precision::Float64}
    D::Matrix{Float64}      # N×2M state at decision time        } reference geometry
    s1::Vector{Float64}     # 1-day ruler                        } (the data themselves)
    draws::Vector{Draw}     # geometry-bootstrap draws, each with its own response posterior
    mean_r::Vector{Float64} # sample mean next-day log return of each asset (raw), over its own history
    drift::DriftPost        # hierarchical posterior of the per-asset drift
    resid::Matrix{Float64}  # n×N residuals after the response (return units); NaN before an asset exists
    r0::Matrix{Float64}     # n×N demeaned returns (no-response baseline); NaN before an asset exists
    n::Int                  # rows of resid
    nrows::Int              # price rows of the longest asset: every one enters the fit (warm-up, regression, geometry)
    ngeom::Int              # rows of the geometry window (ruler, modes, noise covariance)
    nreg::Int               # state rows of the response evidence (disjoint from the geometry window)
end

"Runs of consecutive rows of the Bool matrix `V` with an identical row pattern."
function patterns(V)
    rs = UnitRange{Int}[]; a = 1
    for t in 2:size(V, 1)
        V[t, :] == V[t-1, :] || (push!(rs, a:t-1); a = t)
    end
    push!(rs, a:size(V, 1))
end

"""
Fit. Every regressor channel is a rank-one field φ_k ⊗ c_k(t) (mode vector × scalar state
series), so with whitener Γ the normal equations need no N×n×P design tensor:
  X'X = (AᵀA) ∘ (CᵀC),  X'y = Σ_t C ∘ (Aᵀ ΓY),  A = Γ Φ.
Exact reformulation (same numbers), O(nP² + NnP/…) instead of materialising N·n·P (with ragged panels per run of identical listing pattern).

Information flow (theory layer; see the header):
  geometry window  last G = max(GEOM_MIN, GEOM_FRAC·T) rows  → ruler, modes, noise covariance
  response block   state rows before it                      → evidence, coefficients (ARD)
  The two blocks share no return, so the eigenvectors are not selected on the response noise.
  `draws` stationary-bootstrap resamples of the window give geometry draws; each has its own
  response posterior; a band keeps mode k only if it is stable across the draws.
Ragged panel: each asset enters at its own listing. Covariances are pairwise-available, the
Gaussian likelihood of every row is that of the observed sub-panel (Γ from Σ restricted to the
listed assets), states are built from the listed assets, drift/long-run variance and the
volatility history use every asset's own rows. Needs ≥ `MINROWS` rows per asset.
"""
function fit_response(adj::AbstractMatrix; rng = Random.Xoshiro(0), draws = GEOM_DRAWS)
    f = firstrows(adj); T0, N = size(adj)
    all(f .<= T0 - MINROWS + 1) || error("history too short: every asset needs $MINROWS rows")
    draws >= 2 || error("need at least 2 geometry draws")
    r1 = minimum(f); x = log.(@view adj[r1:end, :]); T = size(x, 1); f = f .- r1 .+ 1
    ok = isfinite.(x); x0 = ifelse.(ok, x, 0.0)

    G = clamp(round(Int, GEOM_FRAC * T), GEOM_MIN, 1500); nreg = T - G - WARMUP
    rlo = WARMUP                                         # first state row (asset listed at row 1)
    gd = geodata(x, f, G)
    ref = geometry(x, f, gd, rng, false)
    bs = [geometry(x, f, gd, rng, true) for _ in 1:draws]
    K = stable_modes(ref, bs)

    labels = [(; τ, k) for (τ, Kb) in zip(BANDS, K) for k in 1:Kb]
    M = length(labels); P = 2M
    idx = [(p + 1) >> 1 for p in 1:P]                    # channel → mode
    Cn = zeros(2, P)                                     # neutrality on non-macro modes: Σa = Σb = 0
    for (m, l) in enumerate(labels)
        l.k >= 2 && (Cn[1, 2m-1] = 1; Cn[2, 2m] = 1)
    end
    Z = any(!iszero, Cn) ? nullspace(Cn) : Matrix{Float64}(I, P, P)

    # raw (un-normalised) state series, independent of the geometry: rows rlo:T
    rows = rlo:T; n1 = length(rows)
    S = zeros(T + 1, N); cumsum!(view(S, 2:T+1, :), x0; dims = 1)
    valid = [t >= f[j] + WARMUP - 1 for t in rows, j in 1:N]
    raw = map(BANDS) do τ
        c0 = (view(S, rows .+ 1, :) .- view(S, rows .+ 1 .- τ, :)) ./ τ
        cl = (view(S, rows .+ 1 .- τ, :) .- view(S, rows .+ 1 .- 2τ, :)) ./ τ
        ((view(x0, rows, :) .- c0) .* valid, (c0 .- cl) .* valid)          # (Q, p) in log-price units
    end
    Rraw = view(x, 2:T, :) .- view(x, 1:T-1, :)          # NaN until the asset exists
    own = [filter(isfinite, view(Rraw, :, j)) for j in 1:N]
    mean_r = mean.(own)
    Rdm = Rraw .- mean_r'
    Rdm0 = ifelse.(isfinite.(Rdm), Rdm, 0.0)
    groups = patterns(view(valid, 1:nreg, :))

    function respond(g)
        s1 = g.s[:, 1]
        Φm = reduce(hcat, [g.Φ[b][:, 1:K[b]] for b in eachindex(BANDS)])
        chans = map(eachindex(BANDS)) do b
            Φt = g.Φ[b][:, 1:K[b]] ./ g.s[:, BANDCOL[b]]
            Cb = Matrix{Float64}(undef, n1, 2K[b])
            Cb[:, 1:2:end] .= .-(raw[b][1] * Φt); Cb[:, 2:2:end] .= raw[b][2] * Φt
            Cb
        end
        C = reduce(hcat, chans)
        cbar = vec(mean(view(C, 1:nreg, :); dims = 1))
        Cc = view(C, 1:nreg, :) .- cbar'
        Y = (view(Rdm0, rlo:rlo+nreg-1, :) ./ s1') .* view(valid, 1:nreg, :)
        Gm = zeros(P, P); c = zeros(P); yy = 0.0; nobs = 0
        for rg in groups
            O = findall(view(valid, first(rg), :))
            λ, V = eigen(Symmetric(g.Σ[O, O]))
            Γ = V * Diagonal(1 ./ sqrt.(max.(λ, 1e-10))) * V'              # whitener of the listed sub-panel
            A = (Γ * Φm[O, :])[:, idx]; Cg = Cc[rg, :]; GY = Γ * Y[rg, O]'
            Gm .+= (A' * A) .* (Cg' * Cg)
            c .+= vec(sum(Cg' .* (A' * GY); dims = 2))
            yy += sum(abs2, GY); nobs += length(O) * length(rg)
        end
        post = bayes_ard(Gm, c, yy, nobs, Z)
        D = Φm[:, idx] .* (view(C, n1, :) .- cbar)'
        (; post, D, s1, Φm, C, cbar)
    end

    r = respond(ref)
    β = Z * r.post.m
    E = (view(r.C, 1:n1-1, :) .- r.cbar') .* β'                            # fitted state-by-coefficient
    fitted = (r.Φm * (E[:, 1:2:end] .+ E[:, 2:2:end])')' .* r.s1' .* view(valid, 1:n1-1, :)
    resid = copy(Rdm); resid[rlo:T-1, :] .-= fitted        # NaN stays NaN

    drift = shrink_drift(mean_r, [sqrt(lrv(reshape(o, 1, :))[1] / length(o)) for o in own])
    dr = map(bs) do g
        r2 = respond(g)
        Draw(r2.post, r2.D, r2.s1)
    end
    Fit(labels, Z, r.post, r.D, r.s1, dr, mean_r, drift, resid, Rdm, T - 1, T, G, nreg)
end

"""
Posterior over the belief Π(dθ_k, dρ_k | H) for every (band, mode): (a_k,b_k) is jointly Gaussian
under the posterior (reference geometry), a_k = ρ_k cosθ_k on the observed deviation −q (mean
reversion axis), b_k = ρ_k sinθ_k on the observed centre-slope p (belief axis: b>0 "confirm the
move", b<0 "front-run it"). q,p are observation coordinates, (ρ,θ) are belief parameters. Nothing is
collapsed to a point θ; per mode we report
  a, b       posterior mean coordinates
  ρ5/ρ50/ρ95 quantiles of ρ
  θmean, R1  circular mean and concentration |E e^{iθ}| (R1≈0: no preferred direction)
  R2         |E e^{2iθ}| (R2 > R1: two opposite peaks — the data cannot tell the two
             opposite beliefs apart, e.g. reversion vs trend)
  p_revert   P(a_k > 0)  — belief "deviation reverts"      (cosθ > 0)
  p_right    P(b_k > 0)  — belief axis b: "the move is worth confirming" (sinθ > 0)
  alpha      evidence-maximised prior precision (→ ALPHA_MAX: no evidence, ρ→0)
"""
function theta_posterior(fit::Fit; draws = 4000, rng = Random.MersenneTwister(0))
    M = length(fit.labels)
    U = cholesky(fit.post.A).U
    B = fit.Z * (fit.post.m .+ U \ randn(rng, size(U, 1), draws))             # P×draws, β draws
    map(enumerate(fit.labels)) do (g, l)
        a, b = B[2g-1, :], B[2g, :]
        ρ = hypot.(a, b); θ = atan.(b, a)
        (; l.τ, l.k, macro_ = l.k == 1, a = mean(a), b = mean(b),
           ρ5 = quantile(ρ, 0.05), ρ50 = median(ρ), ρ95 = quantile(ρ, 0.95),
           θmean = mod(atan(mean(sin.(θ)), mean(cos.(θ))), 2π),
           R1 = hypot(mean(cos.(θ)), mean(sin.(θ))), R2 = hypot(mean(cos.(2θ)), mean(sin.(2θ))),
           p_revert = mean(a .> 0), p_right = mean(b .> 0), alpha = fit.post.alpha[g])
    end
end


"Posterior-mean response in return units: mean over the geometry draws (what `predict` moment-matches to)."
response_mean(fit::Fit) = sum(d.s1 .* (d.D * (fit.Z * d.post.m)) for d in fit.draws) / length(fit.draws)

"Posterior-mean conditional return (deterministic): drift posterior mean + `response_mean`."
conditional_mean(fit::Fit) = fit.drift.mean .+ response_mean(fit)

"One standardised row of `Z`: a random row, a missing entry replaced by the asset's own random entry."
function innovation(Z, own, rng)
    x = Z[rand(rng, 1:size(Z, 1)), :]
    for j in eachindex(x)
        isfinite(x[j]) || (x[j] = Z[rand(rng, own[j]), j])
    end
    x
end

"Rows of each column of E that exist (finite)."
own_rows(E) = [findall(isfinite, view(E, :, j)) for j in axes(E, 2)]

"""
Posterior-predictive next-day log-return scenarios, S×N. Per scenario: one geometry draw from the
block bootstrap (`fit.draws`, uniform) with its own response posterior, one drift draw, and one
innovation from the mode-space long-memory conditional covariance (`volmodel=true`, `mode_vol_model`)
or the iid bootstrap of the residual rows (`volmodel=false`).
"""
function predict(fit::Fit; S = 1000, rng = Random.default_rng(), response = true, volmodel = true)
    N = length(fit.s1)
    Us = [cholesky(d.post.A).U for d in fit.draws]
    E = response ? fit.resid : fit.r0
    vm = volmodel ? mode_vol_model(E) : nothing
    own = volmodel ? nothing : own_rows(E)
    out = zeros(S, N)
    for i in 1:S
        g = min(searchsortedfirst(cumsum(fit.drift.p), rand(rng)), length(fit.drift.p))
        drift = fit.drift.m[g, :] .+ randn(rng) * fit.drift.mu0sd[g] .+ randn(rng, N) .* fit.drift.sd[g, :]
        k = rand(rng, 1:length(fit.draws)); d = fit.draws[k]
        resp = response ? d.s1 .* (d.D * (fit.Z * (d.post.m .+ Us[k] \ randn(rng, size(Us[k], 1))))) : zeros(N)
        eps = vm === nothing ? innovation(E, own, rng) : draw(vm, rng)
        out[i, :] = drift .+ resp .+ eps
    end
    # Numerics only: the scenario mean must equal the analytic predictive mean exactly. With S
    # draws the sample mean of one asset carries ~σ/√S noise (≈29 %/yr at S=300), far above
    # any real cross-sectional signal, and Kelly would chase it. Moment-matching the first
    # moment removes that error and leaves the objective (S→∞) unchanged.
    out .+= (fit.drift.mean .+ (response ? response_mean(fit) : zeros(N)))' .- mean(out, dims = 1)
    out
end

predictive_log_returns(adj; S = 1000, rng = Random.default_rng(), response = true, volmodel = true) =
    predict(fit_response(adj); S, rng, response, volmodel)

"""
Exact Kelly over free assets f ≥ 0, Σf = budget:  max mean log(base + X f).
`base` is the (scenario-wise) wealth already committed to locked positions; with the
defaults this is plain full-investment Kelly.
"""
function kelly_weights(X::AbstractMatrix; base = nothing, budget = 1.0)
    S, n = size(X)
    f = Variable(n)
    wealth = base === nothing ? X * f : X * f + base
    prob = maximize(sum(log(wealth)) / S, [f >= 0, sum(f) == budget])
    solve!(prob, Clarabel.Optimizer; silent = true)
    prob.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        error("Kelly solve failed: $(prob.status)")
    out = clamp.(vec(evaluate(f)), 0, budget)
    out .* (budget / sum(out))
end

"""
First row with a real price per column (T+1 if never listed). `adj` is NaN before the
listing and finite after it (marking prices are carried through halts, see `Bars`).
"""
firstrows(adj) = [something(findfirst(isfinite, @view(adj[:, j])), size(adj, 1) + 1) for j in axes(adj, 2)]

"""
History-eligible columns: listed for at least `MINROWS` rows — the numerical minimum `fit_response`
needs from one asset, not a statistical threshold (weak evidence is the posterior's business).
Says nothing about tradability today. Eligible assets keep their entire own history in the fit.
"""
eligible(f::AbstractVector{<:Integer}, T) = (T .- f .+ 1) .>= MINROWS
eligible(adj::AbstractMatrix) = eligible(firstrows(adj), size(adj, 1))

"Posterior-predictive gross-return scenarios (S×|mask|) for the history-eligible assets."
function scenarios(adj::AbstractMatrix; S = 1000, rng = Random.default_rng(), response = true, volmodel = true)
    mask = eligible(adj)
    any(mask) || error("no asset has $MINROWS rows of history")
    mask, exp.(predictive_log_returns(adj[:, mask]; S, rng, response, volmodel))
end

"""
Weights over all N assets (Σ=1) from scenarios `R` of the `mask` assets.
  free   assets that are history-eligible AND tradable today (free ⊂ mask)
  held   current portfolio weights, or nothing before the first trade
Positions in assets that are not free cannot be changed: they keep their weight `held`
(including the risk of those that are modelled), and the free assets share the remaining
budget 1 − Σlocked by exact Kelly given that fixed exposure. If nothing is free (or
nothing is left to allocate) the portfolio is simply held.
"""
function allocate(R, mask, free, held = nothing)
    N = length(mask)
    all(mask[free]) || error("free assets must be history-eligible")
    locked = held === nothing ? zeros(N) : held .* .!free
    L = sum(locked)
    fi = findall(free)
    if isempty(fi) || 1 - L <= 1e-9
        held === nothing && error("nothing tradable and nothing held")
        return copy(held)
    end
    col = cumsum(mask)
    out = copy(locked)
    if L > 0
        lm = findall((locked .> 0) .& mask)
        base = fill(sum(locked[.!mask]), size(R, 1))   # locked but unmodelled: constant wealth
        isempty(lm) || (base .+= R[:, col[lm]] * locked[lm])
        out[fi] = kelly_weights(R[:, col[fi]]; base, budget = 1 - L)
    else
        out[fi] = kelly_weights(R[:, col[fi]])
    end
    out
end

"""
Target weights from the price history. `tradable` (default: has a price) says which assets
can be traded now; `held` the current portfolio. `response=false` is the no-edge baseline.
"""
function path_kelly(adj::AbstractMatrix; S = 1000, rng = Random.default_rng(), response = true,
                    tradable = nothing, held = nothing, volmodel = true)
    mask, R = scenarios(adj; S, rng, response, volmodel)
    tr = tradable === nothing ? isfinite.(adj[end, :]) : tradable
    allocate(R, mask, mask .& tr, held)
end
