const DGRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

"""
Fractional-integration kernel π_k(d) = Γ(k+d)/(Γ(k+1)Γ(d)), k = 0..n−1: the (1−L)^{-d} coefficients
(Granger–Joyeux / Hosking). d = 1 is constant variance (all weights 1), d → 0 is no memory.
"""
frac_weights(d, n) = accumulate((w, k) -> w * (k - 1 + d) / k, 1:n-1; init = 1.0) |> w -> pushfirst!(w, 1.0)

"""
Conditional risk on the natural modes.

    e_{t+1} = Φ z_{t+1} + u_{t+1},     z = Φ'e (mode innovations),  u = (I − ΦΦ')e (idiosyncratic part),
    Σ_{t+1} = Φ diag(v_{1..K,t+1}) Φ' + P D_{t+1} P,     P = I − ΦΦ'.

The idiosyncratic part lives in the orthogonal complement of the modes BY DEFINITION (u = P e), so its
rescaled draw is projected back with P: rescaling each asset by its own forecast would otherwise push
u into the modes' near-null directions and put variance where the data has almost none (measured: 3.7
QLIKE points lost on a nearly singular iid world, none lost once u is projected).

Every mode k has its OWN long-memory variance forecast v_k = Σ_j π_j(d_k) z²_{k,T−j} / Σ_j π_j(d_k)
(`frac_weights`, exact FFT convolution; FIGARCH idea of Baillie–Bollerslev–Mikkelsen 1996), and the
idiosyncratic block has one exponent for its diagonal. Every exponent keeps its full posterior on
`grid` (flat prior, Gaussian quasi-likelihood of the next-day square given the forecast) and is
sampled, never collapsed, so macro, sector and idiosyncratic risk share one geometry and sector
rotation is energy moving between modes.

Modes. Φ is the eigenbasis of the available-case covariance of E (PSD-repaired). With `K = :auto` the
kept modes are
  (i)  the leading ones: as many as there are eigenvalues of cor(E) above the same-rank eigenvalue of
       every one of 20 circularly shifted copies of E (Horn's parallel analysis with a shift null: every
       column keeps its marginal law, volatility clustering and serial dependence, only the
       cross-sectional dependence is destroyed, so no iid assumption enters), and
  (ii) every TAIL direction whose variance is below `TAILFRAC` × the mean asset variance: a near exact
       linear relation among assets. It is kept as an explicit mode (its own variance forecast keeps the
       relation) instead of being left to the diagonal idiosyncratic block, which has no way to
       represent a direction of almost zero variance.
`K = k::Int` keeps exactly the k leading modes (no tail rule).

Ragged panels (NaN before an asset's listing): Φ comes from the pairwise-available covariance, the
mode series uses every asset that exists at that date (a missing asset contributes zero), the
idiosyncratic series of an asset exists only on its own rows, its forecast is built from its own history
only, and in `draw` a missing standardised entry is replaced by the asset's own random entry. Nothing
is invented and no asset's history is cut to a common window.
"""
const TAILFRAC = 0.1

struct LMBlock
    d::Vector{Float64}                 # kept grid values
    p::Vector{Float64}                 # posterior weights (sum 1)
    cp::Vector{Float64}                # cumulative p, for sampling
    σ::Vector{Vector{Float64}}         # σ_{T+1}(d), one entry per kept d (length = columns of the block)
    Z::Vector{Matrix{Float64}}         # standardised innovations (m × columns), NaN outside an asset's own rows
    own::Vector{Vector{Int}}           # rows of Z each column has
    loglik::Vector{Float64}            # over the full grid, for inspection
end

struct ModeVol
    Phi::Matrix{Float64}               # N×K orthonormal modes
    modes::Vector{LMBlock}             # one block (one column) per mode, own d_k
    idio::LMBlock                      # N columns, one common idiosyncratic d
end

"""
Long-memory forecast block for the columns of `X` (n×c, NaN before a column's first row): one
exponent, posterior over `grid`; each column uses only its own history.
"""
function lm_block(X::AbstractMatrix, grid, burn, keep)
    n, c = size(X)
    a = [findfirst(isfinite, view(X, :, j)) for j in 1:c]
    ok = isfinite.(X); X0 = ifelse.(ok, X, 0.0); X2 = X0 .^ 2
    floor = 1e-12 * sum(X2) / count(ok)
    V = Vector{Matrix{Float64}}(undef, length(grid)); ll = zeros(length(grid))
    for (g, d) in enumerate(grid)
        π = frac_weights(d, n); cs = cumsum(π)
        v = ones(n, c)
        for j in 1:c
            m = n - a[j] + 1
            v[a[j]:n, j] = max.(DSP.conv(view(X2, a[j]:n, j), view(π, 1:m))[1:m] ./ view(cs, 1:m), floor)
        end
        V[g] = v
        ll[g] = -0.5 * sum(log(v[s, j]) + X2[s+1, j] / v[s, j] for j in 1:c for s in a[j]+burn-1:n-1)
    end
    p = exp.(ll .- maximum(ll)); p ./= sum(p)
    ks = findall(>(keep), p); pk = p[ks] ./ sum(p[ks])
    σ = [sqrt.(V[g][n, :]) for g in ks]
    Z = map(ks) do g
        z = X[burn+1:end, :] ./ sqrt.(V[g][burn:n-1, :])
        for j in 1:c; z[1:a[j]-1, j] .= NaN; end                    # burn-in of the column's own history
        z
    end
    LMBlock(grid[ks], pk, cumsum(pk), σ, Z, [findall(isfinite, view(Z[1], :, j)) for j in 1:c], ll)
end

"Available-case correlation matrix of the columns of E (NaN = missing), PSD-repaired when ragged."
function avail_cor(E)
    M = isfinite.(E); C = wcov(ifelse.(M, E, 0.0), Float64.(M), ones(size(E, 1)))
    all(M) || (C = repair(C))
    d = sqrt.(diag(C)); C ./ (d * d')
end

"Circularly shift every column inside its own observed block, preallocating out buffer."
function shift_cols!(out, E, rng)
    n, N = size(E)
    for j in 1:N
        a = findfirst(isfinite, view(E, :, j))
        m = n - a + 1
        sh = rand(rng, m÷10:m-m÷10)
        # copy NaNs
        @inbounds for t in 1:a-1
            out[t, j] = NaN
        end
        # circular shift observed block
        @inbounds for k in 0:m-1
            src_idx = a + ((k + sh) % m)
            out[a + k, j] = E[src_idx, j]
        end
    end
    out
end

"Number of leading modes: eigenvalues of cor(E) above the circular-shift null (see `ModeVol` docs)."
function nmodes(E, nulls = 20)
    n, N = size(E); rng = Random.Xoshiro(0)
    λ = reverse(eigvals(Symmetric(avail_cor(E)))); null = zeros(N)
    E_shift = Matrix{Float64}(undef, n, N)
    for _ in 1:nulls
        shift_cols!(E_shift, E, rng)
        null .= max.(null, reverse(eigvals(Symmetric(avail_cor(E_shift)))))
    end
    k = findfirst(i -> λ[i] <= null[i], 1:N-1)
    k === nothing ? N - 1 : k - 1
end

function mode_vol_model(E::AbstractMatrix; K = :auto, grid = DGRID, burn = 60, keep = 1e-4, max_history = 2500)
    n0, N = size(E)
    n0 > burn + 100 || error("too little history for the mode volatility model")
    # To prevent OOM and huge FFT allocations on 50-year histories (14000 rows),
    # bound the volatility calibration window to the recent 10 years (2500 sessions).
    r_start = max(1, n0 - max_history + 1)
    E = @view E[r_start:end, :]
    n = size(E, 1)
    ok = isfinite.(E)
    C = wcov(ifelse.(ok, E, 0.0), Float64.(ok), ones(n)); all(ok) || (C = repair(C))
    eg = eigen(Symmetric(C))
    Q, λ = eg.vectors[:, end:-1:1], eg.values[end:-1:1]
    if K === :auto
        lead = 1:nmodes(E)
        tail = findall(<(TAILFRAC * tr(C) / N), λ)
        idx = sort!(union(lead, tail)); length(idx) < N || (idx = idx[1:N-1])
    else
        0 <= K < N || error("K must be in 0:N-1")
        idx = 1:K
    end
    Phi = Q[:, idx]
    E0 = ifelse.(ok, E, 0.0)
    Z = E0 * Phi                                                  # a missing asset contributes zero
    modes = [lm_block(view(Z, :, k:k), grid, burn, keep) for k in eachindex(idx)]
    U = ifelse.(ok, E0 - Z * Phi', NaN)                          # idiosyncratic part on each asset's own rows
    ModeVol(Phi, modes, lm_block(U, grid, burn, keep))
end

pick(cp, u) = min(searchsortedfirst(cp, u), length(cp))

"""
One next-day innovation vector: every block draws its exponent from its own posterior, one common row
index resamples the standardised residuals jointly (cross-mode dependence is preserved), each is scaled
by its forecast and the modes are rotated back with Φ. The idiosyncratic output is projected onto the complement of the modes. A missing idiosyncratic entry (asset not yet
listed on that row) is replaced by that asset's own random entry.
"""
function draw(vm::ModeVol, rng::AbstractRNG)
    Phi = vm.Phi; N, K = size(Phi)
    x = zeros(N)
    t = rand(rng, 1:size(vm.idio.Z[1], 1))
    b = vm.idio; g = pick(b.cp, rand(rng)); Zg = b.Z[g]
    for i in 1:N
        z = Zg[t, i]
        isfinite(z) || (z = Zg[rand(rng, b.own[i]), i])
        x[i] = b.σ[g][i] * z
    end
    for k in 1:K                                   # project the idiosyncratic part onto the complement of the modes
        c = 0.0
        for i in 1:N; c += Phi[i, k] * x[i]; end
        for i in 1:N; x[i] -= c * Phi[i, k]; end
    end
    for k in 1:K
        b = vm.modes[k]; g = pick(b.cp, rand(rng))
        y = b.σ[g][1] * b.Z[g][t, 1]
        for i in 1:N
            x[i] += y * Phi[i, k]
        end
    end
    x
end

"""
Second-moment matrix E[xxᵀ] of `draw` (N×N, PSD; diagnostics), exact. x = Σ_b B_b y_b(t, d_b) with one
shared row t and independent exponents d_b per block, so
  E[xxᵀ] = M̄ᵀM̄/m + Σ_b B_bᵀ [ Σ_d p_d (Ȳ_d − ȳ)ᵀ(Ȳ_d − ȳ)/m + Σ_d p_d diag(fill-variance_d) ] B_b,
M̄ = Σ_b ȳ_b B_b the exponent-averaged outputs. For a late-listed asset a missing standardised entry is
replaced in `draw` by an independent random own-row entry (same exponent): Ȳ_d holds its own-row mean
there and `fill-variance` the own-row variance.
"""
function cond_cov(vm::ModeVol)
    N = size(vm.Phi, 1); m = size(vm.idio.Z[1], 1)
    function parts(b::LMBlock)
        Y, F = Matrix{Float64}[], Vector{Float64}[]
        for (Z, σ) in zip(b.Z, b.σ)
            Yd = Z .* σ'; fv = zeros(size(Z, 2))
            for j in axes(Z, 2)
                r = b.own[j]; length(r) == m && continue
                mu = mean(view(Yd, r, j)); fv[j] = (m - length(r)) / m * mean(abs2, view(Yd, r, j) .- mu)
                Yd[setdiff(1:m, r), j] .= mu
            end
            push!(Y, Yd); push!(F, fv)
        end
        ybar = sum(p * Y_ for (p, Y_) in zip(b.p, Y))
        ybar, sum(p * (Y_ - ybar)' * (Y_ - ybar) for (p, Y_) in zip(b.p, Y)) / m + Diagonal(sum(p * f for (p, f) in zip(b.p, F)))
    end
    Mbar = zeros(m, N); M2 = zeros(N, N)
    for (k, bk) in enumerate(vm.modes)
        yb, C = parts(bk); B = reshape(vm.Phi[:, k], 1, N)
        Mbar .+= yb * B; M2 .+= B' * C * B
    end
    yb, C = parts(vm.idio)
    P = I - vm.Phi * vm.Phi'
    Mi = Mbar .+ yb * P
    Symmetric(Mi' * Mi / m + M2 + P * C * P)
end

nan0(x) = isfinite(x) ? x : 0.0
