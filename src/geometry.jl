"""
KTrader V0.95: Geometry Layer
- Per-asset fractal ruler s_i(τ) = c_i τ^{H_i} using each asset's OWN complete history
- Center-of-mass macro decomposition: e_0 = (1, ..., 1)' / √N
- Fixed Numerical Gauge: Analytical Helmert basis Q (N × (N-1))
    Q' * e0 = 0,  Q' * Q = I_{N-1}
- Relative covariance C_Q = Q' * C_avail * Q strictly in (N-1) space with PSD repair
- No data-dependent eigenvector tracking across folds; modes are operator spectra, not coordinate axes
"""

const TAUS  = 2 .^ (0:8)      # ruler fit horizons (1, 2, 4, 8, 16, 32, 64, 128, 256)
const BANDS = 2 .^ (1:7)      # multi-scale bands (2, 4, 8, 16, 32, 64, 128)
const BANDCOL = [findfirst(==(τ), TAUS) for τ in BANDS]
const WARMUP = 2 * maximum(BANDS)

"""
Analytical Helmert basis matrix Q of size N × (N-1):
Columns are strictly orthonormal and strictly orthogonal to e_0 = 1/√N.
Deterministic, coordinate-free, invariant to sample estimation noise.
"""
function helmert_basis(N::Int)
    N >= 2 || error("N must be >= 2 for relative subspace decomposition")
    Q = zeros(Float64, N, N - 1)
    for j in 1:(N - 1)
        # Column j has j identical entries, one negative entry, rest zeros
        c = 1.0 / sqrt(j * (j + 1))
        for i in 1:j
            Q[i, j] = c
        end
        Q[j + 1, j] = -j * c
    end
    Q
end

"""
Center-of-mass unit vector in asset space:
    e_0 = (1, 1, ..., 1)' / √N
"""
center_of_mass(N::Int) = fill(1.0 / sqrt(N), N)

"""
Per-asset fractal ruler s_i(τ) on TAUS: power-law fit of weighted RMS τ-increments.
Uses ALL of asset j's own available history.
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
                    x1 = x[t, j]; x0 = x[t - τ, j]
                    if isfinite(x1) && isfinite(x0)
                        d = x1 - x0
                        wt = w[t]
                        acc += wt * d * d
                        ws += wt
                    end
                end
                val = log(sqrt(max(acc / max(ws, 1e-12), 1e-12)))
                L_buf[n_use] = val
                sum_l += val
                sum_log_tau += lτ[a]
            end
        end
        if n_use >= 2
            m_l = sum_l / n_use
            m_tau = sum_log_tau / n_use
            num = 0.0; den = 0.0
            for a in 1:n_use
                d_tau = lτ[a] - m_tau
                num += (L_buf[a] - m_l) * d_tau
                den += d_tau * d_tau
            end
            H = num / max(den, 1e-12)
            for a in 1:Ntaus
                out[j, a] = exp(m_l + H * (lτ[a] - m_tau))
            end
        else
            out[j, :] .= 1.0
        end
    end
    out
end

"""
Decompose price increments u (T × N) into:
1. Macro center-of-mass component m_t = (1/√N_t) Σ_{j ∈ alive(t)} u_{t, j}
2. Relative subspace u_perp_{t, j} = u_{t, j} - m_t / √N_t
This preserves ALL history: long-history assets contribute to macro decades before newer assets list.
"""
function center_of_mass_decomposition(u::AbstractMatrix{Float64})
    T, N = size(u)
    m = zeros(Float64, T)
    u_perp = fill(NaN, T, N)
    e0 = center_of_mass(N)
    
    for t in 1:T
        alive = findall(j -> isfinite(u[t, j]), 1:N)
        Nt = length(alive)
        if Nt > 0
            sum_u = sum(u[t, j] for j in alive)
            m[t] = sum_u / sqrt(Nt)
            shift = sum_u / Nt
            for j in alive
                u_perp[t, j] = u[t, j] - shift
            end
        end
    end
    (; e0, m, u_perp)
end

"""
Pairwise-available weighted covariance matrix on ragged data.
w: Optional row weights.
"""
function pairwise_covariance(X::AbstractMatrix{Float64}, w::AbstractVector{Float64} = ones(size(X, 1)))
    T, N = size(X)
    C = zeros(Float64, N, N)
    
    for j in 1:N
        for i in j:N
            count = 0
            W_sum = 0.0
            s_i = 0.0; s_j = 0.0; s_ij = 0.0
            for t in 1:T
                xi = X[t, i]; xj = X[t, j]
                if isfinite(xi) && isfinite(xj)
                    wt = w[t]
                    count += 1
                    W_sum += wt
                    s_i += wt * xi
                    s_j += wt * xj
                    s_ij += wt * xi * xj
                end
            end
            if count >= 30 && W_sum > 1e-6
                cov_val = (s_ij - s_i * s_j / W_sum) / max(W_sum * (count - 1) / count, 1e-6)
            else
                cov_val = 0.0
            end
            C[i, j] = cov_val
            C[j, i] = cov_val
        end
        if C[j, j] <= 1e-12
            C[j, j] = 1.0
        end
    end
    C
end

"""
Project relative increments u_perp into the fixed Helmert gauge:
    z_rel = u_perp * Q   (T × (N-1))
Computes PSD-repaired covariance C_Q in the (N-1) space:
    C_Q = Q' * C_avail * Q
    Repaired via eigenvalues: U * max(Λ, 1e-6) * U'
Strictly free from any e_0 zero/negative eigenvalue sorting hazard.
"""
function project_helmert_gauge(u_perp::AbstractMatrix{Float64}, Q::AbstractMatrix{Float64})
    T, N = size(u_perp)
    K_rel = N - 1
    
    # Project each row t into Helmert coordinates
    z_rel = zeros(Float64, T, K_rel)
    for t in 1:T
        for k in 1:K_rel
            val = 0.0
            for j in 1:N
                uj = u_perp[t, j]
                if isfinite(uj)
                    val += uj * Q[j, k]
                end
            end
            z_rel[t, k] = val
        end
    end
    
    # Available-case covariance in N-space
    C_raw = pairwise_covariance(u_perp)
    # Strictly project into (N-1) space
    C_Q_raw = Q' * C_raw * Q
    
    # Exact PSD repair in (N-1) space
    E = eigen(Symmetric(C_Q_raw))
    max_ev = maximum(E.values)
    repaired_vals = max.(E.values, max(max_ev * 1e-4, 1e-6))
    C_Q = E.vectors * Diagonal(repaired_vals) * E.vectors'
    
    (; z_rel, C_Q, E)
end
