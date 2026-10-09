"""
KTrader V1.0: Geometry Layer
- Per-asset fractal ruler s_i(τ) = c_i τ^{H_i} using each asset's OWN complete history
- Center-of-mass macro decomposition on Ragged Panels:
    m_t = (1/√N_t) Σ_{j ∈ alive(t)} u_{t, j}
    u_perp_{t, j} = u_{t, j} - m_t / √N_t  (exact zero sum on alive assets)
- Strict orthogonal complement relative modes:
    P_0 = I - e_0 e_0',  C_perp = P_0 * C_avail * P_0
    Ensures e_0' * Phi_perp ≡ 0 strictly, preserving all N-1 relative degrees of freedom.
- Deterministic sign convention for eigenvectors (eliminates arbitrary sign flips across folds).
"""

const TAUS  = 2 .^ (0:8)      # ruler fit horizons (1, 2, 4, 8, 16, 32, 64, 128, 256)
const BANDS = 2 .^ (1:7)      # multi-scale bands (2, 4, 8, 16, 32, 64, 128)
const BANDCOL = [findfirst(==(τ), TAUS) for τ in BANDS]
const WARMUP = 2 * maximum(BANDS)
const DEFAULT_L_BLOCK = 64

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

function center_of_mass_decomposition(u::AbstractMatrix{Float64})
    T, N = size(u)
    m = zeros(Float64, T)
    u_perp = fill(NaN, T, N)
    e0 = fill(1.0 / sqrt(N), N)
    
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

function relative_modes(u_perp::AbstractMatrix{Float64}, e0::AbstractVector{Float64}, w::AbstractVector{Float64} = ones(size(u_perp, 1)))
    T, N = size(u_perp)
    C_raw = pairwise_covariance(u_perp, w)
    
    P0 = I - e0 * e0'
    C_perp = P0 * C_raw * P0
    
    E = eigen(Symmetric(C_perp))
    
    idx = sortperm(E.values, rev=true)
    vals = E.values[idx]
    vecs = E.vectors[:, idx]
    
    K_rel = N - 1
    Phi_perp = vecs[:, 1:K_rel]
    
    # Gram-Schmidt cleanup against e0 and canonical sign convention
    for k in 1:K_rel
        v = Phi_perp[:, k]
        v .-= dot(e0, v) * e0
        v ./= norm(v)
        # Canonical sign convention: largest absolute component is strictly positive
        max_idx = argmax(abs.(v))
        if v[max_idx] < 0
            v .= -v
        end
        Phi_perp[:, k] .= v
    end
    
    (; Phi_perp, eigenvalues = vals[1:K_rel])
end

function stationary_bootstrap_weights(rng::AbstractRNG, T::Int, L::Int = DEFAULT_L_BLOCK)
    w = zeros(Float64, T)
    p_geom = 1.0 / max(L, 1)
    
    idx = rand(rng, 1:T)
    for _ in 1:T
        w[idx] += 1.0
        idx = (rand(rng) < p_geom) ? rand(rng, 1:T) : mod1(idx + 1, T)
    end
    w
end

function bootstrap_relative_modes(u_perp::AbstractMatrix{Float64}, e0::AbstractVector{Float64};
                                  D_draws = 4, L_block = DEFAULT_L_BLOCK, rng = Random.default_rng())
    T, N = size(u_perp)
    modes_draws = Vector{Matrix{Float64}}(undef, D_draws)
    
    for d in 1:D_draws
        w_boot = stationary_bootstrap_weights(rng, T, L_block)
        rel = relative_modes(u_perp, e0, w_boot)
        modes_draws[d] = rel.Phi_perp
    end
    modes_draws
end
