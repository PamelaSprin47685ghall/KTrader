"""
KTrader V1.0 Final: Geometry Layer
- Per-asset fractal ruler s_i(τ) = c_i τ^{H_i} using each asset's OWN complete history
- Center-of-mass macro decomposition: e_0 = (1, ..., 1)' / √N
- Pure Orthogonal Complement Geometry: P_0 = I - e_0 e_0'
    m_t = e_0' u_t (scalar macro flow)
    u_perp_t = P_0 u_t (N-dimensional relative field with exact zero center of mass)
- Zero coordinate-ordering bias: 100% permutation symmetric across all assets!
"""

const TAUS  = 2 .^ (0:8)      # ruler fit horizons (1, 2, 4, 8, 16, 32, 64, 128, 256)
const BANDS = 2 .^ (1:7)      # multi-scale bands (2, 4, 8, 16, 32, 64, 128)
const BANDCOL = [findfirst(==(τ), TAUS) for τ in BANDS]
const WARMUP = 2 * maximum(BANDS)

"""
Center-of-mass unit vector in asset space:
    e_0 = (1, 1, ..., 1)' / √N
"""
center_of_mass(N::Int) = fill(1.0 / sqrt(N), N)

"""
Per-asset fractal ruler s_i(τ) on TAUS: power-law fit of weighted RMS τ-increments.
Uses ALL of asset j's own available history.
"""
function ruler(x::AbstractMatrix{Float64}, f::AbstractVector{<:Integer},
               w::Union{Nothing,AbstractVector{Float64}} = nothing)
    T, N = size(x)
    Ntaus = length(TAUS)
    out = Matrix{Float64}(undef, N, Ntaus)
    lτ = log.(TAUS)
    L_buf = zeros(Float64, Ntaus)
    tau_buf = similar(L_buf)
    for j in 1:N
        n_use = 0; sum_l = 0.0; sum_log_tau = 0.0; fj = f[j]
        for a in 1:Ntaus
            τ = TAUS[a]
            if 4τ <= T - fj + 1
                acc = 0.0; ws = 0.0
                @inbounds @simd for t in (fj + τ):T
                    x1 = x[t, j]; x0 = x[t - τ, j]
                    if isfinite(x1) && isfinite(x0)
                        d = x1 - x0
                        wt = w === nothing ? 1.0 : w[t]
                        acc += wt * d * d
                        ws += wt
                    end
                end
                ws > 0 || continue
                n_use += 1
                val = log(sqrt(max(acc / max(ws, 1e-12), 1e-12)))
                L_buf[n_use] = val
                tau_buf[n_use] = lτ[a]
                sum_l += val
                sum_log_tau += lτ[a]
            end
        end
        if n_use >= 2
            m_l = sum_l / n_use
            m_tau = sum_log_tau / n_use
            num = 0.0; den = 0.0
            for a in 1:n_use
                d_tau = tau_buf[a] - m_tau
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
Prefix statistics for O(1) fractal ruler evaluations across decision days:
acc[t, j, a]: cumulative sum of (x[s, j] - x[s - τ, j])^2 for s <= t
cnt[t, j, a]: count of valid pairs
"""
struct PrefixRulerStats
    acc::Array{Float64, 3}  # T × N × Ntaus
    cnt::Array{Int32, 3}    # T × N × Ntaus
end

function build_prefix_ruler_stats(x::AbstractMatrix{Float64}, f::AbstractVector{<:Integer})
    T, N = size(x)
    Ntaus = length(TAUS)
    acc = zeros(Float64, T, N, Ntaus)
    cnt = zeros(Int32, T, N, Ntaus)
    for j in 1:N
        fj = f[j]
        for (a, τ) in enumerate(TAUS)
            running_acc = 0.0
            running_cnt = Int32(0)
            for t in (fj + τ):T
                x1 = x[t, j]; x0 = x[t - τ, j]
                if isfinite(x1) && isfinite(x0)
                    d = x1 - x0
                    running_acc += d * d
                    running_cnt += Int32(1)
                end
                acc[t, j, a] = running_acc
                cnt[t, j, a] = running_cnt
            end
        end
    end
    PrefixRulerStats(acc, cnt)
end

function ruler_from_stats(stats::PrefixRulerStats, t::Int, f::AbstractVector{<:Integer},
                          columns::AbstractVector{<:Integer} = collect(eachindex(f)))
    N = length(f)
    length(columns) == N || throw(DimensionMismatch("ruler column mapping"))
    Ntaus = length(TAUS)
    out = Matrix{Float64}(undef, N, Ntaus)
    lτ = log.(TAUS)
    L_buf = zeros(Float64, Ntaus)
    tau_buf = similar(L_buf)
    for j in 1:N
        fj = f[j]
        column = columns[j]
        n_use = 0; sum_l = 0.0; sum_log_tau = 0.0
        for a in 1:Ntaus
            τ = TAUS[a]
            if 4τ <= t - fj + 1
                c = stats.cnt[t, column, a]
                if c > 0
                    n_use += 1
                    val = log(sqrt(max(stats.acc[t, column, a] / c, 1e-12)))
                    L_buf[n_use] = val
                    tau_buf[n_use] = lτ[a]
                    sum_l += val
                    sum_log_tau += lτ[a]
                end
            end
        end
        if n_use >= 2
            m_l = sum_l / n_use
            m_tau = sum_log_tau / n_use
            num = 0.0; den = 0.0
            for a in 1:n_use
                d_tau = tau_buf[a] - m_tau
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
Preserves ALL history: strictly zero-sum in relative space, fully permutation symmetric.
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
