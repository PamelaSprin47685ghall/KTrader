"""
KTrader V1.0 Final: Response Layer
- Multi-scale causal path basis functions B_m[H_t] over cumulative price coordinates
- Pure Permutation-Symmetric Relative Response Operator in zero-sum space
- Exact Gaussian Conditioning on Trace Neutrality:
    tr(A_b) = 0, tr(B_b) = 0 on parameter support strictly
- Empirical Bayes Evidence alpha optimization
"""

"""
Construct causal multi-scale path basis functions for 1D integrated coordinate X(t):
For each dyadic timescale τ in BANDS:
- Q_τ(t) = (X(t) - c0(t)) / scale  (displacement from trailing mean)
- P_τ(t) = (c0(t) - c1(t)) / scale  (velocity of trailing mean)
"""
function path_basis_1d(X::AbstractVector{Float64}, s::AbstractVector{Float64})
    T = length(X)
    M = length(BANDS)
    B = zeros(Float64, T, 2 * M)
    S = cumsum(vcat(0.0, X))
    
    for (idx, τ) in enumerate(BANDS)
        scale = length(s) == length(BANDS) ? max(s[idx], 1e-6) : max(s[BANDCOL[idx]], 1e-6)
        for t in (2τ + 1):T
            c0 = (S[t + 1] - S[t + 1 - τ]) / τ
            c1 = (S[t + 1 - τ] - S[t + 1 - 2τ]) / τ
            q = (X[t] - c0) / scale
            p = (c0 - c1) / scale
            B[t, 2idx - 1] = -q
            B[t, 2idx]     = p
        end
    end
    B
end

function build_path_basis(m::AbstractVector{Float64}, X_rel::AbstractMatrix{Float64}, s_macro::AbstractVector{Float64}, s_perp::AbstractVector{Float64})
    T = length(m)
    N = size(X_rel, 2)
    X_m = cumsum(m)
    
    B_m = path_basis_1d(X_m, s_macro)
    # Permutation-symmetric: each asset j gets its path basis directly from its own X_rel[:, j]
    B_rel = [path_basis_1d(view(X_rel, :, j), s_perp) for j in 1:N]
    (; B_m, B_rel)
end

struct ResponseOperator
    G_macro::Vector{Float64}                # Length 2 * n_bands
    post_cov_m::Matrix{Float64}             # Macro parameter covariance
    A_matrices::Vector{Matrix{Float64}}     # Conditioned posterior mean A_b (N × N)
    B_matrices::Vector{Matrix{Float64}}     # Conditioned posterior mean B_b (N × N)
    G_raw_mean::Matrix{Float64}             # N × P_features
    L_Sigma_rel::Matrix{Float64}            # Cholesky of Sigma_eps
    L_V_rel::Matrix{Float64}                # Matrix-normal factor
    C_constraint::Matrix{Float64}           # 14 × P_features constraint matrix
    alpha_macro::Float64
    alpha_rel::Float64
    trace_real::Float64                     # Σ_b tr(A_b) ≡ 0
    trace_imag::Float64                     # Σ_b tr(B_b) ≡ 0
end

function optimize_evidence_alpha(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}; iters = 15)
    n_samples, P = size(X)
    K = size(Y, 2)
    
    if P <= n_samples
        XtX = X' * X
        XtY = X' * Y
        ev = max.(eigvals(Symmetric(XtX)), 0.0)
        alpha = 1.0
        gamma = 1.0
        for _ in 1:iters
            inv_V = inv(Symmetric(XtX + alpha * I))
            G_hat = (inv_V * XtY)'
            g_sq = sum(abs2, G_hat)
            gamma = sum(ev ./ (ev .+ alpha))
            alpha_new = clamp((gamma * K) / max(g_sq, 1e-12), 1e-4, 1e6)
            if abs(log(alpha_new) - log(alpha)) < 1e-3
                alpha = alpha_new; break
            end
            alpha = alpha_new
        end
        return alpha, gamma
    else
        XXt = X * X'
        ev = max.(eigvals(Symmetric(XXt)), 0.0)
        alpha = 1.0
        gamma = 1.0
        for _ in 1:iters
            inv_W = inv(Symmetric(XXt + alpha * I))
            G_hat = Y' * inv_W * X
            g_sq = sum(abs2, G_hat)
            gamma = sum(ev ./ (ev .+ alpha))
            alpha_new = clamp((gamma * K) / max(g_sq, 1e-12), 1e-4, 1e6)
            if abs(log(alpha_new) - log(alpha)) < 1e-3
                alpha = alpha_new; break
            end
            alpha = alpha_new
        end
        return alpha, gamma
    end
end

function build_trace_constraint_matrix(N::Int, n_bands::Int)
    P_features = 2 * n_bands * N
    n_constraints = 2 * n_bands
    C = zeros(Float64, n_constraints, P_features)
    
    for b in 1:n_bands
        col_offset = (b - 1) * (2 * N)
        row_A = 2b - 1
        row_B = 2b
        for j in 1:N
            idx_q = col_offset + 2*(j - 1) + 1
            idx_p = col_offset + 2*j
            C[row_A, idx_q] = 1.0
            C[row_B, idx_p] = 1.0
        end
    end
    C
end

function fit_response_operator(B_m, B_rel, y_m, y_rel; ridge_alpha = nothing, ts = WARMUP:length(y_m)-1)
    N = length(B_rel)
    n_bands = length(BANDS)
    n_samples = length(ts)
    P_features = 2 * n_bands * N
    
    # 1. Macro Response
    X_m = B_m[ts, :]
    y_target_m = y_m[ts .+ 1]
    alpha_m, gamma_m = ridge_alpha === nothing ? optimize_evidence_alpha(X_m, reshape(y_target_m, :, 1)) : (ridge_alpha, size(X_m, 2))
    inv_m = inv(Symmetric(X_m' * X_m + alpha_m * I))
    g_macro_vec = inv_m * (X_m' * y_target_m)
    res_m = y_target_m - X_m * g_macro_vec
    sig2_m = max(sum(abs2, res_m) / max(n_samples - gamma_m, 1.0), 1e-8)
    post_cov_m = Matrix(Symmetric(sig2_m * inv_m))
    
    # 2. Relative Response in Permutation-Symmetric Space
    X_rel_stacked = zeros(Float64, n_samples, P_features)
    for (idx, t) in enumerate(ts)
        col = 1
        for b in 1:n_bands
            col_q = 2b - 1
            col_p = 2b
            for j in 1:N
                X_rel_stacked[idx, col]     = B_rel[j][t, col_q]
                X_rel_stacked[idx, col + 1] = B_rel[j][t, col_p]
                col += 2
            end
        end
    end
    
    Y_target_rel = y_rel[ts .+ 1, :] # n_samples × N
    alpha_rel, gamma_rel = ridge_alpha === nothing ? optimize_evidence_alpha(X_rel_stacked, Y_target_rel) : (ridge_alpha, min(P_features, n_samples - 1))
    
    XtX_rel = X_rel_stacked' * X_rel_stacked
    V_unconstrained = inv(Symmetric(XtX_rel + alpha_rel * I))
    G_unconstrained = (V_unconstrained * (X_rel_stacked' * Y_target_rel))' # N × P_features
    
    G_raw = copy(G_unconstrained)
    
    # Exact Conditioning on Trace Neutrality:
    # Under isotropic Frobenius prior, exact Gaussian conditioning on Σ_j G[j, diag_col] = 0
    # corresponds to orthogonal projection subtracting the trace mean across the diagonal!
    A_mats = [zeros(Float64, N, N) for _ in 1:n_bands]
    B_mats = [zeros(Float64, N, N) for _ in 1:n_bands]
    
    for b in 1:n_bands
        col_offset = (b - 1) * (2 * N)
        sum_A = 0.0; sum_B = 0.0
        for j in 1:N
            sum_A += G_raw[j, col_offset + 2*(j - 1) + 1]
            sum_B += G_raw[j, col_offset + 2*j]
        end
        mean_A = sum_A / N
        mean_B = sum_B / N
        for j in 1:N
            G_raw[j, col_offset + 2*(j - 1) + 1] -= mean_A
            G_raw[j, col_offset + 2*j]           -= mean_B
            A_mats[b][j, j] = G_raw[j, col_offset + 2*(j - 1) + 1]
            B_mats[b][j, j] = G_raw[j, col_offset + 2*j]
        end
        for l in 1:N, j in 1:N
            l == j && continue
            A_mats[b][l, j] = G_raw[l, col_offset + 2*(j - 1) + 1]
            B_mats[b][l, j] = G_raw[l, col_offset + 2*j]
        end
    end
    
    E_v = eigen(Symmetric(V_unconstrained))
    L_V = E_v.vectors * Diagonal(sqrt.(max.(E_v.values, 1e-8)))
    
    Y_pred = X_rel_stacked * G_raw'
    res_rel = Y_target_rel - Y_pred
    df_eff = max(n_samples - gamma_rel, 1.0)
    Sigma_eps = (res_rel' * res_rel) / df_eff
    
    E_sig = eigen(Symmetric(Sigma_eps))
    Sigma_eps_repaired = E_sig.vectors * Diagonal(max.(E_sig.values, 1e-8)) * E_sig.vectors'
    L_Sigma = Matrix(cholesky(Symmetric(Sigma_eps_repaired)).L)
    
    tr_A = sum(sum(diag(A_mats[b])) for b in 1:n_bands)
    tr_B = sum(sum(diag(B_mats[b])) for b in 1:n_bands)
    C = build_trace_constraint_matrix(N, n_bands)
    
    ResponseOperator(g_macro_vec, post_cov_m, A_mats, B_mats, G_raw, L_Sigma, L_V, C, alpha_m, alpha_rel, tr_A, tr_B)
end

function predict_modes(resp::ResponseOperator, B_m_t::AbstractVector{Float64}, B_rel_t::Vector{<:AbstractVector{Float64}};
                       sample_posterior = false, rng = Random.default_rng())
    N = size(resp.G_raw_mean, 1)
    n_bands = length(BANDS)
    
    g_m = resp.G_macro
    if sample_posterior
        U_m = cholesky(Symmetric(resp.post_cov_m)).U
        g_m = g_m .+ U_m' * randn(rng, length(g_m))
    end
    mu_m = dot(g_m, B_m_t)
    
    P_features = 2 * n_bands * N
    x_features = zeros(Float64, P_features)
    col = 1
    for b in 1:n_bands
        col_q = 2b - 1
        col_p = 2b
        for j in 1:N
            x_features[col]     = B_rel_t[j][col_q]
            x_features[col + 1] = B_rel_t[j][col_p]
            col += 2
        end
    end
    
    if !sample_posterior
        mu_rel = resp.G_raw_mean * x_features
    else
        Z_rand = randn(rng, N, P_features)
        G_sample = resp.G_raw_mean .+ resp.L_Sigma_rel * Z_rand * resp.L_V_rel'
        # Project each sample strictly onto trace neutral manifold
        for b in 1:n_bands
            col_offset = (b - 1) * (2 * N)
            sum_A = 0.0; sum_B = 0.0
            for j in 1:N
                sum_A += G_sample[j, col_offset + 2*(j - 1) + 1]
                sum_B += G_sample[j, col_offset + 2*j]
            end
            mean_A = sum_A / N
            mean_B = sum_B / N
            for j in 1:N
                G_sample[j, col_offset + 2*(j - 1) + 1] -= mean_A
                G_sample[j, col_offset + 2*j]           -= mean_B
            end
        end
        mu_rel = G_sample * x_features
    end
    
    (; mu_m, mu_rel)
end
