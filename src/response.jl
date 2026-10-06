"""
KTrader V1.0: Response Layer
- Multi-scale causal path basis functions B_m[H_t] over cumulative price coordinates
- Full multivariate response operator G_b = A_b + i B_b:
    A_b: cross-mode displacement / mean-reversion matrix
    B_b: cross-mode velocity / momentum rotation matrix
- Matrix-Normal Posterior on Trace-Neutral Subspace:
    tr(A_b) = 0, tr(B_b) = 0 enforced strictly on the parameter support.
    G | Y ~ MN(G_hat, Sigma_eps, V)
- Matrix-Free Woodbury Identity:
    (X'X + αI)⁻¹ X' = X' (XX' + αI)⁻¹
    Prevents P × P memory allocation explosion when P = 14K reaches thousands.
- Empirical Bayes Evidence: alpha is determined analytically from data, zero hand-tuned parameters.
"""

"""
Construct causal multi-scale path basis functions for a 1D integrated price coordinate X(t):
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
        scale = max(s[BANDCOL[idx]], 1e-6)
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

function build_path_basis(m::AbstractVector{Float64}, z::AbstractMatrix{Float64}, s_macro::AbstractVector{Float64}, s_rel::AbstractMatrix{Float64})
    T = length(m)
    K_rel = size(z, 2)
    X_m = cumsum(m)
    X_rel = cumsum(z, dims=1)
    
    B_m = path_basis_1d(X_m, s_macro)
    B_rel = [path_basis_1d(view(X_rel, :, k), view(s_rel, k, :)) for k in 1:K_rel]
    (; B_m, B_rel)
end

struct ResponseOperator
    G_macro::Vector{Float64}                # Length 2 * n_bands
    post_cov_m::Matrix{Float64}             # Covariance of macro parameter posterior
    A_matrices::Vector{Matrix{Float64}}     # Analytic posterior mean A_b
    B_matrices::Vector{Matrix{Float64}}     # Analytic posterior mean B_b
    G_raw_mean::Matrix{Float64}             # Posterior mean: K_rel × P_features
    L_Sigma_rel::Matrix{Float64}            # Cholesky of Sigma_eps
    inv_V_factor::Matrix{Float64}           # Cholesky factor of V = (X'X + alpha*I)^-1
    alpha_macro::Float64                    # Evidence-optimised macro prior precision
    alpha_rel::Float64                      # Evidence-optimised relative prior precision
    trace_real::Float64                     # Σ_b tr(A_b) ≡ 0
    trace_imag::Float64                     # Σ_b tr(B_b) ≡ 0
end

"""
MacKay Empirical Bayes Evidence Maximization for operator precision alpha:
    gamma = Σ λ_i / (alpha + λ_i)
    alpha <- gamma / ||G||_F^2
"""
function optimize_evidence_alpha(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}; iters = 15)
    n_samples, P = size(X)
    K = size(Y, 2)
    
    # Use dual representation if P > n_samples (Woodbury)
    if P <= n_samples
        XtX = X' * X
        XtY = X' * Y
        ev = max.(eigvals(Symmetric(XtX)), 0.0)
        alpha = 1.0
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
        return alpha
    else
        # Dual Woodbury: XX' is n_samples × n_samples
        XXt = X * X'
        ev = max.(eigvals(Symmetric(XXt)), 0.0)
        alpha = 1.0
        for _ in 1:iters
            inv_W = inv(Symmetric(XXt + alpha * I))
            # G_hat' = (1/alpha) * (X'Y - X' * inv_W * (XXt * Y / alpha))
            G_hat = (Y' * inv_W * X) ./ alpha
            g_sq = sum(abs2, G_hat)
            gamma = sum(ev ./ (ev .+ alpha))
            alpha_new = clamp((gamma * K) / max(g_sq, 1e-12), 1e-4, 1e6)
            if abs(log(alpha_new) - log(alpha)) < 1e-3
                alpha = alpha_new; break
            end
            alpha = alpha_new
        end
        return alpha
    end
end

"""
Joint multivariate Bayesian estimation of the full response operator G.
Uses Woodbury dual inversion when P_features > n_samples to prevent P × P memory explosion.
"""
function fit_response_operator(B_m, B_rel, y_m, y_rel; ridge_alpha = nothing)
    T = length(y_m)
    K_rel = length(B_rel)
    n_bands = length(BANDS)
    ts = WARMUP:T-1
    n_samples = length(ts)
    P_features = 2 * n_bands * K_rel
    
    # 1. Macro Response
    X_m = B_m[ts, :]
    y_target_m = y_m[ts .+ 1]
    alpha_m = ridge_alpha === nothing ? optimize_evidence_alpha(X_m, reshape(y_target_m, :, 1)) : ridge_alpha
    inv_m = inv(Symmetric(X_m' * X_m + alpha_m * I))
    g_macro_vec = inv_m * (X_m' * y_target_m)
    res_m = y_target_m - X_m * g_macro_vec
    sig2_m = max(sum(abs2, res_m) / max(n_samples - size(X_m, 2), 1), 1e-8)
    post_cov_m = Matrix(Symmetric(sig2_m * inv_m))
    
    # 2. Relative Response
    X_rel_stacked = zeros(Float64, n_samples, P_features)
    for (idx, t) in enumerate(ts)
        col = 1
        for b in 1:n_bands
            col_q = 2b - 1
            col_p = 2b
            for k in 1:K_rel
                X_rel_stacked[idx, col]     = B_rel[k][t, col_q]
                X_rel_stacked[idx, col + 1] = B_rel[k][t, col_p]
                col += 2
            end
        end
    end
    
    Y_target_rel = y_rel[ts .+ 1, :]
    alpha_rel = ridge_alpha === nothing ? optimize_evidence_alpha(X_rel_stacked, Y_target_rel) : ridge_alpha
    
    # Fast solve for G_raw: K_rel × P_features
    if P_features <= n_samples
        XtX_rel = X_rel_stacked' * X_rel_stacked
        V_rel = inv(Symmetric(XtX_rel + alpha_rel * I))
        G_raw = (V_rel * (X_rel_stacked' * Y_target_rel))'
        E_v = eigen(Symmetric(V_rel))
        V_rep = E_v.vectors * Diagonal(max.(E_v.values, 1e-8)) * E_v.vectors'
        L_V = Matrix(cholesky(Symmetric(V_rep)).L)
    else
        # Woodbury dual representation: XXt is n_samples × n_samples
        inv_W = inv(Symmetric(X_rel_stacked * X_rel_stacked' + alpha_rel * I))
        G_raw = (Y_target_rel' * inv_W * X_rel_stacked)
        # Low-rank factor for sampling: V ≈ (1/alpha) * I - (1/alpha^2) * X' * inv_W * X
        # Diagonal approximation for massive P
        L_V = fill(1.0 / sqrt(alpha_rel), P_features, 1)
    end
    
    # Enforce exact trace neutrality on the mean operator across all bands
    A_mats = [zeros(Float64, K_rel, K_rel) for _ in 1:n_bands]
    B_mats = [zeros(Float64, K_rel, K_rel) for _ in 1:n_bands]
    
    for b in 1:n_bands
        col_offset = (b - 1) * (2 * K_rel)
        for l in 1:K_rel
            for k in 1:K_rel
                idx_q = col_offset + 2*(k - 1) + 1
                idx_p = col_offset + 2*k
                A_mats[b][l, k] = G_raw[l, idx_q]
                B_mats[b][l, k] = G_raw[l, idx_p]
            end
        end
        diag_mean_A = sum(diag(A_mats[b])) / K_rel
        diag_mean_B = sum(diag(B_mats[b])) / K_rel
        for k in 1:K_rel
            A_mats[b][k, k] -= diag_mean_A
            B_mats[b][k, k] -= diag_mean_B
            idx_q = col_offset + 2*(k - 1) + 1
            idx_p = col_offset + 2*k
            G_raw[k, idx_q] -= diag_mean_A
            G_raw[k, idx_p] -= diag_mean_B
        end
    end
    
    # 3. Residual Covariance Sigma_eps
    Y_pred = X_rel_stacked * G_raw'
    res_rel = Y_target_rel - Y_pred
    Sigma_eps = (res_rel' * res_rel) / max(n_samples - min(P_features, n_samples - 1), 1)
    
    E_sig = eigen(Symmetric(Sigma_eps))
    Sigma_eps_repaired = E_sig.vectors * Diagonal(max.(E_sig.values, 1e-8)) * E_sig.vectors'
    L_Sigma = Matrix(cholesky(Symmetric(Sigma_eps_repaired)).L)
    
    tr_A = sum(sum(diag(A_mats[b])) for b in 1:n_bands)
    tr_B = sum(sum(diag(B_mats[b])) for b in 1:n_bands)
    
    ResponseOperator(g_macro_vec, post_cov_m, A_mats, B_mats, G_raw, L_Sigma, L_V, alpha_m, alpha_rel, tr_A, tr_B)
end

function predict_modes(resp::ResponseOperator, B_m_t::AbstractVector{Float64}, B_rel_t::Vector{<:AbstractVector{Float64}};
                       sample_posterior = false, rng = Random.default_rng())
    K_rel = size(resp.G_raw_mean, 1)
    n_bands = length(BANDS)
    
    g_m = resp.G_macro
    if sample_posterior
        U_m = cholesky(Symmetric(resp.post_cov_m)).U
        g_m = g_m .+ U_m' * randn(rng, length(g_m))
    end
    mu_m = dot(g_m, B_m_t)
    
    P_features = 2 * n_bands * K_rel
    x_features = zeros(Float64, P_features)
    col = 1
    for b in 1:n_bands
        col_q = 2b - 1
        col_p = 2b
        for k in 1:K_rel
            x_features[col]     = B_rel_t[k][col_q]
            x_features[col + 1] = B_rel_t[k][col_p]
            col += 2
        end
    end
    
    if !sample_posterior
        mu_rel = resp.G_raw_mean * x_features
    else
        if size(resp.inv_V_factor, 2) == P_features
            Z_rand = randn(rng, K_rel, P_features)
            G_sample = resp.G_raw_mean .+ resp.L_Sigma_rel * Z_rand * resp.inv_V_factor'
        else
            # Woodbury fast diagonal perturbation
            Z_rand = randn(rng, K_rel, P_features)
            G_sample = resp.G_raw_mean .+ (resp.L_Sigma_rel * Z_rand) .* resp.inv_V_factor'
        end
        
        # Exact trace neutrality on every sample
        for b in 1:n_bands
            col_offset = (b - 1) * (2 * K_rel)
            sum_diag_q = 0.0
            sum_diag_p = 0.0
            for k in 1:K_rel
                sum_diag_q += G_sample[k, col_offset + 2*(k - 1) + 1]
                sum_diag_p += G_sample[k, col_offset + 2*k]
            end
            mean_q = sum_diag_q / K_rel
            mean_p = sum_diag_p / K_rel
            for k in 1:K_rel
                G_sample[k, col_offset + 2*(k - 1) + 1] -= mean_q
                G_sample[k, col_offset + 2*k]           -= mean_p
            end
        end
        mu_rel = G_sample * x_features
    end
    
    (; mu_m, mu_rel)
end
