"""
KTrader V1.0: Prediction Layer
- Pure Ragged Panel with Fixed Helmert Gauge Q: preserves all history, zero eigenvector tracking artifacts
- Gauge-Invariant Scalar Ruler s_perp(tau) per band
- True Out-of-Fold (OOF) Response Residuals: epsilon_t = r_{t+1} - mu_t^{(-fold)}
- True Absolute Fractional Innovation Scaling: scale_d = sqrt(v_{T+1}(d) / v_bootstrap)
- Exact Matrix-Free Matrix-Normal Posterior Predictive Sampling: G^(s) ~ MN(G_hat, Sigma_eps, V)
"""

const DGRID_V1 = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

frac_weights(d, n) = accumulate((w, k) -> w * (k - 1 + d) / k, 1:n-1; init = 1.0) |> w -> pushfirst!(w, 1.0)

struct V1Model
    e0::Vector{Float64}                     # Center-of-mass weights per asset
    Q_helmert::Matrix{Float64}              # Fixed Helmert basis (N × (N-1))
    s1::Vector{Float64}                     # 1-day fractal ruler per asset
    s_macro::Vector{Float64}                # Macro fractal ruler
    s_perp::Vector{Float64}                 # Gauge-invariant scalar relative ruler per band
    resp::ResponseOperator                  # Fitted response operator
    mu_pred::Vector{Float64}                # Analytic asset-level conditional expected returns
    res_history::Matrix{Float64}            # True out-of-fold response residuals
    own_res_rows::Vector{Vector{Int}}       # Valid residual rows for each asset
    basis_now::NamedTuple                   # Decision-time basis coordinates
    d_posterior::Vector{Float64}            # Posterior probabilities over DGRID_V1
    v_forecasts::Vector{Float64}            # Forecasted conditional variance per grid point d
    v_bootstrap::Float64                    # Historical variance of macro residual
end

"""
Causal sequential log-likelihood for fractional long-memory parameter d on residuals:
v_t(d) = Σ_{j=1}^{t-1} π_{t-j}(d) e_j² / Σ π
ll(d) = -1/2 Σ_t [log v_t(d) + e_t² / v_t(d)]
"""
function causal_fractional_posterior(e_series::AbstractVector{Float64}; burn = 30)
    T = length(e_series)
    e2 = e_series .^ 2
    G = length(DGRID_V1)
    ll = zeros(Float64, G)
    v_forecasts = zeros(Float64, G)
    
    for (g, d) in enumerate(DGRID_V1)
        w = frac_weights(d, T)
        cs = cumsum(w)
        v_seq = zeros(Float64, T)
        v_seq[1] = max(e2[1], 1e-8)
        for t in 2:T
            v_seq[t] = max(dot(view(e2, t-1:-1:1), view(w, 1:t-1)) / cs[t-1], 1e-10)
        end
        ll[g] = -0.5 * sum(log(v_seq[t]) + e2[t] / v_seq[t] for t in burn:T)
        v_forecasts[g] = max(dot(view(e2, T:-1:1), view(w, 1:T)) / cs[T], 1e-10)
    end
    
    p_d = exp.(ll .- maximum(ll))
    p_d ./= sum(p_d)
    
    (; p_d, v_forecasts)
end

"""
Fit the complete V1.0 causal path response model on price history adj (T × N).
Zero hand-tuned parameters: ridge_alpha defaults to nothing (Empirical Bayes Evidence).
"""
function fit_v1(adj::AbstractMatrix{Float64}; ridge_alpha = nothing, F_folds = 3)
    T, N = size(adj)
    log_adj = log.(adj)
    f = [something(findfirst(isfinite, view(adj, :, j)), T + 1) for j in 1:N]
    
    # 1. Fractal rulers for each asset on its OWN complete available history
    w_ones = ones(T)
    s_raw = ruler(log_adj, f, w_ones)
    s1 = s_raw[:, 1]
    
    # 2. Daily log returns (T-1 × N)
    r = fill(NaN, T - 1, N)
    for j in 1:N
        for t in (f[j] + 1):T
            p1 = adj[t, j]; p0 = adj[t - 1, j]
            if isfinite(p1) && isfinite(p0)
                r[t - 1, j] = log(p1) - log(p0)
            end
        end
    end
    
    # Normalized increments
    u_norm = fill(NaN, T - 1, N)
    for j in 1:N
        scale = max(s1[j], 1e-6)
        for t in 1:(T - 1)
            if isfinite(r[t, j])
                u_norm[t, j] = r[t, j] / scale
            end
        end
    end
    
    # 3. Ragged Center-of-Mass Macro Decomposition
    decomp = center_of_mass_decomposition(u_norm)
    m = decomp.m
    u_perp = decomp.u_perp
    
    # 4. Fixed Numerical Helmert Gauge Q (N × (N-1))
    Q_helmert = helmert_basis(N)
    K_rel = N - 1
    
    helm = project_helmert_gauge(u_perp, Q_helmert)
    z_rel = helm.z_rel
    
    # 5. Cumulative level coordinates & Gauge-Invariant Scalar Ruler s_perp(τ)
    X_m = cumsum(m)
    X_rel = cumsum(z_rel, dims=1)
    
    Tr = length(m)
    w_tr = ones(Tr)
    s_m = ruler(reshape(X_m, :, 1), [1], w_tr)[1, :]
    
    # Scalar ruler per band: s_perp(τ)^2 = (1 / (N-1)) * tr(C_Q(τ))
    # Ensures exact rotation covariance and asset permutation invariance!
    s_perp = zeros(Float64, length(BANDS))
    for (b_idx, τ) in enumerate(BANDS)
        # trace of relative increments at band τ
        u_tau = fill(NaN, T - 1, N)
        for j in 1:N
            scale = max(s1[j], 1e-6)
            for t in (τ + 1):(T - 1)
                p1 = adj[t, j]; p0 = adj[t - τ, j]
                if isfinite(p1) && isfinite(p0)
                    u_tau[t, j] = (log(p1) - log(p0)) / scale
                end
            end
        end
        helm_tau = project_helmert_gauge(center_of_mass_decomposition(u_tau).u_perp, Q_helmert)
        s_perp[b_idx] = sqrt(max(tr(helm_tau.C_Q) / K_rel, 1e-6))
    end
    
    basis = build_path_basis(m, z_rel, s_m, s_perp)
    
    # 6. Fit response operator G on full history using Empirical Bayes Evidence
    resp = fit_response_operator(basis.B_m, basis.B_rel, m, z_rel; ridge_alpha)
    
    # 7. True Out-of-Fold (OOF) Response Residuals via Blocked Cross-Fitting
    # G^{(-fold)} is fitted strictly on train_ts, predicting fold_ts!
    ts_total = WARMUP:T-2
    n_res = length(ts_total)
    block_size = max(1, n_res ÷ F_folds)
    res_history = fill(NaN, n_res, N)
    
    for fold in 1:F_folds
        fold_start = 1 + (fold - 1) * block_size
        fold_end   = (fold == F_folds) ? n_res : (fold_start + block_size - 1)
        fold_eval_indices = fold_start:fold_end
        
        # Truly out-of-fold training time points
        train_indices = setdiff(1:n_res, fold_eval_indices)
        train_ts = ts_total[train_indices]
        
        # Fit out-of-fold response operator G^{(-fold)} on train_ts
        resp_oof = fit_response_operator(basis.B_m, basis.B_rel, m, z_rel; ridge_alpha, ts = train_ts)
        
        # Evaluate true out-of-fold residuals for t in fold
        for idx in fold_eval_indices
            t = ts_total[idx]
            pred_t = predict_modes(resp_oof, basis.B_m[t, :], [b[t, :] for b in basis.B_rel]; sample_posterior = false)
            alive_t = findall(j -> isfinite(adj[t+1, j]), 1:N)
            Nt = length(alive_t)
            e0_t = zeros(Float64, N)
            Nt > 0 && (e0_t[alive_t] .= 1.0 / sqrt(Nt))
            mu_t = (pred_t.mu_m .* e0_t .+ Q_helmert * pred_t.mu_rel) .* s1
            for j in 1:N
                rt = r[t, j]
                if isfinite(rt)
                    res_history[idx, j] = rt - mu_t[j]
                end
            end
        end
    end
    
    own_res = [findall(isfinite, view(res_history, :, j)) for j in 1:N]
    
    # 8. Decision-time projection
    B_m_now = basis.B_m[end, :]
    B_rel_now = [b[end, :] for b in basis.B_rel]
    pred = predict_modes(resp, B_m_now, B_rel_now; sample_posterior = false)
    
    alive_now = findall(j -> isfinite(adj[end, j]), 1:N)
    N_alive = length(alive_now)
    e0_now = zeros(Float64, N)
    if N_alive > 0
        e0_now[alive_now] .= 1.0 / sqrt(N_alive)
    end
    
    mu_norm = pred.mu_m .* e0_now .+ Q_helmert * pred.mu_rel
    mu_asset = mu_norm .* s1
    
    # 9. Causal Sequential Long-Memory Variance Likelihood on Macro Residual
    e_res_m = Float64[]
    for idx in 1:n_res
        valid_j = findall(isfinite, view(res_history, idx, :))
        push!(e_res_m, isempty(valid_j) ? 0.0 : dot(res_history[idx, valid_j], e0_now[valid_j]))
    end
    frac_macro = causal_fractional_posterior(e_res_m)
    v_bootstrap = max(var(e_res_m), 1e-8)
    
    basis_now = (; B_m_now, B_rel_now)
    V1Model(e0_now, Q_helmert, s1, s_m, s_perp, resp, mu_asset, res_history, own_res, basis_now, frac_macro.p_d, frac_macro.v_forecasts, v_bootstrap)
end

"""
Generate S posterior predictive scenarios of gross returns X = exp(r_{t+1}) (S × N):
1. Draws response operator G^(s) ~ MN(G_hat, Sigma_eps, V)
2. Evaluates conditional mean μ^(s) = s1 .* (e0 * mu_m^(s) + Q_helmert * mu_perp^(s))
3. Samples fractional memory parameter d ~ p(d | H) and scales innovation shocks by true baseline
    scale_d = sqrt(v_{T+1}(d) / v_bootstrap)
4. Bootstraps true out-of-fold residuals, using own_rows for younger assets
5. Exact moment matching
"""
function generate_scenarios_v1(model::V1Model, r_history::AbstractMatrix{Float64}; S = 500, rng = Random.MersenneTwister(1))
    N = length(model.mu_pred)
    mu_analytic = model.mu_pred
    T_res = size(model.res_history, 1)
    
    scenarios_log = zeros(Float64, S, N)
    cum_d = cumsum(model.d_posterior)
    
    for s in 1:S
        # 1. Matrix-Normal posterior draw
        pred_s = predict_modes(model.resp, model.basis_now.B_m_now, model.basis_now.B_rel_now; sample_posterior = true, rng)
        mu_norm_s = pred_s.mu_m .* model.e0 .+ model.Q_helmert * pred_s.mu_rel
        mu_s = mu_norm_s .* model.s1
        
        # 2. Sample fractional long-memory exponent d from posterior mixture
        g_d = min(searchsortedfirst(cum_d, rand(rng)), length(cum_d))
        # True physical scaling relative to historical bootstrap residual variance
        vol_scale = sqrt(model.v_forecasts[g_d] / model.v_bootstrap)
        
        # 3. Innovation shock from real out-of-fold response residuals
        sampled_row = rand(rng, 1:T_res)
        eps_shock = zeros(Float64, N)
        for j in 1:N
            val = model.res_history[sampled_row, j]
            if isfinite(val)
                eps_shock[j] = val * vol_scale
            else
                own_r = model.own_res_rows[j]
                eps_shock[j] = !isempty(own_r) ? model.res_history[rand(rng, own_r), j] * vol_scale : 0.0
            end
        end
        
        scenarios_log[s, :] .= mu_s .+ eps_shock
    end
    
    # Exact moment matching
    scenarios_log .+= (mu_analytic' .- mean(scenarios_log, dims=1))
    exp.(scenarios_log)
end
