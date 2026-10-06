"""
KTrader V1.0 Final: Prediction Layer
- Pure Ragged Panel with Center-of-Mass & Zero-Sum Relative Space
- Gauge-Invariant Scalar Ruler s_perp(tau) per band
- True Out-of-Fold (OOF) Response Residuals: epsilon_t = r_{t+1} - mu_t^{(-fold)}
- True Absolute Fractional Innovation Scaling: scale_d = sqrt(v_{T+1}(d) / v_bootstrap)
- Exact Matrix-Free Matrix-Normal Posterior Predictive Sampling
"""

const DGRID_V1 = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

frac_weights(d, n) = accumulate((w, k) -> w * (k - 1 + d) / k, 1:n-1; init = 1.0) |> w -> pushfirst!(w, 1.0)

struct V1Model
    e0::Vector{Float64}                     # Center-of-mass weights per asset
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
    
    Δd = zeros(Float64, G)
    Δd[1] = (DGRID_V1[2] - DGRID_V1[1]) / 2.0
    Δd[end] = (DGRID_V1[end] - DGRID_V1[end-1]) / 2.0
    for g in 2:(G - 1)
        Δd[g] = (DGRID_V1[g + 1] - DGRID_V1[g - 1]) / 2.0
    end
    
    log_weights = ll .+ log.(Δd)
    p_d = exp.(log_weights .- maximum(log_weights))
    p_d ./= sum(p_d)
    
    (; p_d, v_forecasts)
end

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
    u_perp = decomp.u_perp # N-dimensional relative field, row-sum strictly 0!
    
    # Fill leading NaNs with 0 in u_perp for continuous cumulative level integration
    u_perp_clean = ifelse.(isfinite.(u_perp), u_perp, 0.0)
    
    # 4. Cumulative level coordinates & Gauge-Invariant Scalar Ruler s_perp(τ)
    X_m = cumsum(m)
    X_rel = cumsum(u_perp_clean, dims=1)
    
    Tr = length(m)
    w_tr = ones(Tr)
    s_m = ruler(reshape(X_m, :, 1), [1], w_tr)[1, :]
    
    s_perp = zeros(Float64, length(BANDS))
    for (b_idx, τ) in enumerate(BANDS)
        tr_val = 0.0
        for j in 1:N
            col = view(X_rel, :, j)
            acc = 0.0; cnt = 0
            @inbounds @simd for t in (τ + 1):(T - 1)
                d = col[t] - col[t - τ]
                acc += d * d
                cnt += 1
            end
            tr_val += acc / max(cnt, 1)
        end
        s_perp[b_idx] = sqrt(max(tr_val / max(N - 1, 1), 1e-6))
    end
    
    basis = build_path_basis(m, X_rel, s_m, s_perp)
    
    # 5. Fit response operator G on full history using Empirical Bayes Evidence
    resp = fit_response_operator(basis.B_m, basis.B_rel, m, u_perp_clean; ridge_alpha)
    
    # 6. True Out-of-Fold (OOF) Response Residuals via Blocked Cross-Fitting
    ts_total = WARMUP:T-2
    n_res = length(ts_total)
    block_size = max(1, n_res ÷ F_folds)
    res_history = fill(NaN, n_res, N)
    
    e0_full = fill(1.0 / sqrt(N), N)
    P0 = I - e0_full * e0_full'
    
    for fold in 1:F_folds
        fold_start = 1 + (fold - 1) * block_size
        fold_end   = (fold == F_folds) ? n_res : (fold_start + block_size - 1)
        fold_eval_indices = fold_start:fold_end
        
        train_indices = setdiff(1:n_res, fold_eval_indices)
        train_ts = ts_total[train_indices]
        
        resp_oof = fit_response_operator(basis.B_m, basis.B_rel, m, u_perp_clean;
                                         ridge_alpha = resp.alpha_rel, ts = train_ts)
        
        for idx in fold_eval_indices
            t = ts_total[idx]
            alive_t = findall(j -> isfinite(adj[t+1, j]), 1:N)
            Nt = length(alive_t)
            e0_t = zeros(Float64, N)
            Nt > 0 && (e0_t[alive_t] .= 1.0 / sqrt(Nt))
            
            pred_t = predict_modes(resp_oof, basis.B_m[t, :], [b[t, :] for b in basis.B_rel]; sample_posterior = false)
            # Project relative prediction onto P0 zero-sum manifold
            mu_rel_proj = P0 * pred_t.mu_rel
            mu_t = (pred_t.mu_m .* e0_t .+ mu_rel_proj) .* s1
            for j in 1:N
                rt = r[t, j]
                if isfinite(rt)
                    res_history[idx, j] = rt - mu_t[j]
                end
            end
        end
    end
    
    own_res = [findall(isfinite, view(res_history, :, j)) for j in 1:N]
    
    # 7. Decision-time projection
    B_m_now = basis.B_m[end, :]
    B_rel_now = [b[end, :] for b in basis.B_rel]
    pred = predict_modes(resp, B_m_now, B_rel_now; sample_posterior = false)
    
    alive_now = findall(j -> isfinite(adj[end, j]), 1:N)
    N_alive = length(alive_now)
    e0_now = zeros(Float64, N)
    if N_alive > 0
        e0_now[alive_now] .= 1.0 / sqrt(N_alive)
    end
    
    mu_rel_proj = P0 * pred.mu_rel
    mu_norm = pred.mu_m .* e0_now .+ mu_rel_proj
    mu_asset = mu_norm .* s1
    
    # 8. Causal Sequential Long-Memory Variance Likelihood on Macro Residual
    e_res_m = Float64[]
    for idx in 1:n_res
        valid_j = findall(isfinite, view(res_history, idx, :))
        Nt = length(valid_j)
        push!(e_res_m, Nt > 0 ? sum(res_history[idx, valid_j]) / sqrt(Nt) : 0.0)
    end
    frac_macro = causal_fractional_posterior(e_res_m)
    v_bootstrap = max(var(e_res_m), 1e-8)
    
    basis_now = (; B_m_now, B_rel_now)
    V1Model(e0_now, s1, s_m, s_perp, resp, mu_asset, res_history, own_res, basis_now, frac_macro.p_d, frac_macro.v_forecasts, v_bootstrap)
end

function generate_scenarios_v1(model::V1Model, r_history::AbstractMatrix{Float64}; S = 500, rng = Random.MersenneTwister(1))
    N = length(model.mu_pred)
    mu_analytic = model.mu_pred
    T_res = size(model.res_history, 1)
    
    scenarios_log = zeros(Float64, S, N)
    cum_d = cumsum(model.d_posterior)
    P0 = I - fill(1.0 / N, N, N)
    
    for s in 1:S
        pred_s = predict_modes(model.resp, model.basis_now.B_m_now, model.basis_now.B_rel_now; sample_posterior = true, rng)
        mu_rel_s = P0 * pred_s.mu_rel
        mu_norm_s = pred_s.mu_m .* model.e0 .+ mu_rel_s
        mu_s = mu_norm_s .* model.s1
        
        g_d = min(searchsortedfirst(cum_d, rand(rng)), length(cum_d))
        vol_scale = sqrt(model.v_forecasts[g_d] / model.v_bootstrap)
        
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
    
    scenarios_log .+= (mu_analytic' .- mean(scenarios_log, dims=1))
    exp.(scenarios_log)
end
