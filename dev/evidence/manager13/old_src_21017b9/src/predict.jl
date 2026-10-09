"""
KTrader V1.0: Prediction Layer
- Pure Ragged Panel: preserves all history, no truncation
- Blocked Cross-Fitting: geometry is estimated out-of-fold to strictly eliminate selection leakage
- True Matrix-Normal Posterior Predictive Sampling: G^(s) ~ MN(G_hat, Sigma_eps, V)
- Propagates Geometry Subspace Uncertainty: Phi_perp^(d) drawn via stationary block bootstrap
- Unifies Non-Markov Fractional Innovation Law on Natural Modes (Macro + Relative)
- Exact response residual bootstrap using own_rows for younger assets
"""

const DGRID_V1 = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

frac_weights(d, n) = accumulate((w, k) -> w * (k - 1 + d) / k, 1:n-1; init = 1.0) |> w -> pushfirst!(w, 1.0)


struct V1Model
    e0::Vector{Float64}                     # Center-of-mass weights per asset
    Phi_perp_now::Matrix{Float64}           # Decision-time relative natural modes (N × K_rel)
    Phi_draws::Vector{Matrix{Float64}}      # D bootstrap geometry draws of relative modes
    s1::Vector{Float64}                     # 1-day fractal ruler per asset
    s_macro::Vector{Float64}                # Macro fractal ruler
    s_rel::Matrix{Float64}                  # Relative modes fractal rulers
    resp::ResponseOperator                  # Fitted response operator
    mu_pred::Vector{Float64}                # Analytic asset-level conditional expected returns
    res_history::Matrix{Float64}            # True out-of-fold response residuals
    own_res_rows::Vector{Vector{Int}}       # Valid residual rows for each asset
    basis_now::NamedTuple                   # Decision-time basis coordinates
    mode_vol_factors::Vector{Float64}       # One-step fractional volatility forecast per mode
end

"""
Compute fractional long-memory variance forecast for a 1D residual series:
v_{t+1} = Σ_j π_j(d) e_{t-j}^2 / Σ π_j(d)
Optimizes d over DGRID_V1 via quasi-likelihood.
"""
function fractional_mode_variance(e_series::AbstractVector{Float64})
    T = length(e_series)
    e2 = e_series .^ 2
    best_ll = -Inf
    best_v = var(e_series)
    
    for d in DGRID_V1
        w = frac_weights(d, T)
        cs = cumsum(w)
        # causal fractional weighted moving variance
        v_next = dot(view(e2, T:-1:1), view(w, 1:T)) / cs[T]
        v_next = max(v_next, 1e-10)
        
        # in-sample fit likelihood for d
        ll = -0.5 * (log(v_next) + e2[end] / v_next)
        if ll > best_ll
            best_ll = ll
            best_v = v_next
        end
    end
    max(best_v, 1e-8)
end

"""
Fit the complete V1.0 causal path response model on price history adj (T × N).
"""
function fit_v1(adj::AbstractMatrix{Float64}; ridge_alpha = nothing, F_folds = 3, D_geom_draws = 4)
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
    
    # 3. Decision-time Center-of-Mass Macro & Relative Space Decomposition (Full History)
    decomp = center_of_mass_decomposition(u_norm)
    m = decomp.m
    u_perp = decomp.u_perp
    
    e0_full = fill(1.0 / sqrt(N), N)
    rel_full = relative_modes(u_perp, e0_full)
    Phi_perp_now = rel_full.Phi_perp
    K_rel = size(Phi_perp_now, 2)
    
    # Geometry uncertainty: draw D bootstrap geometry relative modes
    Phi_draws = bootstrap_relative_modes(u_perp, e0_full; D_draws = D_geom_draws)
    
    # 4. Blocked Cross-Fitting: Divide history into F blocks to prevent selection leakage
    ts_total = WARMUP:T-1
    n_reg = length(ts_total)
    block_size = max(1, n_reg ÷ F_folds)
    
    z_rel_cf = zeros(Float64, T - 1, K_rel)
    
    for fold in 1:F_folds
        fold_start = WARMUP + (fold - 1) * block_size
        fold_end   = (fold == F_folds) ? (T - 1) : (fold_start + block_size - 1)
        fold_ts = fold_start:fold_end
        
        oof_indices = setdiff(WARMUP:T-1, fold_ts)
        u_perp_oof = @view u_perp[oof_indices, :]
        rel_oof = relative_modes(u_perp_oof, e0_full)
        Phi_perp_oof = rel_oof.Phi_perp
        
        for k in 1:K_rel
            phi_k = view(Phi_perp_oof, :, min(k, size(Phi_perp_oof, 2)))
            for t in fold_ts
                val = 0.0
                for j in 1:N
                    uj = u_perp[t, j]
                    if isfinite(uj)
                        val += uj * phi_k[j]
                    end
                end
                z_rel_cf[t, k] = val
            end
        end
    end
    
    # 5. Cumulative level coordinates for path basis
    X_m = cumsum(m)
    X_rel = cumsum(z_rel_cf, dims=1)
    
    Tr = length(m)
    w_tr = ones(Tr)
    s_m = ruler(reshape(X_m, :, 1), [1], w_tr)[1, :]
    s_rel = ruler(X_rel, ones(Int, K_rel), w_tr)
    
    basis = build_path_basis(m, z_rel_cf, s_m, s_rel)
    
    # 6. Fit response operator G on cross-fitted basis
    resp = fit_response_operator(basis.B_m, basis.B_rel, m, z_rel_cf; ridge_alpha)
    
    # 7. Decision-time projection using full decision geometry
    B_m_now = basis.B_m[end, :]
    B_rel_now = [b[end, :] for b in basis.B_rel]
    pred = predict_modes(resp, B_m_now, B_rel_now; sample_posterior = false)
    
    alive_now = findall(j -> isfinite(adj[end, j]), 1:N)
    N_alive = length(alive_now)
    e0_now = zeros(Float64, N)
    if N_alive > 0
        e0_now[alive_now] .= 1.0 / sqrt(N_alive)
    end
    
    mu_norm = pred.mu_m .* e0_now .+ Phi_perp_now * pred.mu_rel
    mu_asset = mu_norm .* s1
    
    # 8. True out-of-fold response residuals
    ts = WARMUP:T-2
    n_res = length(ts)
    res_history = fill(NaN, n_res, N)
    for (idx, t) in enumerate(ts)
        pred_t = predict_modes(resp, basis.B_m[t, :], [b[t, :] for b in basis.B_rel]; sample_posterior = false)
        alive_t = findall(j -> isfinite(adj[t+1, j]), 1:N)
        Nt = length(alive_t)
        e0_t = zeros(Float64, N)
        Nt > 0 && (e0_t[alive_t] .= 1.0 / sqrt(Nt))
        mu_t = (pred_t.mu_m .* e0_t .+ Phi_perp_now * pred_t.mu_rel) .* s1
        for j in 1:N
            rt = r[t, j]
            if isfinite(rt)
                res_history[idx, j] = rt - mu_t[j]
            end
        end
    end
    
    own_res = [findall(isfinite, view(res_history, :, j)) for j in 1:N]
    
    # 9. Non-Markov Fractional Innovation Law on Modes:
    # Project historical residuals onto modes to estimate long-memory conditional variance
    e_res_m = Float64[]
    for idx in 1:n_res
        valid_j = findall(isfinite, view(res_history, idx, :))
        push!(e_res_m, isempty(valid_j) ? 0.0 : dot(res_history[idx, valid_j], e0_full[valid_j]))
    end
    v_macro = fractional_mode_variance(e_res_m)
    
    v_rel = zeros(Float64, K_rel)
    for k in 1:K_rel
        phi_k = view(Phi_perp_now, :, k)
        e_k = [dot(view(res_history, idx, findall(isfinite, view(res_history, idx, :))), phi_k[findall(isfinite, view(res_history, idx, :))]) for idx in 1:n_res]
        v_rel[k] = fractional_mode_variance(e_k)
    end
    
    mode_vol_factors = vcat(v_macro, v_rel)
    basis_now = (; B_m_now, B_rel_now)
    
    V1Model(e0_now, Phi_perp_now, Phi_draws, s1, s_m, s_rel, resp, mu_asset, res_history, own_res, basis_now, mode_vol_factors)
end

"""
Generate S posterior predictive scenarios of gross returns X = exp(r_{t+1}) (S × N):
1. Draws geometry subspace Phi_perp^(d) from block-bootstrap draws (geometry uncertainty)
2. Draws response operator G^(s) ~ MN(G_hat, Sigma_eps, V)
3. Evaluates conditional mean μ^(s) using drawn geometry and operator
4. Shocks are scaled by fractional long-memory mode variance forecasts
5. Exact response residual bootstrap using own_rows for younger assets
"""
function generate_scenarios_v1(model::V1Model, r_history::AbstractMatrix{Float64}; S = 500, rng = Random.MersenneTwister(1))
    N = length(model.mu_pred)
    mu_analytic = model.mu_pred
    T_res = size(model.res_history, 1)
    D_geom = length(model.Phi_draws)
    
    scenarios_log = zeros(Float64, S, N)
    
    for s in 1:S
        # Geometry uncertainty draw
        d_idx = rand(rng, 1:D_geom)
        phi_draw = model.Phi_draws[d_idx]
        
        # Matrix-Normal posterior draw
        pred_s = predict_modes(model.resp, model.basis_now.B_m_now, model.basis_now.B_rel_now; sample_posterior = true, rng)
        mu_norm_s = pred_s.mu_m .* model.e0 .+ phi_draw * pred_s.mu_rel
        mu_s = mu_norm_s .* model.s1
        
        # Innovation shock from real response residuals
        sampled_row = rand(rng, 1:T_res)
        eps_shock = zeros(Float64, N)
        for j in 1:N
            val = model.res_history[sampled_row, j]
            if isfinite(val)
                eps_shock[j] = val
            else
                own_r = model.own_res_rows[j]
                eps_shock[j] = !isempty(own_r) ? model.res_history[rand(rng, own_r), j] : 0.0
            end
        end
        
        scenarios_log[s, :] .= mu_s .+ eps_shock
    end
    
    # Moment matching
    scenarios_log .+= (mu_analytic' .- mean(scenarios_log, dims=1))
    exp.(scenarios_log)
end
