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

const DELTA_D_V1 = [0.025, 0.075, 0.15, 0.2, 0.2, 0.2, 0.1]

struct V1Model
    active_indices::Vector{Int}             # Indices of active assets in universe
    N_universe::Int                         # Total universe columns
    e0::Vector{Float64}                     # Center-of-mass weights (length N_active)
    s1::Vector{Float64}                     # 1-day fractal ruler (length N_active)
    s_macro::Vector{Float64}                # Macro fractal ruler
    s_perp::Vector{Float64}                 # Relative scalar ruler per band
    relative_observed::BitMatrix            # Actual observations; field padding is not a return
    resp::ResponseOperator                  # Trace-conditioned response operator
    mu_pred::Vector{Float64}                # Conditional expected returns (length N_universe)
    res_history::Matrix{Float64}            # True OOF residuals (n_res × N_active)
    own_res_rows::Vector{Vector{Int}}       # Valid residual rows per active asset
    pred_moments::NamedTuple                # Decision-time moments (μ_m, var_m, μ_rel, L_rel)
    d_posterior::Vector{Float64}            # Posterior probabilities over DGRID_V1
    v_forecasts::Vector{Float64}            # Forecasted conditional variance per grid point d
    v_bootstrap::Float64                    # Historical variance of macro residual
end

"""Observed relative field embedded in the current asset coordinate space.
Absent components are defined to be zero in the field, not in the return data.
The separate observation mask is retained. This is the explicit zero-embedding
model, not a Gaussian likelihood marginalizing unknown asset returns.
"""
function embedded_relative_field(r::AbstractMatrix{Float64},s1::AbstractVector{Float64})
    T,N=size(r)
    observed=isfinite.(r)
    macro_flow=zeros(T)
    embedded=zeros(T,N)
    inv_scale=1.0 ./ max.(s1,1e-6)
    @inbounds for t in 1:T
        total=0.0; count=0
        for j in 1:N
            if observed[t,j]
                total+=r[t,j]*inv_scale[j]
                count+=1
            end
        end
        if count>0
            macro_flow[t]=total/sqrt(count)
            shift=total/count
            for j in 1:N
                observed[t,j] && (embedded[t,j]=r[t,j]*inv_scale[j]-shift)
            end
        end
    end
    (; macro_flow,embedded,observed)
end

"""
Causal fractional volatility posterior computed via fast FFT convolution:
Reduces O(|D| T^2) direct sums to O(|D| T log T).
"""
function causal_fractional_posterior(e_series::AbstractVector{Float64}; burn = 30)
    T = length(e_series)
    T > 0 || throw(ArgumentError("fractional posterior needs observations"))
    burn >= 2 || throw(ArgumentError("burn must be at least 2"))
    all(isfinite, e_series) || throw(ArgumentError("fractional residuals must be observed"))
    e2 = e_series .^ 2
    G = length(DGRID_V1)
    ll = zeros(Float64, G)
    v_forecasts = zeros(Float64, G)
    
    L = nextpow(2, 2T)
    work = fractional_fft_workspace(L)
    try
        fractional_likelihood!(ll,v_forecasts,e2,work,burn)
    finally
        release_fractional_fft_workspace(work)
    end
    
    log_weights = ll .+ log.(DELTA_D_V1)
    p_d = exp.(log_weights .- maximum(log_weights))
    p_d ./= sum(p_d)
    
    (; p_d, v_forecasts)
end

function fractional_likelihood!(ll,forecasts,e2,work::FractionalFFTWorkspace,burn)
    T=length(e2)
    fill!(work.input,0.0)
    work.input[1:T].=e2
    mul!(work.spectrum,work.forward,work.input)
    for g in eachindex(DGRID_V1)
        work.product.=work.spectrum.*work.kernels[g]
        mul!(work.output,work.inverse,work.product)
        cs=work.counts[g]
        value=0.0
        @inbounds for t in burn:T
            variance=max(work.output[t-1]/cs[t-1],1e-10)
            value-=0.5*(log(variance)+e2[t]/variance)
        end
        ll[g]=value
        forecasts[g]=max(work.output[T]/cs[T],1e-10)
    end
    nothing
end

"""
Identify active assets in the decision prefix: assets with at least one observed price.
Dummy assets (all-NaN prefix) are strictly excluded from the active coordinate space!
"""
function active_universe_indices(adj::AbstractMatrix{Float64})
    T, N = size(adj)
    active = Int[]
    for j in 1:N
        # Asset is active iff it has at least one valid daily return
        if any(t -> isfinite(adj[t, j]) && isfinite(adj[t - 1, j]), 2:T)
            push!(active, j)
        end
    end
    active
end

function fold_sufficient_statistics(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}, folds::Int;
                                   workspace=nothing)
    n,P = size(X); N = size(Y,2)
    2 <= folds <= n || throw(ArgumentError("F_folds must be in 2:n_training_rows"))
    ranges = [((f-1)*div(n,folds)+1):(f == folds ? n : f*div(n,folds)) for f in 1:folds]
    need_primal = n >= P
    xx = need_primal ? [matrix_buffer!(workspace,:fold_xx,P,P;slot=f) for f in 1:folds] : nothing
    xy = [matrix_buffer!(workspace,:fold_xy,P,N;slot=f) for f in 1:folds]
    yy = [matrix_buffer!(workspace,:fold_yy,N,N;slot=f) for f in 1:folds]
    # Parallelize fold block Gram computations across threads when inside a single decision
    for f in 1:folds
        Xf=view(X,ranges[f],:)
        Yf=view(Y,ranges[f],:)
        if need_primal
            BLAS.syrk!('U','T',1.0,Xf,0.0,xx[f])
            LinearAlgebra.copytri!(xx[f],'U')
        end
        BLAS.gemm!('T','N',1.0,Xf,Yf,0.0,xy[f])
        BLAS.syrk!('U','T',1.0,Yf,0.0,yy[f])
        LinearAlgebra.copytri!(yy[f],'U')
    end
    full_xx=need_primal ? matrix_buffer!(workspace,:full_xx,P,P) : nothing
    full_xy=matrix_buffer!(workspace,:full_xy,P,N)
    full_yy=matrix_buffer!(workspace,:full_yy,N,N)
    if need_primal
        copyto!(full_xx, xx[1])
        for f in 2:folds
            BLAS.axpy!(1.0, xx[f], full_xx)
        end
        # Only copytri! once on the final full matrix for routines that require full symmetry
        LinearAlgebra.copytri!(full_xx, 'U')
    end
    copyto!(full_xy, xy[1])
    copyto!(full_yy, yy[1])
    for f in 2:folds
        BLAS.axpy!(1.0, xy[f], full_xy)
        BLAS.axpy!(1.0, yy[f], full_yy)
    end
    (; ranges,xx,xy,yy,full_xx,full_xy,full_yy)
end

function fit_v1(adj::AbstractMatrix{Float64}; ridge_alpha = nothing, F_folds = 3, ruler_stats = nothing,
                history_cache=nothing, alpha_initial=nothing, timing=nothing, workspace=nothing)
    prep_start=time_ns()
    T_raw, N_universe = size(adj)
    active_idx = history_cache === nothing ? active_universe_indices(adj) :
                 findall(<(T_raw),history_cache.first_return)
    isempty(active_idx) && error("No active assets with price history in prefix")
    
    # ----------------------------------------------------
    # Work strictly in active asset space A_t (ragged closure)
    # ----------------------------------------------------
    adj_act = view(adj, :, active_idx)
    T, N = size(adj_act)
    log_adj = history_cache === nothing ? log.(adj_act) : view(history_cache.log_prices,1:T,active_idx)
    f = history_cache === nothing ?
        [something(findfirst(isfinite, view(adj_act, :, j)), T + 1) for j in 1:N] :
        history_cache.first_price[active_idx]
    
    # 1. Fractal rulers for each active asset
    s_raw = if ruler_stats === nothing
        ruler(log_adj, f)
    else
        ruler_from_stats(ruler_stats, T, f, active_idx)
    end
    s1 = s_raw[:, 1]
    
    # 2. Daily log returns (T-1 × N)
    r = history_cache === nothing ? diff(log_adj; dims=1) :
        view(history_cache.returns,1:T-1,active_idx)
    
    # 3. Explicit field embedding; raw r and its observation mask retain missingness.
    field=embedded_relative_field(r,s1)
    m=field.macro_flow
    relative_embedding=field.embedded
    
    # 4. Cumulative level coordinates & Gauge-Invariant Scalar Ruler s_perp(τ)
    X_m = matrix_buffer!(workspace, :X_m, length(m), 1; grow_rows=true)
    X_rel = matrix_buffer!(workspace, :X_rel, size(relative_embedding, 1), N; grow_rows=true)
    running_m = 0.0
    @inbounds for t in 1:length(m)
        running_m += m[t]
        X_m[t, 1] = running_m
    end
    @inbounds for j in 1:N
        running_rel = 0.0
        for t in 1:size(relative_embedding, 1)
            running_rel += relative_embedding[t, j]
            X_rel[t, j] = running_rel
        end
    end
    X_m_vec = view(X_m, 1:length(m), 1)
    
    timing === nothing || (timing.seconds[1]+=(time_ns()-prep_start)*1e-9)
    basis_start=time_ns()
    s_m = fast_s_m(X_m_vec)
    
    s_perp = compute_s_perp(X_rel)
    
    B_m = path_basis_1d(X_m_vec, s_m)
    
    # ----------------------------------------------------
    # 5. Shared Sufficient Statistics across full & folds
    # ----------------------------------------------------
    ts_total = WARMUP:T-2
    n_res = length(ts_total)
    2 <= F_folds <= n_res || throw(ArgumentError("F_folds must leave nonempty training and evaluation rows"))
    n_bands = length(BANDS)
    P_features = 2 * n_bands * N
    
    X_rel_stacked = build_X_rel_stacked(X_rel, s_perp, ts_total;workspace)
    Y_target_rel = relative_embedding[ts_total .+ 1, :] # Field targets, not imputed asset returns
    timing === nothing || (timing.seconds[2]+=(time_ns()-basis_start)*1e-9)
    stats=timed(timing,:gram) do
        fold_sufficient_statistics(X_rel_stacked,Y_target_rel,F_folds;workspace)
    end
    initials=alpha_initial === nothing ? fill((1.0,1.0),F_folds+1) : alpha_initial
    
    # Fit full model
    resp = fit_response_operator(B_m, fill(Float64[], N), m, relative_embedding;
                                 ridge_alpha, ts = ts_total,
                                 S_xx_rel = stats.full_xx, S_xy_rel = stats.full_xy, S_yy_rel = stats.full_yy,
                                 X_design=X_rel_stacked, alpha_initial=initials[1], timing)
                                 
    # ----------------------------------------------------
    # 6. True Out-of-Fold Residuals with Independent EB alphas
    # ----------------------------------------------------
    res_history = fill(NaN, n_res, N)
    fitted_alphas=[(resp.alpha_macro,resp.alpha_rel)]
    
    for fold in 1:F_folds
        fold_eval = stats.ranges[fold]
        S_xx_train = stats.xx === nothing ? nothing : stats.full_xx-stats.xx[fold]
        S_xy_train = stats.full_xy-stats.xy[fold]
        S_yy_train = stats.full_yy-stats.yy[fold]
        train_indices = [1:first(fold_eval)-1; last(fold_eval)+1:n_res]
        train_ts = ts_total[train_indices]
        # Only dual fits need the held-out-row design; give BLAS contiguous storage.
        train_design=length(train_indices)<P_features ? X_rel_stacked[train_indices,:] : nothing
        
        # Fit response operator on train fold: independent EB alpha_m and alpha_rel!
        # Fixed at full model's optimal alpha (Phase 4), solved via fast Cholesky.
        resp_oof = fit_response_operator(B_m, fill(Float64[], N), m, relative_embedding;
                                         ridge_alpha = resp.alpha_rel, ts = train_ts,
                                         S_xx_rel = S_xx_train, S_xy_rel = S_xy_train, S_yy_rel = S_yy_train,
                                         X_design=train_design,
                                         alpha_initial=(resp.alpha_macro, resp.alpha_rel),timing,
                                         need_uncertainty = false)
        push!(fitted_alphas,(resp_oof.alpha_macro,resp_oof.alpha_rel))
                                         
        # Predict evaluation fold rows using fast matrix-vector operations
        eval_start=time_ns()
        X_eval = view(X_rel_stacked, fold_eval, :) # n_eval × P_features
        mu_rel_eval = matrix_buffer!(workspace, :mu_rel_eval, length(fold_eval), N; grow_rows=true)
        BLAS.gemm!('N', 'T', 1.0, X_eval, resp_oof.G_c_mean, 0.0, mu_rel_eval)
        eval_ts = ts_total[fold_eval]
        mu_m_eval = matrix_buffer!(workspace, :mu_m_eval, length(fold_eval), 1; grow_rows=true)
        mul!(mu_m_eval, view(B_m, eval_ts, :), resp_oof.G_macro)
        inv_N = 1.0 / N
        
        @inbounds for local_idx in 1:length(fold_eval)
            global_idx = fold_eval[local_idx]
            t = eval_ts[local_idx]
            inv_sqrt = history_cache === nothing ? 
                ( (cnt = count(isfinite, view(r, t+1, :))) > 0 ? 1.0 / sqrt(cnt) : 0.0 ) :
                history_cache.inv_sqrt_alive[t + 1]
            macro_shift = mu_m_eval[local_idx, 1] * inv_sqrt
            sum_row = 0.0
            @simd for j in 1:N
                sum_row += mu_rel_eval[local_idx, j]
            end
            shift = sum_row * inv_N
            
            # Match the same next-return index used by Y_target_rel and the macro fit.
            @simd for j in 1:N
                rt = r[t + 1, j]
                if isfinite(rt)
                    prediction=(macro_shift + mu_rel_eval[local_idx, j] - shift) * s1[j]
                    res_history[global_idx, j] = rt-prediction
                end
            end
        end
    end
    
    own_res = [findall(isfinite, view(res_history, :, j)) for j in 1:N]
    
    # ----------------------------------------------------
    # 7. Decision-time projection
    # ----------------------------------------------------
    moments = timed(timing,:condition) do
        B_rel_now=compute_B_rel_at_t(X_rel,s_perp,T-1)
        predictive_moments(resp,view(B_m,size(B_m,1),:),B_rel_now)
    end
    
    alive_now = findall(j -> isfinite(adj_act[end, j]), 1:N)
    N_alive = length(alive_now)
    e0_now = zeros(Float64, N)
    if N_alive > 0
        e0_now[alive_now] .= 1.0 / sqrt(N_alive)
    end
    
    mu_rel_proj = moments.mu_rel .- mean(moments.mu_rel)
    mu_norm = moments.mu_m .* e0_now .+ mu_rel_proj
    mu_asset_act = mu_norm .* s1
    
    # Embed mu_asset into full universe
    mu_asset_full = fill(0.0, N_universe)
    mu_asset_full[active_idx] .= mu_asset_act
    
    # ----------------------------------------------------
    # 8. Causal Fractional Likelihood on Macro Residual via FFT
    # ----------------------------------------------------
    e_res_m = Vector{Float64}(undef,n_res)
    @inbounds for idx in 1:n_res
        total=0.0; count=0
        @simd for j in 1:N
            value=res_history[idx,j]
            if isfinite(value)
                total+=value
                count+=1
            end
        end
        e_res_m[idx]=count>0 ? total/sqrt(count) : 0.0
    end
    frac_macro = timed(timing,:fracFFT) do
        causal_fractional_posterior(e_res_m)
    end
    v_bootstrap = max(var(e_res_m), 1e-8)
    
    alpha_initial === nothing || copyto!(alpha_initial,fitted_alphas)
    V1Model(active_idx, N_universe, e0_now, s1, s_m, s_perp,field.observed,resp,
            mu_asset_full, res_history, own_res, moments,
            frac_macro.p_d, frac_macro.v_forecasts, v_bootstrap)
end

"""Nested randomized Halton quadrature. All prefix sizes share one random shift."""
struct ScenarioQuadrature
    primes::Vector{Int}
    shifts::Vector{Float64}
end
function ScenarioQuadrature(dim::Int,rng)
    primes=Int[]; candidate=2
    while length(primes)<dim
        if all(p -> candidate%p != 0, Iterators.takewhile(p -> p*p<=candidate,primes))
            push!(primes,candidate)
        end
        candidate+=1
    end
    ScenarioQuadrature(primes,rand(rng,dim))
end
function quadrature_uniform(rule::ScenarioQuadrature,s::Int,coordinate::Int)
    base=rule.primes[coordinate]
    x=0.0; factor=1.0/base; index=s
    while index>0
        index,digit=divrem(index,base)
        x+=digit*factor
        factor/=base
    end
    clamp(mod(x+rule.shifts[coordinate],1.0),eps(Float64),prevfloat(1.0))
end

"""IID posterior draws, or quadrature of the same law. No post-hoc mean matching."""
function generate_scenarios_v1(model::V1Model; S=500,rng=Random.MersenneTwister(1),quadrature=nothing)
    S>0 || throw(ArgumentError("scenario count must be positive"))
    N=length(model.active_indices)
    T=size(model.res_history,1)
    z=Matrix{Float64}(undef,N,S)
    macro_draws=Vector{Float64}(undef,S)
    uniforms=quadrature === nothing ? rand(rng,S,N+2) : nothing
    normal=Normal()
    if quadrature === nothing
        randn!(rng,z); randn!(rng,macro_draws)
    else
        for s in 1:S
            macro_draws[s]=quantile(normal,quadrature_uniform(quadrature,s,1))
            for j in 1:N
                z[j,s]=quantile(normal,quadrature_uniform(quadrature,s,j+1))
            end
        end
    end
    relative=model.pred_moments.L_rel*z
    relative .+= model.pred_moments.mu_rel
    cumulative=cumsum(model.d_posterior)
    gross=fill(NaN,S,model.N_universe)
    for s in 1:S
        shift=mean(view(relative,:,s))
        mu_m=model.pred_moments.mu_m+sqrt(model.pred_moments.var_m)*macro_draws[s]
        ud=quadrature === nothing ? uniforms[s,1] : quadrature_uniform(quadrature,s,N+2)
        ur=quadrature === nothing ? uniforms[s,2] : quadrature_uniform(quadrature,s,N+3)
        d=min(searchsortedfirst(cumulative,ud),length(cumulative))
        scale=sqrt(model.v_forecasts[d]/model.v_bootstrap)
        row=min(floor(Int,ur*T)+1,T)
        for j in 1:N
            residual=model.res_history[row,j]
            if !isfinite(residual)
                own=model.own_res_rows[j]
                isempty(own) && error("active asset has no observed OOF residual")
                u=quadrature === nothing ? uniforms[s,j+2] : quadrature_uniform(quadrature,s,N+3+j)
                residual=model.res_history[own[min(floor(Int,u*length(own))+1,length(own))],j]
            end
            loggross=(mu_m*model.e0[j]+relative[j,s]-shift)*model.s1[j]+residual*scale
            gross[s,model.active_indices[j]]=exp(loggross)
        end
    end
    gross
end

function adaptive_scenario_weights(model,tradable,held=nothing; rng=Random.MersenneTwister(1),
                                   min_scenarios=64,max_scenarios=512,tol=1e-5,weight_tol=1e-3)
    2<=min_scenarios<max_scenarios || throw(ArgumentError("quadrature must allow refinement"))
    rule=ScenarioQuadrature(2length(model.active_indices)+3,rng)
    S=min_scenarios
    X=generate_scenarios_v1(model; S,rng,quadrature=rule)
    previous=scenario_weights(X,model.active_indices,tradable,held)
    while S<max_scenarios
        S=min(2S,max_scenarios)
        X=generate_scenarios_v1(model; S,rng,quadrature=rule)
        weights=scenario_weights(X,model.active_indices,tradable,held)
        indices=[j for j in model.active_indices if tradable[j]]
        current=held === nothing ? zeros(model.N_universe) : held
        locked=copy(current); locked[indices].=0.0
        base=locked_wealth(X,locked)
        if isempty(indices)
            return (; weights,scenarios=X,S,converged=true,objective_gap=0.0)
        end
        budget=1-sum(locked)
        certificate=kelly_certificate(view(X,:,indices),previous[indices]; budget,base)
        if norm(weights-previous,1)<=weight_tol && certificate.objective_gap<=tol
            return (; weights,scenarios=X,S,converged=true,objective_gap=certificate.objective_gap)
        end
        previous=weights
    end
    error("posterior quadrature did not converge by $max_scenarios scenarios")
end
