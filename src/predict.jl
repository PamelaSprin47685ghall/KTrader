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
    res_history::AbstractMatrix{Float64}    # OOF residuals (n_res × N_active); a read-only
                                             # lazy ResidualOracle in production, any
                                             # AbstractMatrix with identical value semantics
                                             # in tests. Indexing/slicing/size semantics
                                             # are unchanged from the dense era.
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
    macro_flow=zeros(T)
    embedded=zeros(T,N)
    inv_scale=1.0 ./ max.(s1,1e-6)
    @inbounds for t in 1:T
        total=0.0; count=0
        @simd for j in 1:N
            val=r[t,j]
            if isfinite(val)
                total+=val*inv_scale[j]
                count+=1
            end
        end
        if count>0
            macro_flow[t]=total/sqrt(count)
            shift=total/count
            @simd for j in 1:N
                val=r[t,j]
                if isfinite(val)
                    embedded[t,j]=val*inv_scale[j]-shift
                end
            end
        end
    end
    observed=isfinite.(r)
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

# Internal-only keyword channels (ruler_override / scale_override /
# statistics_builder): these are the incremental engine's PRIVATE wiring —
# incremental.jl _prepare_current! passes state-owned values (its self-
# produced ruler, regime s_perp, and the regime statistics builder) into
# this same math. They are NOT part of any public surface: the public
# reference entry prepare_reference (prepare.jl) is closed to a lawful
# keyword whitelist and rejects these three names at keyword dispatch,
# even when passed as `nothing` (refusal precedes the generation bump
# below, any cache read, and any builder execution). _prepare_v1 itself
# is not a public entry; its signature and default math here are
# unchanged. fit_v1 and the batch fallbacks pass no internal keywords.
function _prepare_v1(adj::AbstractMatrix{Float64}; ridge_alpha = nothing, F_folds = 3, ruler_stats = nothing,
                history_cache=nothing, alpha_initial=nothing, timing=nothing, workspace=nothing,
                ruler_override=nothing, scale_override=nothing, statistics_builder=nothing)
    # Prepare-generation binding: bump BEFORE any buffer use, so a failed or
    # partial prepare has already invalidated every older PreparedProblem
    # holding this workspace owner (no stale-buffer solve can succeed).
    workspace !== nothing && (workspace.generation[] += 1)
    ws_generation = workspace === nothing ? 0 : workspace.generation[]
    prep_start=time_ns()
    T_raw, N_universe = size(adj)
    # SPEC §56 fail-loud cache guard: BEFORE the first cache read (the
    # active-set decision below), the consumed prefix — logs, returns,
    # first_price, first_return — must be element-exact with this adj.
    # fit_v1, prepare_reference and the backtest batch path all route
    # through here, so no public entry can bypass it; a mismatch is never
    # silently consumed as an override. No-lookahead: only rows 1:T_raw are
    # verified, so a longer cache stays reusable for any of its prefixes.
    history_cache === nothing || verify_history_cache_prefix(adj, history_cache)
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
    s_raw = if ruler_override !== nothing
        ruler_override
    elseif ruler_stats === nothing
        ruler(log_adj, f)
    else
        # SPEC §56 fail-loud ruler-cache guard: ruler_stats is an exact
        # acceleration cache of THIS panel prefix ruler statistics
        # (geometry.jl PrefixRulerStats), never a license to swap the
        # SPEC §10 ruler. Before the consumed read points (row T, active
        # columns, every tau) reach ruler_from_stats, they are verified
        # against a streaming recomputation from log_adj — the same facts
        # the no-cache path consumes. A foreign same-shape cache, a
        # mutated acc/cnt payload, a lying first-observation row, or
        # NaN/Inf pollution is rejected loudly; rows beyond T are never
        # read (a longer lawful cache serves any of its prefixes).
        # initialize_inference enforces the same provenance semantics
        # against its own accumulated statistics; the incremental engine
        # self-produced ruler_override channel stays unguarded by design
        # (state-internal, never caller input).
        verify_ruler_stats_prefix(log_adj, ruler_stats, active_idx, f)
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
    # scale_override.s_m === nothing means "recompute from the public X_m path"
    # (the regime engine delegates macro scaling to this same batch call).
    s_m = scale_override === nothing ? fast_s_m(X_m_vec) :
        (scale_override.s_m === nothing ? fast_s_m(X_m_vec) : scale_override.s_m)

    s_perp = scale_override === nothing ? compute_s_perp(X_rel) : scale_override.s_perp

    B_m = path_basis_1d(X_m_vec, s_m)

    # ----------------------------------------------------
    # 5. Shared Sufficient Statistics across full & folds
    # ----------------------------------------------------
    ts_total = WARMUP:T-2
    n_res = length(ts_total)
    2 <= F_folds <= n_res || throw(ArgumentError("F_folds must leave nonempty training and evaluation rows"))
    n_bands = length(BANDS)
    P_features = 2 * n_bands * N

    # Lazy design: a statistics_builder owns its Gram source (the regime
    # engine contracts metric-free aggregates), so the n_res x P design is
    # NOT built here. _fit_prepared_v1 rebuilds it on demand only when a
    # dual fit (n < P) or a fold's train block actually needs the rows.
    X_rel_stacked = statistics_builder === nothing ?
        build_X_rel_stacked(X_rel, s_perp, ts_total;workspace) : nothing
    Y_target_rel = view(relative_embedding, ts_total .+ 1, :) # Field targets, not imputed asset returns
    timing === nothing || (timing.seconds[2]+=(time_ns()-basis_start)*1e-9)
    result=timed(timing,:gram) do
        statistics_builder === nothing ?
            (; stats=fold_sufficient_statistics(X_rel_stacked,Y_target_rel,F_folds;workspace),macro_stats=nothing) :
            statistics_builder(X_rel_stacked,Y_target_rel,B_m,m,ts_total)
    end
    (; stats,macro_stats)=result
    typed_stats=to_fold_statistics(stats)
    typed_macro=macro_stats === nothing ? nothing : to_macro_statistics(macro_stats)
    # Ownership as implemented (prepare.jl states the authoritative
    # contract): r/m are owned copies, and the decision-day observation
    # mask alive_now defaults from THIS adj_act's last row inside the
    # constructor - frozen at this moment as an owned copy. adj_act stays
    # a zero-copy view that solve never reads, so later caller-side
    # mutation of the price panel (last row included) cannot change any
    # solve result of this object. X_rel, the lazy design and the
    # full/fold Grams are workspace-leased Matrix objects when a
    # workspace is shared (the generation gate serializes their
    # lifetime; a later same-owner prepare retires this lease loudly),
    # and freshly allocated otherwise.
    PreparedProblem(; active_idx=active_idx, N_universe=N_universe, T=T, N=N,
        s1=s1, s_m=s_m, s_perp=s_perp, adj_act=adj_act, r=Matrix(r), m=copy(m),
        relative_embedding=relative_embedding, observed=field.observed,
        X_rel=X_rel, B_m=B_m, ts_total=collect(ts_total), n_res=n_res,
        P_features=P_features, stats=typed_stats, macro_stats=typed_macro,
        X_rel_stacked=X_rel_stacked, F_folds=F_folds,
        ws_owner=workspace, ws_generation=ws_generation)
end

function fit_v1(adj::AbstractMatrix{Float64}; ridge_alpha=nothing,F_folds=3,ruler_stats=nothing,
                history_cache=nothing,alpha_initial=nothing,timing=nothing,workspace=nothing)
    prep=_prepare_v1(adj;ridge_alpha,F_folds,ruler_stats,history_cache,timing,workspace)
    solve(prep;ridge_alpha,F_folds=F_folds,alpha_initial,timing,workspace)
end

# Both preparations feed this one posterior/OOF/fractional implementation.
# The single solve owner lives in prepare.jl (solve(prepared::PreparedProblem));
# the historical private name below is a pure forward, kept only so existing
# dev-side callers keep working. There is no second pipeline.
function _fit_prepared_v1(prep;ridge_alpha=nothing,F_folds=3,alpha_initial=nothing,timing=nothing,workspace=nothing)
    solve(prep;ridge_alpha,F_folds=F_folds,alpha_initial,timing,workspace)
end

"""The single posterior/OOF/fractional solve owner (SPEC M1: actual_fit has
one production owner). Both prepare_reference and prepare_incremental
produce PreparedProblem; this function is the only consumer. The fold loop
(independent per-fold EB, held-out isolation, warm starts), the trace
conditioning and the lazy ResidualOracle are unchanged; the oof_shadow
factory machinery and its typed error were removed from production (probe
diagnostics must not shape solver signatures)."""
function solve(prep::PreparedProblem; ridge_alpha=nothing,F_folds=nothing,
               alpha_initial=nothing,timing=nothing,workspace=nothing)
    # Fail-loud generation gate (NOT a blind-state acceptance: this only
    # checks that no later prepare has bumped the shared workspace owner -
    # a routing/lifetime fact that never decides math). Default reference
    # preparations share no owner and skip the check entirely.
    prep.ws_owner !== nothing && prep.ws_generation != prep.ws_owner.generation[] &&
        error("PreparedProblem is stale: workspace generation moved on (a later prepare reused this workspace); re-prepare before solving")
    F = F_folds === nothing ? prep.F_folds : F_folds
    F == prep.F_folds || throw(ArgumentError("solve F_folds must match the prepared layout"))
    (; active_idx,N_universe,T,N,r,s1,m,relative_embedding,X_rel,
       s_m,s_perp,B_m,ts_total,n_res,P_features,X_rel_stacked,
       stats,macro_stats,observed)=prep
    n_bands=length(BANDS)
    initials=alpha_initial === nothing ? fill((1.0,1.0),F+1) : alpha_initial

    macro_full=macro_stats === nothing ? (;) : (;macro_stats=macro_stats.full)
    # 构造性 relative support witness（_fit_prepared_v1 是 batch/full、OOF
    # fold 与 incremental solve_current! 的共同拟合核心——此处一处构造，
    # 三路同源一致）：embedded_relative_field 行和构造性为零、X_rel 为其
    # 累积、Y 同源相对场、gauge=relative_gauge(N)。support 维数=N-1，仅
    # N==2（一维）时三前提成立；非数值判定、非裸 N 分支。
    relative_witness = N==2 ? RelativeSupportWitness(1,1,N) : nothing
    # Fit full model. Lazy design: rebuild rows only when a dual fit needs
    # them (n_res < P); primal fits consume Grams alone.
    X_design_full = X_rel_stacked
    if X_design_full === nothing && n_res < P_features
        X_design_full = build_X_rel_stacked(X_rel, s_perp, ts_total)
    end
    resp = fit_response_operator(B_m, fill(Float64[], N), m, relative_embedding;
                                 ridge_alpha, ts = ts_total,
                                 S_xx_rel = stats.full_xx, S_xy_rel = stats.full_xy, S_yy_rel = stats.full_yy,
                                 X_design=X_design_full, alpha_initial=initials[1], timing,
                                 degenerate_witness=relative_witness,macro_full...)
                                 
    # ----------------------------------------------------
    # 6. True Out-of-Fold Residuals via the lazy ResidualOracle
    # ----------------------------------------------------
    # The fold refits (independent EB, held-out isolation, warm starts) are
    # unchanged. What is gone: the per-fold X_eval*G' GEMM over every
    # evaluation row and the T×N residual matrix. The oracle freezes this
    # solve's inputs (r, s1, s_perp, Bm, X_rel prefix sums — never a T×P
    # design copy) and reproduces the identical per-cell formula on demand.
    # Scenario rows are evaluated only for the rows actually sampled; the
    # fractional series uses the O(P)-per-row (fold, mask) h-contraction.
    # The dense formula survives as the test-only reference
    # dense_oof_residuals (residual_oracle.jl).
    # Probe E bucket semantics after lazy wiring:
    #   :OOF_fit      per-fold response refits (independent EB) — unchanged
    #   :OOF_predict  oracle construction: input freezing + prefix table
    #   :OOF_residual scalar macro series (O(P)/row, no N-dim predictions)
    #                 + own-row lists from the raw observation mask
    # No GEMM hides inside the scalar path.
    fold_models = Vector{ResponseOperator}(undef, F)
    fitted_alphas=[(resp.alpha_macro,resp.alpha_rel)]

    for fold in 1:F
        fold_eval = stats.ranges[fold]
        S_xx_train = stats.xx === nothing ? nothing : stats.full_xx-stats.xx[fold]
        S_xy_train = stats.full_xy-stats.xy[fold]
        S_yy_train = stats.full_yy-stats.yy[fold]
        train_indices = [1:first(fold_eval)-1; last(fold_eval)+1:n_res]
        train_ts = ts_total[train_indices]
        # Only dual fits need the held-out-row design; give BLAS contiguous
        # storage. Lazy path rebuilds exactly the train rows from the
        # cumulative field (same fill_design_matrix! formula).
        train_design = if length(train_indices) < P_features
            if X_rel_stacked === nothing
                build_X_rel_stacked(X_rel, s_perp, train_ts)
            else
                X_rel_stacked[train_indices,:]
            end
        else
            nothing
        end

        macro_train=macro_stats === nothing ? (;) : (;macro_stats=(;
            xx=macro_stats.full.xx-macro_stats.folds[fold].xx,
            xy=macro_stats.full.xy-macro_stats.folds[fold].xy,
            yy=macro_stats.full.yy-macro_stats.folds[fold].yy,n=length(train_indices)))
        # Fit response operator on train fold: EB alpha_m and alpha_rel are solved
        # from THIS fold's train Grams only (full data minus fold f). The held-out
        # rows must not reach the fold model through the full-data EB alphas, so
        # ridge_alpha stays exactly what the caller fixed globally (nothing = EB),
        # never the full fit's data-dependent alpha. The EB warm start is the
        # previous decision day's alpha for the same fold (initials[fold+1]):
        # strictly past information, causally independent of today's held-out rows.
        resp_oof = timed(timing, :OOF_fit) do
            fit_response_operator(B_m, fill(Float64[], N), m, relative_embedding;
                                          ridge_alpha, ts = train_ts,
                                          S_xx_rel = S_xx_train, S_xy_rel = S_xy_train, S_yy_rel = S_yy_train,
                                          X_design=train_design,
                                          alpha_initial=initials[fold+1],
                                          degenerate_witness=relative_witness,
                                          need_uncertainty = false, workspace = workspace,macro_train...)
        end
        fold_models[fold] = resp_oof
        push!(fitted_alphas,(resp_oof.alpha_macro,resp_oof.alpha_rel))
    end

    # Frozen-input lazy oracle; construction copies/reduces everything (X_rel
    # is a workspace buffer — the prefix table inside the oracle owns its own
    # storage, so later workspace reuse cannot leak into this model).
    res_history = timed(timing, :OOF_predict) do
        ResidualOracle(fold_models, stats.ranges, ts_total, r, s1, s_perp, B_m, X_rel)
    end

    own_res = timed(timing, :OOF_residual) do
        own_residual_rows(res_history)
    end
    
    # ----------------------------------------------------
    # 7. Decision-time projection
    # ----------------------------------------------------
    moments = timed(timing,:condition) do
        B_rel_now=compute_B_rel_at_t(X_rel,s_perp,T-1)
        predictive_moments(resp,view(B_m,size(B_m,1),:),B_rel_now)
    end
    
    # Decision-day observation support: the mask frozen at prepare time
    # (owned copy). The caller's price panel is never re-read here, so
    # post-prepare mutation of the last price row cannot silently change
    # e0/mu/scenarios of THIS prepared object; a fresh prepare reflects
    # the new information.
    alive_now = findall(prep.alive_now)
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
    # O(P)-per-row (fold, mask) h-contraction over the full history; no T×N
    # OOF prediction matrix is ever materialized for this series.
    e_res_m = timed(timing, :OOF_residual) do
        macro_residual_series(res_history)
    end
    frac_macro = timed(timing,:fracFFT) do
        causal_fractional_posterior(e_res_m)
    end
    v_bootstrap = max(var(e_res_m), 1e-8)
    
    alpha_initial === nothing || copyto!(alpha_initial,fitted_alphas)
    V1Model(active_idx, N_universe, e0_now, s1, s_m, s_perp, observed, resp,
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

"""IID posterior draws, or quadrature of the same law. No post-hoc mean matching.

Lazy residual access: the sampling law is identical to the dense era — same
rng consumption order (rand(S,N+2), randn!(z), randn!(macro_draws), nothing
inside the loop), same uniform layout, the same single shared row per
scenario across assets, the same own-row fallback for missing cells, the
same fractional scaling. The only change: the residual rows each scenario
will need are decided from observation masks and uniforms alone (pass 1, no
rng consumption, no value evaluation), the unique requested rows are
batch-evaluated into a LOCAL matrix (fold-grouped GEMM for the oracle;
plain row indexing for a dense test matrix), and consumed in the original
per-scenario order (pass 2). The model is never mutated and the oracle holds
no cache, so same-seed replays reproduce bit-identically."""
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
    # 规范耦合（Manager 契约 2026-10-07）：从 moments covariance 构造
    # 唯一对称 PSD 主平方根用于 relative draw——消除 L 因子的特征向量
    # 符号/简并基旋转自由度（同 seed 同 Σ → 逐点到 roundoff 一致的
    # 场景）。每 generator 恰好构造一次（z 循环外）。概率律/秩/资产
    # 次序/随机变量数量均不变；Σ 取 L·Lᵀ（old transient model 的 L
    # 字段经同一 adapter 归一，producer/consumer 同规范）。
    canonical_root = principal_sqrt_root(model.pred_moments.L_rel)
    relative = canonical_root * z
    relative .+= model.pred_moments.mu_rel
    cumulative=cumsum(model.d_posterior)
    res=model.res_history

    # Pass 1: decide every (row, j) request from masks and uniforms only.
    isobs(row,j)=res isa ResidualOracle ? observed(res,row,j) : isfinite(res[row,j])
    ur_of(s)=quadrature === nothing ? uniforms[s,2] : quadrature_uniform(quadrature,s,N+3)
    u_of(s,j)=quadrature === nothing ? uniforms[s,j+2] : quadrature_uniform(quadrature,s,N+3+j)
    row_of=Vector{Int}(undef,S)
    back_of=Matrix{Int}(undef,S,N)      # 0 = use the shared row; else own-fallback row
    needed=Int[]
    position=Dict{Int,Int}()
    addrow(row)=begin
        if !haskey(position,row)
            push!(needed,row)
            position[row]=length(needed)
        end
    end
    for s in 1:S
        row=min(floor(Int,ur_of(s)*T)+1,T)
        row_of[s]=row
        addrow(row)
        for j in 1:N
            back_of[s,j]=0
            if !isobs(row,j)
                own=model.own_res_rows[j]
                isempty(own) && error("active asset has no observed OOF residual")
                back=own[min(floor(Int,u_of(s,j)*length(own))+1,length(own))]
                back_of[s,j]=back
                addrow(back)
            end
        end
    end

    # Pass 2: batch-evaluate the unique requested rows locally, then consume
    # them in the original per-scenario order. Nothing lands on the model.
    rows_matrix = res isa ResidualOracle ? residual_rows(res,needed) : Matrix{Float64}(res[needed,:])
    gross=fill(NaN,S,model.N_universe)
    for s in 1:S
        shift=mean(view(relative,:,s))
        mu_m=model.pred_moments.mu_m+sqrt(model.pred_moments.var_m)*macro_draws[s]
        ud=quadrature === nothing ? uniforms[s,1] : quadrature_uniform(quadrature,s,N+2)
        d=min(searchsortedfirst(cumulative,ud),length(cumulative))
        scale=sqrt(model.v_forecasts[d]/model.v_bootstrap)
        k0=position[row_of[s]]
        for j in 1:N
            k=back_of[s,j]==0 ? k0 : position[back_of[s,j]]
            residual=rows_matrix[k,j]
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
