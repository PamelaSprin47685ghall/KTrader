"""Causal path coordinates and trace-neutral Matrix-Normal regression."""
function path_basis_1d(X::AbstractVector{Float64}, s::AbstractVector{Float64})
    T = length(X)
    B = zeros(T, 2length(BANDS))
    sums = cumsum(vcat(0.0, X))
    for (b, tau) in enumerate(BANDS)
        scale = max(s[length(s) == length(BANDS) ? b : BANDCOL[b]], 1e-6)
        for t in (2tau+1):T
            c0 = (sums[t+1] - sums[t+1-tau]) / tau
            c1 = (sums[t+1-tau] - sums[t+1-2tau]) / tau
            B[t,2b-1] = -(X[t] - c0) / scale
            B[t,2b] = (c0 - c1) / scale
        end
    end
    B
end

function compute_s_perp(X::AbstractMatrix{Float64})
    T, N = size(X)
    out = zeros(length(BANDS))
    for (b, tau) in enumerate(BANDS)
        count = 0; acc = 0.0
        for j in 1:N, t in (tau+1):T
            a = X[t,j]; z = X[t-tau,j]
            if isfinite(a) && isfinite(z)
                acc += (a-z)^2
                count += 1
            end
        end
        out[b] = count == 0 ? 1e-3 : sqrt(max(acc / count * N / max(N-1,1), 1e-6))
    end
    out
end
fast_s_m(X::AbstractVector{Float64}) = vec(ruler(reshape(X, :, 1), [1]))

function build_X_rel_stacked(X::AbstractMatrix{Float64}, scales::AbstractVector{Float64}, ts::AbstractVector{Int};
                             workspace=nothing)
    T, N = size(X)
    sums = matrix_buffer!(workspace,:path_sums,T+1,N;grow_rows=true)
    counts = matrix_buffer!(workspace,:path_counts,T+1,N;grow_rows=true)
    sums[1,:].=0.0; counts[1,:].=0.0
    @inbounds for j in 1:N
        running_sum = 0.0
        running_cnt = 0
        @simd for t in 1:T
            val = X[t, j]
            if isfinite(val)
                running_sum += val
                running_cnt += 1
            end
            sums[t+1, j] = running_sum
            counts[t+1, j] = running_cnt
        end
    end
    out = matrix_buffer!(workspace,:design,length(ts),2length(BANDS)*N;grow_rows=true)
    fill!(out,0.0)
    inv_scales = 1.0 ./ max.(scales, 1e-6)
    @inbounds for (b,tau) in enumerate(BANDS)
        offset = (b-1)*2N
        inv_scale = inv_scales[b]
        inv_tau = 1.0 / tau
        two_tau = 2tau
        for j in 1:N
            col_q = offset + j
            col_p = offset + N + j
            @simd for row in 1:length(ts)
                t = ts[row]
                if t >= two_tau + 1 && isfinite(X[t,j]) && (counts[t+1,j] - counts[t+1-two_tau,j] == two_tau)
                    c0 = (sums[t+1,j] - sums[t+1-tau,j]) * inv_tau
                    c1 = (sums[t+1-tau,j] - sums[t+1-two_tau,j]) * inv_tau
                    out[row, col_q] = -(X[t,j] - c0) * inv_scale
                    out[row, col_p] = (c0 - c1) * inv_scale
                end
            end
        end
    end
    out
end
function compute_B_rel_at_t(X::AbstractMatrix{Float64}, scales::AbstractVector{Float64}, t::Int)
    N = size(X,2)
    out = [zeros(2length(BANDS)) for _ in 1:N]
    for (b,tau) in enumerate(BANDS), j in 1:N
        t >= 2tau+1 || continue
        window = view(X, t-2tau+1:t, j)
        all(isfinite, window) || continue
        c0 = sum(view(X,t-tau+1:t,j)) / tau
        c1 = sum(view(X,t-2tau+1:t-tau,j)) / tau
        scale = max(scales[b],1e-6)
        out[j][2b-1] = -(X[t,j]-c0)/scale
        out[j][2b] = (c0-c1)/scale
    end
    out
end

struct ResponseOperator
    G_macro::Vector{Float64}
    post_cov_m::Matrix{Float64}
    G_c_mean::Matrix{Float64}
    covariance::RidgeCovariance
    Sigma_rel::Matrix{Float64}
    inv_M_constraint::Matrix{Float64}
    constraint_cols::Vector{UnitRange{Int}}
    alpha_macro::Float64
    alpha_rel::Float64
    trace_real::Float64
    trace_imag::Float64
end

function get_constraint_columns(N::Int, n_bands::Int)
    [(((b-1)*2+channel-1)*N+1):(((b-1)*2+channel)*N) for b in 1:n_bands for channel in 1:2]
end
function constraint_moments(G, V::RidgeCovariance, Sigma, cols)
    nc = length(cols); rank = size(V.basis,2)
    N = length(cols[1])
    # Vectorized computation: T_r = (Sigma * V_r) .* V.weights'
    V_mat = zeros(nc, N * rank)
    T_mat = zeros(nc, N * rank)
    for r in 1:nc
        Vr = view(V.basis, cols[r], :)
        V_mat[r, :] .= vec(Vr)
        Tr = (Sigma * Vr) .* V.weights'
        T_mat[r, :] .= vec(Tr)
    end
    M_raw = V_mat * T_mat'
    M = 0.5 * (M_raw + M_raw')
    if V.baseline != 0
        for r in 1:nc
            M[r, r] += V.baseline * tr(Sigma)
        end
    end
    h = [tr(view(G, :, cols[r])) for r in 1:nc]
    M, h
end
function condition_trace_neutrality(G::AbstractMatrix{Float64}, V::RidgeCovariance,
                                    Sigma::AbstractMatrix{Float64}, N::Int, n_bands::Int)
    cols = get_constraint_columns(N,n_bands)
    M,h = constraint_moments(G,V,Sigma,cols)
    chol = cholesky(Symmetric(M))
    inv_M = Matrix(inv(chol))
    lambda = chol \ h
    K = zeros(size(G))
    for r in eachindex(cols)
        K[:, cols[r]] .+= lambda[r] .* Sigma
    end
    (; G_c=G-covariance_right_product(K,V), inv_M, cols)
end
function condition_trace_neutrality(G::AbstractMatrix{Float64}, V::AbstractMatrix{Float64},
                                    Sigma::AbstractMatrix{Float64}, N::Int, n_bands::Int)
    ev = eigen(Symmetric(V))
    condition_trace_neutrality(G,RidgeCovariance(ev.vectors,ev.values,0.0),Sigma,N,n_bands)
end

function ridge_spectrum(X, Sxx, Sxy; dual=nothing)
    use_dual = X !== nothing && size(X,1) < size(X,2)
    dual !== nothing && (use_dual = dual)
    if use_dual
        X === nothing && throw(ArgumentError("dual fit requires design matrix"))
        ev = eigen!(Symmetric(X*X'))
        # Exact zero eigenvalues carry no data direction; prior null-space variance is retained.
        keep = findall(>(0.0),ev.values)
        values = ev.values[keep]
        basis = X' * view(ev.vectors,:,keep)
        basis ./= sqrt.(values)'
        B = basis' * Sxy
        return (; values,basis,B,dual=true)
    end
    ev = eigen!(Symmetric(copy(Sxx)))
    values = max.(ev.values,0.0)
    (; values,basis=ev.vectors,B=ev.vectors'*Sxy,dual=false)
end
function ridge_covariance(spectrum,alpha)
    baseline = spectrum.dual ? 1/alpha : 0.0
    RidgeCovariance(spectrum.basis,1.0 ./ (spectrum.values .+ alpha) .- baseline,baseline)
end
function positive_covariance(S)
    ev = eigen(Symmetric(S))
    (ev.vectors .* max.(ev.values,1e-8)') * ev.vectors'
end
function supported_covariance(S,gauge)
    gauge === nothing && return positive_covariance(S)
    gauge*positive_covariance(gauge'*S*gauge)*gauge'
end
function covariance_geometry(Sigma,gauge)
    if gauge === nothing
        chol=cholesky(Symmetric(Sigma))
        return (; precision=Matrix(inv(chol)),logdet=logdet(chol),rank=size(Sigma,1))
    end
    chol=cholesky(Symmetric(gauge'*Sigma*gauge))
    (; precision=gauge*(chol\gauge'),logdet=logdet(chol),rank=size(gauge,2))
end

"""Projection in the covariance natural metric, with the same spectral floor.
Euclidean clipping after a natural step can turn it into a descent direction.
"""
function project_covariance_step(Sigma,direction,gauge)
    S=gauge === nothing ? Sigma : gauge'*Sigma*gauge
    D=gauge === nothing ? direction : gauge'*direction*gauge
    root=cholesky(Symmetric(S)).L
    margin=S+D-1e-8I
    whitened=root\margin/root'
    ev=eigen(Symmetric(whitened))
    positive=(ev.vectors .* max.(ev.values,0.0)')*ev.vectors'
    projected=Matrix(Symmetric(root*positive*root'+1e-8I))
    gauge === nothing ? projected : gauge*projected*gauge'
end

# Bounded maximization of the same scalar evidence. Warm-start is an additional
# candidate, never a different stopping rule or a reduced iteration budget.
function maximize_logalpha(f, derivative; initial=1.0,tol=1e-7)
    nodes=collect(range(log(1e-4),log(1e6);length=33))
    slopes=derivative.(nodes)
    candidates=[first(nodes),last(nodes)]
    warm=clamp(log(initial),first(nodes),last(nodes))
    for i in 1:length(nodes)-1
        if slopes[i]>0 && slopes[i+1]<0
            lo=nodes[i]; hi=nodes[i+1]
            if lo<warm<hi
                left=max(lo,warm-1e-3); right=min(hi,warm+1e-3)
                if derivative(left)>0 && derivative(right)<0
                    lo=left;hi=right
                end
            end
            while hi-lo>tol
                mid=(lo+hi)/2
                derivative(mid)>0 ? (lo=mid) : (hi=mid)
            end
            push!(candidates,(lo+hi)/2)
        end
    end
    scores=f.(candidates)
    exp(candidates[argmax(scores)])
end

function optimize_matrix_normal_eb(lambdas::AbstractVector{Float64}, B::AbstractMatrix{Float64},
                                  YtY::AbstractMatrix{Float64}, n::Int; initial = 1.0, iters = 20, tol = 1e-4)
    K = size(B, 2)
    alpha = clamp(initial, 1e-4, 1e6); gamma = 1.0
    converged = false
    for _ in 1:iters
        inv_d = 1.0 ./ (lambdas .+ alpha)
        B_inv = B .* inv_d
        GGt = B_inv' * B_inv
        B_scaled = B .* sqrt.(inv_d)
        quad = B_scaled' * B_scaled
        Sigma = (YtY - quad) ./ n
        chol_S = cholesky(Symmetric(Sigma + 1e-8 * I); check=false)
        tr_term = issuccess(chol_S) ? tr(chol_S \ GGt) : sum(abs2, B_inv)
        gamma = sum(lambdas ./ (lambdas .+ alpha))
        alpha_new = clamp((gamma * K) / max(tr_term, 1e-12), 1e-4, 1e6)
        if abs(log(alpha_new) - log(alpha)) < tol
            alpha = alpha_new
            converged = true
            break
        end
        alpha = alpha_new
    end
    gamma = sum(lambdas ./ (lambdas .+ alpha))
    # Stationarity certificate: relative fixed point residual |gamma*K - alpha*tr(Sigma^-1 G G')| / (gamma*K)
    inv_d_final = 1.0 ./ (lambdas .+ alpha)
    B_inv_final = B .* inv_d_final
    GGt_final = B_inv_final' * B_inv_final
    B_scaled_final = B .* sqrt.(inv_d_final)
    quad_final = B_scaled_final' * B_scaled_final
    Sigma_final = (YtY - quad_final) ./ n
    chol_final = cholesky(Symmetric(Sigma_final + 1e-8 * I); check=false)
    tr_final = issuccess(chol_final) ? tr(chol_final \ GGt_final) : sum(abs2, B_inv_final)
    rel_res = abs(gamma * K - alpha * tr_final) / max(gamma * K, 1e-12)
    (; alpha, gamma, rel_res, converged)
end

"""Conditional-support evidence, up to constants independent of alpha and Sigma.
L_C = L + log density(Cg=0|Y) - log density(Cg=0).
The prior constraint covariance is tr(Sigma)/alpha times the identity.
"""
function conditioned_evidence(spectrum,YtY,n,alpha,Sigma; gradient=false,gauge=nothing)
    d=1.0 ./ (spectrum.values .+ alpha)
    G=(spectrum.B .* d)'*spectrum.basis'
    R=YtY-spectrum.B'*(spectrum.B .* d)
    V=ridge_covariance(spectrum,alpha)
    cols=get_constraint_columns(size(Sigma,1),length(BANDS))
    M,h=constraint_moments(G,V,Sigma,cols)
    cholM=cholesky(Symmetric(M)); geometry=covariance_geometry(Sigma,gauge)
    z=cholM \ h
    nc=length(cols); K=size(Sigma,1)
    value=-geometry.rank/2*sum(log1p.(spectrum.values ./ alpha))-n/2*geometry.logdet-0.5dot(geometry.precision,R)
    value += -0.5logdet(cholM)-0.5dot(h,z)+nc/2*log(tr(Sigma)/alpha)
    gradient || return value
    invS=geometry.precision; invM=Matrix(inv(cholM))
    J=zeros(K,K)
    multipliers=eigen(Symmetric(invM-z*z'))
    work=zeros(K,size(V.basis,2))
    for c in 1:nc
        fill!(work,0.0)
        for r in 1:nc
            work .+= multipliers.vectors[r,c].*view(V.basis,cols[r],:)
        end
        J .+= multipliers.values[c].*((work .* V.weights')*work')
    end
    for j in 1:K
        J[j,j]+=V.baseline*sum(multipliers.values)
    end
    grad=(-n.*invS + invS*R*invS - J) ./ 2
    for j in 1:K
        grad[j,j] += nc/(2tr(Sigma))
    end
    # Cancel Sigma*inv(Sigma) analytically before the natural-gradient step.
    # Multiplying the ill-conditioned inverse products back by Sigma loses digits.
    direction=(-n.*Sigma+R-Sigma*J*Sigma+(nc/tr(Sigma)).*(Sigma*Sigma))./n
    if gauge !== nothing
        direction=gauge*(gauge'*direction*gauge)*gauge'
        grad=gauge*(gauge'*grad*gauge)*gauge'
    end
    (; value,grad=Matrix(Symmetric(grad)),direction=Matrix(Symmetric(direction)))
end

function conditioned_evidence_cache(spectrum,YtY,n,Sigma;gauge=nothing)
    N=size(Sigma,1)
    cols=get_constraint_columns(N,length(BANDS))
    nc=length(cols); rank=length(spectrum.values)
    geometry=covariance_geometry(Sigma,gauge)
    solved=geometry.precision*spectrum.B'
    quadratic=[dot(view(spectrum.B,k,:),view(solved,:,k)) for k in 1:rank]
    tensors=zeros(rank,nc,nc)
    hcoef=zeros(rank,nc)
    for r in 1:nc
        Ur=view(spectrum.basis,cols[r],:)
        transformed=Sigma*Ur
        for k in 1:rank
            hcoef[k,r]=dot(view(spectrum.B,k,:),view(Ur,:,k))
        end
        for s in r:nc
            Us=view(spectrum.basis,cols[s],:)
            for k in 1:rank
                tensors[k,r,s]=tensors[k,s,r]=dot(view(Us,:,k),view(transformed,:,k))
            end
        end
    end
    constant=-n/2*geometry.logdet-0.5dot(geometry.precision,YtY)
    function moments(logalpha)
        alpha=exp(logalpha)
        d=1.0 ./ (spectrum.values .+ alpha)
        baseline=spectrum.dual ? 1/alpha : 0.0
        M=zeros(nc,nc)
        for r in 1:nc, s in r:nc
            M[r,s]=M[s,r]=dot(d .- baseline,view(tensors,:,r,s))
        end
        for r in 1:nc
            M[r,r]+=baseline*tr(Sigma)
        end
        (; alpha,d,baseline,M,h=hcoef'*d)
    end
    function evidence(logalpha)
        state=moments(logalpha)
        cholM=cholesky(Symmetric(state.M))
        constant-geometry.rank/2*sum(log1p.(spectrum.values./state.alpha))+0.5dot(state.d,quadratic) -
            0.5logdet(cholM)-0.5dot(state.h,cholM\state.h)+nc/2*log(tr(Sigma)/state.alpha)
    end
    function derivative(logalpha)
        state=moments(logalpha)
        alpha=state.alpha; d=state.d; baseline=state.baseline
        cholM=cholesky(Symmetric(state.M)); z=cholM\state.h
        dd=-alpha.*d.^2
        Md=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            Md[r,s]=Md[s,r]=dot(dd .+ baseline,view(tensors,:,r,s))
        end
        for r in 1:nc
            Md[r,r]-=baseline*tr(Sigma)
        end
        geometry.rank/2*sum(spectrum.values.*d)+0.5dot(dd,quadratic)-
            0.5tr(cholM\Md)+0.5dot(z,Md*z)-dot(z,hcoef'*dd)-nc/2
    end
    (; evidence,derivative)
end

function optimize_conditioned_eb(spectrum,YtY,n; initial=1.0,tol=1e-6,gauge=nothing)
    alpha=initial
    # Warm starts accelerate scalar roots only; covariance initialization is fixed.
    # Different worker schedules must not choose a different covariance basin.
    d=1.0 ./ (spectrum.values .+ 1.0)
    Sigma=supported_covariance((YtY-spectrum.B'*(spectrum.B .* d))./n,gauge)
    for iteration in 1:100
        cache=conditioned_evidence_cache(spectrum,YtY,n,Sigma;gauge)
        alpha=maximize_logalpha(cache.evidence,cache.derivative; initial=alpha,tol)
        state=conditioned_evidence(spectrum,YtY,n,alpha,Sigma; gradient=true,gauge)
        value=state.value
        # Natural covariance step; projected floor is the existing 1e-8 noise bound.
        direction=state.direction
        projected=project_covariance_step(Sigma,direction,gauge)
        if norm(projected-Sigma) <= tol*max(norm(Sigma),1.0)
            return alpha,Sigma
        end
        displacement=projected-Sigma
        # Best-effort precision: when evidence differences fall below floating-point
        # noise, the optimizer has reached its numerical limit. Accept and stop.
        noise=8eps(max(abs(value),1.0))
        step=1.0; improved=false
        for _ in 1:30
            # Interpolation between feasible covariances stays feasible; do not
            # repeatedly re-eigendecompose an almost-identical matrix at the floor.
            candidate=Sigma+step.*displacement
            trial=conditioned_evidence(spectrum,YtY,n,alpha,candidate;gauge)
            if trial >= value-noise
                Sigma=candidate; value=trial; improved=true
                break
            end
            step*=0.5
        end
        improved || return alpha,Sigma
    end
    error("trace-neutral evidence failed to converge")
end

function fit_response_operator(B_m,B_rel,y_m,y_rel; ridge_alpha=nothing,ts=WARMUP:length(y_m)-1,
                               S_xx_rel=nothing,S_xy_rel=nothing,S_yy_rel=nothing,
                               X_design=nothing,alpha_initial=(1.0,1.0),timing=nothing,dual=nothing,need_uncertainty=true)
    N=length(B_rel); n=length(ts)
    n>0 || throw(ArgumentError("response fit needs training rows"))
    Xm=B_m[ts,:]; ym=y_m[ts .+ 1]
    Sxx=Xm'*Xm; Sxy=reshape(Xm'*ym,:,1); Syy=fill(dot(ym,ym),1,1)
    ev=timed(timing,:eigen) do
        eigen!(Symmetric(Sxx))
    end
    values=max.(ev.values,0.0); B=ev.vectors'*Sxy
    alpha_m,gamma_m=timed(timing,:EB) do
        ridge_alpha === nothing ? (res_m = optimize_matrix_normal_eb(values,B,Syy,n; initial=alpha_initial[1]); (res_m.alpha, res_m.gamma)) :
                                  (ridge_alpha,sum(values ./ (values .+ ridge_alpha)))
    end
    dm=1.0 ./ (values .+ alpha_m)
    gm=vec(ev.vectors*(B .* dm))
    residual=ym-Xm*gm
    sig2=max(sum(abs2,residual)/max(n-gamma_m,1.0),1e-8)
    covm=sig2.*((ev.vectors .* dm')*ev.vectors')
    if N==1
        # The relative subspace has dimension zero: its response and uncertainty are exactly zero.
        P=2length(BANDS)
        alpha_rel=ridge_alpha === nothing ? 1.0 : ridge_alpha
        V=RidgeCovariance(zeros(P,0),Float64[],1/alpha_rel)
        return ResponseOperator(gm,covm,zeros(1,P),V,zeros(1,1),zeros(P,P),
                                get_constraint_columns(1,length(BANDS)),alpha_m,alpha_rel,0.0,0.0)
    end
    if S_xy_rel === nothing
        X_design=zeros(n,2length(BANDS)*N)
        for (row,t) in enumerate(ts), b in eachindex(BANDS), j in 1:N
            offset=(b-1)*2N
            X_design[row,offset+j]=B_rel[j][t,2b-1]
            X_design[row,offset+N+j]=B_rel[j][t,2b]
        end
        Y=y_rel[ts .+ 1,:]
        S_xx_rel=X_design'*X_design; S_xy_rel=X_design'*Y; S_yy_rel=Y'*Y
    end
    if !need_uncertainty && ridge_alpha !== nothing
        # Fast path for OOF folds: alpha is fixed from full model, no eigensolver needed!
        # Solve (S_xx_rel + alpha*I) \ S_xy_rel via single Cholesky (3x faster than eigen).
        P_dim = size(S_xx_rel, 1)
        chol_rel = timed(timing,:eigen) do
            cholesky(Symmetric(S_xx_rel + ridge_alpha * I))
        end
        G = (chol_rel \ S_xy_rel)'
        cols = get_constraint_columns(N, length(BANDS))
        G_c = copy(G)
        for r in 1:length(cols)
            mean_val = tr(view(G_c, :, cols[r])) / N
            for j in 1:N
                G_c[j, cols[r][j]] -= mean_val
            end
        end
        V = RidgeCovariance(zeros(P_dim, 0), Float64[], 0.0)
        Sigma = zeros(N, N)
        trA=sum(sum(G_c[j,cols[2b-1][j]] for j in 1:N) for b in eachindex(BANDS))
        trB=sum(sum(G_c[j,cols[2b][j]] for j in 1:N) for b in eachindex(BANDS))
        return ResponseOperator(gm,covm,G_c,V,Sigma,zeros(length(cols),length(cols)),cols,alpha_m,ridge_alpha,trA,trB)
    end
    spectrum=timed(timing,:eigen) do
        ridge_spectrum(X_design,S_xx_rel,S_xy_rel; dual)
    end
    gauge=relative_gauge(N)
    alpha_rel,Sigma=timed(timing,:EB) do
        if ridge_alpha === nothing
            eb_res = optimize_matrix_normal_eb(spectrum.values, spectrum.B, S_yy_rel, n; initial=alpha_initial[2])
            a_rel = eb_res.alpha
            dr = 1.0 ./ (spectrum.values .+ a_rel)
            B_scaled = spectrum.B .* sqrt.(dr)
            sig = supported_covariance((S_yy_rel - B_scaled' * B_scaled) ./ n, gauge)
            a_rel, sig
        else
            dr=1.0 ./ (spectrum.values .+ ridge_alpha)
            ridge_alpha,supported_covariance((S_yy_rel-spectrum.B'*(spectrum.B .* dr))./n,gauge)
        end
    end
    dr=1.0 ./ (spectrum.values .+ alpha_rel)
    G=(spectrum.B .* dr)'*spectrum.basis'
    V = need_uncertainty ? ridge_covariance(spectrum,alpha_rel) : RidgeCovariance(zeros(size(spectrum.basis, 1), 0), Float64[], 0.0)
    cond=timed(timing,:condition) do
        if need_uncertainty
            condition_trace_neutrality(G,V,Sigma,N,length(BANDS))
        else
            # For OOF evaluation, only G_c is needed to predict fold means.
            # Trace projection subtracting mean diagonal per band:
            cols = get_constraint_columns(N, length(BANDS))
            G_c = copy(G)
            for r in 1:length(cols)
                mean_val = tr(view(G_c, :, cols[r])) / N
                for j in 1:N
                    G_c[j, cols[r][j]] -= mean_val
                end
            end
            (; G_c, inv_M = zeros(length(cols), length(cols)), cols)
        end
    end
    trA=sum(sum(cond.G_c[j,cond.cols[2b-1][j]] for j in 1:N) for b in eachindex(BANDS))
    trB=sum(sum(cond.G_c[j,cond.cols[2b][j]] for j in 1:N) for b in eachindex(BANDS))
    ResponseOperator(gm,covm,cond.G_c,V,Sigma,cond.inv_M,cond.cols,alpha_m,alpha_rel,trA,trB)
end

function predictive_moments(resp::ResponseOperator,Bm::AbstractVector{Float64},Br::Vector{<:AbstractVector{Float64}})
    N=size(resp.G_c_mean,1)
    x=zeros(2length(BANDS)*N)
    for b in eachindex(BANDS), j in 1:N
        offset=(b-1)*2N
        x[offset+j]=Br[j][2b-1]; x[offset+N+j]=Br[j][2b]
    end
    mu_m=dot(resp.G_macro,Bm)
    var_m=max(dot(Bm,resp.post_cov_m*Bm),0.0)
    mu_rel=resp.G_c_mean*x
    v=covariance_product(resp.covariance,x)
    H=zeros(N,length(resp.constraint_cols))
    for (r,cr) in enumerate(resp.constraint_cols)
        mul!(view(H,:,r),resp.Sigma_rel,view(v,cr))
    end
    cov=Symmetric(dot(x,v).*resp.Sigma_rel-H*resp.inv_M_constraint*H')
    ev=eigen(cov)
    L_rel=ev.vectors .* sqrt.(max.(ev.values,0.0))'
    (; mu_m,var_m,mu_rel,L_rel,x_features=x)
end
