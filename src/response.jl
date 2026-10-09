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
    sums[1,:].=0.0
    @inbounds for j in 1:N
        running_sum = 0.0
        @simd for t in 1:T
            running_sum += X[t, j]
            sums[t+1, j] = running_sum
        end
    end
    out = matrix_buffer!(workspace,:design,length(ts),2length(BANDS)*N;grow_rows=true)
    fill!(out,0.0)
    inv_scales = 1.0 ./ max.(scales, 1e-6)
    fill_design_matrix!(out, sums, X, ts, inv_scales, BANDS, N, length(ts))
    out
end

function fill_design_matrix!(out, sums, X, ts, inv_scales, bands, N, n_res)
    @inbounds for (b,tau) in enumerate(BANDS)
        offset = (b-1)*2N
        inv_scale = inv_scales[b]
        inv_tau = 1.0 / tau
        two_tau = 2tau
        for j in 1:N
            col_q = offset + j
            col_p = offset + N + j
            @simd for row in 1:n_res
                t = ts[row]
                if t >= two_tau + 1
                    c0 = (sums[t+1,j] - sums[t+1-tau,j]) * inv_tau
                    c1 = (sums[t+1-tau,j] - sums[t+1-two_tau,j]) * inv_tau
                    out[row, col_q] = -(X[t,j] - c0) * inv_scale
                    out[row, col_p] = (c0 - c1) * inv_scale
                end
            end
        end
    end
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
    # Block-direct contraction: T_r = (Sigma * V_r) .* V.weights' (10x faster, zero allocation)
    T_all = [ (Sigma * view(V.basis, cols[r], :)) .* V.weights' for r in 1:nc ]
    M = zeros(nc, nc)
    for r in 1:nc
        Vr = view(V.basis, cols[r], :)
        for s in r:nc
            val = dot(Vr, T_all[s])
            M[r, s] = val
            M[s, r] = val
        end
    end
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
    # Use LAPACK syevr! (Relatively Robust Representations) for 2x faster eigensolve with exact eigenvalues
    vals, vecs = LAPACK.syevr!('V', 'A', 'U', copy(Sxx), 0.0, 0.0, 0, 0, -1.0)
    values = max.(vals, 0.0)
    (; values, basis = vecs, B = vecs' * Sxy, dual = false)
end
function ridge_covariance(spectrum,alpha)
    baseline = spectrum.dual ? 1/alpha : 0.0
    weights=spectrum.dual ? -spectrum.values ./ (alpha .* (spectrum.values .+ alpha)) :
            1.0 ./ (spectrum.values .+ alpha)
    RidgeCovariance(spectrum.basis,weights,baseline)
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

const EB_ALPHA_MIN = 1e-4
const EB_ALPHA_MAX = 1e6
const EB_COVARIANCE_FLOOR = 1e-8

# 所有 warm start 只缩小已验证的符号 bracket；候选集合及证书不变。
function maximize_logalpha(f, derivative; initial=1.0,tol=1e-9,maxiters=80)
    isfinite(initial) && initial>0 || throw(ArgumentError("alpha warm start must be finite and positive"))
    tol>0 && maxiters>=0 || throw(ArgumentError("invalid alpha solver budget"))
    nodes=collect(range(log(EB_ALPHA_MIN),log(EB_ALPHA_MAX);length=33))
    slopes=derivative.(nodes)
    all(isfinite,slopes) || error("nonfinite EB alpha derivative")
    candidates=[first(nodes),last(nodes)]
    warm=clamp(log(initial),first(nodes),last(nodes))
    for i in eachindex(nodes)
        slopes[i]==0 && push!(candidates,nodes[i])
    end
    for i in 1:length(nodes)-1
        if slopes[i]>0 && slopes[i+1]<0
            lo=nodes[i]; hi=nodes[i+1]
            if lo<warm<hi
                left=max(lo,warm-1e-3); right=min(hi,warm+1e-3)
                if derivative(left)>0 && derivative(right)<0
                    lo=left; hi=right
                end
            end
            for _ in 1:maxiters
                hi-lo<=tol && break
                mid=(lo+hi)/2
                slope=derivative(mid)
                isfinite(slope) || error("nonfinite EB alpha derivative")
                slope>0 ? (lo=mid) : (hi=mid)
            end
            hi-lo<=tol || error("EB alpha bracket did not converge")
            # Bracket endpoints keep opposite slope signs. The midpoint still
            # carries a first-order stationarity residual (curvature*width/2),
            # which a flat-top maximum leaves far above the certificate scale;
            # the secant zero of the slope removes that first-order term.
            slo=derivative(lo); shi=derivative(hi)
            isfinite(slo) && isfinite(shi) || error("nonfinite EB alpha derivative")
            root = slo>0 && shi<0 ? lo+(hi-lo)*slo/(slo-shi) : (lo+hi)/2
            push!(candidates,clamp(root,lo,hi))
        end
    end
    scores=f.(candidates)
    all(isfinite,scores) || error("nonfinite EB evidence")
    chosen=candidates[argmax(scores)]
    # scale-aware 确定性 tie-break：最优与次优的 evidence 差低于
    # 256·eps(Float64)·max(|best|,1)（机器精度×尺度）时，argmax 由实现顺序
    # 与输入微扰决定（浅平坦 evidence 区，warm start 会成为事实上的 basin
    # 选择器）。注意语义：eps(x) 是 x 处的 ULP（2 的幂栅格），比
    # eps(Float64)*x 小最多 2 倍——上一版误用 ULP 语义，|best|≈1e4 处
    # τ≈4.66e-10 与实测分歧 gap 4.65e-10 贴边（严格小于不成立），本版
    # 用 eps(Float64)*scale：|L|=1e4 时 τ≈5.68e-10 稳定盖住。锚定到 log
    # 网格几何中心，不依赖 warm start；warm start 仍只缩 bracket。
    # 不掩盖真实结构差异：即使 |L|=1e6，τ≈5.7e-8，比似然比检验的
    # 可分辨尺度 O(1) 低七个数量级以上。
    if length(candidates)>1
        order=sortperm(scores,rev=true)
        if scores[order[1]]-scores[order[2]]<256*eps(Float64)*max(abs(scores[order[1]]),1.0)
            anchor=(first(nodes)+last(nodes))/2
            chosen=candidates[argmin(abs.(candidates .- anchor))]
        end
    end
    chosen==first(nodes) && return EB_ALPHA_MIN
    chosen==last(nodes) && return EB_ALPHA_MAX
    exp(chosen)
end

"""有界最大化的 KKT 残差：下边界导数非正，上边界导数非负。"""
function bounded_alpha_certificate(alpha,slope,scale;tol=1e-6)
    bounded=isfinite(alpha) && EB_ALPHA_MIN<=alpha<=EB_ALPHA_MAX
    location=alpha==EB_ALPHA_MIN ? :lower : alpha==EB_ALPHA_MAX ? :upper : :interior
    violation=location===:lower ? max(slope,0.0) : location===:upper ? max(-slope,0.0) : abs(slope)
    rel_res=violation/max(scale,1e-12)
    (; valid=bounded && isfinite(rel_res) && rel_res<=tol,location,slope,rel_res)
end

"""原 support/floor 上的 KKT 驻点证书。
自由方向使用旧 interior 判据的 RMS 白化位移度量
（||Λ_F^{-1/2}DVΛ_F^{-1/2}||_F/sqrt(|F|)≤tol；全自由时与旧 norm/√d 完全
等价）。勘误：上一版分母误写为 |F| 并声称与旧判据"同语义"——代数错误
（sqrt(d·d)=d 不成立，旧归一化是 √d 即 RMS），实际放宽了 √|F| 倍，已被
独立数值实验证伪（比值恒 √d、区间点旧拒新收）；本版恢复 RMS 语义。
floor 激活方向改为单边 KKT 符号：最大化受特征值下界约束时，evidence
想继续减小被 floor 钉住的方向是合法边界驻点，不是 interior 失败。旧 interior
判据把被剪掉的界外位移按 1/λ≈1/floor 白化放大约 1e5 倍，既误杀合法边界解
（全历史第一决策抛 "no certified ascent direction"），也误放行非法边界点
（激活方向梯度指向增大时旧判据只看剪后位移，可能为零）。激活子空间内部的
旋转分量对特征值下界约束一阶不变，KKT 只要求激活块白化方向半负定
（λ_max≤tol，即 evidence 不得想增大被 floor 钉住的方向）。"""
function covariance_certificate(Sigma,direction;gauge=nothing,tol=1e-6)
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    D=gauge===nothing ? direction : gauge'*direction*gauge
    scale=max(norm(Sigma),EB_COVARIANCE_FLOOR)
    symmetry=norm(Sigma-Sigma')/scale
    support=gauge===nothing ? 0.0 : norm(Sigma-gauge*S*gauge')/scale
    if !all(isfinite,S) || !all(isfinite,D)
        return (; valid=false,symmetry,support,min_eigenvalue=NaN,projected_residual=Inf,
                free_rms=NaN,cross_max=NaN,active_block_max=NaN)
    end
    d=size(S,1)
    ev=eigen(Symmetric(S))
    λ=ev.values; V=ev.vectors
    min_eigenvalue=λ[1]
    floor_roundoff=128eps(Float64)*max(opnorm(S),EB_COVARIANCE_FLOOR)
    feasible=symmetry<=1e-10 && support<=1e-10 && min_eigenvalue>0 &&
             min_eigenvalue>=EB_COVARIANCE_FLOOR-floor_roundoff
    feasible || return (; valid=false,symmetry,support,min_eigenvalue,projected_residual=Inf,
                         free_rms=NaN,cross_max=NaN,active_block_max=NaN)
    # 诊断字段保留旧 interior 度量；valid 不再由它判定（边界情形它无 KKT 含义）。
    projected=project_covariance_step(S,D,nothing)
    root=cholesky(Symmetric(S)).L
    residual=root\(projected-S)/root'
    projected_residual=norm(residual)/sqrt(d)
    # KKT 判据：特征基白化位移 W_ij=DV_ij/sqrt(λ_iλ_j)，与旧度量同变换。
    # 分项数值（free_rms/cross_max/active_block_max）进入返回值，供耗尽
    # error 直接定位未收敛部分（alpha 交替 vs Sigma 驻点的分项诊断）。
    active=findall(<=(EB_COVARIANCE_FLOOR+floor_roundoff),λ)
    free=setdiff(1:d,active)
    invroot=1.0 ./ sqrt.(max.(λ,EB_COVARIANCE_FLOOR))
    DV=V'*D*V
    W=DV .* (invroot*invroot')
    valid=true
    free_rms=0.0; cross_max=0.0; active_block_max=-Inf
    if !isempty(free)
        # 自由块 RMS：全自由时 norm(W)/sqrt(d) 与旧判据完全等价（勘误修正）。
        free_rms=norm(W[free,free])/sqrt(length(free))
        valid = valid && free_rms<=tol
        if !isempty(active)
            cross_max=maximum(abs.(W[free,active]))
            valid = valid && cross_max<=tol
        end
    end
    if !isempty(active)
        # 单边 KKT：激活块白化方向半负定（数值容差与自由方向同 tol）。
        active_block_max=maximum(eigvals(Symmetric(W[active,active])))
        valid = valid && active_block_max<=tol
    end
    (; valid=isfinite(projected_residual) && valid,symmetry,support,min_eigenvalue,
       projected_residual,active_directions=length(active),free_rms,cross_max,active_block_max)
end

# 普通（macro）EB：同一 evidence 的 profile covariance 有精确 floor 解。
# 不再用未收敛的 20 步 fixed point 或失效 Cholesky 的无关替代统计量。
function optimize_matrix_normal_eb(lambdas::AbstractVector{Float64},B::AbstractMatrix{Float64},
                                  YtY::AbstractMatrix{Float64},n::Int;
                                  initial=1.0,iters=80,tol=1e-4)
    n>0 && tol>0 || throw(ArgumentError("EB requires observations and positive tolerance"))
    iters>0 || error("matrix-normal EB iteration budget exhausted")
    K=size(B,2)
    function profile(logalpha)
        # Local name `a` is deliberately distinct from the enclosing `alpha`:
        # the caller's alpha must keep the exact boundary value returned by
        # maximize_logalpha (EB_ALPHA_MAX is exact; exp(log(EB_ALPHA_MAX)) is
        # not), so this closure must never assign the enclosing name.
        a=exp(logalpha); d=1.0 ./ (lambdas .+ a)
        R=Matrix(Symmetric(YtY-B'*(B .* d)))
        Sigma=positive_covariance(R./n)
        chol=cholesky(Symmetric(Sigma))
        GGt=(B .* d)'*(B .* d)
        gamma=sum(lambdas .* d)
        value=-K/2*sum(log1p.(lambdas./a))-n/2*logdet(chol)-0.5tr(chol\R)
        slope=(K*gamma-a*tr(chol\GGt))/2
        (; alpha=a,R,Sigma,gamma,value,slope)
    end
    alpha=maximize_logalpha(a->profile(a).value,a->profile(a).slope;
                            initial,tol=min(tol/10,1e-12),maxiters=iters)
    state=profile(log(alpha))
    ac=bounded_alpha_certificate(alpha,state.slope,K*state.gamma/2;tol)
    sc=covariance_certificate(state.Sigma,(state.R-n.*state.Sigma)./n;tol)
    certificate=(; valid=ac.valid && sc.valid,alpha=ac,covariance=sc,evidence=state.value)
    certificate.valid || error("matrix-normal EB certificate failed: $certificate")
    (; alpha,gamma=state.gamma,rel_res=ac.rel_res,converged=true,Sigma=state.Sigma,certificate)
end

"""固定 alpha 的回归量。Sigma 候选绝不重建 G、V、R。
所有矩阵统计在原 N-1 support 坐标计算，不增加 noisy macro 方向。

Sigma-iteration invariants (cached here, valid ONLY at this alpha):
  bwcat  = [blocks[1]*diag(w) | ... | blocks[nc]*diag(w)]  (column-weighted)
  bcat   = [blocks[1] | ... | blocks[nc]]                  (unweighted)
  jcore  = symmetric d x d constraint-Jacobian cores, one per (r <= s):
           jcore[r,s] = (bw_r*B_s' + B_r*bw_s')/2,  bw_r = blocks[r]*diag(w).
Rationale: with unlimited compute the target is UNCHANGED - these are exact
algebraic identities of the same evidence/gradient, not approximations; the
cache is numerical engineering, never theory. The Jacobian identity is the
spectral reconstruction  J = sum_{r,s} K[r,s] * bw_r*B_s' + baseline*tr(K)*I
with K = invM - z z' (sum_c lambda_c m_c m_c' = K by eigendecomposition), and
the single-sided weight w enters the cores once. J's skew part is discarded
by every consumer (grad/direction/stationary target all pass through
Matrix(Symmetric(...))), so the symmetric-core contraction is identical to
the original eigen+work loop after that truncation. Rejected alternatives:
no cross-day Sigma warm start (basin change, PosteriorAudit locks the cold
reference at fixed alpha=1); no held-out alpha reuse (leakage). Evidence
that would reopen this: any input where core-contracted value/grad/direction
diverges from the reference loop beyond roundoff - then the cache must be
deleted, not patched. Invalidations are structural: alpha, fold Grams,
metric or active-set changes rebuild this cache from scratch (it is a local
object owned by one fit), so the cores can never outlive their key.
"""
# Only the diagonal of each constraint block is needed here:
# h[c] = sum_i sum_k Bd[k,i] * U[cols[c][i],k], Bd = B .* d.
# Contract those entries directly: O(N * nc * rank), not O(N * P * rank)
# for the entire G = Bd' * U'. Keep the spectral dot inside the trace sum;
# no gauge projection, rank truncation or cross-fit cache is introduced.
function ridge_constraint_traces(Bd,basis,cols)
    h=zeros(Float64,length(cols))
    for (c,indices) in enumerate(cols), (i,p) in enumerate(indices)
        h[c]+=dot(view(Bd,:,i),view(basis,p,:))
    end
    h
end

function _conditioned_block_geometry(spectrum,N,gauge)
    cols=get_constraint_columns(N,length(BANDS))
    blocks=[gauge===nothing ? Matrix(view(spectrum.basis,c,:)) : gauge'*view(spectrum.basis,c,:) for c in cols]
    (; cols,blocks)
end

function _conditioned_alpha_geometry(geometry)
    (; cols,blocks)=geometry
    (; cols,blocks,bcat=reduce(hcat,blocks))
end

# One private workspace per optimizer invocation; never on the model, never
# shared between folds/tasks. Reuse STORAGE, not old numerical answers.
function _conditioned_alpha_storage(geometry)
    (; owner=geometry.bcat,weighted=similar(geometry.bcat),product=similar(geometry.bcat),
       cores=Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing),generation=Ref(0))
end

function _check_alpha_storage(cache)
    if hasproperty(cache,:scratch)
        cache.storage_generation==cache.scratch.generation[] ||
            error("alpha cache storage retired by a later alpha preparation")
    end
    nothing
end

function conditioned_alpha_cache(spectrum,YtY,n,alpha;gauge=nothing)
    geometry=_conditioned_alpha_geometry(_conditioned_block_geometry(spectrum,size(YtY,1),gauge))
    _conditioned_alpha_cache(spectrum,YtY,n,alpha,geometry;gauge)
end

# Only the optimizer passes its own fit-local geometry. Direct callers
# rebuild it above; neither geometry nor mutable alpha state crosses fits.
function _conditioned_alpha_cache(spectrum,YtY,n,alpha,geometry,scratch=nothing;gauge=nothing)
    if scratch!==nothing
        scratch.owner===geometry.bcat || throw(ArgumentError("alpha storage belongs to another fit geometry"))
        # Invalidate old leases before touching storage, even if prepare fails.
        scratch.generation[]+=1
    end
    d=1.0 ./ (spectrum.values .+ alpha)
    Bd=spectrum.B .* d
    R=Matrix(Symmetric(YtY-spectrum.B'*Bd))
    V=ridge_covariance(spectrum,alpha)
    (; cols,blocks,bcat)=geometry
    h=ridge_constraint_traces(Bd,spectrum.basis,cols)
    compact_R=gauge===nothing ? R : gauge'*R*gauge
    rank=gauge===nothing ? size(YtY,1) : size(gauge,2)
    constant=-rank/2*sum(log1p.(spectrum.values./alpha))
    nc=length(cols)
    dd=size(blocks[1],1); rk=size(blocks[1],2)
    w=V.weights
    bwcat=scratch===nothing ? Matrix{Float64}(undef,dd,nc*rk) : scratch.weighted
    for r in 1:nc
        lo=(r-1)*rk+1; hi=r*rk
        @inbounds for l in 1:rk, j in 1:dd
            bwcat[j,lo+l-1]=blocks[r][j,l]*w[l]
        end
    end
    # Jacobian cores are LAZY: assembled once on the first gradient
    # evaluation at this alpha (see build_jacobian_cores). Value-only
    # consumers of this cache (evidence profiles, alpha brackets use the
    # separate fixed-Sigma cache) never pay the ~nc^2/2 d x d GEMM
    # assembly. The Ref lives inside this local cache object - no state
    # escapes the alpha/fold/metric lifetime.
    jcore=Ref{Union{Nothing,Vector{Matrix{Float64}}}}(nothing)
    cache=(; V,R=Matrix(Symmetric(compact_R)),blocks,h,n,alpha,rank,constant,bwcat,bcat,jcore)
    scratch===nothing ? cache : (;cache...,scratch,storage_generation=scratch.generation[])
end

"""Assemble the symmetric d x d constraint-Jacobian cores on first use:
  core[r,s] = (bw_r*B_s' + B_r*bw_s')/2 for r < s (exact symmetrization),
  core[r,r] = (bw_r*B_r' + B_r*bw_r')/2.
The theoretical Jacobian J = sum_{r,s} K[r,s] * bw_r*B_s' is EXACTLY
symmetric (K symmetric; the single-sided column weight cancels in the
(j,k)/(k,j) product), so the original loop only produced skew at roundoff
and every consumer truncated it through Matrix(Symmetric(...)) - Julia's
Symmetric keeps the UPPER triangle, which equals the theoretical value up
to that roundoff. The cores therefore construct the exact symmetric J
directly; agreement with the original upper-triangle path is roundoff-level
and is pinned by the reference tests (including dual negative weights,
baseline and kappa~1e8)."""
function build_jacobian_cores(cache)
    _check_alpha_storage(cache)
    nc=length(cache.blocks)
    d=size(cache.bcat,1)
    cores=[Matrix{Float64}(undef,d,d) for _ in 1:div(nc*(nc+1),2)]
    _fill_jacobian_cores!(cores,cache)
end

function _fill_jacobian_cores!(cores,cache)
    _check_alpha_storage(cache)
    nc=length(cache.blocks)
    rk=div(size(cache.bcat,2),nc)
    for r in 1:nc, s in r:nc
        lo_r=(r-1)*rk+1; hi_r=r*rk; lo_s=(s-1)*rk+1; hi_s=s*rk
        A=cores[(r-1)*nc+s-div(r*(r-1),2)]
        mul!(A,view(cache.bwcat,:,lo_r:hi_r),view(cache.bcat,:,lo_s:hi_s)')
        # Use BOTH triangles: sym(bw_r*B_s'), not an upper-triangle
        # mirror. Same (a+b)/2 arithmetic, without three matrix copies.
        @inbounds for j in axes(A,2), i in 1:j
            v=(A[i,j]+A[j,i])/2
            A[i,j]=A[j,i]=v
        end
        cores[(r-1)*nc+s-div(r*(r-1),2)]=A
    end
    cores
end

# M 是 Sigma 的线性收缩。允许不定 direction；只有证据求值需要 SPD。
# Exact refactor of dot(blocks[r],(S*blocks[s]).*w') = dot(bw_r, S*blocks_s)
# = Frobenius(S, T_rs) with T_rs = bw_r*B_s'; for the SYMMETRIC S (and the
# covariance-ray's symmetric indefinite D) that every caller supplies,
# <S, T> = <S, sym(T)> exactly, and sym(T_rs) is precisely the lazily
# assembled jacobian core. So when cores already exist (a gradient
# evaluation has run at this alpha) the contraction is 105 O(d^2)
# Frobenius dots instead of the d x d x d x nc*rank GEMM; when they do
# not, the original direct GEMM path runs UNCHANGED and this function
# never triggers assembly (value-only callers stay free of the 105-core
# build). The baseline diagonal is added exactly once on both paths.
# Caller contract: S (or D) symmetric; a non-symmetric input would give
# <sym(S), T> here vs <S, T> on the direct path - pinned by the
# symmetric-indefinite-D test in conditioned_contraction_kernel_tests.jl.
function conditioned_constraint_contraction(cache,S)
    _check_alpha_storage(cache)
    nc=length(cache.blocks)
    jcore=cache.jcore[]
    if jcore!==nothing
        M=zeros(nc,nc)
        for r in 1:nc, s in r:nc
            v=dot(S,jcore[(r-1)*nc+s-div(r*(r-1),2)])
            M[r,s]=M[s,r]=v
        end
        for r in 1:nc
            M[r,r]+=cache.V.baseline*tr(S)
        end
        return M
    end
    _conditioned_constraint_direct(cache,S)
end

# The certificate keeps the original fresh-cache M summation order even
# when it reuses already-built alpha-only Jacobian cores. Do not clear or
# mutate jcore to force this route; other evaluations own that cache too.
function _conditioned_constraint_direct(cache,S)
    _check_alpha_storage(cache)
    nc=length(cache.blocks)
    rk=div(size(cache.bcat,2),nc)
    SB=if hasproperty(cache,:scratch)
        mul!(cache.scratch.product,S,cache.bcat)
    else
        S*cache.bcat
    end
    M=zeros(nc,nc)
    for r in 1:nc,s in r:nc
        lo_r=(r-1)*rk+1; hi_r=r*rk; lo_s=(s-1)*rk+1; hi_s=s*rk
        M[r,s]=M[s,r]=dot(view(cache.bwcat,:,lo_r:hi_r),view(SB,:,lo_s:hi_s))
    end
    for r in 1:nc
        M[r,r]+=cache.V.baseline*tr(S)
    end
    M
end

function conditioned_constraint_jacobian(cache,cholM,z)
    _check_alpha_storage(cache)
    nc=length(cache.h)
    invM=Matrix(inv(cholM))
    # Jacobian by symmetric-core contraction (exact identity; see the
    # rationale on conditioned_alpha_cache / build_jacobian_cores):
    #   J = sum_{r<=s} c_rs * core[r,s] + baseline*tr(K)*I,
    #   K = invM - z*z', c_rr = K[r,r], c_rs = 2K[r,s] for r < s.
    # Derived from the original eigen+work loop by the spectral
    # reconstruction sum_c lambda_c m_c m_c' = K and by moving the
    # single-sided column weight into the lazily assembled cores. The
    # theoretical J is exactly symmetric, so the original loop's roundoff
    # skew was discarded by every consumer's Matrix(Symmetric(...))
    # upper-triangle truncation; the cores construct the same symmetric J
    # directly, without the per-c work assembly, the nc d x rank GEMMs and
    # the eigendecomposition. Cores are built once per (alpha-cache) on
    # this first gradient use; value-only paths never assemble them.
    jcore=cache.jcore[]
    if jcore===nothing
        if hasproperty(cache,:scratch)
            storage=cache.scratch.cores[]
            jcore=storage===nothing ? build_jacobian_cores(cache) : _fill_jacobian_cores!(storage,cache)
            cache.scratch.cores[]=jcore
        else
            jcore=build_jacobian_cores(cache)
        end
        cache.jcore[]=jcore
    end
    K=invM-z*z'
    J=zeros(size(cache.R))
    for r in 1:nc
        J .+= K[r,r].*jcore[(r-1)*nc+r-div(r*(r-1),2)]
        for s in (r+1):nc
            J .+= (2*K[r,s]).*jcore[(r-1)*nc+s-div(r*(r-1),2)]
        end
    end
    btrK=cache.V.baseline*tr(K)
    for j in axes(J,1)
        J[j,j]+=btrK
    end
    J
end

function _conditioned_cached_state(cache,S;direction=false,M=nothing)
    _check_alpha_storage(cache)
    M===nothing && (M=conditioned_constraint_contraction(cache,S))
    cholM=cholesky(Symmetric(M)); cholS=cholesky(Symmetric(S))
    z=cholM\cache.h; nc=length(cache.h)
    value=cache.constant-cache.n/2*logdet(cholS)-0.5tr(cholS\cache.R)-
          0.5logdet(cholM)-0.5dot(cache.h,z)+nc/2*log(tr(S)/cache.alpha)
    direction || return value
    J=conditioned_constraint_jacobian(cache,cholM,z)
    # The natural direction uses no inverse or spectral gradient. Outer
    # fixed-point acceptance consumes this direction, J and evidence only.
    trS=tr(S)
    # Preserve both matrix products and left-associated scalar operations;
    # fuse only the elementwise work, avoiding three full temporary arrays.
    SJS=S*J*S
    SS=S*S
    step=(-cache.n.*S .+ cache.R .- SJS .+ (nc/trS).*SS)./cache.n
    (; value,direction=Matrix(Symmetric(step)),M,J)
end

function _conditioned_covariance_gradient(cache,S,J,evS=eigen(Symmetric(S)))
    nc=length(cache.h)
    # grad 的特征基稳定求值。原实现 invS=Matrix(inv(cholS)) 后两次矩阵
    # 乘：在 κ(S)~1e8（floor 钉住方向）下 invS 的求逆绝对误差
    # ~κ·eps·||invS||~O(1)，而 invS·R·invS 的自由块信号仅 O(1e2)——
    # 信号被求逆误差淹没（t=14294 iter=6 实测：V'gradV 自由块范数
    # 3.97e2，恒等式 direction=(2/n)·S·grad·S 的相对失配 1.4e4），
    # dot(grad,·) 符号由噪声决定，evidence 平台上 ascent 判据误触
    # fail-closed。改为特征坐标组装（数学恒等的白化重排）：
    #   V'grad V = (-n·Λ⁻¹ + Λ⁻¹(V'RV)Λ⁻¹ - V'JV)/2 + diag(nc/(2tr))
    # 1/λ 只出现在精确对角/逐元素位置，不产生矩阵级误差传播；direction
    # 公式（不经过 invS）不变，恒等式随之在浮点下恢复。
    Λ=evS.values; V=evS.vectors
    VRV=V'*cache.R*V
    VJV=V'*J*V
    G=(-cache.n ./ Λ) .* Matrix{Float64}(I,size(S,1),size(S,1))
    G .+= VRV ./ (Λ .* Λ')
    G .-= VJV
    G ./= 2
    trS=tr(S)
    for j in axes(G,1)
        G[j,j]+=nc/(2trS)
    end
    grad=V*G*V'
    Matrix(Symmetric(grad))
end

# Preserve the public value/full-gradient interface and its return fields.
# Only the private optimizer defers grad until it is actually consumed.
function conditioned_cached_state(cache,S;gradient=false,M=nothing)
    gradient || return _conditioned_cached_state(cache,S;M)
    state=_conditioned_cached_state(cache,S;direction=true,M)
    grad=_conditioned_covariance_gradient(cache,S,state.J)
    (; value=state.value,grad,direction=state.direction,M=state.M,J=state.J)
end

"""Conditional-support evidence，常数独立于 alpha 与 Sigma。
L_C = L + log density(Cg=0|Y) - log density(Cg=0)。
"""
function conditioned_evidence(spectrum,YtY,n,alpha,Sigma;gradient=false,gauge=nothing)
    cache=conditioned_alpha_cache(spectrum,YtY,n,alpha;gauge)
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    state=conditioned_cached_state(cache,S;gradient)
    gradient || return state
    grad=gauge===nothing ? state.grad : gauge*state.grad*gauge'
    direction=gauge===nothing ? state.direction : gauge*state.direction*gauge'
    (; value=state.value,grad,direction)
end

# Fixed within ONE fit: these projections and trace coefficients depend on
# its spectrum/targets/gauge, never on alpha or Sigma. The optimizer owns
# this local tuple; it is neither stored on a model nor shared across folds.
function _conditioned_scalar_geometry(spectrum,YtY;gauge=nothing)
    N=size(YtY,1)
    block_geometry=_conditioned_block_geometry(spectrum,N,gauge)
    (; cols,blocks)=block_geometry
    nc=length(cols); rank=length(spectrum.values)
    B=gauge===nothing ? spectrum.B : spectrum.B*gauge
    R=gauge===nothing ? YtY : gauge'*YtY*gauge
    hcoef=zeros(rank,nc)
    for r in 1:nc, k in 1:rank
        hcoef[k,r]=dot(view(B,k,:),view(blocks[r],:,k))
    end
    (; B,R,hcoef,block_geometry...)
end

# Public/direct callers still build from their own inputs on every call.
# Only the private optimizer loop reuses its fit-local geometry.
function conditioned_evidence_cache(spectrum,YtY,n,Sigma;gauge=nothing)
    geometry=_conditioned_scalar_geometry(spectrum,YtY;gauge)
    _conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
end

# Fixed-Sigma scalar cache: rebuild ALL Sigma-dependent terms. Within an
# alpha bracket, only spectral weights and nc x nc algebra remain.
function _conditioned_scalar_tensors(blocks,S)
    nc=length(blocks); d,rank=size(first(blocks))
    # Only upper pairs are read. Scratch is local to construction, never
    # captured by evidence/derivative and never shared across Sigma/folds.
    tensors=Matrix{Float64}(undef,rank,div(nc*(nc+1),2))
    transformed=Matrix{Float64}(undef,d,rank)
    for r in 1:nc
        mul!(transformed,S,blocks[r])
        for s in r:nc
            pair=(r-1)*nc+s-div(r*(r-1),2)
            for k in 1:rank
                tensors[k,pair]=dot(view(blocks[s],:,k),view(transformed,:,k))
            end
        end
    end
    tensors
end

function _conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge=nothing)
    (; B,R,blocks,hcoef)=geometry
    nc=length(blocks); rank=length(spectrum.values)
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    chol=cholesky(Symmetric(S)); solved=chol\B'
    quadratic=[dot(view(B,k,:),view(solved,:,k)) for k in 1:rank]
    tensors=_conditioned_scalar_tensors(blocks,S)
    constant=-n/2*logdet(chol)-0.5tr(chol\R)
    function moments(logalpha)
        alpha=exp(logalpha); d=1.0 ./ (spectrum.values .+ alpha)
        baseline=spectrum.dual ? 1/alpha : 0.0
        weights=spectrum.dual ? -spectrum.values .* d ./ alpha : d
        M=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            M[r,s]=M[s,r]=dot(weights,view(tensors,:,(r-1)*nc+s-div(r*(r-1),2)))
        end
        for r in 1:nc
            M[r,r]+=baseline*tr(S)
        end
        (; alpha,d,baseline,M,h=hcoef'*d)
    end
    function evidence(logalpha)
        state=moments(logalpha); cholM=cholesky(Symmetric(state.M))
        constant-size(S,1)/2*sum(log1p.(spectrum.values./state.alpha))+0.5dot(state.d,quadratic)-
            0.5logdet(cholM)-0.5dot(state.h,cholM\state.h)+nc/2*log(tr(S)/state.alpha)
    end
    function derivative(logalpha)
        state=moments(logalpha); alpha=state.alpha; d=state.d
        cholM=cholesky(Symmetric(state.M)); z=cholM\state.h
        dd=-alpha.*d.^2
        wd=spectrum.dual ? spectrum.values .* d .* (d .+ 1/alpha) : dd
        Md=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            Md[r,s]=Md[s,r]=dot(wd,view(tensors,:,(r-1)*nc+s-div(r*(r-1),2)))
        end
        for r in 1:nc
            Md[r,r]-=state.baseline*tr(S)
        end
        size(S,1)/2*sum(spectrum.values.*d)+0.5dot(dd,quadratic)-
            0.5tr(cholM\Md)+0.5dot(z,Md*z)-dot(z,hcoef'*dd)-nc/2
    end
    (; evidence,derivative)
end

"""每条 Sigma 线搜索仅收缩一次 direction；M(S+tD)=M(S)+t M(D)。"""
function conditioned_covariance_ray(cache,S,D;M=conditioned_constraint_contraction(cache,S))
    Md=conditioned_constraint_contraction(cache,D)
    t->conditioned_cached_state(cache,S+t.*D;M=M+t.*Md)
end

function conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma;gauge=nothing,tol=1e-6)
    geometry=_conditioned_scalar_geometry(spectrum,YtY;gauge)
    alpha_geometry=_conditioned_alpha_geometry(geometry)
    _conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,alpha_geometry;gauge,tol)
end

# Only the optimizer supplies its own immutable fit geometry; no new public
# cache-injection keyword. R/h/spectral weights and Jacobian cores depend
# on this fit and alpha, not Sigma; the optimizer may pass its own cache.
# EVERY Sigma-dependent term and check is recomputed. Reproject Sigma as the public certificate does,
# rather than substituting the optimizer's pre-embedding compact S.
function _conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,alpha_geometry,cache=nothing;
                                     gauge=nothing,tol=1e-6)
    cache===nothing || (cache.alpha==alpha && cache.n==n) ||
        throw(ArgumentError("certificate cache belongs to another alpha or sample count"))
    scalar=_conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
    slope=scalar.derivative(log(alpha))
    rank=gauge===nothing ? size(Sigma,1) : size(gauge,2)
    scale=max(rank*sum(spectrum.values./(spectrum.values.+alpha)),2length(BANDS),1.0)
    ac=bounded_alpha_certificate(alpha,slope,scale;tol)
    cache===nothing && (cache=_conditioned_alpha_cache(spectrum,YtY,n,alpha,alpha_geometry;gauge))
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    M=_conditioned_constraint_direct(cache,S)
    state=_conditioned_cached_state(cache,S;direction=true,M)
    direction=gauge===nothing ? state.direction : gauge*state.direction*gauge'
    sc=covariance_certificate(Sigma,direction;gauge,tol)
    (; valid=ac.valid && sc.valid && isfinite(state.value),alpha=ac,covariance=sc,evidence=state.value)
end

"""构造性 relative support witness：由真正 producer（embedded_relative_field
→X_rel 累积、relative_embedding 同源输出、relative_gauge 压缩）建立的三前提
结构声明——(1) 每通道输入的相对 support 一维（embedded 行和构造性为零 →
N=2 时 design 通道两列反对称）；(2) 输出 Σ=Q·S·Q' 同一 gauge、support 一
维；(3) 每通道 trace-neutral 约束在该 support 内强制。三前提成立时
G[:,c1]=[a;-a]、G[:,c2]=[c;-c] 且 trace 钉 a=c，任意合法相对输入的预测恒为
零——可观测系数 support 为空，input common 系数是保留的未识别 prior（由
producer 构造证明，不由数值判定）。拒绝裸 N==2/浮点 rank 推断：N==2 只是
support 维数=1 的构造性等价，语义由 witness 携带；primal/dual 是同一模型
的表示，不是 flat 选择的理论前提。canonical alpha 只选被证明不可识别的
参数（旧 default 参考 1.0）；S 取条件似然的解析噪声协方差解。失效证据：
任一前提被证伪（如非零和输入混入）或原证书拒绝解析候选时，分支失效、
回退通用求解（见 optimize_conditioned_eb 的接线与 covariance_certificate
契约）。"""
struct RelativeSupportWitness
    input_support_dim::Int    # 每通道输入相对 support 维数（构造性，一维消元要求=1）
    output_support_dim::Int   # 输出 Σ 的 gauge 压缩维数（=1）
    assets::Int               # 资产空间大小（须与 YtY/gauge 同一空间）
end

"""N=2 相对退化（可观测系数 support 为空）的解析候选——仅在 producer
witness 下生效，无 witness 一律通用。alpha=旧 default 参考 1.0；
S=max(q'YtYq/n, floor)（条件化 ỹ|Cg=0~N(0,S·I) 的解析噪声协方差解，
ỹ=Y·q）。witness 的维度/资产空间声明与实际输入精确核对（防谎报），
不符即 nothing。候选必须过原 conditioned_eb_certificate（源码公式、含
alpha slope 证书）验收；失败回退通用循环——不伪造 valid、不放宽门禁、
不削 coefficient prior（G/V 由下游照常构造）。"""
function degenerate_relative_solution(YtY,n,gauge,witness)
    witness===nothing && return nothing
    (gauge!==nothing && size(gauge,2)==witness.output_support_dim==1 &&
     witness.input_support_dim==1 && size(YtY,1)==witness.assets==2) || return nothing
    q=vec(gauge)
    (; alpha=1.0, S=max(dot(q,YtY*q)/n,EB_COVARIANCE_FLOOR))
end

"""冻结方程 S+S·K·S=Q 的闭式候选；K=(J−(nc/τ)I)/n，Q=R/n（τ=tr S）。
S⁺=√Q·W·√Q，W=2(I+(I+4B̃)^{1/2})^{-1}，B̃=√Q·K·√Q：恒等式
S⁺+S⁺KS⁺=√Q(W+WB̃W)√Q=Q 精确成立，不依赖 K 与 Q 对易，也不要求 Q
非奇异（零方向自动 S⁺→0，由调用方的 floor clamp 接住）。仅当
I+4B̃⪰0（合法域）时返回解；否则返回 nothing——该步不产生候选，
不冒充认证。返回值未做 floor clamp，与不动点候选同交
positive_covariance 统一投影。"""
function riccati_stationary_candidate(Qroot,J,τ,nc,n)
    d=size(J,1)
    K=(J .- (nc/τ)*Matrix{Float64}(I,d,d))./n
    evB=eigen(Symmetric(Qroot*(K*Qroot)))
    scaled=1.0 .+ 4.0 .* evB.values
    minimum(scaled)<0.0 && return nothing
    W=(evB.vectors .* (2.0./(1.0 .+ sqrt.(scaled)))') * evB.vectors'
    Matrix(Symmetric(Qroot*W*Qroot))
end

"""固定 alpha 的 Sigma 驻点方程求解器（残差单调 + Riccati 闭式候选）。
方程 -nS⁻¹+S⁻¹RS⁻¹−J(S)+(nc/tr S)I=0 等价于 direction=0，即
nS = R − S·J·S + (nc/tr S)·S²。冻结 (J,τ) 的子问题 S+S·K·S=Q 的
不动点迭代 X⁺=Q−X·K·X 的雅可比 −(S*K+K*S) 在病态 M（多激活方向使
J 大且对 S 敏感）下谱半径远超 1：本质发散的迭代，阻尼只能镇定、不能
恢复收敛率——t=14294 单线程 BLAS 下 free_rms=1.4097897e-5 停滞的实测
机制（DevOps 固化 fixture：test/fixtures/t14294_conditioned_fold3.jls）。
本版在原不动点候选之外补充冻结方程的闭式候选
riccati_stationary_candidate（合法域内方程精确解；域外不产生候选，
退回不动点候选），两候选走同一回溯接受框架：同一未 clamp 方程残差
||G(X)−X||_F、同残差单调判据（≤current·(1+1e-12)）、同一 ω 减半
回溯与凸组合可行（Weyl），取残差更小的接受者，均失败则 break（返回
当前点，外层走自然梯度回退）。闭式候选不宣称普遍收敛——K 不定、
Q 奇异时存在性无保证；验收权全在外层 evidence 不降 + free_rms 不升
双条件与 covariance_certificate。nc=0 时 K=0、W=I，退化为 R/n 一步
收敛（与旧版语义一致）。"""
function sigma_stationary_fixed_point(cache,S;iters=40,tol=1e-10)
    nc=length(cache.h)
    function frozen_step(X)
        M=conditioned_constraint_contraction(cache,X)
        cholM=cholesky(Symmetric(M))
        cholesky(Symmetric(X)) # retain the old SPD rejection, even for direct callers
        J=conditioned_constraint_jacobian(cache,cholM,cholM\cache.h)
        # The fixed-point proposal consumes J only. Evidence, the spectral
        # gradient and direction are still evaluated by the outer acceptance
        # and certificate path, not for every inner trial that discards them.
        target=Matrix(Symmetric((cache.R .+ (nc/tr(X)).*(X*X) .- X*J*X)./cache.n))
        (; target,residual=norm(target-X),J)
    end
    Qroot=let ev=eigen(Symmetric(Matrix(cache.R)./cache.n))
        (ev.vectors .* sqrt.(max.(ev.values,0.0))') * ev.vectors'
    end
    current=frozen_step(S)
    for _ in 1:iters
        current.residual<=tol*max(norm(S),1.0) && break
        candidates=[positive_covariance(current.target)]
        rc=riccati_stationary_candidate(Qroot,current.J,tr(S),nc,cache.n)
        rc===nothing || push!(candidates,positive_covariance(rc))
        best=nothing
        for candidate in candidates
            ω=1.0
            for _ in 1:8
                trial=Matrix(Symmetric(S .+ ω.*(candidate .- S)))
                probe=frozen_step(trial)
                if probe.residual<=current.residual*(1+1e-12)
                    if best===nothing || probe.residual<best[2].residual
                        best=(trial,probe)
                    end
                    break
                end
                ω*=0.5
            end
        end
        best===nothing && break
        S=best[1]; current=best[2]
    end
    Matrix(Symmetric(S))
end

# iters 默认 200→1000（数值工程层收敛预算，非数学定义；SPEC §55 同数学目标
# 的工程调整）：实测 N=3 relative-support fixture 在途慢收敛、264 次迭代收敛
# （确定性复现，约 4 倍余量）；fail-loudly 契约完整保留——真停滞类（如
# docstring 记载的 t=14294 发散迭代）在预算耗尽时照样抛错。证据：
# archive/evidence/manager11/eb_convergence_probe.log
function optimize_conditioned_eb(spectrum,YtY,n;initial=1.0,tol=1e-6,gauge=nothing,
                                 iters=1000,max_backtracks=30,alpha_iters=80,return_certificate=false,
                                 fp_iters=40,fp_tol=1e-10,degenerate_witness=nothing)
    n>0 && tol>0 || throw(ArgumentError("conditioned EB requires observations and positive tolerance"))
    iters>0 && max_backtracks>0 && alpha_iters>0 || error("conditioned EB iteration budget exhausted")
    # 构造性 witness 分支：只在 producer 证明可观测系数 support 为空时给出
    # canonical 候选（alpha=1.0 旧 default 参考、解析噪声协方差 S），仍须原
    # conditioned_eb_certificate 验收（含 alpha slope 与 covariance KKT），
    # 失败回退通用循环——不伪造 valid；witness=nothing 时本函数行为不变。
    degenerate=degenerate_relative_solution(YtY,n,gauge,degenerate_witness)
    if degenerate!==nothing
        Sigma0=gauge*degenerate.S*gauge'
        cert0=conditioned_eb_certificate(spectrum,YtY,n,degenerate.alpha,Sigma0;gauge,tol)
        if cert0.valid
            return return_certificate ? (; alpha=degenerate.alpha,Sigma=Sigma0,
                                           certificate=cert0,iterations=0) :
                                       (degenerate.alpha,Sigma0)
        end
        # 证书拒绝：witness 与输入不一致或推导失效，回退通用循环。
    end
    # 确定的参考 alpha=1 covariance 起点；暖 alpha 不选择 covariance basin。
    scalar_geometry=_conditioned_scalar_geometry(spectrum,YtY;gauge)
    # Scalar-only direct/certificate calls do not allocate this packed copy.
    alpha_geometry=_conditioned_alpha_geometry(scalar_geometry)
    alpha_storage=_conditioned_alpha_storage(alpha_geometry)
    reference=_conditioned_alpha_cache(spectrum,YtY,n,1.0,alpha_geometry,alpha_storage;gauge)
    S=positive_covariance(reference.R./n)
    alpha=initial
    for iteration in 1:iters
        Sigma=gauge===nothing ? S : gauge*S*gauge'
        scalar=_conditioned_scalar_cache(spectrum,n,Sigma,scalar_geometry;gauge)
        alpha=maximize_logalpha(scalar.evidence,scalar.derivative;initial=alpha,
                                tol=min(tol/10,1e-9),maxiters=alpha_iters)
        cache=_conditioned_alpha_cache(spectrum,YtY,n,alpha,alpha_geometry,alpha_storage;gauge)
        state=_conditioned_cached_state(cache,S;direction=true)
        # [选项 A + 双条件采纳] 驻点方程不动点（残差单调版）直接跳到当前
        # alpha 下的 Sigma 驻点。采纳需同时满足：evidence 不降（全局，防坏
        # basin）AND 自由块白化残差 free_rms 不升（局部，防 evidence 数值
        # 平台上的无信息漂移——实测 t=14294 的平台上 evidence 检验放行过
        # 3.4 倍的残差恶化，平台上唯一可靠判据是残差本身）。任一不满足
        # 则退回下方自然梯度步。
        sc_now=covariance_certificate(S,state.direction;tol)
        S_fp=sigma_stationary_fixed_point(cache,S;iters=fp_iters,tol=fp_tol)
        fp_state=_conditioned_cached_state(cache,S_fp;direction=true)
        fp_sc=covariance_certificate(S_fp,fp_state.direction;tol)
        # Causal duplicate elimination: conditioned_evidence at (alpha, S_fp)
        # would rebuild the SAME alpha cache and re-evaluate the SAME
        # cached_state value; fp_state.value is that exact number. This is a
        # local pass-down inside one iteration (the caller owns both the
        # cache and the state), not a cache: alpha bits or Sigma content
        # changes rebuild everything by construction.
        fp_value=fp_state.value
        if isfinite(fp_value) && fp_value>=state.value-8eps(max(abs(state.value),1.0)) &&
           isfinite(fp_sc.free_rms) && fp_sc.free_rms<=sc_now.free_rms+1e-12
            S=S_fp
            state=fp_state
            sc=fp_sc
        else
            sc=sc_now
        end
        if sc.valid
            Sigma=gauge===nothing ? S : gauge*S*gauge'
            certificate=_conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,
                                                   scalar_geometry,alpha_geometry,cache;gauge,tol)
            if certificate.valid
                return return_certificate ? (; alpha,Sigma,certificate,iterations=iteration) : (alpha,Sigma)
            end
            # Sigma 已驻点但 alpha 未到当前 Sigma 下的驻点：继续交替。总证书
            # 仍是唯一返回出口（门禁不变）；旧流程把这一坐标上升中间态误报
            # 为 error，不动点使 Sigma 驻点先于 alpha 驻点到达，在此修正流程。
         else
            # [free-tangent 搜索方向] 白化特征基中 direction 的自由块正交
            # 投影；active 块精确置零。依据（t=14294 iter=7 冻结实测）：
            # direction 是 evidence 的 S-预条件上升方向（代数恒等式
            # direction=(n/4)·S·grad·S ⟹ dot(grad,direction)=
            # (n/4)·||S^{1/2}gradS^{1/2}||²_F≥0，自由块贡献同样非负）；
            # 但 project_covariance_step 的特征值截断在存在 O(1) 白化
            # active 块（KKT 单边判据合法的边界行为）时对自由块产生畸变
            # （实测截断后位移的白化自由范数为 direction 自由块的 7 倍），
            # 使位移的真实方向导数翻负（-3.4e-4），ascent 判据在 evidence
            # 平台上误触 fail-closed。本分支因此不再使用截断位移，而是
            # 构造新的合法搜索方向：direction 的自由块（上升性由恒等式
            # 保证），floor 钉住的 active 方向不动（KKT 单边条件已满足，
            # 步进不穿越正定锥——自由特征值相对变化 ~1e-5 量级）。这不是
            # 从原位移丢弃非零项：实测位移的 active/active 与 active/free
            # 块本就为零至 1e-16（floor 投影的语义），被丢弃的只有把方向
            # 导数符号污染的截断畸变。验收出口不变：covariance_certificate
            # 与 conditioned_eb_certificate 的原值门禁、8eps 采纳容差、
            # line search 的 Armijo 条件全部保留。
            evS=eigen(Symmetric(S))
            floor_roundoff=128eps(Float64)*max(opnorm(S),EB_COVARIANCE_FLOOR)
            active_idx=findall(<=(EB_COVARIANCE_FLOOR+floor_roundoff),evS.values)
            free_idx=setdiff(1:size(S,1),active_idx)
            displacement=if isempty(free_idx)
                # 全部方向被 floor 钉住且 active 块 KKT 违反（sc 无效）：
                # 唯一合法切向为零，保留原投影路径的 fail-closed 语义。
                projected=project_covariance_step(S,state.direction,nothing)
                projected-S
            else
                Vf=evS.vectors[:,free_idx]
                Matrix(Symmetric(Vf*(Vf'*state.direction*Vf)*Vf'))
            end
            # This is the only optimizer consumer of the full gradient.
            # Reuse the eigendecomposition of THIS S already needed above;
            # candidates and final certificates retain the original formulas.
            grad=_conditioned_covariance_gradient(cache,S,state.J,evS)
            ascent=dot(grad,displacement)
            trial=conditioned_covariance_ray(cache,S,displacement;M=state.M)
            step=1.0; accepted=false
            if isfinite(ascent) && ascent>0
                for _ in 1:max_backtracks
                    # 允许浮点求值的最后几位不同，但它绝不充当返回证书。
                    candidate_value=trial(step)
                    roundoff=8eps(max(abs(state.value),1.0))
                    if isfinite(candidate_value) && candidate_value>=state.value+1e-4step*ascent-roundoff
                        S=Matrix(Symmetric(S+step.*displacement))
                        accepted=true
                        break
                    end
                    step*=0.5
                end
            else
                # [ray 实测严格上升] evidence 平台上 dot(grad,·) 的数值
                # 可靠性崩坏（t=14294 iter=6 实测：dot=-6.6e-5 判定无上升，
                # 而 ray 上 step=1/2 处 evidence 真实上升 +7.6e-5 且 KKT
                # 证书 valid——dot 符号与实测矛盾；两个独立 grad 求值路径
                # 同号，理论恒等 dot(grad,(2/n)S·grad·S)≥0 却给出负值，
                # 证明该量级上判据不可分辨）。此时以 ray 上的实测严格上
                # 升为步进依据：要求 trial(step) 超过当前值加 8eps 噪声带
                # （比 Armijo 部分上升更严），任一步长通过即步进；全部
                # 回溯无上升才 fail-closed（"真的没有上升方向"的语义保
                # 留）。最终验收仍走原完整 KKT 证书。
                for _ in 1:max_backtracks
                    candidate_value=trial(step)
                    roundoff=8eps(max(abs(state.value),1.0))
                    if isfinite(candidate_value) && candidate_value>=state.value+roundoff
                        S=Matrix(Symmetric(S+step.*displacement))
                        accepted=true
                        break
                    end
                    step*=0.5
                end
                # A successful strict-rise step follows the original loop.
                # A failed ray must also reach the existing independently
                # certified terminal candidate below; it was accidentally
                # bypassed when the directional derivative was nonpositive.
                # No new candidate, tolerance or acceptance rule is added.
                accepted && continue
            end
            if !accepted && max_backtracks>0
                # [确定性收敛候选] 多线程 BLAS 求和顺序下，浅盆地上 evidence
                # 求值噪声可超过 Armijo 的 8eps 判定带，使全部回溯步长都过不了
                # 「可测上升」判据；此时 line search 既不能迈步也证不出无上升方向。
                # 在 fail-closed 之前，用当前 J 构造冻结驻点方程 S+SKS=Q 的闭式
                # 解（与 sigma_stationary_fixed_point 内部候选同一数学，不新增
                # 近似），交给完整证书原样验收；仅当证书通过才返回，否则走原
                # fail-closed。该候选是代数闭式，不经「逐次可测上升」瓶颈。
                # 本分支只在此前必抛错的路径上执行：所有已成功返回的轨迹
                # （含 BLAS1 下已绿的 R6）不受影响；max_backtracks=0 的预算
                # 拒绝路径由守卫排除，语义保持。
                Qroot=let ev=eigen(Symmetric(Matrix(cache.R)./cache.n))
                    (ev.vectors .* sqrt.(max.(ev.values,0.0))') * ev.vectors'
                end
                rc=riccati_stationary_candidate(Qroot,state.J,tr(S),length(cache.h),cache.n)
                if rc!==nothing
                    S_rc=positive_covariance(rc)
                    Sigma_rc=gauge===nothing ? S_rc : gauge*S_rc*gauge'
                    cert_rc=_conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma_rc,
                                                       scalar_geometry,alpha_geometry,cache;gauge,tol)
                    if cert_rc.valid
                        return return_certificate ? (; alpha,Sigma=Sigma_rc,certificate=cert_rc,
                                                       iterations=iteration) : (alpha,Sigma_rc)
                    end
                end
            end
            if !accepted
                isfinite(ascent) && ascent>0 ||
                    error("conditioned EB has no certified ascent direction: $sc")
                error("conditioned EB line search failed before stationarity")
            end
        end
    end
    # 耗尽：分项证书直接定位未收敛部分（alpha KKT vs Sigma 自由块/交叉/
    # 激活块），交下一次运行观察裁决，不再只报总数。
    Sigma=gauge===nothing ? S : gauge*S*gauge'
    final=_conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,
                                     scalar_geometry,alpha_geometry;gauge,tol)
    error("conditioned EB failed joint alpha/Sigma stationarity after $iters iterations: $final")
end

function fit_response_operator(B_m,B_rel,y_m,y_rel; ridge_alpha=nothing,ts=WARMUP:length(y_m)-1,degenerate_witness=nothing,
                               S_xx_rel=nothing,S_xy_rel=nothing,S_yy_rel=nothing,
                               X_design=nothing,alpha_initial=(1.0,1.0),timing=nothing,dual=nothing,need_uncertainty=true,workspace=nothing,
                               macro_stats=nothing)
    N=length(B_rel); n=length(ts)
    n>0 || throw(ArgumentError("response fit needs training rows"))
    if macro_stats===nothing
        Xm=B_m[ts,:]; ym=y_m[ts .+ 1]
        Sxx=Xm'*Xm; Sxy=reshape(Xm'*ym,:,1); Syy=fill(dot(ym,ym),1,1)
    else
        C=size(B_m,2)
        macro_stats.n==n && size(macro_stats.xx)==(C,C) && size(macro_stats.xy)==(C,1) &&
            size(macro_stats.yy)==(1,1) || throw(ArgumentError("macro_stats must describe exactly the current training rows"))
        Sxx=macro_stats.xx; Sxy=macro_stats.xy; Syy=macro_stats.yy
    end
    ridge_alpha===nothing || (isfinite(ridge_alpha) && ridge_alpha>0) ||
        throw(ArgumentError("fixed ridge alpha must be finite and positive"))
    ev=timed(timing,:eigen) do
        # optional stats 属于调用方；谱分解不能改写增量缓存或 SSE 的 Gram。
        eigen!(Symmetric(macro_stats===nothing ? Sxx : copy(Sxx)))
    end
    values=max.(ev.values,0.0); B=ev.vectors'*Sxy
    alpha_m,gamma_m=timed(timing,:EB) do
        ridge_alpha === nothing ? (res_m = optimize_matrix_normal_eb(values,B,Syy,n; initial=alpha_initial[1]); (res_m.alpha, res_m.gamma)) :
                                  (ridge_alpha,sum(values ./ (values .+ ridge_alpha)))
    end
    dm=1.0 ./ (values .+ alpha_m)
    gm=vec(ev.vectors*(B .* dm))
    sse=macro_stats===nothing ? sum(abs2,ym-Xm*gm) :
        Syy[1,1]-2dot(gm,vec(Sxy))+dot(gm,Sxx*gm)
    sig2=max(sse/max(n-gamma_m,1.0),1e-8)
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
    spectrum=timed(timing,:eigen) do
        ridge_spectrum(X_design,S_xx_rel,S_xy_rel; dual)
    end
    gauge=relative_gauge(N)
    alpha_rel,Sigma=timed(timing,:EB) do
        if ridge_alpha === nothing
            optimize_conditioned_eb(spectrum,S_yy_rel,n;initial=alpha_initial[2],gauge,degenerate_witness)
        else
            dr=1.0 ./ (spectrum.values .+ ridge_alpha)
            ridge_alpha,supported_covariance((S_yy_rel-spectrum.B'*(spectrum.B .* dr))./n,gauge)
        end
    end
    dr=1.0 ./ (spectrum.values .+ alpha_rel)
    # In-place GEMM: G = (spectrum.B .* dr)' * spectrum.basis' = B_scaled' * spectrum.basis'
    B_scaled_G = spectrum.B .* dr
    P_dim = size(spectrum.basis, 1)
    G = zeros(N, P_dim)
    BLAS.gemm!('T', 'T', 1.0, B_scaled_G, spectrum.basis, 0.0, G)
    # The exact constrained posterior mean G_c = G - Omega C' (C Omega C')^-1 C G
    # needs the posterior covariance V even when covariance factors are not
    # exported (need_uncertainty=false); wrapping the spectral factors is free.
    # The former Euclidean per-band trace subtraction is NOT that mean: it
    # spreads the correction uniformly across assets, while the exact mean
    # weights it by Omega (band/channel posterior variance through V, asset
    # correlation through Sigma). The 14x14 solve is negligible in cost.
    V = ridge_covariance(spectrum, alpha_rel)
    cond=timed(timing,:condition) do
        condition_trace_neutrality(G,V,Sigma,N,length(BANDS))
    end
    # OOF folds consume only the conditioned mean; the covariance factor itself
    # is deliberately not exported for them.
    V_out = need_uncertainty ? V : RidgeCovariance(zeros(P_dim, 0), Float64[], 0.0)
    trA=sum(sum(cond.G_c[j,cond.cols[2b-1][j]] for j in 1:N) for b in eachindex(BANDS))
    trB=sum(sum(cond.G_c[j,cond.cols[2b][j]] for j in 1:N) for b in eachindex(BANDS))
    ResponseOperator(gm,covm,cond.G_c,V_out,Sigma,cond.inv_M,cond.cols,alpha_m,alpha_rel,trA,trB)
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
