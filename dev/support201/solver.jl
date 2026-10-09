# Structural support / joint-Newton candidate for 2.0.1. Development only
# until original-law, reference-output and wall-time acceptance completes.
module Support201
using KTrader, LinearAlgebra

const DELTA=KTrader.EB_COVARIANCE_FLOOR
const ALO=KTrader.EB_ALPHA_MIN
const AHI=KTrader.EB_ALPHA_MAX
sym(A)=Matrix(Symmetric((A+A')/2))
function svec(A)
    d=size(A,1); v=Vector{Float64}(undef,d*(d+1)÷2); k=0
    for j in 1:d, i in 1:j
        k+=1; v[k]=i==j ? A[i,j] : sqrt(2.0)*A[i,j]
    end
    v
end
function smat(v,d)
    length(v)==d*(d+1)÷2 || throw(DimensionMismatch("symmetric coordinates"))
    A=zeros(d,d); k=0
    for j in 1:d, i in 1:j
        k+=1; A[i,j]=A[j,i]=i==j ? v[k] : v[k]/sqrt(2.0)
    end
    A
end
pair(i,j,m)=(i-1)*m+j-i*(i-1)÷2

"""Construct support from all masks needed by the training feature prefix.
No eigenvalue threshold or response-value inference determines this space."""
function support_basis(returns,cutoff)
    1<=cutoff<=size(returns,1) || throw(ArgumentError("invalid support history"))
    N=size(returns,2); parent=collect(1:N)
    function root(i)
        while parent[i]!=i
            parent[i]=parent[parent[i]]; i=parent[i]
        end
        i
    end
    for t in 1:cutoff
        first_seen=0
        for j in 1:N
            isfinite(returns[t,j]) || continue
            if first_seen==0
                first_seen=j
            else
                parent[root(j)]=root(first_seen)
            end
        end
    end
    groups=Dict{Int,Vector{Int}}()
    for j in 1:N; push!(get!(groups,root(j),Int[]),j); end
    components=sort!(collect(values(groups));by=first)
    rank=sum(length(g)-1 for g in components)
    Z=zeros(N,rank); start=0
    for g in components
        k=length(g)-1; k==0 && continue
        Z[g,start+1:start+k]=KTrader.relative_gauge(length(g)); start+=k
    end
    (;Z,components,cutoff)
end

"""Data-only support compression. All original null-prior terms survive in
the objective and in lift_spectrum; a structural witness, not a tolerance,
selects the coordinates. Residual checks only reject inconsistent witnesses."""
function prepare(xx,xy,yy,n,witness)
    Z=witness.Z; N,r=size(Z); m=2length(KTrader.BANDS); d=N-1; a=d-r
    n>0 && 1<r<d || throw(ArgumentError("proper nontrivial relative support required"))
    all(isfinite,Z) && norm(Z'*Z-I)<=1e-12*max(r,1) && norm(sum(Z;dims=1))<=1e-12*max(r,1) ||
        throw(ArgumentError("support basis must be finite, orthonormal and relative"))
    size(xx)==(m*N,m*N) && size(xy)==(m*N,N) && size(yy)==(N,N) ||
        throw(DimensionMismatch("support sufficient statistics"))
    P=m*r; XR=zeros(P,P); XY=zeros(P,r)
    residual_x=0.0; residual_y=0.0
    for c in 1:m
        rows=(c-1)*N+1:c*N; cr=(c-1)*r+1:c*r
        block=view(xy,rows,:); XY[cr,:]=Z'*block*Z
        residual_y+=sum(abs2,block-Z*view(XY,cr,:)*Z')
        for e in c:m
            cols=(e-1)*N+1:e*N; er=(e-1)*r+1:e*r
            source=view(xx,rows,cols); reduced=Z'*source*Z
            XR[cr,er]=reduced; XR[er,cr]=reduced'
            residual_x+=(c==e ? 1 : 2)*sum(abs2,source-Z*reduced*Z')
        end
    end
    YY=sym(Z'*yy*Z)
    leaks=(sqrt(residual_x)/max(norm(xx),eps()),sqrt(residual_y)/max(norm(xy),eps()),
        norm(yy-Z*YY*Z')/max(norm(yy),eps()))
    maximum(leaks)<=1e-10 || throw(ArgumentError("structural support witness does not describe this fit: $leaks"))
    eig=eigen(Symmetric(XR)); values=max.(eig.values,0.0); U=eig.vectors
    B=U'*XY
    blocks=[Matrix(view(U,(c-1)*r+1:c*r,:)) for c in 1:m]
    hc=zeros(P,m)
    for c in 1:m, k in 1:P; hc[k,c]=dot(view(B,k,:),view(blocks[c],:,k)); end
    # Keep a private basis copy: a caller mutating a witness must not change
    # the coordinate meaning of an already prepared candidate.
    (;Z=copy(Z),N,r,d,a,m,n,values,U,B,YY,blocks,hc,leaks,xx=XR,xy=XY)
end

function alpha_cache(g,alpha)
    ALO<=alpha<=AHI || throw(ArgumentError("alpha outside original bounds"))
    v=1.0 ./ (g.values .+ alpha); va=-alpha .* v.^2
    vaa=va .+ (2alpha^2) .* v.^3
    R=sym(g.YY-g.B'*(g.B.*v))
    Ra=sym(-g.B'*(g.B.*va)); Raa=sym(-g.B'*(g.B.*vaa))
    h=g.hc'*v; ha=g.hc'*va; haa=g.hc'*vaa
    q=g.m*(g.m+1)÷2; H=Matrix{Float64}(undef,g.r*g.r,q+1)
    weighted=[block.*v' for block in g.blocks]
    index=0
    for j in 1:g.m, i in 1:j
        index+=1; factor=i==j ? 1.0 : sqrt(2.0)
        H[:,index]=factor.*vec(sym(weighted[i]*g.blocks[j]'))
    end
    H[:,end]=vec(Matrix{Float64}(I,g.r,g.r))
    (;alpha,v,va,vaa,R,Ra,Raa,h,ha,haa,H)
end

# Contract the derivative of the cores directly:14 GEMMs, not105 derivative
# core matrices. No low-rank cutoff or finite-difference derivative is used.
function weighted_adjoint(g,weights,K)
    J=zeros(g.r,g.r); mixed=zeros(size(first(g.blocks)))
    for i in 1:g.m
        fill!(mixed,0.0)
        for j in 1:g.m
            mixed .+= K[i,j].*g.blocks[j]
        end
        mul!(J,g.blocks[i].*weights',mixed',1.0,1.0)
    end
    sym(J)
end

"""Exact reduced negative evidence and derivatives w.r.t. (S,logalpha).
Original output d, null-prior M offset and trace offset are retained."""
function state(g,c,S;second=true)
    cs=cholesky(Symmetric(S)); P=Matrix(inv(cs)); t=tr(S)+g.a*DELTA
    ms=c.H'*vec(S); M=smat(ms[1:end-1],g.m)
    tensors=KTrader._conditioned_scalar_tensors(g.blocks,S)
    Ma=zeros(g.m,g.m); Maa=similar(Ma)
    for i in 1:g.m,j in i:g.m
        column=view(tensors,:,pair(i,j,g.m))
        Ma[i,j]=Ma[j,i]=dot(c.va,column)
        Maa[i,j]=Maa[j,i]=dot(c.vaa,column)
    end
    offset=g.a*DELTA/c.alpha
    for i in 1:g.m; M[i,i]+=offset; Ma[i,i]-=offset; Maa[i,i]+=offset; end
    cm=cholesky(Symmetric(M)); V=Matrix(inv(cm)); z=cm\c.h; K=V-z*z'
    za=cm\(c.ha-Ma*z); Ka=-V*Ma*V-za*z'-z*za'
    moments=vcat(svec(K)/2,-g.m/(2t))
    gradient=sym((g.n.*P-P*c.R*P)/2+reshape(c.H*moments,g.r,g.r))
    ga=-g.d/2*sum(g.values.*c.v)+tr(P*c.Ra)/2+dot(K,Ma)/2+dot(z,c.ha)+g.m/2
    # By retaining original moment contractions, joint alpha derivatives
    # include BOTH data spectral weights and the analytic null prior.
    ua=sym(-P*c.Ra*P/2+weighted_adjoint(g,c.va,K)/2+
        reshape(c.H*vcat(svec(Ka)/2,0.0),g.r,g.r))
    aa=g.d/2*sum(g.values.*(c.alpha.*c.v.^2))+tr(P*c.Raa)/2+
        dot(Ka,Ma)/2+dot(K,Maa)/2+dot(za,c.ha)+dot(z,c.haa)
    parts=[g.d/2*sum(log1p.(g.values./c.alpha)),g.n/2*(logdet(cs)+g.a*log(DELTA)),
        tr(cs\c.R)/2,logdet(cm)/2,dot(c.h,z)/2,-g.m/2*log(t/c.alpha)]
    G=zeros(size(c.H,2),size(c.H,2))
    if second
        index=0
        for j in 1:g.m, i in 1:j
            index+=1; E=zeros(g.m,g.m)
            E[i,j]=E[j,i]=i==j ? 1.0 : inv(sqrt(2.0))
            G[1:end-1,index]=svec((-V*E*V+V*E*(z*z')+(z*z')*E*V)/2)
        end
        G[end,end]=g.m/(2t^2); G=sym(G)
    end
    J=reshape(c.H[:,1:end-1]*svec(K),g.r,g.r)
    direction=sym((-g.n.*S.+c.R.-S*J*S.+(g.m/t).*(S*S))./g.n)
    root=Matrix(cs.L); residual=norm(root\direction/root')/sqrt(g.r)
    gamma=-g.n/(2DELTA)-tr(K)/(2c.alpha)+g.m/(2t)
    scale=max(g.d*sum(g.values.*c.v),2length(KTrader.BANDS),1.0)
    ac=KTrader.bounded_alpha_certificate(c.alpha,-ga,scale)
    (;value=sum(parts),parts,gradient,ga,ua,aa,G,M,Ma,Maa,z,K,P,t,residual,gamma,ac,direction)
end

function hessian_action(g,c,S,s,D)
    P=s.P
    base=(-g.n.*(P*D*P)+P*D*P*c.R*P+P*c.R*P*D*P)/2
    sym(base+reshape(c.H*(s.G*(c.H'*vec(D))),g.r,g.r))
end

function newton_direction(g,c,S,s;joint=true)
    L=Matrix(cholesky(Symmetric(S)).L); ev=eigen(Symmetric(L\c.R/L'))
    D=(ev.values .+ ev.values' .- g.n)./2
    minimum(D)>0 || return nothing
    T=L*ev.vectors; q=size(c.H,2); F=similar(c.H)
    for j in 1:q; F[:,j]=vec(T'*reshape(view(c.H,:,j),g.r,g.r)*T); end
    W=F'*(F./vec(D)); B=T'*s.gradient*T; U=T'*s.ua*T
    b=F'*vec(B./D); v=F'*vec(U./D)
    if joint
        system=[Matrix{Float64}(I,q,q)+W*s.G v; -v'*s.G s.aa-dot(U,U./D)]
        rhs=vcat(-b,-s.ga+dot(U,B./D))
        sol=system\rhs; y=sol[1:q]; da=sol[end]
    else
        y=(Matrix{Float64}(I,q,q)+W*s.G)\(-b); da=0.0
    end
    dS=sym(T*((-B-reshape(F*(s.G*y),g.r,g.r)-da*U)./D)*T')
    predicted=dot(s.gradient,dS)+s.ga*da
    all(isfinite,dS) && isfinite(da) && predicted<0 || return nothing
    (;dS,da,predicted)
end

# This is the ORIGINAL full alpha-search candidate policy applied to an
# algebraically compressed evidence, not a local warm-alpha root search.
function scalar_functions(g,S)
    cs=cholesky(Symmetric(S)); solved=cs\g.B'
    quad=[dot(view(g.B,k,:),view(solved,:,k)) for k in axes(g.B,1)]
    tensors=KTrader._conditioned_scalar_tensors(g.blocks,S)
    t=tr(S)+g.a*DELTA
    constant=g.n/2*(logdet(cs)+g.a*log(DELTA))+tr(cs\g.YY)/2
    function profile(logalpha)
        alpha=exp(logalpha); v=1.0./(g.values.+alpha); va=-alpha.*v.^2
        M=zeros(g.m,g.m); Ma=similar(M)
        for i in 1:g.m, j in i:g.m
            idx=pair(i,j,g.m)
            M[i,j]=M[j,i]=dot(v,view(tensors,:,idx))
            Ma[i,j]=Ma[j,i]=dot(va,view(tensors,:,idx))
        end
        offset=g.a*DELTA/alpha
        for i in 1:g.m; M[i,i]+=offset; Ma[i,i]-=offset; end
        h=g.hc'*v; ha=g.hc'*va; cm=cholesky(Symmetric(M)); z=cm\h
        value=constant+g.d/2*sum(log1p.(g.values./alpha))-dot(v,quad)/2+
            logdet(cm)/2+dot(h,z)/2-g.m/2*log(t/alpha)
        slope=-g.d/2*sum(g.values.*v)-dot(va,quad)/2+
            tr(cm\Ma)/2-dot(z,Ma*z)/2+dot(z,ha)+g.m/2
        (;value=-value,slope=-slope)
    end
    (;evidence=x->profile(x).value,derivative=x->profile(x).slope)
end

function lift_spectrum(g)
    P=g.m*g.N; rank=length(g.values); basis=zeros(P,rank)
    for c in 1:g.m
        basis[(c-1)*g.N+1:c*g.N,:]=g.Z*view(g.U,(c-1)*g.r+1:c*g.r,:)
    end
    # Exact unresolved input variance remains 1/alpha in RidgeCovariance.
    (;values=g.values,basis,B=g.B*g.Z',dual=true)
end
function lift_covariance(g,S)
    Q=KTrader.relative_gauge(g.N); Pa=Q*Q'-g.Z*g.Z'
    sym(g.Z*S*g.Z'+DELTA*Pa)
end

function fixed_value(g,c,S)
    cs=cholesky(Symmetric(S)); moments=c.H'*vec(S)
    M=smat(moments[1:end-1],g.m)
    for i in 1:g.m; M[i,i]+=g.a*DELTA/c.alpha; end
    cm=cholesky(Symmetric(M)); t=tr(S)+g.a*DELTA
    g.d/2*sum(log1p.(g.values./c.alpha))+g.n/2*(logdet(cs)+g.a*log(DELTA))+
        tr(cs\c.R)/2+logdet(cm)/2+dot(c.h,cm\c.h)/2-g.m/2*log(t/c.alpha)
end

"""Sufficient fixed-alpha uniqueness enclosure in the retained support.
H_phi >= -(m/2)||S^-1/2 H S^-1/2||_F^2 and H0 has coefficient
lambda_min(S^-1/2 R S^-1/2)-n/2. Outside S < beta*R/n,
the Gaussian part exceeds its minimum by n/2*(log(beta)+1/beta-1).
This proves a unique minimum for a sublevel enclosed by this boundary;
it is NOT a joint-alpha/global-nullspace uniqueness assertion."""
function fixed_enclosure(g,c,S,s=state(g,c,S;second=false))
    g.n>g.m || return (;valid=false,margin=-Inf)
    Rn=c.R./g.n
    isposdef(Symmetric(Rn)) || return (;valid=false,margin=-Inf)
    L=Matrix(cholesky(Symmetric(Rn)).L)
    mu=eigvals(Symmetric(L\S/L'))
    minimum(mu)>0 || return (;valid=false,margin=-Inf)
    beta=2g.n/(g.n+g.m)
    gaussian_gap=g.n/2*sum(log.(mu).+1.0./mu.-1.0)
    phi=s.parts[4]+s.parts[5]+s.parts[6]
    phi_lower=-g.m/2*log1p(maximum(g.values)/c.alpha)
    boundary=g.n/2*(log(beta)+1/beta-1)
    margin=boundary+phi_lower-gaussian_gap-phi
    roundoff=4096eps(Float64)*max(abs(boundary),abs(phi),abs(gaussian_gap),1.0)
    (;valid=maximum(mu)<beta && margin>roundoff,margin,beta,maximum_mu=maximum(mu))
end

function solve_fixed(g,c,start;tol=1e-13,maxiters=20)
    S=copy(start)
    for i in 1:maxiters
        s=state(g,c,S)
        s.residual<=tol && return (;valid=true,S,state=s,steps=i)
        direction=newton_direction(g,c,S,s;joint=false)
        direction===nothing && return (;valid=false,reason=:base_or_direction)
        eta=1.0; accepted=false
        for _ in 1:24
            trial=sym(S+eta*direction.dS)
            if minimum(eigvals(Symmetric(trial)))>DELTA
                value=fixed_value(g,c,trial)
                if value<=s.value+1e-4eta*direction.predicted+8eps(max(abs(s.value),1.0))
                    S=trial; accepted=true; break
                end
            end
            eta*=0.5
        end
        accepted || return (;valid=false,reason=:fixed_line_search)
    end
    (;valid=false,reason=:fixed_budget)
end

"""Preserve reference alpha alternation and its final1e-6 certificate.
Only the covariance subsolve is accelerated. When its sublevel is not
enclosed, use the EXISTING reference fixed-point proposal rather than a
warm answer. A trajectory seed only enters an enclosed fixed-alpha problem.
Original-output comparisons still gate promotion of this development path."""
function solve_reference_schedule(g;initial=1.0,seeds=nothing,maxiters=32,trace=false)
    first_cache=alpha_cache(g,1.0); S=KTrader.positive_covariance(first_cache.R./g.n)
    minimum(eigvals(Symmetric(S)))>DELTA*(1+1e-8) || return (;accepted=false,reason=:nonstructural_floor)
    alpha=initial; spectrum=lift_spectrum(g); Q=KTrader.relative_gauge(g.N)
    yy=g.Z*g.YY*g.Z'; trajectory=Matrix{Float64}[]; newton_steps=0; fixed_fallbacks=0
    for iteration in 1:maxiters
        scalar=scalar_functions(g,S)
        alpha=KTrader.maximize_logalpha(scalar.evidence,scalar.derivative;initial=alpha)
        c=alpha_cache(g,alpha); before=state(g,c,S)
        enclosed=fixed_enclosure(g,c,S,before)
        if enclosed.valid
            start=seeds!==nothing && iteration<=length(seeds) ? seeds[iteration] : S
            size(start)==size(S) || return (;accepted=false,reason=:seed_shape)
            correction=solve_fixed(g,c,start)
            if !correction.valid || correction.state.value>before.value+8eps(max(abs(before.value),1.0)) ||
                !fixed_enclosure(g,c,correction.S,correction.state).valid
                correction=solve_fixed(g,c,S)
            end
            correction.valid || return (;accepted=false,reason=:fixed_corrector)
            Snew=correction.S; newton_steps+=correction.steps
        else
            fixed_fallbacks+=1
            original_cache=KTrader.conditioned_alpha_cache(spectrum,yy,g.n,alpha;gauge=Q)
            fullS=Q'*lift_covariance(g,S)*Q
            proposal=KTrader.sigma_stationary_fixed_point(original_cache,fullS)
            Snew=sym(g.Z'*Q*proposal*Q'*g.Z)
        end
        after=state(g,c,Snew;second=false)
        after.value<=before.value+8eps(max(abs(before.value),1.0)) &&
            after.residual<=before.residual+1e-12 && after.gamma<=0 ||
            return (;accepted=false,reason=:reference_acceptance,iteration,before=before.residual,after=after.residual)
        S=Snew; push!(trajectory,copy(S))
        Sigma=lift_covariance(g,S)
        cert=KTrader.conditioned_eb_certificate(spectrum,yy,g.n,alpha,Sigma;gauge=Q)
        trace && (println("reference-schedule iter=",iteration," alpha=",alpha,
            " cov_res=",after.residual," alpha_res=",cert.alpha.rel_res,
            " enclosure_margin=",enclosed.margin," enclosed=",enclosed.valid,
            " steps=",newton_steps," fixed_fallbacks=",fixed_fallbacks);flush(stdout))
        if cert.valid
            return (;accepted=true,alpha,S,Sigma,spectrum,certificate=cert,iterations=iteration,
                newton_steps,fixed_fallbacks,trajectory)
        end
    end
    (;accepted=false,reason=:outer_budget)
end

"""Deterministic cold-start joint corrector. No cross-fold/day posterior
warm start is accepted: the old branch-selection contract remains a gate.
Returns only after original ambient certificates are evaluated; acceptance
against the reference outputs is still mandatory before production wiring."""
function solve_candidate(g;initial=1.0,initial_state=nothing,maxiters=32,tol=1e-10,trace=false)
    initial>0 && isfinite(initial) && maxiters>0 || throw(ArgumentError("candidate budget/initial"))
    # A warm state is a RESEARCH continuation experiment, not production
    # branch acceptance; its cold-reference/output gates cannot be skipped.
    first=alpha_cache(g,1.0)
    S=initial_state===nothing ? KTrader.positive_covariance(first.R./g.n) : copy(initial_state)
    size(S)==(g.r,g.r) || throw(DimensionMismatch("candidate continuation coordinates"))
    minimum(eigvals(Symmetric(S)))>DELTA*(1+1e-8) || return (;accepted=false,reason=:nonstructural_floor)
    alpha=initial; evaluations=0; scans=0; spectrum=lift_spectrum(g); Q=KTrader.relative_gauge(g.N)
    for iteration in 1:maxiters
        scalar=scalar_functions(g,S)
        alpha=KTrader.maximize_logalpha(scalar.evidence,scalar.derivative;initial=alpha)
        scans+=1; c=alpha_cache(g,alpha); s=state(g,c,S); evaluations+=1
        trace && (println("candidate iter=",iteration," alpha=",alpha," cov_res=",s.residual,
            " alpha_res=",s.ac.rel_res," null_gradient=",s.gamma," objective=",s.value);flush(stdout))
        if s.residual<=tol && s.ac.rel_res<=tol && s.gamma<=0
            Sigma=lift_covariance(g,S)
            certificate=KTrader.conditioned_eb_certificate(spectrum,g.Z*g.YY*g.Z',g.n,alpha,Sigma;gauge=Q)
            certificate.valid || return (;accepted=false,reason=:ambient_certificate,certificate)
            return (;accepted=true,alpha,S,Sigma,certificate,iterations=iteration,evaluations,scans,spectrum)
        end
        proposal=newton_direction(g,c,S,s;joint=true)
        if proposal===nothing
            proposal=newton_direction(g,c,S,s;joint=false)
        end
        proposal===nothing && return (;accepted=false,reason=:non_descent,iteration)
        eta=1.0; success=false
        for _ in 1:24
            candidateS=sym(S+eta*proposal.dS)
            logalpha=log(alpha)+eta*proposal.da
            if log(ALO)<=logalpha<=log(AHI) && minimum(eigvals(Symmetric(candidateS)))>DELTA
                candidate_alpha=exp(logalpha)
                scalar_trial=scalar_functions(g,candidateS)
                nextvalue=-scalar_trial.evidence(logalpha); evaluations+=1
                decrease=nextvalue-s.value
                roundoff=8eps(max(abs(s.value),1.0))
                if isfinite(nextvalue) && decrease<=1e-4eta*proposal.predicted+roundoff
                    S=candidateS; alpha=candidate_alpha; success=true; break
                end
            end
            eta*=0.5
        end
        success || return (;accepted=false,reason=:line_search,iteration)
    end
    (;accepted=false,reason=:iteration_budget)
end
end
