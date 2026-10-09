# Algebraic research only. Not included by KTrader, not a production solver.
# The Newton proposal below has no acceptance policy and cannot replace the
# original alpha/covariance/Kelly certificates or reference branch selection.
module IncrementalTheory
using LinearAlgebra

sym(A) = Matrix(Symmetric((A+A')/2))
function symmetric_basis(n)
    out=Matrix{Float64}[]
    for j in 1:n, i in 1:j
        E=zeros(n,n)
        if i==j
            E[i,j]=1.0
        else
            E[i,j]=E[j,i]=inv(sqrt(2.0))
        end
        push!(out,E)
    end
    out
end
svec(A)=Float64[dot(A,E) for E in symmetric_basis(size(A,1))]
function smat(v,n)
    length(v)==n*(n+1)÷2 || throw(DimensionMismatch("symmetric coordinates"))
    out=zeros(n,n)
    for (x,E) in zip(v,symmetric_basis(n)); out .+= x.*E; end
    out
end

function mask_projector(mask)
    v=Float64.(mask); c=sum(v)
    c==0 && return zeros(length(v),length(v))
    Matrix(Diagonal(v))-v*v'/c
end

# moment_map maps Sigma to [svec(M(Sigma)); tr(Sigma)].  The ordering of
# source cores is row-upper (r,s), whereas svec is column-upper (i,j).
function moment_map(cores,m,d;baseline=0.0)
    length(cores)==m*(m+1)÷2 || throw(DimensionMismatch("constraint cores"))
    index(r,s)=(r-1)*m+s-r*(r-1)÷2
    maps=Matrix{Float64}[]
    for j in 1:m, i in 1:j
        A=copy(cores[index(i,j)])
        if i==j
            A .+= baseline.*Matrix{Float64}(I,d,d)
        else
            A .*= sqrt(2.0)
        end
        push!(maps,A)
    end
    push!(maps,Matrix{Float64}(I,d,d))
    maps
end
moments(S,maps)=Float64[dot(A,S) for A in maps]
function adjoint_moments(v,maps)
    length(v)==length(maps) || throw(DimensionMismatch("moment adjoint"))
    out=zeros(size(first(maps)))
    for (x,A) in zip(v,maps); out .+= x.*A; end
    out
end

# Negative fixed-alpha log evidence; constant independent of Sigma omitted.
function loss(S,R,n,maps,h)
    m=length(h); t=moments(S,maps); M=smat(t[1:end-1],m)
    cs=cholesky(Symmetric(S)); cm=cholesky(Symmetric(M))
    n/2*logdet(cs)+0.5tr(cs\R)+0.5logdet(cm)+0.5dot(h,cm\h)-m/2*log(t[end])
end
function gradient(S,R,n,maps,h)
    m=length(h); t=moments(S,maps); M=smat(t[1:end-1],m)
    P=inv(Symmetric(S)); V=inv(Symmetric(M)); z=V*h
    J=adjoint_moments(vcat(0.5svec(V-z*z'),-m/(2t[end])),maps)
    sym((n.*P-P*R*P)/2+J)
end
function moment_hessian(S,maps,h)
    m=length(h); t=moments(S,maps); M=smat(t[1:end-1],m)
    V=inv(Symmetric(M)); z=V*h; zz=z*z'
    q=length(maps); G=zeros(q,q)
    for (j,E) in enumerate(symmetric_basis(m))
        G[1:end-1,j]=svec((-V*E*V+V*E*zz+zz*E*V)/2)
    end
    G[end,end]=m/(2t[end]^2)
    sym(G)
end
function base_hessian(H,S,R,n)
    P=inv(Symmetric(S))
    sym((-n.*(P*H*P)+P*H*P*R*P+P*R*P*H*P)/2)
end
function hessian_action(H,S,R,n,maps,h)
    base_hessian(H,S,R,n)+adjoint_moments(moment_hessian(S,maps,h)*moments(H,maps),maps)
end

# Exact Newton linear solve by a q-dimensional Schur system, provided the
# base operator is positive definite. q=m(m+1)/2+1, independent of asset d.
# This is NOT an assertion that the full Hessian is positive definite, or
# that the step respects the covariance floor or the alpha search contract.
function reduced_newton_direction(S,R,n,maps,h)
    L=Matrix(cholesky(Symmetric(S)).L)
    ev=eigen(Symmetric(L\R/L'))
    D=(ev.values .+ ev.values' .- n)./2
    minimum(D)>0 || throw(DomainError(minimum(D),"base Hessian not positive definite; no local Schur proposal"))
    T=L*ev.vectors; q=length(maps); d=size(S,1)
    F=hcat([vec(T'*A*T) for A in maps]...)
    W=F'*(F./vec(D))
    g=gradient(S,R,n,maps,h); B=T'*g*T
    G=moment_hessian(S,maps,h)
    y=(Matrix{Float64}(I,q,q)+W*G)\(-F'*vec(B./D))
    delta=sym(T*((-B-reshape(F*(G*y),d,d))./D)*T')
    linear_residual=norm(hessian_action(delta,S,R,n,maps,h)+g)/(1+norm(g))
    (;delta,linear_residual,q,base_minimum=minimum(D))
end

# Joint scalar-alpha/covariance block algebra. cross and aa MUST be the
# exact current problem's mixed derivative and alpha-alpha derivative;
# this function does not provide or approximate those expensive derivatives.
# Tests supply a manufactured joint quadratic to verify elimination only.
function reduced_joint_direction(S,R,n,maps,h,cross,aa,ga)
    L=Matrix(cholesky(Symmetric(S)).L)
    ev=eigen(Symmetric(L\R/L'))
    D=(ev.values .+ ev.values' .- n)./2
    minimum(D)>0 || throw(DomainError(minimum(D),"base Hessian not positive definite"))
    T=L*ev.vectors; q=length(maps); d=size(S,1)
    F=hcat([vec(T'*A*T) for A in maps]...)
    W=F'*(F./vec(D)); G=moment_hessian(S,maps,h)
    g=gradient(S,R,n,maps,h); B=T'*g*T; U=T'*cross*T
    v=F'*vec(U./D); b=F'*vec(B./D)
    system=[Matrix{Float64}(I,q,q)+W*G v; -v'*G aa-dot(U,U./D)]
    solution=system\vcat(-b,-ga+dot(U,B./D))
    y=solution[1:q]; da=solution[end]
    dS=sym(T*((-B-reshape(F*(G*y),d,d)-da*U)./D)*T')
    (;dS,da,system_size=q+1)
end

# Differential of the floor projection away from its nondifferentiable
# threshold. Supplies a semismooth-system action, NOT a floor shortcut.
function floor_projection_differential(Z,H,floor)
    ev=eigen(Symmetric(Z)); v=ev.values; U=ev.vectors
    any(==(floor),v) && throw(DomainError(floor,"projection derivative at threshold needs generalized Jacobian"))
    a=max.(v,floor); G=zeros(length(v),length(v))
    for j in eachindex(v), i in eachindex(v)
        G[i,j]=v[i]==v[j] ? Float64(v[i]>floor) : (a[i]-a[j])/(v[i]-v[j])
    end
    sym(U*(G.*(U'*H*U))*U')
end
end
