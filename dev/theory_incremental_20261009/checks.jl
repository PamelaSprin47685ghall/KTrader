# Small deterministic identity/counterexample checks; zero fits/backtests.
using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"identities.jl"))
using .IncrementalTheory
const IT=IncrementalTheory
BLAS.set_num_threads(1)
rng=MersenneTwister(20261009)

@testset "Metric transport: mask commutator, rank and lost common component" begin
    for N in (3,7,12), K in (1,2,3)
        masks=[rand(rng,Bool,N) for _ in 1:K]
        foreach(m->(m[1]=true),masks)
        ell=0.9 .+ 0.2rand(rng,N); L=Diagonal(ell)
        old=zeros(50,N); new=similar(old)
        for t in axes(old,1)
            mask=masks[mod1(t,K)]; m=Float64.(mask); P=IT.mask_projector(mask)
            u=randn(rng,N).*m
            correction=((L*m)*dot(m,u)-m*dot(L*m,u))/sum(m)
            @test P*L*u≈L*P*u+correction atol=1e-13 rtol=1e-12
            old[t,:]=P*u; new[t,:]=P*L*u
        end
        residual=new-old*L
        s=svdvals(residual)
        @test count(>(maximum(s)*1e-10),s)<=min(N,2K)
        # Gram difference after transport, using exact factorization of the
        # residual only for this small proof check, not rank truncation.
        U=residual; V=Matrix{Float64}(I,N,N)
        Z=hcat(L*old'*U,V)
        C=[zeros(N,N) Matrix{Float64}(I,N,N); Matrix{Float64}(I,N,N) U'*U]
        @test new'*new-L*old'*old*L≈Z*C*Z' atol=1e-11 rtol=1e-10
    end
    N=5; P=IT.mask_projector(trues(N)); L=Diagonal(collect(1.0:1.0:N))
    u=randn(rng,N)
    @test P*u≈P*(u.+2) atol=1e-14
    @test norm(P*L*u-P*L*(u.+2))>1.0
    # Congruence does not preserve an isotropic ridge prior in new units.
    X=randn(rng,30,N); alpha=0.7; A=X'*X+alpha*I
    D=Diagonal(collect(range(0.8,1.2;length=N)))
    next=(X*D)'*(X*D)+alpha*I
    defect=alpha*(Matrix{Float64}(I,N,N)-D*D)
    @test next-D*A*D≈defect atol=1e-12
    @test norm(defect)>0.1
end

@testset "Moving contiguous folds only change bounded many rows at fixed metric" begin
    F=3; C=7; X=randn(rng,70,C)
    for n in 30:49
        ranges(n)=[((f-1)*div(n,F)+1):(f==F ? n : f*div(n,F)) for f in 1:F]
        for f in 1:F
            a=setdiff(1:n,ranges(n)[f]); b=setdiff(1:n+1,ranges(n+1)[f])
            plus=setdiff(b,a); minus=setdiff(a,b)
            @test length(plus)+length(minus)<=2F-2
            difference=X[b,:]'*X[b,:]-X[a,:]'*X[a,:]
            update=X[plus,:]'*X[plus,:]-X[minus,:]'*X[minus,:]
            @test difference≈update atol=1e-12 rtol=1e-12
        end
    end
end

@testset "Fixed-alpha objective: exact moment Hessian and reduced Newton equation" begin
    worst_fd=0.0; worst_step=0.0; worst_original=0.0
    for N in (3,5,7), dual in (false,true)
        P=14N; rows=dual ? N+6 : P+8
        X=randn(rng,rows,P); Q=KTrader.relative_gauge(N)
        Y=randn(rng,rows,N)*Q*Q'
        spectrum=KTrader.ridge_spectrum(X,X'*X,X'*Y;dual)
        cache=KTrader.conditioned_alpha_cache(spectrum,Y'*Y,rows,0.7;gauge=Q)
        d=N-1; m=length(cache.h)
        cores=KTrader.build_jacobian_cores(cache)
        maps=IT.moment_map(cores,m,d;baseline=cache.V.baseline)
        @test length(maps)==106
        Z=randn(rng,d,d); S=IT.sym(Z*Z'/d+I)
        # Match the actual production gradient on its own R/n/h, including
        # dual null-space baseline. This is a pure cache evaluation, no fit.
        source=KTrader.conditioned_cached_state(cache,S;gradient=true)
        g=IT.gradient(S,cache.R,cache.n,maps,cache.h)
        original_error=norm(g+source.grad)/(1+norm(source.grad))
        worst_original=max(worst_original,original_error)
        @test original_error<1e-10
        @test IT.smat(IT.moments(S,maps)[1:end-1],m)≈source.M atol=1e-10 rtol=1e-11
        # Controlled positive-base fixture for the linear Newton identity;
        # this is NOT a production posterior or a claim all real bases pass.
        n=500; R=n*S; h=0.1cache.h
        E=IT.symmetric_basis(d)
        H0=hcat([IT.svec(IT.base_hessian(A,S,R,n)) for A in E]...)
        U=hcat([IT.moments(A,maps) for A in E]...)
        G=IT.moment_hessian(S,maps,h)
        full=H0+U'*G*U
        D=IT.sym(randn(rng,d,d)); epsilon=1e-5
        fd=(IT.gradient(S+epsilon*D,R,n,maps,h)-IT.gradient(S-epsilon*D,R,n,maps,h))/(2epsilon)
        fd_error=norm(IT.svec(fd)-full*IT.svec(D))/(1+norm(fd))
        worst_fd=max(worst_fd,fd_error)
        @test fd_error<1e-7
        proposal=IT.reduced_newton_direction(S,R,n,maps,h)
        dense=IT.smat(full\(-IT.svec(IT.gradient(S,R,n,maps,h))),d)
        error=norm(proposal.delta-dense)/(1+norm(dense))
        worst_step=max(worst_step,error)
        @test error<1e-10
        @test proposal.linear_residual<1e-10
        # General scalar/covariance joint block: exact linear algebra only,
        # not a fabricated analytic derivative of production alpha evidence.
        cross=IT.sym(randn(rng,d,d)); aa=1000.0; ga=0.2
        joint=IT.reduced_joint_direction(S,R,n,maps,h,cross,aa,ga)
        dense_joint=[full IT.svec(cross); IT.svec(cross)' aa]\
            vcat(-IT.svec(IT.gradient(S,R,n,maps,h)),-ga)
        @test joint.system_size==107
        @test joint.dS≈IT.smat(dense_joint[1:end-1],d) atol=1e-10 rtol=1e-10
        @test joint.da≈dense_joint[end] atol=1e-10 rtol=1e-10
        @test_throws DomainError IT.reduced_newton_direction(S,0.1R,n,maps,h)
        # Manufacture exact interior stationary points on a smooth path.
        # One current-problem Newton correction should square the small drift.
        drift=IT.sym(randn(rng,d,d)); drift./=opnorm(drift)
        errors=Float64[]
        for epsilon in (1e-2,5e-3,2.5e-3)
            target=S+epsilon*drift
            tm=IT.moments(target,maps); M=IT.smat(tm[1:end-1],m)
            inverse=inv(Symmetric(M)); z=inverse*h
            J=IT.adjoint_moments(vcat(IT.svec(inverse-z*z'),-m/tm[end]),maps)
            Rnew=IT.sym(n*target+target*J*target)
            @test norm(IT.gradient(target,Rnew,n,maps,h))<1e-9
            prediction=IT.reduced_newton_direction(S,Rnew,n,maps,h).delta
            push!(errors,norm(S+prediction-target))
        end
        @test 3.5<errors[1]/errors[2]<4.5
        @test 3.5<errors[2]/errors[3]<4.5
    end
    println("max original-gradient relative error=",worst_original,
        " finite-difference Hessian error=",worst_fd," reduced/dense Newton error=",worst_step)
end

@testset "Floor constraint: clipping a Riccati solution is not generally KKT" begin
    S0=Matrix(Diagonal([0.5,2.0])); K=[0.2 0.1;0.1 0.2]
    R=S0+S0*K*S0
    grad(S)=(inv(S)-inv(S)*R*inv(S)+K)/2
    @test norm(grad(S0))<1e-14
    clipped=Matrix(Diagonal([1.0,2.0]))
    @test isapprox(grad(clipped)[1,2],0.025;atol=1e-14,rtol=0)
    @test abs(grad(clipped)[1,2])>1e-6 # nonzero active/free rotation residual
    # Matrix projection differential, a building block for the true floor
    # KKT/semismooth system rather than an unproved clipping substitute.
    W=Matrix(qr(randn(rng,4,4)).Q)
    Z=W*Diagonal([0.2,0.7,1.8,3.0])*W'; H=IT.sym(randn(rng,4,4))
    projection(A)=let e=eigen(Symmetric(A)); e.vectors*Diagonal(max.(e.values,1.0))*e.vectors' end
    e=1e-5; fd=(projection(Z+e*H)-projection(Z-e*H))/(2e)
    @test IT.floor_projection_differential(Z,H,1.0)≈fd atol=1e-8 rtol=1e-7
    @test_throws DomainError IT.floor_projection_differential(Matrix{Float64}(I,2,2),ones(2,2),1.0)
end

@testset "Constructive output support: retain null prior and lift the floor face" begin
    N=7; observed=5; m=14; alpha=2.0; delta=0.01; n=130
    # delta is a toy algebra parameter, not a change to the production floor.
    # Support is constructed from unavailable coordinates, NOT spectral cutoff.
    Z=zeros(N,observed-1); Z[1:observed,:]=KTrader.relative_gauge(observed)
    r=size(Z,2); Q=KTrader.relative_gauge(N); Pa=Q*Q'-Z*Z'; a=N-1-r
    L=kron(Matrix{Float64}(I,m,m),Z)
    Xsmall=randn(rng,n,m*r); X=Xsmall*L'
    Y=2X[:,1:N]
    spectrum=KTrader.ridge_spectrum(X,X'*X,X'*Y;dual=false)
    cache=KTrader.conditioned_alpha_cache(spectrum,Y'*Y,n,alpha;gauge=Q)
    B=Matrix{Float64}(I,r,r); Sigma=Z*B*Z'+delta*Pa; S=Q'*Sigma*Q
    V=inv(Symmetric(X'*X+alpha*I)); G=Y'*X*V
    full=KTrader.condition_trace_neutrality(G,V,Sigma,N,7)
    M,h=KTrader.constraint_moments(G,KTrader.ridge_covariance(spectrum,alpha),Sigma,
                                  KTrader.get_constraint_columns(N,7))
    coefficients=M\h
    VF=inv(Symmetric(Xsmall'*Xsmall+alpha*I))
    Msmall=zeros(m,m)
    for j in 1:m, i in 1:m
        block=VF[(i-1)*r+1:i*r,(j-1)*r+1:j*r]
        Msmall[i,j]=tr(B*block)
    end
    Msmall .+= (a*delta/alpha).*Matrix{Float64}(I,m,m)
    @test M≈Msmall atol=1e-11 rtol=1e-11
    @test logdet(Symmetric(I+X'*X/alpha))≈logdet(Symmetric(I+Xsmall'*Xsmall/alpha)) atol=1e-10
    @test norm(Pa*Y')<1e-12
    @test norm(Pa*Sigma*Z)<1e-12
    nullnorm=0.0
    for c in 1:m
        cr=(c-1)*N+1:c*N
        expected=-(delta/alpha)*coefficients[c].*Pa
        @test Pa*full.G_c[:,cr]*Pa≈expected atol=1e-10 rtol=1e-10
        nullnorm=max(nullnorm,norm(expected))
    end
    @test nullnorm>1e-5 # dropping these "unused" coefficient directions is wrong
    Vc=inv(Symmetric(M)); z=Vc*h
    scalar=-n/(2delta)-tr(Vc-z*z')/(2alpha)+m/(2tr(Sigma))
    source=KTrader.conditioned_cached_state(cache,S;gradient=true)
    ambient_gradient=Q*source.grad*Q'
    @test norm(Pa*ambient_gradient*Z)<1e-9
    @test Pa*ambient_gradient*Pa≈scalar.*Pa atol=1e-8 rtol=1e-10
    @test scalar<0 # null block satisfies the one-sided maximum KKT condition
    # Full objective keeps the ORIGINAL output dimension N-1 in its
    # log-determinant coefficient, not only r observed output coordinates.
    R=Y'*Y-Y'*X*V*X'*Y
    reduced_value=-(N-1)/2*logdet(Symmetric(I+Xsmall'*Xsmall/alpha))-
        n/2*(logdet(Symmetric(B))+a*log(delta))-0.5tr(B\(Z'*R*Z))-
        0.5logdet(Symmetric(Msmall))-0.5dot(h,Msmall\h)+m/2*log((tr(B)+a*delta)/alpha)
    @test source.value≈reduced_value atol=1e-8 rtol=1e-10
    println("constructed support r=",r," null=",a," null mean norm=",nullnorm,
        "; full objective and prior retained")
end
println("THEORY IDENTITIES CHECKED; zero posterior fits/backtests; no production candidate accepted")
