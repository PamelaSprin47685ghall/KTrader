using Test, Random, LinearAlgebra, Statistics
using KTrader

# Pre-hoist scalar-cache formula, kept test-only. In particular, this does
# not call the new geometry/cache helpers to produce its own expectations.
function reuse_scalar_reference(spectrum,YtY,n,Sigma;gauge=nothing)
    cols=KTrader.get_constraint_columns(size(Sigma,1),length(KTrader.BANDS))
    nc=length(cols); rank=length(spectrum.values)
    S=gauge===nothing ? Sigma : gauge'*Sigma*gauge
    B=gauge===nothing ? spectrum.B : spectrum.B*gauge
    R=gauge===nothing ? YtY : gauge'*YtY*gauge
    chol=cholesky(Symmetric(S)); solved=chol\B'
    quadratic=[dot(view(B,k,:),view(solved,:,k)) for k in 1:rank]
    blocks=[gauge===nothing ? Matrix(view(spectrum.basis,c,:)) : gauge'*view(spectrum.basis,c,:) for c in cols]
    tensors=zeros(rank,nc,nc); hcoef=zeros(rank,nc)
    for r in 1:nc
        Ur=blocks[r]; transformed=S*Ur
        for k in 1:rank
            hcoef[k,r]=dot(view(B,k,:),view(Ur,:,k))
        end
        for s in r:nc,k in 1:rank
            tensors[k,r,s]=tensors[k,s,r]=dot(view(blocks[s],:,k),view(transformed,:,k))
        end
    end
    constant=-n/2*logdet(chol)-0.5tr(chol\R)
    function moments(logalpha)
        alpha=exp(logalpha); d=1.0 ./ (spectrum.values .+ alpha)
        baseline=spectrum.dual ? 1/alpha : 0.0
        weights=spectrum.dual ? -spectrum.values .* d ./ alpha : d
        M=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            M[r,s]=M[s,r]=dot(weights,view(tensors,:,r,s))
        end
        for r in 1:nc
            M[r,r]+=baseline*tr(S)
        end
        (; alpha,d,baseline,M,h=hcoef'*d)
    end
    function evidence(a)
        state=moments(a); cholM=cholesky(Symmetric(state.M))
        constant-size(S,1)/2*sum(log1p.(spectrum.values./state.alpha))+0.5dot(state.d,quadratic)-
            0.5logdet(cholM)-0.5dot(state.h,cholM\state.h)+nc/2*log(tr(S)/state.alpha)
    end
    function derivative(a)
        state=moments(a); alpha=state.alpha; d=state.d
        cholM=cholesky(Symmetric(state.M)); z=cholM\state.h
        dd=-alpha.*d.^2
        wd=spectrum.dual ? spectrum.values .* d .* (d .+ 1/alpha) : dd
        Md=zeros(nc,nc)
        for r in 1:nc,s in r:nc
            Md[r,s]=Md[s,r]=dot(wd,view(tensors,:,r,s))
        end
        for r in 1:nc
            Md[r,r]-=state.baseline*tr(S)
        end
        size(S,1)/2*sum(spectrum.values.*d)+0.5dot(dd,quadratic)-
            0.5tr(cholM\Md)+0.5dot(z,Md*z)-dot(z,hcoef'*dd)-nc/2
    end
    (; evidence,derivative)
end

# Old allocation pattern, with the same scalar arithmetic. The new function
# must retain immutable Dict keys when its single scratch mask changes.
function reuse_macro_reference(o)
    N=o.N; e=zeros(o.n_res)
    hcache=Dict{Tuple{Int,Vector{Bool}},Vector{Float64}}()
    onesN=ones(N); x=zeros(2length(KTrader.BANDS)*N)
    for idx in 1:o.n_res
        t=o.ts_total[idx]; rt=view(o.r_frozen,t+1,:)
        mask=Vector{Bool}(undef,N); c=0; sumr=0.0
        for j in 1:N
            v=rt[j]; mask[j]=isfinite(v)
            if mask[j]
                c+=1; sumr+=v
            end
        end
        c==0 && continue
        sw=0.0
        for j in 1:N
            mask[j] && (sw+=o.s1[j])
        end
        f=KTrader.fold_of(o,idx); model=o.fold_models[f]
        h=get!(hcache,(f,mask)) do
            w=zeros(N)
            for j in 1:N
                mask[j] && (w[j]=o.s1[j])
            end
            model.G_c_mean'*(w .- onesN.*(sw/N))
        end
        KTrader.design_row!(x,o,t)
        mu_macro=dot(view(o.Bm_frozen,t,:),model.G_macro)
        e[idx]=sumr/sqrt(c)-mu_macro*sw/c-dot(h,x)/sqrt(c)
    end
    e
end

function reuse_residual_fixture(;nrows=150,N=5)
    rng=MersenneTwister(8164); C=2length(KTrader.BANDS); P=C*N
    ts=collect(KTrader.WARMUP:KTrader.WARMUP+nrows-1); T=last(ts)+1
    ranges=[1:div(nrows,3),div(nrows,3)+1:2div(nrows,3),2div(nrows,3)+1:nrows]
    r=0.01randn(rng,T,N)
    # A -> B -> A -> empty -> all, recurring within and across folds.
    # A permanently mutable stored key or a missing fold key must fail.
    for (idx,t) in enumerate(ts), j in 1:N
        observed=mod(idx,5)==0 || (mod(idx,5) in (1,3) ? isodd(j) : mod(idx,5)==2 && iseven(j))
        observed || (r[t+1,j]=NaN)
    end
    covariance=KTrader.RidgeCovariance(zeros(P,0),Float64[],0.0)
    models=[KTrader.ResponseOperator(0.01randn(rng,C),zeros(C,C),
        0.01randn(rng,N,P),covariance,zeros(N,N),zeros(C,C),
        KTrader.get_constraint_columns(N,length(KTrader.BANDS)),1.0,1.0,0.0,0.0) for _ in ranges]
    KTrader.ResidualOracle(models,ranges,ts,r,0.1 .+ rand(rng,N),
        0.2 .+ rand(rng,length(KTrader.BANDS)),randn(rng,T,C),cumsum(randn(rng,T,N);dims=1))
end

@testset "Fit-local scalar geometry: reuse changes no evidence or derivative" begin
    rng=MersenneTwister(8163)
    for (N,n,dual) in ((3,70,false),(3,20,true)), gauged in (false,true)
        P=2length(KTrader.BANDS)*N
        gauge=gauged ? KTrader.relative_gauge(N) : nothing
        X=randn(rng,n,P); Y=randn(rng,n,N)
        gauged && (Y=Y*gauge*gauge')
        spectrum=KTrader.ridge_spectrum(X,X'X,X'Y;dual)
        YtY=Y'Y
        geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
        frozen=deepcopy(geometry)
        d=gauged ? N-1 : N
        for scale in (0.1,1.0,7.0)
            S=Matrix(Diagonal(collect(range(scale,2scale;length=d))))
            Sigma=gauged ? gauge*S*gauge' : S
            reused=KTrader._conditioned_scalar_cache(spectrum,n,Sigma,geometry;gauge)
            fresh=KTrader.conditioned_evidence_cache(spectrum,YtY,n,Sigma;gauge)
            reference=reuse_scalar_reference(spectrum,YtY,n,Sigma;gauge)
            for alpha in (0.01,0.7,11.0,1e4)
                a=log(alpha)
                @test reused.evidence(a) ≈ reference.evidence(a) atol=1e-11 rtol=1e-12
                @test reused.derivative(a) ≈ reference.derivative(a) atol=1e-11 rtol=1e-12
                @test reused.evidence(a) == fresh.evidence(a)
                @test reused.derivative(a) == fresh.derivative(a)
            end
            a=log(0.7); delta=1e-5
            fd=(reference.evidence(a+delta)-reference.evidence(a-delta))/(2delta)
            @test reused.derivative(a) ≈ fd atol=1e-6 rtol=1e-6
            # Old closures must remain valid after another Sigma cache is built.
            before=reused.evidence(a)
            other=KTrader._conditioned_scalar_cache(spectrum,n,2Sigma,geometry;gauge)
            @test !isapprox(other.evidence(a),before;atol=1e-8,rtol=1e-10)
            @test reused.evidence(a) == before
        end
        @test isequal(geometry,frozen)
        bad=-(gauged ? gauge*gauge' : Matrix{Float64}(I,N,N))
        @test_throws PosDefException KTrader._conditioned_scalar_cache(spectrum,n,bad,geometry;gauge)
        # Separate target/fold data always get new geometry; no process cache.
        target2=2YtY
        g2=KTrader._conditioned_scalar_geometry(spectrum,target2;gauge)
        @test !isequal(g2.R,geometry.R)
        @test isequal(geometry,frozen)
    end
end

@testset "Residual mask scratch: recurrent masks, empty rows and independent calls" begin
    oracle=reuse_residual_fixture()
    frozen=copy(oracle.r_frozen)
    expected=reuse_macro_reference(oracle)
    actual=KTrader.macro_residual_series(oracle)
    @test isequal(actual,expected)
    @test isequal(KTrader.macro_residual_series(oracle),expected)
    @test isequal(oracle.r_frozen,frozen)
    # Independent row-level prediction, not the optimized scalar contraction.
    dense=[let row=KTrader.residual_row(oracle,i), vals=filter(isfinite,row)
        isempty(vals) ? 0.0 : sum(vals)/sqrt(length(vals))
    end for i in 1:oracle.n_res]
    @test actual ≈ dense atol=1e-10 rtol=1e-8
    @test all(iszero,actual[4:5:end])
    # A subsequent call sees changed inputs, rather than an escaped h cache.
    oracle.fold_models[1].G_c_mean[1,1]+=0.25
    changed=KTrader.macro_residual_series(oracle)
    @test isequal(changed,reuse_macro_reference(oracle))
    @test !isequal(changed,actual)
end

@testset "Inference panel packing preserves values and ownership" begin
    bars=[[1.0,NaN,-0.0],[2.0,Inf,3.0],[4.0,-Inf,5.0]]
    old=reduce(vcat,permutedims.(bars))
    packed=KTrader.inference_prices((;bars))
    @test isequal(packed,old)
    @test size(packed)==(3,3)
    packed[1,1]=7.0
    @test bars[1][1]==1.0
    @test isequal(KTrader.inference_prices((;bars=[bars[1]])),permutedims(bars[1]))
    @test_throws ArgumentError KTrader.inference_prices((;bars=Vector{Float64}[]))
    @test_throws DimensionMismatch KTrader.inference_prices((;bars=[[1.0],[2.0,3.0]]))
end
