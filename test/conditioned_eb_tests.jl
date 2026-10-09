using Test, Random, LinearAlgebra, Statistics
using Serialization
using KTrader

# 固定前处理；OOF 仅删除监督行，不重估设计坐标。
function conditioned_fixture(;seed=710,n=120,N=3)
    rng=MersenneTwister(seed); P=2length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); Pi=Q*Q'
    X=randn(rng,n,P)
    for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        X[:,c]=X[:,c]*Pi
    end
    A=zeros(N,P)
    A[:,1:N]=3.0Pi # 故意违反 trace prior，区分两种 evidence。
    Y=X*A'+0.25randn(rng,n,N)*Pi
    Bm=randn(rng,n+1,2length(KTrader.BANDS))
    ym=zeros(n+1); ym[2:end]=Bm[1:n,:]*fill(0.12,size(Bm,2))+0.4randn(rng,n)
    embedding=vcat(zeros(1,N),Y)
    (; X,Y,Q,Bm,ym,embedding,N,n)
end
function conditioned_fixture_fit(f;train=collect(1:f.n),uncertainty=true,initial=(1.0,1.0),fixed=nothing,macro_stats=nothing)
    X=f.X[train,:]; Y=f.Y[train,:]
    fit_response_operator(f.Bm,fill(Float64[],f.N),f.ym,f.embedding;
        ridge_alpha=fixed,ts=train,S_xx_rel=X'*X,S_xy_rel=X'*Y,S_yy_rel=Y'*Y,
        X_design=X,alpha_initial=initial,need_uncertainty=uncertainty,macro_stats)
end
function conditioned_fixture_spectrum(f,train=collect(1:f.n);dual=false)
    X=f.X[train,:]; Y=f.Y[train,:]
    KTrader.ridge_spectrum(X,X'*X,X'*Y;dual)
end

@testset "Support-conditioned EB production and certificates" begin
    f=conditioned_fixture()
    spectrum=conditioned_fixture_spectrum(f)
    solved=KTrader.optimize_conditioned_eb(spectrum,f.Y'*f.Y,f.n;gauge=f.Q,return_certificate=true)
    @test solved.certificate.valid
    @test solved.certificate.alpha.rel_res<=1e-6
    @test solved.certificate.covariance.projected_residual<=1e-6
    @test solved.certificate.covariance.support<=1e-10
    @test solved.certificate.covariance.min_eigenvalue>=1e-8-1e-14
    @test norm(solved.Sigma*ones(f.N))<=1e-10
    @test KTrader.conditioned_eb_certificate(spectrum,f.Y'*f.Y,f.n,solved.alpha,solved.Sigma;gauge=f.Q).valid

    @testset "生产 conditional evidence；uncertainty 路径均值相同" begin
        full=conditioned_fixture_fit(f)
        mean_only=conditioned_fixture_fit(f;uncertainty=false)
        @test full.alpha_rel ≈ solved.alpha rtol=1e-10
        @test full.Sigma_rel ≈ solved.Sigma rtol=1e-10
        @test full.alpha_macro==mean_only.alpha_macro
        @test full.alpha_rel==mean_only.alpha_rel
        @test full.G_macro==mean_only.G_macro
        @test full.G_c_mean==mean_only.G_c_mean
        @test full.Sigma_rel==mean_only.Sigma_rel
        @test isempty(mean_only.covariance.weights)
        @test mean_only.covariance.baseline==0
        @test full.trace_real ≈ 0.0 atol=1e-10
        @test full.trace_imag ≈ 0.0 atol=1e-10
        ordinary=optimize_matrix_normal_eb(spectrum.values,spectrum.B*f.Q,f.Q'*(f.Y'*f.Y)*f.Q,f.n)
        @test ordinary.certificate.valid
        @test !isapprox(log(ordinary.alpha),log(full.alpha_rel);atol=1e-3)
        fixed=conditioned_fixture_fit(f;fixed=2.7)
        @test fixed.alpha_macro==2.7
        @test fixed.alpha_rel==2.7
        fixed_mean=conditioned_fixture_fit(f;fixed=2.7,uncertainty=false)
        @test fixed_mean.G_c_mean==fixed.G_c_mean
    end

    @testset "OOF train-only 独立 alpha 与 optional macro Grams" begin
        train=setdiff(collect(1:f.n),31:60)
        fold=conditioned_fixture_fit(f;train,uncertainty=false)
        fs=conditioned_fixture_spectrum(f,train); Ytrain=f.Y[train,:]
        @test KTrader.conditioned_eb_certificate(fs,Ytrain'*Ytrain,length(train),fold.alpha_rel,fold.Sigma_rel;gauge=f.Q).valid
        Xm=f.Bm[train,:]; ym=f.ym[train .+ 1]
        ev=eigen(Symmetric(Xm'*Xm))
        macro_alpha=optimize_matrix_normal_eb(max.(ev.values,0.0),ev.vectors'*reshape(Xm'*ym,:,1),fill(dot(ym,ym),1,1),length(train))
        @test fold.alpha_macro ≈ macro_alpha.alpha rtol=1e-10
        stats=(;xx=Xm'*Xm,xy=reshape(Xm'*ym,:,1),yy=fill(dot(ym,ym),1,1),n=length(train))
        untouched=(copy(stats.xx),copy(stats.xy),copy(stats.yy))
        incremental=conditioned_fixture_fit(f;train,uncertainty=false,macro_stats=stats)
        @test stats.xx==untouched[1]
        @test stats.xy==untouched[2]
        @test stats.yy==untouched[3]
        @test incremental.G_macro ≈ fold.G_macro atol=1e-12
        @test incremental.post_cov_m ≈ fold.post_cov_m atol=1e-12
        @test incremental.alpha_macro==fold.alpha_macro
        @test_throws ArgumentError conditioned_fixture_fit(f;train,macro_stats=merge(stats,(;n=length(train)+1)))
        changed=merge(f,(;ym=copy(f.ym),Y=copy(f.Y),embedding=copy(f.embedding)))
        changed.ym[32:61].+=100.0
        changed.Y[31:60,:].+=20.0.*(f.Q[:,1]')
        changed.embedding[32:61,:].=changed.Y[31:60,:]
        other=conditioned_fixture_fit(changed;train,uncertainty=false)
        @test other.alpha_macro==fold.alpha_macro
        @test other.alpha_rel==fold.alpha_rel
        @test other.G_macro==fold.G_macro
        @test other.G_c_mean==fold.G_c_mean
        @test other.Sigma_rel==fold.Sigma_rel
    end

    @testset "warm/cold 同 covariance basin、目标和证书" begin
        warm=KTrader.optimize_conditioned_eb(spectrum,f.Y'*f.Y,f.n;gauge=f.Q,initial=solved.alpha,return_certificate=true)
        @test warm.certificate.valid
        @test warm.certificate.alpha.rel_res<=1e-6
        @test warm.certificate.covariance.projected_residual<=1e-6
        @test log(warm.alpha) ≈ log(solved.alpha) atol=5e-5
        @test warm.Sigma ≈ solved.Sigma rtol=3e-5
        @test warm.certificate.evidence ≈ solved.certificate.evidence atol=1e-7
        lambdas=[0.0,0.1,1.0,10.0]; B=reshape([0.0,0.3,0.8,2.0],:,1); yy=fill(8.0,1,1)
        cold=optimize_matrix_normal_eb(lambdas,B,yy,50)
        mwarm=optimize_matrix_normal_eb(lambdas,B,yy,50;initial=cold.alpha)
        @test mwarm.certificate.valid
        @test mwarm.alpha ≈ cold.alpha rtol=1e-7
    end

    @testset "失败明确：边界、floor、support、预算" begin
        @test KTrader.bounded_alpha_certificate(1e-4,-1.0,1.0).valid
        @test !KTrader.bounded_alpha_certificate(1e-4,1.0,1.0).valid
        @test KTrader.bounded_alpha_certificate(1e6,1.0,1.0).valid
        @test !KTrader.bounded_alpha_certificate(1e6,-1.0,1.0).valid
        @test !KTrader.bounded_alpha_certificate(1.0,1.0,1.0).valid
        @test !KTrader.covariance_certificate(0.5e-8Matrix{Float64}(I,2,2),zeros(2,2)).valid
        @test !KTrader.covariance_certificate(Matrix{Float64}(I,2,2),Matrix{Float64}(I,2,2)).valid
        @test !KTrader.covariance_certificate(solved.Sigma+0.1ones(f.N,f.N),zeros(f.N,f.N);gauge=f.Q).valid
        @test_throws ErrorException KTrader.optimize_conditioned_eb(spectrum,f.Y'*f.Y,f.n;gauge=f.Q,iters=0)
        @test_throws ErrorException KTrader.optimize_conditioned_eb(spectrum,f.Y'*f.Y,f.n;gauge=f.Q,max_backtracks=0)
        @test_throws ErrorException optimize_matrix_normal_eb([1.0],fill(1.0,1,1),fill(2.0,1,1),10;iters=0)
        @test_throws ErrorException KTrader.maximize_logalpha(x->-(x-0.17)^2,x->-2(x-0.17);maxiters=0)
        @test_throws ErrorException KTrader.maximize_logalpha(x->NaN,x->0.0)
        floored=optimize_matrix_normal_eb([1.0],zeros(1,1),zeros(1,1),10)
        @test floored.certificate.valid
        @test floored.Sigma[1,1]==1e-8
        @test floored.alpha==1e6
        N=2; P=2length(KTrader.BANDS)*N; Q=KTrader.relative_gauge(N)
        zero_spectrum=KTrader.ridge_spectrum(zeros(20,P),zeros(P,P),zeros(P,N);dual=false)
        zero_fit=KTrader.optimize_conditioned_eb(zero_spectrum,zeros(N,N),20;gauge=Q,return_certificate=true)
        @test zero_fit.certificate.valid
        @test Q'*zero_fit.Sigma*Q ≈ fill(1e-8,1,1) atol=1e-20
    end

    @testset "单资产 relative support 恰为空" begin
        one=conditioned_fixture(N=1)
        fit=conditioned_fixture_fit(one)
        @test fit.Sigma_rel==zeros(1,1)
        @test iszero(norm(fit.G_c_mean))
        @test fit.alpha_rel==1.0
    end
end

@testset "Dense Gaussian oracle and fixed-alpha cached covariance rays" begin
    rng=MersenneTwister(721); n=18; N=3; P=2length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); d=N-1
    X=randn(rng,n,P); Yc=randn(rng,n,d); Y=Yc*Q'
    S=[0.8 0.12;0.12 0.6]; Sigma=Q*S*Q'
    primal=KTrader.ridge_spectrum(X,X'*X,X'*Y;dual=false)
    dual=KTrader.ridge_spectrum(X,X'*X,X'*Y;dual=true)
    cols=KTrader.get_constraint_columns(N,length(KTrader.BANDS)); C=zeros(length(cols),N*P)
    for c in eachindex(cols),j in 1:N
        C[c,j+(cols[c][j]-1)*N]=1.0
    end
    Cc=C*kron(Matrix{Float64}(I,P,P),Q)
    observation=kron(X,Matrix{Float64}(I,d,d)); target=vec(Yc')
    function oracle(alpha,S)
        prior=kron(Matrix{Float64}(I,P,P)./alpha,S)
        constrained=prior-prior*Cc'*((Cc*prior*Cc')\(Cc*prior))
        cov=kron(Matrix{Float64}(I,n,n),S)+observation*constrained*observation'
        chol=cholesky(Symmetric(cov))
        -0.5logdet(chol)-0.5dot(target,chol\target)
    end
    a1,a2=0.4,9.0
    optimized=KTrader.optimize_conditioned_eb(dual,Y'*Y,n;gauge=Q,return_certificate=true)
    @test optimized.certificate.valid
    fitted_S=Q'*optimized.Sigma*Q
    @test KTrader.conditioned_evidence(dual,Y'*Y,n,a1,optimized.Sigma;gauge=Q)-
          KTrader.conditioned_evidence(dual,Y'*Y,n,a2,optimized.Sigma;gauge=Q) ≈
          oracle(a1,fitted_S)-oracle(a2,fitted_S) atol=1e-8
    for spectrum in (primal,dual)
        cache=KTrader.conditioned_evidence_cache(spectrum,Y'*Y,n,Sigma;gauge=Q)
        @test cache.evidence(log(a1))-cache.evidence(log(a2)) ≈ oracle(a1,S)-oracle(a2,S) atol=1e-9
        logstep=1e-5
        @test cache.derivative(log(a1)) ≈ (cache.evidence(log(a1)+logstep)-cache.evidence(log(a1)-logstep))/(2logstep) atol=1e-7
        fixed=KTrader.conditioned_alpha_cache(spectrum,Y'*Y,n,a1;gauge=Q)
        D=[0.06 0.02;0.02 -0.03]; ray=KTrader.conditioned_covariance_ray(fixed,S,D)
        base=KTrader.conditioned_cached_state(fixed,S;gradient=true)
        for t in (0.0,0.5,1.0)
            @test ray(t) ≈ KTrader.conditioned_evidence(spectrum,Y'*Y,n,a1,Q*(S+t*D)*Q';gauge=Q) atol=1e-9
            @test ray(t)-ray(0.0) ≈ oracle(a1,S+t*D)-oracle(a1,S) atol=1e-9
        end
        step=1e-5
        @test dot(base.grad,D) ≈ (ray(step)-ray(-step))/(2step) atol=1e-7
        V=inv(Symmetric(X'*X+a1*I)); G=(V*(X'*Y))'
        Omega=kron(V,Sigma); M=C*Omega*C'
        dense=reshape(vec(G)-Omega*C'*(M\(C*vec(G))),N,P)
        # 新契约的 alpha cache 刻意不含未约束均值 G（deadwork 测试以
        # !hasproperty(c,:G) 锁定该契约）；按生产同式从 spectrum 重建
        # G = Bd'*basis'，Bd = B .* d，数值等于旧 cache.G，断言与容差不变。
        G_ref=((spectrum.B .* (1.0 ./ (spectrum.values .+ a1)))')*spectrum.basis'
        exact=condition_trace_neutrality(G_ref,fixed.V,Sigma,N,length(KTrader.BANDS))
        @test exact.G_c ≈ dense atol=1e-10
        @test exact.inv_M ≈ inv(Symmetric(M)) atol=1e-10
    end
    O=[0.6 -0.8;0.8 0.6]
    @test KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma;gauge=Q) ≈
          KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma;gauge=Q*O) atol=1e-10
end

@testset "Floor 激活的边界 KKT 驻点与平坦 evidence 锚定" begin
    # eigen 升序：diagm([1.0,floor]) 的 λ=[floor,1.0]，激活方向是 e2（D 的 [2,2]）。
    S=diagm([1.0,KTrader.EB_COVARIANCE_FLOOR])
    # 合法边界驻点：evidence 想减小被 floor 钉住的方向（单边 KKT 允许）。
    # 旧 interior 判据在真实全历史（14310×65）上把这类状态误判为未驻点并抛
    # "no certified ascent direction"（白化按 1/λ≈1/floor 放大剪裁位移）。
    legal=KTrader.covariance_certificate(S,diagm([0.0,-1e-13]))
    @test legal.valid
    @test legal.active_directions==1
    # 破坏 KKT 符号：evidence 想增大被 floor 钉住的方向，不是合法驻点。
    # 旧 interior 判据只看剪后位移（此处为零），会误放行；新判据必须红。
    @test !KTrader.covariance_certificate(S,diagm([0.0,1e-13])).valid
    # 自由方向仍受原白化度量约束：判据修正不是放宽。
    @test !KTrader.covariance_certificate(S,diagm([2e-6,0.0])).valid
    @test KTrader.covariance_certificate(S,diagm([5e-7,-1e-13])).valid
    # 全激活（evidence 想整体缩小被 floor 挡住）：合法边界驻点。
    @test KTrader.covariance_certificate(KTrader.EB_COVARIANCE_FLOOR*Matrix{Float64}(I,2,2),
                                         -1e-13*Matrix{Float64}(I,2,2)).valid

    # 自由×激活交叉块：KKT 乘子 M 的支撑在激活子空间，交叉梯度必须为零。
    # 上面所有用例的 D 都是对角阵（交叉恒零），把 cross_max 判据删成恒 0 不会
    # 让它们变红——这两条专门锁定交叉判据的变异拒绝与白化语义：交叉位移按
    # 1/√(λ_free·λ_active)≈1e4 白化，1e-9 的交叉位移放大成 1e-5 > tol 必须红；
    # 若判据退化成裸 DV（不白化），1e-9 ≤ tol 会放行该非法点，第一条即失败。
    # 同形状的 1e-13 交叉白化后 1e-9 ≤ tol，是合法驻点，不得误杀。
    @test !KTrader.covariance_certificate(S,[0.0 1e-9; 1e-9 0.0]).valid
    @test KTrader.covariance_certificate(S,[0.0 1e-13; 1e-13 0.0]).valid

    # 端到端：退化方向能量低于 floor 的 Y，从 optimize_conditioned_eb 入口
    # 返回合法边界证书而非抛错（全历史回测解锁路径）。
    rng=MersenneTwister(731); n=18; N=3; P=2length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); d=N-1
    X=randn(rng,n,P)
    for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        X[:,c]=X[:,c]*(Q*Q')
    end
    Yc=randn(rng,n,d).*[1.0 1e-12]  # 第二方向方差远低于 floor
    Y=Yc*Q'
    spectrum=KTrader.ridge_spectrum(X,X'*X,X'Y)  # n<P ⟹ dual，baseline 保证 M SPD
    solved=KTrader.optimize_conditioned_eb(spectrum,Y'*Y,n;gauge=Q,return_certificate=true)
    @test solved.certificate.valid
    @test eigmin(Symmetric(Q'*solved.Sigma*Q))<=KTrader.EB_COVARIANCE_FLOOR*(1+1e-6)
    @test eigmin(Symmetric(Q'*solved.Sigma*Q))>=KTrader.EB_COVARIANCE_FLOOR-1e-15

    # 平坦 evidence 的确定性锚定：不依赖 warm start（增量/batch 分歧根因）。
    flat=x->0.0
    a_warm=KTrader.maximize_logalpha(flat,x->0.0;initial=1e-3)
    a_cold=KTrader.maximize_logalpha(flat,x->0.0;initial=1e5)
    @test a_warm==a_cold
    @test a_warm≈10.0 rtol=1e-12  # log 网格几何中心 exp((log 1e-4+log 1e6)/2)
end

@testset "RMS 等价、驻点方程求解器与浅平坦 tie-break" begin
    # [1] sqrt(d) RMS 等价：修正分母后全自由方向的判据数值与旧 Cholesky
    # RMS 路径一致；norm(W)∈(tol·sqrt(d), tol·d) 的点必须 invalid
    # （上一版分母 length(free) 在此区间误放行——DevOps 数值实验证伪点）。
    rng=MersenneTwister(741)
    for _ in 1:5
        A=randn(rng,3,3); S0=A*A'+0.5I
        E=randn(rng,3,3); D=1e-3.*(E+E')/2
        root=cholesky(Symmetric(S0)).L
        old_rms=norm(root\(D)/root')/sqrt(3)  # 旧判据精确复现（可行时 projected-S=D）
        cert=KTrader.covariance_certificate(S0,D)
        ev=eigen(Symmetric(S0)); λ=ev.values; V=ev.vectors
        invsqrt=1.0./sqrt.(λ)
        Wnew=(V'*D*V).*(invsqrt*invsqrt')
        @test norm(Wnew)/sqrt(3) ≈ old_rms rtol=1e-10  # 数值对照：比值=1
        @test cert.valid == (norm(Wnew)/sqrt(3)<=1e-6)
    end
    # 区间点：||W||_F=2e-6 ∈ (tol·sqrt(3), tol·3)，旧 RMS 拒绝；分母错误版会放行。
    S0=Matrix{Float64}(I,3,3); D=(2e-6/sqrt(3))*Matrix{Float64}(I,3,3)
    @test norm(D)≈2e-6 atol=1e-18
    @test !KTrader.covariance_certificate(S0,D).valid

    # [2] 驻点方程求解器（选项 A）：fixture n=90（此前 200 轮不收敛）与
    # 多激活方向退化 fixture 都应收敛并给出 valid 证书。
    f=conditioned_fixture()
    train=setdiff(collect(1:f.n),31:60)
    fs=conditioned_fixture_spectrum(f,train); Ytrain=f.Y[train,:]
    solved90=KTrader.optimize_conditioned_eb(fs,Ytrain'*Ytrain,length(train);
        gauge=f.Q,return_certificate=true)
    @test solved90.certificate.valid
    # 多激活：N=4（d=3），两个方向能量低于 floor。
    rng2=MersenneTwister(742); n2=18; N4=4; P4=2length(KTrader.BANDS)*N4
    Q4=KTrader.relative_gauge(N4); d4=N4-1
    X4=randn(rng2,n2,P4)
    for c in KTrader.get_constraint_columns(N4,length(KTrader.BANDS))
        X4[:,c]=X4[:,c]*(Q4*Q4')
    end
    Yc4=randn(rng2,n2,d4).*[1.0 1e-12 1e-12]
    Y4=Yc4*Q4'
    sp4=KTrader.ridge_spectrum(X4,X4'*X4,X4'Y4)
    multi=KTrader.optimize_conditioned_eb(sp4,Y4'*Y4,n2;gauge=Q4,return_certificate=true)
    @test multi.certificate.valid
    @test multi.certificate.covariance.active_directions==2
    compact4=Q4'*multi.Sigma*Q4
    @test count(<=(KTrader.EB_COVARIANCE_FLOOR*1.000001),eigvals(Symmetric(compact4)))==2
    # 全历史形状（19 激活方向 + 45 自由）用 N=20 的合成近似复现。
    rng3=MersenneTwister(743); n3=60; N20=20; P20=2length(KTrader.BANDS)*N20
    Q20=KTrader.relative_gauge(N20); d20=N20-1
    X20=randn(rng3,n3,P20)
    for c in KTrader.get_constraint_columns(N20,length(KTrader.BANDS))
        X20[:,c]=X20[:,c]*(Q20*Q20')
    end
    scale20=vcat(fill(1.0,5),fill(1e-12,14))  # 14 激活 + 5 自由（N=20 合成近似全历史多激活形状）
    Yc20=randn(rng3,n3,d20).*scale20'
    Y20=Yc20*Q20'
    sp20=KTrader.ridge_spectrum(X20,X20'*X20,X20'Y20)
    hist=KTrader.optimize_conditioned_eb(sp20,Y20'Y20,n3;gauge=Q20,return_certificate=true)
    @test hist.certificate.valid

    # [2b] 不动点残差单调守护：残差单调+阻尼版 sigma_stationary_fixed_point
    # 返回点的方程残差不超过起点（t=14294 振荡教训的回归锁定）。
    fp_cache=KTrader.conditioned_alpha_cache(fs,f.Y'*f.Y,f.n,1.0;gauge=f.Q)
    fp_residual(cache,X)=begin
        st=KTrader.conditioned_cached_state(cache,X;gradient=true)
        nc=length(cache.h)
        norm((cache.R .+ (nc/tr(X)).*(X*X) .- X*st.J*X)./cache.n .- X)
    end
    S_start=KTrader.positive_covariance(fp_cache.R./f.n)
    S_fp2=KTrader.sigma_stationary_fixed_point(fp_cache,S_start)
    @test fp_residual(fp_cache,S_fp2)<=fp_residual(fp_cache,S_start)+1e-12

    # [3] 浅平坦 tie-break（scale-aware 阈值 256·eps·|L|）：|L|~1e4 时
    # τ=256·eps(Float64)·1e4=5.68e-10（机器精度×尺度；上一版误用 eps(x) 的
    # ULP 语义，|L|≈1e4 处 τ=4.66e-10 与 gap 4.65e-10 贴边，已修）。
    # 双峰差 2e-10 < τ 锚定（远离阈值边界，避免 flaky）；差 1e-8 > τ 保留。
    # 该量级正是实测 alpha_rel 分歧 gap，tie-break 对它同样不触发（E 项根因）。
    dual(δ)=x->1e4+max(-(x-2.0)^2,δ-(x-5.0)^2)
    dd=x->x<3.5 ? -2(x-2.0) : -2(x-5.0)
    a_w=KTrader.maximize_logalpha(dual(2e-10),dd;initial=exp(5.0))
    a_c=KTrader.maximize_logalpha(dual(2e-10),dd;initial=exp(2.0))
    @test a_w ≈ a_c atol=1e-6
    @test a_w ≈ exp(2.0) atol=1e-6  # 锚定到最接近几何中心的候选（峰1）
    # 实测分歧量级 4.65e-10（|L|=1e4 处舍入为 256 ulp）：新阈值
    # 256·eps(Float64)·|L|≈5.68e-10 将其盖住（锚定）；旧 ULP 语义
    # 256·eps(1e4)=4.66e-10 恰好盖不住（舍入后相等，严格小于不成立）。
    a_mid=KTrader.maximize_logalpha(dual(4.65e-10),dd;initial=exp(5.0))
    @test a_mid ≈ exp(2.0) atol=1e-6
    a_true=KTrader.maximize_logalpha(dual(1e-8),dd;initial=exp(2.0))
    @test a_true ≈ exp(5.0) atol=1e-6  # 真高 1e-8 的峰不被锚定
end

@testset "Riccati 闭式候选：方程恒等式、合法域、退化结构与 t=14294 回归" begin
    # [R1a] 恒等式（合法域：K PSD 且与 Q 不对齐；Q SPD 非对角）：S⁺+S⁺KS⁺=Q
    # 精确到机器精度，且 S⁺ 正定。合法性证明：K⪰0 ⟹ B̃=√QK√Q⪰0（Sylvester
    # 惯性/合同保半正定）⟹ 1+4λ(B̃)≥1>0，域门禁恒过；K 与 Q 独立随机构造
    # 几乎必然不对齐，恒等式在非对易下仍精确——把 g(b)=2/(1+√(1+4b)) 公式
    # 改错（如去掉 4 倍因子）必红。SPD Q 且不对齐本身不是合法充分条件。
    rng=MersenneTwister(751)
    A=randn(rng,4,4); Q=A*A'
    Bm=randn(rng,4,4); K=Bm*Bm'+0.5I
    τ=2.3; ncR=3; nR=17
    J=K.*nR .+ (ncR/τ)*Matrix{Float64}(I,4,4)
    evQ=eigen(Symmetric(Q))
    Qroot=(evQ.vectors .* sqrt.(max.(evQ.values,0.0))') * evQ.vectors'
    @test norm(K*Q-Q*K)>1e-6  # 非对齐非平凡
    S=KTrader.riccati_stationary_candidate(Qroot,J,τ,ncR,nR)
    @test S!==nothing
    @test norm(S+S*K*S-Q)<=1e-12*norm(Q)
    @test minimum(eigvals(Symmetric(S)))>0
    # [R1b] 非法域拒绝（原 R1 输入保留为拒绝反例，非逃绿——实测 seed 751
    # 下 candidate=nothing）：随机不定对称 K 经 B̃=√QK√Q 保持惯性
    # （Sylvester），负特征值使 1+4λ_min(B̃)<0，域门禁必须拒绝。域判据：
    # 合法 ⟺ λ_min(√QK√Q)≥−1/4；K⪰0 是充分条件，K 不定则依谱而定。
    Ku=(Bm+Bm')/2  # 原 R1 的 Km：seed 751 第二次 randn(4,4) 即 Bm，精确复原实测红的失败输入
    Ju=Ku.*nR .+ (ncR/τ)*Matrix{Float64}(I,4,4)
    @test minimum(eigvals(Symmetric(Qroot*(Ku*Qroot))))<-0.25  # 越域确证
    @test KTrader.riccati_stationary_candidate(Qroot,Ju,τ,ncR,nR)===nothing
    # [R2] 非法域：I+4√QK√Q 非半正定 → 不产生候选（不是错误、不是放行）。
    Qd=diagm([2.0,1.0]); evQd=eigen(Symmetric(Qd))
    Qrootd=(evQd.vectors .* sqrt.(max.(evQd.values,0.0))') * evQd.vectors'
    Jbad=(-1.0).*Matrix{Float64}(I,2,2).*nR  # nc=0 → K=-I，B̃=-Qd，1-8<0
    @test KTrader.riccati_stationary_candidate(Qrootd,Jbad,τ,0,nR)===nothing
    # [R3] 退化（奇异）Q：零方向 S⁺→0（该方向方程 0+0=0 精确），非零方向
    # 满足方程；floor clamp 是调用方 positive_covariance 的职责，不在候选内。
    Q0=diagm([1.0,0.0]); evQ0=eigen(Symmetric(Q0))
    Qroot0=(evQ0.vectors .* sqrt.(max.(evQ0.values,0.0))') * evQ0.vectors'
    S0=KTrader.riccati_stationary_candidate(Qroot0,diagm([0.5,0.7]).*nR,τ,0,nR)
    @test S0!==nothing
    @test S0[2,2]==0.0
    @test abs(S0[1,1]+S0[1,1]^2*0.5-1.0)<=1e-12
    # [R4] nc=0、K=0：W=I → S⁺=Q（R/n 一步解的闭式等价）。
    Szero=KTrader.riccati_stationary_candidate(Qroot,zeros(4,4),τ,0,nR)
    @test Szero!==nothing
    @test norm(Szero-Q)<=1e-12*norm(Q)
    # [R5] 根因锁定（纯数学事实，不依赖实现）：冻结迭代 X⁺=Q−X·K·X 从
    # 明确的非解初值一步越出正定锥——K 大时不动点本质发散，阻尼只能
    # 镇定不能恢复收敛率。真解是不动点（从解出发不动），此处用非解初值。
    Kbig=1e3*Matrix{Float64}(I,2,2); X0=0.1*Matrix{Float64}(I,2,2)
    X1=Matrix{Float64}(I,2,2)-X0*Kbig*X0
    @test eigmax(Symmetric(X1))<0

    # [R6] t=14294 真实切片（DevOps 固化的不可变 fixture，fold 3，n=9358）：
    # 单线程 BLAS 下旧阻尼不动点 100% 复现 free_rms=1.4097897e-5 的
    # fail-closed；多线程求和顺序差异会把该浅盆地翻绿——复现契约固定
    # 单线程（保存并 finally 恢复 BLAS 线程数）。本回归要求求解器真实
    # 收敛到门禁内：valid 证书 + free_rms≤原 tol；证书/容差/预算全部
    # 原值，不是放水。若 Riccati 候选不能解决该切片，此测试保持红——
    # 诚实暴露，不得以改证书遮掩。
    saved=BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        c=deserialize(joinpath(@__DIR__,"fixtures","t14294_conditioned_fold3.jls"))
        solved=KTrader.optimize_conditioned_eb(c.spectrum,c.YtY,c.n;
            initial=c.initial,tol=c.tol,gauge=c.gauge,iters=c.iters,
            max_backtracks=c.max_backtracks,alpha_iters=c.alpha_iters,
            return_certificate=true,fp_iters=c.fp_iters,fp_tol=c.fp_tol)
        @test solved.certificate.valid
        @test solved.certificate.covariance.free_rms<=c.tol
    finally
        BLAS.set_num_threads(saved)
    end
end

@testset "t14294 actual-BLAS6 single-solve companion (no thread switch)" begin
    # 原 R6 testset 内部 set_num_threads(1)：两启动环境都只证明 BLAS1
    # 求解路径。本 companion 全程不切线程，验证实际 BLAS6 路径下同一
    # fixture、原 residual/cert 门禁（单次，非 repeats）。BLAS<6 的环境
    # 跳过（无多线程 BLAS 可验证时 not run，不假绿）。
    if BLAS.get_num_threads() >= 6
        c = deserialize(joinpath(@__DIR__, "fixtures", "t14294_conditioned_fold3.jls"))
        solved = KTrader.optimize_conditioned_eb(c.spectrum, c.YtY, c.n;
            initial = c.initial, tol = c.tol, gauge = c.gauge, iters = c.iters,
            max_backtracks = c.max_backtracks, alpha_iters = c.alpha_iters,
            return_certificate = true, fp_iters = c.fp_iters, fp_tol = c.fp_tol)
        @test BLAS.get_num_threads() >= 6          # 全程未切换
        @test solved.certificate.covariance.valid
        @test solved.certificate.covariance.free_rms <= c.tol   # 原门禁原值
        @test solved.certificate.alpha.valid
    else
        @test_skip "BLAS threads < 6: actual-BLAS6 path not verifiable here"
    end
end
