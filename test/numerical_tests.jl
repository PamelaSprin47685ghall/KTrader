@testset "Exact numerical kernels" begin
    rng=MersenneTwister(901)
    @testset "Ragged field support is explicit, not a price observation" begin
        r=[1.0 NaN 3.0;NaN NaN NaN;4.0 -2.0 NaN]
        field=KTrader.embedded_relative_field(r,[1.0,2.0,1.0])
        @test field.embedded ≈ [-1.0 0.0 1.0;0.0 0.0 0.0;2.5 -2.5 0.0]
        @test field.macro_flow ≈ [4/sqrt(2),0.0,3/sqrt(2)]
        @test field.observed == Bool[1 0 1;0 0 0;1 1 0]
        @test all(isnan,(r[1,2],r[2,1],r[2,2],r[2,3],r[3,3]))
    end
    @testset "Arbitrary folds retain the same observations" begin
        X=randn(rng,29,7); Y=randn(rng,29,3)
        for F in (2,3,4,7)
            stats=KTrader.fold_sufficient_statistics(X,Y,F)
            @test stats.full_xx ≈ X'*X atol=1e-12
            @test stats.full_xy ≈ X'*Y atol=1e-12
            @test stats.full_yy ≈ Y'*Y atol=1e-12
            for f in 1:F
                train=setdiff(1:29,stats.ranges[f])
                @test stats.full_xx-stats.xx[f] ≈ X[train,:]'*X[train,:] atol=1e-12
                @test stats.full_xy-stats.xy[f] ≈ X[train,:]'*Y[train,:] atol=1e-12
            end
        end
        @test_throws ArgumentError KTrader.fold_sufficient_statistics(X,Y,1)
        @test_throws ArgumentError KTrader.fold_sufficient_statistics(X,Y,30)
    end
    @testset "Primal and dual retain posterior null-space uncertainty" begin
        n,P,N=9,28,2
        X=randn(rng,n,P); Y=randn(rng,n,N)
        Sxx=X'*X; Sxy=X'*Y
        primal=KTrader.ridge_spectrum(X,Sxx,Sxy;dual=false)
        dual=KTrader.ridge_spectrum(X,Sxx,Sxy;dual=true)
        Sigma=[1.0 0.2;0.2 0.7]
        for alpha in (0.3,12.0)
            Vp=KTrader.ridge_covariance(primal,alpha)
            Vd=KTrader.ridge_covariance(dual,alpha)
            vector=randn(rng,P)
            expected=(Sxx+alpha*I)\vector
            @test KTrader.covariance_product(Vp,vector) ≈ expected atol=1e-10
            @test KTrader.covariance_product(Vd,vector) ≈ expected atol=1e-10
            Gp=(primal.B.*(1.0./(primal.values.+alpha)))'*primal.basis'
            Gd=(dual.B.*(1.0./(dual.values.+alpha)))'*dual.basis'
            cp=condition_trace_neutrality(Gp,Vp,Sigma,N,7)
            cd=condition_trace_neutrality(Gd,Vd,Sigma,N,7)
            @test cd.G_c ≈ cp.G_c atol=1e-10
            @test cd.inv_M ≈ cp.inv_M atol=1e-10
        # Dense Gaussian conditioning oracle test: G_cond = G - Omega * C' * (C * Omega * C')^-1 * C * g
        cols_7 = KTrader.get_constraint_columns(N, 7)
        C_7 = zeros(14, N * P)
        for r in 1:14, j in 1:N
            C_7[r, j + (cols_7[r][j] - 1) * N] = 1.0
        end
        Omega_dense = kron(inv(Sxx + alpha * I), Sigma)
        g_uncond = vec(Gp)
        M_dense = C_7 * Omega_dense * C_7'
        inv_M_dense = inv(Symmetric(M_dense))
        lambda_dense = inv_M_dense * (C_7 * g_uncond)
        G_dense = reshape(g_uncond - Omega_dense * C_7' * lambda_dense, N, P)
        @test cp.G_c ≈ G_dense atol=1e-10
        @test cp.inv_M ≈ inv_M_dense atol=1e-10
        end
        # Marginal likelihood of Y under the explicitly conditioned prior.
        C=zeros(14,N*P)
        cols=KTrader.get_constraint_columns(N,7)
        for r in 1:14,j in 1:N
            C[r,j+(cols[r][j]-1)*N]=1
        end
        observation=kron(X,Matrix{Float64}(I,N,N))
        target=vec(Y')
        function dense_evidence(alpha)
            prior=kron(Matrix{Float64}(I,P,P)./alpha,Sigma)
            conditioned=prior-prior*C'*((C*prior*C')\(C*prior))
            cov=kron(Matrix{Float64}(I,n,n),Sigma)+observation*conditioned*observation'
            chol=cholesky(Symmetric(cov))
            -0.5logdet(chol)-0.5dot(target,chol\target)
        end
        a1,a2=0.3,12.0
        cache=KTrader.conditioned_evidence_cache(primal,Y'*Y,n,Sigma)
        @test cache.evidence(log(a1)) ≈ KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma) atol=1e-9
        @test cache.evidence(log(a1))-cache.evidence(log(a2)) ≈ dense_evidence(a1)-dense_evidence(a2) atol=1e-9
        logstep=1e-5
        @test cache.derivative(log(a1)) ≈ (cache.evidence(log(a1)+logstep)-cache.evidence(log(a1)-logstep))/(2logstep) atol=1e-7
        state=KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma;gradient=true)
        direction=[0.1 0.03;0.03 -0.07]; step=1e-5
        derivative=(KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma+step*direction)-
                    KTrader.conditioned_evidence(primal,Y'*Y,n,a1,Sigma-step*direction))/(2step)
        @test dot(state.grad,direction) ≈ derivative atol=1e-7
        # Relative observations have rank N-1, not an artificial noisy macro direction.
        Q=KTrader.relative_gauge(N)
        relative_cov=Q*fill(0.8,1,1)*Q'
        target_relative=Y*Q
        Yrelative=target_relative*Q'
        spectrum_relative=KTrader.ridge_spectrum(X,Sxx,X'*Yrelative;dual=false)
        compact_C=C*kron(Matrix{Float64}(I,P,P),Q)
        function dense_relative_evidence(alpha)
            prior=Matrix{Float64}(I,P,P).*(0.8/alpha)
            conditioned=prior-prior*compact_C'*((compact_C*prior*compact_C')\(compact_C*prior))
            covariance=0.8Matrix{Float64}(I,n,n)+X*conditioned*X'
            chol=cholesky(Symmetric(covariance))
            -0.5logdet(chol)-0.5dot(vec(target_relative),chol\vec(target_relative))
        end
        relative_cache=KTrader.conditioned_evidence_cache(spectrum_relative,Yrelative'*Yrelative,n,relative_cov;gauge=Q)
        @test relative_cache.evidence(log(a1))-relative_cache.evidence(log(a2)) ≈
              dense_relative_evidence(a1)-dense_relative_evidence(a2) atol=1e-9
    end
    @testset "Covariance floor cannot reverse an evidence ascent step" begin
        Sigma=diagm([1e-8,1.0])
        gradient=[-1e9 1e4;1e4 0.1]
        direction=Sigma*gradient*Sigma
        projected=KTrader.project_covariance_step(Sigma,direction,nothing)
        @test minimum(eigvals(Symmetric(projected)))>=1e-8-1e-14
        @test dot(gradient,projected-Sigma)>0
    end
    @testset "Dynamic FFT is the causal convolution, including long histories" begin
        for T in (65,129,17001)
            e=0.1randn(rng,T)
            posterior=causal_fractional_posterior(e)
            expected=zeros(length(DGRID_V1))
            ll=zeros(length(DGRID_V1))
            for (g,d) in enumerate(DGRID_V1)
                weights=frac_weights(d,T); cs=cumsum(weights)
                expected[g]=max(sum(weights[T-s+1]*e[s]^2 for s in 1:T)/cs[T],1e-10)
                if T<200
                    for t in 30:T
                        variance=max(sum(weights[t-s]*e[s]^2 for s in 1:t-1)/cs[t-1],1e-10)
                        ll[g]-=0.5*(log(variance)+e[t]^2/variance)
                    end
                end
            end
            @test posterior.v_forecasts ≈ expected rtol=1e-10
            if T<200
                probabilities=exp.(ll.+log.(KTrader.DELTA_D_V1).-maximum(ll.+log.(KTrader.DELTA_D_V1)))
                probabilities./=sum(probabilities)
                @test posterior.p_d ≈ probabilities atol=1e-11
            end
        end
    end
    @testset "Kelly certification preserves the original objective" begin
        X=[1.2 0.8;1.05 0.95;1.1 0.9]
        w=fast_kelly_solver(X)
        c=KTrader.kelly_certificate(X,w)
        @test c.objective_gap<=1e-8
        @test c.feasibility<=1e-10
        @test w[1]>1-1e-6
        # Forced custom-solver exhaustion uses the same exact log objective.
        fallback=fast_kelly_solver(X;max_outer=0)
        @test KTrader.kelly_certificate(X,fallback).objective_gap<=1e-8
        @test sum(log.(X*fallback))/3 ≈ sum(log.(X*w))/3 atol=1e-8
        random=exp.(0.1randn(rng,80,4))
        base=0.3.*exp.(0.2randn(rng,80))
        locked=fast_kelly_solver(random;budget=0.7,base)
        @test KTrader.kelly_certificate(random,locked;budget=0.7,base).objective_gap<=1e-8
        @test_throws ArgumentError fast_kelly_solver([1.0 -0.01])
        @test_throws ArgumentError fast_kelly_solver([1.0 NaN])
        # Small positive returns must not be silently raised to the old 1e-4 floor.
        tiny=[1e-8 1.0;1e-8 1.1]
        exact=fast_kelly_solver(tiny)
        @test exact[2]>1-1e-6
    end
    @testset "Nested quadrature and sequential chunk semantics" begin
        T=380; P=exp.(cumsum(0.01randn(rng,T,3);dims=1))
        model=fit_v1(P)
        rule=KTrader.ScenarioQuadrature(9,MersenneTwister(40))
        small=generate_scenarios_v1(model;S=64,quadrature=rule)
        large=generate_scenarios_v1(model;S=128,quadrature=rule)
        @test small ≈ large[1:64,:] atol=1e-14
        # The sample mean is not forcibly pinned to the analytic model mean.
        iid=generate_scenarios_v1(model;S=64,rng=MersenneTwister(41))
        @test norm(vec(mean(log.(iid);dims=1))-model.mu_pred)>1e-8
        flat=fit_v1(ones(320,2))
        adaptive=KTrader.adaptive_scenario_weights(flat,trues(2);rng=MersenneTwister(6))
        @test adaptive.S==128
        @test adaptive.weights ≈ [0.5,0.5] atol=1e-9
        @test_throws ErrorException KTrader.adaptive_scenario_weights(model,trues(3);
            rng=MersenneTwister(42),max_scenarios=128,tol=0.0,weight_tol=0.0)
        dates=collect(Date(2020,1,1).+Day.(0:T-1))
        bars=Bars(dates,["A","B","C"],P,P)
        one=backtest_v1(bars;from=dates[end-3],S=64,date_tasks=1,chunk_size=1)
        many=backtest_v1(bars;from=dates[end-3],S=64,date_tasks=2,chunk_size=3)
        @test many.ret ≈ one.ret atol=1e-8
        @test many.weights ≈ one.weights atol=1e-6
    end
end
