module ConditionedDeadworkTests

using Test, Random, LinearAlgebra
using KTrader

# Pre-refactor inner loop: the same proposal/acceptance arithmetic, but each
# trial goes through the FULL state evaluator. Test-only; never a second
# production solver. This pins the fixed-point trajectory, not just its limit.
function full_state_fixed_point(cache,S;iters=40,tol=1e-10)
    nc=length(cache.h)
    function frozen_step(X)
        state=KTrader.conditioned_cached_state(cache,X;gradient=true)
        target=Matrix(Symmetric((cache.R .+ (nc/tr(X)).*(X*X) .- X*state.J*X)./cache.n))
        (;target,residual=norm(target-X),state)
    end
    ev=eigen(Symmetric(Matrix(cache.R)./cache.n))
    Qroot=(ev.vectors .* sqrt.(max.(ev.values,0.0))') * ev.vectors'
    current=frozen_step(S)
    for _ in 1:iters
        current.residual<=tol*max(norm(S),1.0) && break
        candidates=[KTrader.positive_covariance(current.target)]
        rc=KTrader.riccati_stationary_candidate(Qroot,current.state.J,tr(S),nc,cache.n)
        rc===nothing || push!(candidates,KTrader.positive_covariance(rc))
        best=nothing
        for candidate in candidates
            omega=1.0
            for _ in 1:8
                trial=Matrix(Symmetric(S .+ omega.*(candidate .- S)))
                probe=frozen_step(trial)
                if probe.residual<=current.residual*(1+1e-12)
                    if best===nothing || probe.residual<best[2].residual
                        best=(trial,probe)
                    end
                    break
                end
                omega*=0.5
            end
        end
        best===nothing && break
        S=best[1]; current=best[2]
    end
    Matrix(Symmetric(S))
end

function fixture(n,N,dual,gauged)
    rng=MersenneTwister(812+n+N)
    P=2length(KTrader.BANDS)*N
    Q=gauged ? KTrader.relative_gauge(N) : nothing
    X=randn(rng,n,P); Y=randn(rng,n,N)
    if Q!==nothing
        Pi=Q*Q'
        for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
            X[:,c]=X[:,c]*Pi
        end
        Y=Y*Pi
    end
    spectrum=KTrader.ridge_spectrum(X,X'X,X'Y;dual)
    (;spectrum,YtY=Y'Y,n,N,Q)
end

@testset "Conditioned EB: trace-only cache and J-only proposals" begin
    for (n,N,dual) in ((72,3,false),(18,3,true),(80,4,false)), gauged in (false,true)
        f=fixture(n,N,dual,gauged)
        cols=KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        dS=gauged ? N-1 : N
        fresh(alpha)=KTrader.conditioned_alpha_cache(f.spectrum,f.YtY,n,alpha;gauge=f.Q)
        for alpha in (0.01,1.0,100.0)
            c=fresh(alpha)
            Bd=f.spectrum.B .* (1.0 ./ (f.spectrum.values .+ alpha))
            G=Bd'*f.spectrum.basis'
            h_ref=[tr(view(G,:,indices)) for indices in cols]
            @test c.h ≈ h_ref atol=1e-12 rtol=1e-12
            @test !hasproperty(c,:G) # no unused N x P payload in an alpha cache
            @test c.jcore[]===nothing
            S=Matrix{Float64}(I,dS,dS)
            dense_trace_cache=merge(fresh(alpha),(;h=h_ref))
            st=KTrader.conditioned_cached_state(c,S;gradient=true)
            ref=KTrader.conditioned_cached_state(dense_trace_cache,S;gradient=true)
            @test st.value ≈ ref.value atol=1e-12 rtol=1e-12
            @test st.grad ≈ ref.grad atol=1e-12 rtol=1e-12
            @test st.direction ≈ ref.direction atol=1e-12 rtol=1e-12
            # Cold-cache order, including lazy core assembly, must be identical.
            cj=fresh(alpha)
            M=KTrader.conditioned_constraint_contraction(cj,S)
            cholM=cholesky(Symmetric(M))
            J=KTrader.conditioned_constraint_jacobian(cj,cholM,cholM\cj.h)
            @test J==st.J
            @test cj.jcore[]!==nothing
        end
        # Check every prefix of a short proposal trajectory, plus a floor-active
        # starting covariance. No tolerance relaxation and no solver-budget change.
        for small in (1.0,KTrader.EB_COVARIANCE_FLOOR), steps in (0,1,3)
            S=Matrix{Float64}(I,dS,dS); S[1,1]=small
            before=copy(S)
            actual=KTrader.sigma_stationary_fixed_point(fresh(1.0),S;iters=steps)
            expected=full_state_fixed_point(fresh(1.0),S;iters=steps)
            @test actual==expected
            @test S==before
        end
        bad=-Matrix{Float64}(I,dS,dS)
        @test_throws PosDefException KTrader.sigma_stationary_fixed_point(fresh(1.0),bad)
        @test_throws PosDefException full_state_fixed_point(fresh(1.0),bad)
    end
end

end # module
