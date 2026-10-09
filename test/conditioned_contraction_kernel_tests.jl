using Test, Random, LinearAlgebra
using KTrader

# Independent oracle for the constraint contraction M(S): the ORIGINAL
# per-block broadcast formula, re-derived here from the mathematics (not a
# call into production). Both the current direct path (S*bcat GEMM + block
# dots) and any future kernel-Frobenius path
#   M[r,s] = Frobenius(S, core[r,s]),  core = (T_rs + T_rs')/2,  T_rs = bw_r*B_s'
# must pass every assertion below at the original tolerance, because
#   dot(bw_r, S*blocks_s) = sum_{j,k} S[j,k]*T_rs[j,k] = <S, T_rs>
# and for SYMMETRIC S (the only caller contract: Sigma candidates and the
# ray's symmetric indefinite directions D) <S, T> = <S, sym(T)> exactly.
# The baseline diagonal M[r,r] += baseline*tr(S) must appear exactly once.

function ref_M(blocks, w, baseline, S)
    nc=length(blocks)
    transformed=[(S*U) .* w' for U in blocks]
    M=zeros(nc,nc)
    for r in 1:nc, s in r:nc
        M[r,s]=M[s,r]=dot(blocks[r],transformed[s])
    end
    for r in 1:nc
        M[r,r]+=baseline*tr(S)
    end
    M
end

function contraction_fixture(;seed=721,n=120,N=3,dual=false)
    rng=MersenneTwister(seed); P=2*length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); Pi=Q*Q'
    X=randn(rng,n,P)
    for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        X[:,c]=X[:,c]*Pi
    end
    A=zeros(N,P); A[:,1:N]=3.0 .* Pi
    # Noise must be RIGHT-PROJECTED onto the relative support (matrix
    # product, n x N times N x N). The earlier `.*Pi` broadcast is a
    # DimensionMismatch whenever n != N (e.g. 120 vs 3) and, when n == N,
    # would silently apply an elementwise mask instead of the projection
    # the matrix-normal contract requires.
    Y=X*A' .+ 0.25 .* randn(rng,n,N) * Pi
    spectrum=KTrader.ridge_spectrum(X,X'X,X'Y;dual)
    (; X, Y, Q, spectrum, N, n, rng)
end

# Deterministic 2x2 hand counterexample for the cross-core symmetrization
# (mirrors the comment on build_jacobian_cores): B_r=[1;2], B_s=[3;4],
# w=[1] -> T = bw_r*B_s' = [3 4; 6 8]; the correct symmetric core is
# (T+T')/2 = [3 5; 5 8]; a naive upper-triangle mirror gives [3 4; 4 8].
@testset "M(S) kernel Frobenius contraction identities" begin
    # [E1] Hand counterexample pins the symmetrization semantics.
    Br=[1.0,2.0]; Bs=[3.0,4.0]
    T=Br*Bs'
    @test (T .+ T') ./ 2 == [3.0 5.0; 5.0 8.0]
    @test Matrix(Symmetric(T)) == [3.0 4.0; 4.0 8.0]   # mirror != average
    @test Matrix(Symmetric(T)) != (T .+ T') ./ 2

    rng=MersenneTwister(818)
    for (n,dual,N) in ((120,false,3),(30,true,3),(120,false,5))
        f=contraction_fixture(;n,N,dual)
        d=size(f.Q,2)
        for alpha in (0.5,3.0)
            cache=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,alpha;gauge=f.Q)
            # Sigma candidates: two random SPD, one kappa~1e8; plus one
            # SYMMETRIC INDEFINITE direction D (covariance-ray shape).
            B1=randn(rng,d,d); S1=B1*B1' .+ 0.5 .* Matrix{Float64}(I,d,d)
            B2=randn(rng,d,d); S2=B2*B2' .+ 0.5 .* Matrix{Float64}(I,d,d)
            S3=Matrix{Float64}(I,d,d); S3[1,1]=1e-4; S3[d,d]=1e4
            D=randn(rng,d,d); D=Matrix(Symmetric((D .+ D') ./ 2)); D[1,1]-=2.0
            for (tag,Xm) in (("spd1",S1),("spd2",S2),("kappa1e8",S3),("symmetric-indefinite-D",D))
                # [E2] Current direct path matches the independent oracle.
                Mref=ref_M(cache.blocks,cache.V.weights,cache.V.baseline,Xm)
                @test KTrader.conditioned_constraint_contraction(cache,Xm) ≈ Mref atol=1e-12 rtol=1e-12
                # [E3] After a gradient call assembles the cores, the
                # contraction must STILL match the oracle on fresh inputs
                # (the future kernel path is identity-bound here; the
                # current path is unchanged by assembly).
                KTrader.conditioned_cached_state(cache,S1;gradient=true)
                @test cache.jcore[]!==nothing
                @test KTrader.conditioned_constraint_contraction(cache,Xm) ≈ Mref atol=1e-12 rtol=1e-12
            end
            # [E4] Baseline appears exactly once: M - M_without_baseline is
            # baseline*tr(S) on the diagonal and zero elsewhere (no leak,
            # no double add). Dual spectra exercise a nonzero baseline.
            Mfull=KTrader.conditioned_constraint_contraction(cache,S1)
            b=cache.V.baseline
            Mno=ref_M(cache.blocks,cache.V.weights,0.0,S1)
            @test Mfull ≈ Mno .+ b .* tr(S1) .* Matrix{Float64}(I,length(cache.blocks),length(cache.blocks)) atol=1e-12 rtol=1e-12
            # [E5] Cores themselves are the exact symmetrized T (recomputed
            # independently from bwcat/bcat blocks).
            KTrader.conditioned_cached_state(cache,S1;gradient=true)
            cores=cache.jcore[]
            nc=length(cache.blocks); rk=div(size(cache.bcat,2),nc)
            for r in 1:nc, s in r:nc
                lo_r=(r-1)*rk+1; hi_r=r*rk; lo_s=(s-1)*rk+1; hi_s=s*rk
                T=view(cache.bwcat,:,lo_r:hi_r)*view(cache.bcat,:,lo_s:hi_s)'
                idx=(r-1)*nc+s-div(r*(r-1),2)
                @test cores[idx] ≈ (T .+ T') ./ 2 atol=1e-12 rtol=1e-12
                @test cores[idx] ≈ Matrix(Symmetric(cores[idx])) atol=0 atol=0
            end
            # [E6] Mutation: a naive upper-triangle-mirror core family must
            # NOT reproduce the oracle through Frobenius contraction (this
            # is the wrong-index variant; correct cores must stay green).
            nc=length(cache.blocks)
            mirror=Vector{Matrix{Float64}}(undef,div(nc*(nc+1),2))
            for r in 1:nc, s in r:nc
                lo_r=(r-1)*rk+1; hi_r=r*rk; lo_s=(s-1)*rk+1; hi_s=s*rk
                T=view(cache.bwcat,:,lo_r:hi_r)*view(cache.bcat,:,lo_s:hi_s)'
                mirror[(r-1)*nc+s-div(r*(r-1),2)]=Matrix(Symmetric(T))
            end
            # Frobenius contraction with the mirror cores vs oracle:
            Mmirror=zeros(nc,nc)
            for r in 1:nc, s in r:nc
                idx=(r-1)*nc+s-div(r*(r-1),2)
                v=dot(S1,mirror[idx])
                Mmirror[r,s]=Mmirror[s,r]=v
            end
            for r in 1:nc
                Mmirror[r,r]+=b .* tr(S1)
            end
            Mtrue=ref_M(cache.blocks,cache.V.weights,b,S1)
            @test !isapprox(Mmirror, Mtrue; atol=1e-8)
        end
        # [E7] Alpha invalidation: contractions differ across alphas (a
        # cache that reused cores/contractions across alphas goes red).
        c1=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,0.5;gauge=f.Q)
        c2=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,3.0;gauge=f.Q)
        Bk=randn(rng,d,d); Sk=Bk*Bk' .+ 0.5 .* Matrix{Float64}(I,d,d)
        @test !isapprox(KTrader.conditioned_constraint_contraction(c1,Sk),
                        KTrader.conditioned_constraint_contraction(c2,Sk); atol=1e-8)
        # [E8] Laziness: value-only evaluation never assembles cores.
        c3=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,1.7;gauge=f.Q)
        @test c3.jcore[]===nothing
        @test KTrader.conditioned_cached_state(c3,Sk;gradient=false) isa Float64
        @test c3.jcore[]===nothing
    end
end
