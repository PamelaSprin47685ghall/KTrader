using Test, Random, LinearAlgebra
using KTrader

# Independent reference for the Sigma-iteration identities of the fixed-alpha
# cache: the ORIGINAL per-block broadcast contraction and the ORIGINAL
# eigen+work Jacobian loop, re-implemented here from the mathematics. The
# production path (symmetric-core contraction) must match these at roundoff
# for every SPD Sigma at a FIXED actual alpha - primal and dual spectra, well
# conditioned and kappa~1e8. The cache lives only inside one alpha cache; a
# broken alpha key (cores reused across alphas) must turn the counterexample
# test red.

function ref_contraction(blocks, weights, baseline, S)
    nc=length(blocks)
    transformed=[(S*U) .* weights' for U in blocks]
    M=zeros(nc,nc)
    for r in 1:nc, s in r:nc
        M[r,s]=M[s,r]=dot(blocks[r],transformed[s])
    end
    for r in 1:nc
        M[r,r]+=baseline*tr(S)
    end
    M
end

function ref_state(cache, S)
    M=ref_contraction(cache.blocks, cache.V.weights, cache.V.baseline, S)
    cholM=cholesky(Symmetric(M)); cholS=cholesky(Symmetric(S))
    z=cholM\cache.h; nc=length(cache.h)
    value=cache.constant-cache.n/2*logdet(cholS)-0.5tr(cholS\cache.R)-
          0.5logdet(cholM)-0.5dot(cache.h,z)+nc/2*log(tr(S)/cache.alpha)
    invM=Matrix(inv(cholM))
    J=zeros(size(S)); multipliers=eigen(Symmetric(invM-z*z'))
    work=zeros(size(S,1),size(cache.V.basis,2))
    for c in 1:nc
        fill!(work,0.0)
        for r in 1:nc
            work .+= multipliers.vectors[r,c].*cache.blocks[r]
        end
        J .+= multipliers.values[c].*((work .* cache.V.weights')*work')
    end
    for j in axes(J,1)
        J[j,j]+=cache.V.baseline*sum(multipliers.values)
    end
    (; value, M, J)
end

function jcore_fixture(;seed=711,n=120,N=3,dual=false)
    rng=MersenneTwister(seed); P=2*length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); Pi=Q*Q'
    X=randn(rng,n,P)
    for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        X[:,c]=X[:,c]*Pi
    end
    A=zeros(N,P); A[:,1:N]=3.0 .* Pi   # deliberately violates the trace prior
    Y=X*A' .+ 0.25 .* (randn(rng,n,N) * Pi)
    spectrum=KTrader.ridge_spectrum(X,X'X,X'Y;dual)
    (; X, Y, Q, spectrum, N, n, rng)
end

@testset "Fixed-alpha Jacobian core cache: exact Sigma-iteration identities" begin
    rng=MersenneTwister(4242)
    for (n,dual,N) in ((120,false,3),(30,true,3),(120,false,5))
        f=jcore_fixture(;n,N,dual)
        d=size(f.Q,2)
        for alpha in (0.5,3.0)
            cache=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,alpha;gauge=f.Q)
            # Sigma candidates: two random SPD, one kappa~1e8 SPD.
            B1=randn(rng,d,d); S1=B1*B1' .+ 0.5 .* Matrix{Float64}(I,d,d)
            B2=randn(rng,d,d); S2=B2*B2' .+ 0.5 .* Matrix{Float64}(I,d,d)
            S3=Matrix{Float64}(I,d,d)
            S3[1,1]=1e-4; S3[d,d]=1e4
            for (tag,S) in (("spd1",S1),("spd2",S2),("kappa1e8",S3))
                Mref=ref_contraction(cache.blocks,cache.V.weights,cache.V.baseline,S)
                @test KTrader.conditioned_constraint_contraction(cache,S) ≈ Mref atol=1e-12 rtol=1e-12
                st=KTrader.conditioned_cached_state(cache,S;gradient=true)
                rs=ref_state(cache,S)
                @test st.M ≈ rs.M atol=1e-12 rtol=1e-12
                @test st.value ≈ rs.value atol=1e-12 rtol=1e-12
                # The reference J is theoretically symmetric (K symmetric,
                # single-sided weight cancels in the (j,k)/(k,j) product);
                # its roundoff skew was truncated by the original consumers
                # via Matrix(Symmetric(...)) - Julia keeps the UPPER
                # triangle. The core contraction builds the exact symmetric
                # J; the comparison anchors the upper-triangle behavior of
                # the original path (dual negative weights and baseline are
                # covered by the dual fixture; kappa by S3).
                @test st.J ≈ Matrix(Symmetric(rs.J)) atol=1e-12 rtol=1e-12
                @test st.J ≈ Matrix(Symmetric(st.J)) atol=0 atol=0  # exactly symmetric
            end
            # Full free directional derivative on the well-conditioned
            # candidates: an independent finite-difference check of the
            # returned gradient, not a mirror of the implementation.
            for S in (S1,S2)
                st=KTrader.conditioned_cached_state(cache,S;gradient=true)
                D=randn(rng,d,d); D=Matrix(Symmetric((D .+ D') ./ 2))
                t=1e-6
                v0=KTrader.conditioned_cached_state(cache,S;gradient=false)
                vp=KTrader.conditioned_cached_state(cache,S .+ t .* D;gradient=false)
                vm=KTrader.conditioned_cached_state(cache,S .- t .* D;gradient=false)
                fd=(vp-vm)/(2t)
                @test dot(st.grad,D) ≈ fd atol=1e-5 rtol=1e-5
            end
        end
    end
    # Alpha-cache-key counterexample: the cache contents must depend on
    # alpha. Cores are lazy, so drive one gradient evaluation per cache and
    # compare both the immediately assembled weighted blocks and the cores.
    f=jcore_fixture()
    c1=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,0.5;gauge=f.Q)
    c2=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,3.0;gauge=f.Q)
    @test !isapprox(c1.bwcat, c2.bwcat; atol = 1e-10)
    d=size(f.Q,2)
    Bk=randn(rng,d,d); Sk=Bk*Bk' .+ 0.5 .* Matrix{Float64}(I,d,d)
    KTrader.conditioned_cached_state(c1,Sk;gradient=true)
    KTrader.conditioned_cached_state(c2,Sk;gradient=true)
    @test length(c1.jcore[])==length(c2.jcore[])
    @test any(i -> !isapprox(c1.jcore[][i], c2.jcore[][i]; atol = 1e-10), eachindex(c1.jcore[]))
    # Same target, same cold-alpha basin, certificate unchanged: the cache is
    # engineering, the optimizer must still certify exactly as before.
    solved=KTrader.optimize_conditioned_eb(f.spectrum,f.Y'f.Y,f.n;gauge=f.Q,
        return_certificate=true)
    @test solved.certificate.valid
    @test solved.certificate.covariance.min_eigenvalue >= KTrader.EB_COVARIANCE_FLOOR-1e-12
    @test solved.certificate.alpha.rel_res <= 1e-6
    # Laziness gate: a freshly built cache has no cores; value-only
    # evaluation must not assemble them; the first gradient call does.
    alpha=0.9
    cache=KTrader.conditioned_alpha_cache(f.spectrum,f.Y'f.Y,f.n,alpha;gauge=f.Q)
    @test cache.jcore[]===nothing
    d=size(f.Q,2)
    B=randn(rng,d,d); S0=B*B' .+ 0.5 .* Matrix{Float64}(I,d,d)
    @test KTrader.conditioned_cached_state(cache,S0;gradient=false) isa Float64
    @test cache.jcore[]===nothing
    st=KTrader.conditioned_cached_state(cache,S0;gradient=true)
    @test cache.jcore[]!==nothing
    @test length(cache.jcore[])==div(14*15,2)
end

@testset "cross-core symmetry: deterministic 2x1 counterexample (kernel owner)" begin
    # B_r=[1;2], B_s=[3;4], w=ones: A=bw_r*B_s'=[3 4;6 8]; the correct
    # symmetric cross core is (A+A')/2=[3 5;5 8]. The old wrong code used
    # (A+B)/2 with B=B_r*bw_s'==A (same column weights), which only
    # mirrors the upper triangle -> [3 4;4 8]. Enters via the real kernel
    # owner build_jacobian_cores; the reference is NOT relaxed.
    B1=[1.0;2.0]; B2=[3.0;4.0]
    cache=(blocks=[B1, B2], bcat=[B1 B2], bwcat=[B1 B2])  # w=1: bwcat==bcat
    cores=KTrader.build_jacobian_cores(cache)
    cross=cores[2]  # pair (r=1,s=2) -> flat index 2
    @test cross == [3.0 5.0; 5.0 8.0]          # correct average, exact
    @test cross != [3.0 4.0; 4.0 8.0]          # old wrong mirror must differ
    @test cross == cross'                       # exactly symmetric
    # diagonal cores unchanged by this fix: (A+A')/2 with A=bw_r*B_r'
    @test cores[1] == Matrix(Symmetric([1.0 2.0; 2.0 4.0]))
    @test cores[3] == Matrix(Symmetric([9.0 12.0; 12.0 16.0]))
end
