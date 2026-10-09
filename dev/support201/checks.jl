using KTrader, Test, LinearAlgebra, Random
include(joinpath(@__DIR__,"solver.jl"))
const SP=Support201
BLAS.set_num_threads(1)
function fixture(N=6,observed=4,n=100;seed=201)
    rng=MersenneTwister(seed); m=14; Z=zeros(N,observed-1)
    Z[1:observed,:]=KTrader.relative_gauge(observed)
    E=kron(Matrix{Float64}(I,m,m),Z)
    x=randn(rng,n,m*(observed-1)); y=randn(rng,n,observed-1)
    X=x*E'; Y=y*Z'
    (;X,Y,witness=(;Z,components=[collect(1:observed)],cutoff=n),rng)
end
@testset "2.0.1 candidate complete mathematical and boundary checks" begin
@testset "2.0.1 candidate: original law and analytic joint derivatives" begin
    for (N,observed) in ((5,3),(7,5))
        f=fixture(N,observed); g=SP.prepare(f.X'*f.X,f.X'*f.Y,f.Y'*f.Y,size(f.X,1),f.witness)
        r=g.r; S=Matrix{Float64}(I,r,r); alpha=0.7; c=SP.alpha_cache(g,alpha); s=SP.state(g,c,S)
        spectrum=KTrader.ridge_spectrum(f.X,f.X'*f.X,f.X'*f.Y;dual=false)
        Sigma=SP.lift_covariance(g,S); Q=KTrader.relative_gauge(N)
        old=KTrader.conditioned_evidence(spectrum,f.Y'*f.Y,g.n,alpha,Sigma;gauge=Q,gradient=true)
        cache=KTrader.conditioned_alpha_cache(spectrum,f.Y'*f.Y,g.n,alpha;gauge=Q)
        fullS=Q'*Sigma*Q; cholS=cholesky(Symmetric(fullS))
        fullM=KTrader._conditioned_constraint_direct(cache,fullS); cholM=cholesky(Symmetric(fullM))
        original_parts=[-cache.constant,g.n/2*logdet(cholS),tr(cholS\cache.R)/2,
            logdet(cholM)/2,dot(cache.h,cholM\cache.h)/2,-g.m/2*log(tr(fullS)/alpha)]
        println("EVIDENCE_PART_DIFFERENCES N=",N," ambient-minus-reduced=",original_parts.-s.parts)
        @test isapprox(-s.value,old.value;atol=1e-7,rtol=1e-11)
        @test isapprox(-s.gradient,g.Z'*old.grad*g.Z;atol=1e-5,rtol=1e-7)
        scalar=KTrader.conditioned_evidence_cache(spectrum,f.Y'*f.Y,g.n,Sigma;gauge=Q)
        @test isapprox(-s.ga,scalar.derivative(log(alpha));atol=1e-8,rtol=1e-9)
        reduced=SP.scalar_functions(g,S)
        @test isapprox(-s.value,reduced.evidence(log(alpha));atol=1e-9,rtol=1e-12)
        @test isapprox(-s.ga,reduced.derivative(log(alpha));atol=1e-10,rtol=1e-10)
        epsilon=1e-5
        plus=SP.state(g,SP.alpha_cache(g,alpha*exp(epsilon)),S)
        minus=SP.state(g,SP.alpha_cache(g,alpha*exp(-epsilon)),S)
        @test isapprox(s.aa,(plus.ga-minus.ga)/(2epsilon);atol=1e-6,rtol=1e-7)
        @test isapprox(s.ua,(plus.gradient-minus.gradient)/(2epsilon);atol=1e-6,rtol=1e-7)
        D=SP.sym(randn(f.rng,r,r))
        gp=SP.state(g,c,S+epsilon*D).gradient; gm=SP.state(g,c,S-epsilon*D).gradient
        @test isapprox(SP.hessian_action(g,c,S,s,D),(gp-gm)/(2epsilon);atol=1e-5,rtol=1e-7)
        @test isapprox(dot(s.ua,D),(SP.state(g,c,S+epsilon*D).ga-SP.state(g,c,S-epsilon*D).ga)/(2epsilon);atol=1e-6,rtol=1e-7)
        lifted=SP.lift_spectrum(g)
        V=KTrader.ridge_covariance(lifted,alpha); W=KTrader.ridge_covariance(spectrum,alpha)
        z=randn(f.rng,14N)
        @test isapprox(KTrader.covariance_product(V,z),KTrader.covariance_product(W,z);atol=1e-10,rtol=1e-10)
        @test g.a==N-observed
    end
end
@testset "Candidate ownership and rejection boundaries" begin
    f=fixture(); originalZ=copy(f.witness.Z)
    g=SP.prepare(f.X'*f.X,f.X'*f.Y,f.Y'*f.Y,size(f.X,1),f.witness)
    @test g.Z!==f.witness.Z
    @test g.Z==originalZ
    f.witness.Z[1,1]+=0.1
    @test g.Z==originalZ
    @test_throws ArgumentError SP.prepare(f.X'*f.X,f.X'*f.Y,f.Y'*f.Y,size(f.X,1),
        merge(f.witness,(;Z=2f.witness.Z)))
    @test_throws ArgumentError SP.alpha_cache(g,0.0)
    @test_throws ArgumentError SP.solve_candidate(g;maxiters=0)
    @test_throws DimensionMismatch SP.solve_candidate(g;initial_state=ones(g.r+1,g.r+1))
    @test_throws ArgumentError SP.support_basis(ones(10,3),11)
    source=randn(MersenneTwister(86),20,5); source[:,4:5].=NaN
    saved=copy(source); witness=SP.support_basis(source,20)
    @test isequal(source,saved)
    @test size(witness.Z)==(5,2)
end
@testset "Fixed-alpha sublevel enclosure and covariance correction" begin
    f=fixture(6,4,400;seed=6201)
    g=SP.prepare(f.X'*f.X,f.X'*f.Y,f.Y'*f.Y,size(f.X,1),f.witness)
    c=SP.alpha_cache(g,7.0)
    S=SP.sym(c.R./g.n)
    enclosed=SP.fixed_enclosure(g,c,S)
    @test enclosed.valid
    @test enclosed.margin>0
    @test !SP.fixed_enclosure(g,c,3S).valid
    correction=SP.solve_fixed(g,c,S)
    @test correction.valid
    @test correction.state.residual<=1e-13
    @test SP.fixed_enclosure(g,c,correction.S,correction.state).valid
end
end # complete testset; failed children do not hide later boundary checks
println("Candidate derivative checks only; no production fit or release change")
