# Isolate ONE original third-fold structure. Avoid rerunning all eight
# spectral decompositions or the transported-design rank diagnostic.
include(joinpath(@__DIR__,"local_structure.jl"))
using Test
function face_main()
    isempty(ARGS) || error("one fixed third-fold support diagnostic")
    BLAS.set_num_threads(1); current=sources(); record=saved_day(1,current)
    bars=KTrader.load_bars(joinpath(ROOT,"data")); signal=KTrader.signal_prices(bars)
    println("BEGIN one third-fold support audit; no posterior fit; date=",record.date); flush(stdout)
    prep=KTrader.prepare_reference(view(signal,1:record.t,:);F_folds=3)
    training=1:(first(prep.stats.ranges[3])-1)
    cutoff=maximum(prep.ts_total[training])+1
    Z=observation_support(prep.r,cutoff); Q=KTrader.relative_gauge(prep.N)
    Pz=Z*Z'; Pa=Q*Q'-Pz; r=size(Z,2); a=prep.N-1-r
    response=record.model.res_history.fold_models[3]
    println("GRAPH r=",r," null_relative=",a," likelihood_dimensions=",14r," instead of ",prep.P_features); flush(stdout)
    residual=0.0; reference=0.0
    for cr in response.constraint_cols
        Xc=view(prep.X_rel_stacked,training,cr)
        residual+=sum(abs2,Xc-Xc*Pz); reference+=sum(abs2,Xc)
    end
    Y=view(prep.relative_embedding,prep.ts_total[training].+1,:)
    feature_leak=sqrt(residual/reference); target_leak=norm(Y-Y*Pz)/norm(Y)
    cross=norm(Pa*response.Sigma_rel*Pz)/norm(response.Sigma_rel)
    delta=KTrader.EB_COVARIANCE_FLOOR
    floor_error=norm(Pa*response.Sigma_rel*Pa-delta*Pa)
    println("FEATURE/TARGET relative leaks=",feature_leak," / ",target_leak,
        " Sigma cross=",cross," null floor absolute error=",floor_error); flush(stdout)
    xx=prep.stats.full_xx-prep.stats.xx[3]
    xy=prep.stats.full_xy-prep.stats.xy[3]
    yy=prep.stats.full_yy-prep.stats.yy[3]; n=length(training)
    spectrum=KTrader.ridge_spectrum(nothing,xx,xy)
    cache=KTrader.conditioned_alpha_cache(spectrum,yy,n,response.alpha_rel;gauge=Q)
    S=Matrix(Symmetric(Q'*response.Sigma_rel*Q))
    M=KTrader._conditioned_constraint_direct(cache,S); lambda=M\cache.h
    B=Matrix(Symmetric(Z'*response.Sigma_rel*Z))
    Rz=Matrix(Symmetric(Z'*Q*cache.R*Q'*Z)); Lz=Matrix(cholesky(Symmetric(B)).L)
    curv=minimum(eigvals(Symmetric(Lz\Rz/Lz')))/n-0.5
    gamma=-n/(2delta)-(tr(inv(Symmetric(M)))-dot(lambda,lambda))/(2response.alpha_rel)+
        length(cache.h)/(2tr(S))
    mean_error=0.0; mean_norm=0.0
    for (c,cr) in enumerate(response.constraint_cols)
        actual=Pa*response.G_c_mean[:,cr]*Pa
        expected=-(delta/response.alpha_rel)*lambda[c].*Pa
        mean_error=max(mean_error,norm(actual-expected)); mean_norm=max(mean_norm,norm(actual))
    end
    println("REDUCED base curvature/n=",curv," null evidence gradient=",gamma,
        " null mean norm=",mean_norm," null mean formula error=",mean_error); flush(stdout)
    @testset "Authenticated third fold satisfies constructive support identities" begin
        @test r==45 && a==19
        @test feature_leak<1e-10
        @test target_leak<1e-10
        @test cross<1e-10
        @test floor_error<1e-12
        @test curv>0.4
        @test gamma<0
        @test mean_error<1e-12
    end
    @assert current==sources()
    println("ONE_FOLD_STRUCTURE VERIFIED; no altered covariance, alpha, mean, fit or backtest")
end
face_main()
