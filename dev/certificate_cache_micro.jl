include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Statistics, Test
function cc_measure(f)
    f() # explicit primitive warm-up
    a=[@timed(f()) for _ in 1:5]
    (;seconds=median(x.time for x in a),bytes=median(x.bytes for x in a),compile=sum(x.compile_time for x in a))
end
function cc_main()
    isempty(ARGS) || error("fixed bounded primitive probe; no fit")
    BLAS.set_num_threads(6); sources=panel_sources()
    meta=TOML.parsefile(joinpath(PANEL_EVIDENCE,"prepared.toml"))
    meta["sha256"]=="82b18e9b0283aa5aa9ef31e95e99d4870b5af49e70d800029d9fda8e2e4bfedc" || error("wrong input")
    prep=panel_load("prepared",meta["source_hashes"]).prep
    mm=TOML.parsefile(joinpath(PANEL_EVIDENCE,"model.toml"))
    mm["sha256"]=="691c3297813f2511efaffbee276c11274fb4ce03f32b19bf9cd5fa8eb0cd3577" || error("wrong model")
    model=panel_load("model",mm["source_hashes"]).model
    spectrum=KTrader.ridge_spectrum(nothing,prep.stats.full_xx,prep.stats.full_xy)
    YtY=prep.stats.full_yy; n=prep.n_res; gauge=KTrader.relative_gauge(prep.N)
    alpha=model.resp.alpha_rel; Sigma=model.resp.Sigma_rel
    geometry=KTrader._conditioned_scalar_geometry(spectrum,YtY;gauge)
    ag=KTrader._conditioned_alpha_geometry(geometry)
    cache=KTrader._conditioned_alpha_cache(spectrum,YtY,n,alpha,ag;gauge)
    before=()->KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,ag;gauge)
    after=()->KTrader._conditioned_eb_certificate(spectrum,YtY,n,alpha,Sigma,geometry,ag,cache;gauge)
    a=before(); b=after()
    @test isequal(a,b) && a.valid
    println("N65/P910 authenticated statistics; zero posterior fits; BLAS=6")
    println("after reuses existing optimizer alpha-cache/cores; construction excluded and not free")
    println("before_A ",cc_measure(before))
    println("after_B ",cc_measure(after))
    println("after_B2 ",cc_measure(after))
    println("before_A2 ",cc_measure(before))
    sources==panel_sources() || error("source changed")
    println("all certificate fields exact, valid=true source=",sources["src/response.jl"])
end
cc_main()
