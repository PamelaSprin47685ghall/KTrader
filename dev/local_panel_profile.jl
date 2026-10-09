# Prepared-stage microbenchmark, admitted only after the one-call cost is known.
# One EXPLICIT specialization warm-up, then ONE profiled numerical-cold solve.
# No warm numerical state, repeated loop, re-prepare or backtest.
include(joinpath(@__DIR__,"local_panel_stages.jl"))
using Profile, Test

function panel_profile_main()
    isempty(ARGS) || error("no arguments; exactly one warm-up and one measured solve")
    profile_path=joinpath(PANEL_EVIDENCE,"profile.txt")
    ispath(profile_path) && error("refusing to overwrite profile")
    BLAS.set_num_threads(6)
    sources=panel_sources()
    previous=panel_load("prepared",sources)
    prep=previous.prep
    println("T=",prep.T," N=",prep.N," F=",prep.F_folds,
            " CPU=",Sys.CPU_NAME," Julia=",VERSION," BLAS=",BLAS.get_config(),
            " BLAS_threads=",BLAS.get_num_threads())
    println("EXPLICIT one specialization warm-up; alpha_initial omitted on BOTH solves")
    panel_measure("warmup_solve") do
        KTrader.solve(prep)
        nothing # do not retain the warm-up model
    end
    GC.gc() # outside measurement; never timed as a throughput improvement
    Profile.init(n=2_000_000,delay=0.001)
    Profile.clear()
    timing=KTrader.DecisionTiming()
    model=panel_measure("profiled_solve") do
        Profile.@profile KTrader.solve(prep;timing)
    end
    open(profile_path,"w") do io
        println(io,"One prepared solve; sampling/inclusive counts are NOT an exclusive wall-time partition.")
        Profile.print(io;format=:flat,sortedby=:count,C=true,mincount=10)
    end
    for (bucket,seconds) in zip(KTrader.TIMING_BUCKETS,timing.seconds)
        println("timing ",bucket,"=",seconds)
    end
    reference=panel_load("model",sources).model
    @testset "Current full N65 prepared solve vs previous same-input cold solve" begin
        @test model.resp.alpha_macro ≈ reference.resp.alpha_macro rtol=1e-10
        @test model.resp.alpha_rel ≈ reference.resp.alpha_rel rtol=1e-10
        @test model.resp.G_c_mean ≈ reference.resp.G_c_mean rtol=1e-10 atol=1e-12
        @test model.resp.Sigma_rel ≈ reference.resp.Sigma_rel rtol=1e-10 atol=1e-12
        @test model.mu_pred ≈ reference.mu_pred rtol=1e-10 atol=1e-12
        @test model.d_posterior ≈ reference.d_posterior rtol=1e-10 atol=1e-12
        @test model.v_forecasts ≈ reference.v_forecasts rtol=1e-10 atol=1e-12
    end
    panel_sources()==sources || error("runtime source changed during profile")
    println("profile_sha256=",panel_digest(profile_path))
end
panel_profile_main()
