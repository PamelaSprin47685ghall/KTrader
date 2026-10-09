using Test, Random, LinearAlgebra, Dates

# M1 reference-isolation contract: the reference core must prepare -> solve
# -> scenarios -> Kelly and run a minimal sequential batch backtest WITHOUT
# including the incremental engine or the ceiling probes. This file builds
# an independent module from the REAL source files (minimal include set,
# no copied math, no re-implemented solver) and asserts the isolation
# structurally: the accelerator/probe symbols are NOT merely unused, they
# must be absent from the module's namespace. If a core file (e.g.
# backtest.jl) actually requires an incremental-only type at load time, the
# include itself fails here - that is the defect being tested, not
# something to isdefined-skip around.

module RefIsolation
using LinearAlgebra, Statistics, Random, Dates
using TimeZones, Convex, Clarabel
using HTTP, JSON3, CSV
using FFTW, Distributions
include(joinpath(@__DIR__, "..", "src", "data.jl"))
include(joinpath(@__DIR__, "..", "src", "geometry.jl"))
include(joinpath(@__DIR__, "..", "src", "numerics.jl"))
include(joinpath(@__DIR__, "..", "src", "response.jl"))
include(joinpath(@__DIR__, "..", "src", "residual_oracle.jl"))
include(joinpath(@__DIR__, "..", "src", "prepare.jl"))
include(joinpath(@__DIR__, "..", "src", "predict.jl"))
include(joinpath(@__DIR__, "..", "src", "kelly.jl"))
include(joinpath(@__DIR__, "..", "src", "backtest.jl"))
end

@testset "Reference isolation: minimal core without accelerator/probes" begin
    REFISO_STAGE = get(ENV, "REFISO_STAGE", "ALL")   # phase-split runner: R2/R3 can run in separate <=45s commands
    REFISO_STAGE in ("ALL", "R2", "R3") ||
        error("unknown REFISO_STAGE $(REFISO_STAGE): must be ALL, R2 or R3 (fail-closed, no silent phase omission)")
    mk(t) = (println(stderr, "[refiso][", t, "] ", time()); flush(stderr))
    # [R1] Structural isolation: the incremental BACKEND and the probe
    # entities are absent from the module namespace (positive absence on a
    # module we built ourselves - not isdefined skips). prepare_incremental
    # is deliberately NOT on this list: it is the high-level forward API
    # defined in prepare.jl whose body references the backend lazily - it
    # legitimately exists without the concrete accelerator and is part of
    # the useful core surface.
    for sym in (:ExactInferenceState, :initialize_inference, :advance_exact!,
                :solve_current!, :inference_checkpoint, :inference_diagnostics,
                :RawInferenceCore, :RegimeRun,
                :CeilingProbeCollector, :print_probe_report,
                :mask_transition_coverage, :solve_floor_probe,
                :sequential_ceiling_probe, :PROBE_COVERAGE_WINDOWS,
                :record_probe_checkpoints!, :collect_probe_window!,
                :finish_ceiling_probe!)
        @test !isdefined(RefIsolation, sym)
    end
    # The reference surface IS present (the module is a working core, not
    # an empty shell) - including the lazy prepare_incremental forward.
    for sym in (:prepare_reference, :solve, :PreparedProblem, :fit_v1,
                :generate_scenarios_v1, :scenario_weights, :kelly_certificate,
                :backtest_v1, :Bars, :signal_prices, :prepare_incremental)
        @test isdefined(RefIsolation, sym)
    end

    mk("R1-end")
    # [R2] Small causal fixture: 3 assets, prefix just past WARMUP, fixed
    # seed, very few scenarios. Correctness only - no throughput claims,
    # no truncated history for performance numbers. NOTE: n_res=39 < P=42
    # here, so this fixture exercises the DUAL branch of the ridge
    # spectrum - a small-scale correctness case, NOT a representative
    # real-day primal system.
    rng=MersenneTwister(2027)
    T=RefIsolation.WARMUP+40; N=3
    ret=0.006 .* randn(rng, T+1, N)
    prices=exp.(cumsum(ret; dims=1))
    dates=Date(2000,1,1) .+ Day.(0:T)
    bars=RefIsolation.Bars(dates, ["A","B","C"], copy(prices), copy(prices))

    REFISO_STAGE in ("ALL", "R2") || mk("R2-fit-skipped")
    if REFISO_STAGE in ("ALL", "R2")
    mk("R2-solve-begin")
    prep=RefIsolation.prepare_reference(prices[1:T,:]; F_folds=3)
    @test prep isa RefIsolation.PreparedProblem
    model=RefIsolation.solve(prep; ridge_alpha=1.0)
    @test all(isfinite, model.mu_pred[model.active_indices])
    @test model.resp.alpha_rel > 0
    # Trace neutrality at the mathematical level: every band/channel block
    # of the conditioned posterior mean G_c_mean has zero trace (the
    # structural identification constraint), computed directly from the
    # matrix - independent of any summary-field semantics.
    Gc=model.resp.G_c_mean
    cols=RefIsolation.get_constraint_columns(model.N_universe == 0 ? N :
        length(model.active_indices), length(RefIsolation.BANDS))
    for c in cols
        @test abs(tr(view(Gc, :, c))) <= 1e-10
    end
    # The summary fields (present in the current ResponseOperator at
    # response.jl L102-103, consumed identically by conditioned_eb_tests
    # and bin/report.jl) agree with the direct computation.
    @test abs(model.resp.trace_real) <= 1e-10
    @test abs(model.resp.trace_imag) <= 1e-10

    scen=RefIsolation.generate_scenarios_v1(model; S=24,
        rng=MersenneTwister(11))
    @test all(isfinite, scen[:, model.active_indices])
    w=RefIsolation.scenario_weights(scen, model.active_indices,
                                     trues(model.N_universe))
    @test sum(w) ≈ 1.0 atol=1e-9
    # Kelly certificate on the exact original objective.
    idx=model.active_indices
    cert=RefIsolation.kelly_certificate(view(scen, :, idx), w[idx])
    @test cert.feasibility <= 1e-8
    @test cert.kkt_residual <= 1e-8
    @test cert.objective_gap <= 1e-8

    end # R2 stage
    mk("R2-end")
    if REFISO_STAGE in ("ALL", "R3")
    mk("R3-begin")
    # [R3] Minimal sequential batch backtest: engine=:batch (the reference
    # path), date_tasks=1, exactly ONE decidable day, one true causal
    # marking advance (weights -> next-day gross -> holdings roll).
    # N2 well-conditioned sub-panel (SAME real price-generation rule as
    # R2's N3 fixture, two assets) so the DEFAULT full-EB solve
    # (ridge_alpha=nothing: full + 3 independent fold EB alternations) fits
    # the <=45s phase budget. Real EB default / F3 / S24 / one decision /
    # channel / spool / causal marking / original tolerances ALL unchanged;
    # the accounting identity uses res's ACTUAL returned weights.
    let rng2=MersenneTwister(2028)
        local T2=RefIsolation.WARMUP+40; local N2=2
        local ret2=0.006 .* randn(rng2, T2+1, N2)
        local prices2=exp.(cumsum(ret2; dims=1))
        local dates2=Date(2000,1,1) .+ Day.(0:T2)
        local bars2=RefIsolation.Bars(dates2, ["A","B"], copy(prices2), copy(prices2))
    res=RefIsolation.backtest_v1(bars2; from=dates2[T2], S=24, seed=5,
                                 engine=:batch, date_tasks=1, blas_threads=1)
    @test length(res.ret) == 1
    @test all(isfinite, res.ret)
    @test isfinite(res.wealth[end])
    @test sum(res.weights[1,:]) ≈ 1.0 atol=1e-9
    # The single day's account advance is a real causal marking: the
    # realized return equals the weighted next-bar gross minus one.
    gross=RefIsolation.signal_prices(bars2)[T2+1,:] ./ RefIsolation.signal_prices(bars2)[T2,:]
    gclean=ifelse.(isfinite.(gross), gross, 1.0)
    @test res.ret[1] ≈ dot(res.weights[1,:], gclean) - 1.0 atol=1e-12
    end # N2 R3 fixture let
    mk("R3-return")
    end # R3 stage
    # No incremental machinery was engaged anywhere above: the module
    # simply does not contain it (R1), and batch engine never looks for it.
end
