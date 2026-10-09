#!/usr/bin/env julia
# Ceiling probes CLI. Probes are diagnostic code only: they do not change
# posterior mathematics, fallback semantics or any decision output.
#
# ---------------------------------------------------------------------------
# Incident 2026-10-07 — why this CLI is shaped the way it is.
#
# A run with the previous defaults (--probes A,B,C,D,E, D repeats=100 with
# cold AND warm modes, each with an automatic warmup solve) hid roughly 202
# solves behind one flag and exceeded the per-command 60 s budget; the logs
# only printed bucket summaries after each stage finished, so an aborted run
# left nothing to locate. Causal notes, recorded here so the same detour is
# not re-explored:
#
#   * --from only crops the BACKTEST evaluation range. D/E never read it:
#     their input size is the full 1:t_star prefix (t_star defaults to the
#     last decision bar). Re-running with different --from values while
#     waiting on the same stage produces no new information — the prepare and
#     the solve do the same work on the same full prefix every time.
#   * A tool-level abort (timeout) is not process death: the flushed stderr
#     stream survives, so every stage prints an immediate start line before
#     doing any work ([stage:prep|solve][start] t_star=... rows=... mode=...
#     run=...) and an immediate end line after. An aborted run therefore
#     always shows which stage it entered.
#   * Admission: one command = at most one prepare + one solve (D and E are
#     two readings of the SAME solve). Multi-date A/C runs, repeats>1 and
#     dual-mode loops are REJECTED — there is no --force and none will be
#     added. Re-admission requires verified single-day correctness and cost.
#     All admitted commands plus teardown must fit in 60 s.
#
# M1 CLI alignment (2026-10-07): the production probe plumbing is gone, so
# this script no longer passes a probe keyword to backtest_v1 (none exists)
# and no longer calls the prepare/solve stage probes through the production
# KTrader namespace (those live in the DevProbes namespace only). A and C are now the SAME single-day
# dev sequential probe — one initialize + one solve_current! — sharing one
# run and one report. When A/C run they REPLACE the phase stages: a legal
# command still performs at most one prepare and one solve. Unknown flags,
# duplicate flags and conflicting combinations are rejected fail-closed
# before any stage runs (parse/admission live in DevProbes and are shared
# verbatim with the tests).
# ---------------------------------------------------------------------------

using Printf
using Statistics
using Dates

if length(ARGS) < 1
    println(stderr, "usage: ceiling_probes.jl <bars-dir> [--from 2018-01-01] [--t-star <bar-index>] [--phase prepare|solve] [--mode cold|warm] [--folds 3] [--ridge-alpha <float>] [--probes B] [--out <file>]")
    println(stderr, "probes: B mask-transition coverage (pure statistics, zero solves);")
    println(stderr, "        A/C the single-day dev sequential run (K=1; multi-day rejected);")
    println(stderr, "        A/C replace the phase stages — the command's single prepare+solve.")
    println(stderr, "        D/E are readings of the phase stages, not standalone runs.")
    println(stderr, "phase:  prepare (DEFAULT) — one prepare on the full 1:t_star prefix, no solve;")
    println(stderr, "        solve — TWO explicit phases (prepare, then ONE solve on that same prepared")
    println(stderr, "        input); D (elapsed baseline) and E (OOF bucket split) are readings of")
    println(stderr, "        that single authorized solve. The default run performs no solve at all.")
    exit(2)
end

bars_dir = ARGS[1]

# Fail-closed parse: parse_probe_cli lives in DevProbes, so the includes run
# first, but NOTHING is loaded from disk and no stage starts before parsing —
# a bad command line is rejected before load_bars ever runs.
include(normpath(joinpath(@__DIR__, "..", "src", "KTrader.jl")))
using .KTrader
# M1: probes live in the dev namespace (module DevProbes in dev/probes.jl),
# which calls the single production prepare_reference/solve boundary. KTrader
# no longer defines or exports any probe symbol.
include(normpath(joinpath(@__DIR__, "..", "dev", "probes.jl")))
using .DevProbes

opts = try
    DevProbes.parse_probe_cli(ARGS[2:end])
catch e
    println(stderr, sprint(showerror, e))
    exit(2)
end

from_date = try
    Date(opts["from"])
catch e
    println(stderr, "--from must be a YYYY-MM-DD date: $(sprint(showerror, e))")
    exit(2)
end
folds = try
    parse(Int, opts["folds"])
catch e
    println(stderr, "--folds must be an integer: $(sprint(showerror, e))")
    exit(2)
end
t_star = isempty(opts["t-star"]) ? nothing : try
    parse(Int, opts["t-star"])
catch e
    println(stderr, "--t-star must be an integer bar index: $(sprint(showerror, e))")
    exit(2)
end
ridge_alpha = isempty(opts["ridge-alpha"]) ? nothing : try
    parse(Float64, opts["ridge-alpha"])
catch e
    println(stderr, "--ridge-alpha must be a float: $(sprint(showerror, e))")
    exit(2)
end
out_path = opts["out"]

b = load_bars(bars_dir)
from = findfirst(>=(from_date), b.dates)
from === nothing && error("from date $from_date not found in panel")
K = size(b.adj, 1) - from

# Admission (shared with the tests): rejects bad phase/mode/probe values and
# every conflicting combination — A/C with K>1, A/C with --phase solve (a
# second solve), A/C with D/E (no stage readings exist for them), A/C with
# --t-star (not read by them), E without --phase solve — all BEFORE any
# stage runs.
admission = try
    DevProbes.validate_probe_cli(opts; K = K)
catch e
    println(stderr, "REJECTED: $(sprint(showerror, e))")
    exit(2)
end
phase = admission.phase
mode = admission.mode
probe_list = admission.probe_list
has_probe(p) = p in probe_list

println("ceiling probes: $(size(b.adj, 1)) bars, $(size(b.adj, 2)) assets, evaluation decisions from $(b.dates[from]) (K=$K)")

# ---------------- Probe B ----------------
if has_probe(:B)
    println("\n=== probe B: mask transition coverage (engine daily observed-return mask) ===")
    # The mask that decides the regime boundaries of e_t(d) is defined on
    # return rows — finite(log s[k+1]-log s[k]), two real price endpoints —
    # so the probe reads exactly the engine's signal matrix. Runs and
    # transitions are computed on the FULL prefix; `from` crops only the
    # in-range statistics.
    signal = signal_prices(b)
    coverage = DevProbes.mask_transition_coverage(signal; from = from)
    @printf("decision days: %d (bars %d..%d, denominator = T - from)\n",
            coverage.decision_days, from, size(signal, 1) - 1)
    @printf("mask transitions (daily observed-return mask, non-monotone): %d in range (rate %.4f/decision)\n",
            coverage.transitions, coverage.transition_rate)
    if coverage.pre_range_transitions > 0
        @printf("pre-range mask transitions (before bar %d, detected on full prefix): %d\n",
                from, coverage.pre_range_transitions)
    end
    @printf("unique masks (decision range): %d  (whole prefix: %d)\n",
            coverage.unique_masks, coverage.prefix_unique_masks)
    if isempty(coverage.transition_positions)
        println("  stable runs: no in-range transitions; one stable run of $(coverage.decision_days) days")
        @printf("  stable run statistics: median %.1f, max %d\n",
                coverage.stable_run_median, coverage.stable_run_max)
    else
        @printf("stable runs (decision range [from, T-1], transition day opens the new segment): median %.1f, p90 %.1f, max %d, min %d\n",
                coverage.stable_run_median, coverage.stable_run_p90,
                coverage.stable_run_max, coverage.stable_run_min)
        println("  transition positions (decision-day indices): ", join(coverage.transition_positions, ","))
    end
    println("transition coverage by forward window (user metric: [t, t+w-1] price bars, full-prefix detection):")
    for (w, c) in zip(coverage.coverage_windows, coverage.coverage)
        @printf("  window %3d days: %.4f\n", w, c)
    end
    @printf("rho_256 (forward window [t, t+255]): %.4f\n", coverage.rho_256)
    # Separately named cost statistic: the transition-driven exact
    # materialization boundary — NOT the rho_256 metric and NOT the exact
    # materialized row count (resource-budget demotion adds rows on top).
    println("boundary cost by window depth (transition-driven materialization boundary, NOT rho_256):")
    for (w, c) in zip(coverage.boundary_cost_windows, coverage.boundary_cost_coverage)
        @printf("  depth %3d rows: %.4f\n", w, c)
    end
    @printf("boundary_cost_256: %.4f\n", coverage.boundary_cost_256)
    # Activation (first observation of an asset): a DIFFERENT, monotone
    # statistic — reported under its own name, never as a mask run.
    @printf("activation transitions (first-observation days, monotone): %d in range, %d pre-range\n",
            coverage.activation_transitions, coverage.pre_range_activation_transitions)
    if !isempty(coverage.activation_positions)
        println("  activation positions (decision-day indices): ",
                join(coverage.activation_positions, ","))
    end
    @printf("partial rows (active space, 0<|obs|<|A|): %d in range, %d pre-range\n",
            coverage.partial_rows_active, coverage.pre_range_partial_rows_active)
    @printf("partial rows (universe-wide, 0<|obs|<N): %d in range, %d pre-range\n",
            coverage.partial_rows_universe, coverage.pre_range_partial_rows_universe)
    @printf("final active (|A| at last decision day; bar %d never read): %d of universe %d\n",
            size(signal, 1) - 1, coverage.final_active, coverage.universe)
    if out_path != ""
        open(out_path, "a") do io
            println(io, "# probe B")
            println(io, "decision_days=", coverage.decision_days)
            println(io, "transitions=", coverage.transitions)
            println(io, "pre_range_transitions=", coverage.pre_range_transitions)
            println(io, "unique_masks=", coverage.unique_masks)
            println(io, "stable_run_median=", coverage.stable_run_median)
            println(io, "stable_run_p90=", coverage.stable_run_p90)
            println(io, "stable_run_max=", coverage.stable_run_max)
            println(io, "stable_run_min=", coverage.stable_run_min)
            println(io, "rho_256=", coverage.rho_256)
            println(io, "activation_transitions=", coverage.activation_transitions)
            println(io, "pre_range_activation_transitions=", coverage.pre_range_activation_transitions)
            println(io, "observed_transitions=", coverage.observed_transitions)
            println(io, "partial_rows_active=", coverage.partial_rows_active)
            println(io, "partial_rows_universe=", coverage.partial_rows_universe)
            println(io, "boundary_cost_256=", coverage.boundary_cost_256)
            println(io, "prefix_transitions=", coverage.prefix_transitions)
            println(io, "prefix_unique_masks=", coverage.prefix_unique_masks)
            for (w, c) in zip(coverage.coverage_windows, coverage.coverage)
                println(io, "coverage_$w=", c)
            end
            for (w, c) in zip(coverage.boundary_cost_windows, coverage.boundary_cost_coverage)
                println(io, "boundary_cost_$w=", c)
            end
        end
    end
end

# ---------------- Probes A & C (shared single-day sequential run) ----------------
# A's old full-backtest composition needed the production probe plumbing M1
# removed; its admitted replacement is the dev-driven sequential probe — the
# SAME single-day run C performs. One initialize + ONE solve_current!; the
# collector records before the solve and collects after it. When A/C run
# they REPLACE the phase stages below, so the command still performs at
# most one prepare and one solve.
if admission.runs_ac
    println("\n=== probes A/C: shared single-day sequential run (incremental engine) ===")
    println("phase stages replaced by this run: it is the command's single authorized prepare+solve")
    t0 = time_ns()
    sequential = try
        DevProbes.sequential_ceiling_probe(b; from = from_date, F_folds = folds,
                                           ridge_alpha = ridge_alpha)
    catch e
        println(stderr, "sequential probe failed: $(sprint(showerror, e))")
        exit(1)
    end
    wall = (time_ns() - t0) * 1e-9
    rep = sequential.report
    @printf("  wall: %.2fs  decisions: 1  checkpoints: %d\n", wall, sequential.checkpoints)
    @printf("  solves=%d fast=%d fallback=%d\n", rep.solves, rep.fast, rep.fallback)
    if !isempty(rep.fallback_reasons)
        for (reason, count) in sort(collect(rep.fallback_reasons); by = first)
            @printf("    %-28s %6d\n", reason, count)
        end
    end
    @printf("  partial return rows: %d (prefix %d)\n", rep.async_rows, rep.prefix_async_rows)
    @printf("  core rebuilds: %d (prefix %d)\n", rep.rebuilds, rep.prefix_rebuilds)
    @printf("  coordinator advances: %d\n", rep.coordinator_advances)
    if out_path != ""
        open(out_path, "a") do io
            println(io, "# probes A/C (shared single-day sequential run)")
            println(io, "solves=", rep.solves)
            println(io, "fast=", rep.fast)
            println(io, "fallback=", rep.fallback)
            println(io, "async_rows=", rep.async_rows)
            println(io, "prefix_async_rows=", rep.prefix_async_rows)
            println(io, "rebuilds=", rep.rebuilds)
            println(io, "prefix_rebuilds=", rep.prefix_rebuilds)
            println(io, "coordinator_advances=", rep.coordinator_advances)
            println(io, "checkpoints=", sequential.checkpoints)
            println(io, "wall_seconds=", wall)
        end
    end
end

# ---------------- Phase: prepare (default) or prepare + one solve ----------------
# The DEFAULT phase is prepare-only: one prepare on the full 1:t_star
# prefix, NO solve. --phase solve is TWO explicit phases (prepare, then one
# solve consuming that same prepared input) — never described as a single
# phase. D (elapsed baseline) and E (OOF bucket split) are readings of that
# single authorized solve; there is no warmup and no repeats knob, and no
# cross-process prepared-input cache exists (solve consumes the prep built
# in this same command's prepare phase). When A/C ran above, this section
# prints the replacement notice instead of preparing a second time.
if phase === :prepare
    if admission.runs_ac
        println("\n=== phase: prepare — replaced by the A/C sequential run above (no second prepare) ===")
    else
        println("\n=== phase: prepare (one prepare, no solve) ===")
        println("stage lines (immediate, flushed) go to stderr; --from is not read here —")
        println("the input is the full prefix up to t_star (default: last decision bar).")
        signal = signal_prices(b)
        prepared = DevProbes.prepare_stage_probe(signal; t_star, F_folds = folds,
                                                 ridge_alpha, io = stderr)
        @printf("prepare: t*=%d  history_rows=%d  active_assets=%d\n",
                prepared.t_star, prepared.history_rows, prepared.active_assets)
        @printf("prepare elapsed: %.3fs\n", prepared.elapsed_seconds)
        if out_path != ""
            open(out_path, "a") do io
                println(io, "# phase prepare")
                println(io, "t_star=", prepared.t_star)
                println(io, "history_rows=", prepared.history_rows)
                println(io, "active_assets=", prepared.active_assets)
                println(io, "prepare_seconds=", prepared.elapsed_seconds)
            end
        end
    end
end

if phase === :solve
    # validate_probe_cli already rejected runs_ac with :solve, so this
    # branch never adds a second solve on top of an A/C run.
    println("\n=== phase: solve (TWO phases: prepare, then ONE solve) ===")
    println("This is NOT a single phase: the prepare phase runs first, then one solve")
    println("consumes that same prepared input. Stage lines (immediate, flushed) go to")
    println("stderr; --from is not read here — the input is the full 1:t_star prefix.")
    signal = signal_prices(b)
    prepared = DevProbes.prepare_stage_probe(signal; t_star, F_folds = folds,
                                             ridge_alpha, io = stderr)
    rep = DevProbes.solve_stage_probe(prepared.prep; t_star = prepared.t_star, mode = mode,
                                      F_folds = folds, ridge_alpha, io = stderr)
    # D reading: elapsed baseline of the single authorized solve.
    @printf("[D] frozen day t*=%d: %d history rows, %d active assets\n",
            rep.t_star, rep.history_rows, rep.active_assets)
    @printf("[D] prepare: %.3fs   solve: %.3fs   mode=%s   solves=1\n",
            prepared.elapsed_seconds, rep.elapsed_seconds, rep.mode)
    if rep.failed > 0
        println("[D] PARTIAL: the single solve FAILED; the baseline below is a failed-run report, not a success")
        for failure in rep.failures
            println("[D]   ", failure)
        end
    end
    if rep.observer_error !== nothing
        println("[D] OBSERVER FAILED (recorded separately from the model outcome): ",
                rep.observer_error)
    end
    println("[D] scope: EB + trace conditioning + independent per-fold OOF + predictive")
    println("         moments + fractional posterior (everything solve includes).")
    println("         EXCLUDES generate_scenarios_v1 and the Kelly solve.")
    println("         A prepared-solve TIMING BASELINE, not a mathematical floor and")
    println("         not a CPU ceiling: single-worker latency does not bound worker-")
    println("         parallel end-to-end throughput.")
    # E reading: bucket split of the SAME single solve.
    println("\n[E] OOF timing split (reading of the same single solve's buckets):")
    if rep.failed > 0
        println("[E] PARTIAL: the single solve FAILED; bucket values below are from the failed run")
    end
    buckets = collect(KTrader.TIMING_BUCKETS)
    for (bucket, seconds) in zip(buckets, rep.timing.seconds)
        @printf("[E]   %-14s %10.4fs\n", bucket, seconds)
    end
    oof_idx = [findfirst(==(b), buckets) for b in (:OOF_fit, :OOF_predict, :OOF_residual)]
    oof_seconds = sum(rep.timing.seconds[i] for i in oof_idx if isfinite(rep.timing.seconds[i]))
    solve_stage = (:eigen, :EB, :condition, :fracFFT, :OOF_fit, :OOF_predict, :OOF_residual)
    solve_stage_idx = [findfirst(==(b), buckets) for b in solve_stage]
    solve_seconds = sum(rep.timing.seconds[i] for i in solve_stage_idx if isfinite(rep.timing.seconds[i]))
    @printf("[E] OOF stage (OOF_fit+OOF_predict+OOF_residual, exclusive): %.4fs\n", oof_seconds)
    @printf("[E] solve stage buckets sum (see scope note): %.4fs\n", solve_seconds)
    @printf("[E] oof share of solve stage buckets: %.2f%%\n", 100 * (solve_seconds > 0 ? oof_seconds / solve_seconds : NaN))
    println("[E] bucket scope: OOF_fit is the whole fold-refit wall; eigen/EB/condition")
    println("    are attribution buckets of the FULL-data fit (condition also accumulates")
    println("    the decision-time projection). CURRENT wiring passes timing=nothing into")
    println("    fold refits, so the full-fit buckets and the three OOF buckets do NOT")
    println("    overlap and the solve-stage sum above is exclusive; if fold refits ever")
    println("    receive timing, eigen/EB/condition become nested inside OOF_fit and that")
    println("    sum double-counts.")
    if out_path != ""
        open(out_path, "a") do io
            println(io, "# phase solve (prepare + one solve)")
            println(io, "t_star=", rep.t_star)
            println(io, "history_rows=", rep.history_rows)
            println(io, "active_assets=", rep.active_assets)
            println(io, "mode=", rep.mode)
            println(io, "repeats=", rep.repeats)
            println(io, "succeeded=", rep.succeeded)
            println(io, "failed=", rep.failed)
            println(io, "prepare_seconds=", prepared.elapsed_seconds)
            println(io, "solve_seconds=", rep.elapsed_seconds)
            if rep.observer_error !== nothing
                println(io, "observer_error=\"", rep.observer_error, "\"")
            end
            for (bucket, seconds) in zip(buckets, rep.timing.seconds)
                println(io, "bucket_", bucket, "=", seconds)
            end
            println(io, "oof_seconds=", oof_seconds)
            println(io, "solve_stage_seconds=", solve_seconds)
            println(io, "includes_scenarios_kelly=", rep.includes_scenarios_kelly)
        end
    end
end
