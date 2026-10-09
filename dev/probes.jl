# dev/probes.jl — M1 dev namespace for the ceiling probes.
#
# SPEC 52.12/53/74 + M1: probes live OUTSIDE the production module. This file
# defines module DevProbes and calls the ONE production boundary through
# explicit KTrader.* references: prepare_reference / prepare_incremental /
# solve (the typed PreparedProblem boundary in src/prepare.jl). It expects
# KTrader to be loadable in the including scope (bin loads it from source;
# tests load it via `using KTrader`). Nothing here is included into the
# KTrader namespace. The old oof_shadow factory, its typed error and the
# per-fold identity machinery are gone with the production ports; no second
# solver, no global profiler. Old NamedTuple artifacts are converted ONCE on
# the dev side via the PreparedProblem constructor (its dimension checks
# stay); no production compatibility layer.
#
# Probe A (full-backtest fallback composition) needed the production probe
# plumbing that M1 removes; its admitted replacement is the dev-driven
# sequential probe C below (K==1, collector semantics kept: record before
# solve, collect after, coordinator accounting check).

module DevProbes

using ..KTrader
import ..KTrader: TIMING_BUCKETS, DecisionTiming, Bars, BANDS, WARMUP,
                  signal_prices, PriceHistoryCache, build_prefix_ruler_stats,
                  initialize_inference, solve_current!, prepare_reference,
                  PreparedProblem
using LinearAlgebra, Statistics, Printf, Dates

export CeilingProbeCollector, record_probe_checkpoints!, collect_probe_window!,
       finish_ceiling_probe!, print_probe_report, PROBE_COVERAGE_WINDOWS,
       mask_transition_coverage, prepare_stage_probe, solve_stage_probe,
       solve_floor_probe, oof_timing_split_probe, sequential_ceiling_probe

# ---------------- collector (dev-owned; A/C semantics) ----------------
mutable struct CeilingProbeCollector
    solves::Int
    fast::Int
    fallback::Int
    fallback_reasons::Dict{Symbol,Int}
    async_rows::Int
    rebuilds::Int
    coordinator_advances::Int
    coordinator_async_rows::Int
    coordinator_rebuilds::Int
    prefix_async_rows::Int
    prefix_rebuilds::Int
    windows::Int
    checkpoints::Int
    checkpoint_snapshot::Vector{Tuple{Int,Int}}
end
CeilingProbeCollector()=CeilingProbeCollector(0,0,0,Dict{Symbol,Int}(),0,0,0,0,0,0,0,0,0,Tuple{Int,Int}[])

function record_probe_checkpoints!(probe::CeilingProbeCollector,states)
    empty!(probe.checkpoint_snapshot)
    for state in states
        push!(probe.checkpoint_snapshot,
              (get(state.counters,:async_rows,0),get(state.counters,:rebuilds,0)))
    end
    if probe.windows==0 && !isempty(probe.checkpoint_snapshot)
        probe.prefix_async_rows=probe.checkpoint_snapshot[1][1]
        probe.prefix_rebuilds=probe.checkpoint_snapshot[1][2]
    end
    probe.windows+=1
    probe.checkpoints+=length(states)
    nothing
end

function collect_probe_window!(probe::CeilingProbeCollector,states)
    for (block,state) in enumerate(states)
        counters=state.counters
        probe.solves+=get(counters,:solves,0)
        probe.fast+=get(counters,:fast,0)
        probe.fallback+=get(counters,:fallback,0)
        for (reason,count) in state.fallback_reasons
            probe.fallback_reasons[reason]=get(probe.fallback_reasons,reason,0)+count
        end
        snapshot=block<=length(probe.checkpoint_snapshot) ? probe.checkpoint_snapshot[block] : (0,0)
        probe.async_rows+=get(counters,:async_rows,0)-snapshot[1]
        probe.rebuilds+=get(counters,:rebuilds,0)-snapshot[2]
    end
    nothing
end

function finish_ceiling_probe!(probe::CeilingProbeCollector,coordinator)
    probe.coordinator_advances=get(coordinator.counters,:advances,0)
    probe.coordinator_async_rows=get(coordinator.counters,:async_rows,0)
    probe.coordinator_rebuilds=get(coordinator.counters,:rebuilds,0)
    isempty(probe.checkpoint_snapshot) && error("finish_ceiling_probe! called before any recorded window")
    final_snapshot=probe.checkpoint_snapshot[length(probe.checkpoint_snapshot)]
    (probe.coordinator_async_rows==final_snapshot[1] && probe.coordinator_rebuilds==final_snapshot[2]) ||
        error("probe accounting mismatch: coordinator (async_rows=$(probe.coordinator_async_rows), rebuilds=$(probe.coordinator_rebuilds)) != final checkpoint snapshot $final_snapshot")
    probe.async_rows+=probe.coordinator_async_rows-probe.prefix_async_rows
    probe.rebuilds+=probe.coordinator_rebuilds-probe.prefix_rebuilds
    (solves=probe.solves,fast=probe.fast,fallback=probe.fallback,
     fallback_reasons=copy(probe.fallback_reasons),
     async_rows=probe.async_rows,rebuilds=probe.rebuilds,
     prefix_async_rows=probe.prefix_async_rows,prefix_rebuilds=probe.prefix_rebuilds,
     windows=probe.windows,checkpoints=probe.checkpoints,
     coordinator_advances=probe.coordinator_advances)
end

function print_probe_report(probe::CeilingProbeCollector;decisions::Union{Int,Nothing}=nothing,io::IO=stdout)
    total=decisions===nothing ? probe.solves : decisions
    if probe.solves>0
        @printf(io,"  fallback: %d/%d solves (%.2f%%) [fast=%d (%.1f%%)]\n",
                probe.fallback,probe.solves,100*probe.fallback/probe.solves,
                probe.fast,100*probe.fast/probe.solves)
        for (reason,count) in sort(collect(probe.fallback_reasons);by=first)
            @printf(io,"    %-28s %6d (%.2f%% of solves)\n",reason,count,100*count/probe.solves)
        end
    else
        println(io,"  fallback: no solves recorded (run failed before any decision)")
    end
    @printf(io,"  partial return rows (decision span, universe-wide): %d\n",probe.async_rows)
    probe.prefix_async_rows>0 && @printf(io,"  partial return rows (prefix/warmup): %d\n",probe.prefix_async_rows)
    @printf(io,"  core rebuilds (decision span): %d\n",probe.rebuilds)
    probe.prefix_rebuilds>0 && @printf(io,"  core rebuilds (prefix/warmup): %d\n",probe.prefix_rebuilds)
    @printf(io,"  coordinator advances: %d\n",probe.coordinator_advances)
    if total>0
        @printf(io,"  rebuilds/decision: %.3f  partial-rows/decision: %.3f\n",
                probe.rebuilds/total,probe.async_rows/total)
    end
    return nothing
end

# ---------------- probe B: mask transition coverage (engine semantics) --------
# The engine's observation mask (src/incremental.jl append_return!) is defined
# on RETURN rows: mask[k,j] = isfinite(log signal[k+1,j] - log signal[k,j]) —
# TWO real price endpoints, exactly the ragged field the regime runs split.
# Day axis: k = 1..T-2 return rows (price bar t = k+1); a decision day t in
# from..T-1 is return row t-1, K = T - from. THE LAST BAR T IS NEVER READ:
# masks exist only for bars 2..T-1, so a final-bar mask change or IPO cannot
# move any primary statistic. Initial-mask convention: k=1's mask is the
# first return's mask — no earlier return exists, so the first prefix run
# always opens at k=1.
# PRIMARY statistics (transitions, run lengths, stable_run_*, unique_masks,
# mask_fingerprints) cover the DECISION RANGE [from, T-1] in bars = return
# rows [from-1, T-2]; sum(run_lengths) == decision_days identically. The
# whole prefix (return rows 1..T-2) is used only to DETECT pre-range
# transitions — their forward-window tails still count toward coverage —
# and is reported separately under prefix_* names.
# A transition is k>=2 with mask(k) != mask(k-1); its effective day is bar
# k+1. PRIMARY coverage (user-defined metric): a transition on price bar t
# opens the CLOSED FORWARD window [t, t+w-1] on the price-bar day axis — w
# days starting at the transition day itself. On the return-row axis this
# is the equivalent index change [s, s+w-1] with s = t-1: pure index
# translation, no day added or dropped, no target-spill shift. rho_256 is
# this metric at w=256; the window list keeps the band scales and always
# includes 256.
# A SEPARATELY NAMED cost statistic (boundary_cost_*) measures the
# transition-driven exact-materialization boundary: transition s forces the
# materialized path on training row k iff the feature window [k-w+1, k]
# straddles s (k-w+1 < s <= k) or the target row k+1 opens the new run
# (s == k+1) — together k in [max(1, s-1), min(Reff, s+w-2)], a window DEPTH
# of w rows (2*tau per band; 256 = 2*128, the deepest index
# fill_design_matrix! touches). That boundary is NOT the rho_256 metric and
# NOT the exact materialized row count: the real count also includes rows
# demoted by the resource budget (budget_allows! demotion). Activation
# (first observation of an asset) is reported under activation_* names: it
# is a DIFFERENT, monotone membership event and must never be read as a
# mask-run statistic.
const PROBE_COVERAGE_WINDOWS = vcat(collect(Int, BANDS), 256)
const PROBE_BOUNDARY_COST_WINDOWS = 2 .* BANDS

function mask_transition_coverage(signal::AbstractMatrix{Float64}; from::Integer)
    T, N = size(signal)
    2 <= from <= T-1 ||
        throw(ArgumentError("from must be a decision-day bar index in 2:T-1"))
    K = T - from
    Reff = T - 2                   # last return row read (bar T-1); bar T never read
    mask = falses(Reff, N)
    for k in 1:Reff, j in 1:N
        mask[k, j] = isfinite(signal[k+1, j]) && isfinite(signal[k, j])
    end
    trans = Int[]                   # full-prefix transitions, s in [2, T-2]
    prefix_run_lengths = Int[]
    runlen = 1
    for k in 2:Reff
        if view(mask, k, :) != view(mask, k-1, :)
            push!(trans, k)
            push!(prefix_run_lengths, runlen)
            runlen = 1
        else
            runlen += 1
        end
    end
    push!(prefix_run_lengths, runlen)   # close the final prefix run
    starts = vcat(1, trans)              # prefix run starts
    ends = vcat(trans .- 1, Reff)        # prefix run ends
    in_range = Int[]                # decision-day indices (bar s+1 - from + 1)
    pre_trans = 0                   # transitions strictly before the range
    for s in trans
        if s >= from - 1            # the new mask first governs bar s+1 >= from
            push!(in_range, s - from + 2)
        else
            pre_trans += 1
        end
    end
    # PRIMARY run structure: the decision window [from-1, T-2] in return
    # rows, clipped onto the prefix runs; sum(run_lengths) == K identically.
    run_lengths = Int[]
    for i in eachindex(starts)
        lo = max(starts[i], from - 1)
        hi = min(ends[i], Reff)
        hi >= lo && push!(run_lengths, hi - lo + 1)
    end
    # PRIMARY unique masks: masks appearing inside the decision window.
    mask_fingerprints = Vector{Vector{Bool}}()
    seen = Set{Vector{Bool}}()
    for i in eachindex(starts)
        (ends[i] >= from - 1 && starts[i] <= Reff) || continue
        m = Vector{Bool}(mask[starts[i], :])
        m in seen || (push!(mask_fingerprints, m); push!(seen, m))
    end
    # WHOLE-PREFIX statistics under their own name (pre-range detection and
    # audit); the prefix transition list also drives the windows below.
    prefix_mask_fingerprints = Vector{Vector{Bool}}()
    seenp = Set{Vector{Bool}}()
    for i in eachindex(starts)
        m = Vector{Bool}(mask[starts[i], :])
        m in seenp || (push!(prefix_mask_fingerprints, m); push!(seenp, m))
    end
    # PRIMARY coverage: user-defined forward window. Transition s (return
    # row) is the transition day t = s+1 (price bar); the window [t, t+w-1]
    # in bars = [s, s+w-1] in return rows = decision-day indices
    # [s-from+2, s-from+w+1] (decision day d is bar from+d-1, return row
    # from+d-2). Pre-range transitions are in the loop: their window tails
    # still reach into the range.
    coverage = zeros(Float64, length(PROBE_COVERAGE_WINDOWS))
    for (wi, w) in enumerate(PROBE_COVERAGE_WINDOWS)
        covered = falses(K)
        for s in trans
            lo = max(1, s - from + 2)
            hi = min(K, s - from + w + 1)
            lo <= hi && (covered[lo:hi] .= true)
        end
        coverage[wi] = count(covered) / K
    end
    # SEPARATELY NAMED cost boundary (NOT rho_256, NOT the exact materialized
    # row count — resource-budget demotion adds rows on top of this):
    # k in [max(1, s-1), min(Reff, s+w-2)] in return rows = decision-day
    # indices [s-from+1, s+w-from]. One day EARLIER than the user window
    # (the target row whose target opens the new run) — that is the cost
    # semantics, kept distinct from the coverage metric above.
    boundary_cost = zeros(Float64, length(PROBE_BOUNDARY_COST_WINDOWS))
    for (wi, w) in enumerate(PROBE_BOUNDARY_COST_WINDOWS)
        covered = falses(K)
        for s in trans
            lo = max(1, s - from + 1)
            hi = min(K, s + w - from)
            lo <= hi && (covered[lo:hi] .= true)
        end
        boundary_cost[wi] = count(covered) / K
    end
    # activation (first observation) under its OWN name; the same loop feeds
    # active_count for the partial-row conventions. Bar T is not read here
    # either: the loop stops at the last decision day T-1.
    cumulative = falses(N)
    nactive = 0
    active_count = zeros(Int, T)
    activation_positions = Int[]    # decision-day indices
    pre_activation = 0
    activation_fingerprints = Vector{Vector{Int}}()
    for t in 2:T-1
        changed = false
        for j in 1:N
            if mask[t-1, j] && !cumulative[j]
                cumulative[j] = true
                nactive += 1
                changed = true
            end
        end
        active_count[t] = nactive
        if t == from
            push!(activation_fingerprints, findall(cumulative))
        elseif from < t && changed
            push!(activation_fingerprints, findall(cumulative))
        end
        changed || continue
        if t < from
            pre_activation += 1
        else
            push!(activation_positions, t - from + 1)
        end
    end
    obs_count = zeros(Int, T)       # bar t: assets with a finite return pair
    for t in 2:T-1
        obs_count[t] = count(view(mask, t-1, :))
    end
    partial_active = 0
    partial_universe = 0
    for t in from:T-1
        0 < obs_count[t] < active_count[t] && (partial_active += 1)
        0 < obs_count[t] < N && (partial_universe += 1)
    end
    pre_partial_active = 0
    pre_partial_universe = 0
    for t in 2:from-1
        0 < obs_count[t] < active_count[t] && (pre_partial_active += 1)
        0 < obs_count[t] < N && (pre_partial_universe += 1)
    end
    (; decision_days = K,
       transitions = length(in_range),
       transition_positions = in_range,
       pre_range_transitions = pre_trans,
       transition_rate = length(in_range) / K,
       coverage_windows = copy(PROBE_COVERAGE_WINDOWS),
       coverage = coverage,
       rho_256 = coverage[findfirst(==(256), PROBE_COVERAGE_WINDOWS)],
       boundary_cost_windows = copy(PROBE_BOUNDARY_COST_WINDOWS),
       boundary_cost_coverage = boundary_cost,
       boundary_cost_256 = boundary_cost[findfirst(==(256), PROBE_BOUNDARY_COST_WINDOWS)],
       unique_masks = length(mask_fingerprints),
       mask_fingerprints = mask_fingerprints,
       run_lengths = run_lengths,
       stable_run_median = median(run_lengths),
       stable_run_p90 = quantile(run_lengths, 0.9),
       stable_run_max = maximum(run_lengths),
       stable_run_min = minimum(run_lengths),
       prefix_transitions = length(trans),
       prefix_unique_masks = length(prefix_mask_fingerprints),
       prefix_mask_fingerprints = prefix_mask_fingerprints,
       prefix_run_lengths = prefix_run_lengths,
       prefix_stable_run_median = median(prefix_run_lengths),
       prefix_stable_run_p90 = quantile(prefix_run_lengths, 0.9),
       prefix_stable_run_max = maximum(prefix_run_lengths),
       prefix_stable_run_min = minimum(prefix_run_lengths),
       observed_transitions = length(in_range),
       observed_transition_positions = in_range,
       activation_transitions = length(activation_positions),
       activation_positions = activation_positions,
       pre_range_activation_transitions = pre_activation,
       activation_rate = length(activation_positions) / K,
       activation_fingerprints = activation_fingerprints,
       partial_rows_active = partial_active,
       partial_rows_universe = partial_universe,
       pre_range_partial_rows_active = pre_partial_active,
       pre_range_partial_rows_universe = pre_partial_universe,
       final_active = active_count[T-1],
       universe = N)
end

# ---------------- stage admission & live logging ----------------
round3(x)=round(x;digits=3)
function stage_line(io::IO,stage::Symbol,boundary::Symbol;pairs...)
    print(io,"[stage:",stage,"][",boundary,"]")
    for (key,value) in pairs
        print(io," ",key,"=",value)
    end
    println(io)
    flush(io)
    nothing
end
function stage_bucket_sink(io::IO,phase::Symbol;pairs...)
    fields=join(["$k=$v" for (k,v) in pairs]," ")
    return function(event::Symbol,bucket::Symbol,stamp::Integer)
        println(io,"[stage:",phase,"][bucket:",event,"] ",bucket," ",fields)
        flush(io)
    end
end

# -------- probe D, stage 1: prepare only (thin over prepare_reference) -------
function prepare_stage_probe(signal::AbstractMatrix{Float64};t_star=nothing,
                             F_folds::Integer=3,ridge_alpha=nothing,io::IO=stderr)
    T,N=size(signal)
    t=t_star===nothing ? T-1 : Int(t_star)
    2<=t<=T-1 || throw(ArgumentError("t_star must be a decision-day bar index in 2:T-1"))
    stage_line(io,:prep,:start,t_star=t,history_rows=t,universe=N,
               folds=Int(F_folds),mode="-",run="1/1")
    timing=DecisionTiming()
    timing.sink=stage_bucket_sink(io,:prep;t_star=t)
    started=time_ns()
    prep=prepare_reference(view(signal,1:t,:);ridge_alpha,F_folds,timing)
    elapsed=(time_ns()-started)*1e-9
    i_prep=findfirst(==(:prep),TIMING_BUCKETS)
    i_basis=findfirst(==(:basis),TIMING_BUCKETS)
    i_gram=findfirst(==(:gram),TIMING_BUCKETS)
    stage_line(io,:prep,:end,t_star=t,history_rows=t,universe=N,
               active_assets=length(prep.active_idx),
               elapsed_s=round3(elapsed),
               prep_s=round3(timing.seconds[i_prep]),
               basis_s=round3(timing.seconds[i_basis]),
               gram_s=round3(timing.seconds[i_gram]))
    (;prep,timing,t_star=t,history_rows=t,elapsed_seconds=elapsed,
      active_assets=length(prep.active_idx))
end

# -------- probe D, stage 2: exactly ONE solve (thin over solve) --------------
function solve_stage_probe(prep;t_star=nothing,mode::Symbol=:cold,warmup::Bool=false,
                           F_folds::Integer=3,ridge_alpha=nothing,io::IO=stderr)
    mode in (:cold,:warm) ||
        throw(ArgumentError("mode must be a single :cold or :warm (dual-mode runs are rejected)"))
    warmup && throw(ArgumentError("warmup=true is rejected before any data is touched: a warmup is a second full solve, and the single-solve cost is not yet verified (2026-10-07 gate). Re-admission requires a measured single-day solve cost plus an explicit decision; no override exists"))
    t=t_star===nothing ? prep.T : Int(t_star)
    t==prep.T || throw(ArgumentError("t_star ($t) does not match the prepared prefix (prep.T=$(prep.T)); the solve must consume the prepared input unchanged"))
    length(prep.stats.ranges)==Int(F_folds) ||
        throw(ArgumentError("F_folds ($F_folds) does not match the prepared fold ranges ($(length(prep.stats.ranges))); pass the same folds used to build prep"))
    warm=fill((1.0,1.0),Int(F_folds)+1)
    stage_line(io,:solve,:start,t_star=t,history_rows=prep.T,mode=mode,run="1/1")
    timing=DecisionTiming()
    timing.sink=stage_bucket_sink(io,:solve;t_star=t,mode=mode)
    started=time_ns()
    failure=nothing
    model=nothing
    try
        model=KTrader.solve(prep;ridge_alpha,F_folds,
                            alpha_initial=mode===:warm ? warm : nothing,
                            timing)
    catch error
        failure=sprint(showerror,error)
    end
    elapsed=(time_ns()-started)*1e-9
    bucket_str=join(["$(b)=$(round3(timing.seconds[findfirst(==(b),TIMING_BUCKETS)]))"
                     for b in (:eigen,:EB,:condition,:fracFFT,
                               :OOF_fit,:OOF_predict,:OOF_residual)],",")
    stage_line(io,:solve,:end,t_star=t,history_rows=prep.T,mode=mode,run="1/1",
               elapsed_s=round3(elapsed),failed=failure===nothing ? "-" : "yes",
               buckets=bucket_str)
    (;t_star=t,history_rows=prep.T,active_assets=length(prep.active_idx),
      repeats=1,mode=mode,succeeded=failure===nothing ? 1 : 0,
      failed=failure===nothing ? 0 : 1,
      failures=failure===nothing ? String[] : ["run 1: $failure"],
      observer_error=timing.sink_error,
      model=model,
      elapsed_seconds=elapsed,
      seconds=[failure===nothing ? elapsed : NaN],
      timing,includes_scenarios_kelly=false)
end

# ---------------- compatibility wrappers (D and E) ---------------------------
function solve_floor_probe(signal::AbstractMatrix{Float64};t_star=nothing,F_folds::Integer=3,
                           ridge_alpha=nothing,repeats::Integer=1,mode::Symbol=:cold,
                           warmup::Bool=false,io::IO=stderr)
    repeats==1 || throw(ArgumentError("repeats must be 1 (got $repeats): single-solve admission; no override is provided"))
    mode in (:cold,:warm) || throw(ArgumentError("mode must be a single :cold or :warm (dual-mode runs are rejected)"))
    warmup && throw(ArgumentError("warmup=true is rejected before any data is touched (2026-10-07 gate)"))
    prepared=prepare_stage_probe(signal;t_star,F_folds,ridge_alpha,io)
    rep=solve_stage_probe(prepared.prep;t_star=prepared.t_star,mode,warmup,
                          F_folds,ridge_alpha,io)
    s=rep.seconds[1]
    (;t_star=rep.t_star,active_assets=rep.active_assets,
      repeats=1,mode=mode,
      succeeded=rep.succeeded,failed=rep.failed,failures=rep.failures,
      warmup_error=nothing,
      mean_seconds=s,median_seconds=s,min_seconds=s,max_seconds=s,
      seconds=rep.seconds,includes_scenarios_kelly=false,timing=rep.timing)
end

function oof_timing_split_probe(signal::AbstractMatrix{Float64};t_star=nothing,F_folds::Integer=3,
                                ridge_alpha=nothing,repeats::Integer=1,warmup::Bool=false,
                                io::IO=stderr)
    repeats==1 || throw(ArgumentError("repeats must be 1 (got $repeats): single-solve admission; no override"))
    prepared=prepare_stage_probe(signal;t_star,F_folds,ridge_alpha,io)
    rep=solve_stage_probe(prepared.prep;t_star=prepared.t_star,mode=:warm,
                          warmup,F_folds,ridge_alpha,io)
    buckets=TIMING_BUCKETS
    bucket_mean=copy(rep.timing.seconds)
    oof_idx=[findfirst(==(b),buckets) for b in (:OOF_fit,:OOF_predict,:OOF_residual)]
    oof_seconds=sum(bucket_mean[i] for i in oof_idx if isfinite(bucket_mean[i]))
    solve_stage=(:eigen,:EB,:condition,:fracFFT,:OOF_fit,:OOF_predict,:OOF_residual)
    solve_seconds=sum(bucket_mean[findfirst(==(b),buckets)] for b in solve_stage if isfinite(bucket_mean[findfirst(==(b),buckets)]))
    (;t_star=rep.t_star,active_assets=rep.active_assets,
      repeats=1,succeeded=rep.succeeded,failed=rep.failed,failures=rep.failures,
      warmup_error=nothing,observer_error=rep.observer_error,
      timing=rep.timing,
      buckets=collect(buckets),
      bucket_mean_seconds=bucket_mean,bucket_median_seconds=copy(bucket_mean),
      oof_seconds=oof_seconds,solve_stage_seconds=solve_seconds,
      oof_share=solve_seconds>0 ? oof_seconds/solve_seconds : NaN,
      includes_scenarios_kelly=false)
end

# -------- probe C: dev-driven single-day sequential --------------------------
function sequential_ceiling_probe(b::Bars;from::Date,ridge_alpha=nothing,
                                  F_folds::Integer=3,
                                  probe::CeilingProbeCollector=CeilingProbeCollector())
    i0=findfirst(>=(from),b.dates)
    i0===nothing && error("no decision day on/after $from")
    K=size(b.adj,1)-i0
    K==1 || throw(ArgumentError("multi-day sequential probe rejected: $K decision days imply $K solves, exceeding the single-command 60s budget until single-day correctness and cost are verified; no override is provided (use prepare_stage_probe/solve_stage_probe for one-shot single-day diagnostics)"))
    signal=signal_prices(b)
    cache=PriceHistoryCache(signal)
    ruler_stats=build_prefix_ruler_stats(cache.log_prices,cache.first_price)
    state=initialize_inference(view(signal,1:i0,:);ridge_alpha,F_folds,
                               history_cache=cache,ruler_stats)
    record_probe_checkpoints!(probe,[state])
    model=solve_current!(state)
    collect_probe_window!(probe,[state])
    report=finish_ceiling_probe!(probe,state)
    print_probe_report(probe;decisions=K)
    (;report,model,checkpoints=probe.checkpoints)
end


# ---------------- CLI boundary (shared with bin/ceiling_probes.jl) ------------
# Pure parse/admission core: the bin script and the tests exercise the SAME
# decision path (the supported entrance). Tests call these functions
# directly; no command is ever spawned from them.
const PROBE_CLI_FLAGS = ("--from", "--t-star", "--phase", "--mode", "--folds",
                         "--ridge-alpha", "--probes", "--out")
const PROBE_CLI_DEFAULTS = Dict{String,String}(
    "from" => "2018-01-01", "t-star" => "", "phase" => "prepare", "mode" => "cold",
    "folds" => "3", "ridge-alpha" => "", "probes" => "B", "out" => "")
const PROBE_CLI_KNOWN_PROBES = Set{Symbol}((:A, :B, :C, :D, :E))

"""Parse probe-CLI arguments. Fail-closed BEFORE anything is loaded or run:
unknown flags, duplicates, missing values and empty values all throw."""
function parse_probe_cli(args)
    parsed = copy(PROBE_CLI_DEFAULTS)
    seen = Set{String}()
    i = 1
    while i <= length(args)
        flag = args[i]
        flag in PROBE_CLI_FLAGS || throw(ArgumentError(
            "unknown argument $flag (supported: $(join(PROBE_CLI_FLAGS, " "))"))
        flag in seen && throw(ArgumentError("duplicate flag $flag"))
        push!(seen, flag)
        i + 1 <= length(args) || throw(ArgumentError("missing value for $flag"))
        value = args[i+1]
        isempty(strip(value)) && throw(ArgumentError("empty value for $flag"))
        parsed[flag[3:end]] = value
        i += 2
    end
    parsed
end

"""Admission for the probe CLI given the decision-day count K (bars already
loaded; nothing has been solved yet). Enforces the single-authorized-solve
rule: any legal command performs at most ONE prepare and ONE solve. B is
pure statistics (zero solves). D/E are readings of the phase stages. A/C run
the single-day sequential probe — their own one prepare+solve — and REPLACE
the phase stages. Every conflicting combination is rejected fail-closed."""
function validate_probe_cli(parsed; K::Integer)
    phase = Symbol(parsed["phase"])
    phase in (:prepare, :solve) ||
        throw(ArgumentError("--phase must be prepare or solve"))
    mode = Symbol(parsed["mode"])
    mode in (:cold, :warm) ||
        throw(ArgumentError("--mode must be cold or warm (one mode per run; dual-mode loops are rejected)"))
    probe_list = Symbol.(strip.(split(parsed["probes"], ",")))
    for p in probe_list
        p in PROBE_CLI_KNOWN_PROBES ||
            throw(ArgumentError("unknown probe $p (supported: A,B,C,D,E)"))
    end
    isempty(probe_list) &&
        throw(ArgumentError("--probes must name at least one probe"))
    has(p) = p in probe_list
    runs_ac = has(:A) || has(:C)
    if runs_ac
        K == 1 || throw(ArgumentError(
            "probes A/C requested with K=$K decision days — a multi-date run implies " *
            "$K solves, exceeding the single-command 60s budget until single-day " *
            "correctness and cost are verified; no override exists (set --from to " *
            "the last decision date for a single-day run)"))
        phase === :solve && throw(ArgumentError(
            "probes A/C already perform the single authorized prepare+solve of this " *
            "command; --phase solve would add a second solve"))
        (has(:D) || has(:E)) && throw(ArgumentError(
            "probes A/C replace the phase stages and produce no D/E stage readings; " *
            "request D/E via --phase prepare|solve without A/C"))
        !isempty(parsed["t-star"]) && throw(ArgumentError(
            "probes A/C derive their decision day from --from; --t-star is not read " *
            "by them"))
    elseif has(:E) && phase !== :solve
        throw(ArgumentError(
            "probe E is a reading of the single authorized solve; request it with " *
            "--phase solve"))
    end
    (; phase, mode, probe_list, runs_ac)
end

end # module