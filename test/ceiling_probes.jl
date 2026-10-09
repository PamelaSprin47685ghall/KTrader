using Test, Random, LinearAlgebra, Statistics, Dates
using KTrader
# M1: probes live in the dev namespace; KTrader no longer defines or
# exports any probe symbol. Load DevProbes explicitly; qualified probe calls
# below are rerouted to it and BARE probe names (CeilingProbeCollector(),
# solve_floor_probe) resolve via `using .DevProbes`. Core math stays KTrader.
isdefined(@__MODULE__, :DevProbes) || include(joinpath(@__DIR__, "..", "dev", "probes.jl"))
using .DevProbes

@testset "ceiling probes" begin
    # ---------- Collector accounting: fast unit tests, no backtest ----------
    # A minimal stand-in for an inference state: the collector only reads
    # .counters and .fallback_reasons, exactly as backtest_v1's states do.
    mkstate(async, rebuilds; advances = async + rebuilds) =
        (counters = Dict(:advances => advances, :async_rows => async, :rebuilds => rebuilds,
                         :solves => 0, :fast => 0, :fallback => 0),
         fallback_reasons = Dict{Symbol,Int}())

    # Single continuous block (probe C's topology): one record before block
    # work, one collect after join. The first snapshot is the prefix/warmup
    # baseline; the block's own delta is counted exactly once.
    probe1 = CeilingProbeCollector()
    s1 = mkstate(2, 1)
    DevProbes.record_probe_checkpoints!(probe1, [s1])
    @test probe1.windows == 1
    @test probe1.checkpoints == 1
    @test probe1.prefix_async_rows == 2
    @test probe1.prefix_rebuilds == 1
    s1.counters[:solves] += 5
    s1.counters[:fast] += 5
    s1.counters[:async_rows] += 3
    s1.fallback_reasons[:x] = 1
    DevProbes.collect_probe_window!(probe1, [s1])
    coordinator = (counters = Dict(:advances => 0, :async_rows => 2, :rebuilds => 1),
                   fallback_reasons = Dict{Symbol,Int}())
    report = DevProbes.finish_ceiling_probe!(probe1, coordinator)
    @test report.solves == 5
    @test report.fast == 5
    @test report.fallback == 0
    @test report.fallback_reasons[:x] == 1
    @test report.async_rows == 3          # block decision-span delta only
    @test report.prefix_async_rows == 2   # warmup rows reported separately
    @test report.rebuilds == 0
    @test report.coordinator_advances == 0

    # Multiple blocks in one window (the current backtest_v1 topology): each
    # state's delta counted exactly once; the coordinator's decision-span
    # contribution is its final counter minus the first snapshot.
    probe2 = CeilingProbeCollector()
    b1 = mkstate(2, 1)
    b2 = mkstate(4, 2; advances = 7)
    DevProbes.record_probe_checkpoints!(probe2, [b1, b2])
    @test probe2.checkpoints == 2
    b1.counters[:solves] += 5
    b1.counters[:async_rows] += 3
    b2.counters[:solves] += 5
    DevProbes.collect_probe_window!(probe2, [b1, b2])
    coord2 = (counters = Dict(:advances => 7, :async_rows => 4, :rebuilds => 2),
              fallback_reasons = Dict{Symbol,Int}())
    rep2 = DevProbes.finish_ceiling_probe!(probe2, coord2)
    @test rep2.solves == 10
    @test rep2.async_rows == 5   # 3 (block delta) + (4 - 2) coordinator span
    @test rep2.rebuilds == 1     # 0 (block delta) + (2 - 1) coordinator span
    @test rep2.coordinator_advances == 7
    @test rep2.prefix_async_rows == 2
    @test rep2.prefix_rebuilds == 1

    # Accounting guard: the coordinator is the single writer up to the last
    # checkpoint, so a final counter disagreeing with the last snapshot is a
    # loud failure, not a silent report.
    probe_bad = CeilingProbeCollector()
    sbad = mkstate(2, 1)
    DevProbes.record_probe_checkpoints!(probe_bad, [sbad])
    sbad.counters[:solves] += 1
    DevProbes.collect_probe_window!(probe_bad, [sbad])
    coord_bad = (counters = Dict(:advances => 0, :async_rows => 99, :rebuilds => 1),
                 fallback_reasons = Dict{Symbol,Int}())
    @test_throws ErrorException DevProbes.finish_ceiling_probe!(probe_bad, coord_bad)

    # finish before any recorded window fails loudly.
    @test_throws ErrorException DevProbes.finish_ceiling_probe!(CeilingProbeCollector(), mkstate(0, 0))

    # ---------- Probe B: engine daily observed-return mask semantics ----------
    # The mask that decides the regime boundaries of e_t(d) is defined on
    # RETURN rows: finite(log s[k+1]-log s[k]), two real price endpoints.
    # Day axis: k = 1..T-1 return rows (bar t = k+1); decision day t is
    # return row t-1. Asset 3 prints its first bar at row 101, so its mask
    # turns on at k=101 (both endpoints real); the activation DAY is 102.
    # Expected values below are derived by hand on that axis.
    signal = fill(NaN, 400, 3)
    signal[:, 1] .= 100 .+ cumsum(0.01 .* randn(MersenneTwister(1), 400))
    signal[:, 2] .= 50 .+ cumsum(0.01 .* randn(MersenneTwister(2), 400))
    signal[101:end, 3] .= 30 .+ cumsum(0.01 .* randn(MersenneTwister(3), 300))
    cov = mask_transition_coverage(signal; from = 50)
    @test cov.decision_days == 350            # bars 50..399
    @test cov.transitions == 1                # mask changes at k=101 (day 102)
    @test cov.transition_positions == [53]    # day 102 - 50 + 1
    @test cov.pre_range_transitions == 0      # no full-prefix mask transition before bar 49
    @test cov.transition_rate ≈ 1 / 350
    @test cov.observed_transitions == 1       # same mask semantics (compat name)
    @test cov.observed_transition_positions == [53]
    @test cov.unique_masks == 2               # [TT F] and [TT T]
    @test cov.final_active == 3
    @test cov.universe == 3
    @test cov.mask_fingerprints == [[true, true, false], [true, true, true]]
    # PRIMARY runs: the decision window [50..399] in bars = return rows
    # [49..398], cut by the k=101 transition: [52, 298]; sum == K.
    @test cov.run_lengths == [52, 298]
    @test sum(cov.run_lengths) == cov.decision_days
    @test cov.stable_run_min == 52
    @test cov.stable_run_max == 298
    @test cov.stable_run_median == 175.0
    @test cov.stable_run_p90 ≈ 273.4 atol = 1e-12
    # Whole-prefix runs under their own name: [100, 299] over rows 1..398.
    # Prefix runs over return rows 1..T-2 (T-2 rows total): first asset
    # price 101 -> first return bar 102, so run lengths are [100, 298]
    # (298 = 398 - 100), NOT [100, 299].
    @test cov.prefix_run_lengths == [100, 298]
    @test cov.prefix_stable_run_max == 298
    @test cov.prefix_unique_masks == 2
    @test cov.coverage_windows == vcat(collect(Int, BANDS), 256)  # forward window DAYS, 256 included
    # User metric: transition day 102 opens [102, 102+w-1] — decision days
    # [53, 52+w]; fully inside the range at every scale.
    for (i, w) in enumerate(cov.coverage_windows)
        @test cov.coverage[i] ≈ w / 350 atol = 1e-12
    end
    @test cov.rho_256 ≈ 256 / 350 atol = 1e-12
    # Cost boundary (separately named; one day earlier than the user
    # window): numerically equal here — a single unclipped transition; the
    # semantic split shows up in the clipped fixtures below.
    @test cov.boundary_cost_windows == 2 .* BANDS
    @test cov.boundary_cost_256 ≈ 256 / 350 atol = 1e-12
    # Activation (first observation) is its OWN statistic: t=2 (pre-range,
    # assets 1-2) and t=102 (in-range, asset 3, position 53).
    @test cov.activation_transitions == 1
    @test cov.activation_positions == [53]
    @test cov.pre_range_activation_transitions == 1
    @test cov.activation_fingerprints == [[1, 2], [1, 2, 3]]

    # Fully synchronous panel: no mask transitions anywhere, one run over
    # the whole prefix, zero coverage at every depth.
    sync_signal = fill(NaN, 200, 2)
    sync_signal[:, 1] .= 100 .+ cumsum(0.01 .* randn(MersenneTwister(4), 200))
    sync_signal[:, 2] .= 40 .+ cumsum(0.01 .* randn(MersenneTwister(5), 200))
    cov0 = mask_transition_coverage(sync_signal; from = 50)
    @test cov0.decision_days == 150
    @test cov0.transitions == 0
    @test cov0.pre_range_transitions == 0     # the t=2 activation is NOT a mask transition
    @test cov0.unique_masks == 1
    @test cov0.pre_range_activation_transitions == 1
    @test all(==(0.0), cov0.coverage)
    @test cov0.rho_256 == 0.0
    @test cov0.run_lengths == [150]
    @test sum(cov0.run_lengths) == cov0.decision_days
    @test cov0.stable_run_max == 150
    @test cov0.stable_run_median == 150.0

    # Two staggered activations: assets 3 and 4 print from bars 101 and 200;
    # masks turn on at k=101 and k=199.
    signal2 = fill(NaN, 400, 4)
    signal2[:, 1:2] .= signal[:, 1:2]
    signal2[101:end, 3] .= signal[101:end, 3]
    signal2[200:end, 4] .= 20 .+ cumsum(0.01 .* randn(MersenneTwister(6), 201))
    cov2 = mask_transition_coverage(signal2; from = 50)
    @test cov2.transitions == 2               # k=101 (day 102) and k=199 (day 200)
    # Asset 4 first price 200 -> first return bar 201; from=50 means
    # position 201-50+1 = 152 (not 151: the transition day 200 is one
    # return-row BEFORE its first observed return).
    @test cov2.transition_positions == [53, 152]
    @test cov2.unique_masks == 3
    @test cov2.activation_positions == [53, 152]
    @test cov2.pre_range_activation_transitions == 1
    # Full-prefix runs [100, 99, 199]; the two 256-deep windows overlap from
    # day 150, so rho_256 covers [52, 350].
    # Runs [52, 99, 199]: second run 99 = 151-52 (transition at 152),
    # third 199 = 350-151 (clipped at range end 350).
    @test cov2.run_lengths == [52, 99, 199]
    @test sum(cov2.run_lengths) == cov2.decision_days
    @test cov2.stable_run_min == 52
    @test cov2.stable_run_median == 99
    @test cov2.stable_run_max == 199
    @test cov2.coverage[findfirst(==(4), cov2.coverage_windows)] ≈ 8 / 350 atol = 1e-12
    @test cov2.coverage[findfirst(==(64), cov2.coverage_windows)] ≈ 128 / 350 atol = 1e-12
    # Clipped at the range end the two metrics differ by exactly one day:
    # forward [53,308]∪[151,350] = 298 vs boundary [52,307]∪[150,350] = 299.
    @test cov2.rho_256 ≈ 298 / 350 atol = 1e-12
    @test cov2.boundary_cost_256 ≈ 299 / 350 atol = 1e-12

    # ---------- Probe B: suspension/resumption, observed-mask windows ----------
    # Asset 2 suspends for bars 200..210 and resumes. The DAILY
    # observed-return mask changes at k=199 (obs off) and k=211 (back):
    # these ARE transitions now — they split the runs and drive coverage —
    # not just a side count. Days 200..211 stay partially observed.
    susp = fill(NaN, 400, 3)
    susp[:, 1] .= 100 .+ cumsum(0.01 .* randn(MersenneTwister(11), 400))
    susp[:, 2] .= 50 .+ cumsum(0.01 .* randn(MersenneTwister(12), 400))
    susp[:, 3] .= 30 .+ cumsum(0.01 .* randn(MersenneTwister(13), 400))
    susp[200:210, 2] .= NaN
    covs = mask_transition_coverage(susp; from = 50)
    @test covs.transitions == 2               # k=199 (day 200) and k=211 (day 212)
    @test covs.transition_positions == [151, 163]
    @test covs.observed_transition_positions == [151, 163]
    @test covs.pre_range_transitions == 0
    @test covs.unique_masks == 2
    @test covs.mask_fingerprints == [[true, true, true], [true, false, true]]
    @test covs.final_active == 3
    @test covs.activation_transitions == 0   # activation never saw the suspension
    @test covs.pre_range_activation_transitions == 1
    @test covs.partial_rows_active == 12      # days 200..211: 0 < 2 < |A|=3
    @test covs.partial_rows_universe == 12
    @test covs.pre_range_partial_rows_active == 0
    # PRIMARY runs: the decision window cut by k=199/211: [150, 12, 188] —
    # the suspension itself is a 12-day run; sum == K.
    @test covs.run_lengths == [150, 12, 188]
    @test sum(covs.run_lengths) == covs.decision_days
    @test covs.stable_run_min == 12
    @test covs.stable_run_median == 150.0
    @test covs.stable_run_max == 188
    # User forward windows: [151,150+w] and [163,162+w]; they overlap once
    # w >= 13 (163 <= 151+12). Hand-derived:
    #   w=2/4/8 disjoint -> 2w; w=16: 32-4 = 28; w=32: 64-20 = 44;
    #   w=64: 128-52 = 76; w=128: 256-116 = 140; w=256: [151,350] = 200.
    for (i, w) in enumerate(covs.coverage_windows)
        local expected = w <= 8 ? 2w : (w == 16 ? 28 : (w == 32 ? 44 :
                           (w == 64 ? 76 : (w == 128 ? 140 : 200))))
        @test covs.coverage[i] ≈ expected / 350 atol = 1e-12
    end
    @test covs.rho_256 ≈ 200 / 350 atol = 1e-12
    # Cost boundary starts one day earlier ([150,149+w], [162,161+w]): the
    # same counts while unclipped, one more day at depth 256 ([150,350] =
    # 201) — distinct from rho_256, and still not the exact materialized
    # row count (resource-budget demotion adds rows on top of this).
    @test covs.boundary_cost_256 ≈ 201 / 350 atol = 1e-12
    i64c = findfirst(==(64), covs.boundary_cost_windows)
    @test covs.boundary_cost_coverage[i64c] ≈ 76 / 350 atol = 1e-12

    # ---------- Probe B: the last bar T is never read ----------
    # A mask change on bar T itself — a suspension starting at the final
    # bar, or a brand-new asset printing only bar T — must not move any
    # primary statistic: masks exist only for bars 2..T-1.
    susp_last = copy(susp)
    susp_last[400, 2] = NaN                    # mask change on the final bar
    covl = mask_transition_coverage(susp_last; from = 50)
    @test covl.transitions == covs.transitions == 2
    @test covl.transition_positions == covs.transition_positions
    @test covl.run_lengths == covs.run_lengths
    @test covl.stable_run_median == covs.stable_run_median
    @test covl.unique_masks == covs.unique_masks
    @test covl.rho_256 ≈ covs.rho_256 atol = 1e-12
    @test covl.boundary_cost_256 ≈ covs.boundary_cost_256 atol = 1e-12
    ipo_last = hcat(signal, fill(NaN, 400, 1))  # asset 4 prints ONLY bar 400
    ipo_last[400, 4] = 30.0
    covi = mask_transition_coverage(ipo_last; from = 50)
    @test covi.transitions == cov.transitions == 1
    @test covi.run_lengths == cov.run_lengths
    @test covi.stable_run_median == cov.stable_run_median
    @test covi.unique_masks == cov.unique_masks
    @test covi.rho_256 ≈ cov.rho_256 atol = 1e-12
    @test covi.universe == 4                  # only the universe field widens

    # ---------- Probe B: pre-range transition coverage ----------
    # Asset 3 activates at bar 30, twenty bars BEFORE from=50. No in-range
    # transition exists, but the full-prefix run structure sees the k=29
    # transition and its windows still cover the start of the range.
    pre = fill(NaN, 400, 3)
    pre[:, 1] .= 100 .+ cumsum(0.01 .* randn(MersenneTwister(14), 400))
    pre[:, 2] .= 50 .+ cumsum(0.01 .* randn(MersenneTwister(15), 400))
    pre[30:end, 3] .= 30 .+ cumsum(0.01 .* randn(MersenneTwister(16), 371))
    covp = mask_transition_coverage(pre; from = 50)
    @test covp.transitions == 0
    @test covp.pre_range_transitions == 1     # k=29: before bar 49
    @test covp.unique_masks == 1              # decision range sees only [T T T]
    @test covp.prefix_unique_masks == 2       # whole prefix adds the pre-range [T T F]
    @test covp.final_active == 3
    @test covp.run_lengths == [350]
    @test sum(covp.run_lengths) == covp.decision_days
    @test covp.stable_run_min == 350
    @test covp.stable_run_max == 350
    # Pre fixture: first price bar 30 -> first observed return bar 31
    # (return row 30). Valid prefix return rows are 1:T-2 = 398, so the
    # pre-range cut gives runs [29, 369] (29 = 30-1 rows before the cut,
    # 369 = 398-29), NOT [28, 370].
    @test covp.prefix_stable_run_min == 29
    @test covp.prefix_stable_run_max == 369
    # User forward window: transition day 30 opens [1, w-20] when w > 20.
    i2 = findfirst(==(2), covp.coverage_windows)
    i16 = findfirst(==(16), covp.coverage_windows)
    i32 = findfirst(==(32), covp.coverage_windows)
    i128 = findfirst(==(128), covp.coverage_windows)
    @test covp.coverage[i2] == 0.0
    @test covp.coverage[i16] == 0.0
    # Forward w32 covers bars 31..62; intersect with range from 50 gives
    # 13 days (50..62), w128 gives 109, w256 gives 237.
    @test covp.coverage[i32] ≈ 13 / 350 atol = 1e-12
    @test covp.coverage[i128] ≈ 109 / 350 atol = 1e-12
    @test covp.rho_256 ≈ 237 / 350 atol = 1e-12        # [1, 237]
    # Cost boundary is one day EARLIER than the forward metric at the same
    # depths: 12 / 108 / 236.
    i32c = findfirst(==(32), covp.boundary_cost_windows)
    @test covp.boundary_cost_coverage[i32c] ≈ 12 / 350 atol = 1e-12
    @test covp.boundary_cost_256 ≈ 236 / 350 atol = 1e-12

    # ---------- Probe B: never-active asset, the two conventions diverge ----------
    # Asset 4 never prints: fully observed in the ACTIVE space (3 of 3, the
    # engine's async_seen never flips) yet partially observed UNIVERSE-wide
    # (3 of 4) every single day.
    dead = fill(NaN, 400, 4)
    dead[:, 1] .= 100 .+ cumsum(0.01 .* randn(MersenneTwister(17), 400))
    dead[:, 2] .= 50 .+ cumsum(0.01 .* randn(MersenneTwister(18), 400))
    dead[:, 3] .= 30 .+ cumsum(0.01 .* randn(MersenneTwister(19), 400))
    covd = mask_transition_coverage(dead; from = 50)
    @test covd.transitions == 0
    @test covd.unique_masks == 1
    @test covd.final_active == 3
    @test covd.universe == 4
    @test covd.partial_rows_active == 0       # 3 of |A|=3: not partial
    @test covd.partial_rows_universe == 350   # 3 of 4: partial every day

    # ---------- Probe B: same counts, different membership ----------
    # Two panels with identical transition counts but different mask
    # membership: the fingerprints must tell them apart.
    panelA = fill(NaN, 400, 4)
    panelB = fill(NaN, 400, 4)
    panelA[:, 1:2] .= signal[:, 1:2]
    panelA[101:end, 3] .= signal[101:end, 3]  # asset 3 activates; 4 never
    panelB[:, 1:2] .= signal[:, 1:2]
    panelB[101:end, 4] .= 20 .+ cumsum(0.01 .* randn(MersenneTwister(20), 300))
    covA = mask_transition_coverage(panelA; from = 50)
    covB = mask_transition_coverage(panelB; from = 50)
    @test covA.transitions == covB.transitions == 1
    @test covA.transition_positions == covB.transition_positions
    @test covA.unique_masks == covB.unique_masks == 2
    @test covA.stable_run_max == covB.stable_run_max
    @test covA.mask_fingerprints[2] == [true, true, true, false]
    @test covB.mask_fingerprints[2] == [true, true, false, true]
    @test covA.mask_fingerprints != covB.mask_fingerprints

    # from bounds: it must be a decision-day bar index in 2:T-1.
    @test_throws ArgumentError mask_transition_coverage(signal; from = 1)
    @test_throws ArgumentError mask_transition_coverage(signal; from = 400)

    # ---------- Probe E: engine bucket wiring (unchanged engine behavior) ----------
    @test KTrader.TIMING_BUCKETS[1:9] == (:prep, :basis, :gram, :eigen, :EB, :condition, :fracFFT, :scenario, :Kelly)
    @test KTrader.TIMING_BUCKETS[10:end] == (:OOF_fit, :OOF_predict, :OOF_residual)
    P = exp.(cumsum(0.01 .* randn(MersenneTwister(7), 400, 3); dims = 1))
    timing = KTrader.DecisionTiming()
    model = fit_v1(P; F_folds = 3, timing)
    i_fit = findfirst(==(:OOF_fit), KTrader.TIMING_BUCKETS)
    i_pred = findfirst(==(:OOF_predict), KTrader.TIMING_BUCKETS)
    i_res = findfirst(==(:OOF_residual), KTrader.TIMING_BUCKETS)
    @test timing.seconds[i_fit] > 0           # three fold refits with EB
    @test timing.seconds[i_pred] > 0          # oracle construction
    @test timing.seconds[i_res] > 0           # own-row lists + macro series
    @test all(>=(0), timing.seconds)
    # The model itself is untouched by instrumentation.
    @test length(model.active_indices) == 3
    @test all(isfinite, model.mu_pred[model.active_indices])

    # ---------- Single-stage entry points: one prepare, one solve, shared input ----------
    # Default admission: exactly ONE prepare + ONE solve, no warmup, no repeats.
    # The bucket-boundary observer hook lives in numerics.jl's `timed` (the
    # single owner of bucket timing); default DecisionTiming observes nothing.
    @test KTrader.DecisionTiming().sink === nothing
    io_p = IOBuffer()
    prepared = DevProbes.prepare_stage_probe(P; io = io_p)
    log_p = String(take!(io_p))
    @test prepared.t_star == 399                 # default: last decision day
    @test prepared.history_rows == 399           # the FULL prefix: no history truncation
    @test prepared.prep.T == 399                 # the prepared input IS the full prefix
    @test prepared.active_assets == 3
    @test occursin("[stage:prep][start]", log_p)
    @test occursin("[stage:prep][end]", log_p)
    @test !occursin("[stage:solve]", log_p)      # the prepare stage never solves
    @test !occursin("[stage:warmup]", log_p)
    @test prepared.timing.seconds[findfirst(==(:prep), KTrader.TIMING_BUCKETS)] > 0
    # :gram flows through timed(), so the observer fires for it; :prep/:basis
    # accumulate by direct addition in predict.jl and expose NO bucket
    # boundary — a documented limitation of where the hook lives, not a gap
    # the ceiling side may paper over by replicating the algorithm.
    @test occursin("[stage:prep][bucket:start] gram", log_p)
    @test occursin("[stage:prep][bucket:stop] gram", log_p)
    @test !occursin("[stage:prep][bucket:start] prep", log_p)
    @test !occursin("[stage:prep][bucket:start] basis", log_p)
    # The solve consumes the SAME prepared input: t_star and fold mismatches
    # are rejected rather than silently re-preparing something else.
    @test_throws ArgumentError DevProbes.solve_stage_probe(prepared.prep; t_star = 398)
    @test_throws ArgumentError DevProbes.solve_stage_probe(prepared.prep; F_folds = 2)
    @test_throws ArgumentError DevProbes.solve_stage_probe(prepared.prep; mode = :both)
    io_s = IOBuffer()
    rep = DevProbes.solve_stage_probe(prepared.prep; mode = :cold, io = io_s)
    log_s = String(take!(io_s))
    @test rep.succeeded == 1
    @test rep.repeats == 1
    @test rep.elapsed_seconds > 0
    @test rep.timing.seconds[i_fit] > 0
    @test occursin("[stage:solve][start]", log_s)
    @test occursin("[stage:solve][end]", log_s)
    @test !occursin("[stage:prep]", log_s)       # the solve stage does not re-prepare
    @test !occursin("[stage:warmup]", log_s)     # no automatic warmup before the solve
    @test first(findfirst("[stage:solve][start]", log_s)) <
          first(findfirst("[stage:solve][end]", log_s))
    # Bucket boundaries are immediate: a timeout inside the solve shows WHICH
    # bucket was entered (eigen and OOF_fit both flow through timed()).
    @test occursin("[stage:solve][bucket:start] eigen", log_s)
    @test occursin("[stage:solve][bucket:stop] eigen", log_s)
    @test occursin("[stage:solve][bucket:start] OOF_fit", log_s)
    @test occursin("[stage:solve][bucket:stop] OOF_fit", log_s)
    @test first(findfirst("[stage:solve][bucket:start] eigen", log_s)) <
          first(findfirst("[stage:solve][bucket:stop] eigen", log_s))
    @test rep.observer_error === nothing          # clean run: no observer failure
    # The fitted model is returned so downstream stages can be chained on
    # this ONE fit without repeating it; it carries no logging resource.
    @test rep.model !== nothing
    @test rep.model isa V1Model
    @test :sink ∉ propertynames(rep.model)
    @test :timing ∉ propertynames(rep.model)
    # Chaining works on the returned model directly (no second fit):
    scenarios = generate_scenarios_v1(rep.model; S = 10, rng = MersenneTwister(21))
    @test scenarios isa AbstractMatrix
    @test 10 in size(scenarios)

    # ---------- Observer failure policy (numerics.jl `timed` contract) ----------
    # A PRIMARY failure propagates unchanged with the observer's secondary
    # error recorded separately; a successful bucket with a dead observer
    # RAISES the observer's error (never a silent success); the default
    # sink===nothing path is untouched. Notification overhead stays out of
    # the measured bucket time (stamps are notification wall clocks).
    struct SinkBoom <: Exception end
    boom = (_, _, _) -> throw(SinkBoom())
    timing_primary = KTrader.DecisionTiming()
    timing_primary.sink = boom
    @test_throws ErrorException KTrader.timed(() -> error("model boom"), timing_primary, :eigen)
    @test occursin("SinkBoom", something(timing_primary.sink_error, ""))
    timing_success = KTrader.DecisionTiming()
    timing_success.sink = boom
    @test_throws SinkBoom KTrader.timed(() -> nothing, timing_success, :eigen)
    @test occursin("SinkBoom", something(timing_success.sink_error, ""))
    # Timing still completed for the bucket despite the observer failure.
    @test timing_success.seconds[findfirst(==(:eigen), KTrader.TIMING_BUCKETS)] >= 0
    # Default fast path: no sink, no recorded error, additive timing unchanged.
    timing_plain = KTrader.DecisionTiming()
    @test timing_plain.sink === nothing
    @test timing_plain.sink_error === nothing
    KTrader.timed(() -> nothing, timing_plain, :EB)
    @test timing_plain.seconds[findfirst(==(:EB), KTrader.TIMING_BUCKETS)] >= 0
    # A solve that fails mid-stage keeps its start line in the log (a tool
    # abort is not process death — the log stream survives) and reports a
    # PARTIAL result, never a silent success. Broken statistics make the
    # fit fail after the start line has been printed.
    bad_prep = deepcopy(prepared.prep)
    bad_prep.stats.full_xy .= NaN          # non-finite statistics fail the fit
    io_f = IOBuffer()
    rep_f = DevProbes.solve_stage_probe(bad_prep; io = io_f)
    log_f = String(take!(io_f))
    @test rep_f.succeeded == 0
    @test rep_f.failed == 1
    @test rep_f.model === nothing          # no model escapes a failed solve
    @test isnan(rep_f.seconds[1])
    @test occursin("[stage:solve][start]", log_f)   # start survived the failure
    @test occursin("[stage:solve][end]", log_f)
    @test occursin("failed=yes", log_f)
    # ---------- Probe D: prepared-statistics solve baseline (compatibility wrapper) ----------
    io_d = IOBuffer()
    floor = solve_floor_probe(P; io = io_d)
    log_d = String(take!(io_d))
    @test floor.repeats == 1
    @test floor.succeeded == 1
    @test floor.failed == 0
    @test floor.timing.sink_error === nothing   # observer isolation via stage timing
    @test floor.mean_seconds > 0
    @test floor.median_seconds > 0
    @test length(floor.seconds) == 1
    @test all(isfinite, floor.seconds)
    @test floor.includes_scenarios_kelly == false
    @test floor.t_star == 399
    @test floor.mode == :cold
    # full prefix, no truncation (shared stage t_star; wrapper omits history_rows)
    # Exactly ONE prepare and ONE solve in the default run; no warmup stage.
    lines_d = split(log_d, "\n")   # Base.split; a local `split` shadow is a fixture bug
    @test count(l -> startswith(l, "[stage:prep][start]"), lines_d) == 1
    @test count(l -> startswith(l, "[stage:prep][end]"), lines_d) == 1
    @test count(l -> startswith(l, "[stage:solve][start]"), lines_d) == 1
    @test count(l -> startswith(l, "[stage:solve][end]"), lines_d) == 1
    @test !occursin("[stage:warmup]", log_d)
    # Admission: repeated solves and dual modes are REJECTED, no override.
    @test_throws ArgumentError solve_floor_probe(P; repeats = 5)
    @test_throws ArgumentError solve_floor_probe(P; repeats = 100)
    @test_throws ArgumentError solve_floor_probe(P; mode = :both)
    @test_throws ArgumentError solve_floor_probe(P; mode = :hot)
    # WARMUP ADMISSION (2026-10-07 gate): warmup=true is rejected by every
    # stage and compatibility entry BEFORE any data is touched — a warmup is
    # a second full solve and the single-solve cost is not yet verified.
    # The tests falsify the old opt-in behavior instead of keeping it green;
    # this is a gate, re-admitted only by an explicit decision once a
    # measured single-day solve cost exists.
    @test_throws ArgumentError solve_floor_probe(P; warmup = true)
    @test_throws ArgumentError DevProbes.solve_stage_probe(prepared.prep; warmup = true)
    @test_throws ArgumentError DevProbes.oof_timing_split_probe(P; warmup = true)
    # A single explicit warm mode is one solve (the persistent EB alpha chain
    # solve_current! uses), measured separately from cold.
    io_w = IOBuffer()
    floor_warm = solve_floor_probe(P; mode = :warm, io = io_w)
    @test floor_warm.mode == :warm
    @test floor_warm.succeeded == 1
    @test floor_warm.mean_seconds > 0
    @test floor_warm.timing.sink_error === nothing # observer isolation via stage timing
    @test floor_warm.includes_scenarios_kelly == false
    @test !occursin("[stage:warmup]", String(take!(io_w)))

    # ---------- Probe E: frozen-day OOF timing split (one solve) ----------
    io_e = IOBuffer()
    split_rep = DevProbes.oof_timing_split_probe(P; io = io_e)
    log_e = String(take!(io_e))
    @test split_rep.repeats == 1
    @test split_rep.succeeded == 1
    @test split_rep.failed == 0
    @test split_rep.observer_error === nothing   # real observer isolation, transparently forwarded
    @test split_rep.buckets == collect(KTrader.TIMING_BUCKETS)
    @test split_rep.bucket_mean_seconds[i_fit] > 0
    @test all(x -> x >= 0, split_rep.bucket_mean_seconds)
    @test split_rep.oof_seconds > 0
    @test 0 < split_rep.oof_share <= 1
    @test split_rep.includes_scenarios_kelly == false
    # prep/basis/gram belong to the frozen preparation (probe D's prepare stage)
    # and stay zero in the solve's timing; scenario/Kelly are outside the
    # solve stage and stay zero too.
    for bucket in (:prep, :basis, :gram, :scenario, :Kelly)
        @test split_rep.bucket_mean_seconds[findfirst(==(bucket), split_rep.buckets)] == 0
    end
    # Admission: same single-stage policy as D.
    @test_throws ArgumentError DevProbes.oof_timing_split_probe(P; t_star = 1)
    @test_throws ArgumentError DevProbes.oof_timing_split_probe(P; repeats = 5)
    @test_throws ArgumentError DevProbes.oof_timing_split_probe(P; repeats = 100)
    @test occursin("[stage:solve][start]", log_e)
    @test !occursin("[stage:warmup]", log_e)

    # ---------- Probes A & C: single-day dev sequential run (K=1) ----------
    # M1 removed the production probe plumbing: backtest_v1 has no probe
    # keyword, the collector is dev-owned, and A/C are the SAME single-day
    # sequential probe (initialize + ONE solve_current!). The collector's
    # window accounting (record before solve, collect after, coordinator
    # check) is covered by the unit fixtures at the top of this file.
    # Engine math regressions — the ragged fast path, backtest topology
    # equality — live in test/incremental_tests.jl and test/timeblock_tests.jl,
    # their owners; this file does not hide a 30-day backtest behind them.
    dates = collect(Date(2020, 1, 1) .+ Day.(0:599))
    P4 = exp.(cumsum(0.01 .* randn(MersenneTwister(8), 600, 4); dims = 1))
    sync = Bars(dates, ["A", "B", "C", "D"], P4, P4)

    # Admission: multi-day runs are rejected (K=30 here would imply 30 solves
    # — the incident's hidden workload); there is no override, and the dev
    # sequential interface has no S/seed knobs to pass through.
    @test_throws ArgumentError sequential_ceiling_probe(sync; from = dates[end-30])
    # The single admitted shape is K=1: one advance + one solve.
    out = sequential_ceiling_probe(sync; from = dates[end-1])
    @test out.report.solves == 1
    @test out.report.fast == 1          # fully synchronous panel: fast path
    @test out.report.fallback == 0
    @test out.report.async_rows == 0
    @test out.checkpoints == 1
    @test out.model isa V1Model
    @test DevProbes.print_probe_report(DevProbes.CeilingProbeCollector();
                                       decisions = 1, io = devnull) === nothing
end
