"""M1 architecture boundary: the typed PreparedProblem.

History -> prepare -> PreparedProblem -> solve -> V1Model. This file owns
the TYPE and the single solve entry; the batch recomputation lives in
predict.jl (_prepare_v1, aliased as prepare_reference) and the exact
incremental preparation lives in incremental.jl (_prepare_current!,
aliased as prepare_incremental). Both producers return THIS type; there is
no production NamedTuple pipeline anymore (dev-side conversions only).

Field groups (SPEC 44.1): identity/universe, geometry & scaling, raw panel
& relative field (the ResidualOracle's frozen inputs), decision-time
basis source, training layout, full/fold sufficient statistics, and the
lazy design source. NO diagnostic state ever enters this struct.

Ownership & lifetime (frozen-input contract, stated exactly as
implemented): a PreparedProblem mixes three ownership classes.

  OWNED (materialized copies; caller-side mutation cannot reach them):
    r, m, and alive_now - the decision-day observation mask, frozen from
    the panel's last row at construction time (the constructor stores an
    owned Vector{Bool} copy even when the mask is supplied explicitly).
    On the DEFAULT no-workspace path every other matrix field (s1, s_m,
    s_perp, relative_embedding, observed, B_m, X_rel, X_rel_stacked, the
    full/fold Grams) is freshly allocated, hence also owned.

  CALLER-VIEWED (zero-copy alias): adj_act is a view of the caller's
    price panel. solve never reads it: the decision-day observation
    support it used to carry is frozen into the owned alive_now mask at
    prepare time, so a caller mutating the panel AFTER prepare (including
    its last row) cannot silently change any solve result of this object.
    The field remains as metadata; a fresh prepare reflects the mutation.

  WORKSPACE-LEASED (only when a caller explicitly shares a FitWorkspace):
    X_rel, X_rel_stacked and the full/fold Grams are the SAME Matrix
    objects the workspace buffers handed back; NO copies are taken. Their
    lifetime is protected serially by the generation binding: a later
    prepare on the same owner (successful, failed, or partial) bumps the
    generation before touching buffers, and solving an older prepared
    object then fails loudly - use-after-reprepare is REJECTED at solve,
    never silently solved with reused buffer content. This is a lease,
    not ownership: do not cache a workspace-prepared object across
    prepares.

solve itself never copies and never mutates. state.advance! on the
incremental side writes only the inference core, never these matrices.
The constructor validates shape consistency cheaply (O(#fields)) so
corrupted data fails at the boundary.
"""

struct FoldStatistics
    ranges::Vector{UnitRange{Int}}
    xx::Union{Nothing,Vector{Matrix{Float64}}}   # per-fold primal Grams (nothing in dual regime)
    xy::Vector{Matrix{Float64}}
    yy::Vector{Matrix{Float64}}
    full_xx::Union{Nothing,Matrix{Float64}}
    full_xy::Matrix{Float64}
    full_yy::Matrix{Float64}
end

struct MacroFoldBlock
    xx::Matrix{Float64}
    xy::Matrix{Float64}
    yy::Matrix{Float64}
    n::Int
end

struct MacroStatistics
    full::MacroFoldBlock
    folds::Vector{MacroFoldBlock}
end

"""Assemble the typed statistics from the producer-side NamedTuple shape
(batch fold_sufficient_statistics and the incremental regime_statistics
both emit this shape). Pure re-wrapping: no numeric change."""
function to_fold_statistics(st)
    FoldStatistics(st.ranges, st.xx, st.xy, st.yy, st.full_xx, st.full_xy, st.full_yy)
end
function to_macro_statistics(ms)
    MacroStatistics(MacroFoldBlock(ms.full.xx, ms.full.xy, ms.full.yy, ms.full.n),
                    [MacroFoldBlock(b.xx, b.xy, b.yy, b.n) for b in ms.folds])
end

struct PreparedProblem
    # identity / universe
    active_idx::Vector{Int}
    N_universe::Int
    T::Int                              # price rows in the active panel
    N::Int                              # active assets
    # geometry & scaling
    s1::Vector{Float64}
    s_m::Vector{Float64}
    s_perp::Vector{Float64}
    # raw panel & relative field (ResidualOracle frozen inputs)
    adj_act::AbstractMatrix{Float64}    # view of the caller's price panel (metadata only; solve never reads it)
    alive_now::Vector{Bool}             # decision-day observation mask (owned copy, frozen at prepare time)
    r::AbstractMatrix{Float64}          # raw log returns (NaN = missing)
    m::Vector{Float64}                  # macro flow
    relative_embedding::Matrix{Float64}
    observed::BitMatrix
    X_rel::AbstractMatrix{Float64}      # cumulative relative field (may be a workspace buffer)
    B_m::Matrix{Float64}                # macro path basis (decision row = last row)
    # training layout
    ts_total::Vector{Int}
    n_res::Int
    P_features::Int
    # full/fold sufficient statistics
    stats::FoldStatistics
    macro_stats::Union{Nothing,MacroStatistics}
    # lazy design source (nothing on the incremental/lazy path; rebuilt on
    # demand by solve only when a dual fit or fold train block needs rows)
    X_rel_stacked::Union{Nothing,Matrix{Float64}}
    # fit layout
    F_folds::Int
    # Prepare-generation binding (minimal, not a lease framework): the
    # workspace owner observed at prepare time plus its generation counter.
    # solve fails loudly if the owner's counter has moved on (a later
    # prepare - successful, failed, or partial - bumped it BEFORE touching
    # buffers, so stale-buffer solves cannot succeed). The DEFAULT
    # reference path shares no workspace: ws_owner === nothing, no check,
    # independent owner, stable. advance! never touches the workspace, so
    # a prepared object stays valid across a later advance. The generation
    # is a routing/lifetime fact only and never decides math.
    ws_owner::Union{Nothing,FitWorkspace}
    ws_generation::Int
end

function PreparedProblem(; active_idx, N_universe, T, N, s1, s_m, s_perp,
                           adj_act, r, m, relative_embedding, observed, X_rel,
                           B_m, ts_total, n_res, P_features, stats, macro_stats,
                           X_rel_stacked, F_folds,
                           alive_now = isfinite.(view(adj_act, size(adj_act, 1), :)),
                           ws_owner=nothing, ws_generation=0)
    N_universe >= N > 0 || throw(DimensionMismatch("active set larger than universe"))
    length(active_idx) == N || throw(DimensionMismatch("active_idx length vs N"))
    size(adj_act) == (T, N) || throw(DimensionMismatch("adj_act shape"))
    size(r) == (T - 1, N) || throw(DimensionMismatch("r shape"))
    size(relative_embedding) == (T - 1, N) || throw(DimensionMismatch("relative_embedding shape"))
    size(observed) == (T - 1, N) || throw(DimensionMismatch("observed mask shape"))
    size(X_rel) == (T - 1, N) || throw(DimensionMismatch("X_rel shape"))
    length(m) == T - 1 || throw(DimensionMismatch("macro flow length"))
    length(s1) == N || throw(DimensionMismatch("s1 length"))
    length(alive_now) == N || throw(DimensionMismatch("alive_now length vs N"))
    size(B_m) == (T - 1, 2 * length(BANDS)) || throw(DimensionMismatch("B_m shape"))
    length(ts_total) == n_res > 0 || throw(DimensionMismatch("ts_total length"))
    P_features == 2 * length(BANDS) * N || throw(DimensionMismatch("P_features"))
    F_folds >= 2 && F_folds <= n_res || throw(ArgumentError("F_folds must be in 2:n_res"))
    length(stats.ranges) == F_folds || throw(DimensionMismatch("fold ranges count"))
    size(stats.full_xy) == (P_features, N) || throw(DimensionMismatch("full_xy shape"))
    size(stats.full_yy) == (N, N) || throw(DimensionMismatch("full_yy shape"))
    length(stats.xy) == F_folds && length(stats.yy) == F_folds ||
        throw(DimensionMismatch("fold gram counts"))
    stats.xx === nothing || length(stats.xx) == F_folds ||
        throw(DimensionMismatch("fold primal gram count"))
    stats.full_xx === nothing || size(stats.full_xx) == (P_features, P_features) ||
        throw(DimensionMismatch("full_xx shape"))
    macro_stats === nothing || length(macro_stats.folds) == F_folds ||
        throw(DimensionMismatch("macro fold count"))
    X_rel_stacked === nothing || size(X_rel_stacked) == (n_res, P_features) ||
        throw(DimensionMismatch("X_rel_stacked shape"))
    alive_own = Vector{Bool}(alive_now)   # owned copy (BitVector/alias inputs included)
    PreparedProblem(active_idx, N_universe, T, N, s1, s_m, s_perp, adj_act, alive_own, r,
        m, relative_embedding, observed, X_rel, B_m, ts_total, n_res,
        P_features, stats, macro_stats, X_rel_stacked, F_folds,
        ws_owner, ws_generation)
end

"""Reference preparation: recompute from the complete prefix (executable
specification; predict.jl's _prepare_v1 does the work and returns this type).

Closed public keyword surface (Manager decision, 2026-10-08): this entry
accepts EXACTLY the lawful caller keywords — ridge_alpha, F_folds,
ruler_stats, history_cache, alpha_initial, timing, workspace (the
non-internal subset of _prepare_v1's signature, defaults unchanged) — and
forwards them unchanged. Any other keyword is rejected by Julia keyword
dispatch before a single line of this function or _prepare_v1 runs; in
particular the three internal channels ruler_override / scale_override /
statistics_builder are rejected EVEN when passed as `nothing`. The refusal
therefore strictly precedes any workspace generation bump (which is the
first statement inside _prepare_v1), any cache read, and any builder
execution — it cannot be observed through side effects.

This is a whitelist closure, not a blacklist filter: nothing is silently
dropped or ignored. The three internal channels remain available on
_prepare_v1 itself, which is not a public entry: they exist for the
incremental engine's private wiring (incremental.jl _prepare_current!
passes state-owned ruler/scale/statistics values into the same math), and
that wiring is unaffected by this closure. No second solver or
configuration framework is introduced."""
function prepare_reference(adj::AbstractMatrix{Float64};
        ridge_alpha=nothing, F_folds=3, ruler_stats=nothing,
        history_cache=nothing, alpha_initial=nothing, timing=nothing,
        workspace=nothing)
    _prepare_v1(adj; ridge_alpha, F_folds, ruler_stats, history_cache,
                alpha_initial, timing, workspace)
end

"""Incremental preparation: contract the exact metric-free state into the
same PreparedProblem type (incremental.jl's _prepare_current!)."""
prepare_incremental(state; kwargs...) = _prepare_current!(state; kwargs...)
