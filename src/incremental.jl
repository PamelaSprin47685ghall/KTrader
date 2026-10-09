"""
Exact prefix inference with mask-run (regime) factorization.

The relative field is strictly linear in the decision-day metric:
    e_t(d) = R_t * diag(z_t) * d,   R_t = P_t - m_t m_t'/c_t
where z_t is the zero-filled raw return row (missing -> 0; the zero is a
FIELD definition, never a data imputation - the raw observation mask stays
with `r` and `field.observed`) and macro_t = d'z_t/sqrt(c_t). The batch
oracle (`_prepare_v1`) evaluates this directly. This engine keeps the
metric-free factors and contracts with today's d at solve time.

Runs: the return-row axis is partitioned into maximal intervals of constant
observation mask ("runs"). Within a run, R is constant, so interior feature
prototypes are single-mask; rows whose 2*tau window (tau <= 128 => depth
256, the deepest index touched by fill_design_matrix!'s
sums[t+1-2*tau]) or whose target row crosses a run boundary carry
multi-mask prototypes and are solved exactly via per-row materialization
(BLAS syrk/gemm on the materialized block) - never via permanent batch
fallback. Prototypes are MERGED per unique mask (registry), so repeated
masks (e.g. alternating availability) cost one Gram entry, not one per
transition.

async_seen is a DIAGNOSTIC ONLY: a partially observed row opens/extends runs
like any other row and never disables the fast path. The only fallbacks are
numerical (InferenceContractionError, same semantics as before) and
:resource_budget (materialized rows beyond the configured row limit; the
batch oracle is exact, so this routes, never changes outputs).

IPO/activation: `active` grows by sorted insertion, which can permute old
coordinates ([1,3] -> [1,2,3]). History is preserved by an explicit
permutation embed of all N-vector / NxN statistics and stored row records
(new asset columns are exact zeros, matching the zero-filled history). No
return-row replay; `:embeds` counts these structural embeds while
`:rebuilds` stays at zero.

Memory budgeting: aggregated Grams are allocated lazily per (mask, fold);
when the byte budget (default 512 MiB per state, configurable, route-only)
would be exceeded, new rows take the materialized path. The materialized
row count beyond `materialize_row_limit` (default 4096) routes the solve to
the batch oracle with reason :resource_budget. IPO embeds check the SAME
budget BEFORE enlarging any held Gram (embed_active! rebuilds the dict
directly, bypassing budget_allows!): within budget the Grams are embedded
and the ledger resynchronizes to the true held payload; over budget NO
enlarged matrix is allocated - the held Grams are released (ledger -> 0)
and their rows are demoted to the exact materialized path, so raw history,
coordinates and fold membership keep evolving with the same mathematics and
new rows aggregate again as budget allows (no replay, no permanent
fallback). The ledger counts HELD FoldGram matrix payload (V+Xc+R2) only -
not rows, runs or zsum_hist, not the transient contraction workspace, and
it is not an OS RSS cap.
"""
const REGIME_GRAM_BUDGET_DEFAULT = 512 * 2^20
const REGIME_MATERIALIZE_ROW_LIMIT_DEFAULT = 4096
const REGIME_WINDOW_DEPTH = 2 * maximum(BANDS)  # 256: deepest fill_design_matrix! index

mutable struct RegimeRun
    maskid::Int
    mask::Vector{Bool}
    first::Int
    last::Int
    zbase::Vector{Float64}   # zsum at first-1 (N)
    zfull::Vector{Float64}   # zsum at last minus zbase; maintained live
end

mutable struct PendingPrototype
    regs::Vector{Int}              # unique mask ids in the window support
    v::Vector{Matrix{Float64}}     # per mask: N x C, column c = channel
end
const REGIME_CHANNELS = 2 * length(BANDS)

mutable struct RowRecord
    regs::Vector{Int}
    v::Vector{Matrix{Float64}}     # per mask: N x C prototypes
    target::Vector{Float64}        # z_{t+1}
    tmask::Int                     # mask id of the target row's run
    fold::Int
    aggregated::Bool               # true: contributions live in FoldGram
end

mutable struct FoldGram
    n::Int
    V::Vector{Matrix{Float64}}     # symmetric channel pairs c <= c'
    Xc::Vector{Matrix{Float64}}    # per channel: sum v_c z'
    R2::Matrix{Float64}            # sum z z'
end

mutable struct RawInferenceCore
    active::Vector{Int}
    k::Int
    zsum::Vector{Float64}
    zsum_hist::Vector{Vector{Float64}}   # hist[s+1] = zsum_s (hist[1] = 0)
    runs::Vector{RegimeRun}
    maskids::Dict{Vector{Bool},Int}
    mask_of::Vector{Vector{Bool}}
    rows::Vector{RowRecord}
    grams::Dict{Tuple{Int,Int},FoldGram}
    gram_bytes::Int
    gram_budget::Int
    materialize_row_limit::Int
    membership::Vector{Int}
    boundary_moves::Int
    async_rows::Int
    async_seen::Bool
    pending::Union{Nothing,PendingPrototype}
    embeds::Int
    resource_limited::Bool
end

function RawInferenceCore(active, F; gram_budget = REGIME_GRAM_BUDGET_DEFAULT,
                          materialize_row_limit = REGIME_MATERIALIZE_ROW_LIMIT_DEFAULT)
    N = length(active)
    RawInferenceCore(copy(active), 0, zeros(N), [zeros(N)],
        RegimeRun[], Dict{Vector{Bool},Int}(), Vector{Vector{Bool}}(),
        RowRecord[], Dict{Tuple{Int,Int},FoldGram}(), 0, gram_budget,
        materialize_row_limit, Int[], 0, 0, false, nothing, 0, false)
end

mutable struct ExactInferenceState
    bars::Vector{Vector{Float64}}
    logs::Vector{Vector{Float64}}
    first_price::Vector{Int}
    ruler_acc::Matrix{Float64}
    ruler_cnt::Matrix{Int}
    active::Vector{Int}
    core::Union{Nothing,RawInferenceCore}
    ridge_alpha::Union{Nothing,Float64}
    F_folds::Int
    gram_budget::Int
    materialize_row_limit::Int
    warm::Vector{Tuple{Float64,Float64}}
    workspace::FitWorkspace
    history_cache::Any
    ruler_stats::Any
    counters::Dict{Symbol,Int}
    fallback_reasons::Dict{Symbol,Int}
    last_reason::Union{Nothing,Symbol}
end

const INFERENCE_COUNTER_KEYS =
    (:advances, :solves, :fast, :fallback, :rebuilds, :async_rows, :embeds)

function fresh_counters()
    Dict(k => 0 for k in INFERENCE_COUNTER_KEYS)
end

function inference_diagnostics(state::ExactInferenceState)
    core = state.core
    (; day=length(state.bars), active=copy(state.active), counters=copy(state.counters),
       fallback_reasons=copy(state.fallback_reasons), last_reason=state.last_reason,
       training_rows=core === nothing ? 0 : length(core.rows),
       boundary_moves=core === nothing ? 0 : core.boundary_moves,
       runs=core === nothing ? 0 : length(core.runs),
       unique_masks=core === nothing ? 0 : length(core.mask_of),
       materialized_rows=core === nothing ? 0 : count(r -> !r.aggregated, core.rows),
       gram_bytes=core === nothing ? 0 : core.gram_bytes,
       resource_limited=core !== nothing && core.resource_limited,
       async_rows=get(state.counters, :async_rows, 0),
       async_seen=core === nothing ? false : core.async_seen)
end

function initialize_inference(adj_prefix::AbstractMatrix{Float64}; ridge_alpha=nothing, F_folds=3,
                              history_cache=nothing, ruler_stats=nothing,
                              gram_budget=REGIME_GRAM_BUDGET_DEFAULT,
                              materialize_row_limit=REGIME_MATERIALIZE_ROW_LIMIT_DEFAULT)
    T,N = size(adj_prefix)
    T>=1 && N>=1 || throw(ArgumentError("inference requires a nonempty price prefix"))
    F_folds>=2 || throw(ArgumentError("F_folds must be at least two"))
    ridge_alpha === nothing || (isfinite(ridge_alpha) && ridge_alpha>0) ||
        throw(ArgumentError("ridge_alpha must be positive and finite"))
    gram_budget>0 || throw(ArgumentError("gram_budget must be positive"))
    materialize_row_limit>0 || throw(ArgumentError("materialize_row_limit must be positive"))
    # SPEC §56 fail-loud cache guard, unified with the batch path (_prepare_v1
    # runs the same check): BEFORE any state is built or any row advanced, the
    # cache must be element-exact with adj_prefix on the full consumed fact
    # set (logs, returns, first_price, first_return) — replacing the old
    # logs-only, post-advance check. advance_exact! and _prepare_current!
    # never read the cache (the inference state is the sole mathematical
    # source after this point; their _prepare_v1 fallbacks pass no cache), so
    # this entry plus _prepare_v1's guard close the cache boundary for both
    # engines. No-lookahead: rows beyond T are never read; a longer lawful
    # cache stays reusable for any of its prefixes.
    history_cache === nothing || verify_history_cache_prefix(adj_prefix, history_cache)
    state=ExactInferenceState(Vector{Float64}[],Vector{Float64}[],zeros(Int,N),
        zeros(N,length(TAUS)),zeros(Int,N,length(TAUS)),Int[],nothing,ridge_alpha,
        F_folds,Int(gram_budget),Int(materialize_row_limit),
        fill((1.0,1.0),F_folds+1),FitWorkspace(),history_cache,ruler_stats,
        fresh_counters(),Dict{Symbol,Int}(),nothing)
    for t in 1:T
        advance_exact!(state,view(adj_prefix,t,:))
    end
    if ruler_stats !== nothing
        # Unified provenance semantics with the batch entry (predict.jl
        # wires verify_ruler_stats_prefix before ruler_from_stats):
        # ruler_stats is an exact acceleration cache of THIS prefix
        # statistics. This entry verifies it against the statistics the
        # state itself accumulated while advancing the prefix (row T,
        # full column set; cnt exact, acc at the same 64eps production
        # roundoff tolerance).
        size(ruler_stats.acc,1)>=T && size(ruler_stats.acc,2)==N ||
            throw(DimensionMismatch("ruler cache prefix"))
        isapprox(view(ruler_stats.acc,T,:,:),state.ruler_acc;rtol=64eps(Float64),atol=0.0) &&
            view(ruler_stats.cnt,T,:,:)==state.ruler_cnt ||
            throw(ArgumentError("ruler cache does not match the observed prefix"))
    end
    state.counters[:advances]=0
    state
end

function inference_checkpoint(state::ExactInferenceState)
    ExactInferenceState([copy(x) for x in state.bars],[copy(x) for x in state.logs],
        copy(state.first_price),copy(state.ruler_acc),copy(state.ruler_cnt),copy(state.active),
        deepcopy(state.core),state.ridge_alpha,state.F_folds,state.gram_budget,
        state.materialize_row_limit,copy(state.warm),FitWorkspace(),
        state.history_cache,state.ruler_stats,copy(state.counters),copy(state.fallback_reasons),state.last_reason)
end

function inference_ranges(n,F)
    q=div(n,F)
    [((f-1)*q+1):(f==F ? n : f*q) for f in 1:F]
end

# ---------------- mask registry ----------------
function registry_id!(core::RawInferenceCore, mask::Vector{Bool})
    get!(core.maskids, mask) do
        push!(core.mask_of, copy(mask))
        length(core.mask_of)
    end
end

# ---------------- metric-free history accessors ----------------
zsum_at(core, s) = core.zsum_hist[s+1]

function run_of_row_start(core::RawInferenceCore, u::Int)
    # runs are contiguous in row order; find the index of the run containing u
    i = length(core.runs)
    while i > 1 && core.runs[i].first > u
        i -= 1
    end
    i
end

"""window z-sum of run rho over [a,b]: Zw^rho (metric-free)."""
function window_z_sum!(out, core, rho::RegimeRun, a::Int, b::Int)
    fill!(out, 0.0)
    (b < a || rho.first > b || rho.last < a) && return out
    lo = max(a, rho.first)
    hi = min(b, rho.last)
    out .= zsum_at(core, hi) .- zsum_at(core, lo - 1)
    out
end

# ---------------- pending prototypes ----------------
# Exact triangular local filter on RAW returns, derived from the batch
# identity with c0 = mean(X[k-tau+1:k]), c1 = mean(X[k-2*tau+1:k-tau]) and
# X_s = sum_{u<=s} e_u (e_u = R^{run(u)}(z_u .* d)):
#   Q = -(X_k - c0) = -sum_{j=0..tau-1} ((tau-1-j)/tau) e[k-j]
#       (the j = tau-1 term is zero; pre-window constants cancel exactly,
#        so NO run older than the window ever contributes)
#   P = c0 - c1     =  sum_{j=0..2tau-2} (min(j+1, 2tau-1-j)/tau) e[k-j]
# Writing e[k-j] = R^{run(k-j)}(z_{k-j} .* d), the per-mask metric-free
# prototype is the weight-summed raw z over window-intersect-run; runs
# sharing a mask merge (same R). Q carries its own minus sign: the stored
# Q column is the NEGATIVE displacement prototype (weights are negative,
# peaking at -(tau-1)/tau on the newest row), P stores the positive
# triangular weights. Band activation keeps the legacy k >= 2tau+1.
function compute_pending(core::RawInferenceCore, k::Int)
    N = length(core.zsum)
    acc = Dict{Int,Tuple{Matrix{Float64},Matrix{Float64}}}()
    zu = zeros(N)
    for (b, tau) in enumerate(BANDS)
        k >= 2tau + 1 || continue
        w1a = k - tau + 1
        w2a = k - 2tau + 1
        ri = run_of_row_start(core, w2a)
        for u in w2a:k
            rho = core.runs[ri]
            if rho.last < u
                ri += 1
                rho = core.runs[ri]
            end
            zu .= zsum_at(core, u) .- zsum_at(core, u - 1)
            j = k - u
            slot = get!(acc, rho.maskid) do
                (zeros(N, length(BANDS)), zeros(N, length(BANDS)))
            end
            if u >= w1a
                wq = -(tau - 1 - j) / tau
                # Compound indexed broadcast reads its old column too.
                # Use a view on BOTH sides instead of copying that column
                # for every lag; accumulation order and arithmetic stay put.
                qcol = view(slot[1], :, b)
                qcol .+= wq .* zu
            end
            wp = min(j + 1, 2tau - 1 - j) / tau
            pcol = view(slot[2], :, b)
            pcol .+= wp .* zu
        end
    end
    isempty(acc) && return nothing
    regs = collect(keys(acc))
    v = Vector{Matrix{Float64}}(undef, length(regs))
    for (i, mid) in enumerate(regs)
        q_m, p_m = acc[mid]
        mat = Matrix{Float64}(undef, N, REGIME_CHANNELS)
        for b in 1:length(BANDS)
            copyto!(view(mat, :, 2b - 1), view(q_m, :, b))
            copyto!(view(mat, :, 2b), view(p_m, :, b))
        end
        v[i] = mat
    end
    PendingPrototype(regs, v)
end

# ---------------- fold gram updates ----------------
function gram_pair_index(c::Int, c2::Int)
    C = REGIME_CHANNELS
    c <= c2 || ((c, c2) = (c2, c))
    (c - 1) * C + c2 - div(c * (c - 1), 2)
end

const REGIME_PAIR_COUNT = let C = REGIME_CHANNELS
    div(C * (C + 1), 2)
end

function gram_update!(g::FoldGram, rec::RowRecord, sign::Int)
    v = rec.v[1]
    z = rec.target
    for c in 1:REGIME_CHANNELS
        vc = view(v, :, c)
        for c2 in c:REGIME_CHANNELS
            BLAS.ger!(Float64(sign), vc, view(v, :, c2), g.V[gram_pair_index(c, c2)])
        end
        BLAS.ger!(Float64(sign), vc, z, g.Xc[c])
    end
    BLAS.ger!(Float64(sign), z, z, g.R2)
    g.n += sign
    nothing
end

function gram_bytes_for(N::Int)
    (REGIME_PAIR_COUNT * N * N + REGIME_CHANNELS * N * N + N * N) * 8
end

function budget_allows!(core::RawInferenceCore, maskid::Int, fold::Int)
    key = (maskid, fold)
    haskey(core.grams, key) && return true
    alloc = gram_bytes_for(length(core.zsum))
    if core.gram_bytes + alloc > core.gram_budget
        core.resource_limited = true
        return false
    end
    N = length(core.zsum)
    g = FoldGram(0, [zeros(N, N) for _ in 1:REGIME_PAIR_COUNT],
                 [zeros(N, N) for _ in 1:REGIME_CHANNELS], zeros(N, N))
    core.grams[key] = g
    core.gram_bytes += alloc
    true
end

# ---------------- training row append (with fold moves) ----------------
function append_training_row!(core::RawInferenceCore, pending::PendingPrototype,
                              z_target::Vector{Float64}, target_maskid::Int, F::Int)
    oldn = length(core.rows); n = oldn + 1
    oldranges = inference_ranges(oldn, F); ranges = inference_ranges(n, F)
    moved = Set{Int}()
    for fold in 1:F
        new = ranges[fold]; old = oldranges[fold]
        if isempty(old)
            for i in new
                i <= oldn && push!(moved, i)
            end
        else
            for i in first(new):min(last(new), first(old) - 1)
                i <= oldn && push!(moved, i)
            end
            for i in max(first(new), last(old) + 1):last(new)
                i <= oldn && push!(moved, i)
            end
        end
    end
    for i in sort!(collect(moved))
        old = core.membership[i]
        new = findfirst(r -> i in r, ranges)
        old == new && continue
        rec = core.rows[i]
        if rec.aggregated
            key_old = (rec.regs[1], old)
            key_new = (rec.regs[1], new)
            g_old = get(core.grams, key_old, nothing)
            g_old === nothing || gram_update!(g_old, rec, -1)
            g_new = get(core.grams, key_new, nothing)
            # a moved aggregated row keeps its aggregated status; the target
            # gram must exist (created on demand within budget)
            if g_new === nothing
                if budget_allows!(core, rec.regs[1], new)
                    g_new = core.grams[(rec.regs[1], new)]
                else
                    rec.aggregated = false  # demote to materialized on budget
                end
            end
            g_new === nothing || gram_update!(g_new, rec, 1)
        end
        core.membership[i] = new
        core.boundary_moves += 1
    end
    aggregated = (length(pending.regs) == 1 && pending.regs[1] == target_maskid &&
                  budget_allows!(core, target_maskid, F))
    rec = RowRecord(copy(pending.regs), [copy(m) for m in pending.v],
                    copy(z_target), target_maskid, F, aggregated)
    push!(core.rows, rec); push!(core.membership, F)
    if aggregated
        gram_update!(core.grams[(target_maskid, F)], rec, 1)
    end
    nothing
end

# ---------------- return row append ----------------
function append_return!(core::RawInferenceCore, r::AbstractVector{Float64}, F::Int)
    N = length(r); k = core.k + 1
    z = ifelse.(isfinite.(r), r, 0.0)
    mask = Bool[isfinite(x) for x in r]
    if any(x -> !x, mask) && any(mask)
        core.async_seen = true
        core.async_rows += 1
    end
    # run management for row k (zsum still at k-1: new run zbase = zsum_{k-1})
    if isempty(core.runs)
        push!(core.runs, RegimeRun(registry_id!(core, mask), mask, k, k,
             zeros(N), zeros(N)))
    elseif mask != core.runs[end].mask
        prev = core.runs[end]
        prev.zfull .= core.zsum .- prev.zbase
        push!(core.runs, RegimeRun(registry_id!(core, mask), mask, k, k,
             copy(core.zsum), zeros(N)))
    else
        core.runs[end].last = k
    end
    # zsum update (the only cumulative statistic the filter needs)
    core.zsum .+= z
    push!(core.zsum_hist, copy(core.zsum))
    core.runs[end].zfull .= core.zsum .- core.runs[end].zbase  # live zfull after row k
    target_maskid = core.runs[end].maskid
    # training row: feature at k-1 (pending), target z at k
    if core.pending !== nothing
        append_training_row!(core, core.pending, z, target_maskid, F)
    end
    core.pending = k >= WARMUP ? compute_pending(core, k) : nothing
    core.k = k
    core
end

# ---------------- permutation embed (IPO; no replay) ----------------
# All embed helpers return NEW arrays sized for the new coordinate space;
# callers rebind the stored references. Old entries land at their permuted
# positions; the activating asset's historical columns are exact zeros.
function embed_vec(old::Vector{Float64}, pos::Vector{Int}, newN::Int)
    out = zeros(newN)
    for (old_i, new_i) in enumerate(pos)
        out[new_i] = old[old_i]
    end
    out
end
function embed_cols(old::Matrix{Float64}, pos::Vector{Int}, newN::Int)
    out = zeros(newN, size(old, 2))
    for (old_i, new_i) in enumerate(pos), j in axes(old, 2)
        out[new_i, j] = old[old_i, j]
    end
    out
end
function embed_sq(old::Matrix{Float64}, pos::Vector{Int}, newN::Int)
    out = zeros(newN, newN)
    for (oi, ni) in enumerate(pos), (oj, nj) in enumerate(pos)
        out[ni, nj] = old[oi, oj]
    end
    out
end
function embed_mask(old::Vector{Bool}, pos::Vector{Int}, newN::Int)
    out = falses(newN)
    for (old_i, new_i) in enumerate(pos)
        out[new_i] = old[old_i]
    end
    out
end

function embed_active!(state::ExactInferenceState, new_active::Vector{Int})
    core = state.core
    core === nothing && return nothing
    old_active = core.active
    newN = length(new_active)
    newN == length(old_active) && return nothing
    pos = [findfirst(==(a), new_active) for a in old_active]
    # registry rebuild: old maskid -> embedded mask -> new maskid
    old_to_new = Dict{Int,Int}()
    old_mask_of = core.mask_of
    core.maskids = Dict{Vector{Bool},Int}()
    core.mask_of = Vector{Vector{Bool}}()
    for old_mid in 1:length(old_mask_of)
        m = embed_mask(old_mask_of[old_mid], pos, newN)
        new_mid = get!(core.maskids, m) do
            push!(core.mask_of, m)
            length(core.mask_of)
        end
        old_to_new[old_mid] = new_mid
    end
    # runs
    for run in core.runs
        run.mask = embed_mask(run.mask, pos, newN)
        run.zbase = embed_vec(run.zbase, pos, newN)
        run.zfull = embed_vec(run.zfull, pos, newN)
        run.maskid = old_to_new[run.maskid]
    end
    # rows
    for rec in core.rows
        rec.v = [embed_cols(m, pos, newN) for m in rec.v]
        rec.target = embed_vec(rec.target, pos, newN)
        rec.tmask = old_to_new[rec.tmask]
        rec.regs = [old_to_new[m] for m in rec.regs]
    end
    # pending
    if core.pending !== nothing
        core.pending = PendingPrototype(
            [old_to_new[m] for m in core.pending.regs],
            [embed_cols(m, pos, newN) for m in core.pending.v])
    end
    # grams: budget-checked embed or exact release (route-only).
    # embed_active! rebuilds the dict directly - budget_allows!'s haskey
    # shortcut never runs here - so the byte budget is enforced BEFORE
    # any enlarged matrix is allocated. target_bytes is the true held
    # payload after embedding (each FoldGram carries PAIR+C+1 matrices
    # of newN x newN), computed from the dict size and newN - NOT from
    # the historical ledger - so every embed resynchronizes the ledger
    # regardless of drift, and repeated embeds (multi-IPO / insertion
    # reordering) stay consistent.
    # Ledger/payload sync, static proof: within budget each embedded
    # FoldGram holds exactly gram_bytes_for(newN) bytes (PAIR*newN^2 +
    # C*newN^2 + newN^2 Float64s), ng entries sum to target_bytes, and
    # the ledger is set to exactly that; on release the dict is empty
    # and the ledger is 0. Later budget_allows! allocations use
    # length(core.zsum) == newN (zsum is embedded below), so the
    # invariant  gram_bytes == held payload  is preserved across the
    # whole state life, including inference_checkpoint (deepcopy copies
    # both the payload and the ledger value).
    # Complexity: O(1) budget check; embed path O(ng*(PAIR+C+1)*newN^2)
    # (identical to the previous rebuild); release path O(n_rows) flag
    # flips - cheaper than embedding. Nothing on either path can throw,
    # so an embed never leaves a torn state.
    ng = length(core.grams)
    target_bytes = ng * gram_bytes_for(newN)
    if target_bytes <= core.gram_budget
        new_grams = Dict{Tuple{Int,Int},FoldGram}()
        for ((mid, f), g) in core.grams
            nmid = old_to_new[mid]
            new_grams[(nmid, f)] = FoldGram(g.n,
                [embed_sq(M, pos, newN) for M in g.V],
                [embed_sq(M, pos, newN) for M in g.Xc],
                embed_sq(g.R2, pos, newN))
        end
        core.grams = new_grams
        core.gram_bytes = target_bytes   # strict ledger/payload sync
    else
        # Over budget: NO enlarged allocation happens. Release the
        # held Grams (payload -> 0, ledger -> 0) and demote their rows
        # to the exact materialized path: regime_statistics recomputes
        # every demoted row from the embedded row records - the same
        # mathematics, a per-row route. resource_limited records the
        # route-only event; raw history, coordinates and fold
        # membership are untouched, and new rows may aggregate again
        # as budget allows (the IPO-day training row is itself a
        # boundary row - its target opens the new run - so no Gram is
        # created for it on the release day either).
        empty!(core.grams)
        core.gram_bytes = 0
        core.resource_limited = true
        for rec in core.rows
            rec.aggregated = false
        end
    end
    # history and current vectors
    core.zsum = embed_vec(core.zsum, pos, newN)
    for s in 1:length(core.zsum_hist)
        core.zsum_hist[s] = embed_vec(core.zsum_hist[s], pos, newN)
    end
    core.active = copy(new_active)
    core.embeds += 1
    nothing
end

# ---------------- advance ----------------
function advance_exact!(state::ExactInferenceState,new_bar_vector::AbstractVector)
    N=length(state.first_price)
    length(new_bar_vector)==N || throw(DimensionMismatch("new inference bar"))
    bar=Float64.(new_bar_vector)
    all(x->isnan(x)||(isfinite(x)&&x>0),bar) ||
        throw(ArgumentError("prices must be positive finite observations or NaN"))
    logs=log.(bar); t=length(state.bars)+1
    push!(state.bars,bar); push!(state.logs,logs)
    for j in 1:N
        isfinite(logs[j]) && state.first_price[j]==0 && (state.first_price[j]=t)
        for (a,tau) in enumerate(TAUS)
            if t>tau && isfinite(logs[j]) && isfinite(state.logs[t-tau][j])
                delta=logs[j]-state.logs[t-tau][j]
                state.ruler_acc[j,a]+=delta*delta
                state.ruler_cnt[j,a]+=1
            end
        end
    end
    if t>1
        r=logs.-state.logs[t-1]
        if any(isfinite,r)&&!all(isfinite,r)
            state.counters[:async_rows]+=1
        end
        active=sort!(union(state.active,findall(isfinite,r)))
        if active!=state.active
            # coordinate permutation embed: no history replay. The embed
            # counter only advances when a core actually existed to embed;
            # the very first activation creates the core below instead.
            if state.core !== nothing
                embed_active!(state,active)
                state.counters[:embeds]+=1
            end
            state.active=active
        end
        if state.core === nothing
            # The budgets validated by initialize_inference must reach the
            # core: dropping them here silently re-defaults to 512 MiB /
            # 4096 and voids every caller-supplied budget contract (the
            # audited dead-kwarg defect; route-only, never outputs).
            state.core=RawInferenceCore(active,state.F_folds;
                gram_budget=state.gram_budget,
                materialize_row_limit=state.materialize_row_limit)
        end
        append_return!(state.core,r[state.active],state.F_folds)
    end
    state.counters[:advances]+=1
    state
end

function inference_ruler(acc,cnt,ages)
    N=size(acc,1); out=ones(N,length(TAUS)); lt=log.(TAUS)
    for j in 1:N
        usable=[a for a in eachindex(TAUS) if 4TAUS[a]<=ages[j] && cnt[j,a]>0]
        length(usable)>=2 || continue
        vals=[log(sqrt(max(acc[j,a]/cnt[j,a],1e-12))) for a in usable]
        ml=sum(vals)/length(vals); mt=sum(lt[usable])/length(vals)
        num=0.0; den=0.0
        for (i,a) in enumerate(usable)
            delta=lt[a]-mt; num+=(vals[i]-ml)*delta; den+=delta*delta
        end
        H=num/max(den,1e-12)
        out[j,:].=exp.(ml.+H.*(lt.-mt))
    end
    out
end

struct InferenceContractionError <: Exception
    reason::Symbol
end

function apply_R!(out, mask::Vector{Bool}, x::AbstractVector{Float64})
    c = count(mask)
    if c == 0
        fill!(out, 0.0)
        return out
    end
    s = 0.0
    @inbounds for j in eachindex(x)
        out[j] = mask[j] ? x[j] : 0.0
        mask[j] && (s += x[j])
    end
    shift = s / c
    @inbounds for j in eachindex(x)
        mask[j] && (out[j] -= shift)
    end
    out
end

# R * M * R' for symmetric M, O(N^2): no matmul. out and M MUST be distinct
# buffers (out is zeroed on entry).
function rwrap_sym!(out, M::Matrix{Float64}, mask::Vector{Bool}, work1)
    N = size(M, 1)
    c = count(mask)
    fill!(out, 0.0)
    c == 0 && return out
    # term1: M .* (m m')
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            out[j, k] = M[j, k]
        end
    end
    # M * m (matvec)
    fill!(work1, 0.0)
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            work1[j] += M[j, k]
        end
    end
    # term2: -( (m .* (M m)) m' + m ((M m) .* m)' ) / c
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            out[j, k] -= (work1[j] + work1[k]) / c
        end
    end
    # term3: + (m' M m) m m' / c^2
    s = 0.0
    @inbounds for j in 1:N
        mask[j] && (s += work1[j])
    end
    s /= c * c
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            out[j, k] += s
        end
    end
    out
end

# R * M * R' for general M (feature-target cross), O(N^2). out and M MUST
# be distinct buffers (out is zeroed on entry).
function rwrap_gen!(out, M::Matrix{Float64}, mask::Vector{Bool}, work1, work2)
    N = size(M, 1)
    c = count(mask)
    fill!(out, 0.0)
    c == 0 && return out
    # M * m and M' * m
    fill!(work1, 0.0); fill!(work2, 0.0)
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            work1[j] += M[j, k]
            work2[k] += M[j, k]
        end
    end
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            out[j, k] = M[j, k] - (mask[j] ? work2[k] : 0.0) / c - (mask[k] ? work1[j] : 0.0) / c
        end
    end
    s = 0.0
    @inbounds for j in 1:N
        mask[j] && (s += work1[j])
    end
    s /= c * c
    @inbounds for j in 1:N, k in 1:N
        if mask[j] && mask[k]
            out[j, k] += s
        end
    end
    out
end

"""Exact ragged s_perp: batch semantics (X_rel is always finite via zero
embedding, so the count is the full N*(L-tau); the denominator matches
compute_s_perp's N/(N-1) normalization)."""
function regime_s_perp(core::RawInferenceCore, d::AbstractVector{Float64})
    N = length(d)
    L = core.k
    out = zeros(length(BANDS))
    w = zeros(N); x = zeros(N); acc_out = zeros(N)
    zw = zeros(N)
    for (b, tau) in enumerate(BANDS)
        acc = 0.0
        cnt = max(L - tau, 0)
        cnt == 0 && (out[b] = 1e-3; continue)
        for t in (tau + 1):L
            # window sum_j = sum over runs rho of [R^rho (Zw^rho .* d)]_j
            fill!(w, 0.0)
            for i in length(core.runs):-1:1
                rho = core.runs[i]
                rho.last < t - tau + 1 && break
                rho.first > t && continue
                window_z_sum!(zw, core, rho, t - tau + 1, t)
                x .= zw .* d
                apply_R!(acc_out, rho.mask, x)
                w .+= acc_out
            end
            acc += dot(w, w)
        end
        out[b] = sqrt(max(acc / (cnt * max(N - 1, 1)), 1e-6))
    end
    out
end

"""Contract aggregated + materialized Grams with today's metric.
Produces the same statistics shape as fold_sufficient_statistics (and the
previous inference_statistics): per-fold and full xx/xy/yy with
xx = nothing when n < P (dual regime), plus macro Grams from Bm/m."""
function regime_statistics(core::RawInferenceCore, d::AbstractVector{Float64},
                           sp::AbstractVector{Float64}, Bm::AbstractMatrix{Float64},
                           m::AbstractVector{Float64}, ts::AbstractVector{Int}, F::Int)
    N = length(d); C = REGIME_CHANNELS; P = C * N
    n = length(core.rows)
    n == length(ts) || throw(ArgumentError("row/feature count mismatch: rows=$n ts=$(length(ts))"))
    ir = repeat(1.0 ./ max.(sp, 1e-6); inner = 2)
    dd = d * d'
    xx = n >= P ? [zeros(P, P) for _ in 1:F] : nothing
    xy = [zeros(P, N) for _ in 1:F]
    yy = [zeros(N, N) for _ in 1:F]
    work1 = zeros(N); work2 = zeros(N)
    # Two INDEPENDENT buffers: src receives the Hadamard (V/Xc/R2 .* dd'),
    # dst receives the R-wrapped result. Aliasing src and dst would zero
    # the input inside rwrap (fill!(out,0)) and silently produce all-zero
    # Grams - the exact defect the static audit caught.
    blk_src = zeros(N, N)
    blk_dst = zeros(N, N)
    # 1) aggregated interior rows
    for ((maskid, f), g) in core.grams
        mask = core.mask_of[maskid]
        for c in 1:C
            for c2 in c:C
                M = g.V[gram_pair_index(c, c2)]
                good = true
                @inbounds for j in 1:N, k in 1:N
                    blk_src[j, k] = M[j, k] * dd[j, k]
                    isfinite(blk_src[j, k]) || (good = false)
                end
                good || throw(InferenceContractionError(:nonfinite_gram))
                # V^{c,c2} = Σ v_c v_c2' 只在 c==c2 时对称；cross 块必须走
                # 通用收缩器 R·M·R'。对称收缩器会把非对称输入的行/列和差
                # 投影成错误的对称残差（N=2 全真 mask 反例：M=[1 2; 3 4] 的
                # RMR'=0，而 sym 路径给出 [0.5 -0.5; 0.5 -0.5]——mutation
                # 验证实测 0.5，与解析预测一致）。c==c2 的 V 块自身对称，
                # 保留对称收缩器（数学等价且省一半计算）。
                if c == c2
                    rwrap_sym!(blk_dst, blk_src, mask, work1)
                else
                    rwrap_gen!(blk_dst, blk_src, mask, work1, work2)
                end
                s = ir[c] * ir[c2]
                if xx !== nothing
                    cr = (c - 1) * N .+ (1:N); br = (c2 - 1) * N .+ (1:N)
                    xx[f][cr, br] .+= s .* blk_dst
                    c == c2 || (xx[f][br, cr] .+= s .* transpose(blk_dst))
                end
            end
            # feature-target cross (general, non-symmetric)
            Mx = g.Xc[c]
            @inbounds for j in 1:N, k in 1:N
                blk_src[j, k] = Mx[j, k] * dd[j, k]
            end
            rwrap_gen!(blk_dst, blk_src, mask, work1, work2)
            cr = (c - 1) * N .+ (1:N)
            xy[f][cr, :] .+= ir[c] .* blk_dst
        end
        # target gram (symmetric)
        @inbounds for j in 1:N, k in 1:N
            blk_src[j, k] = g.R2[j, k] * dd[j, k]
        end
        rwrap_sym!(blk_dst, blk_src, mask, work1)
        yy[f] .+= blk_dst
    end
    # 2) materialized rows (boundary windows / boundary targets / budget)
    for f in 1:F
        idxs = Int[i for i in 1:n if !core.rows[i].aggregated && core.membership[i] == f]
        isempty(idxs) && continue
        X_b = zeros(length(idxs), P)
        Y_b = zeros(length(idxs), N)
        x = zeros(N); Rx = zeros(N)
        for (li, i) in enumerate(idxs)
            rec = core.rows[i]
            for (ri, mid) in enumerate(rec.regs)
                vm = rec.v[ri]
                mask = core.mask_of[mid]
                for c in 1:C
                    x .= view(vm, :, c) .* d
                    apply_R!(Rx, mask, x)
                    X_b[li, (c - 1) * N .+ (1:N)] .+= ir[c] .* Rx
                end
            end
            x .= rec.target .* d
            apply_R!(Rx, core.mask_of[rec.tmask], x)
            Y_b[li, :] .= Rx
        end
        if xx !== nothing
            xx[f] .+= transpose(X_b) * X_b
        end
        xy[f] .+= transpose(X_b) * Y_b
        yy[f] .+= transpose(Y_b) * Y_b
    end
    all(isfinite, reduce(+, xy)) && all(isfinite, reduce(+, yy)) &&
        (xx === nothing || all(isfinite, reduce(+, xx))) ||
        throw(InferenceContractionError(:nonfinite_gram))
    full_xx = xx === nothing ? nothing : reduce(+, xx)
    full_xy = reduce(+, xy); full_yy = reduce(+, yy)
    # 3) macro Grams (identical construction to the previous engine)
    Xm = Bm[ts, :]; ym = m[ts .+ 1]
    macro_full = (; xx = Xm' * Xm, xy = reshape(Xm' * ym, :, 1), yy = fill(dot(ym, ym), 1, 1), n)
    ranges = inference_ranges(n, F)
    macro_folds = [let Xf = Bm[ts[r], :], yf = m[ts[r] .+ 1]
        (; xx = Xf' * Xf, xy = reshape(Xf' * yf, :, 1), yy = fill(dot(yf, yf), 1, 1), n = length(r)) end
        for r in ranges]
    (; stats = (; ranges, xx, xy, yy, full_xx, full_xy, full_yy),
       macro_stats = (; full = macro_full, folds = macro_folds))
end

function inference_prices(state)
    # One dense panel allocation, rather than one transposed row object per day
    # followed by a concatenation. Preserve row order and missing values.
    T=length(state.bars)
    T>0 || throw(ArgumentError("inference state has no bars"))
    N=length(first(state.bars))
    prices=Matrix{Float64}(undef,T,N)
    for t in 1:T
        row=state.bars[t]
        length(row)==N || throw(DimensionMismatch("inconsistent inference bar width"))
        @inbounds for j in 1:N
            prices[t,j]=row[j]
        end
    end
    prices
end

function inference_fallback!(state,reason)
    state.last_reason=reason
    state.counters[:fallback]+=1
    state.fallback_reasons[reason]=get(state.fallback_reasons,reason,0)+1
end

function _prepare_current!(state::ExactInferenceState;timing=nothing)
    prices=inference_prices(state); T=length(state.bars)
    state.core === nothing && error("No active assets with price history in prefix")
    core=state.core
    n=length(core.rows)
    2<=state.F_folds<=n || throw(ArgumentError("F_folds must leave nonempty training and evaluation rows"))
    materialized=count(r -> !r.aggregated, core.rows)
    if materialized > core.materialize_row_limit
        # Engineering budget route ONLY (configurable, affects routing never
        # outputs): the materialized path is exact; beyond this row limit its
        # solve cost approaches the batch oracle, which is also exact. This
        # is a resource decision, NOT a mathematical limitation - the ragged
        # field remains strictly factorable (e_t(d) = R_t diag(z_t) d).
        inference_fallback!(state, :resource_budget)
        return _prepare_v1(prices;F_folds=state.F_folds,timing,workspace=state.workspace)
    end
    sraw=inference_ruler(state.ruler_acc[state.active,:],state.ruler_cnt[state.active,:],
        T .- state.first_price[state.active] .+ 1)
    try
        s1=sraw[:,1]; d=1.0./max.(s1,1e-6)
        sp=regime_s_perp(core,d)
        all(isfinite,sp) || throw(InferenceContractionError(:nonfinite_metric))
        builder=(X,Y,Bm,m,ts)->regime_statistics(core,d,sp,Bm,m,ts,state.F_folds)
        prep=_prepare_v1(prices;F_folds=state.F_folds,timing,workspace=state.workspace,
            ruler_override=sraw,scale_override=(;s_m=nothing,s_perp=sp),statistics_builder=builder)
        state.last_reason=nothing; state.counters[:fast]+=1
        prep
    catch error
        error isa InferenceContractionError || rethrow()
        inference_fallback!(state,error.reason)
        _prepare_v1(prices;F_folds=state.F_folds,timing,workspace=state.workspace)
    end
end

function solve_current!(state::ExactInferenceState;timing=nothing)
    prep=_prepare_current!(state;timing)
    model=_fit_prepared_v1(prep;ridge_alpha=state.ridge_alpha,F_folds=state.F_folds,
        alpha_initial=state.warm,timing,workspace=state.workspace)
    state.counters[:solves]+=1
    model
end
