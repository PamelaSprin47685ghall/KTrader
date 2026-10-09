"""
Lazy OOF ResidualOracle — the production OOF residual path.

predict.jl's `solve` constructs this oracle as the solved model's
`res_history`; the pre-lazy dense formula survives only as the TEST-ONLY
reference `dense_oof_residuals` at the bottom of this file. It is
measurement-free, model-free plumbing: the oracle reproduces that dense
OOF residual formula on demand, from inputs frozen at construction time.

Contract:
- Inputs are FROZEN copies taken at construction: r, s1, s_perp, Bm are
  copied, X_rel is reduced to a prefix-sum table `field_sums` (O(TN), never a
  T×P design copy). Nothing is shared with a FitWorkspace or with a
  history-cache view that later mutates. `ResponseOperator` fold models are
  immutable structs owned exclusively by the oracle after the solve that
  produced them.
- Row identity: row idx (1..n_res) is global day t = ts_total[idx]; its
  residual target is the observed return row r[t+1, :]. The observed set of a
  row is O = {j : isfinite(r[t+1,j])}, c = |O| — the row's OWN historical
  observation mask, evaluated at its own historical date. s1 is the CURRENT
  decision day's metric (from this solve's prep) and belongs to every row's
  scale mapping; the two must never be conflated.
- Fold membership: idx belongs to the unique fold f with idx in ranges[f]
  (contiguous cover of 1:n_res, verified at construction).
- Per-cell formula (identical symbol-for-symbol to the dense path):
    res[idx,j] = r[t+1,j] - (mu_macro/sqrt(c) + mu_rel_j - sum(mu_rel)/N) * s1[j]
  for j in O, NaN otherwise. mu_rel = G_f * x_t (fold model's G_c_mean times
  the design row), mu_macro = dot(Bm[t,:], G_f.G_macro).
- Design row x_t is rebuilt from field_sums with exactly fill_design_matrix!'s
  indexing: for each band (b,tau) with t >= 2tau+1,
    c0 = (field_sums[t+1,j] - field_sums[t+1-tau,j]) / tau
    c1 = (field_sums[t+1-tau,j] - field_sums[t+1-2tau,j]) / tau
    x[(b-1)*2N + j]   = -(X_rel[t,j] - c0) * inv_scale_b
    x[(b-1)*2N + N+j] = (c0 - c1) * inv_scale_b
  where X_rel[t,j] = field_sums[t+1,j] - field_sums[t,j] and inv_scale_b =
  1/max(s_perp[b], 1e-6). Each row touches only the window
  [t+1-2*tau_max, t+1] = [t-255, t+1] of the field (tau_max = 128); the full
  prefix table is the O(TN) frozen acceleration of those windows.
- Scalar macro residual per row (algebraically identical to the dense row
  sum scaled by 1/sqrt(c) — the pre-lazy macro-series formula; verified
  term by term against dense_macro_series in residual_oracle_tests.jl):
  with w = s1 .* mask, sw = sum(w),
    c > 0: e[idx] = sum(r[O])/sqrt(c) - mu_macro*sw/c - dot(h, x_t)/sqrt(c)
           where h = G_f' * (w - ones(N)*sw/N), cached per (fold, mask)
           WITHIN one macro_residual_series call (never across calls).
    c == 0: e[idx] = 0.0   (matches the dense path's `count>0 ? ... : 0.0`).
  This is O(P) per row — no N-dimensional prediction is ever materialized for
  the scalar series. The full-history scalar series is recomputed on every
  call ("fractional still recomputes scalars").
- Full N-dimensional residual vectors are computed ONLY for explicitly
  requested rows (residual_row / residual_rows); residual_rows groups rows by
  fold and uses one GEMM per fold group, the same call shape as the dense
  path's X_eval * G_c_mean'.
- own rows are derived from the raw observation mask alone:
    own[j] = { idx : isfinite(r_frozen[ts_total[idx]+1, j]) }
  No prediction is involved.
- The oracle is a read-only AbstractMatrix{Float64}: size, getindex, column
  slicing, collect and broadcasting all work through the standard machinery
  with unchanged value semantics. There is NO internal shared mutable cache;
  every query recomputes (scenario wiring will keep its unique-row cache
  local to the scenario function, never on the model).
- Sampling law and rng order are NOT touched here; they stay exactly as in
  generate_scenarios_v1 (the production scenario generator).

Floating-point honesty: row evaluation and the scalar contraction do not
reproduce the dense path's summation order bit-for-bit; agreement is expected
at roundoff scale and must be checked against the dense reference at the
existing tolerances (atol=1e-10, rtol=1e-8 style), not claimed as ULP-exact.
"""

struct ResidualOracle <: AbstractMatrix{Float64}
    fold_models::Vector{ResponseOperator}
    ranges::Vector{UnitRange{Int}}
    ts_total::Vector{Int}
    r_frozen::Matrix{Float64}
    s1::Vector{Float64}
    s_perp::Vector{Float64}
    Bm_frozen::Matrix{Float64}
    field_sums::Matrix{Float64}
    N::Int
    n_res::Int
    T::Int
end

function ResidualOracle(fold_models, ranges, ts_total, r, s1, s_perp, Bm, X_rel)
    T, N = size(X_rel)
    length(fold_models) == length(ranges) || throw(DimensionMismatch("one fold model per fold"))
    size(r, 1) >= T - 1 && size(r, 2) == N || throw(DimensionMismatch("r must have >= T-1 rows and N columns"))
    size(Bm, 1) >= T || throw(DimensionMismatch("Bm must cover T rows"))
    length(s1) == N || throw(DimensionMismatch("s1 length"))
    length(s_perp) == length(BANDS) || throw(DimensionMismatch("s_perp length"))
    n = length(ts_total)
    n == 0 && throw(ArgumentError("no residual rows"))
    # X_rel carries return rows (T_price-1); caller ts_total is WARMUP:T_price-2.
    # With T = size(X_rel,1) = T_price-1, the legal upper bound is T-1 = T_price-2.
    all(t -> WARMUP <= t <= T - 1, ts_total) || throw(ArgumentError("ts_total must lie in WARMUP:T-1 (return-row indexing)"))
    issorted(ts_total) && allunique(ts_total) || throw(ArgumentError("ts_total must be strictly increasing"))
    vcat(ranges...) == 1:n || throw(ArgumentError("ranges must contiguously cover 1:n_res"))
    # Freeze: prefix sums of the cumulative relative field, built with the
    # same accumulation order as build_X_rel_stacked's `sums`.
    sums = Matrix{Float64}(undef, T + 1, N)
    sums[1, :] .= 0.0
    @inbounds for j in 1:N
        acc = 0.0
        for t in 1:T
            acc += X_rel[t, j]
            sums[t + 1, j] = acc
        end
    end
    ResidualOracle(copy(fold_models), copy(ranges), copy(ts_total),
        Matrix{Float64}(r), Vector{Float64}(s1), Vector{Float64}(s_perp),
        Matrix{Float64}(Bm), sums, N, n, T)
end

Base.size(o::ResidualOracle) = (o.n_res, o.N)
Base.IndexStyle(::Type{ResidualOracle}) = Base.IndexCartesian()

"""Fold owning residual row idx (contiguous ranges verified at construction)."""
function fold_of(o::ResidualOracle, idx::Integer)
    1 <= idx <= o.n_res || throw(BoundsError(o, (idx, 1)))
    f = findfirst(r -> idx in r, o.ranges)
    f === nothing && throw(ArgumentError("row $idx belongs to no fold"))
    f
end

"""Rebuild design row for global day t into x (length 2*length(BANDS)*N),
using exactly fill_design_matrix!'s indexing into the frozen prefix table."""
function design_row!(x::AbstractVector{Float64}, o::ResidualOracle, t::Int)
    N = o.N
    fill!(x, 0.0)
    @inbounds for (b, tau) in enumerate(BANDS)
        two = 2tau
        t >= two + 1 || continue
        offset = (b - 1) * 2N
        iscale = 1.0 / max(o.s_perp[b], 1e-6)
        itau = 1.0 / tau
        for j in 1:N
            c0 = (o.field_sums[t + 1, j] - o.field_sums[t + 1 - tau, j]) * itau
            c1 = (o.field_sums[t + 1 - tau, j] - o.field_sums[t + 1 - two, j]) * itau
            xtj = o.field_sums[t + 1, j] - o.field_sums[t, j]
            x[offset + j] = -(xtj - c0) * iscale
            x[offset + N + j] = (c0 - c1) * iscale
        end
    end
    x
end

"""Full N-dimensional OOF residual vector for row idx (NaN where unobserved),
recomputed on every call — no shared mutable cache on the oracle."""
function residual_row(o::ResidualOracle, idx::Integer)
    1 <= idx <= o.n_res || throw(BoundsError(o, (idx,)))
    t = o.ts_total[idx]
    model = o.fold_models[fold_of(o, idx)]
    N = o.N
    x = zeros(2 * length(BANDS) * N)
    design_row!(x, o, t)
    mu_rel = model.G_c_mean * x
    mu_macro = dot(view(o.Bm_frozen, t, :), model.G_macro)
    rt = view(o.r_frozen, t + 1, :)
    c = count(isfinite, rt)
    inv_sqrt = c > 0 ? 1.0 / sqrt(c) : 0.0
    shift = sum(mu_rel) / N
    base = mu_macro * inv_sqrt
    out = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        v = rt[j]
        out[j] = isfinite(v) ? v - (base + mu_rel[j] - shift) * o.s1[j] : NaN
    end
    out
end

"""Batch residual rows (idxs -> length(idxs) × N, NaN where unobserved).
Rows are grouped per fold and evaluated with one GEMM per group — the same
call shape as the dense path's X_eval * G_c_mean'. Pure function of (o, idxs)."""
function residual_rows(o::ResidualOracle, idxs::AbstractVector{<:Integer})
    length(idxs) == 0 && return fill(NaN, 0, o.N)
    all(i -> 1 <= i <= o.n_res, idxs) || throw(BoundsError(o, (0,)))
    out = fill(NaN, length(idxs), o.N)
    N = o.N
    P = 2 * length(BANDS) * N
    for f in 1:length(o.ranges)
        local_rows = Int[]
        for k in eachindex(idxs)
            idxs[k] in o.ranges[f] && push!(local_rows, k)
        end
        isempty(local_rows) && continue
        model = o.fold_models[f]
        X = zeros(length(local_rows), P)
        ts_f = [o.ts_total[idxs[k]] for k in local_rows]
        for (li, t) in enumerate(ts_f)
            design_row!(view(X, li, :), o, t)
        end
        MU = X * model.G_c_mean'                       # |grp| × N GEMM
        mu_m = o.Bm_frozen[ts_f, :] * model.G_macro    # |grp| × 1, same shape as dense mul!
        @inbounds for (li, k) in enumerate(local_rows)
            t = ts_f[li]
            rt = view(o.r_frozen, t + 1, :)
            c = count(isfinite, rt)
            inv_sqrt = c > 0 ? 1.0 / sqrt(c) : 0.0
            shift = sum(view(MU, li, :)) / N
            base = mu_m[li] * inv_sqrt
            for j in 1:N
                v = rt[j]
                if isfinite(v)
                    out[k, j] = v - (base + MU[li, j] - shift) * o.s1[j]
                end
            end
        end
    end
    out
end

"""Scalar macro residual series (n_res), recomputed on every call via the
verified (fold, mask) h-contraction — O(P) per row, no N-dim predictions.
h is cached only WITHIN this call, keyed by (fold, observed mask)."""
function macro_residual_series(o::ResidualOracle)
    N = o.N
    e = zeros(Float64, o.n_res)
    hcache = Dict{Tuple{Int,Vector{Bool}},Vector{Float64}}()
    onesN = fill(1.0, N)
    x = zeros(2 * length(BANDS) * N)
    # Reuse the lookup mask, but NEVER put this mutable scratch in the Dict.
    # Only a new (fold, mask) owns a copied key; repeated masks allocate none.
    mask = Vector{Bool}(undef, N)
    for idx in 1:o.n_res
        t = o.ts_total[idx]
        rt = view(o.r_frozen, t + 1, :)
        c = 0
        sumr = 0.0
        @inbounds for j in 1:N
            v = rt[j]
            m = isfinite(v)
            mask[j] = m
            if m
                c += 1
                sumr += v
            end
        end
        c == 0 && (e[idx] = 0.0; continue)
        sw = 0.0
        @inbounds for j in 1:N
            mask[j] && (sw += o.s1[j])
        end
        f = fold_of(o, idx)
        model = o.fold_models[f]
        h = get(hcache, (f, mask), nothing)
        if h === nothing
            w = zeros(N)
            @inbounds for j in 1:N
                mask[j] && (w[j] = o.s1[j])
            end
            h = model.G_c_mean' * (w .- onesN .* (sw / N))
            hcache[(f, copy(mask))] = h
        end
        design_row!(x, o, t)
        mu_macro = dot(view(o.Bm_frozen, t, :), model.G_macro)
        e[idx] = sumr / sqrt(c) - mu_macro * sw / c - dot(h, x) / sqrt(c)
    end
    e
end

"""Per-asset valid residual rows, derived from the raw observation mask alone
(no prediction is involved): own[j] = {idx : isfinite(r[ts_total[idx]+1, j])}."""
function own_residual_rows(o::ResidualOracle)
    out = Vector{Vector{Int}}(undef, o.N)
    for j in 1:o.N
        rows_j = Int[]
        for idx in 1:o.n_res
            isfinite(o.r_frozen[o.ts_total[idx] + 1, j]) && push!(rows_j, idx)
        end
        out[j] = rows_j
    end
    out
end

"""Materialize the full n_res × N residual matrix (diagnostic path; the dense
reference for tests). Batch-evaluated per fold."""
materialize(o::ResidualOracle) = residual_rows(o, collect(1:o.n_res))

function Base.getindex(o::ResidualOracle, idx::Integer, j::Integer)
    Base.checkbounds(o, idx, j)
    t = o.ts_total[idx]
    isfinite(o.r_frozen[t + 1, j]) || return NaN   # fast path: unobserved cell
    residual_row(o, idx)[j]
end

"""O(1) observation-mask query: is residual cell (idx, j) observed, i.e.
isfinite(r[ts_total[idx]+1, j])? Scenario row collection uses this instead of
getindex so that the collection pass never triggers row evaluation."""
observed(o::ResidualOracle, idx::Integer, j::Integer) =
    (Base.checkbounds(o, idx, j); isfinite(o.r_frozen[o.ts_total[idx] + 1, j]))

"""Dense old-formula OOF residuals — TEST-ONLY explicit reference entry point.

Replicates the pre-lazy predict.jl computation exactly: per-fold
X_eval * G_c_mean' GEMM over the full evaluation block, per-cell residual
construction with the (mu_macro/sqrt(c) + mu_rel_j - shift)*s1_j prediction.
This is the oracle of truth for tests; production never calls it. Both batch
and incremental engines use the lazy path now, so without this reference the
suite would compare the new method against itself."""
function dense_oof_residuals(prep, fold_models)
    stats = prep.stats
    N = prep.N
    n_res = prep.n_res
    length(fold_models) == length(stats.ranges) || throw(DimensionMismatch("one fold model per fold"))
    res = fill(NaN, n_res, N)
    for fold in eachindex(fold_models)
        fold_eval = stats.ranges[fold]
        model = fold_models[fold]
        X_eval = view(prep.X_rel_stacked, fold_eval, :)   # n_eval × P
        mu_rel = Matrix{Float64}(undef, length(fold_eval), N)
        BLAS.gemm!('N', 'T', 1.0, X_eval, model.G_c_mean, 0.0, mu_rel)
        eval_ts = prep.ts_total[fold_eval]
        mu_m = Matrix{Float64}(undef, length(fold_eval), 1)
        mul!(mu_m, view(prep.B_m, eval_ts, :), model.G_macro)
        inv_N = 1.0 / N
        for li in 1:length(fold_eval)
            t = eval_ts[li]
            cnt = count(isfinite, view(prep.r, t + 1, :))
            inv_sqrt = cnt > 0 ? 1.0 / sqrt(cnt) : 0.0
            macro_shift = mu_m[li, 1] * inv_sqrt
            sum_row = 0.0
            for j in 1:N
                sum_row += mu_rel[li, j]
            end
            shift = sum_row * inv_N
            for j in 1:N
                rt = prep.r[t + 1, j]
                if isfinite(rt)
                    prediction = (macro_shift + mu_rel[li, j] - shift) * prep.s1[j]
                    res[fold_eval[li], j] = rt - prediction
                end
            end
        end
    end
    res
end
