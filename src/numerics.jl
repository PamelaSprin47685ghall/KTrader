# Probe E buckets (:OOF_fit, :OOF_predict, :OOF_residual) are appended so every
# pre-existing index keeps its meaning. They account for the out-of-fold stage
# only and are disjoint from each other and from the full-fit buckets:
#   :OOF_fit      per-fold response refits (EB + conditioning inside folds)
#   :OOF_predict  evaluation GEMM X_eval*G' and residual-row construction
#   :OOF_residual res_history allocation, own-res row lists, macro residual series
# Fold refits no longer double-count into :eigen/:EB/:condition, which now
# measure the full-data fit only. Timing only; no computation changes.
const TIMING_BUCKETS = (:prep, :basis, :gram, :eigen, :EB, :condition, :fracFFT, :scenario, :Kelly,
                        :OOF_fit, :OOF_predict, :OOF_residual)

# Optional diagnostic observer at the bucket boundary. This file is the
# single owner of bucket timing, so the boundary hook lives here and only
# here. Contract: when `sink !== nothing`, `timed` notifies
#   sink(:start, bucket, stamp_ns)  immediately BEFORE the bucket's work
#   sink(:stop,  bucket, stamp_ns)  after the accumulated work, in the
# finally, so a failing bucket still closes.
# Failure policy (minimal, existing exceptions only — no registry):
#   * a PRIMARY (bucket work) failure always propagates unchanged; an
#     observer failure at the same boundary is recorded separately as
#     `timing.sink_error` (first one, as a string) and never replaces it;
#   * a bucket whose work SUCCEEDED but whose observer failed RAISES the
#     observer's own error — a run cannot silently succeed while its
#     diagnostics are dead;
#   * notification overhead stays OUT of the measured bucket time: the
#     stamps passed to the sink are notification wall-clock times, while
#     `seconds[i]` accumulates only the span around the work itself.
# Default `sink === nothing` takes a byte-identical fast path: the numeric
# and additive timing semantics are exactly the pre-observer
# implementation. The sink lives on the DecisionTiming instance its CALLER
# owns — never global, never stored in fitted models (V1Model and friends
# never see it), and its scope is the single diagnostic phase that created
# it. A sink should not throw, but if it does the policy above applies.
# Note: :prep/:basis are accumulated by direct addition in predict.jl, not
# through `timed`, so they expose no bucket boundary here — their
# stage-level lines come from the ceiling probes' prepare stage.
mutable struct DecisionTiming
    seconds::Vector{Float64}
    sink::Any
    sink_error::Union{Nothing,String}
end
DecisionTiming() = DecisionTiming(zeros(length(TIMING_BUCKETS)), nothing, nothing)
mutable struct FitWorkspace
    matrices::Dict{Tuple{Symbol,Int},Matrix{Float64}}
    # Prepare-generation binding (minimal, NOT a lease framework): every
    # _prepare_v1 call that reuses this workspace bumps the counter BEFORE
    # touching any buffer, which retroactively invalidates every
    # PreparedProblem holding an older generation of the same owner; solve
    # fails loudly on a mismatch. A single Int in a Ref - no clock, no
    # global registry, no profiler. This is a routing/lifetime fact only:
    # it must never decide mathematical outcomes.
    # CONCURRENCY SCOPE: this is a staleness criterion for SERIAL reuse of
    # one caller-owned workspace, NOT a thread-safety or cross-task sharing
    # license. The production boundary stays task-private single-writer:
    # each worker/checkpoint owns its own FitWorkspace (inference_checkpoint
    # constructs a fresh one; backtest workers build per-block). A caller
    # who concurrently prepares/solves through ONE shared workspace gets no
    # protection here - no atomic, no global mutex, no cross-field
    # atomicity is pretended. Do not build new services for unsupported
    # concurrency modes.
    generation::Base.RefValue{Int}
end
FitWorkspace() = FitWorkspace(Dict{Tuple{Symbol,Int},Matrix{Float64}}(), Ref(0))
function matrix_buffer!(workspace::Union{Nothing,FitWorkspace},name::Symbol,rows::Int,columns::Int;
                        slot=0,grow_rows=false)
    workspace === nothing && return zeros(rows,columns)
    key=(name,slot)
    storage=get(workspace.matrices,key,nothing)
    # Allocate exact dimensions so memory stride strictly equals rows (leading dimension = rows).
    # A non-matching leading dimension causes OpenBLAS Level-3 GEMM/SYRK to degrade by 4.7x!
    if storage === nothing || size(storage, 2) != columns || size(storage, 1) != rows
        storage = Matrix{Float64}(undef, rows, columns)
        workspace.matrices[key]=storage
    end
    storage
end

function timed(f, timing::Union{Nothing,DecisionTiming}, bucket::Symbol)
    timing === nothing && return f()
    i = something(findfirst(==(bucket), TIMING_BUCKETS))
    observer = timing.sink
    if observer === nothing
        # Fast path: transparently passes through f's return value.
        # The old bare `return` after try/finally discarded the try
        # expression's value (returning nothing), breaking every
        # `spectrum=timed(...){...}` / tuple-destructure consumer when
        # timing was non-nothing but sink was nothing (e.g. shadow
        # diagnostics). Fix: return the try expression's value directly.
        start = time_ns()
        try
            return f()
        finally
            timing.seconds[i] += (time_ns() - start) * 1e-9
        end
    end
    # Observed path. The notification stamps are the wall clock of the
    # notifications themselves and their overhead stays OUT of the measured
    # bucket time: `seconds[i]` accumulates only the span around the work.
    observer_failure = nothing
    try
        observer(:start, bucket, time_ns())
    catch err
        observer_failure = err
    end
    start = time_ns()
    primary_failed = false
    try
        f()
    catch err
        primary_failed = true
        rethrow()
    finally
        stop = time_ns()
        timing.seconds[i] += (stop - start) * 1e-9
        if observer_failure === nothing
            try
                observer(:stop, bucket, time_ns())
            catch err
                observer_failure = err
            end
        end
        if observer_failure !== nothing
            timing.sink_error === nothing &&
                (timing.sink_error = sprint(showerror, observer_failure))
            if !primary_failed
                # The bucket's work succeeded but its diagnostics are dead:
                # raise the observer's own error — never a silent success.
                throw(observer_failure)
            end
            # A primary failure is already propagating; the observer's
            # secondary error stays recorded on the timing and never
            # replaces the primary.
        end
    end
end

struct PriceHistoryCache
    log_prices::Matrix{Float64}
    returns::Matrix{Float64}
    first_price::Vector{Int}
    first_return::Vector{Int}
    alive_returns::Vector{Int}
    inv_sqrt_alive::Vector{Float64}
end
function PriceHistoryCache(prices::AbstractMatrix{Float64})
    T, N = size(prices)
    logs = log.(prices)
    returns = diff(logs; dims=1)
    first_price = [something(findfirst(isfinite, view(logs, :, j)), T + 1) for j in 1:N]
    first_return = [something(findfirst(isfinite, view(returns, :, j)), T) for j in 1:N]
    alive_returns = [count(isfinite, view(returns, t, :)) for t in 1:(T - 1)]
    inv_sqrt_alive = [cnt > 0 ? 1.0 / sqrt(cnt) : 0.0 for cnt in alive_returns]
    PriceHistoryCache(logs, returns, first_price, first_return, alive_returns, inv_sqrt_alive)
end

"""SPEC §56 input/cache consistency guard (shared numerics owner).

`_prepare_v1` consumes from a caller-supplied cache: the log-price prefix
`log_prices[1:T, :]`, the return prefix `returns[1:T-1, :]`, `first_return`
(the active-set decision) and `first_price` (the ruler start). This guard
verifies exactly those consumed facts against the caller panel `adj`,
element-by-element, NaN-aware, BEFORE any mathematical consumption:

- shape: the universe width must match; the panel must not exceed the
  cache; the cache's derived arrays must have constructor-consistent shapes
  and lengths (a hand-built malformed cache fails here, not downstream);
- logs: `log(adj[t, j])` must equal `cache.log_prices[t, j]` for every
  prefix cell (finite cells compared with `==`; a NaN price must pair with
  a NaN log; a finite non-positive price has no legal cached log and
  fails);
- returns: `cache.returns[t, j]` must equal the exact difference of the
  already-verified `cache.log_prices` pair — a mutated internal return
  payload is caught even when the log payload is intact;
- first_price / first_return: recomputed on the verified prefix; a future
  first observation (row > T) is legal. Rows beyond T are never READ and
  never required to stay unchanged (no-lookahead: one full cache may be
  lawfully reused for any of its prefixes).

The comparison is O(TN) with no panel-sized allocations. It recomputes the
same quantities the constructor computed (`log.`, `diff`, `findfirst`), so
a cache built from this exact panel prefix passes, while a same-shape
foreign panel (including a pure price-level rescale whose returns nearly
coincide — the reason logs, not returns, must be compared), any
post-construction mutation of the panel OR of the cache payload, and any
NaN<->finite flip is rejected loudly. Mismatched data is never silently
consumed as an override (SPEC §56: fail loudly, not silently mix).

Deliberately not content-checked: `alive_returns` / `inv_sqrt_alive`
(lengths only — `_prepare_v1` does not read their values). The separate
`ruler_stats` argument is NOT a theory choice either: it is an exact
acceleration cache of THIS panel ruler statistics (geometry.jl
PrefixRulerStats), so it gets its own fail-loud provenance guard
(verify_ruler_stats_prefix, below) wired in `_prepare_v1` before the
consumed read points reach ruler_from_stats. One semantic corollary of
the guard passing: the cache-path
active set `findall(<(T), first_return)` equals the no-cache active set
`active_universe_indices(adj)` on this prefix.
"""
function verify_history_cache_prefix(adj::AbstractMatrix{Float64}, cache::PriceHistoryCache)
    T, N = size(adj)
    Tc, Nc = size(cache.log_prices)
    N == Nc || throw(ArgumentError(
        "input/cache mismatch: panel universe width $N != history_cache width $Nc"))
    T <= Tc || throw(ArgumentError(
        "input/cache mismatch: panel has $T rows but history_cache holds $Tc"))
    size(cache.returns, 1) == Tc - 1 && size(cache.returns, 2) == Nc ||
        throw(ArgumentError(
            "input/cache mismatch: history_cache.returns shape inconsistent with log_prices"))
    length(cache.first_price) == Nc && length(cache.first_return) == Nc ||
        throw(ArgumentError(
            "input/cache mismatch: first_price/first_return lengths != universe width"))
    length(cache.alive_returns) == Tc - 1 && length(cache.inv_sqrt_alive) == Tc - 1 ||
        throw(ArgumentError(
            "input/cache mismatch: alive_returns/inv_sqrt_alive lengths != Tc-1"))
    # 1) log-price prefix, element-exact and NaN-aware.
    @inbounds for j in 1:Nc, t in 1:T
        a = adj[t, j]
        c = cache.log_prices[t, j]
        ok = if isnan(a)
            isnan(c)
        elseif a > 0.0
            isfinite(c) && c == log(a)
        else
            false
        end
        ok || throw(ArgumentError(
            "input/cache mismatch: log_prices[$t,$j] does not match log(panel[$t,$j]) " *
            "(foreign panel, mutated panel, or mutated cache payload)"))
    end
    # 2) return prefix: the exact difference of the verified logs. NaN
    #    propagates exactly as in `diff`, so a NaN on either side must pair
    #    with a NaN on the other; finite pairs compare with `==`.
    @inbounds for j in 1:Nc
        for t in 1:(T - 1)
            cr = cache.returns[t, j]
            d = cache.log_prices[t + 1, j] - cache.log_prices[t, j]
            ok = if isnan(cr)
                isnan(d)
            else
                isfinite(d) && cr == d
            end
            ok || throw(ArgumentError(
                "input/cache mismatch: returns[$t,$j] does not match diff(log_prices) " *
                "(mutated cache payload)"))
        end
    end
    # 3) derived metadata, recomputed on the verified prefix only (a
    #    future first observation — row > T — is legal and never read).
    @inbounds for j in 1:Nc
        fp = findfirst(isfinite, view(cache.log_prices, 1:T, j))
        fpc = cache.first_price[j]
        (fp === nothing ? fpc >= T + 1 : fpc == fp) ||
            throw(ArgumentError(
                "input/cache mismatch: first_price[$j] inconsistent with the verified prefix " *
                "(mutated cache metadata)"))
        frp = findfirst(isfinite, view(cache.returns, 1:(T - 1), j))
        frc = cache.first_return[j]
        (frp === nothing ? frc >= T : frc == frp) ||
            throw(ArgumentError(
                "input/cache mismatch: first_return[$j] inconsistent with the verified prefix " *
                "(mutated cache metadata)"))
    end
    nothing
end

"""SPEC §56 ruler-cache provenance guard (shared numerics owner; the
ruler-stats sibling of verify_history_cache_prefix).

_prepare_v1 consumes from a caller-supplied PrefixRulerStats exactly
one read point per (active asset, tau): stats.acc[T, col, a] and
stats.cnt[T, col, a], where T is this panel row count. This guard
verifies those consumed facts against a streaming recomputation from the
panel own log-prices BEFORE ruler_from_stats trusts them:

- shape: acc/cnt must be constructor-consistent (Tc x Nc x |TAUS|,
  Tc >= T, columns in range, first-observation rows >= 1);
- cnt: exact integer equality (a counting fact has no roundoff);
- acc: isapprox(rtol = 64eps(Float64), atol = 0.0) — the same
  production roundoff tolerance initialize_inference already uses
  for its accumulated-statistics comparison; not loosened here;
- non-finite: an acc read point that is NaN/Inf is rejected explicitly
  (these sums must be finite; NaN/Inf is pollution, not roundoff).

No-lookahead: rows beyond T are never read and never required to stay
unchanged — a cache built on a longer lawful history stays reusable for
any of its prefixes. Only ACTIVE columns are verified: inactive columns
are never consumed by ruler_from_stats on this panel, and validating
unconsumed cells would silently widen the guard meaning.

Complexity, stated honestly (this is a correctness guard, NOT a
performance optimization): O(N_active * |TAUS| * T) time with O(1)
extra allocations — the same data-scan order as the no-cache ruler
path, paid once per guarded prepare. The cache remaining benefit is
avoiding the T x N x |TAUS| materialization and the repeated log-space
fit, not avoiding this scan. A foreign same-shape cache (e.g. a
same-shape panel with different per-column log amplitude), a mutated
acc/cnt payload, a lying first-observation row, and NaN/Inf pollution
are all rejected loudly; mismatched data is never silently consumed as
an override (SPEC §56: fail loudly, not silently mix).
"""
function verify_ruler_stats_prefix(x::AbstractMatrix{Float64},
                                   stats::PrefixRulerStats,
                                   columns::AbstractVector{<:Integer},
                                   f::AbstractVector{<:Integer})
    T, N_active = size(x)
    N_active == length(columns) == length(f) ||
        throw(ArgumentError("input/cache mismatch: ruler_stats column/first-observation mapping lengths != active width"))
    all(fi -> fi >= 1, f) ||
        throw(ArgumentError("input/cache mismatch: ruler_stats first-observation rows must be >= 1"))
    Tc, Nc, Ntaus = size(stats.acc)
    size(stats.cnt) == (Tc, Nc, Ntaus) ||
        throw(ArgumentError(
            "input/cache mismatch: ruler_stats cnt shape inconsistent with acc"))
    Ntaus == length(TAUS) ||
        throw(ArgumentError("input/cache mismatch: ruler_stats tau axis != length(TAUS)"))
    T <= Tc || throw(ArgumentError(
        "input/cache mismatch: panel has $T rows but ruler_stats holds $Tc"))
    N_active > 0 || throw(ArgumentError(
        "input/cache mismatch: no active columns to verify"))
    all(c -> 1 <= c <= Nc, columns) ||
        throw(ArgumentError("input/cache mismatch: ruler_stats column index out of range"))
    # Streaming recomputation of the consumed read points (row T, active
    # columns, every tau) from THIS panel log-prices: the same
    # accumulation order as build_prefix_ruler_stats; no T x N x |TAUS|
    # materialization, two scalar accumulators only.
    @inbounds for jj in 1:N_active
        col = columns[jj]
        fj = f[jj]
        for a in eachindex(TAUS)
            tau = TAUS[a]
            acc = 0.0
            cnt = 0
            for t in (fj + tau):T
                x1 = x[t, jj]
                x0 = x[t - tau, jj]
                if isfinite(x1) && isfinite(x0)
                    d = x1 - x0
                    acc += d * d
                    cnt += 1
                end
            end
            c_st = stats.cnt[T, col, a]
            a_st = stats.acc[T, col, a]
            c_st == cnt || throw(ArgumentError(
                "input/cache mismatch: ruler_stats cnt[T,col=$col,tau=$tau] does not match the recomputed prefix count " *
                "(foreign panel, mutated stats payload, or wrong first-observation row)"))
            isfinite(a_st) || throw(ArgumentError(
                "input/cache mismatch: ruler_stats acc[T,col=$col,tau=$tau] is not finite " *
                "(NaN/Inf pollution of a sum that must be finite)"))
            isapprox(a_st, acc; rtol = 64eps(Float64), atol = 0.0) ||
                throw(ArgumentError(
                "input/cache mismatch: ruler_stats acc[T,col=$col,tau=$tau] does not match the recomputed prefix statistic " *
                "(foreign panel, mutated stats payload, or wrong first-observation row)"))
        end
    end
    nothing
end

# A spectral representation of (X'X + alpha I)^-1, including the dual null space.
struct RidgeCovariance
    basis::Matrix{Float64}
    weights::Vector{Float64}
    baseline::Float64
end

const RELATIVE_GAUGE_LOCK = ReentrantLock()
const RELATIVE_GAUGES = Dict{Int,Matrix{Float64}}()
function relative_gauge(N::Int)
    lock(RELATIVE_GAUGE_LOCK) do
        get!(RELATIVE_GAUGES,N) do
            Q=zeros(N,max(N-1,0))
            for k in 1:N-1
                scale=sqrt(k*(k+1))
                Q[1:k,k].=1/scale
                Q[k+1,k]=-k/scale
            end
            Q
        end
    end
end
function covariance_product(V::RidgeCovariance, x::AbstractVecOrMat{Float64})
    z = V.basis' * x
    z .*= V.weights
    out = V.basis * z
    V.baseline == 0 || (out .+= V.baseline .* x)
    out
end
function covariance_right_product(x::AbstractMatrix{Float64}, V::RidgeCovariance)
    z = x * V.basis
    z .*= V.weights'
    out = z * V.basis'
    V.baseline == 0 || (out .+= V.baseline .* x)
    out
end

# Per-call FFT storage is leased by a task, not by threadid(): Julia tasks may migrate.
mutable struct FractionalFFTWorkspace{F,I}
    input::Vector{Float64}
    spectrum::Vector{ComplexF64}
    product::Vector{ComplexF64}
    output::Vector{Float64}
    forward::F
    inverse::I
    kernels::Vector{Vector{ComplexF64}}
    counts::Vector{Vector{Float64}}
end
const FRACTIONAL_FFT_LOCK = ReentrantLock()
const FRACTIONAL_FFT_POOL = Dict{Int,Vector{FractionalFFTWorkspace}}()
function fractional_fft_workspace(L::Int)
    lock(FRACTIONAL_FFT_LOCK) do
        pool = get!(FRACTIONAL_FFT_POOL, L, FractionalFFTWorkspace[])
        !isempty(pool) && return pop!(pool)
        input = zeros(L)
        forward = plan_rfft(input; flags=FFTW.ESTIMATE)
        spectrum = forward * input
        product = similar(spectrum)
        inverse = plan_irfft(spectrum, L; flags=FFTW.ESTIMATE)
        kernels = Vector{Vector{ComplexF64}}()
        counts = Vector{Vector{Float64}}()
        for d in DGRID_V1
            weights = frac_weights(d, L ÷ 2)
            fill!(input, 0.0)
            input[1:length(weights)] .= weights
            push!(kernels, forward * input)
            push!(counts, cumsum(weights))
        end
        FractionalFFTWorkspace(input, spectrum, product, zeros(L), forward, inverse, kernels, counts)
    end
end
function release_fractional_fft_workspace(work::FractionalFFTWorkspace)
    # Do not serialize native plan pointers in a precompiled module image.
    ccall(:jl_generating_output,Cint,()) == 1 && return nothing
    lock(FRACTIONAL_FFT_LOCK) do
        push!(FRACTIONAL_FFT_POOL[length(work.input)], work)
    end
end

"""协方差矩阵的唯一对称 PSD 主平方根（scenario 数值计划的规范耦合）。

数学唯一性根据：Σ=VΛVᵀ 的对称 PSD 平方根 R=V√Λ⁺Vᵀ 是唯一解
（Σ 的谱定理）；简并特征子空间内 V 的任意正交旋转 Q 被 √Λ 的常数
块吸收（Q√ΛQᵀ=√Λ），特征向量的列符号翻转同理被吸收——因此
R 与底层分解器的基选择/符号自由度完全解耦。这比"列首符号约定/
简并排序"强得多：那些只是坐标约定，不构成唯一性证明。

数值契约（Manager 2026-10-07 裁决）：同 Σ（含任意列 sign、正交
旋转、简并与 rank-deficient、微 roundoff 扰动）在同 seed 有限场景
下逐点到 roundoff 一致；不掉小正特征值、不降 rank（无 rank 阈值），
负特征值沿用现有数值裁剪策略（max(λ,0)）。只改工程表示，不改
Gaussian 概率律/秩/资产次序/Kelly 目标。
"""
function principal_sqrt_root(L::AbstractMatrix{Float64})
    # SVD(L)=U·Diag(s)·V'；U·Diag(s)·U' 是 (L·Lᵀ) 的唯一对称 PSD 主根
    # （谱定理），且与 V 无关——L 的右正交变换（列符号翻转、列空间内
    # 旋转，含简并与秩亏）不改变主根。相对先 materialize LLᵀ 再 eigen
    # 的优势：不平方条件数、零奇异值保持零模（无 sqrt(eps) 假噪声）、
    # 保留全部 s（不截秩、无阈值抹小正值）。s≥0 恒成立，无需裁剪。
    F = svd(L)
    Matrix(Symmetric((F.U .* F.S') * F.U'))
end
