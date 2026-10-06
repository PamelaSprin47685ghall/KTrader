using Test, Dates, Random, LinearAlgebra, Statistics
using KTrader

BLAS.set_num_threads(1)     # same as bin/backtest.jl: concurrent days must not fight over BLAS threads

# ── synthetic panel: 9 assets, 1500 sessions ──────────────────────────────────────────────────
#   A2 halts for 15 sessions inside the test window (the model holds it → locked position)
#   A6 lists late and becomes history-eligible inside the test window
#   A7 stops printing for good before the end (never tradable at the end, still held/drifting)
#   A8 never lists (never eligible)
const S, SEED = 40, 3
const HALT = 1460:1474
const FROM_IDX = 1440

function panel(; T = 1500, N = 9)
    rng = MersenneTwister(7)
    f = 0.006 .* randn(rng, T)
    drift = [0.0008, 0.0016, 0.0009, 0.0006, 0.0, 0.0008, 0.0008, 0.0, 0.0007]
    raw = exp.(cumsum(drift' .+ f .+ 0.008 .* randn(rng, T, N), dims = 1))
    raw[1:870, 6] .= NaN
    raw[HALT, 2] .= NaN
    raw[1481:end, 7] .= NaN
    raw[:, 8] .= NaN
    dates = [d for d in Date(2015, 1, 1):Day(1):Date(2030, 1, 1) if dayofweek(d) <= 5][1:T]
    dates, ["A$i" for i in 1:N], raw
end

dates, syms, raw = panel()
b = Bars(dates, syms, raw, raw)
T, N = size(b.adj)

# ── plain sequential reference implementation of the specification, public API only ──────────
function reference(b; from, S, seed)
    T, N = size(b.adj)
    ds = findfirst(>=(from), b.dates):T-1
    K = length(ds)
    W, We, ret, ew = zeros(K, N), zeros(K, N), zeros(K), zeros(K)
    h = he = nothing
    locked_days = locked_days_ew = 0
    for (k, t) in enumerate(ds)
        adj = b.adj[1:t, :]
        free = eligible(adj) .& b.bar[t, :]
        h !== nothing && any(h .* .!free .> 0) && (locked_days += 1)
        he !== nothing && any(he .* .!free .> 0) && (locked_days_ew += 1)
        w = path_kelly(adj; S, rng = MersenneTwister(seed + t), tradable = b.bar[t, :], held = h)
        we = equal_weights(free, he)
        g = b.adj[t+1, :] ./ b.adj[t, :]
        g[.!isfinite.(g)] .= 1.0
        W[k, :], We[k, :] = w, we
        ret[k], ew[k] = dot(w, g) - 1, dot(we, g) - 1
        h, he = w .* g ./ (1 + ret[k]), we .* g ./ (1 + ew[k])
    end
    wealth(r) = [1.0; cumprod(1 .+ r)]
    (dates = b.dates[ds], weights = W, weights_ew = We, ret, ew, locked_days, locked_days_ew,
     wealth = wealth(ret), wealth_ew = wealth(ew))
end

from = dates[FROM_IDX]
bt = backtest(b; from, S, seed = SEED)
K = T - FROM_IDX

@testset "Fixture exercises every case" begin
    @test bt.weights[HALT[1]-FROM_IDX, 2] > 0.01           # the model holds A2 when it halts …
    @test bt.locked_days >= length(HALT) - 1               # … so the position is locked through the halt
    @test bt.locked_days_ew >= length(HALT) - 1
    @test all(bt.weights[:, 8] .== 0) && all(bt.weights_ew[:, 8] .== 0)     # never listed
    @test any(bt.weights_ew[:, 6] .== 0) && any(bt.weights_ew[:, 6] .> 0)   # late listing becomes eligible mid-window
    @test !b.bar[T, 7] && b.bar[T, 1]                      # A7 untradable at the end
end

@testset "backtest ≡ sequential reference (threaded stage 1 = sequential; per-day RNG)" begin
    ref = reference(b; from, S, seed = SEED)
    @info "threads" Threads.nthreads()
    @test bt.dates == ref.dates
    @test bt.weights == ref.weights
    @test bt.weights_ew == ref.weights_ew
    @test bt.ret == ref.ret
    @test bt.ew == ref.ew
    @test bt.wealth == ref.wealth
    @test bt.wealth_ew == ref.wealth_ew
    @test bt.locked_days == ref.locked_days
    @test bt.locked_days_ew == ref.locked_days_ew
    # result does not depend on scheduling: same again
    bt2 = backtest(b; from, S, seed = SEED)
    @test bt2.weights == bt.weights && bt2.ret == bt.ret
end

@testset "Accounting" begin
    @test length(bt.ret) == K && size(bt.weights) == (K, N)
    @test bt.wealth[1] == 1 && all(bt.wealth[2:end] .== bt.wealth[1:end-1] .* (1 .+ bt.ret))
    @test all(bt.wealth_ew[2:end] .== bt.wealth_ew[1:end-1] .* (1 .+ bt.ew))
    for W in (bt.weights, bt.weights_ew)
        @test all(W .>= 0) && all(isapprox.(sum(W, dims = 2), 1; atol = 1e-8))
    end
    # a locked asset keeps its drifted weight: w_t = w_{t-1} g / (1 + r) exactly; free assets share the rest
    nlock = 0
    for k in 2:K, (W, r) in ((bt.weights, bt.ret), (bt.weights_ew, bt.ew))
        t = FROM_IDX + k - 1
        g = b.adj[t, :] ./ b.adj[t-1, :]
        for j in findall(.!b.bar[t, :] .& (W[k-1, :] .> 0))
            @test W[k, j] == W[k-1, j] * g[j] / (1 + r[k-1])
            nlock += 1
        end
    end
    @test nlock >= 2 * (length(HALT) - 1)
    # asset 7 stopped printing: its marking price is flat, so its drifted weight is the only thing that moves
    @test all(b.adj[1482:end, 7] .== b.adj[1481, 7]) && all(.!b.bar[1481:end, 7])
    # return = Σ w g − 1 on marking prices
    k = 5; t = FROM_IDX + k - 1
    @test bt.ret[k] ≈ dot(bt.weights[k, :], [isfinite(x) ? x : 1.0 for x in b.adj[t+1, :] ./ b.adj[t, :]]) - 1
end

@testset "Full-backtest causality: prices after day t+1 cannot change weights ≤ t" begin
    tc = FROM_IDX + 30
    raw2 = copy(raw)
    raw2[tc+2:end, :] .*= exp.(0.4 .* randn(MersenneTwister(99), 1, N))               # one ±40% level shift per asset after day t+1
    b2 = Bars(dates, syms, raw2, raw2)
    @test raw2 != raw
    bt3 = backtest(b2; from, S, seed = SEED)
    k = tc - FROM_IDX + 1
    @test bt3.weights[1:k, :] == bt.weights[1:k, :]
    @test bt3.weights_ew[1:k, :] == bt.weights_ew[1:k, :]
    @test bt3.ret[1:k] == bt.ret[1:k] && bt3.ew[1:k] == bt.ew[1:k]
    @test bt3.weights[k+2:end, :] != bt.weights[k+2:end, :]                           # and the perturbation did matter later
end

@testset "`from` handling" begin
    i = findlast(d -> dayofweek(d) == 5, dates[1:T-6])
    sat = dates[i] + Day(1)
    @test dayofweek(sat) == 6 && !(sat in dates)
    j = findfirst(>=(sat), dates)
    bs = backtest(b; from = sat, S, seed = SEED)
    @test bs.dates[1] == dates[j] && dates[j] > sat && dayofweek(dates[j]) == 1      # next session
    @test length(bs.ret) == T - 1 - j + 1 && bs.dates == dates[j:T-1]
    be = backtest(b; from = dates[i], S, seed = SEED)
    @test be.dates[1] == dates[i] && length(be.ret) == T - 1 - i + 1                  # decides at `from` exactly …
    w0 = path_kelly(b.adj[1:i, :]; S, rng = MersenneTwister(SEED + i), tradable = b.bar[i, :])
    @test be.weights[1, :] == w0                                                      # … on adj[1:from] only
    @test_throws ErrorException backtest(b; from = dates[end] + Day(1), S)   # after the last row
    @test_throws ErrorException backtest(b; from = dates[end], S)           # no next session to earn
    @test Date(2026, 10, 6) - Year(10) == Date(2016, 10, 6)                           # bin/backtest.jl default
end

@testset "Regression: finite returns; held asset without a next mark is an error, not a silent 0%" begin
    @test all(isfinite, bt.ret) && all(isfinite, bt.ew)          # late / never-listed assets (NaN marks) carry zero weight
    adj = copy(b.adj); adj[T, :] .= NaN
    bn = Bars(dates, syms, copy(b.close), adj, b.bar)
    @test_throws ErrorException backtest(bn; from = dates[T-1], S, seed = SEED)
end

@testset "Regression: benchmark lock rule is as robust as allocate" begin
    # everything locked and Σlocked a rounding hair above 1 must hold, not hand out negative weights
    h = [1.0 + 2e-16, 0.0, 0.0]
    @test equal_weights([false, true, true], h) == h
    @test all(equal_weights([false, true, true], [1 - 1e-12, 0.0, 0.0]) .>= 0)
    @test equal_weights([true, true, true], nothing) ≈ fill(1 / 3, 3)
    @test equal_weights([true, false, true, false], [0.25, 0.4, 0.35, 0.0]) ≈ [0.3, 0.4, 0.3, 0.0]
    @test equal_weights([false, false], [0.3, 0.7]) == [0.3, 0.7]
    @test_throws ErrorException equal_weights([false, false], nothing)
end
