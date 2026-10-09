"""
Backtest from calendar date `from` (earlier history is visible to the model, nothing earlier
is traded). Day t: the model sees adj[1:t] (close t included); assets with a real bar at t
are filled at close t, instantly and in any size; assets without one cannot be touched —
their position keeps its (drifted) weight and the tradable assets share what is left
(`allocate`). The book earns the close-to-close return t→t+1 on marking prices. No fees,
financing or slippage (by design).

Benchmark: equal weight over the *same* history-eligible set (it needs the same warm-up
length but does not look at it), rebalanced daily over tradable assets only, locked
positions drifting — the identical lock rule.

Stage 1 (scenarios + unlocked weights) is independent per day → all Julia threads, one
seeded RNG per day. Stage 2 (accounting with locks) is a cheap sequential pass; it only
re-solves Kelly on days where a held asset is locked.
"""
function backtest(b::Bars; from::Date, S = 500, seed = 1, response = true, volmodel = true)
    T, N = size(b.adj)
    i0 = findfirst(>=(from), b.dates)
    (i0 === nothing || i0 == T) && error("no decision day with a next session on/after $from (last row $(b.dates[end]))")
    ds = i0:T-1; K = length(ds)

    plan = Vector{NamedTuple}(undef, K)
    Threads.@threads :greedy for k in 1:K
        t = ds[k]
        mask, R = scenarios(@view(b.adj[1:t, :]); S, rng = MersenneTwister(seed + t), response, volmodel)
        free = mask .& b.bar[t, :]
        plan[k] = (; mask, free, w = any(free) ? allocate(R, mask, free) : nothing,
                   R = free == mask ? nothing : R)
    end

    W, We = zeros(K, N), zeros(K, N)
    ret, ew = zeros(K), zeros(K)
    h = he = nothing; locked_days = locked_days_ew = 0
    for k in 1:K
        t = ds[k]; p = plan[k]
        lk = h === nothing ? zeros(N) : h .* .!p.free
        lke = he === nothing ? zeros(N) : he .* .!p.free
        any(>(0), lk) && (locked_days += 1)
        any(>(0), lke) && (locked_days_ew += 1)
        w = if h === nothing
            p.w === nothing && error("nothing tradable at the first decision")
            p.w
        elseif !any(>(0), lk)
            p.w
        else
            p.R === nothing && error("locked position outside the modelled set")
            allocate(p.R, p.mask, p.free, h)
        end
        we = equal_weights(p.free, he)
        g = b.adj[t+1, :] ./ b.adj[t, :]
        held = (w .> 0) .| (we .> 0)
        all(isfinite, g[held]) || error("held asset without a finite marking return on $(b.dates[t+1])")
        g[.!held] .= 1.0                                 # not yet listed: zero weight anyway
        W[k, :], We[k, :] = w, we
        ret[k], ew[k] = dot(w, g) - 1, dot(we, g) - 1
        h, he = w .* g ./ (1 + ret[k]), we .* g ./ (1 + ew[k])
    end
    wealth(r) = [1.0; cumprod(1 .+ r)]
    (dates = b.dates[ds], symbols = b.symbols, weights = W, weights_ew = We, ret, ew,
     locked_days, locked_days_ew, wealth = wealth(ret), wealth_ew = wealth(ew))
end

"""
Equal weight over `free` assets; positions in non-free assets keep their weight `h`. Like
`allocate`: if nothing is free or nothing is left to share (1 − Σlocked ≤ 1e-9), `h` is held.
"""
function equal_weights(free, h)
    fi = findall(free)
    locked = h === nothing ? zeros(length(free)) : h .* .!free
    if isempty(fi) || 1 - sum(locked) <= 1e-9
        h === nothing && error("nothing tradable and nothing held")
        return copy(h)
    end
    out = copy(locked)
    out[fi] .= (1 - sum(locked)) / length(fi)
    out
end

function summarize(r::AbstractVector; periods = 252)
    w = [1.0; cumprod(1 .+ r)]
    (cagr = expm1(mean(log1p.(r)) * periods), vol = std(r) * sqrt(periods),
     sharpe = mean(r) / std(r) * sqrt(periods),
     maxdd = maximum(1 .- w ./ accumulate(max, w)), final = w[end])
end
