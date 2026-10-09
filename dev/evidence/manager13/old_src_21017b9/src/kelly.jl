"""
KTrader V1 Correctness: Kelly Optimization Layer
- Direct convex optimization: max E_Q [log(w' X)]
- Simplex constraints: w >= 0, sum(w) = 1 (leverage 1, fully invested, no cash asset)
- Exact Locked-Risk Kelly: locked assets carry their true scenario gross returns
    base_s = Σ_{j ∈ locked} w_j X_{s, j}
- Strict theory: NO solver fallback to mean-variance Kelly.
- Solver: Clarabel.Optimizer via Convex.jl
"""

"""
Exact Kelly weights on the asset simplex.
X: S × n matrix of gross return scenarios for free assets.
base: S-vector of wealth already locked in untradable assets across scenarios.
budget: Remaining capital to allocate (1 - Σ w_locked).
"""
function kelly_weights_v1(X::AbstractMatrix{Float64}; budget = 1.0, base = nothing)
    S, n = size(X)
    X_safe = max.(X, 1e-4) # numerical positivity
    w = Variable(n)
    
    wealth = base === nothing ? X_safe * w : X_safe * w + base
    prob = maximize(sum(log(wealth)) / S, [w >= 0, sum(w) == budget])
    solve!(prob, Clarabel.Optimizer; silent = true)
    
    if !(prob.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL))
        error("Clarabel Kelly solver failed with status $(prob.status). No mean-variance fallback permitted by theory.")
    end
    
    raw_w = clamp.(vec(evaluate(w)), 0.0, budget)
    raw_w .* (budget / max(sum(raw_w), 1e-12))
end

"""
Complete Path Kelly V1 decision function:
Price history adj (T × N) -> Target weights w (length N, fully invested)
"""
function path_kelly_v1(adj::AbstractMatrix{Float64}; S = 300, rng = Random.MersenneTwister(1), ridge_alpha = 10.0,
                       tradable = nothing, held = nothing)
    N = size(adj, 2)
    tr = tradable === nothing ? trues(N) : tradable
    
    model = fit_v1(adj; ridge_alpha)
    r_hist = diff(log.(adj), dims=1)
    X = generate_scenarios_v1(model, r_hist; S, rng)
    
    # If all assets are tradable or no prior holdings
    if all(tr) || held === nothing
        return kelly_weights_v1(X)
    end
    
    # Untradable/locked positions stay fixed at held weight
    locked = held .* .!tr
    L = sum(locked)
    free_idx = findall(tr)
    
    if isempty(free_idx) || L >= 1.0 - 1e-6
        return copy(held)
    end
    
    budget = 1.0 - L
    
    # Exact locked-risk Kelly: locked assets carry their true scenario return shocks
    # base_s = Σ_{j ∈ locked} w_j * X_{s, j}
    base = X[:, .!tr] * locked[.!tr]
    
    X_free = X[:, free_idx]
    w_free = kelly_weights_v1(X_free; budget, base)
    
    out = copy(locked)
    out[free_idx] .= w_free
    out
end

const path_kelly = path_kelly_v1
