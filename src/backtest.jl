"""
KTrader V1 Correctness: Backtest Engine (Two-Stage High-Performance Architecture)
- Stage 1 (Multi-threaded embarrassingly parallel):
    Fits model, generates posterior scenarios X_t, solves unconstrained Kelly w_t.
    Fully utilizes all CPU threads (6 or 12 cores).
- Stage 2 (Sequential stateful execution matching live semantics):
    Simulates sequential drifting holdings h_t, tracks locked/untradable positions.
    Re-solves locked-risk Kelly in 4 ms using precomputed scenarios only on halt days.
- Zero lookahead, fully causal.
- Benchmark: daily equal weight under identical lock semantics.
"""

function backtest_v1(b::Bars; from::Date, S = 300, seed = 1, ridge_alpha = nothing)
    T, N = size(b.adj)
    i0 = findfirst(>=(from), b.dates)
    (i0 === nothing || i0 == T) && error("no decision day on/after $from")
    ds = i0:T-1
    K = length(ds)
    
    # Preallocated results for Stage 1
    scenarios_list = Vector{Matrix{Float64}}(undef, K)
    unconstrained_w = Vector{Vector{Float64}}(undef, K)
    
    # ----------------------------------------------------
    # Stage 1: Embarrassingly Parallel Model & Scenario Fitting
    # ----------------------------------------------------
    Threads.@threads :greedy for k in 1:K
        t = ds[k]
        sub_adj = @view b.adj[1:t, :]
        rng = Random.MersenneTwister(seed + t)
        
        # Fit model and generate scenarios
        model = fit_v1(sub_adj; ridge_alpha)
        r_hist = diff(log.(sub_adj), dims=1)
        X = generate_scenarios_v1(model, r_hist; S, rng)
        w_raw = kelly_weights_v1(X)
        
        scenarios_list[k] = X
        unconstrained_w[k] = w_raw
    end
    
    # ----------------------------------------------------
    # Stage 2: Sequential Stateful Account Evolution
    # ----------------------------------------------------
    W = zeros(Float64, K, N)
    We = zeros(Float64, K, N)
    ret = zeros(Float64, K)
    ew = zeros(Float64, K)
    
    h = nothing
    he = nothing
    locked_days = 0
    locked_days_ew = 0
    
    for k in 1:K
        t = ds[k]
        is_tradable = b.bar[t, :]
        X = scenarios_list[k]
        w_raw = unconstrained_w[k]
        
        # Check locked positions
        has_locked = false
        if h !== nothing
            lk = h .* .!is_tradable
            if any(>(0.0), lk)
                locked_days += 1
                has_locked = true
            end
        end
        if he !== nothing
            lke = he .* .!is_tradable
            any(>(0.0), lke) && (locked_days_ew += 1)
        end
        
        # Determine actual portfolio weights
        w = if !has_locked || h === nothing
            # If all tradable or first decision, use unconstrained solution directly
            w_raw
        else
            locked = h .* .!is_tradable
            L = sum(locked)
            free_idx = findall(is_tradable)
            if isempty(free_idx) || L >= 1.0 - 1e-6
                copy(h)
            else
                budget = 1.0 - L
                base = X[:, .!is_tradable] * locked[.!is_tradable]
                X_free = X[:, free_idx]
                w_free = kelly_weights_v1(X_free; budget, base)
                out = copy(locked)
                out[free_idx] .= w_free
                out
            end
        end
        
        # Benchmark weights
        we = equal_weights_v1(is_tradable, he)
        
        # Realized gross return
        gross = b.adj[t+1, :] ./ b.adj[t, :]
        gross_clean = ifelse.(isfinite.(gross), gross, 1.0)
        
        ret_k = dot(w, gross_clean) - 1.0
        ew_k  = dot(we, gross_clean) - 1.0
        
        W[k, :]  .= w
        We[k, :] .= we
        ret[k] = ret_k
        ew[k]  = ew_k
        
        # Holdings drift
        h  = (w  .* gross_clean) ./ (1.0 + ret_k)
        he = (we .* gross_clean) ./ (1.0 + ew_k)
    end
    
    wealth(r) = [1.0; cumprod(1.0 .+ r)]
    (dates = b.dates[ds], symbols = b.symbols, weights = W, weights_ew = We,
     ret = ret, ew = ew, locked_days = locked_days, locked_days_ew = locked_days_ew,
     wealth = wealth(ret), wealth_ew = wealth(ew))
end

function equal_weights_v1(free::AbstractVector{Bool}, he)
    N = length(free)
    fi = findall(free)
    locked = he === nothing ? zeros(Float64, N) : he .* .!free
    L = sum(locked)
    
    if isempty(fi) || L >= 1.0 - 1e-6
        return he === nothing ? fill(1.0 / N, N) : copy(he)
    end
    
    out = copy(locked)
    budget = 1.0 - L
    out[fi] .= budget / length(fi)
    out
end

function summarize(r::AbstractVector; periods = 252)
    w = [1.0; cumprod(1.0 .+ r)]
    (cagr = expm1(mean(log1p.(r)) * periods), vol = std(r) * sqrt(periods),
     sharpe = mean(r) / std(r) * sqrt(periods),
     maxdd = maximum(1.0 .- w ./ accumulate(max, w)), final = w[end])
end
