"""
KTrader V0.95: Backtest Engine
- Live/Backtest Isomorphism: tracks real drifted holdings h_{t, j}
- Strict separation of Marking Prices and Signal Prices:
    Signal prices strictly NaN on untradable days (no fake zero returns)
- Stage 1: multi-threaded embarrassingly parallel scenario generation
- Stage 2: sequential stateful account evolution
- Exact Locked-Risk Kelly: locked assets carry their true scenario return shocks
- Benchmark: daily equal-weight rebalancing under identical lock semantics
"""

function backtest_v1(b::Bars; from::Date, S = 300, seed = 1, ridge_alpha = nothing)
    T, N = size(b.adj)
    i0 = findfirst(>=(from), b.dates)
    (i0 === nothing || i0 == T) && error("no decision day on/after $from")
    ds = i0:T-1
    K = length(ds)
    
    # Pre-extract signal prices (strictly NaN when bar[t, j] == false)
    sig_prices = signal_prices(b)
    
    scenarios_list = Vector{Matrix{Float64}}(undef, K)
    unconstrained_w = Vector{Vector{Float64}}(undef, K)
    
    # ----------------------------------------------------
    # Stage 1: Embarrassingly Parallel Model & Scenario Fitting
    # ----------------------------------------------------
    Threads.@threads :greedy for k in 1:K
        t = ds[k]
        sub_signal = @view sig_prices[1:t, :]
        rng = Random.MersenneTwister(seed + t)
        is_tr = b.bar[t, :]
        
        # Fit model on signal prices and generate scenarios
        model = fit_v1(sub_signal; ridge_alpha)
        r_hist = diff(log.(sub_signal), dims=1)
        X = generate_scenarios_v1(model, r_hist; S, rng)
        
        # Stage 1: Solve Kelly strictly over free (tradable) assets
        free_idx = findall(is_tr)
        w_raw = zeros(Float64, N)
        if !isempty(free_idx)
            X_free = X[:, free_idx]
            w_free = kelly_weights_v1(X_free)
            w_raw[free_idx] .= w_free
        else
            w_raw .= fill(1.0 / N, N)
        end
        
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
        
        w = if !has_locked || h === nothing
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
        
        we = equal_weights_v1(is_tradable, he)
        
        # Realized gross return on MARKING prices
        gross = b.adj[t+1, :] ./ b.adj[t, :]
        gross_clean = ifelse.(isfinite.(gross), gross, 1.0)
        
        ret_k = dot(w, gross_clean) - 1.0
        ew_k  = dot(we, gross_clean) - 1.0
        
        W[k, :]  .= w
        We[k, :] .= we
        ret[k] = ret_k
        ew[k]  = ew_k
        
        # Drift holdings
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
