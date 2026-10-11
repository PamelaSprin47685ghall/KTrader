# Optional file adapter. Included by the CLI, not by the numerical core.
# Loading prices must not import the frozen KTrader model or its workloads.
using CSV

function load_market_facts(dir::AbstractString)
    tables = [CSV.File(joinpath(dir, name * ".csv"); normalizenames=false)
              for name in ("close", "adj")]
    close_table, adj_table = tables
    names = propertynames(close_table)
    names == propertynames(adj_table) ||
        error("load_market_facts: close/adj columns differ in identity or order")
    :date in names || error("load_market_facts: missing date column")
    symbols = String.(filter(!=(:date), names))
    isempty(symbols) && error("load_market_facts: no asset columns")
    length(unique(symbols)) == length(symbols) || error("load_market_facts: duplicate symbols")
    dates = Date.(close_table.date)
    dates == Date.(adj_table.date) || error("load_market_facts: close/adj dates differ")
    isempty(dates) && error("load_market_facts: empty calendar")
    all(>(Day(0)), diff(dates)) || error("load_market_facts: dates must be strictly increasing")
    T, N = length(dates), length(symbols)
    raw = [Matrix{Float64}(undef, T, N) for _ in 1:2]
    for (k, table) in enumerate(tables), (j, symbol) in enumerate(symbols)
        col = table[Symbol(symbol)]
        for i in 1:T
            v = ismissing(col[i]) ? NaN : Float64(col[i])
            (isnan(v) || (isfinite(v) && v > 0)) ||
                error("load_market_facts: invalid $(k == 1 ? "close" : "adj") price for $symbol at $(dates[i]): $v")
            raw[k][i, j] = v
        end
    end
    observed = BitMatrix(isfinite.(raw[1]) .& isfinite.(raw[2]))
    # Carry for account valuation only. observed remains the real-bar mask;
    # signal_prices restores NaN on every absent observation.
    marked = [fill(NaN, T, N) for _ in 1:2]
    for k in 1:2, j in 1:N
        last = NaN
        for i in 1:T
            observed[i, j] && (last = raw[k][i, j])
            marked[k][i, j] = last
        end
    end
    MarketFacts(dates, symbols, marked[1], marked[2], observed)
end
