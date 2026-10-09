"""
Daily bars. `close`/`adj` (split- / total-return-adjusted) are the *marking* prices: NaN
before an asset's first bar, carried forward from the last real bar afterwards. `bar[t,j]`
is true iff a real bar printed that day — i.e. the asset was tradable. The 4-argument
constructor takes raw matrices (NaN = no bar) and derives the rest.
"""
struct Bars
    dates::Vector{Date}
    symbols::Vector{String}
    close::Matrix{Float64}
    adj::Matrix{Float64}
    bar::BitMatrix
end

function carry(raw, bar)
    out = fill(NaN, size(raw))
    for j in axes(raw, 2)
        last = NaN
        for t in axes(raw, 1)
            bar[t, j] && (last = raw[t, j])
            out[t, j] = last
        end
    end
    out
end

function Bars(dates, symbols, rawclose, rawadj)
    bar = BitMatrix(isfinite.(rawclose) .& isfinite.(rawadj))
    Bars(dates, symbols, carry(rawclose, bar), carry(rawadj, bar), bar)
end

"""
Signal prices for causal theoretical inference:
Strictly NaN when an asset did not trade on day t (bar[t, j] == false).
Never injects fake zero-returns r = 0 from carried marking prices into the model.
"""
signal_prices(b::Bars) = ifelse.(b.bar, b.adj, NaN)

load_universe(path) = [uppercase(strip(l)) for l in eachline(path)
                       if !isempty(strip(l)) && !startswith(strip(l), "#")]

ny_now() = now(tz"America/New_York")

"A daily bar is final only once its date has passed, or after 17:00 New York (Yahoo's final print)."
is_final(d::Date, asof) = d < Date(asof) || (d == Date(asof) && hour(asof) >= 17)

http_json(url) = JSON3.read(HTTP.get(url, ["User-Agent" => "Mozilla/5.0"]; retry = true, read_idle_timeout = 30).body)

function fetch_symbol(sym; asof = ny_now(), getjson = http_json)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$sym?period1=0&period2=$(round(Int, time()))&interval=1d&events=div%2Csplit"
    res = getjson(url).chart.result[1]
    off = res.meta.gmtoffset
    dates = [Date(unix2datetime(t + off)) for t in res.timestamp]
    c = res.indicators.quote[1].close
    a = res.indicators.adjclose[1].adjclose
    Dict(d => (Float64(ci), Float64(ai)) for (d, ci, ai) in zip(dates, c, a)
         if ci !== nothing && ai !== nothing && is_final(d, asof))
end

yahoo_symbol(s) = replace(s, "." => "-")

"""
Download all symbols onto the union of real sessions (a date counts only if at least half
of the already-listed symbols printed a bar, which drops unpublished partial days).
Missing bars stay NaN in the raw matrices; `Bars` marks them non-tradable.
"""
function download_bars(symbols; fetch = fetch_symbol, asof = ny_now())
    per = [fetch(yahoo_symbol(s); asof) for s in symbols]
    alldates = sort!(collect(union((keys(p) for p in per)...)))
    firsts = [minimum(keys(p)) for p in per]
    dates = filter(alldates) do d
        count(p -> haskey(p, d), per) * 2 >= count(<=(d), firsts)
    end
    raw(k) = reduce(hcat, [[haskey(p, d) ? p[d][k] : NaN for d in dates] for p in per])
    Bars(dates, collect(String, symbols), raw(1), raw(2))
end

function save_bars(b::Bars, dir)
    mkpath(dir)
    for (name, M) in (("close", ifelse.(b.bar, b.close, NaN)), ("adj", ifelse.(b.bar, b.adj, NaN)))
        cols = (; date = b.dates, (Symbol(s) => M[:, j] for (j, s) in enumerate(b.symbols))...)
        CSV.write(joinpath(dir, name * ".csv"), cols)
    end
end

function load_bars(dir)
    fc, fa = CSV.File(joinpath(dir, "close.csv")), CSV.File(joinpath(dir, "adj.csv"))
    syms = String.(filter(!=(:date), propertynames(fc)))
    mat(f) = reduce(hcat, [Vector{Float64}(f[Symbol(s)]) for s in syms])
    Bars(Vector{Date}(fc.date), syms, mat(fc), mat(fa))
end
