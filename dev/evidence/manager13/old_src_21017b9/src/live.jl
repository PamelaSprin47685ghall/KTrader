"""
Live loop state. During the session the current price of each symbol is the *preview
close* of today: a TOTAL-RETURN price appended to the stored history (see `preview_history`)
and fed through the same `path_kelly` as the backtest. After the close, `settle!` replaces the
preview by the real close for tomorrow.

`actions(sym, asof)` returns today's corporate actions `(dividend, split)` of `sym` (injectable;
default `fetch_actions`). They are fetched once per day and symbol; a failed fetch degrades to "no
action today" with an `@warn` and is retried on the next pass. `base` holds the foreign quantities
of the day's first pass (the basis of progressive liquidation); a restart resets it for that day.
"""
mutable struct LiveState
    bars::Bars          # history up to the last settled close
    symbols::Vector{String}
    actions::Function
    day::Date
    acts::Dict{String,NamedTuple{(:dividend, :split),Tuple{Float64,Float64}}}
    base::Dict{String,Float64}
end

LiveState(bars::Bars, symbols; actions = fetch_actions) =
    LiveState(bars, symbols, actions, Date(0), Dict{String,NamedTuple{(:dividend, :split),Tuple{Float64,Float64}}}(),
              Dict{String,Float64}())

"""
Today's corporate actions of `sym` from Yahoo's chart endpoint (`events=div,split`, the source of
`download_bars`, so the preview and tomorrow's settled history agree): the cash dividend per share and
the share multiplier `numerator/denominator` of the splits whose ex-date is `Date(asof)`; `(0, 1)`
when there is none. Dividends are in post-split dollars, like today's quotes.
"""
function fetch_actions(sym, asof = ny_now())
    url = "https://query1.finance.yahoo.com/v8/finance/chart/$(yahoo_symbol(sym))?range=5d&interval=1d&events=div%2Csplit"
    parse_actions(JSON3.read(HTTP.get(url, ["User-Agent" => "Mozilla/5.0"]; retry = false, read_idle_timeout = 10).body).chart.result[1], asof)
end

"The `(dividend, split)` of Yahoo chart result `res` whose ex-date (exchange-local) is `Date(asof)`."
function parse_actions(res, asof)
    ev = get(res, :events, nothing)
    ex_today(e) = Date(unix2datetime(e.date + res.meta.gmtoffset)) == Date(asof)
    evs(k) = ev === nothing || !haskey(ev, k) ? [] : filter(ex_today, collect(values(ev[k])))
    (dividend = sum((Float64(e.amount) for e in evs(:dividends)); init = 0.0),
     split = prod((Float64(e.numerator) / Float64(e.denominator) for e in evs(:splits)); init = 1.0))
end

"""
Total-return preview of today's adjusted price: `(last + dividend) · split · adj/close` of the last
stored bar. `last` and `dividend` are today's post-split quote units, `split` the share multiplier of a
split effective today (stored history is pre-split, so quotes are scaled back by it). On an ex-dividend
day the price gaps down by the dividend while total return does not, so the dividend is added back.
"""
function preview_history(st::LiveState, last::AbstractVector;
                         dividend = zeros(length(last)), split = ones(length(last)))
    b = st.bars
    ratio = b.adj[end, :] ./ b.close[end, :]
    vcat(b.adj, ((last .+ dividend) .* split .* ratio)')
end

"Today's corporate actions of `x`: cached per day on success; a failed fetch is `(0, 1)` with an `@warn`, not cached."
function actions_today!(st::LiveState, x, asof)
    haskey(st.acts, x) && return st.acts[x]
    try
        a = st.actions(x, asof)
        st.acts[x] = (dividend = Float64(a.dividend), split = Float64(a.split))
    catch e
        @warn "corporate actions unavailable, assuming none today" symbol = x exception = e
        (dividend = 0.0, split = 1.0)
    end
end

"""
One decision+execution pass; returns (weights, orders) or nothing when the market is closed.
Our open orders are cancelled and confirmed first (`cancel_open!`), then the account is snapshotted.
Quotes → total-return preview close; tradability from the live market (halted/unquoted assets keep their
weight, only the others are re-optimised). The weights are over the KTrader budget
`equity − foreign positions still held` (`liquidation`, at most `liq_rate` of each foreign position per
pass), so the held weights are measured against that budget as well.
"""
function live_step!(st::LiveState, br::Broker; S = 1000, rng = Random.default_rng(), liq_rate = 0.25, asof = ny_now())
    market_open(br) || return nothing
    if st.day != Date(asof)
        empty!(st.acts); empty!(st.base); st.day = Date(asof)
    end
    cancel_open!(br, st.symbols)
    s = snapshot(br, st.symbols)
    plan = liquidation(s, st.symbols; base = st.base, rate = liq_rate)
    last = [s.quotes[x].last for x in st.symbols]
    free = [tradable(s.quotes[x]) for x in st.symbols]
    acts = [f ? actions_today!(st, x, asof) : (dividend = 0.0, split = 1.0) for (f, x) in zip(free, st.symbols)]
    H = preview_history(st, [f ? l : st.bars.close[end, j] for (j, (f, l)) in enumerate(zip(free, last))];
                        dividend = [a.dividend for a in acts], split = [a.split for a in acts])
    w = path_kelly(H; S, rng, tradable = free, held = held_weights(s, st.symbols; budget = plan.budget))
    (weights = w, orders = rebalance!(br, st.symbols, w; s, plan))
end

"After the close: refresh stored history with the actual closes."
settle!(st::LiveState, dir) = (st.bars = download_bars(st.symbols); save_bars(st.bars, dir); st)

"True when today's final bars should be stored: a weekday, after 17:00 New York, not yet stored."
settle_due(st::LiveState, asof) = dayofweek(asof) <= 5 && hour(asof) >= 17 && st.bars.dates[end] < Date(asof)
