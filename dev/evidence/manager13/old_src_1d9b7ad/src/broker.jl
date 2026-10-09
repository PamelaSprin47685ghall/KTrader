# Execution layer, derived from rebalance.mjs: weights -> target shares -> limit orders.
# The account is dedicated to KTrader: universe symbols are traded toward the weights; positions in
# symbols OUTSIDE the universe ("foreign") are only ever sold, progressively (`liquidation`).
# Non-live mode cannot issue any write (enforced at the HTTP entry point, not at call sites).

"""
Tradier session. `request(method, path, form)` returns parsed JSON (injectable for tests); for GET,
`form` (or `nothing`) is the query string. `pause(seconds)` and `retries` bound the wait for order
cancellations to be confirmed (see `cancel_open!`).
"""
struct Broker
    base::String
    account::String
    token::String
    live::Bool
    request::Function
    pause::Function
    retries::Int
end

Broker(base, account, token, live, request; pause = sleep, retries = 10) =
    Broker(base, account, token, live, request, pause, retries)

function tradier(; live = false,
                 base = get(ENV, "TRADIER_BASE_URL", "https://api.tradier.com"),
                 account = ENV["TRADIER_ACCOUNT_ID"], token = ENV["TRADIER_TOKEN"], kw...)
    function request(method, path, form)
        if method != "GET" && !live
            @info "DRY $method $path $form"
            return nothing
        end
        h = ["Authorization" => "Bearer $token", "Accept" => "application/json"]
        r = method == "GET" ? HTTP.get(base * path * (form === nothing ? "" : "?" * HTTP.URIs.escapeuri(Dict(form))), h) :
            HTTP.request(method, base * path, [h; "Content-Type" => "application/x-www-form-urlencoded"],
                         HTTP.URIs.escapeuri(Dict(form)))
        isempty(r.body) ? nothing : JSON3.read(r.body)
    end
    Broker(base, account, token, live, request; kw...)
end

aslist(x) = x === nothing ? [] : x isa AbstractVector ? collect(x) : [x]
get_(b::Broker, path, form = nothing) = b.request("GET", path, form)
num(x, k) = (v = get(x, k, nothing); v === nothing ? 0.0 : Float64(v))

"Order statuses that can still change a position or reserve cash."
const NONTERMINAL = ("pending", "pending_cancel", "open", "partially_filled")

market_open(b::Broker) = get_(b, "/v1/markets/clock").clock.state == "open"

"Today's orders of the account (all statuses)."
function orders(b::Broker)
    r = get_(b, "/v1/accounts/$(b.account)/orders", ["limit" => "1000"])
    o = r === nothing ? nothing : r.orders
    aslist(o === nothing || o isa AbstractString ? nothing : get(o, :order, nothing))
end

live_orders(b::Broker) = filter(o -> String(o.status) in NONTERMINAL, orders(b))

"Held quantities by symbol (zero lines included)."
function positions(b::Broker)
    pj = get_(b, "/v1/accounts/$(b.account)/positions").positions
    Dict(String(p.symbol) => Float64(p.quantity)
         for p in aslist(pj isa AbstractString ? nothing : get(pj, :position, nothing)))
end

"Long positions held outside `symbols`, sorted."
foreign(pos, symbols) = sort!([x for (x, q) in pos if q > 0 && !(x in symbols)])

"""
Cancel every non-terminal order on our symbols (the universe and the foreign holdings), then, live
only, poll `/orders` until none of them is non-terminal: a cancel response is not a cancel
confirmation, and a partially filled order must be in the positions before we snapshot. Sleeps
`b.pause(1)` before each of up to `b.retries` polls and throws if orders are still live: never trade
on a stale snapshot. Non-live mode issues no DELETE (the request entry drops it), so cannot confirm.
"""
function cancel_open!(b::Broker, symbols)
    mine = Set(String.(symbols)) ∪ foreign(positions(b), symbols)
    ours() = filter(o -> String(o.symbol) in mine, live_orders(b))
    open = ours()
    isempty(open) && return
    for o in open
        b.request("DELETE", "/v1/accounts/$(b.account)/orders/$(o.id)", nothing)
    end
    b.live || return
    for _ in 1:b.retries
        b.pause(1)
        isempty(ours()) && return
    end
    error("orders still live after cancel: ", join(sort!(unique(String[o.symbol for o in ours()])), ", "))
end

"""
Cash usable for buys by account type (Tradier balances): `cash.cash_available` for cash accounts
(settled cash only), `total_cash` for margin and pdt accounts. Buying power is never used: no
borrowing. A balances record without `account_type` is read as margin.
"""
function usable_cash(bal)
    type = String(get(bal, :account_type, "margin"))
    type == "cash" ? Float64(bal.cash.cash_available) :
    type in ("margin", "pdt") ? Float64(bal.total_cash) : error("unsupported account type: ", type)
end

"Cash held by a resting equity buy order: remaining quantity × limit price (market order: the quoted ask)."
function reserved(o, quotes)
    String(get(o, :side, "")) == "buy" && String(get(o, :class, "equity")) == "equity" || return 0.0
    left = haskey(o, :remaining_quantity) ? Float64(o.remaining_quantity) : num(o, :quantity) - num(o, :exec_quantity)
    px = get(o, :price, nothing)
    px === nothing && (q = get(quotes, String(o.symbol), nothing); px = q === nothing ? 0.0 : max(q.ask, q.last))
    left * Float64(px)
end

"""
Account state `(equity, cash, pos, quotes, unpriced)`. `cash` = usable cash (`usable_cash`) minus the
cash reserved by resting buy orders. `quotes` covers the universe, every held symbol and every resting
order's symbol. Positions without a usable last price cannot be valued from quotes: their combined value
is `unpriced = equity − total_cash − priced positions` (0 when every position is priced), so halted or
unquotable holdings still count against the capital.
"""
function snapshot(b::Broker, symbols)
    bal = get_(b, "/v1/accounts/$(b.account)/balances").balances
    pos = positions(b)
    resting = live_orders(b)
    want = unique(String[symbols; [x for (x, q) in pos if q != 0]; String[o.symbol for o in resting]])
    q = get_(b, "/v1/markets/quotes", ["symbols" => join(want, ",")]).quotes
    quotes = Dict(String(x.symbol) => (bid = num(x, :bid), ask = num(x, :ask), last = num(x, :last))
                  for x in aslist(q isa AbstractString ? nothing : get(q, :quote, nothing)))
    equity = Float64(bal.total_equity)
    priced(x) = haskey(quotes, x) && quotes[x].last > 0
    held = [x for (x, qty) in pos if qty != 0]
    value = sum((pos[x] * quotes[x].last for x in held if priced(x)); init = 0.0)
    (; equity, cash = usable_cash(bal) - sum(o -> reserved(o, quotes), resting; init = 0.0), pos, quotes,
       unpriced = all(priced, held) ? 0.0 : max(0.0, equity - Float64(bal.total_cash) - value))
end

"Tradable now = a live two-sided market (positive bid, ask and last). Halted/unquoted → false."
tradable(q) = q.bid > 0 && q.ask > 0 && q.last > 0

"Portfolio weights over `symbols` as a fraction of `budget` (the capital KTrader manages; default: equity; none when it is 0)."
held_weights(s, symbols; budget = s.equity) =
    [budget > 0 ? get(s.pos, x, 0.0) * s.quotes[x].last / budget : 0.0 for x in symbols]

"Integer share targets from weights (Σw=1 → notional ≤ budget ≤ equity: leverage ≤ 1, floor to whole shares)."
target_shares(w, budget, last) = [floor(Int, max(wi, 0) * budget / p) for (wi, p) in zip(w, last)]

"""
Progressive liquidation of foreign positions (outside `symbols`; the account is dedicated to KTrader).
Each pass sells at most `rate` ∈ (0, 1] of the quantity held at the day's first pass (`base[sym]`,
recorded here on first sight), rounded up to whole shares and never more than held, as sell limit
orders at the (cent-rounded) ask. A foreign symbol that is not tradable now (or has no quote) is skipped without
error and still counts as held. Returns `(sells, budget)` with
`budget = equity − market value of the foreign positions still held after these sells`: the capital
KTrader may allocate this pass, so total exposure never exceeds 100% of equity. No foreign position
→ no sells and `budget = equity`.
"""
function liquidation(s, symbols; base = Dict{String,Float64}(), rate = 0.25)
    0 < rate <= 1 || throw(ArgumentError("liquidation rate must be in (0, 1], got $rate"))
    sells = NamedTuple[]
    held = s.unpriced
    for x in foreign(s.pos, symbols)
        q = s.pos[x]
        qt = get(s.quotes, x, nothing)
        (qt === nothing || qt.last <= 0) && continue          # valued in s.unpriced
        n = tradable(qt) ? min(ceil(Int, round(rate * get!(base, x, q); digits = 9)), floor(Int, q)) : 0
        n > 0 && push!(sells, (; sym = x, side = "sell", qty = n, price = round(qt.ask; digits = 2)))
        held += (q - n) * qt.last
    end
    (; sells, budget = max(0.0, s.equity - held))
end

"""
One rebalance pass toward weights `w` (aligned with `symbols`, Σw=1 over the KTrader budget). Without
`s` it cancels our open orders (confirmed, see `cancel_open!`) and takes the snapshot itself; a given
`s` must already be post-cancel. `plan` is the `liquidation` of `s` (default: computed here with
`liq_rate`). Order of business: foreign sells, universe sells, then buys limited to the cash ALREADY
in the account (unfilled sale proceeds are not counted and nothing is borrowed: leverage ≤ 1); the
next pass picks up freed cash. Idempotent and convergent.
"""
function rebalance!(b::Broker, symbols, w; s = nothing, plan = nothing, liq_rate = 0.25)
    if s === nothing
        cancel_open!(b, symbols)
        s = snapshot(b, symbols)
    end
    plan === nothing && (plan = liquidation(s, symbols; rate = liq_rate))
    last = [s.quotes[x].last for x in symbols]
    tgt = target_shares(w, plan.budget, last)
    delta = [t - get(s.pos, x, 0.0) for (t, x) in zip(tgt, symbols)]
    cash = max(0.0, s.cash) * 0.98
    acts = NamedTuple[plan.sells...]
    for i in sortperm(delta)                       # most negative (largest sells) first
        d = round(Int, delta[i]); d == 0 && continue
        q = s.quotes[symbols[i]]
        tradable(q) || continue                        # cannot operate: keep the position
        if d < 0
            push!(acts, (; sym = symbols[i], side = "sell", qty = -d, price = round(q.ask; digits = 2)))
        else
            bid = round(q.bid; digits = 2)             # the price actually sent is what costs cash
            n = bid > 0 ? min(d, floor(Int, cash / bid)) : 0
            n > 0 && (cash -= n * bid; push!(acts, (; sym = symbols[i], side = "buy", qty = n, price = bid)))
        end
    end
    for a in acts
        b.request("POST", "/v1/accounts/$(b.account)/orders", ["class" => "equity", "symbol" => a.sym, "side" => a.side,
            "quantity" => string(a.qty), "type" => "limit", "duration" => "day",
            "price" => string(a.price)])
    end
    acts
end
