using Test, JSON3, Random, Dates, Statistics
using KTrader

# ---- mock Tradier --------------------------------------------------------------------------
J(x) = JSON3.read(JSON3.write(x))

"""
Mock broker. `quotes`: symbol => (bid, ask, last); `pos`: symbol => qty; `orders(n)`: the order list
returned by the n-th GET /orders (n counts from 1). Returns (broker, log, pauses); `log` holds
(method, path, form) of every request.
"""
function mock(; type = "margin", equity = 10000.0, total_cash = 1000.0, avail = total_cash,
              pos = Dict{String,Float64}(), quotes, orders = n -> [], live = true, retries = 4)
    log = Tuple[]; polls = Ref(0); pauses = Ref(0)
    function req(m, p, f)
        push!(log, (m, p, f))
        m != "GET" && return nothing
        p == "/v1/markets/clock" && return J((clock = (state = "open",),))
        endswith(p, "/balances") && return J((balances = (total_equity = equity, total_cash = total_cash,
            account_type = type, cash = (cash_available = avail,)),))
        endswith(p, "/positions") && return J((positions = isempty(pos) ? "null" :
            (position = [(symbol = s, quantity = q) for (s, q) in pos],),))
        if endswith(p, "/orders")
            o = orders(polls[] += 1)
            return J((orders = isempty(o) ? "null" : (order = o,),))
        end
        if p == "/v1/markets/quotes"
            want = split(Dict(f)["symbols"], ",")
            return J((quotes = (var"quote" = [(symbol = s, bid = quotes[s][1], ask = quotes[s][2], last = quotes[s][3])
                                         for s in want if haskey(quotes, s)],),))
        end
        error("unexpected request $m $p")
    end
    KTrader.Broker("", "X", "T", live, req; pause = _ -> pauses[] += 1, retries), log, pauses
end

writes(log) = filter(c -> c[1] != "GET", log)
posts(log) = [Dict(c[3]) for c in log if c[1] == "POST"]
notional(ps, side) = sum((parse(Int, p["quantity"]) * parse(Float64, p["price"]) for p in ps if p["side"] == side); init = 0.0)

const Q = Dict("A" => (99.9, 100.1, 100.0), "B" => (9.9, 10.1, 10.0), "F" => (19.9, 20.1, 20.0))

# ---- total-return preview ------------------------------------------------------------------
@testset "Preview is total-return: ex-dividend and split days" begin
    T = 30; P = fill(100.0, T, 2)
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:T-1)), ["A", "B"], P, P .* 0.5)    # adj/close = 0.5
    st = LiveState(b, b.symbols)
    # market +1%, A goes ex a 1.5 dividend: the quote gaps DOWN to 99.5 while total return is +1%
    H = preview_history(st, [99.5, 101.0]; dividend = [1.5, 0.0])
    @test H[end, 1] / H[end-1, 1] ≈ 1.01 && H[end, 2] / H[end-1, 2] ≈ 1.01
    @test preview_history(st, [99.5, 101.0])[end, 1] / 50 ≈ 0.995       # the gap a dividend-blind preview would feed the model
    # 2:1 split today, market +1%: the quote halves (50.5), history stays pre-split
    H = preview_history(st, [50.5, 101.0]; split = [2.0, 1.0])
    @test H[end, 1] / H[end-1, 1] ≈ 1.01 && size(H, 1) == T + 1
    # dividend and split together (quote units are post-split)
    H = preview_history(st, [49.75, 101.0]; dividend = [0.75, 0.0], split = [2.0, 1.0])
    @test H[end, 1] / H[end-1, 1] ≈ 1.01
end

@testset "Corporate actions: Yahoo events, cache, degradation" begin
    ex = Dates.datetime2unix(DateTime(2026, 10, 6, 13, 30))
    res = J((meta = (gmtoffset = -14400,), events = (
        dividends = Dict("1" => (amount = 0.5, date = ex), "2" => (amount = 9.0, date = ex - 86400 * 30)),
        splits = (var"x" = (date = ex, numerator = 2, denominator = 1, splitRatio = "2:1"),))))
    a = KTrader.parse_actions(res, DateTime(2026, 10, 6, 11))
    @test a.dividend == 0.5 && a.split == 2.0                       # only today's ex-date counts
    @test KTrader.parse_actions(J((meta = (gmtoffset = -14400,),)), DateTime(2026, 10, 6)) == (dividend = 0.0, split = 1.0)
    @test KTrader.parse_actions(res, DateTime(2026, 10, 7)) == (dividend = 0.0, split = 1.0)

    P = exp.(cumsum(0.01 .* randn(MersenneTwister(1), 40, 2), dims = 1))
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:39)), ["A", "B"], P, P)
    calls = String[]
    st = LiveState(b, b.symbols; actions = (x, asof) -> (push!(calls, x); x == "B" ? error("yahoo down") : (dividend = 0.5, split = 1)))
    asof = DateTime(2026, 10, 6, 11)
    @test KTrader.actions_today!(st, "A", asof) == (dividend = 0.5, split = 1.0)
    @test KTrader.actions_today!(st, "A", asof) == (dividend = 0.5, split = 1.0)
    @test calls == ["A"]                                            # cached per day
    @test_logs (:warn, r"corporate actions unavailable") @test KTrader.actions_today!(st, "B", asof) == (dividend = 0.0, split = 1.0)
    @test_logs (:warn, r"corporate actions unavailable") KTrader.actions_today!(st, "B", asof)
    @test calls == ["A", "B", "B"]                                  # failure is not cached: retried next pass
end

# ---- progressive liquidation ---------------------------------------------------------------
@testset "Progressive liquidation of foreign positions" begin
    syms = ["A", "B"]
    br, log, _ = mock(pos = Dict("F" => 10.0), quotes = Q, total_cash = 9800.0)
    s = KTrader.snapshot(br, syms)
    base = Dict{String,Float64}()
    plan = KTrader.liquidation(s, syms; base, rate = 0.25)
    @test [(a.sym, a.side, a.qty, a.price) for a in plan.sells] == [("F", "sell", 3, 20.1)]   # ceil(0.25 × 10)
    @test plan.budget ≈ 10000 - 7 * 20                                                     # equity − foreign still held
    @test base == Dict("F" => 10.0)

    acts = rebalance!(br, syms, [1.0, 0.0]; liq_rate = 0.25)
    @test acts[1].sym == "F" && acts[1].side == "sell" && acts[1].qty == 3                 # foreign sells first
    @test [a.qty for a in acts if a.sym == "A"] == [min(floor(Int, (10000 - 140) / 100), floor(Int, 0.98 * 9800 / 99.9))]  # KTrader budget (not equity), cash-capped
    @test notional(posts(log), "buy") <= (10000 - 140) + 1e-9                             # exposure ≤ 100% of equity

    # later passes sell a fraction of the day's FIRST-pass quantity, never more than held
    base = Dict("F" => 10.0)
    for (held, n) in ((7.0, 3), (2.0, 2), (1.0, 1))
        s = KTrader.snapshot(first(mock(pos = Dict("F" => held), quotes = Q)), syms)
        @test only(KTrader.liquidation(s, syms; base, rate = 0.25).sells).qty == n
    end
    s = KTrader.snapshot(first(mock(pos = Dict("F" => 1.0), quotes = Q)), syms)
    @test only(KTrader.liquidation(s, syms; rate = 0.25).sells).qty == 1                   # ceil to whole shares
    @test length(KTrader.liquidation(s, syms; rate = 1.0).sells) == 1
    @test_throws ArgumentError KTrader.liquidation(s, syms; rate = 0.0)

    # no foreign position: identical to before (budget = equity, no extra orders)
    br, log, _ = mock(quotes = Q, total_cash = 9800.0)
    s = KTrader.snapshot(br, syms)
    @test KTrader.liquidation(s, syms) == (sells = NamedTuple[], budget = 10000.0)
    @test [a.qty for a in rebalance!(br, syms, [1.0, 0.0]) if a.sym == "A"] == [floor(Int, 0.98 * 9800 / 99.9)]
end

@testset "Foreign untradable / unquoted symbols are skipped but reserved" begin
    syms = ["A", "B"]
    q = merge(Q, Dict("G" => (0.0, 0.0, 5.0)))                       # halted: last only
    br, log, _ = mock(pos = Dict("F" => 10.0, "G" => 100.0), quotes = q, total_cash = 9800.0)
    acts = rebalance!(br, syms, [1.0, 0.0])
    @test [(a.sym, a.qty) for a in acts if a.side == "sell"] == [("F", 3)]                 # G untouched, no error
    @test all(c -> !occursin("G", string(c[3])), writes(log))
    s = KTrader.snapshot(br, syms)
    @test KTrader.liquidation(s, syms).budget ≈ 10000 - 7 * 20 - 100 * 5                   # G keeps counting
    @test [a.qty for a in acts if a.sym == "A"] == [floor(Int, (10000 - 140 - 500) / 100)]

    # no quote at all: valued as equity − cash − priced positions, still reserved
    br, log, _ = mock(pos = Dict("H" => 10.0), quotes = Q, total_cash = 3000.0)
    s = KTrader.snapshot(br, syms)
    @test s.unpriced ≈ 7000
    @test KTrader.liquidation(s, syms) == (sells = NamedTuple[], budget = 3000.0)
    @test isempty(writes(log))
    @test all(a -> a.sym != "H", rebalance!(br, syms, [1.0, 0.0]))
end

# ---- account semantics ---------------------------------------------------------------------
@testset "Balance fields by account type" begin
    syms = ["A", "B"]
    for (type, cap) in (("margin", 1000.0), ("pdt", 1000.0), ("cash", 300.0))
        br, log, _ = mock(type = type, total_cash = 1000.0, avail = 300.0, quotes = Q)
        @test KTrader.snapshot(br, syms).cash == cap
        acts = rebalance!(br, syms, [1.0, 0.0])
        @test 0 < notional(posts(log), "buy") <= 0.98 * cap + 1e-9
    end
    br, log, _ = mock(type = "cash", total_cash = 1000.0, avail = 300.0, quotes = Q)
    @test only(rebalance!(br, syms, [1.0, 0.0])).qty == floor(Int, 0.98 * 300 / 99.9)       # settled cash only
    @test_throws ErrorException KTrader.usable_cash(J((account_type = "weird", total_cash = 1)))
end

@testset "Resting buy orders reduce the buy budget" begin
    syms = ["A", "B"]
    resting = n -> [(id = 9, symbol = "Z", side = "buy", status = "open", quantity = 10, price = 50.0, class = "equity", exec_quantity = 0)]
    br, log, _ = mock(quotes = merge(Q, Dict("Z" => (49.9, 50.1, 50.0))), orders = resting)
    @test KTrader.snapshot(br, syms).cash == 1000 - 500                                    # 10 × 50 reserved
    acts = rebalance!(br, syms, [1.0, 0.0])
    @test notional(posts(log), "buy") <= 0.98 * 500 + 1e-9
    @test !any(c -> c[1] == "DELETE", log)                                                # Z is not ours: left alone
    # partially filled: only the remainder is reserved
    part = n -> [(id = 9, symbol = "Z", side = "buy", status = "partially_filled", quantity = 10, price = 50.0, remaining_quantity = 4)]
    br, _, _ = mock(quotes = merge(Q, Dict("Z" => (49.9, 50.1, 50.0))), orders = part)
    @test KTrader.snapshot(br, syms).cash == 1000 - 200
    # resting SELL orders reserve nothing (and their proceeds are not cash)
    sell = n -> [(id = 9, symbol = "Z", side = "sell", status = "open", quantity = 10, price = 50.0)]
    br, _, _ = mock(quotes = merge(Q, Dict("Z" => (49.9, 50.1, 50.0))), orders = sell)
    @test KTrader.snapshot(br, syms).cash == 1000
end

@testset "Cancel is confirmed by polling before the snapshot" begin
    syms = ["A", "B"]
    order(status) = [(id = 7, symbol = "A", side = "buy", status = status, quantity = 5, price = 99.0)]
    # becomes terminal after 2 polls
    br, log, pauses = mock(quotes = Q, orders = n -> order(n <= 2 ? (n == 1 ? "open" : "pending_cancel") : "canceled"))
    acts = rebalance!(br, syms, [1.0, 0.0])
    @test count(c -> c[1] == "DELETE", log) == 1 && pauses[] == 2
    iorders = findall(c -> c[1] == "GET" && endswith(c[2], "/orders"), log)
    @test findfirst(c -> endswith(c[2], "/balances"), log) > iorders[3]                    # snapshot after confirmation
    @test findfirst(c -> c[1] == "POST", log) > findfirst(c -> endswith(c[2], "/balances"), log)
    @test !isempty(acts)
    # partially filled counts as live and is cancelled too
    br, log, _ = mock(quotes = Q, orders = n -> order(n == 1 ? "partially_filled" : "canceled"))
    rebalance!(br, syms, [1.0, 0.0])
    @test count(c -> c[1] == "DELETE", log) == 1
    # never terminal: error, no POST, no snapshot
    for st in ("pending_cancel", "open", "pending", "partially_filled")
        br, log, pauses = mock(quotes = Q, orders = n -> order(st), retries = 3)
        @test_throws ErrorException rebalance!(br, syms, [1.0, 0.0])
        @test isempty(filter(c -> c[1] == "POST", log)) && pauses[] == 3
        @test !any(c -> endswith(c[2], "/balances"), log)
    end
    # terminal orders are neither cancelled nor waited for
    br, log, pauses = mock(quotes = Q, orders = n -> order("filled"))
    rebalance!(br, syms, [1.0, 0.0])
    @test !any(c -> c[1] == "DELETE", log) && pauses[] == 0
    # foreign holdings' orders are ours as well
    br, log, _ = mock(quotes = Q, pos = Dict("F" => 4.0),
                      orders = n -> [(id = 3, symbol = "F", side = "sell", status = n == 1 ? "open" : "canceled", quantity = 2, price = 25.0)])
    rebalance!(br, syms, [1.0, 0.0])
    @test ("DELETE", "/v1/accounts/X/orders/3", nothing) in log
end

@testset "Buys never exceed cash on hand, even with pending sells" begin
    syms = ["A", "B"]
    # A (50 sh = 5000) must be sold for B; only 500 cash is in the account
    br, log, _ = mock(pos = Dict("A" => 50.0), total_cash = 500.0, quotes = Q)
    acts = rebalance!(br, syms, [0.0, 1.0])
    ps = posts(log)
    @test any(p -> p["side"] == "sell" && p["symbol"] == "A" && p["quantity"] == "50", ps)
    @test 0 < notional(ps, "buy") <= 0.98 * 500 + 1e-9                                     # sale proceeds not counted
    @test findfirst(p -> p["side"] == "sell", ps) < findfirst(p -> p["side"] == "buy", ps)  # sells first
    # no cash at all: no buys
    br, log, _ = mock(pos = Dict("A" => 50.0), total_cash = 0.0, quotes = Q)
    rebalance!(br, syms, [0.0, 1.0])
    @test notional(posts(log), "buy") == 0
    # negative cash (borrowed) is never spent
    br, log, _ = mock(pos = Dict("A" => 50.0), total_cash = -300.0, quotes = Q)
    rebalance!(br, syms, [0.0, 1.0])
    @test notional(posts(log), "buy") == 0
end

@testset "Non-live issues no writes" begin
    syms = ["A", "B"]
    open_order = n -> [(id = 7, symbol = "A", side = "buy", status = "open", quantity = 5, price = 99.0)]
    br = KTrader.tradier(; live = false, base = "http://127.0.0.1:9", account = "X", token = "T")
    @test_logs (:info, r"DRY POST") @test br.request("POST", "/x", ["a" => "b"]) === nothing
    @test_logs (:info, r"DRY DELETE") @test br.request("DELETE", "/x", nothing) === nothing
    @test_logs (:info, r"DRY PUT") @test br.request("PUT", "/x", ["a" => "b"]) === nothing
    # a non-live pass does not wait for cancel confirmation (the dry DELETE changes nothing)
    mk, log, pauses = mock(quotes = Q, live = false, orders = open_order)
    acts = rebalance!(mk, syms, [1.0, 0.0])
    @test pauses[] == 0 && !isempty(acts)
end

# ---- end to end ----------------------------------------------------------------------------
@testset "live_step!: foreign liquidation, actions, budget" begin
    rng = MersenneTwister(5); T = 1300
    P = exp.(cumsum(0.01 .* randn(rng, T, 3), dims = 1))
    b = KTrader.Bars(collect(Date(2026, 10, 5) .- Day.(T-1:-1:0)), ["A", "B", "C"], P, P)
    last = P[end, :]
    quotes = Dict("A" => (last[1] * 0.999, last[1] * 1.001, last[1]), "B" => (last[2] * 0.999, last[2] * 1.001, last[2]),
                  "C" => (last[3] * 0.999, last[3] * 1.001, last[3]), "F" => (19.9, 20.1, 20.0))
    br, log, _ = mock(pos = Dict("F" => 100.0), total_cash = 8000.0, quotes = quotes)          # equity 10000, F worth 2000
    calls = String[]
    st = LiveState(b, b.symbols; actions = (x, asof) -> (push!(calls, x); (dividend = 0.0, split = 1.0)))
    asof = DateTime(2026, 10, 6, 11)
    r = live_step!(st, br; S = 60, rng = MersenneTwister(1), asof)
    @test isapprox(sum(r.weights), 1; atol = 1e-8) && all(r.weights .>= 0)
    @test any(a -> a.sym == "F" && a.side == "sell" && a.qty == 25, r.orders)            # ceil(0.25 × 100)
    @test sort(calls) == ["A", "B", "C"]                                                  # foreign symbols need no actions
    budget = 10000 - 75 * 20
    @test notional(posts(log), "buy") <= min(0.98 * 8000, budget) + 1e-6
    # second pass the same day: first-pass base quantity (100) is kept, actions are cached
    br2, log2, _ = mock(pos = Dict("F" => 75.0), total_cash = 8500.0, quotes = quotes)
    r2 = live_step!(st, br2; S = 60, rng = MersenneTwister(1), asof)
    @test any(a -> a.sym == "F" && a.side == "sell" && a.qty == 25, r2.orders) && sort(calls) == ["A", "B", "C"]
    # a new day resets base and actions
    live_step!(st, br2; S = 60, rng = MersenneTwister(1), asof = DateTime(2026, 10, 7, 11), liq_rate = 0.5)
    @test sort(calls) == ["A", "A", "B", "B", "C", "C"] && st.base == Dict("F" => 75.0)
    @test any(c -> c[1] == "POST" && Dict(c[3])["symbol"] == "F" && Dict(c[3])["quantity"] == "38", log2)   # ceil(0.5 × 75)
end

@testset "Default fetch_actions has the signature live_step! calls (no mock)" begin
    st = LiveState(KTrader.Bars([Date(2026, 1, 1)], ["A"], ones(1, 1), ones(1, 1)), ["A"])
    @test hasmethod(st.actions, Tuple{String, Any})
end
