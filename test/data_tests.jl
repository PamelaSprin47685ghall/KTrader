using Test, Dates, JSON3
using KTrader

# Canned Yahoo chart payload. `nothing` -> JSON null. `ts` are UTC epoch seconds.
function yahoo(ts, close, adj; off = -18000, events = nothing)
    res = Dict("meta" => Dict("gmtoffset" => off),
               "timestamp" => ts,
               "indicators" => Dict("quote" => [Dict("close" => close)],
                                    "adjclose" => [Dict("adjclose" => adj)]))
    events === nothing || (res["events"] = events)
    JSON3.read(JSON3.write(Dict("chart" => Dict("result" => [res], "error" => nothing))))
end
epoch(dt) = Int(datetime2unix(dt))
nyopen(d) = epoch(DateTime(d) + Hour(14) + Minute(30))      # 09:30 EST = 14:30 UTC (Date -> DateTime)
sessions(from, n) = [d for d in from:Day(1):from+Day(3n) if dayofweek(d) <= 5][1:n]

asof = DateTime(2026, 12, 31, 18)                          # after 17:00: every listed day is final
fetchfrom(canned) = (sym; asof) -> KTrader.fetch_symbol(sym; asof, getjson = url -> canned[sym])

@testset "is_final" begin
    d = Date(2026, 10, 6)
    @test is_final(d - Day(1), DateTime(2026, 10, 6, 0))
    @test !is_final(d, DateTime(2026, 10, 6, 16, 59, 59))
    @test is_final(d, DateTime(2026, 10, 6, 17))
    @test is_final(d, DateTime(2026, 10, 6, 23, 59))
    @test !is_final(d + Day(1), DateTime(2026, 10, 6, 18))   # later date never final
    @test !is_final(d + Day(1), DateTime(2026, 10, 6, 23, 59))
    @test !is_final(d + Day(30), DateTime(2026, 10, 6, 17))
end

@testset "yahoo_symbol" begin
    @test KTrader.yahoo_symbol("BRK.B") == "BRK-B"
    @test KTrader.yahoo_symbol("SPY") == "SPY"
    seen = String[]
    url = Ref("")
    KTrader.fetch_symbol(KTrader.yahoo_symbol("BRK.B"); asof,
                         getjson = u -> (url[] = u; yahoo([nyopen(Date(2026, 10, 5))], [1.0], [1.0])))
    @test occursin("/chart/BRK-B?", url[])
    rec = (s; asof) -> (push!(seen, s); Dict(Date(2026, 10, 5) => (1.0, 1.0)))
    b = download_bars(["BRK.B", "SPY"]; fetch = rec, asof)
    @test seen == ["BRK-B", "SPY"]
    @test b.symbols == ["BRK.B", "SPY"]                      # Bars keeps the universe names
end

@testset "fetch_symbol: gmtoffset, nulls, finality" begin
    # DST boundary: 2026-03-08 02:00 EST -> EDT. Fri 03-06 opens 14:30 UTC, Mon 03-09 opens 13:30 UTC.
    fri, mon = Date(2026, 3, 6), Date(2026, 3, 9)
    ts = [epoch(DateTime(2026, 3, 6, 14, 30)), epoch(DateTime(2026, 3, 9, 13, 30))]
    for off in (-18000, -14400)                              # Yahoo reports one offset for the whole history
        p = KTrader.fetch_symbol("SPY"; asof, getjson = u -> yahoo(ts, [10.0, 11.0], [9.0, 10.5]; off))
        @test sort(collect(keys(p))) == [fri, mon]
        @test p[fri] == (10.0, 9.0) && p[mon] == (11.0, 10.5)
    end
    # Exchange east of UTC: Sydney opens 10:00 AEDT = 23:00 UTC the *previous* UTC day.
    syd = epoch(DateTime(2026, 2, 1, 23))                    # = Mon 2026-02-02 10:00 local
    p = KTrader.fetch_symbol("BHP.AX"; asof, getjson = u -> yahoo([syd], [1.0], [1.0]; off = 39600))
    @test collect(keys(p)) == [Date(2026, 2, 2)]
    # null close / null adjclose rows are dropped, not carried
    d = sessions(Date(2026, 10, 5), 4)
    p = KTrader.fetch_symbol("X"; asof, getjson = u ->
        yahoo(nyopen.(d), [1.0, nothing, 3.0, nothing], [1.0, 2.0, nothing, nothing]))
    @test collect(keys(p)) == [d[1]]
    # today's row is dropped before 17:00 NY and kept after
    ts = nyopen.(d[1:2])
    early = DateTime(d[2]) + Hour(15)
    late = DateTime(d[2]) + Hour(17)
    @test collect(keys(KTrader.fetch_symbol("X"; asof = early, getjson = u -> yahoo(ts, [1.0, 2.0], [1.0, 2.0])))) == [d[1]]
    @test sort(collect(keys(KTrader.fetch_symbol("X"; asof = late, getjson = u -> yahoo(ts, [1.0, 2.0], [1.0, 2.0]))))) == d[1:2]
    # HTTP / parse errors are not swallowed
    @test_throws ErrorException KTrader.fetch_symbol("X"; asof, getjson = u -> error("HTTP 429"))
end

@testset "adj vs close: dividend and split" begin
    d = sessions(Date(2026, 10, 5), 6)
    # Yahoo `close` is split-adjusted only; `adjclose` also removes dividends.
    # 2:1 split before d[3] (history already halved: no jump in either series);
    # a 1.00 dividend goes ex on d[5]: adj history before it is scaled by (1 - 1/close_prev).
    close = [50.0, 50.5, 51.0, 51.5, 51.0, 51.8]
    f = 1 - 1.0 / close[4]
    adj = [50.0, 50.5, 51.0, 51.5] .* f
    adj = [adj; 51.0; 51.8]
    ev = Dict("splits" => Dict("t" => Dict("date" => nyopen(d[3]), "numerator" => 2, "denominator" => 1)),
              "dividends" => Dict("t" => Dict("date" => nyopen(d[5]), "amount" => 1.0)))
    p = KTrader.fetch_symbol("X"; asof, getjson = u -> yahoo(nyopen.(d), close, adj; events = ev))
    @test [p[x][1] for x in d] == close
    @test [p[x][2] for x in d] == adj
    b = Bars(d, ["X"], reshape([p[x][1] for x in d], :, 1), reshape([p[x][2] for x in d], :, 1))
    r(M) = diff(log.(M[:, 1]))
    rc, ra = r(b.close), r(b.adj)
    @test rc ≈ log.(close[2:end] ./ close[1:end-1]) rtol = 1e-12
    @test rc[2] ≈ log(51.0 / 50.5) rtol = 1e-12               # across the split: history already split-adjusted, no jump
    @test ra[[1, 2, 3, 5]] ≈ rc[[1, 2, 3, 5]] rtol = 1e-10    # adj tracks close away from the dividend
    @test ra[4] ≈ rc[4] - log(f) rtol = 1e-12                 # ex-dividend: close drops, adj return adds the dividend back
    @test ra[4] > rc[4]
end

@testset "Bars constructor" begin
    nan = NaN
    dates = [Date(2026, 10, 5) + Day(i) for i in 0:4]
    rc = [nan 1.0 nan; nan 2.0 10.0; 3.0 nan 11.0; 4.0 4.0 nan; 5.0 5.0 13.0]
    ra = [nan 1.5 nan; nan 2.5 20.0; 3.5 4.5 21.0; 4.5 4.5 nan; 5.5 5.5 23.0]
    b = Bars(dates, ["A", "B", "C"], rc, ra)
    # bar false exactly where either raw price is non-finite (B[3]: close NaN, adj finite)
    @test b.bar == (isfinite.(rc) .& isfinite.(ra))
    @test b.bar isa BitMatrix
    @test b.bar == Bool[0 1 0; 0 1 1; 1 0 1; 1 1 0; 1 1 1]
    # NaN before first bar, carry after
    @test isequal(b.close[:, 1], [nan, nan, 3.0, 4.0, 5.0])
    @test isequal(b.adj[:, 1], [nan, nan, 3.5, 4.5, 5.5])
    @test isequal(b.close[:, 2], [1.0, 2.0, 2.0, 4.0, 5.0])      # halted day carries 2.0, not NaN
    @test isequal(b.adj[:, 2], [1.5, 2.5, 2.5, 4.5, 5.5])
    @test isequal(b.close[:, 3], [nan, 10.0, 11.0, 11.0, 13.0])
    @test isequal(b.adj[:, 3], [nan, 20.0, 21.0, 21.0, 23.0])
    @test b.dates == dates && b.symbols == ["A", "B", "C"]
    # a symbol that never prints stays NaN and never tradable
    z = Bars(dates, ["Z"], fill(nan, 5, 1), fill(nan, 5, 1))
    @test !any(z.bar) && all(isnan, z.close) && all(isnan, z.adj)
end

@testset "download_bars: coverage filter" begin
    d = sessions(Date(2026, 10, 5), 8)          # d[1..8]
    syms = ["A", "B", "C", "D", "E"]
    # E lists on d[3]. Unpublished-day shape: Yahoo's trailing row has only some symbols printed.
    function series(s)
        keep = Dict(
            "A" => collect(1:8), "B" => [1, 2, 3, 4, 5, 6, 8], "C" => [1, 2, 3, 4, 6, 8],
            "D" => [1, 2, 3, 5, 6, 7, 8],                   # D halted on d[4]
            "E" => [3, 4, 5, 6, 8])[s]
        keep
    end
    # d[7]: printed by A, D only; listed symbols on d[7] = 5 -> 2*2 < 5 -> dropped
    # d[6]: A,B,C,D,E print -> kept ; d[4]: A,B,C,E print, D halted (4/5) -> kept with bar=false for D
    price(s, i) = 100.0 * (findfirst(==(s), syms)) + i
    canned = Dict(s => yahoo(nyopen.(d[series(s)]),
                             [price(s, i) for i in series(s)], [price(s, i) + 0.5 for i in series(s)])
                  for s in syms)
    b = download_bars(syms; fetch = fetchfrom(canned), asof)
    @test b.dates == d[[1, 2, 3, 4, 5, 6, 8]]                  # d[7] dropped (2 of 5 listed)
    @test b.symbols == syms
    j(s) = findfirst(==(s), syms)
    t4 = findfirst(==(d[4]), b.dates)
    @test !b.bar[t4, j("D")] && all(b.bar[t4, [j("A"), j("B"), j("C"), j("E")]])   # real session, one halt: kept
    @test b.close[t4, j("D")] == price("D", 3) && b.adj[t4, j("D")] == price("D", 3) + 0.5   # marked at last print
    @test isnan(b.close[1, j("E")]) && isnan(b.close[2, j("E")]) && !any(b.bar[1:2, j("E")])  # NaN before listing
    @test b.bar[3, j("E")]
    @test b.close[findfirst(==(d[8]), b.dates), j("A")] == price("A", 8)

    # exactly half of the listed symbols counts (2*2 >= 4); one fewer does not
    d4 = sessions(Date(2026, 11, 2), 3)
    c4(sel) = Dict(s => yahoo(nyopen.(d4[sel[s]]), fill(1.0, length(sel[s])), fill(1.0, length(sel[s])))
                   for s in ["A", "B", "C", "D"])
    half = Dict("A" => [1, 2, 3], "B" => [1, 2, 3], "C" => [1, 2], "D" => [1, 2])
    b = download_bars(["A", "B", "C", "D"]; fetch = fetchfrom(c4(half)), asof)
    @test b.dates == d4
    one = Dict("A" => [1, 2, 3], "B" => [1, 2], "C" => [1, 2], "D" => [1, 2])
    b = download_bars(["A", "B", "C", "D"]; fetch = fetchfrom(c4(one)), asof)
    @test b.dates == d4[1:2]
    @test all(b.bar)

    # Yahoo's pre-publication shape: after the close, trailing close:null for every symbol.
    dn = sessions(Date(2026, 10, 5), 3)
    cn = Dict(s => yahoo(nyopen.(dn), [1.0, 2.0, nothing], [1.0, 2.0, nothing]) for s in ["A", "B"])
    b = download_bars(["A", "B"]; fetch = fetchfrom(cn), asof)
    @test b.dates == dn[1:2] && all(b.bar) && all(isfinite, b.close)
    # ... and only 1 of 4 symbols has printed the last day: dropped.
    cn = Dict(s => yahoo(nyopen.(dn), [1.0, 2.0, s == "A" ? 3.0 : nothing], [1.0, 2.0, s == "A" ? 3.0 : nothing])
              for s in ["A", "B", "C", "D"])
    b = download_bars(["A", "B", "C", "D"]; fetch = fetchfrom(cn), asof)
    @test b.dates == dn[1:2]
end

@testset "save_bars / load_bars round trip" begin
    nan = NaN
    dates = [Date(2026, 10, 5) + Day(i) for i in 0:4]
    rc = [nan 1.0 nan; nan 2.0 10.1; 3.3 nan 11.0; 4.0 4.0 nan; 5.0 (0.1 + 0.2) 13.0]
    ra = [nan 1.5 nan; nan 2.5 20.0; 3.5 4.5 21.0; 4.5 4.5 nan; 5.5 5.5 23.0]
    b = Bars(dates, ["AAA", "BRK.B", "C"], rc, ra)
    mktempdir() do dir
        out = joinpath(dir, "nested", "bars")
        save_bars(b, out)
        c = load_bars(out)
        @test c.dates == b.dates && c.symbols == b.symbols
        @test isequal(c.close, b.close) && isequal(c.adj, b.adj)
        @test c.bar == b.bar && c.bar isa BitMatrix
        @test typeof(c.close) == Matrix{Float64}
        # no-bar cells are written as NaN, not as the carried marking price
        raw = readlines(joinpath(out, "close.csv"))
        @test split(raw[4], ',')[3] == "NaN"                     # BRK.B halted on 3rd day
        # one-symbol universe also survives
        b1 = Bars(dates, ["A"], rc[:, 1:1], ra[:, 1:1])
        save_bars(b1, joinpath(dir, "one"))
        c1 = load_bars(joinpath(dir, "one"))
        @test isequal(c1.close, b1.close) && c1.bar == b1.bar && c1.symbols == ["A"]
    end
end

@testset "load_universe" begin
    mktemp() do path, io
        write(io, "# comment\nspy\n\n  brk.b  \n#x\nqqq\n"); close(io)
        @test load_universe(path) == ["SPY", "BRK.B", "QQQ"]
    end
end
