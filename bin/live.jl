# Usage: TRADIER_ACCOUNT_ID=.. TRADIER_TOKEN=.. julia --project=. bin/live.jl [--live]
# Env: REBALANCE_INTERVAL_S (60), LIQUIDATE_RATE (0.25: share of each non-universe position sold per pass,
# of its quantity at the day's first pass; the account is dedicated to KTrader), TRADIER_BASE_URL.
# Intraday: every INTERVAL seconds, last price = preview close -> path_kelly -> rebalance!.
# After the close: store the real closes (settle!) ready for tomorrow.
using KTrader, Dates
root = joinpath(@__DIR__, "..")
dir = joinpath(root, "data")
syms = load_universe(joinpath(root, "universe.txt"))
br = tradier(; live = "--live" in ARGS)
st = LiveState(load_bars(dir), syms)
interval = parse(Int, get(ENV, "REBALANCE_INTERVAL_S", "60"))
liq_rate = parse(Float64, get(ENV, "LIQUIDATE_RATE", "0.25"))
settled = Date(0)
while true
    try
        r = live_step!(st, br; liq_rate)
        if r === nothing
            asof = KTrader.ny_now()
            if Date(asof) != settled && settle_due(st, asof)
                settle!(st, dir); settled = Date(asof)
                @info "settled" last = st.bars.dates[end]
            end
            sleep(300)
        else
            @info "pass" weights = round.(r.weights, digits = 3) orders = length(r.orders)
            sleep(interval)
        end
    catch e
        @error "pass failed" exception = (e, catch_backtrace())
        sleep(interval)
    end
end
