# Bounded diagnostic of the actual 65-asset recent-date posterior, not a small
# universe substitute. Does not invoke the broker or run a legacy model.
using Dates, LinearAlgebra
BLAS.set_num_threads(1)
module ReadinessEnv
    using Dates
    include(joinpath(@__DIR__, "../src/gate0/market.jl"))
    include(joinpath(@__DIR__, "../src/gate0/market_io.jl"))
    include(joinpath(@__DIR__, "../src/gate0/geometry.jl"))
    include(joinpath(@__DIR__, "../src/gate0/modes.jl"))
    include(joinpath(@__DIR__, "../src/gate0/response.jl"))
    include(joinpath(@__DIR__, "../src/gate0/posterior.jl"))
    include(joinpath(@__DIR__, "../src/gate0/driver.jl"))
end
using .ReadinessEnv

mf = ReadinessEnv.load_market_facts(get(ENV, "GATE0_DATA_DIR", "data"))
el = Eligibility(mf, effective_bars_252(mf))
date = Date(get(ENV, "PROBE_DATE", "2024-10-02"))
t = findfirst(==(date), mf.dates)
t === nothing && error("Probe date not present in market calendar: $date")
println("REAL_PANEL $(size(mf.observed)) forecast_origin=$date row=$t")
flush(stdout)
started = time_ns()
prep = ReadinessEnv._response_prefix(mf, el, t)
X, Y = prep.mp.X[prep.rows, :], prep.mp.Y[prep.rows, :]
println("PREPARED n=$(size(X,1)) P=$(size(X,2)) N=$(size(Y,2)) seconds=$((time_ns()-started)/1e9)")
flush(stdout)
started = time_ns()
post = fit_full_posterior(X, Y) # production prior, full integration, default strict tol
println("POSTERIOR_OK nodes=$(length(post.alpha_weights)) bytes=$(Base.summarysize(post)) seconds=$((time_ns()-started)/1e9)")
println("mean_log_return=", ReadinessEnv._posterior_log_forecast(post, prep, t).mean_log_return)
flush(stdout)
