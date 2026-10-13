# One-date, original-mathematics CPU cost probe. Never enters the multi-day
# accounting loop and never builds the entire historical forecast cache.
# Use distinct, representative PROBE_DATE values in separate bounded runs.
using Dates, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","1")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const CostProbe = KTraderGate0

const mf = CostProbe.load_market_facts(get(ENV, "GATE0_DATA_DIR", "data"))
const el = CostProbe.Eligibility(mf)
start_d = Date(get(ENV, "WINDOW_START", "2024-10-01"))
end_d = Date(get(ENV, "WINDOW_END", "2026-10-02"))
first_day = findfirst(==(start_d), mf.dates)
final_day = findfirst(==(end_d), mf.dates)
first_day === nothing && error("Missing WINDOW_START session")
final_day === nothing && error("Missing WINDOW_END session")

# Exact number of forecast origins required for at least one realised return
# before the last decision. A forecast only exists after WARMUP and an asset's
# first observed adjacent-price pair.
required = Int[]
for u in (CostProbe.WARMUP+1):(final_day-1)
    any(view(el.model_admitted, u-1, :)) || continue
    any(mf.observed[u-1, j] && mf.observed[u, j]
        for j in eachindex(mf.symbols)) || continue
    push!(required, u-1)
end
println("WORKLOAD sessions=$(final_day-first_day) forecast_origins=$(length(required)) " *
        "assets=$(length(mf.symbols)) total_sessions=$(length(mf.dates))")
flush(stdout)

# At most eight actual complete causal mean forecasts, one after another.
# The user has explicitly forbidden launching any long backtest before a
# reliable runtime admission estimate, so this is diagnostic only.
series = strip(get(ENV,"PROBE_SERIES_DATES",""))
if !isempty(series)
    dates_to_probe = Date.(split(series,','))
    length(dates_to_probe) <= 8 || error("series probe cap is eight origins")
    repeats = parse(Int,get(ENV,"PROBE_SERIES_REPEATS","1"))
    1 <= repeats <= 2 || error("at most two series passes in a bounded probe")
    static_inputs = get(ENV,"PROBE_SERIES_STATIC","false")=="true" ?
        CostProbe._ForecastCache(mf,el,(1e-6,2048,5.0)) : nothing
    for lap in 1:repeats, d in dates_to_probe
        idx = findfirst(==(d),mf.dates)
        idx === nothing && error("missing session $d")
        idx in required || error("origin $d does not require forecast")
        local t0 = time_ns()
        p = CostProbe._response_prefix(mf,el,idx;
                                      compact=true,prepared=static_inputs)
        tprep=(time_ns()-t0)/1e9
        t0=time_ns()
        mu=CostProbe.fit_full_posterior(p.Xtr,p.Ytr; mean_at=p.xnow)
        tpost=(time_ns()-t0)/1e9
        @assert length(mu)==length(p.act) && all(isfinite,mu)
        @printf("SERIES_OK lap=%d date=%s N=%d n=%d prep=%.4f posterior=%.4f total=%.4f\n",
            lap,d,length(p.act),size(p.Xtr,1),tprep,tpost,tprep+tpost)
        flush(stdout)
    end
    exit(0)
end

date = Date(get(ENV, "PROBE_DATE", "2024-10-02"))
origin = findfirst(==(date), mf.dates)
origin === nothing && error("Probe date not present: $date")
origin in required || error("Probe origin does not require a forecast")

if get(ENV,"PROBE_STATIC_PREP","false")=="true"
    cache=CostProbe._ForecastCache(mf,el,(1e-6,2048,5.0))
    for k in 1:3
        local t0=time_ns()
        a=CostProbe._response_prefix(mf,el,origin;compact=true)
        ta=(time_ns()-t0)/1e9
        t0=time_ns()
        b=CostProbe._response_prefix(mf,el,origin;compact=true,prepared=cache)
        tb=(time_ns()-t0)/1e9
        @assert a.act==b.act && isapprox(a.s1,b.s1;atol=1e-12)
        @assert isapprox(a.Xtr,b.Xtr;atol=1e-11)
        @assert isapprox(a.Ytr,b.Ytr;atol=1e-11)
        @printf("STATIC_PREP_OK repeat=%d old=%.5f cached=%.5f\n",k,ta,tb)
        flush(stdout)
    end
    exit(0)
end

started = time_ns()
prep = CostProbe._response_prefix(mf, el, origin)
X = prep.mp.X[prep.rows, :]
Y = prep.mp.Y[prep.rows, :]
prep_seconds = (time_ns()-started)/1e9
println("PREP origin=$(date) n=$(size(X,1)) P=$(size(X,2)) N=$(size(Y,2)) " *
        "seconds=$(round(prep_seconds,digits=4))")
flush(stdout)

if get(ENV, "PREP_ONLY", "false") == "true"
    for k in 1:3
        local started = time_ns()
        p2 = CostProbe._response_prefix(mf, el, origin)
        @assert size(p2.mp.X) == size(prep.mp.X)
        @printf("PREP_STEADY run=%d seconds=%.5f\n", k, (time_ns()-started)/1e9)
    end
    exit(0)  # Bounded microbenchmark, never a decision or account backtest
end

started = time_ns()
mu = CostProbe.fit_full_posterior(X, Y; mean_at=vec(prep.mp.X_full[origin,:]))
post_seconds = (time_ns()-started)/1e9
@assert length(mu) == size(Y,2) && all(isfinite,mu)
@printf("MEAN_ONLY_OK origin=%s prep_sec=%.4f posterior_sec=%.4f total_sec=%.4f\n",
        date, prep_seconds, post_seconds, prep_seconds+post_seconds)
flush(stdout)
if get(ENV, "POST_REPEAT", "false") == "true"
    for k in 1:2
        local started = time_ns()
        mu2 = CostProbe.fit_full_posterior(X,Y;
                      mean_at=vec(prep.mp.X_full[origin,:]))
        @assert isapprox(mu2,mu; atol=1e-11,rtol=1e-10)
        @printf("POST_STEADY run=%d seconds=%.5f\n",k,(time_ns()-started)/1e9)
        flush(stdout)
    end
end
