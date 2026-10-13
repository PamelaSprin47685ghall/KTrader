# Bounded microbenchmark: a handful of real prequential forecasts, never a
# trading/backtest run. Compare independent historical posterior forecasts
# under sequential and threaded scheduling, with identical causal inputs.
using Dates, LinearAlgebra
BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","1")))
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const ForecastProbe = KTraderGate0

mf = ForecastProbe.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el = ForecastProbe.Eligibility(mf)
start_d = Date(get(ENV,"PROBE_DATE","2024-10-02"))
start = findfirst(==(start_d), mf.dates)
start === nothing && error("invalid probe start date")
K = parse(Int,get(ENV,"PROBE_K","4"))
1 <= K <= 8 || error("probe is strictly bounded to at most 8 forecasts")
ids = start:(start+K-1)
println("BOUNDED_FORECAST_PROBE k=$K nthreads=$(Threads.nthreads()) from=$start_d")
flush(stdout)

# Compile all model code before measuring either scheduler.
warm = ForecastProbe._historical_forecast(mf,el,start)
warm === nothing && error("unexpected absence of forecast")
single = Vector{Any}(undef,K)
parallel = Vector{Any}(undef,K)
st=time_ns()
for (k,t) in enumerate(ids)
    single[k] = ForecastProbe._historical_forecast(mf,el,t)
end
ts=(time_ns()-st)/1e9
# Sequential reference allocates many dead expanded-history design matrices;
# collect before measuring independent parallel tasks. This does not modify
# any forecast and prevents GC timing from contaminating the memory probe.
GC.gc(true)
st=time_ns()
Threads.@threads for k in 1:K
    parallel[k] = ForecastProbe._historical_forecast(mf,el,ids[k])
end
tp=(time_ns()-st)/1e9
for k in 1:K
    @assert single[k].active==parallel[k].active
    @assert isapprox(single[k].mean_log_return,parallel[k].mean_log_return;
                     atol=1e-11,rtol=1e-10)
end
println("SCHEDULER_OK serial_s=$ts parallel_s=$tp speedup=$(ts/tp)")
