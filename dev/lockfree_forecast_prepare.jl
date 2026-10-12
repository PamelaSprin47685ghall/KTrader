# Prepare strictly causal, complete response forecast means for the unique
# historical origins consumed by ANY 2024-10-01..2026-10-02 lock-free decision.
# This is NOT a trading run: no Kelly solve, holdings or account equity.
# A source/data/eligibility-bound checkpoint guarantees exact resume.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"GATE0_BLAS_THREADS","4")))
include(joinpath(@__DIR__,"../src/gate0/KTraderGate0.jl"))
Base.include(KTraderGate0,joinpath(@__DIR__,"../src/gate0/market_io.jl"))
const G=KTraderGate0

mf=G.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el=G.Eligibility(mf)
start=Date(get(ENV,"GATE0_START_DATE","2024-10-01"))
finish=Date(get(ENV,"GATE0_END_DATE","2026-10-02"))
a=findfirst(==(start),mf.dates)
b=findfirst(==(finish),mf.dates)
(a!==nothing && b!==nothing) || error("date interval missing from market session calendar")
G._assert_lockfree_independent_window(mf,el,a,b,zeros(length(mf.symbols)))
origins=G._lockfree_required_origins(mf,el,a,b)
options=(1e-6,2048,5.0)
sig=G._backtest_signature(mf,el,
             (:lockfree_forecast_preparation_v2,a,b,options))
dest=abspath(get(ENV,"GATE0_FORECAST_PREP_PATH",
                "/tmp/ktrader-two-year-20261011/lockfree-full-forecast-cache.jls"))
workers=parse(Int,get(ENV,"GATE0_FORECAST_WORKERS","1"))
1<=workers<=Threads.nthreads() || error("workers must be <= Julia thread count")
maxnew=parse(Int,get(ENV,"GATE0_PREP_MAX_NEW","32"))
1<=maxnew<=64 || error("bounded forecast preparation only allows 1..64 new origins per call")

cache=G._ForecastCache(mf,el,options)
if isfile(dest)
    saved=deserialize(dest)
    saved.version==2 && saved.signature==sig && saved.origins==origins ||
        error("forecast checkpoint is incompatible with the current source, data or calendar")
    all(o->o in origins,keys(saved.forecasts)) ||
        error("forecast checkpoint includes a nonrequired historical origin")
    merge!(cache.entries,saved.forecasts)
end
save()=G._write_backtest_checkpoint(dest,
                (;version=2,signature=sig,options,origins,forecasts=cache.entries))
println("LOCKFREE_PREP_ONLY calendar=$(mf.dates[a]):$(mf.dates[b]) " *
        "decision_days=$(b-a) unique_required_origins=$(length(origins)) " *
        "cached=$(length(cache.entries)) workers=$workers BLAS=$(BLAS.get_num_threads())")
flush(stdout)
pending=[o for o in origins if !haskey(cache.entries,o)]
newcount=0
started=time_ns()
for first in 1:workers:min(length(pending),maxnew)
    last=min(first+workers-1,min(length(pending),maxnew))
    batch=pending[first:last]
    compute(o)=G._historical_forecast(mf,el,o;
                   tol=options[1],max_cells=options[2],
                   u_span=options[3],preprocessed=cache)
    forecasts=length(batch)==1 ? [compute(batch[1])] :
              fetch.([Threads.@spawn compute(o) for o in batch])
    for (o,f) in zip(batch,forecasts)
        f!==nothing && all(isfinite,f.mean_log_return) ||
            error("required causal forecast failed at $(mf.dates[o])")
        cache.entries[o]=f
        global newcount+=1
        if newcount%8==0
            save()
            @printf("LOCKFREE_PREP_PROGRESS completed=%d/%d elapsed=%.4f\n",
                    length(cache.entries),length(origins),(time_ns()-started)/1e9)
            flush(stdout)
        end
    end
end
save()
remaining=length(origins)-length(cache.entries)
@printf("LOCKFREE_PREP_SLICE completed=%d/%d added=%d remaining=%d elapsed=%.4f\n",
        length(cache.entries),length(origins),newcount,remaining,
        (time_ns()-started)/1e9)
flush(stdout)
if remaining>0
    println("PREFLIGHT_CHECKPOINTED_AND_PAUSED: never a full backtest completion")
    exit(75)
end
println("LOCKFREE_HISTORICAL_FORECASTS_READY exact=true signature=$sig")
