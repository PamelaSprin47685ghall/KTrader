# Prepare strictly causal, complete response forecast means for the unique
# historical origins consumed by ANY 2024-10-01..2026-10-02 lock-free decision.
# This is NOT a trading run: no Kelly solve, holdings or account equity.
# A source/data/eligibility-bound checkpoint guarantees exact resume.
using Dates, Serialization, LinearAlgebra, Printf
BLAS.set_num_threads(parse(Int,get(ENV,"GATE0_BLAS_THREADS","4")))
# Precompiled carrier of the unchanged src/gate0 code (no per-process JIT).
# Setup once: julia --project=. -e 'push!(LOAD_PATH,"gate0pkg"); using KTraderGate0Precompiled'
push!(LOAD_PATH,joinpath(@__DIR__,"..","gate0pkg"))
using KTraderGate0Precompiled
const KTraderGate0=KTraderGate0Precompiled.KTraderGate0  # checkpoint type paths
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
# Multi-process sharding: shard i of n owns origins[i:n:end] and writes its own
# checkpoint (no shared writer). Every origin is computed by exactly one shard
# with the identical routine; Julia thread/process count does not change
# values (measured bitwise), BLAS thread count does and is signed.
nshard=parse(Int,get(ENV,"GATE0_PREP_SHARD_COUNT","1"))
ishard=parse(Int,get(ENV,"GATE0_PREP_SHARD_ID","1"))
1<=ishard<=nshard<=16 || error("invalid preparation shard $ishard of $nshard")
mine=origins[ishard:nshard:end]
# v3: in-window origins (a <= o <= b-2) use the decision-path forecast that the
# sequential driver itself caches; older origins use _historical_forecast, as
# the sequential _prime_historical_forecasts! does. Script bytes are signed.
sig=G._backtest_signature(mf,el,
             (:lockfree_forecast_preparation_v3,a,b,options,
              read(joinpath(@__DIR__,"lockfree_forecast_prepare.jl")),
              BLAS.get_num_threads(),nshard,ishard))
# BLAS thread count changes forecasts at ~1e-13 (measured): a checkpoint is
# only resumable under the identical BLAS configuration.
base=abspath(get(ENV,"GATE0_FORECAST_PREP_PATH",
                "/workspace/ktrader-run/lockfree-full-forecast-cache.jls"))
dest=nshard==1 ? base : base*".shard$(ishard)of$(nshard)"
workers=parse(Int,get(ENV,"GATE0_FORECAST_WORKERS","1"))
1<=workers<=Threads.nthreads() || error("workers must be <= Julia thread count")
maxnew=parse(Int,get(ENV,"GATE0_PREP_MAX_NEW","32"))
1<=maxnew<=64 || error("bounded forecast preparation only allows 1..64 new origins per call")

cache=G._ForecastCache(mf,el,options)
# Exclusive writer: a second preparation process on the same checkpoint
# refuses to start (a lock whose pid is gone is stale and reclaimed).
lockdir=dest*".lock"
mkpath(dirname(dest))
if !(try mkdir(lockdir); true catch; false end)
    pidf=joinpath(lockdir,"pid")
    owner=isfile(pidf) ? tryparse(Int,strip(read(pidf,String))) : nothing
    (owner!==nothing && isdir("/proc/$owner")) &&
        error("forecast preparation $dest is locked by live process $owner")
    owner===nothing && (time()-mtime(lockdir))<30 &&
        error("forecast preparation lock is being created by another process")
    rm(lockdir;recursive=true,force=true); mkdir(lockdir)
end
write(joinpath(lockdir,"pid"),string(getpid()))
atexit(()->rm(lockdir;recursive=true,force=true))
if isfile(dest)
    saved=deserialize(dest)
    saved.version==3 && saved.signature==sig && saved.origins==origins ||
        error("forecast checkpoint is incompatible with the current source, data or calendar")
    all(o->o in mine,keys(saved.forecasts)) ||
        error("forecast checkpoint includes a nonrequired historical origin")
    merge!(cache.entries,saved.forecasts)
end
save()=G._write_backtest_checkpoint(dest,
                (;version=3,signature=sig,blas=BLAS.get_num_threads(),options,origins,
                 shard=ishard,shard_count=nshard,assigned=mine,forecasts=cache.entries))
println("LOCKFREE_PREP_ONLY calendar=$(mf.dates[a]):$(mf.dates[b]) " *
        "decision_days=$(b-a) unique_required_origins=$(length(origins)) " *
        "cached=$(length(cache.entries)) workers=$workers BLAS=$(BLAS.get_num_threads())")
flush(stdout)
pending=[o for o in mine if !haskey(cache.entries,o)]
newcount=0
started=time_ns()
for first in 1:workers:min(length(pending),maxnew)
    last=min(first+workers-1,min(length(pending),maxnew))
    batch=pending[first:last]
    compute(o)= a<=o<=b-2 ?
        G._decision_forecast(mf,el,o;tol=options[1],max_cells=options[2],
                             u_span=options[3]) :
        G._historical_forecast(mf,el,o;
                   tol=options[1],max_cells=options[2],
                   u_span=options[3],preprocessed=cache)
    forecasts=try
        length(batch)==1 ? [compute(batch[1])] :
            fetch.([Threads.@spawn compute(o) for o in batch])
    catch
        save() # keep every already-certified origin before failing loudly
        rethrow()
    end
    for (o,f) in zip(batch,forecasts)
        f!==nothing && all(isfinite,f.mean_log_return) ||
            (save(); error("required causal forecast failed at $(mf.dates[o])"))
        cache.entries[o]=f
        global newcount+=1
        if newcount%8==0
            save()
            @printf("LOCKFREE_PREP_PROGRESS completed=%d/%d elapsed=%.4f\n",
                    length(cache.entries),length(mine),(time_ns()-started)/1e9)
            flush(stdout)
        end
    end
end
save()
remaining=length(mine)-length(cache.entries)
@printf("LOCKFREE_PREP_SLICE completed=%d/%d added=%d remaining=%d elapsed=%.4f\n",
        length(cache.entries),length(mine),newcount,remaining,
        (time_ns()-started)/1e9)
flush(stdout)
if remaining>0
    println("PREFLIGHT_CHECKPOINTED_AND_PAUSED: never a full backtest completion")
    exit(75)
end
println("LOCKFREE_HISTORICAL_FORECASTS_READY exact=true signature=$sig")
