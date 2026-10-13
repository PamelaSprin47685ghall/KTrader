#!/usr/bin/env julia
# One checkpointable worker of a strictly proved state-INDEPENDENT Gate-0
# backtest. Every daily decision is the original full adaptive model on the
# original complete observed history; no account PnL or scenario approximation
# is calculated here. No worker runs until the 896-origins source-bound causal
# forecast file has completed and its signature verifies.
using Dates,Serialization,LinearAlgebra,Printf

blas=parse(Int,get(ENV,"GATE0_BLAS_THREADS","4"))
1<=blas<=16 || error("unsupported BLAS thread count")
BLAS.set_num_threads(blas)
# Gate-0 code lives in the KTraderNew libs (precompiled as packages); gate0/KTraderGate0.jl re-exports them.
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const G=KTraderGate0

mf=G.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el=G.Eligibility(mf)
a=findfirst(==(Date(get(ENV,"GATE0_START_DATE","2024-10-01"))),mf.dates)
b=findfirst(==(Date(get(ENV,"GATE0_END_DATE","2026-10-02"))),mf.dates)
a!==nothing && b!==nothing || error("worker calendar endpoints missing")
G._assert_lockfree_independent_window(mf,el,a,b,zeros(length(mf.symbols)))
options=(1e-6,2048,5.0)
origins=G._lockfree_required_origins(mf,el,a,b)

prep_path=abspath(get(ENV,"GATE0_FORECAST_PREP_PATH",
             "/workspace/ktrader-run/lockfree-full-forecast-cache.jls"))
prep_shards=parse(Int,get(ENV,"GATE0_PREP_SHARD_COUNT","1"))
1<=prep_shards<=16 || error("invalid preparation shard count")
prepared_forecasts=Dict{Int,Any}()
for i in 1:prep_shards
    path_i=prep_shards==1 ? prep_path : prep_path*".shard$(i)of$(prep_shards)"
    isfile(path_i) || error("causal forecast preparation shard is missing: $path_i")
    part=deserialize(path_i)
    hasproperty(part,:blas) && hasproperty(part,:assigned) ||
        error("forecast preparation lacks BLAS/shard provenance: $path_i")
    part.blas==blas ||
        error("worker BLAS=$blas differs from preparation BLAS=$(part.blas); refusing mixed floating-point paths")
    # The preparation signature binds BLAS configuration and shard identity.
    sig_i=G._backtest_signature(mf,el,
             (:lockfree_forecast_preparation_v3,a,b,options,
              read(joinpath(@__DIR__,"..","dev","lockfree_forecast_prepare.jl")),
              part.blas,prep_shards,i))
    part.version==3 && part.signature==sig_i && part.origins==origins &&
    part.shard==i && part.shard_count==prep_shards &&
    part.assigned==origins[i:prep_shards:end] &&
    length(part.forecasts)==length(part.assigned) &&
    all(o->haskey(part.forecasts,o),part.assigned) ||
        error("forecast preparation shard $i incomplete, stale, or signed for other data/code")
    for (o,f) in part.forecasts
        haskey(prepared_forecasts,o) && error("origin $o prepared by two shards")
        prepared_forecasts[o]=f
    end
end
length(prepared_forecasts)==length(origins) && all(o->haskey(prepared_forecasts,o),origins) ||
    error("forecast preparation shards do not cover every required origin")
prepared=(;forecasts=prepared_forecasts)

numshards=parse(Int,get(ENV,"GATE0_SHARD_COUNT","4"))
shard=parse(Int,get(ENV,"GATE0_SHARD_ID","1"))
1<=numshards<=8 && 1<=shard<=numshards || error("invalid worker shard specification")
maxnew=parse(Int,get(ENV,"GATE0_WORKER_MAX_NEW_DAYS","1"))
1<=maxnew<=8 || error("bounded worker must process 1..8 new dates per invocation")
minM=parse(Int,get(ENV,"GATE0_MIN_SCENARIOS","64"))
maxM=parse(Int,get(ENV,"GATE0_MAX_SCENARIOS","65536"))
minM>=64 && ispow2(minM) && minM*4<=maxM && ispow2(maxM) ||
    error("two-stage confirmation requires a power-of-two scenario interval")
seed=G._DRIVER_DEFAULT_SEED
# The worker program itself is part of the decision semantics (cache reset,
# acceptance gates); bind its exact bytes so an edited worker never resumes
# or reduces an old shard. The reducer recomputes the same signature.
sig=G._backtest_signature(mf,el,(:lockfree_independent_decisions_v3,
          a,b,options,seed,minM,maxM,blas,numshards,
          1e-4,1e-6,1e-6,true,true,
          read(joinpath(@__DIR__,"lockfree_worker.jl"))))
days=G._lockfree_shard_days(a,b,shard,numshards)
out_dir=abspath(get(ENV,"GATE0_SHARD_DIR",
             "/workspace/ktrader-run/lockfree-shards"))
path=joinpath(out_dir,"shard_$(shard)_of_$(numshards).jls")
mkpath(out_dir)
# Exclusive shard ownership: two processes must never write one checkpoint.
# A lock left by a killed process is stale only if its pid is gone.
lockdir=path*".lock"
function acquire_shard_lock(dir)
    for _ in 1:2
        try
            mkdir(dir)
            write(joinpath(dir,"pid"),string(getpid()))
            return
        catch err
            err isa Base.IOError || err isa SystemError || rethrow()
            pidfile=joinpath(dir,"pid")
            owner=isfile(pidfile) ? tryparse(Int,strip(read(pidfile,String))) : nothing
            if owner!==nothing && owner!=getpid() && isdir("/proc/$owner")
                error("shard $shard is locked by live process $owner; refusing concurrent writer")
            end
            owner===nothing && isdir(dir) && (time()-mtime(dir))<30 &&
                error("shard $shard lock is being created by another process")
            rm(dir;recursive=true,force=true)
        end
    end
    error("could not acquire shard lock $dir")
end
acquire_shard_lock(lockdir)
atexit(()->rm(lockdir;recursive=true,force=true))
certified(c)=c.converged && c.A<=1e-4 && abs(c.B)<=1e-6 &&
    c.B_audit<=1e-6 && G.cash_kelly_certified(c.C;tol=1e-6)
results=Dict{Int,G.Gate0DayDecision}()
if isfile(path)
    saved=deserialize(path)
    saved.version==2 && saved.signature==sig && saved.days==days ||
        error("worker checkpoint is stale, out of order, or signed for other settings")
    all(t->t in days,keys(saved.decisions)) ||
        error("worker checkpoint contains a date outside this shard")
    all(t->saved.decisions[t].t==t && saved.decisions[t].mode==:adaptive &&
           certified(saved.decisions[t].certificates),keys(saved.decisions)) ||
        error("worker checkpoint contains a mislabelled or uncertified decision")
    merge!(results,saved.decisions)
end
save()=G._write_backtest_checkpoint(path,
           (;version=2,signature=sig,days,decisions=results))

println("LOCKFREE_WORKER shard=$shard/$numshards assigned=$(length(days)) " *
        "completed=$(length(results)) available_forecasts=$(length(origins)) " *
        "BLAS=$blas minM=$minM maxM=$maxM source_signature_verified=true")
flush(stdout)
cache=G._ForecastCache(mf,el,options)
cache.selective=true
merge!(cache.entries,prepared.forecasts)
precomputed=Set(origins)
newcount=0
started=time_ns()
for t in days
    haskey(results,t) && continue
    # single_day_decision stores TODAY's posterior forecast into the cache.
    # Reset to the signed preparation before every date so that a decision
    # never depends on shard count, processing order or restart history.
    empty!(cache.entries)
    merge!(cache.entries,prepared.forecasts)
    before=Set(keys(cache.entries))
    local t0=time_ns()
    dec=G.single_day_decision(mf,el,zeros(length(mf.symbols));
          t=t,seed=seed,mode=:adaptive,forecast_cache=cache,
          min_scenarios=minM,max_scenarios=maxM,
          posterior_tol=options[1],posterior_max_cells=options[2],
          u_span=options[3],weight_tol=1e-4,utility_tol=1e-6,
          kelly_tol=1e-6,mu_qmc=true,mu_chisq_qmc=true)
    dec.mode==:adaptive ||
        error("lockfree worker refuses $t: risk-domain fallback depends on account holdings")
    dec.t==t || error("worker decision has wrong date index")
    c=dec.certificates
    certified(c) ||
        error("worker rejected a day with red numerical certificates")
    any(o->o!=t && !(o in precomputed),setdiff(Set(keys(cache.entries)),before)) &&
        error("an unexpected historical forecast origin escaped precomputation")
    results[t]=dec
    global newcount+=1
    save()
    @printf("LOCKFREE_WORKER_DAY shard=%d date=%s M=%d seconds=%.4f A=%.7g Bopt=%.7g Baudit=%.7g N=%d cash=%.6g\n",
            shard,mf.dates[t],dec.M,(time_ns()-t0)/1e9,
            c.A,c.B,c.B_audit,length(dec.R_universe),dec.w_cash)
    flush(stdout)
    GC.gc(true)
    if newcount>=maxnew
        break
    end
end
remaining=length(days)-length(results)
@printf("LOCKFREE_WORKER_CHECKPOINT shard=%d added=%d completed=%d/%d remaining=%d elapsed=%.4f\n",shard,newcount,
        length(results),length(days),remaining,(time_ns()-started)/1e9)
flush(stdout)
if remaining>0
    exit(75)
end
println("LOCKFREE_SHARD_COMPLETED shard=$shard/$numshards signature=$sig")
