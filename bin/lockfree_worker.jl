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
include(joinpath(@__DIR__,"../src/gate0/KTraderGate0.jl"))
Base.include(KTraderGate0,joinpath(@__DIR__,"../src/gate0/market_io.jl"))
const G=KTraderGate0

mf=G.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el=G.Eligibility(mf)
a=findfirst(==(Date(get(ENV,"GATE0_START_DATE","2024-10-01"))),mf.dates)
b=findfirst(==(Date(get(ENV,"GATE0_END_DATE","2026-10-02"))),mf.dates)
a!==nothing && b!==nothing || error("worker calendar endpoints missing")
G._assert_lockfree_independent_window(mf,el,a,b,zeros(length(mf.symbols)))
options=(1e-6,2048,5.0)
origins=G._lockfree_required_origins(mf,el,a,b)
prepsig=G._backtest_signature(mf,el,
             (:lockfree_forecast_preparation_v2,a,b,options))
prep_path=abspath(get(ENV,"GATE0_FORECAST_PREP_PATH",
             "/tmp/ktrader-two-year-20261011/lockfree-full-forecast-cache.jls"))
isfile(prep_path) || error("complete causal forecast preparation is missing")
prepared=deserialize(prep_path)
prepared.version==2 && prepared.signature==prepsig &&
prepared.origins==origins && length(prepared.forecasts)==length(origins) &&
all(o->haskey(prepared.forecasts,o),origins) ||
    error("forecast preparation incomplete, stale, or signed for other data/code")

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
sig=G._backtest_signature(mf,el,(:lockfree_independent_decisions_v2,
          a,b,options,seed,minM,maxM,blas,numshards,
          1e-4,1e-6,1e-6,true,true))
days=collect((a+shard-1):numshards:(b-1))
out_dir=abspath(get(ENV,"GATE0_SHARD_DIR",
             "/tmp/ktrader-two-year-20261011/lockfree-shards"))
path=joinpath(out_dir,"shard_$(shard)_of_$(numshards).jls")
results=Dict{Int,G.Gate0DayDecision}()
if isfile(path)
    saved=deserialize(path)
    saved.version==2 && saved.signature==sig && saved.days==days ||
        error("worker checkpoint is stale, out of order, or signed for other settings")
    all(t->t in days,keys(saved.decisions)) ||
        error("worker checkpoint contains a date outside this shard")
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
    c=dec.certificates
    c.converged && c.A<=1e-4 && abs(c.B)<=1e-6 &&
        c.B_audit<=1e-6 && G.cash_kelly_certified(c.C;tol=1e-6) ||
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
