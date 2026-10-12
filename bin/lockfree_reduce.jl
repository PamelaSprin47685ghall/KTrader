#!/usr/bin/env julia
# Strict map/reduce account closing step for the *real*, source-bound
# independently computed Gate-0 adaptive decisions. Full holdings drift and
# cash/EW benchmarks use the original sequential accounting equations.
# Refuse partial, mismatched, duplicate or uncertified worker results.
using Dates,Serialization,LinearAlgebra,Statistics,Printf

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
a!==nothing && b!==nothing || error("map/reduce endpoints absent from market calendar")
G._assert_lockfree_independent_window(mf,el,a,b,zeros(length(mf.symbols)))
options=(1e-6,2048,5.0)
numshards=parse(Int,get(ENV,"GATE0_SHARD_COUNT","4"))
1<=numshards<=8 || error("map/reduce shard count must be 1..8")
minM=parse(Int,get(ENV,"GATE0_MIN_SCENARIOS","64"))
maxM=parse(Int,get(ENV,"GATE0_MAX_SCENARIOS","65536"))
sig=G._backtest_signature(mf,el,(:lockfree_independent_decisions_v2,
          a,b,options,G._DRIVER_DEFAULT_SEED,minM,maxM,blas,numshards,
          1e-4,1e-6,1e-6,true,true))
dir=abspath(get(ENV,"GATE0_SHARD_DIR",
             "/tmp/ktrader-two-year-20261011/lockfree-shards"))
collected=Dict{Int,G.Gate0DayDecision}()
for shard in 1:numshards
    path=joinpath(dir,"shard_$(shard)_of_$(numshards).jls")
    isfile(path) || error("worker shard $shard is missing: $path")
    saved=deserialize(path)
    expected=collect((a+shard-1):numshards:(b-1))
    saved.version==2 && saved.signature==sig && saved.days==expected &&
        length(saved.decisions)==length(expected) &&
        all(t->haskey(saved.decisions,t),expected) ||
        error("worker shard $shard is partial, not signed or has wrong date order")
    for t in expected
        haskey(collected,t) && error("duplicate independent decision t=$t")
        collected[t]=saved.decisions[t]
    end
end
length(collected)==b-a && all(t->haskey(collected,t),a:b-1) ||
    error("missing one or more independent decisions")
decisions=[collected[t] for t in a:(b-1)]
res=G.replay_independent_gate0(mf,el,decisions;t_start=a,t_end=b)
result_path=abspath(get(ENV,"GATE0_LOCKFREE_RESULT_PATH",
           "/tmp/ktrader-two-year-20261011/lockfree-two-year-result.jls"))
G._write_backtest_checkpoint(result_path,(;version=2,signature=sig,
                       data_dates=(mf.dates[a],mf.dates[b]),result=res))
@printf("LOCKFREE_TWO_YEAR_COMPLETED date=%s..%s sessions=%d assets=%d equity=%.8f median_scenarios=%.0f max_scenarios=%d result=%s\n",
        mf.dates[a],mf.dates[b],b-a,length(mf.symbols),
        res.equity[end],median([d.M for d in decisions]),
        maximum(d.M for d in decisions),result_path)
println("ALL_DATES_SIGNED_AND_CERTIFIED original_account_marking=true " *
        "independent_work_allowed_only_by_502_session_lockfree_proof=true")
flush(stdout)
