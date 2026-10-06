#!/usr/bin/env julia
# Same full-history decisions under configurable date-task / BLAS topology.
using KTrader, Dates, Statistics, Random, LinearAlgebra, Printf
b=load_bars("data")
T,N=size(b.adj)
days=parse(Int,get(ENV,"BENCH_DAYS","500"))
date_tasks=parse(Int,get(ENV,"DATE_TASKS",string(Threads.nthreads())))
blas_threads=parse(Int,get(ENV,"BLAS_THREADS","1"))
S=parse(Int,get(ENV,"SCENARIOS","300"))
from=b.dates[max(1,T-days)]
# Compilation warmup is reported separately and does not truncate measured histories.
warmup=@elapsed path_kelly_v1(exp.(cumsum(0.01randn(MersenneTwister(8),600,4);dims=1));S=32)
@printf("Panel T=%d N=%d; warmup=%.3fs; date_tasks=%d BLAS=%d\n",T,N,warmup,date_tasks,blas_threads)
started=time()
result=backtest_v1(b;from,S,date_tasks,blas_threads)
elapsed=time()-started
@printf("%d decisions in %.3fs = %.2f days/s\n",length(result.ret),elapsed,length(result.ret)/elapsed)
for rows in (1:length(result.ret),max(1,length(result.ret)-499):length(result.ret))
    println("CPU-seconds for decisions $(first(rows)):$(last(rows)) (parallel times do not sum to wall time)")
    for (i,bucket) in enumerate(result.timing_buckets)
        @printf("  %-10s %.3f\n",bucket,sum(view(result.timings,rows,i)))
    end
end
@printf("CAGR %.3f, final wealth %.3f\n",summarize(result.ret).cagr,result.wealth[end])
