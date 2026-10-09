# Usage: julia -t 6 --project=. bin/backtest.jl [YYYY-MM-DD]
using KTrader, Dates, Statistics, LinearAlgebra, Printf

b = load_bars(joinpath(@__DIR__, "..", "data"))
from = isempty(ARGS) ? today() - Year(10) : Date(ARGS[1])
S = parse(Int, get(ENV, "SCENARIOS", "300"))
date_tasks = parse(Int,get(ENV,"DATE_TASKS",string(Threads.nthreads())))
blas_threads = parse(Int,get(ENV,"BLAS_THREADS","1"))
adaptive = get(ENV,"ADAPTIVE_SCENARIOS","false") == "true"
engine = Symbol(get(ENV,"ENGINE","batch"))
chunk_size = parse(Int,get(ENV,"CHUNK_SIZE",string(min(24,4date_tasks))))
seed = parse(Int,get(ENV,"SEED","1"))
F_folds = parse(Int,get(ENV,"F_FOLDS","3"))
alpha = get(ENV,"RIDGE_ALPHA","auto")
ridge_alpha = alpha == "auto" ? nothing : parse(Float64,alpha)
quadrature_tol = parse(Float64,get(ENV,"QUADRATURE_TOL","1e-5"))
max_scenarios = parse(Int,get(ENV,"MAX_SCENARIOS","512"))
options = (; S,seed,ridge_alpha,F_folds,date_tasks,blas_threads,chunk_size,adaptive,quadrature_tol,max_scenarios,engine)
println("Julia=$(VERSION) threads=$(Threads.nthreads()) panel=$(size(b.adj)) input=$(first(b.dates)):$(last(b.dates)); from=$from options=$options")
trace = KTrader.BacktestExecutionTiming()

println("Running Path Kelly V1 Backtest from $from...")
measured = @timed backtest_v1(b; from,options...,execution_timing=trace)
bt = measured.value
elapsed = measured.time
@printf("Runtime %.6f s; %.3f days/s; allocated=%.3fMiB GC=%.6fs\n",elapsed,length(bt.ret)/elapsed,measured.bytes/2.0^20,measured.gctime)
@printf("Inference/consumer wall span %.6fs; windows=%d blocks=%d coordinator_advances=%d block_advances=%d\n",
    trace.wall_seconds,trace.windows,trace.blocks,trace.coordinator_advances,trace.block_advances)
for (i,bucket) in enumerate(KTrader.BACKTEST_EXECUTION_BUCKETS)
    @printf("  %-22s %.6fs\n",bucket,trace.seconds[i])
end
println("Worker timings are sums of task elapsed time, not process CPU or total wall time.")
for rows in (1:length(bt.ret),max(1,length(bt.ret)-499):length(bt.ret))
    println("Decision-bucket elapsed sums, decisions $(first(rows)):$(last(rows)) (nested in execution buckets)")
    for (i,bucket) in enumerate(bt.timing_buckets)
        @printf("  %-10s %.3f\n",bucket,sum(view(bt.timings,rows,i)))
    end
end

println("\n==========================================================================")
println("Path Kelly V1 Performance: $(bt.dates[1]) to $(bt.dates[end]) ($(length(bt.ret)) daily decisions, $(length(b.symbols)) symbols)")
println("==========================================================================")

row(n, r) = @printf("%-30s CAGR %6.2f%%  vol %5.1f%%  Sharpe %5.2f  maxDD %5.1f%%  final %6.2fx\n",
    n, 100r.cagr, 100r.vol, r.sharpe, 100r.maxdd, r.final)

row("Path Kelly V1 (Unified Operator)", summarize(bt.ret))
row("Equal-Weight Benchmark (Daily Rebal)", summarize(bt.ew))

neff(W) = mean(1.0 ./ sum(abs2, W, dims = 2))
@printf("\nAverage Effective Number of Holdings (1/Σw²): %.2f / %d\n", neff(bt.weights), length(b.symbols))

println("\nYearly Return Comparison:")
println("Year   Path Kelly V1   Equal-Weight")
for y in unique(year.(bt.dates))
    i = year.(bt.dates) .== y
    ret_y = expm1(sum(log1p.(bt.ret[i])))
    ew_y  = expm1(sum(log1p.(bt.ew[i])))
    @printf("%d      %8.1f%%       %8.1f%%\n", y, 100ret_y, 100ew_y)
end

println("\nTop 10 Average Allocations:")
mean_weights = vec(mean(bt.weights, dims=1))
top_indices = sortperm(mean_weights, rev=true)[1:min(10,length(mean_weights))]
for idx in top_indices
    @printf("  %-8s %5.2f%%\n", b.symbols[idx], 100mean_weights[idx])
end
