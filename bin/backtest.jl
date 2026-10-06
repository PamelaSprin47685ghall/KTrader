# Usage: julia -t 6 --project=. bin/backtest.jl [YYYY-MM-DD]
using KTrader, Dates, Statistics, LinearAlgebra, Printf
BLAS.set_num_threads(1)

b = load_bars(joinpath(@__DIR__, "..", "data"))
from = isempty(ARGS) ? today() - Year(10) : Date(ARGS[1])
S = parse(Int, get(ENV, "SCENARIOS", "300"))

println("Running Path Kelly V1 Backtest from $from...")
bt = backtest_v1(b; from, S)

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
top_indices = sortperm(mean_weights, rev=true)[1:10]
for idx in top_indices
    @printf("  %-8s %5.2f%%\n", b.symbols[idx], 100mean_weights[idx])
end
