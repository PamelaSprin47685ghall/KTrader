# Usage: julia -t 6 --project=. bin/backtest.jl [YYYY-MM-DD]   (default: exactly 10 years ago)
using KTrader, Dates, Statistics, LinearAlgebra, Printf
BLAS.set_num_threads(1)
b = load_bars(joinpath(@__DIR__, "..", "data"))
from = isempty(ARGS) ? today() - Year(10) : Date(ARGS[1])
S = parse(Int, get(ENV, "SCENARIOS", "300"))
runs = [("path-kelly (full)", (response = true, volmodel = true))]
if get(ENV, "ABLATE", "0") == "1"
    push!(runs, ("  no long-memory cov (iid)", (response = true, volmodel = false)))
    push!(runs, ("  no response (drift+cov)", (response = false, volmodel = true)))
    push!(runs, ("  no response, iid cov", (response = false, volmodel = false)))
end
res = [(name, backtest(b; from, S, kw...)) for (name, kw) in runs]
bt = res[1][2]
println("from ", bt.dates[1], " to ", bt.dates[end], " (", length(bt.ret), " daily decisions, ", length(b.symbols), " symbols)")
row(n, r) = @printf("%-26s CAGR %6.2f%%  vol %5.1f%%  Sharpe %5.2f  maxDD %5.1f%%  final %6.2fx\n",
    n, 100r.cagr, 100r.vol, r.sharpe, 100r.maxdd, r.final)
for (name, x) in res; row(name, summarize(x.ret)); end
row("equal-weight (same eligible)", summarize(bt.ew))
println("days with a locked (untradable) position: model ", bt.locked_days, ", equal-weight ", bt.locked_days_ew)
neff(W) = mean(1 ./ sum(abs2, W, dims = 2))
@printf("average effective number of holdings 1/Σw²: model %.2f, equal-weight %.2f\n", neff(bt.weights), neff(bt.weights_ew))
println("\nyear   path-kelly   equal-weight")
for y in unique(year.(bt.dates))
    i = year.(bt.dates) .== y
    @printf("%d  %9.1f%%  %9.1f%%\n", y, 100expm1(sum(log1p.(bt.ret[i]))), 100expm1(sum(log1p.(bt.ew[i]))))
end
println("\nmean weight, top holdings: ",
        join(["$s=$(round(m, digits = 3))" for (s, m) in sort(collect(zip(b.symbols, mean(bt.weights, dims = 1))); by = x -> -x[2])[1:8]], " "))
