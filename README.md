# KTrader — Path Kelly (Julia)

Theory: `AGENTS.md`. Chain: price history → natural modes → ruler-normalised multi-scale
observations (q, p) → belief posterior Π(ρ, θ | H) → predictive scenarios (drift hierarchy,
long-memory covariance) → exact Kelly (`max E log w'X`, long-only, fully invested: Σw = 1,
leverage 1, **no cash asset**) → `rebalance!`.

## Layers (never mixed)

| layer | what | where |
|---|---|---|
| market theory | fixed law over the whole price path | `model.jl`: ruler, modes, (q,p), response, covariance |
| epistemic | what the finite history lets us believe; belongs to theory, not to numerics | grouped ARD (α_k), neutrality as part of the prior, drift hierarchy, geometry uncertainty |
| decision | exact Kelly on the posterior predictive | `kelly_weights`, `allocate` |
| numerics | S scenarios, solver, FFT, moment matching | engineering only; must not change the objective |
| execution | weights → orders on a dedicated account | `broker.jl`, `live.jl` |

## Belief vs observation

`q` (position vs trailing centre) and `p` (slope of the centre) are **observations** computed from
prices. `(ρ_k, θ_k)` are **beliefs** about how a natural mode responds: `a = ρcosθ` multiplies −q
(θ near 0: deviations revert; near π: they persist), `b = ρsinθ` multiplies p (confirm the move vs
front-run it). The prior on every mode is rotation-invariant (θ uniform), the strength ρ_k is set
by evidence (ARD; no evidence ⇒ ρ→0), and the neutrality `Σ a_k = Σ b_k = 0` over non-macro modes
is part of the prior. `theta_posterior(fit)` reports the posterior, never a point θ. Price-level
H ≈ 0.5 does not make this layer empty; it only says that, in this universe, the evidence for ρ is
weak, which the posterior must show.

## Files

| file | role |
|---|---|
| `src/data.jl` | Yahoo daily bars; `Bars.bar` marks real prints (tradable days), NaN before listing |
| `src/model.jl` | `path_kelly(adj; …) -> weights` (one per asset, Σ=1); pure and causal |
| `src/backtest.jl` | fills at each daily close, instantly, any size; no frictions; locked (untradable) positions drift |
| `src/broker.jl` | Tradier executor derived from `rebalance.mjs`; non-live = no writes |
| `src/live.jl` | current price = preview close of today → same `path_kelly`; `settle!` stores the real close after the market closes |
| `universe.txt` | hand-picked universe (the only human prior) |

## Usage

```
julia --project=. bin/fetch.jl                            # download data/
julia -t auto --project=. bin/backtest.jl [YYYY-MM-DD]    # default: exactly 10y ago, vs daily equal-weight
ABLATE=1 julia -t auto --project=. bin/backtest.jl        # also: no long-memory cov / no response / both off
TRADIER_ACCOUNT_ID=.. TRADIER_TOKEN=.. julia --project=. bin/live.jl [--live]
julia --project=. -e 'using Pkg; Pkg.test()'
```

## What the backtest is, and is not

The backtest assumes the book can be filled at every daily close, instantly and in any size, with
no fees, financing or slippage (by design). The model sees close t and trades at close t. That is an
**idealised upper bound** for the theory. It is not a reproduction of the live procedure, which
uses the current intraday price as today's preview close; reproducing that would need intraday
history. Benchmark: daily equal weight over the same history-eligible assets with the same lock
rule, so any gap is the model's, not the universe's.

Untradable days (no real bar) freeze the position; unlisted or too-short-history assets get weight
0; the tradable assets share what is left.

## References (ideas used; none is claimed as new here)

- Kelly criterion, log-optimal growth: Kelly (1956); Thorp; Boyd et al., "Performance bounds and
  suboptimal policies for multi-period investment" (Stanford) for the convex-programming form.
- Fractional Brownian motion and Hurst exponent: Mandelbrot & Van Ness (1968).
- Long-memory volatility and fractional integration: Granger & Joyeux (1980), Hosking (1981),
  FIGARCH — Baillie, Bollerslev & Mikkelsen (1996).
- Rough volatility (H ≈ 0.1 of log-volatility, not of price): Gatheral, Jaisson & Rosenbaum,
  "Volatility is rough" (2018).
- Evidence maximisation / ARD: MacKay (1992); Tipping (2001).
- Empirical-Bayes shrinkage of means: Efron & Morris; Jorion (1986) for portfolio drifts.
- Long-run variance: Newey & West (1987).
- Random-matrix noise edge (Marchenko–Pastur), used only if still present in `model.jl`: Marchenko &
  Pastur (1967).

Numerics are delegated: Convex + Clarabel (Kelly), LinearAlgebra, DSP (FFT convolution), HTTP/JSON3/CSV.
