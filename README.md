# KTrader — Path Kelly 1.0 (Julia)

Theory & Mathematical Closure: `AGENTS.md`. 
Chain: price history → center-of-mass macro & zero-sum relative space → multi-scale cumulative path coordinates → multivariate response operator G_b = A_b + i B_b → exact Gaussian-conditioned trace neutrality tr(A_b)=0, tr(B_b)=0 on parameter support → causal fractional long-memory innovation law → exact matrix-free Matrix-Normal posterior predictive sampling → exact Kelly decision (`max E log w'X`, long-only, fully invested: Σw = 1, leverage 1, **no cash asset**) → `rebalance!`.

## Core Mathematical Principles

1. **Center-of-Mass Macro & Permutation-Symmetric Relative Space**:
   $$m_t = \frac{1}{\sqrt{N_t}} \sum_{j \in \text{alive}(t)} u_{t, j}, \quad u_{\perp, t} = (I - e_0 e_0^T) u_t$$
   Zero coordinate-ordering bias: 100% permutation-symmetric across all assets.

2. **Multivariate Response Operator & Exact Trace Neutrality**:
   $$G_b = A_b + i B_b \quad (\forall b \in \text{BANDS})$$
   Diagonal entries $A_{jj}, B_{jj}$ govern self-continuation and mean-reversion. Off-diagonal entries $A_{lj}, B_{lj} (l \neq j)$ capture energy transfer between assets (sector rotation).
   Operator trace neutrality is enforced directly on the Gaussian parameter support:
   $$\operatorname{tr}(A_b) \equiv 0, \quad \operatorname{tr}(B_b) \equiv 0$$

3. **Causal Fractional Innovation Memory**:
   Continuous uniform prior $p(d) = \text{const}$ integrated via trapezoidal quadrature:
   $$\pi_g \propto \exp(\ell(d_g)) \Delta d_g$$
   Grid refinement strictly improves numerical integral accuracy without altering the theoretical prior.
   Innovation shocks are scaled relative to the real historical bootstrap residual variance $v_{\text{bootstrap}}$.

4. **Strict Separation of Signal Prices and Marking Prices**:
   Untradable days are strictly `NaN` in signal prices (no fake $r = 0$ in the model).

## Verification (214 / 214 Pass)

Run full test suite:
```bash
julia -t 4 --project=. test/runtests.jl
```

Constitutional invariants verified:
- **Price Scale Invariance**: $P_i \to c_i P_i \implies w$ strictly identical.
- **Asset Permutation Invariance**: permuting asset input columns permutes output weights identically.
- **Center-of-Mass Orthogonality**: relative space strictly orthogonal to $e_0$ to machine zero.
- **Conditioned Trace Neutrality**: verified identically on every single posterior draw $G^{(s)}$.
- **Pure Noise Null**: under pure Brownian motion, Evidence $\alpha \to \infty$, $\|\mu\|_\infty \le 0.0005$.
- **Continuous Prior Quadrature**: $\sum p(d) \equiv 1.0$.

## Usage

```bash
julia --project=. bin/fetch.jl                            # download data/
julia -t 6 --project=. bin/backtest.jl [YYYY-MM-DD]       # run backtest
TRADIER_ACCOUNT_ID=.. TRADIER_TOKEN=.. julia --project=. bin/live.jl [--live]
```
