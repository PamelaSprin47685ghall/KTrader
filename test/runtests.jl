using Test, Random, JSON3, LinearAlgebra, Dates, Statistics
using KTrader

# synthetic 3-asset geometric random walks with one positive-drift asset
function synth(T = 1500; seed = 3)
    rng = MersenneTwister(seed)
    r = 0.01 .* randn(rng, T, 3); r[:, 1] .+= 0.002
    exp.(cumsum(r, dims = 1))
end

@testset "Kelly" begin
    rng = MersenneTwister(1)
    X = hcat(exp.(0.02 .+ 0.05 .* randn(rng, 4000)), exp.(-0.01 .+ 0.05 .* randn(rng, 4000)),
             exp.(0.01 .+ 0.05 .* randn(rng, 4000)))
    w = KTrader.kelly_weights(X)
    @test isapprox(sum(w), 1; atol = 1e-6) && all(w .>= 0)
    # KKT: marginal growth equal on held assets, not larger on unheld
    g = vec(mean(X ./ (X * w), dims = 1))
    @test all(g .<= 1 + 1e-4)
    @test g[1] ≈ 1 atol = 1e-3          # asset 1 is held
    @test w[2] < 1e-3                   # negative-drift asset not held
    @test w[1] > w[3] > w[2]
end

@testset "Model causality & units" begin
    P = synth()
    w = path_kelly(P; S = 200, rng = MersenneTwister(5))
    @test isapprox(sum(w), 1; atol = 1e-6) && all(w .>= 0)
    # appending future rows must not change the decision at the earlier time
    P2 = vcat(P, P[end:end, :] .* 3)
    @test path_kelly(P2[1:size(P, 1), :]; S = 200, rng = MersenneTwister(5)) ≈ w
    # rescaling an asset's price level (split-like) must not change weights
    Q = copy(P); Q[:, 2] .*= 50
    @test path_kelly(Q; S = 200, rng = MersenneTwister(5)) ≈ w atol = 1e-4
end

@testset "Backtest accounting" begin
    P = synth(900)
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:899)), ["A", "B", "C"], P, P)
    bt = backtest(b; from = b.dates[600], S = 100)
    @test all(sum(bt.weights, dims = 2) .≈ 1) && all(bt.weights .>= 0)      # fully invested, no cash, leverage 1
    k = 1; t = 600
    manual = sum(bt.weights[k, :] .* (P[t+1, :] ./ P[t, :])) - 1
    @test bt.ret[k] ≈ manual
    @test bt.wealth[end] ≈ prod(1 .+ bt.ret)
end

@testset "Broker" begin
    calls = Tuple[]
    orders = Any[]
    function req(method, path, form)
        push!(calls, (method, path))
        if method != "GET"; return nothing; end
        path == "/v1/markets/clock" && return JSON3.read("""{"clock":{"state":"open"}}""")
        endswith(path, "/balances") && return JSON3.read("""{"balances":{"total_equity":10000,"total_cash":1000}}""")
        endswith(path, "/positions") && return JSON3.read("""{"positions":{"position":[{"symbol":"A","quantity":50},{"symbol":"B","quantity":0}]}}""")
        endswith(path, "/orders") && return JSON3.read("""{"orders":{"order":{"id":7,"symbol":"A","status":"open"}}}""")
        occursin("/quotes", path) && return JSON3.read("""{"quotes":{"quote":[
          {"symbol":"A","bid":99.9,"ask":100.1,"last":100},{"symbol":"B","bid":9.9,"ask":10.1,"last":10}]}}""")
    end
    br = KTrader.Broker("", "X", "T", false, req)
    acts = rebalance!(br, ["A", "B"], [0.2, 0.5])
    @test [(a.sym, a.side, a.qty) for a in acts][1] == ("A", "sell", 30)      # 50 → 20 shares
    @test KTrader.target_shares([0.2, 0.5], 10000, [100, 10]) == [20, 500]
    @test any(c -> c == ("DELETE", "/v1/accounts/X/orders/7"), calls)
    buys = filter(a -> a.side == "buy", acts)
    @test sum(a.qty * a.price for a in buys) <= 1000 * 0.98 + 1e-6            # cash cap, pre-sell
end

@testset "Dry-run issues no writes" begin
    # the real tradier() request closure must refuse non-GET when live=false
    br = KTrader.tradier(; live = false, base = "http://127.0.0.1:9", account = "X", token = "T")
    @test br.request("POST", "/x", ["a" => "b"]) === nothing
    @test br.request("DELETE", "/x", nothing) === nothing
end

@testset "Live preview" begin
    P = synth(300)
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:299)), ["A", "B", "C"], P .* 2, P)
    st = LiveState(b, b.symbols)
    H = preview_history(st, P[end, :] .* 2 .* 1.1)
    @test size(H, 1) == 301 && H[end, :] ≈ P[end, :] .* 1.1
end

# ---- response phase, neutrality, baseline, convergence --------------------------------
function factor_world(kind; T = 4000, N = 6, seed = 1)
    rng = MersenneTwister(seed); f = zeros(T); x = zeros(T); r = zeros(T, N)
    load = 1 .+ 0.2 .* randn(rng, N)
    for t in 2:T
        if kind == :trend
            f[t] = 0.25 * f[t-1] + 0.01 * randn(rng)
        else
            c = t > 21 ? mean(x[t-20:t-1]) : 0.0
            f[t] = -0.15 * (x[t-1] - c) + 0.01 * randn(rng)
        end
        x[t] = x[t-1] + f[t]
        r[t, :] = load .* f[t] .+ 0.004 .* randn(rng, N)
    end
    exp.(cumsum(r, dims = 1))
end
macro_a(ph, τs) = sum(p.a for p in ph if p.macro_ && p.τ in τs)

@testset "θ posterior: beliefs are measured, not assumed" begin
    mac(f) = [m for m in theta_posterior(f; draws = 2000) if m.macro_]
    ft, fr = fit_response(factor_world(:trend)), fit_response(factor_world(:revert))
    mt, mr = mac(ft), mac(fr)
    bt, br = mt[argmin([m.alpha for m in mt])], mr[argmin([m.alpha for m in mr])]    # band with the most evidence
    @test bt.p_revert < 0.05 && bt.R1 > 0.95 && bt.ρ5 > 0.3          # trend world: "reverts" rejected, ρ clearly > 0
    @test br.p_revert > 0.95 && br.R1 > 0.9 && br.ρ5 > 0.05          # reversion world: accepted
    @test 0 <= bt.θmean < 2π && all(m.ρ5 <= m.ρ50 <= m.ρ95 for m in mt)
    # bands without evidence: ρ → 0 and θ is NOT given a direction (uniform, R1 ≈ 0)
    dead = [m for m in mt if m.alpha > 1e6]
    @test !isempty(dead) && all(m.ρ95 < 0.01 && m.R1 < 0.2 for m in dead)

    # two independent factors → non-macro modes exist; neutrality is exact on the *posterior mean*
    rng = MersenneTwister(6)
    f = 0.01 .* randn(rng, 3000, 2); ld = randn(rng, 2, 8)
    fit = fit_response(exp.(cumsum(f * ld .+ 0.004 .* randn(rng, 3000, 8), dims = 1)))
    nm = filter(m -> !m.macro_, theta_posterior(fit))
    @test !isempty(nm)
    @test isapprox(sum(m.a for m in nm), 0; atol = 1e-9) && isapprox(sum(m.b for m in nm), 0; atol = 1e-9)
    # …and on every posterior draw (the constraint is part of the prior, not a projection of the mean)
    U = cholesky(fit.post.A).U
    B = fit.Z * (fit.post.m .+ U \ randn(MersenneTwister(1), size(U, 1), 50))
    idx = [l.k >= 2 for l in fit.labels]
    @test all(abs.(sum(B[2 .* findall(idx) .- 1, :]; dims = 1)) .< 1e-8)
end

@testset "ARD: no evidence ⇒ ρ→0, θ not collapsed" begin
    rng = MersenneTwister(2)
    f = fit_response(exp.(cumsum(0.01 .* randn(rng, 3000, 6), dims = 1)))       # white noise
    post = theta_posterior(f; draws = 3000)
    @test maximum(m.ρ95 for m in post) < 0.05
    @test minimum(m.alpha for m in post) > 1e3 * minimum(m.alpha for m in theta_posterior(fit_response(factor_world(:trend)); draws = 10))
    # macro-mode θ posterior is wide: no direction is preferred when there is no evidence
    @test minimum(m.R1 for m in post if m.macro_) < 0.9
    @test mean(abs.(conditional_mean(f))) < 0.1 * 0.01
end

@testset "ARD: exact recovery of a planted response, rotation-invariant prior" begin
    # Synthetic regression with 3 modes (pairs), neutrality on k≥2; only mode 1 has signal.
    rng = MersenneTwister(31); n = 6000; M = 3; P = 2M
    X = randn(rng, n, P)
    βtrue = [0.3, -0.2, 0, 0, 0, 0]
    y = X * βtrue .+ randn(rng, n)
    Cn = zeros(2, P); for m in 2:3; Cn[1, 2m-1] = 1; Cn[2, 2m] = 1; end
    Z = nullspace(Cn)
    post = KTrader.bayes_ard(X' * X, X' * y, dot(y, y), n, Z)
    β = Z * post.m
    @test β[1:2] ≈ βtrue[1:2] atol = 0.05
    @test all(abs.(β[3:end]) .< 0.03)
    @test post.alpha[1] < 1e3 && all(post.alpha[2:3] .> 1e3)            # evidence kills the empty modes
    @test isapprox(β[3] + β[5], 0; atol = 1e-10) && isapprox(β[4] + β[6], 0; atol = 1e-10)
    # rotating the (a,b) coordinates of every mode leaves the fitted values and α unchanged
    R = kron(Matrix(I, M, M), [cos(0.7) -sin(0.7); sin(0.7) cos(0.7)])
    Xr = X * R                                                          # rotated regressors
    Zr = R' * Z
    pr = KTrader.bayes_ard(Xr' * Xr, Xr' * y, dot(y, y), n, Zr)
    @test pr.alpha ≈ post.alpha rtol = 1e-3
    @test Xr * (Zr * pr.m) ≈ X * β atol = 1e-6
end

@testset "No edge in white noise" begin
    rng = MersenneTwister(2)
    P = exp.(cumsum(0.01 .* randn(rng, 3000, 6), dims = 1))
    f = fit_response(P)
    @test mean(abs.(conditional_mean(f))) < 0.1 * 0.01     # ≪ daily σ
end

@testset "Baseline differs only through the response" begin
    P = factor_world(:trend)
    S = 300
    w1 = path_kelly(P; S, rng = MersenneTwister(4), response = true)
    w0 = path_kelly(P; S, rng = MersenneTwister(4), response = false)
    @test isapprox(sum(w0), 1; atol = 1e-6)
    @test w1 != w0
end

@testset "Numerical convergence is judged by utility loss" begin
    P = factor_world(:trend, T = 2500)
    f = fit_response(P)
    ref = exp.(predict(f; S = 6000, rng = MersenneTwister(9)))
    J(w) = mean(log.(ref * w))
    wstar = KTrader.kelly_weights(ref)
    loss(S) = J(wstar) - J(KTrader.kelly_weights(exp.(predict(f; S, rng = MersenneTwister(S)))))
    @test loss(2000) <= loss(100) + 1e-5
    @test loss(2000) < 1e-3
end

@testset "Live bar finality & settle" begin
    @test KTrader.is_final(Date(2026, 10, 5), DateTime(2026, 10, 6, 10))
    @test !KTrader.is_final(Date(2026, 10, 6), DateTime(2026, 10, 6, 15))
    @test KTrader.is_final(Date(2026, 10, 6), DateTime(2026, 10, 6, 17))
    @test !KTrader.is_final(Date(2026, 10, 7), DateTime(2026, 10, 6, 18))        # later date never final
    P = synth(300)
    st = LiveState(KTrader.Bars(collect(Date(2026, 10, 5) .- Day.(299:-1:0)), ["A", "B", "C"], P, P), ["A", "B", "C"])
    @test KTrader.settle_due(st, DateTime(2026, 10, 6, 17, 30))      # Tue after close
    @test !KTrader.settle_due(st, DateTime(2026, 10, 6, 11))         # intraday
    @test !KTrader.settle_due(st, DateTime(2026, 10, 10, 17, 30))    # Saturday
end

@testset "Eligibility & fully-invested weights with late-listed assets" begin
    P = synth(1500)
    P[1:700, 3] .= NaN                                   # asset 3 lists late, 800 rows ≥ MINROWS
    @test KTrader.eligible(P) == [true, true, true]
    P[1:1100, 3] .= NaN                                  # 400 rows < MINROWS: numerically too short to fit
    @test KTrader.eligible(P) == [true, true, false]
    w = path_kelly(P; S = 100, rng = MersenneTwister(1))
    @test w[3] == 0 && isapprox(sum(w), 1; atol = 1e-8)
end

@testset "Untradable / unlisted / short history" begin
    rng = MersenneTwister(8)
    T = 900
    P = exp.(cumsum(0.01 .* randn(rng, T, 4), dims = 1)); P[:, 1] .*= exp.(0.0)
    P[1:700, 4] .= NaN                                       # lists late: 200 rows < MINROWS
    raw = copy(P); raw[400:420, 2] .= NaN                    # asset 2 halted for 21 sessions
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:T-1)), ["A", "B", "C", "D"], raw, raw)
    @test !any(b.bar[400:420, 2]) && all(isfinite, b.adj[400:420, 2])    # marked at last price, not tradable
    @test b.adj[410, 2] == b.adj[399, 2] && all(.!isfinite.(b.adj[1:700, 4]))

    # model never touches asset D (history) and never trades B while halted
    mask, R = KTrader.scenarios(b.adj[1:T, :]; S = 100, rng = MersenneTwister(1))
    @test mask == [true, true, true, false]
    held = [0.25, 0.4, 0.35, 0.0]
    free = mask .& [true, false, true, false]
    w = KTrader.allocate(R, mask, free, held)
    @test w[2] == 0.4 && w[4] == 0 && isapprox(sum(w), 1; atol = 1e-8) && all(w .>= 0)
    @test isapprox(w[1] + w[3], 0.6; atol = 1e-8)
    @test KTrader.allocate(R, mask, falses(4), held) == held     # nothing tradable → hold
    # locked weight at 100% leaves nothing to allocate → hold
    @test KTrader.allocate(R, mask, free, [0, 1.0, 0, 0]) == [0, 1.0, 0, 0]
    # equal-weight benchmark obeys the same rule
    @test KTrader.equal_weights([true, false, true, false], [0.25, 0.4, 0.35, 0.0]) ≈ [0.3, 0.4, 0.3, 0.0]

    # backtest: weight in a halted asset only drifts with its marking price
    bt = backtest(b; from = b.dates[600], S = 60)
    @test all(bt.weights[:, 4] .== 0 .|| bt.dates .>= b.dates[701])  # D never before it has history
    @test all(isapprox.(sum(bt.weights, dims = 2), 1; atol = 1e-8))
    @test all(isapprox.(sum(bt.weights_ew, dims = 2), 1; atol = 1e-8))
end

@testset "Live: halted symbol is held, never traded" begin
    rng = MersenneTwister(11); T = 1300
    P = exp.(cumsum(0.01 .* randn(rng, T, 3), dims = 1))
    b = KTrader.Bars(collect(Date(2020, 1, 1) .+ Day.(0:T-1)), ["A", "B", "C"], P, P)
    posted = Any[]
    function req(m, p, f)
        m != "GET" && (push!(posted, (m, p, f)); return nothing)
        p == "/v1/markets/clock" && return JSON3.read("""{"clock":{"state":"open"}}""")
        endswith(p, "/balances") && return JSON3.read("""{"balances":{"total_equity":10000,"total_cash":1000}}""")
        endswith(p, "/positions") && return JSON3.read("""{"positions":{"position":[{"symbol":"B","quantity":40}]}}""")
        endswith(p, "/orders") && return JSON3.read("""{"orders":"null"}""")
        occursin("/quotes", p) && return JSON3.read("""{"quotes":{"quote":[
          {"symbol":"A","bid":9.9,"ask":10.1,"last":10},
          {"symbol":"B","bid":0,"ask":0,"last":100},
          {"symbol":"C","bid":19.9,"ask":20.1,"last":20}]}}""")
    end
    br = KTrader.Broker("", "X", "T", true, req)
    r = live_step!(LiveState(b, b.symbols), br; S = 60, rng = MersenneTwister(1))
    @test r.weights[2] ≈ 0.4                                   # 40×100/10000, B halted → untouched
    @test isapprox(sum(r.weights), 1; atol = 1e-8)
    @test all(a -> a.sym != "B", r.orders) && !any(c -> occursin("B", string(c[3])), posted)
    @test !isempty(r.orders)
end

@testset "Fractional kernel & mode-space long-memory covariance" begin
    @test KTrader.frac_weights(1.0, 5) == ones(5)                         # d=1: constant variance
    w = KTrader.frac_weights(0.3, 6)
    @test w[1] == 1 && all(diff(w) .< 0) && w[3] ≈ 0.3 * 1.3 / 2          # Γ(k+d)/(Γ(k+1)Γ(d)) recursion
    rng = MersenneTwister(12)
    lm(d, n) = (w = KTrader.frac_weights(d, n); e = randn(rng, n);
                h = [dot(w[1:t], e[t:-1:1]) for t in 1:n]; exp.(h .* 0.4 / std(h)) .* 0.01 .* randn(rng, n))
    dm(E) = (v = mode_vol_model(E; K = 1); b = v.modes[1]; sum(b.d .* b.p))
    Econst = 0.01 .* randn(rng, 2500, 3)
    Elm = reduce(hcat, [lm(0.4, 2500) for _ in 1:3])
    @test dm(Econst) > 0.5             # no memory → posterior toward d≈1
    @test dm(Elm) < dm(Econst)         # long memory detected
    vm = mode_vol_model(Elm; K = 1)
    @test sum(vm.modes[1].p) ≈ 1 && all(vm.idio.σ[1] .> 0)
end

@testset "Drift shrinkage (tau integrated out)" begin
    se = fill(0.2, 40)
    noise = KTrader.shrink_drift(0.2 .* randn(MersenneTwister(2), 40), se)         # x̄ ~ pure noise
    @test sum(noise.p) ≈ 1 && noise.p[1] > noise.p[end]                           # c=0 favoured over c=3.2
    @test std(noise.mean) < 0.35 * 0.2                                            # almost fully shrunk
    x2 = 5 .* randn(MersenneTwister(3), 40)
    strong = KTrader.shrink_drift(x2, se)
    @test cor(strong.mean, x2) > 0.999 && std(strong.mean) > 0.9 * std(x2)        # signal ≫ noise: ~no shrinkage
    @test strong.p[end] > 0.99
end

@testset "Long-run variance: iid ≈ σ², persistent ≫ σ²" begin
    r = randn(MersenneTwister(4), 3, 4000)
    @test all(0.8 .< KTrader.lrv(r) .< 1.25)
    ar = zeros(1, 4000); e = randn(MersenneTwister(5), 4000)
    for t in 2:4000; ar[1, t] = 0.8 * ar[1, t-1] + e[t]; end                      # LRV = 1/(1-0.8)² = 25 ≫ var 2.8
    @test KTrader.lrv(ar)[1] > 3 * var(ar)
end

@testset "Fractal metric reaches the modes" begin
    rng = MersenneTwister(21); f = randn(rng, 3000)
    u = hcat(f .* 1.0 .+ 0.3 .* randn(rng, 3000), f .* 8.0 .+ 2.4 .* randn(rng, 3000), randn(rng, 3000))
    Φ = KTrader.natural_modes(cov(u))
    @test abs(Φ[2, 1]) > 3 * abs(Φ[1, 1])          # covariance (not correlation): the large-scale asset dominates the macro mode
end

@testset "Scenario first moment is exact (Monte-Carlo noise cannot drive Kelly)" begin
    P = factor_world(:trend, T = 2500)
    f = fit_response(P)
    for kw in ((response = true, volmodel = true), (response = false, volmodel = false))
        L = predict(f; S = 200, rng = MersenneTwister(3), kw...)
        want = f.drift.mean .+ (kw.response ? KTrader.response_mean(f) : 0)
        @test vec(mean(L, dims = 1)) ≈ want atol = 1e-12
    end
    # pure noise: the drift posterior must be nearly flat across assets (Kelly at leverage 1 is
    # hypersensitive to drift dispersion: 0.75 %/yr already concentrates it into ~4 names)
    rng = MersenneTwister(5)
    fn = fit_response(exp.(cumsum(0.01 .* randn(rng, 2500, 20), dims = 1)))
    @test std(fn.drift.mean) * 252 < 0.015
    # zero dispersion ⇒ Kelly spreads over (almost) all assets
    Lz = 0.01 .* randn(MersenneTwister(2), 3000, 20); Lz .-= mean(Lz, dims = 1)
    wz = KTrader.kelly_weights(exp.(Lz))
    @test 1 / sum(wz .^ 2) > 8
end

@testset "No leakage: geometry is estimated on rows the response evidence never sees" begin
    alive = Float64[]; rmax = Float64[]
    for seed in 1:30
        rng = MersenneTwister(seed)
        f = fit_response(exp.(cumsum(0.01 .* randn(rng, 3000, 6), dims = 1)))
        @test f.ngeom + f.nreg + KTrader.WARMUP == f.nrows                 # disjoint blocks: geometry | evidence | warm-up
        ph = theta_posterior(f; draws = 1000)
        push!(alive, mean(m.alpha < 1e6 for m in ph if m.macro_)); push!(rmax, maximum(m.ρ95 for m in ph))
    end
    @test mean(alive) < 0.15                       # spurious macro-mode rate on white noise (was ≈ 0.41 with shared rows)
    @test maximum(rmax) < 0.06
    # planted responses are still found, with the right sign
    ft, fr = fit_response(factor_world(:trend)), fit_response(factor_world(:revert))
    @test sum(l.a for l in theta_posterior(ft; draws = 1000) if l.macro_) < -0.1
    @test sum(l.a for l in theta_posterior(fr; draws = 1000) if l.macro_) > 0.1
end

@testset "Ragged panel: every asset keeps its whole history" begin
    function ragged(young; T = 1800, seed = 1)
        rng = MersenneTwister(seed); f = 0.006 .* randn(rng, T)
        P = exp.(cumsum(f .+ 0.008 .* randn(rng, T, 4), dims = 1)); P[1:T-young, 4] .= NaN; P
    end
    P = ragged(600)
    @test KTrader.eligible(P) == [true, true, true, true]                          # 600 ≥ MINROWS
    fa, fb = fit_response(P[:, 1:3]), fit_response(P)
    @test fa.nrows == 1800 && fb.nrows == 1800                                     # the long assets' rows are all used
    @test fb.nreg + fb.ngeom + KTrader.WARMUP == fb.nrows
    # the young asset enters without cutting the old assets' history: their conditional mean barely moves
    # (the common-window fit would use 600 rows only)
    cut = fit_response(P[end-599:end, :])
    d(f) = maximum(abs.(conditional_mean(f)[1:3] .- conditional_mean(fa))) * 252
    @test d(fb) < 0.02
    @test d(cut) > 5 * d(fb)
    # …and the decision on the old assets is continuous in the new listing
    S = 1000
    wa = path_kelly(P[:, 1:3]; S, rng = MersenneTwister(1))
    wb = path_kelly(P; S, rng = MersenneTwister(1))
    @test sum(abs, wb[1:3] ./ sum(wb[1:3]) .- wa) < 0.15
    @test isapprox(sum(wb), 1; atol = 1e-8) && all(wb .>= 0)
    # a young asset below the numerical minimum is simply not modelled
    @test KTrader.eligible(ragged(KTrader.MINROWS - 1)) == [true, true, true, false]
end

@testset "Geometry bootstrap is block-based (keeps time dependence)" begin
    rng = MersenneTwister(7); n = 20000
    x = zeros(n); for t in 2:n; x[t] = 0.8 * x[t-1] + randn(rng); end
    ac(y) = cor(y[1:end-1], y[2:end])
    L = 50
    idx = KTrader.stationary_bootstrap(rng, n, L)
    @test ac(x) ≈ 0.8 atol = 0.02
    @test ac(x[idx]) ≈ ac(x) atol = 0.1            # blocks keep the serial dependence (iid resampling gives ≈ 0)
    @test abs(ac(x[rand(rng, 1:n, n)])) < 0.05     # the contrast: iid row resample destroys it
    @test isapprox(n / (1 + count(i -> idx[i] != mod1(idx[i-1] + 1, n), 2:n)), L; rtol = 0.25)   # mean block length ≈ L
    f = fit_response(exp.(cumsum(0.01 .* randn(MersenneTwister(2), 2000, 4), dims = 1)))
    @test length(f.draws) == KTrader.GEOM_DRAWS
    @test !all(d.s1 ≈ f.draws[1].s1 for d in f.draws)                              # geometry draws genuinely differ
end

@testset "Identifiability: a mode must be stable across the bootstrap" begin
    rng = MersenneTwister(6)
    fa = 0.01 .* randn(rng, 3000, 2); ld = randn(rng, 2, 8)
    f2 = fit_response(exp.(cumsum(fa * ld .+ 0.004 .* randn(rng, 3000, 8), dims = 1)))
    f0 = fit_response(exp.(cumsum(0.01 .* randn(MersenneTwister(1), 3000, 8), dims = 1)))
    nm(f, τ) = count(l -> l.τ == τ, f.labels)
    @test all(nm(f2, τ) >= 2 for τ in KTrader.BANDS[1:3])        # two real factors → two stable modes at short bands
    @test sum(nm(f0, τ) for τ in KTrader.BANDS) < sum(nm(f2, τ) for τ in KTrader.BANDS)
end

for f in ("data_tests.jl", "execution_tests.jl", "modecov_tests.jl", "backtest_tests.jl", "belief_tests.jl")
    @testset "$f" begin
        include(f)
    end
end

@testset "wcov: a pair with no common weighted row never yields NaN" begin
    X0 = [1.0 0 ; 2 0; 3 5; 4 6]; M = [1.0 0; 1 0; 1 1; 1 1]
    @test_throws ErrorException KTrader.wcov(X0, M, [1.0, 1, 0, 0])               # pair (1,2) unidentified, no fallback
    C = KTrader.wcov(X0, M, [1.0, 1, 0, 0]; fallback = KTrader.wcov(X0, M, ones(4)))
    @test all(isfinite, C) && C[1, 2] == KTrader.wcov(X0, M, ones(4))[1, 2]
    # resampled geometry on a ragged panel whose youngest asset a block resample can miss entirely
    rng = MersenneTwister(5); P = exp.(cumsum(0.01 .* randn(rng, 1500, 5), dims = 1)); P[1:700, 5] .= NaN
    for seed in 1:40
        @test all(isfinite, fit_response(P; rng = Random.Xoshiro(seed)).s1)
    end
end
