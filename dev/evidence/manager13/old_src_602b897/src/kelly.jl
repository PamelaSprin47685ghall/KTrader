"""
Exact sample log-Kelly on the simplex. No clipping of gross returns or change to
mean-variance utility is permitted. The custom solver is accepted only with an
original-objective certificate; Clarabel solves the identical problem otherwise.
"""
function kelly_inputs(X, budget, base)
    S, n = size(X)
    S > 0 && n > 0 || throw(ArgumentError("Kelly needs scenarios and free assets"))
    isfinite(budget) && budget >= 0 || throw(ArgumentError("invalid Kelly budget"))
    all(x -> isfinite(x) && x > 0, X) || throw(ArgumentError("gross returns must be positive and finite"))
    b = base === nothing ? zeros(S) : base
    length(b) == S || throw(DimensionMismatch("locked wealth scenario count"))
    all(x -> isfinite(x) && x >= 0, b) || throw(ArgumentError("invalid locked wealth"))
    b
end

"""Concavity gives F(w*) - F(w) <= budget*maximum(g) - dot(g,w).
Dual slacks maximum(g)-g are nonnegative; their complementarity is the KKT residual.
The bound and residual refer to the original objective, not its log barrier.
"""
function kelly_certificate(X::AbstractMatrix{Float64}, w::AbstractVector{Float64}; budget=1.0, base=nothing)
    S, n = size(X)
    length(w) == n || throw(DimensionMismatch("Kelly weights"))
    b = kelly_inputs(X, budget, base)
    feasibility = max(abs(sum(w) - budget), max(0.0, -minimum(w)))
    wealth = X * w .+ b
    if !all(isfinite, w) || !all(x -> isfinite(x) && x > 0, wealth)
        return (; feasibility=Inf, kkt_residual=Inf, objective_gap=Inf, objective=-Inf)
    end
    g = X' * (1.0 ./ wealth) ./ S
    dual = maximum(g)
    complementarity = w .* (dual .- g)
    gap = max(0.0, budget * dual - dot(w, g))
    (; feasibility, kkt_residual=maximum(abs, complementarity), objective_gap=gap,
       objective=sum(log, wealth) / S)
end
certified(c, tol) = c.feasibility <= tol && c.kkt_residual <= tol && c.objective_gap <= tol

function clarabel_kelly_solver(X::AbstractMatrix{Float64}; budget=1.0, base=nothing, tol=1e-8)
    b = kelly_inputs(X, budget, base)
    n = size(X, 2)
    budget == 0 && return zeros(n)
    n == 1 && return [Float64(budget)]
    w = Variable(n)
    problem = maximize(sum(log(X * w + b)) / size(X, 1), [w >= 0, sum(w) == budget])
    optimizer = Clarabel.Optimizer()
    for setting in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        Convex.MOI.set(optimizer, Convex.MOI.RawOptimizerAttribute(setting), min(tol / 10, 1e-10))
    end
    solve!(problem, () -> optimizer; silent=true)
    problem.status in (Convex.MOI.OPTIMAL, Convex.MOI.ALMOST_OPTIMAL) ||
        error("Clarabel log-Kelly failed: $(problem.status)")
    result = max.(vec(evaluate(w)), 0.0)
    result .*= budget / sum(result)
    c = kelly_certificate(X, result; budget, base=b)
    certified(c, tol) || error("Clarabel log-Kelly certificate failed: $c")
    result
end

function fast_kelly_solver(X::AbstractMatrix{Float64}; budget=1.0, base=nothing,
                           max_outer=20, max_inner=30, tol=1e-8)
    b = kelly_inputs(X, budget, base)
    S, n = size(X)
    budget == 0 && return zeros(n)
    n == 1 && return [Float64(budget)]
    w = fill(budget / n, n)
    wealth = X * w .+ b
    trial_w = similar(w); trial_wealth = similar(wealth)
    inv_wealth = similar(wealth); g = similar(w)
    scaled = Matrix{Float64}(undef, S, n)
    H = Matrix{Float64}(undef, n, n)
    rhs = Matrix{Float64}(undef, n, 2)
    mu = 1e-2
    failed = false
    for _ in 1:max_outer
        for _ in 1:max_inner
            inv_wealth .= 1.0 ./ wealth
            mul!(g, X', inv_wealth, 1.0 / S, 0.0)
            g .+= mu ./ w
            nu = dot(w, g) / budget
            if maximum(abs, g .- nu) <= max(tol / 10, mu / 100)
                break
            end
            scaled .= X .* (inv_wealth ./ sqrt(S))
            BLAS.syrk!('U', 'T', 1.0, scaled, 0.0, H)
            for j in 1:n
                H[j,j] += mu / w[j]^2
            end
            chol = cholesky!(Symmetric(H, :U); check=false)
            if !issuccess(chol)
                failed = true
                break
            end
            rhs[:,1] .= g .- nu
            rhs[:,2] .= 1.0
            ldiv!(chol, rhs)
            dw = view(rhs, :, 1) .- (sum(view(rhs, :, 1)) / sum(view(rhs, :, 2))) .* view(rhs, :, 2)
            step = 1.0
            for j in 1:n
                dw[j] < 0 && (step = min(step, -0.99 * w[j] / dw[j]))
            end
            objective = sum(log, wealth) / S + mu * sum(log, w)
            directional = dot(g, dw)
            accepted = false
            for _ in 1:40
                trial_w .= w .+ step .* dw
                mul!(trial_wealth, X, trial_w)
                trial_wealth .+= b
                if minimum(trial_w) > 0 && minimum(trial_wealth) > 0
                    trial = sum(log, trial_wealth) / S + mu * sum(log, trial_w)
                    if trial >= objective + 1e-4 * step * directional - 8eps(abs(objective) + 1)
                        accepted = true
                        break
                    end
                end
                step *= 0.5
            end
            if !accepted
                failed = true
                break
            end
            copyto!(w, trial_w); copyto!(wealth, trial_wealth)
        end
        failed && break
        c = kelly_certificate(X, w; budget, base=b)
        certified(c, tol) && return w
        mu *= 0.1
    end
    clarabel_kelly_solver(X; budget, base=b, tol)
end

function kelly_weights_v1(X::AbstractMatrix{Float64}; budget=1.0, base=nothing, use_clarabel=false, tol=1e-8)
    use_clarabel ? clarabel_kelly_solver(X; budget, base, tol) : fast_kelly_solver(X; budget, base, tol)
end

# Multiply only genuinely held columns: 0*NaN must never enter locked wealth.
function locked_wealth(X, locked)
    base = zeros(size(X, 1))
    for j in eachindex(locked)
        locked[j] > 0 || continue
        for s in eachindex(base)
            gross = X[s,j]
            isfinite(gross) || error("held asset $j has no predictive return law")
            base[s] += locked[j] * gross
        end
    end
    base
end

function scenario_weights(X, active_indices, tradable, held=nothing; tol=1e-8)
    N = size(X, 2)
    active = falses(N); active[active_indices] .= true
    free = tradable .& active
    current = held === nothing ? zeros(N) : held
    locked = current .* .!free
    budget = 1.0 - sum(locked)
    indices = findall(free)
    if isempty(indices) || budget <= 1e-12
        return copy(current)
    end
    base = locked_wealth(X, locked)
    out = copy(locked)
    out[indices] .= kelly_weights_v1(view(X, :, indices); budget, base, tol)
    out
end

"""Price history -> exact-objective target weights; untradable holdings stay locked."""
function path_kelly_v1(adj::AbstractMatrix{Float64}; S=300, rng=Random.MersenneTwister(1),
                       ridge_alpha=nothing, ruler_stats=nothing, tradable=nothing, held=nothing,
                       adaptive=false, quadrature_tol=1e-5, max_scenarios=512)
    N = size(adj, 2)
    tr = tradable === nothing ? trues(N) : tradable
    model = fit_v1(adj; ridge_alpha, ruler_stats)
    if adaptive
        return adaptive_scenario_weights(model, tr, held; rng, tol=quadrature_tol, max_scenarios).weights
    end
    X = generate_scenarios_v1(model; S, rng)
    scenario_weights(X, model.active_indices, tr, held)
end
const path_kelly = path_kelly_v1
