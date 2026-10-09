using Test, Random, LinearAlgebra, Statistics
using KTrader

# Budgeted direct-file entry only. The standard suite still includes ALL
# phases: no environment selector, filtering by test name or changed fixtures.
const INCREMENTAL_TEST_PHASES = (:prefix, :ragged, :ownership, :kernels)
function incremental_test_phase(args; standalone)
    isempty(args) && return :all
    standalone || throw(ArgumentError("incremental phases require the direct test-file entry; included suite cannot be partial"))
    length(args)==1 && startswith(only(args),"--phase=") ||
        throw(ArgumentError("expected one --phase=prefix|ragged|ownership|kernels|all"))
    phase=Symbol(only(args)[9:end])
    phase in (:all, INCREMENTAL_TEST_PHASES...) || throw(ArgumentError("unknown incremental phase: $phase"))
    phase
end
const INCREMENTAL_TEST_PHASE = incremental_test_phase(ARGS;
    standalone=abspath(PROGRAM_FILE)==abspath(@__FILE__))
incremental_completed_phases = Symbol[]
println("[incremental phase=",INCREMENTAL_TEST_PHASE,"] BEGIN; default/include entry runs all phases"); flush(stdout)

# One mathematical oracle and strict tolerances; do not compare arbitrary seeded
# eigenvector square roots. Scenario points are coupled through a symmetric root.
function incremental_canonical_model(model)
    cov=model.pred_moments.L_rel*model.pred_moments.L_rel'
    ev=eigen(Symmetric(cov))
    root=(ev.vectors.*sqrt.(max.(ev.values,0.0))')*ev.vectors'
    moments=merge(model.pred_moments,(;L_rel=root))
    args=ntuple(i->fieldname(V1Model,i)==:pred_moments ? moments : getfield(model,i),fieldcount(V1Model))
    V1Model(args...)
end
function incremental_assert_preparation(state,prices)
    p=KTrader._prepare_current!(state)
    oracle=KTrader._prepare_v1(prices;F_folds=state.F_folds)
    @test p.active_idx==oracle.active_idx
    @test p.ts_total==oracle.ts_total
    @test p.s1 ≈ oracle.s1 atol=1e-12 rtol=1e-12
    @test p.s_m ≈ oracle.s_m atol=1e-12 rtol=1e-11
    @test p.s_perp ≈ oracle.s_perp atol=1e-12 rtol=1e-11
    @test p.observed==oracle.observed
    # The regime engine takes the lazy design path: rebuild on demand and
    # compare against the batch design at the original tolerance (this is the
    # mutation check for the lazy rebuild - a wrong rebuild goes red here).
    if p.X_rel_stacked === nothing
        rebuilt=KTrader.build_X_rel_stacked(p.X_rel,p.s_perp,p.ts_total)
        @test rebuilt ≈ oracle.X_rel_stacked atol=1e-10 rtol=1e-11
    else
        @test p.X_rel_stacked ≈ oracle.X_rel_stacked atol=1e-10 rtol=1e-11
    end
    @test p.B_m ≈ oracle.B_m atol=1e-10 rtol=1e-11
    # Y_target_rel is no longer a stored field: the typed PreparedProblem
    # derives the field targets as view(relative_embedding, ts_total .+ 1, :)
    # (predict.jl builds the fold statistics from exactly this view), so
    # recompute both sides the same way — the field-target alignment between
    # the incremental and batch preparations keeps its original meaning.
    @test p.relative_embedding[p.ts_total .+ 1, :] ≈
          oracle.relative_embedding[oracle.ts_total .+ 1, :] atol=1e-11 rtol=1e-11
    @test p.stats.ranges==oracle.stats.ranges
    @test (p.stats.full_xx === nothing)==(oracle.stats.full_xx === nothing)
    for name in (:full_xy,:full_yy)
        @test getproperty(p.stats,name) ≈ getproperty(oracle.stats,name) atol=1e-9 rtol=1e-11
    end
    if p.stats.full_xx !== nothing
        @test p.stats.full_xx ≈ oracle.stats.full_xx atol=1e-9 rtol=1e-11
    end
    for f in 1:state.F_folds
        @test p.stats.xy[f] ≈ oracle.stats.xy[f] atol=1e-9 rtol=1e-11
        @test p.stats.yy[f] ≈ oracle.stats.yy[f] atol=1e-9 rtol=1e-11
        p.stats.xx === nothing || (@test p.stats.xx[f] ≈ oracle.stats.xx[f] atol=1e-9 rtol=1e-11)
    end
    if p.macro_stats !== nothing
        index_sets=vcat([collect(1:p.n_res)],[collect(r) for r in p.stats.ranges])
        blocks=vcat([p.macro_stats.full],p.macro_stats.folds)
        for (rows,block) in zip(index_sets,blocks)
            Xm=p.B_m[p.ts_total[rows],:]; ym=p.m[p.ts_total[rows].+1]
            @test block.n==length(rows)
            @test block.xx ≈ Xm'*Xm atol=1e-9 rtol=1e-11
            @test block.xy ≈ reshape(Xm'*ym,:,1) atol=1e-9 rtol=1e-11
            @test block.yy[1,1] ≈ dot(ym,ym) atol=1e-9 rtol=1e-11
        end
    end
    p
end
function incremental_assert_model(state,prices;scenarios=false)
    initial=copy(state.warm)
    oracle=fit_v1(prices;ridge_alpha=state.ridge_alpha,F_folds=state.F_folds,alpha_initial=copy(initial))
    actual=solve_current!(state)
    @test actual.active_indices==oracle.active_indices
    @test actual.relative_observed==oracle.relative_observed
    @test actual.s1 ≈ oracle.s1 atol=1e-12 rtol=1e-12
    @test actual.s_macro ≈ oracle.s_macro atol=1e-12 rtol=1e-11
    @test actual.s_perp ≈ oracle.s_perp atol=1e-12 rtol=1e-11
    @test actual.resp.alpha_macro ≈ oracle.resp.alpha_macro atol=1e-9 rtol=1e-7
    @test actual.resp.alpha_rel ≈ oracle.resp.alpha_rel atol=1e-9 rtol=1e-7
    @test actual.mu_pred ≈ oracle.mu_pred atol=1e-10 rtol=1e-8
    @test actual.pred_moments.mu_rel ≈ oracle.pred_moments.mu_rel atol=1e-9 rtol=1e-8
    @test actual.pred_moments.var_m ≈ oracle.pred_moments.var_m atol=1e-10 rtol=1e-8
    @test actual.pred_moments.L_rel*actual.pred_moments.L_rel' ≈
          oracle.pred_moments.L_rel*oracle.pred_moments.L_rel' atol=1e-9 rtol=1e-8
    @test actual.own_res_rows==oracle.own_res_rows
    @test isfinite.(actual.res_history)==isfinite.(oracle.res_history)
    mask=isfinite.(actual.res_history)
    if any(mask)
        @test actual.res_history[mask] ≈ oracle.res_history[mask] atol=1e-10 rtol=1e-8
    end
    @test actual.d_posterior ≈ oracle.d_posterior atol=1e-9 rtol=1e-8
    @test actual.v_forecasts ≈ oracle.v_forecasts atol=1e-10 rtol=1e-8
    if scenarios
        A=generate_scenarios_v1(incremental_canonical_model(actual);S=128,rng=MersenneTwister(31))
        B=generate_scenarios_v1(incremental_canonical_model(oracle);S=128,rng=MersenneTwister(31))
        active=actual.active_indices
        @test A[:,active] ≈ B[:,active] atol=1e-9 rtol=1e-8
        wa=KTrader.scenario_weights(A,active,trues(actual.N_universe))
        wb=KTrader.scenario_weights(B,active,trues(actual.N_universe))
        @test wa ≈ wb atol=1e-6 rtol=1e-6
    end
    actual
end

@testset "Exact metric-factored prefix inference" begin
    rng=MersenneTwister(714)
    T=1030
    returns=0.007randn(rng,T,3)
    returns[280:290,:].*=18 # strong metric drift, never freeze yesterday's ruler
    prices=exp.(cumsum(returns;dims=1))
    if INCREMENTAL_TEST_PHASE in (:all,:prefix)
    @testset "Append without solve; indices, arbitrary folds and moving boundaries" begin
        for F in (2,3,7)
            state=initialize_inference(prices[1:255,1:2];ridge_alpha=1.0,F_folds=F)
            @test_throws ArgumentError solve_current!(state)
            for t in 256:302
                advance_exact!(state,view(prices,t,1:2))
                t>=257+F || continue
                p=incremental_assert_preparation(state,prices[1:t,1:2])
                @test length(state.core.rows)==t-257
                @test p.ts_total==256:t-2
                if F==3 && t in (282,285,286,302)
                    incremental_assert_model(state,prices[1:t,1:2];scenarios=t==302)
                end
            end
            @test state.counters[:fallback]==0
            @test state.counters[:fast]>0
        end
    end
    @testset "Ruler horizon enablement and exact daily recompression" begin
        state=initialize_inference(prices[1:1018,1:2];ridge_alpha=2.0,F_folds=4)
        for t in 1019:1026
            advance_exact!(state,view(prices,t,1:2))
            incremental_assert_preparation(state,prices[1:t,1:2])
        end
        incremental_assert_model(state,prices[1:1026,1:2])
    end
    @testset "EB warm starts feed the same posterior implementation" begin
        state=initialize_inference(prices[1:400,1:2];F_folds=3)
        for t in 400:402
            t==400 || advance_exact!(state,view(prices,t,1:2))
            incremental_assert_model(state,prices[1:t,1:2])
        end
        @test state.counters[:solves]==3
    end
    push!(incremental_completed_phases,:prefix)
    end
    if INCREMENTAL_TEST_PHASE in (:all,:ragged)
    @testset "Synchronous gaps stay fast; asynchronous rows stay fast and exact" begin
        sync=copy(prices[1:680,1:2]); sync[300:302,:].=NaN
        state=initialize_inference(sync[1:299,:];ridge_alpha=1.0)
        for t in 300:680
            advance_exact!(state,view(sync,t,:))
            t in (300,302,305,560,680) || continue
            p=incremental_assert_preparation(state,sync[1:t,:])
            # Same-mask production prep must be nonzero end to end: an aliased
            # rwrap buffer (the audited all-zero-Gram defect) goes red here.
            @test norm(p.stats.full_xy) > 0
            @test norm(p.stats.full_yy) > 0
            p.stats.full_xx === nothing || @test norm(p.stats.full_xx) > 0
        end
        @test !state.core.async_seen
        @test state.counters[:fallback]==0
        incremental_assert_model(state,sync;scenarios=true)
        # A partially observed row no longer disables the fast path: the
        # regime engine factors the ragged field exactly (e_t(d)=R_t diag(z_t) d).
        ragged=copy(prices[1:680,1:2]); ragged[300:302,2].=NaN
        state=initialize_inference(ragged[1:299,:];ridge_alpha=1.0,F_folds=4)
        for t in 300:680
            advance_exact!(state,view(ragged,t,:))
        end
        @test state.core.async_seen  # diagnostic only
        @test state.counters[:async_rows]>0
        incremental_assert_preparation(state,ragged)
        @test state.last_reason===nothing
        @test state.counters[:fallback]==0
        @test state.counters[:fast]>0
        incremental_assert_model(state,ragged;scenarios=true)
    end
    @testset "IPO embeds coordinates without replay; insertion [1,3]->[1,2,3]" begin
        # Asset 3 activates mid-history: active grows [1,2] -> [1,2,3] (append).
        ipo=copy(prices[1:680,:]); ipo[1:360,3].=NaN
        state=initialize_inference(ipo[1:300,:];ridge_alpha=1.0)
        rebuilds=state.counters[:rebuilds]
        for t in 301:680
            advance_exact!(state,view(ipo,t,:))
            if t in (360,361,362,365,620,680)
                incremental_assert_preparation(state,ipo[1:t,:])
            end
            if t in (360,362,680)
                incremental_assert_model(state,ipo[1:t,:])
            end
        end
        @test state.counters[:rebuilds]==rebuilds  # no replay ever
        @test state.counters[:embeds]==1
        @test state.active==[1,2,3]
        # Diagnostic: with exact IPO coordinate embedding (I3), activation
        # happens at the first valid return day and the new asset's history
        # enters as zero contributions, so the core never sees a partially
        # observed row around activation. async_seen stays reserved for
        # genuinely partial rows of already-active assets (ragged testsets).
        @test !state.core.async_seen
        @test state.counters[:fallback]==0
        @test state.counters[:fast]>0
        inactive=hcat(prices[1:310,1],fill(NaN,310),prices[1:310,2])
        s=initialize_inference(inactive;ridge_alpha=1.0)
        @test s.active==[1,3]
        incremental_assert_model(s,inactive)
        # Insertion in the MIDDLE of the coordinate order: asset 2 activates
        # after asset 3, so active goes [1,3] -> [1,2,3] and asset 3 shifts.
        insert=copy(prices[1:680,:]); insert[1:400,2].=NaN
        state=initialize_inference(insert[1:300,:];ridge_alpha=1.0)
        for t in 301:680
            advance_exact!(state,view(insert,t,:))
        end
        @test state.active==[1,2,3]
        @test state.counters[:embeds]==1
        @test state.counters[:rebuilds]==0
        incremental_assert_preparation(state,insert)
        incremental_assert_model(state,insert)
    end
    @testset "Extreme ragged stays fast, exact, and merged per unique mask" begin
        ragged=copy(prices[1:600,1:2]); ragged[3:5:end,2].=NaN
        state=initialize_inference(ragged;ridge_alpha=1.0)
        incremental_assert_preparation(state,ragged)
        @test state.last_reason===nothing
        @test state.counters[:fallback]==0
        @test state.counters[:fast]>0
        # Alternating availability must merge into few unique masks, not one
        # gram per transition.
        @test state.core.async_seen
        incremental_assert_model(state,ragged)
    end
    @testset "Multi-transition windows across bands; zero and missing rows mixed" begin
        # Two assets with staggered activations, a synchronous all-NaN gap and
        # a per-asset dropout: decision days right after each transition land
        # inside the 256-row boundary window of the deepest band (2*128).
        panel=copy(prices[1:700,1:3])
        panel[1:320,2].=NaN      # asset 2 activates at row 321
        panel[1:430,3].=NaN      # asset 3 activates at row 431
        panel[500:502,:].=NaN    # synchronous gap (all-missing run)
        panel[560,1]=NaN         # single-cell dropout for asset 1
        state=initialize_inference(panel[1:340,:];ridge_alpha=1.0,F_folds=3)
        for t in 341:700
            advance_exact!(state,view(panel,t,:))
        end
        @test state.counters[:embeds]==2   # assets 2 and 3 activated
        @test state.counters[:rebuilds]==0
        incremental_assert_preparation(state,panel)
        incremental_assert_model(state,panel)
        diag=inference_diagnostics(state)
        @test diag.runs>=5  # [..1], [12], [123], gap run, [123], dropout splits
        @test diag.materialized_rows>0  # boundary rows exercised the exact path
    end
    push!(incremental_completed_phases,:ragged)
    end
    if INCREMENTAL_TEST_PHASE in (:all,:ownership)
    @testset "Resource budget routes to the exact batch oracle" begin
        ragged=copy(prices[1:600,1:2]); ragged[3:5:end,2].=NaN
        state=initialize_inference(ragged;ridge_alpha=1.0,materialize_row_limit=4)
        incremental_assert_preparation(state,ragged)
        @test state.last_reason==:resource_budget
        @test state.counters[:fallback]>0
        incremental_assert_model(state,ragged)
    end
    @testset "Numerical failure falls back; external macro Grams stay immutable" begin
        state=initialize_inference(prices[1:310,1:2];ridge_alpha=1.0)
        p=KTrader._prepare_current!(state)
        before=deepcopy(p.macro_stats)
        KTrader._fit_prepared_v1(p;ridge_alpha=1.0,F_folds=3)
        @test p.macro_stats.full.xx==before.full.xx
        @test all(p.macro_stats.folds[f].xx==before.folds[f].xx for f in 1:3)
        # Corrupt one aggregated gram entry: the contraction must detect the
        # non-finite metric and dispatch to the exact batch oracle.
        key1=first(keys(state.core.grams))
        state.core.grams[key1].V[1][1,1]=NaN
        incremental_assert_preparation(state,prices[1:310,1:2])
        @test state.last_reason==:nonfinite_gram
        incremental_assert_model(state,prices[1:310,1:2])
    end
    @testset "Checkpoint isolation, input ownership, no future leakage, stable old models" begin
        allprices=copy(prices[1:310,1:2])
        cache=KTrader.PriceHistoryCache(allprices)
        stats=build_prefix_ruler_stats(cache.log_prices,cache.first_price)
        prefix=copy(allprices[1:300,:])
        root=initialize_inference(prefix;ridge_alpha=1.0,history_cache=cache,ruler_stats=stats)
        prefix[1,1]*=2
        @test root.bars[1][1]==allprices[1,1]
        left=inference_checkpoint(root); right=inference_checkpoint(root)
        @test left.workspace !== right.workspace
        @test left.core.rows !== right.core.rows
        @test left.core.grams !== right.core.grams
        @test left.history_cache === right.history_cache
        saved=solve_current!(root)
        frozen=deepcopy(saved)
        bar=copy(allprices[301,:]); advance_exact!(left,bar); bar[1]*=3
        changed=copy(allprices[301,:]); changed[2]*=1.3
        advance_exact!(right,changed)
        @test length(root.bars)==300
        @test left.bars[end][1]==allprices[301,1]
        incremental_assert_model(left,allprices[1:301,:])
        right_prices=copy(allprices[1:301,:]); right_prices[end,:].=changed
        incremental_assert_model(right,right_prices)
        for t in 301:305
            advance_exact!(root,view(allprices,t,:))
        end
        solve_current!(root)
        @test saved.mu_pred==frozen.mu_pred
        @test saved.res_history==frozen.res_history
        @test saved.resp.G_c_mean==frozen.resp.G_c_mean
        other=copy(allprices); other[301:end,:].*=17
        future=KTrader.PriceHistoryCache(other)
        fs=build_prefix_ruler_stats(future.log_prices,future.first_price)
        a=initialize_inference(allprices[1:300,:];ridge_alpha=1.0,history_cache=cache,ruler_stats=stats)
        b=initialize_inference(other[1:300,:];ridge_alpha=1.0,history_cache=future,ruler_stats=fs)
        @test solve_current!(a).mu_pred ≈ solve_current!(b).mu_pred atol=1e-12
        @test_throws DimensionMismatch advance_exact!(a,[1.0])
        @test_throws ArgumentError advance_exact!(a,[0.0,1.0])
        @test length(a.bars)==300
        single=initialize_inference(allprices[1:300,1:1];ridge_alpha=1.0)
        model=incremental_assert_model(single,allprices[1:300,1:1])
        @test all(iszero,model.pred_moments.L_rel)
    end
    push!(incremental_completed_phases,:ownership)
    end
    if INCREMENTAL_TEST_PHASE in (:all,:kernels)
    @testset "Triangular filter kernel: audit fixture and independent oracle" begin
        # The audited counterexample structure: tau=2, k=5, runs [1,3] mask A
        # and [4,5] mask B. Asset 2 prices finite from ROW 3 ONWARDS, so its
        # first finite return is r[4]=logs[5]-logs[4] (row 4, run B) while
        # r[3]=logs[4]-logs[3] stays NaN because price row 3 is NaN. Correct
        # Q contribution of the OLD run is exactly zero (no ancient
        # constants); current run is -z5/2; P splits as z3/2 (A) + z4 +
        # z5/2 (B). The old zcum=zsum-zbase formula would give the old run
        # z4+z5 and turn this red. (NaN must cover rows 1:2 only: three NaN
        # rows would push the first finite return to row 5 and leave a
        # single run, making runs[2] a BoundsError.)
        audit=exp.(cumsum(0.01 .* randn(rng,5,2);dims=1)); audit[1:3,2].=NaN
        state=initialize_inference(audit)
        core=state.core
        @test core.k==4  # 5 price rows -> 4 return rows
        @test length(core.runs)==2
        @test core.runs[1].last==3 && core.runs[2].first==4
        # band 1 (tau=2) needs k >= 2*tau+1 = 5: nothing active at k=4.
        @test KTrader.compute_pending(core,4)===nothing
        # Grow one more price row so k=5 activates band 1 with the audited
        # window structure (run B becomes [4,5]).
        sixth=exp.(log.(audit[end,:]) .+ 0.01 .* randn(rng,2))
        audit=vcat(audit,reshape(sixth,1,2))
        advance_exact!(state,sixth)
        core=state.core
        @test core.k==5
        @test core.runs[2].first==4 && core.runs[2].last==5
        z=ifelse.(isfinite.(diff(log.(audit);dims=1)),diff(log.(audit);dims=1),0.0)
        z3,z4,z5=view(z,3,:),view(z,4,:),view(z,5,:)
        pending=KTrader.compute_pending(core,5)
        @test pending!==nothing
        @test length(pending.regs)==2
        slotA=slotB=nothing
        for (i,mid) in enumerate(pending.regs)
            m=core.mask_of[mid]
            (m==[true,false]) && (slotA=pending.v[i])
            (m==[true,true]) && (slotB=pending.v[i])
        end
        @test slotA!==nothing && slotB!==nothing
        # band 1 columns: Q = col 1, P = col 2 (tau=2 = BANDS[1]).
        @test slotA[:,1] ≈ zeros(2) atol=1e-14                 # old run: Q exactly zero
        @test slotB[:,1] ≈ -z5 ./ 2 atol=1e-14                 # current run: -z5/2
        @test slotA[:,2] ≈ z3 ./ 2 atol=1e-14                  # P: A gets z3/2
        @test slotB[:,2] ≈ z4 .+ z5 ./ 2 atol=1e-14            # P: B gets z4 + z5/2
        # Independent kernel oracle on a randomized ragged panel: recompute
        # every (mask, band) prototype from the raw-return kernel formula
        # with an independently written per-row run lookup.
        panel=exp.(cumsum(0.008 .* randn(rng,600,3);dims=1))
        panel[1:200,2].=NaN; panel[280:300,1].=NaN; panel[350:352,:].=NaN
        panel[420,3]=NaN; panel[1:500,3].=ifelse.(1:500 .>= 430,panel[1:500,3],NaN)
        st2=initialize_inference(panel)
        for k in (260,320,410,460,560)
            pending=KTrader.compute_pending(st2.core,k)
            oracle=Dict{Int,Tuple{Matrix{Float64},Matrix{Float64}}}()
            N=3
            for (b,tau) in enumerate(KTrader.BANDS)
                k >= 2tau+1 || continue
                for j=0:tau-1
                    u=k-j; rho=nothing
                    for r in st2.core.runs
                        r.first<=u<=r.last && (rho=r)
                    end
                    rho===nothing && continue
                    zu=st2.core.zsum_hist[u+1] .- st2.core.zsum_hist[u]
                    slot=get!(oracle,rho.maskid) do
                        (zeros(N,length(KTrader.BANDS)),zeros(N,length(KTrader.BANDS)))
                    end
                    slot[1][:,b] .+= (-(tau-1-j)/tau) .* zu
                end
                for j=0:2tau-2
                    u=k-j; rho=nothing
                    for r in st2.core.runs
                        r.first<=u<=r.last && (rho=r)
                    end
                    rho===nothing && continue
                    zu=st2.core.zsum_hist[u+1] .- st2.core.zsum_hist[u]
                    slot=get!(oracle,rho.maskid) do
                        (zeros(N,length(KTrader.BANDS)),zeros(N,length(KTrader.BANDS)))
                    end
                    slot[2][:,b] .+= (min(j+1,2tau-1-j)/tau) .* zu
                end
            end
            @test Set(pending.regs)==Set(collect(keys(oracle)))
            for (i,mid) in enumerate(pending.regs)
                q,p=oracle[mid]
                @test pending.v[i][:,1:2:end] ≈ q atol=1e-12
                @test pending.v[i][:,2:2:end] ≈ p atol=1e-12
            end
        end
    end
    @testset "rwrap contractions match the explicit matmul oracle" begin
        # Independent oracle: R is built EXPLICITLY as Diagonal(mask) minus
        # mm'/c and compared against plain matmul R*M*R'. rwrap_sym! is only
        # defined for SYMMETRIC M (its shared row-sum shortcut equals the
        # two-sided correction only when work1==work2); feeding it a general
        # matrix is exactly the audited production defect, so the symmetric
        # branch gets a symmetric Msym. rwrap_gen! takes general M.
        rng2=MersenneTwister(99)
        for trial in 1:8
            N=5
            mask=rand(rng2,Bool,N)
            count(mask)==0 && continue
            c=count(mask)
            R=Matrix{Float64}(Diagonal(mask)) .- (mask .* mask') ./ c
            Msym=Matrix(Symmetric(randn(rng2,N,N)))
            dst=zeros(N,N); w1=zeros(N); w2=zeros(N)
            KTrader.rwrap_sym!(dst,Msym,mask,w1)
            @test dst ≈ R*Msym*R' atol=1e-12
            Mgen=randn(rng2,N,N)
            KTrader.rwrap_gen!(dst,Mgen,mask,w1,w2)
            @test dst ≈ R*Mgen*R' atol=1e-12
            # A symmetric input must also agree between the two contractors
            # (algebraic identity: work1==work2 makes the formulas equal).
            KTrader.rwrap_gen!(dst,Msym,mask,w1,w2)
            @test dst ≈ R*Msym*R' atol=1e-12
            # dst must be a genuinely separate buffer: the production caller
            # passes distinct src/dst; zeroing dst on entry is part of the
            # contract and must never clobber the source.
            before=copy(Msym)
            KTrader.rwrap_sym!(dst,Msym,mask,w1)
            @test Msym==before
        end
    end
    end # kernel phase
end # original outer testset

if INCREMENTAL_TEST_PHASE in (:all,:kernels)
@testset "non-symmetric cross-Gram contraction (production stats path)" begin
    # 确定性反例：N=2、全真 mask、gram 的 V cross 块 M=[1 2; 3 4]。
    # 去心投影 R 下 R·M·R'=0；对称收缩器会给出 [0.5 -0.5; 0.5 -0.5]
    # （mutation 实测 0.5 与解析一致）。regime_statistics 的 V 块 c<c2
    # 必须走 rwrap_gen!（c==c2 对称块保留 rwrap_sym!，数学等价）。
    N=2; C=KTrader.REGIME_CHANNELS; P=C*N
    core=KTrader.RawInferenceCore(collect(1:N),1)
    core.maskids[[true,true]]=1
    push!(core.mask_of,[true,true])
    for _ in 1:P
        push!(core.rows,KTrader.RowRecord([1],[zeros(N,C)],zeros(N),1,1,true))
        push!(core.membership,1)
    end
    M=[1.0 2.0;3.0 4.0]
    V=[zeros(N,N) for _ in 1:KTrader.REGIME_PAIR_COUNT]
    V[KTrader.gram_pair_index(1,2)] .= M
    core.grams[(1,1)]=KTrader.FoldGram(P,V,[zeros(N,N) for _ in 1:C],zeros(N,N))
    out=KTrader.regime_statistics(core,ones(N),ones(length(KTrader.BANDS)),
                                  zeros(P+1,1),zeros(P+1),collect(1:P),1)
    xx=out.stats.xx[1]
    @test maximum(abs.(xx[1:2,3:4]))==0.0     # cross 块 (c=1,c2=2)
    @test maximum(abs.(xx[3:4,1:2]))==0.0     # 对称镜像
    @test maximum(abs.(xx[1:2,1:2]))==0.0     # 对角块（V 对角为零）
end
@testset "Incremental phase admission cannot silently select a partial suite" begin
    @test incremental_test_phase(String[];standalone=true)==:all
    @test incremental_test_phase(String[];standalone=false)==:all
    for phase in (:all,INCREMENTAL_TEST_PHASES...)
        @test incremental_test_phase(["--phase=$phase"];standalone=true)==phase
        @test_throws ArgumentError incremental_test_phase(["--phase=$phase"];standalone=false)
    end
    for args in (["--phase=typo"],["--phase="],["--phase=kernels","--phase=prefix"],["kernels"])
        @test_throws ArgumentError incremental_test_phase(args;standalone=true)
    end
end
push!(incremental_completed_phases,:kernels)
end
@testset "Incremental selected phases completed" begin
    expected=INCREMENTAL_TEST_PHASE==:all ? collect(INCREMENTAL_TEST_PHASES) : [INCREMENTAL_TEST_PHASE]
    @test incremental_completed_phases==expected
end
println("[incremental phase=",INCREMENTAL_TEST_PHASE,"] END completed=",incremental_completed_phases); flush(stdout)
