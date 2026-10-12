#!/usr/bin/env julia
# A strictly ONE-DAY Gate-0 decision preflight on unmodified, real, full-size
# market data. It never enters run_gate0_backtest, books an account, or uses a
# reduced universe. Historical prequential forecasts are computed in bounded
# independent slices and held in an atomic, code/data-bound local checkpoint.
# Do NOT use this file to launch a two-year account backtest.
using Dates, LinearAlgebra, Serialization, Printf, Random, Profile

BLAS.set_num_threads(parse(Int,get(ENV,"PROBE_BLAS_THREADS","4")))
include(joinpath(@__DIR__,"../src/gate0/KTraderGate0.jl"))
using .KTraderGate0
Base.include(KTraderGate0,joinpath(@__DIR__,"../src/gate0/market_io.jl"))
const G=KTraderGate0
G.FFTW.set_num_threads(parse(Int,get(ENV,"PROBE_FFT_THREADS","1")))

date=Date(get(ENV,"PROBE_DATE","2024-10-01"))
mf=G.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el=G.Eligibility(mf)
t=findfirst(==(date),mf.dates)
t === nothing && error("probe date is absent: $date")
t >= G.WARMUP+3 || error("insufficient real history for preflight")
held=zeros(size(mf.observed,2))
act=findall(view(el.model_admitted,t,:))
F=G.free(el,t)
free_act=Int[k for (k,j) in enumerate(act) if F[j]]
locked_act=Int[]
row_ids=Int[u for u in (G.WARMUP+1):t
            if any(mf.observed[u,j] && mf.observed[u-1,j] for j in act)]
row_masks=BitVector[BitVector([mf.observed[u,j] && mf.observed[u-1,j]
                               for j in act]) for u in row_ids]
R, dropped=G.resolve_risk_domain(row_ids,row_masks,free_act,locked_act,t)
isempty(R) && error("preflight cannot prepare an empty risk domain")
floor,peeled=G._forecast_risk_floor(row_ids,row_masks,free_act,locked_act,
                                    R,t,el,act)
required=act[floor]
origins=Int[u-1 for u in row_ids if
    all(j -> mf.observed[u-1,j] && mf.observed[u,j] &&
             el.model_admitted[u-1,j],required)]
sig=G._backtest_signature(mf,el,(:single_day_gate_probe,date,1e-6,2048,5.0))
path=abspath(get(ENV,"PROBE_CHECKPOINT","/tmp/ktrader-two-year-20261011/single-day-probe.jls"))
max_add=parse(Int,get(ENV,"PROBE_MAX_NEW_FORECASTS","40"))
1 <= max_add <= 48 || error("bounded probe permits at most 48 historical fits per invocation")
workers=parse(Int,get(ENV,"PROBE_WORKERS","1"))
1 <= workers <= Threads.nthreads() || error("probe workers require matching Julia threads")

cache=G._ForecastCache(mf,el,(1e-6,2048,5.0))
cache.selective=true
diagnostic_reuse=false
if isfile(path)
    saved=deserialize(path)
    timing_opt_in=(get(ENV,"PROBE_TIMING_REUSE","false")=="true" &&
                   (get(ENV,"PROBE_COMPONENTS_ONLY","false")=="true" ||
                    get(ENV,"PROBE_DIAGNOSTIC_DECISION","false")=="true"))
    older_fixture=(saved.date < date && date-saved.date<=Day(14))
    saved.version == 1 && (saved.date == date ||
                           (timing_opt_in && older_fixture)) ||
        error("preflight checkpoint has incompatible date or schema")
    if saved.signature != sig || saved.date != date
        timing_opt_in ||
            error("preflight checkpoint does not match current code/data")
        # The preserved 395 historical-mean forecasts were computed before
        # changes to Kelly and the QMC draw cache. They are used here ONLY
        # for bounded scenario-cost diagnosis. Never accept this evidence as
        # a production-account or two-year certificate, and never re-sign it.
        diagnostic_reuse=true
        println("TIMING_ONLY_REUSE old signature; output NOT a production certificate")
        flush(stdout)
    end
    merge!(cache.entries,saved.forecasts)
end
save()=diagnostic_reuse ? error("must not re-sign legacy forecasts") :
       G._write_backtest_checkpoint(path,(;
            version=1,signature=sig,date,forecasts=cache.entries))
if diagnostic_reuse
    verified=[o for o in origins if haskey(cache.entries,o)]
    length(verified)>=390 && length(origins)-length(verified)<=8 ||
        error("timing-only historical fixture permits at most eight new origins")
    for k in unique([1,cld(length(verified),3),cld(2length(verified),3),length(verified)])
        o=verified[k]
        ref=cache.entries[o]
        fresh=G._historical_forecast(mf,el,o;
                  tol=1e-6,max_cells=2048,u_span=5.0,preprocessed=cache)
        ref !== nothing && fresh !== nothing && ref.active == fresh.active &&
            isapprox(ref.mean_log_return,fresh.mean_log_return;
                     atol=1e-10,rtol=1e-8) ||
            error("historical forecast spot validation failed at origin=$o")
    end
    println("TIMING_ONLY_SPOT_CHECK_OK checked=4 histories")
    flush(stdout)
end

println("ONE_DAY_PREFLIGHT_ONLY date=$date risk=$(length(R)) " *
        "structural_floor=$(length(floor)) peeled=$peeled total_needed=$(length(origins)) " *
        "already_saved=$(count(o -> haskey(cache.entries,o),origins))")
flush(stdout)
pending=[o for o in origins if !haskey(cache.entries,o)]
diagnostic_reuse && length(pending)>=max_add &&
    error("diagnostic fixture cannot truncate a day's required history")
added=0
start=time_ns()
for batch_start in 1:workers:min(length(pending),max_add)
    last=min(batch_start+workers-1,min(length(pending),max_add))
    batch=pending[batch_start:last]
    compute(origin)=G._historical_forecast(mf,el,origin;
                    tol=1e-6,max_cells=2048,u_span=5.0,preprocessed=cache)
    forecasts=length(batch)==1 ? [compute(batch[1])] :
              fetch.([Threads.@spawn compute(o) for o in batch])
    for (origin,forecast) in zip(batch,forecasts)
        cache.entries[origin]=forecast
        global added+=1
        if added%8==0 && !diagnostic_reuse
            save()
            println("PREP_PROGRESS saved=$(count(o -> haskey(cache.entries,o),origins))/$(length(origins)) " *
                    "last_origin=$(mf.dates[origin]) elapsed=$((time_ns()-start)/1e9)")
            flush(stdout)
        end
    end
end
if added>=max_add
    save()
    println("PREFLIGHT_PAUSED checkpoint=$path completed_day=false " *
            "saved=$(count(o -> haskey(cache.entries,o),origins))/$(length(origins))")
    flush(stdout)
    exit(75)
end
diagnostic_reuse || save()
println("PREPARED_REAL_JOINT_HISTORY date=$date count=$(length(origins)) " *
        "new_fits=$added elapsed=$((time_ns()-start)/1e9)")
flush(stdout)

if get(ENV,"PROBE_COMPONENTS_ONLY","false")=="true"
    # Execute the SAME production response, residual, d quadrature and
    # predictive source functions as the driver, stopping deliberately before
    # invoking cash_kelly. This prevents conic cold JIT from corrupting the
    # timing evidence. It is NOT a successful decision or an account run.
    t0=time_ns()
    mf_t=G.MarketFacts(mf.dates[1:t],mf.symbols,
                        mf.close[1:t,:],mf.adj[1:t,:],mf.observed[1:t,:])
    el_t=G.Eligibility(el.model_admitted[1:t,:],el.trade_eligible[1:t,:],
                       el.executable[1:t,:])
    prep=G._response_prefix(mf_t,el_t,t)
    println("COMPONENT_PREP_OK secs=$((time_ns()-t0)/1e9)")
    flush(stdout)
    rows=Int[s for s in G.WARMUP:(t-1) if any(view(prep.obs,s+1,:))]
    row_ids_today=Int[s+1 for s in rows]
    masks=BitVector[BitVector(view(prep.obs,u,:)) for u in row_ids_today]
    t0=time_ns()
    post=G.fit_full_posterior(prep.mp.X[rows,:],prep.mp.Y[rows,:];
                              tol=1e-6,max_cells=2048,u_span=5.0)
    println("COMPONENT_POST_OK secs=$((time_ns()-t0)/1e9) " *
            "nodes=$(length(post.alpha_weights))")
    flush(stdout)
    if get(ENV,"PROBE_WEIGHTS_ONLY","false")=="true"
        weights=post.alpha_weights
        cdf=G._mu_node_cdf(weights)
        seed=G._DRIVER_DEFAULT_SEED
        node_pts=G.rule_points(G.SobolOwenRule(1,G._mu_node_rule_seed(seed)),32768)
        keys=Int[min(searchsortedfirst(cdf,node_pts[s,1]),length(cdf)) for s in 1:32768]
        counts=zeros(Int,length(cdf))
        for k in keys
            counts[k]+=1
        end
        block_unique=[length(unique(keys[s:min(s+63,length(keys))])) for s in 1:64:32768]
        @printf("REAL_NODE_DISTRIBUTION support=%d positive=%d effective=%.2f max_weight=%.5f unique_32768=%d mean_unique_64=%.2f median_unique_64=%.2f\n",
            length(cdf),count(>(0.0),weights),1/sum(abs2,weights),
            maximum(weights),count(>(0),counts),
            sum(block_unique)/length(block_unique),sort(block_unique)[cld(length(block_unique),2)])
        flush(stdout)
        exit(0)
    end
    t0=time_ns()
    resid_all=G._market_prequential_residuals(mf_t,el_t,row_ids_today,
                 prep.act,prep.s1; tol=1e-6,max_cells=2048,u_span=5.0,
                 focus_risk=required,forecast_cache=cache)
    keep=Int[i for i in 1:size(resid_all,1) if any(isfinite,view(resid_all,i,:))]
    resid=resid_all[keep,:]
    ids_ok=row_ids_today[keep]
    masks_ok=[BitVector(masks[i].&isfinite.(view(resid_all,i,:))) for i in keep]
    R2,dropped2=G.resolve_risk_domain(ids_ok,masks_ok,free_act,locked_act,t;
                                      residual_rows=resid)
    R2 == floor || error("preflight rank chose a different domain: $R2 vs $floor")
    println("COMPONENT_JOINT_OK secs=$((time_ns()-t0)/1e9) " *
            "joint_rows=$(length(G.joint_row_indices(ids_ok,masks_ok,R2,t))) " *
            "domain=$(length(R2))")
    flush(stdout)
    t0=time_ns()
    st=G.innovation_state(resid,ids_ok,masks_ok,R2,prep.E;
                          t=t,require_full_rank=true)
    println("COMPONENT_INNOVATION_OK secs=$((time_ns()-t0)/1e9) " *
            "d_nodes=$(length(st.d_nodes)) joint_rows=$(length(st.J_t))")
    flush(stdout)
    if get(ENV,"PROBE_EXPORT_REAL_INNOVATION","false")=="true"
        dest=get(ENV,"PROBE_INNOVATION_PATH",
             "/tmp/ktrader-two-year-20261011/real-one-day-innovation-input.jls")
        G._write_backtest_checkpoint(dest,(;
             version=1,date,eps_R=copy(st.eps_R),
             u_J=copy(st.z_pool.u_J),target_t=t,
             d_nodes=copy(st.d_nodes),d_weights=copy(st.d_weights),
             timing_only=diagnostic_reuse))
        println("INNOVATION_INPUT_SAVED row_count=$(size(st.eps_R,1)) " *
                "risk=$(size(st.eps_R,2)) dest=$dest")
        flush(stdout)
    end
    for repetition in 1:parse(Int,get(ENV,"PROBE_REPEAT_INNOVATION","0"))
        if get(ENV,"PROBE_GC_BEFORE_INNOVATION","false")=="true"
            local tg=time_ns()
            GC.gc(true)
            @printf("COMPONENT_GC_BEFORE_INNOVATION repetition=%d secs=%.5f\n",
                    repetition,(time_ns()-tg)/1e9)
            flush(stdout)
        end
        local ts=time_ns()
        timed=@timed G.innovation_state(resid,ids_ok,masks_ok,R2,prep.E;
                                        t=t,require_full_rank=true)
        st2=timed.value
        @assert st.d_nodes == st2.d_nodes
        @assert isapprox(st.d_weights,st2.d_weights;rtol=1e-10,atol=1e-12)
        @printf("COMPONENT_INNOVATION_STEADY repetition=%d secs=%.5f gctime=%.5f alloc_mb=%.1f\n",
                repetition,(time_ns()-ts)/1e9,timed.gctime,timed.bytes/(1024^2))
        flush(stdout)
    end
    if get(ENV,"PROBE_INNOV_ONLY","false")=="true"
        println("BOUNDED_INNOVATION_PROFILE_ONLY; no scenarios, Kelly or account")
        exit(0)
    end
    xnow=vec(prep.mp.X_full[t,:])
    mu_qmc=get(ENV,"PROBE_MU_QMC","false")=="true"
    chisq_qmc=get(ENV,"PROBE_CHISQ_QMC","false")=="true"
    prefetch_factors=get(ENV,"PROBE_PREFETCH_FACTORS","true")=="true"
    src=G.predictive_rqmc_source(post,st,xnow;
                                 s1=prep.s1,E_active=prep.E,
                                 mu_qmc=mu_qmc,mu_chisq_qmc=chisq_qmc,
                                 prefetch_mu_factors=prefetch_factors)
    rule=G.SobolOwenRule(2,G._DRIVER_DEFAULT_SEED)
    if get(ENV,"PROBE_PROFILE_SCENARIO","false")=="true"
        M=parse(Int,get(ENV,"PROBE_SCENARIO_TOP","16384"))
        4096<=M<=32768 && ispow2(M) || error("profile is bounded to 4k..32k")
        src.gen(1024,rule,MersenneTwister(G._default_mu_seed(rule.seed)))
        Profile.init(n=2_000_000,delay=0.002)
        Profile.clear()
        local tprofile=time_ns()
        Profile.@profile begin
            X=src.gen(M,rule,MersenneTwister(G._default_mu_seed(rule.seed)))
            @assert size(X)==(M,st.N_R)
        end
        println("PROFILE_SCENARIO_DONE M=$M secs=$((time_ns()-tprofile)/1e9)")
        flush(stdout)
        Profile.print(format=:flat,sortedby=:count)
        exit(0)
    end
    if get(ENV,"PROBE_CAPTURE_8192","false")=="true"
        mu_qmc && chisq_qmc || error("8192 capture requires the fully QMC diagnostic")
        top=parse(Int,get(ENV,"PROBE_CAPTURE_TOP","8192"))
        8192 <= top <= 65536 && ispow2(top) ||
            error("bounded stage capture top must be 8192,16384,32768")
        low=top÷2
        part=Symbol(get(ENV,"PROBE_CAPTURE_PART","opt"))
        part in (:opt,:audit) || error("capture part must be opt or audit")
        dest=abspath(get(ENV,"PROBE_CAPTURE_PATH",
                       "/tmp/ktrader-two-year-20261011/one-day-qmc-8192.jls"))
        signature=G._backtest_signature(mf,el,(:qmc8192_real_one_day,
                                               date,1e-6,2048,5.0,
                                               G._DRIVER_DEFAULT_SEED,top))
        ts=time_ns()
        if part===:opt
            X4096=src.gen(low,rule,MersenneTwister(G._default_mu_seed(rule.seed)))
            X8192=src.gen(top,rule,MersenneTwister(G._default_mu_seed(rule.seed)))
            isequal(X4096,X8192[1:low,:]) ||
                error("optimisation scenarios are not nested")
            @printf("CAPTURE_OPT_COMPUTE full_assets=%d S=%d secs=%.5f\n",
                    size(X8192,2),top,(time_ns()-ts)/1e9)
            flush(stdout)
            audit8192=nothing
            if get(ENV,"PROBE_CAPTURE_BOTH","false")=="true"
                # Same source as production: the independent audit rule gets
                # a distinct QMC seed while sharing only exact factor caches.
                ta=time_ns()
                audit_rng=MersenneTwister(G._default_mu_seed(rule.seed) ⊻
                                         UInt64(0x9E3779B97F4A7C15))
                audit_rule=G.SobolOwenRule(2,G.audit_seed(rule.seed))
                audit8192=src.gen(top,audit_rule,audit_rng)
                @printf("CAPTURE_SHARED_AUDIT_COMPUTE full_assets=%d S=%d secs=%.5f\n",
                        size(audit8192,2),top,(time_ns()-ta)/1e9)
                flush(stdout)
            end
            G._write_backtest_checkpoint(dest,(;version=2,signature,date,
                 count=length(origins),timing_only=diagnostic_reuse,
                 d_nodes=copy(st.d_nodes),d_weights=copy(st.d_weights),
                 Mlow=low,Mhigh=top,X4096,X8192,audit8192))
            @printf("CAPTURE_OPT_SCENARIOS_OK date=%s full_assets=%d S=%d secs=%.5f path=%s\n",
                    date,size(X8192,2),top,(time_ns()-ts)/1e9,dest)
        else
            isfile(dest) || error("audit capture needs the complete optimisation capture")
            previous=deserialize(dest)
            (previous.version==2 && previous.signature==signature &&
             previous.date==date && previous.audit8192===nothing &&
             previous.Mlow==low && previous.Mhigh==top &&
             length(previous.d_nodes)==length(st.d_nodes) &&
             previous.d_nodes==st.d_nodes &&
             isapprox(previous.d_weights,st.d_weights;atol=1e-12,rtol=1e-10)) ||
                error("audit capture does not match current optimisation scenarios")
            audit_rng=MersenneTwister(G._default_mu_seed(rule.seed) ⊻
                                     UInt64(0x9E3779B97F4A7C15))
            audit_rule=G.SobolOwenRule(2,G.audit_seed(rule.seed))
            audit8192=src.gen(top,audit_rule,audit_rng)
            X4096,X8192=previous.X4096,previous.X8192
            G._write_backtest_checkpoint(dest,(;version=2,signature,date,
                count=length(origins),timing_only=diagnostic_reuse,
                d_nodes=copy(st.d_nodes),d_weights=copy(st.d_weights),
                Mlow=low,Mhigh=top,X4096,X8192,audit8192))
            @printf("CAPTURE_AUDIT_SCENARIOS_OK date=%s full_assets=%d S=%d secs=%.5f path=%s\n",
                    date,size(audit8192,2),top,(time_ns()-ts)/1e9,dest)
        end
        flush(stdout)
        exit(0)
    end
    measured=Dict{Int,Matrix{Float64}}()
    top=parse(Int,get(ENV,"PROBE_SCENARIO_TOP","512"))
    512 <= top <= 32768 && ispow2(top) ||
        error("scenario probe cap must be a power of two in 512..32768")
    bottom=parse(Int,get(ENV,"PROBE_SCENARIO_MIN","64"))
    bottom >= 64 && bottom <= top && ispow2(bottom) ||
        error("scenario probe minimum must be a power of two between 64 and top")
    layers=Int[2^k for k in trailing_zeros(bottom):trailing_zeros(top)]
    println("SCENARIO_BACKEND mu_qmc=$mu_qmc chisq_qmc=$chisq_qmc max=$top")
    flush(stdout)
    for ns in layers
        t0=time_ns()
        X=src.gen(ns,rule,MersenneTwister(G._default_mu_seed(rule.seed)))
        @assert size(X)==(ns,st.N_R) && all(v->isfinite(v)&&v>0,X)
        if get(ENV,"PROBE_IPM_ONLY","false")=="true"
            measured[ns]=X
        end
        @printf("COMPONENT_SCENARIOS_OK S=%d secs=%.5f\n",
                ns,(time_ns()-t0)/1e9)
        flush(stdout)
    end
    if get(ENV,"PROBE_IPM_ONLY","false")=="true"
        previous=nothing
        previous_m=0
        for m in layers
            local ts=time_ns()
            X=measured[m]
            r=G._cash_kelly_solve(X,zeros(m),ones(m),1.0,1e-6;
                                  backend=:ipm)
            if !r.status_ok || r.w_val === nothing
                println("IPM_NOT_CERTIFIED m=$m reason=$(r.reason) " *
                        "seconds=$((time_ns()-ts)/1e9)")
                flush(stdout)
                global previous=nothing
                continue
            end
            w=r.w_val
            c=G.cash_kelly_certificate(X,collect(w[1:end-1]),w[end];
                                       base=zeros(m),budget=1.0)
            a=previous === nothing ? NaN : sum(abs, w.-previous)
            f=G.cash_kelly_certified(c;tol=1e-6)
            if previous !== nothing
                X_a=src.gen(m,G.SobolOwenRule(2,G.audit_seed(rule.seed)),
                     MersenneTwister(G._default_mu_seed(rule.seed) ⊻
                                      UInt64(0x9E3779B97F4A7C15)))
                objective(z,xx)=sum(log,xx*z[1:end-1] .+ z[end])/size(xx,1)
                Bopt=objective(w,X)-objective(previous,X)
                Baudit=abs(objective(w,X_a)-objective(previous,X_a))
                @printf("IPM_INTEGRATION m=%d A=%.8g Bopt=%.8g Baudit=%.8g pass=%s\n",
                        m,a,Bopt,Baudit,
                        a<=1e-4 && Bopt<=1e-6 && Baudit<=1e-6 && f)
                flush(stdout)
            end
            @printf("IPM_LAYER m=%d secs=%.5f cert=%s A_vs_%d=%.7g gap=%.7g kkt=%.7g\n",m,(time_ns()-ts)/1e9,
                    f,previous_m,a,c.objective_gap,c.kkt_residual)
            flush(stdout)
            global previous=w
            global previous_m=m
        end
    end
    println("COMPONENT_PREFLIGHT_COMPLETE without Kelly or account equity")
    exit(0)
end

if get(ENV,"PROBE_ALLOW_ONE_DECISION","false") != "true"
    println("PREPARATION_COMPLETE; NO FULL ACCOUNT BACKTEST WAS STARTED")
    exit(0)
end

# Only now call ONE true full-size decision with exact production tolerances.
# Return or fail loudly. No account equity curve is created by this probe.
if diagnostic_reuse
    get(ENV,"PROBE_DIAGNOSTIC_DECISION","false")=="true" ||
        error("stale preflight history cannot authorize a decision")
    println("DIAGNOSTIC_ONLY full one-day decision on spot-checked old forecasts")
else
    cache.on_update=() -> length(cache.entries)%8==0 && save()
end
started=time_ns()
cap=parse(Int,get(ENV,"PROBE_DECISION_SCENARIOS","512"))
cap>=512 && cap<=131072 && ispow2(cap) ||
    error("single-day adaptive scenario budget must be a power of two 512..131072")
min_cap=parse(Int,get(ENV,"PROBE_DECISION_MIN_SCENARIOS","64"))
min_cap>=64 && ispow2(min_cap) && min_cap*4<=cap ||
    error("single-day minimum needs at least two refinement doublings")
mu_qmc=get(ENV,"PROBE_MU_QMC","false")=="true"
chisq_qmc=get(ENV,"PROBE_CHISQ_QMC","false")=="true"
decision=G.single_day_decision(mf,el,held;t=t,mode=:adaptive,
                               forecast_cache=cache,
                               posterior_tol=1e-6,
                               posterior_max_cells=2048,
                               min_scenarios=min_cap,
                               mu_qmc=mu_qmc,
                               mu_chisq_qmc=chisq_qmc,
                               max_scenarios=cap,
                               weight_tol=1e-4,
                               utility_tol=1e-6,
                               kelly_tol=1e-6)
@printf("ONE_REAL_DAY_DECISION_OK date=%s mode=%s M=%d total_seconds=%.5f cash=%.5f\n",
        date,decision.mode,decision.M,(time_ns()-started)/1e9,decision.w_cash)
if diagnostic_reuse
    println("RESULT_NOT_ADMISSIBLE: historical forecasts had old source signature")
end
flush(stdout)
if diagnostic_reuse && get(ENV,"PROBE_REPEAT_ONE_DAY","false")=="true"
    # Only repeat this SAME decision after Julia's compilation has warmed.
    # This is a steady-state cost check: no account advance, no new data, and
    # the first result must be bitwise-consistent in its decision variables.
    tgc=time_ns()
    GC.gc(true)
    @printf("ONE_REAL_DAY_GC_BEFORE_STEADY seconds=%.5f\n",(time_ns()-tgc)/1e9)
    flush(stdout)
    ts=time_ns()
    second=G.single_day_decision(mf,el,held;t=t,mode=:adaptive,
             forecast_cache=cache,posterior_tol=1e-6,
             posterior_max_cells=2048,min_scenarios=min_cap,
             max_scenarios=cap,mu_qmc=mu_qmc,mu_chisq_qmc=chisq_qmc,
             weight_tol=1e-4,utility_tol=1e-6,kelly_tol=1e-6)
    @assert isequal(second.w_universe,decision.w_universe)
    @assert second.w_cash == decision.w_cash && second.M == decision.M
    @printf("ONE_REAL_DAY_STEADY seconds=%.5f M=%d assets=%d original_timing_only=true\n",
            (time_ns()-ts)/1e9,second.M,length(mf.symbols))
    flush(stdout)
end
