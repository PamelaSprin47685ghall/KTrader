# Small, bounded runtime study ONLY. Real 64-asset response posterior at one
# date, synthetic 80-row FULL-DIMENSION innovation merely to exercise scenario
# generation. No Kelly, no live trades, no account backtest.
using Dates, Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "gate0", "KTraderGate0.jl"))
const M = KTraderGate0
mf = M.load_market_facts(get(ENV,"GATE0_DATA_DIR","data"))
el = M.Eligibility(mf)
date = Date(get(ENV,"PROBE_DATE","2024-10-02"))
origin = findfirst(==(date),mf.dates)
origin === nothing && error("session absent")
p = M._response_prefix(mf,el,origin;compact=true)
t0=time_ns()
post=M.fit_full_posterior(p.Xtr,p.Ytr)
println("ACTUAL_RESPONSE_OK N=$(post.N) P=$(post.P) nodes=$(length(post.alpha_weights)) secs=$((time_ns()-t0)/1e9)")
flush(stdout)
Nr=post.N
joint_rows = parse(Int,get(ENV,"PROBE_JOINT_ROWS","80"))
80 <= joint_rows <= 600 || error("bounded joint rows must be 80..600")
resid=0.02 .* randn(MersenneTwister(74),joint_rows,Nr)
ids=collect(1:joint_rows)
masks=[trues(Nr) for _ in ids]
d_nodes,d_cells = if get(ENV,"PROBE_D_NODES","1")=="7"
    M.initial_mesh()
else
    ([0.5],[1.0])
end
st=M.innovation_state(resid,ids,masks,collect(1:Nr),p.E;
          t=joint_rows,d_nodes=d_nodes,d_cells=d_cells,require_full_rank=true)
xnow=p.xnow
qmc = get(ENV,"PROBE_MU_QMC","false") == "true"
lazy_z = get(ENV,"PROBE_LAZY_Z","true") == "true"
src=M.predictive_rqmc_source(post,st,xnow;s1=p.s1,E_active=p.E,
                              mu_qmc=qmc, mu_chisq_qmc=qmc,lazy_z=lazy_z)
rule=M.SobolOwenRule(2,UInt64(0x12345678))
for K in (16,64,256,1024,4096,8192)
    K > parse(Int,get(ENV,"PROBE_MAX_M","256")) && continue
    local t0=time_ns()
    X=src.gen(K,rule,MersenneTwister(79))
    @assert size(X)==(K,Nr) && all(x->isfinite(x) && x>0,X)
    @printf("SCENARIO_GEN_OK M=%d secs=%.5f\n",K,(time_ns()-t0)/1e9)
    flush(stdout)
end
