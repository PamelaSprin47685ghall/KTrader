# dev/eb_convergence_probe.jl — EB 收敛分叉观察（纯诊断脚本）
#
# 目的：裁决 relative_support_tests.jl L56 的 N=3 通用路径 errored 的分叉：
#   「在途慢收敛」（迭代预算不足 → 修预算）vs「真停滞」（solver 数学问题）。
# 纪律：不改任何 production 源码与测试；本脚本不进 REQUIRED、不 export。
#
# 复现对象（逐行对照 test/relative_support_tests.jl）：
#   - L6-16  relative_support_fixture 构造（默认 seed=781, n=120；本 probe 取 N=3）
#   - L17-18 WITNESS_LIAR = RelativeSupportWitness(1,1,3)
#   - L50-52 f3/spec3/YtY3 三行
#   - L56-57 optimize_conditioned_eb 调用形态（唯一差别：显式 iters=…，
#            iters 是函数签名既有参数，零源码修改）
#
# 观察设计：
#   第一步：iters=2000（测试默认为 200，错误原文 "after 200 iterations"）。
#   第二步（阶梯观察，委派认可的替代路径）：iters ∈ 200,400,800,1600,3200，
#   每级 try-catch；收敛则打印终态证书，抛错则从错误原文提取 free_rms/evidence。
#   形态判定：free_rms 随预算单调下降 → 在途；平台不降 → 停滞。
#   （未走手动驱动内部循环的路径：复刻 src/response.jl L800-914 循环结构
#   存在抄错风险，阶梯观察是委派明确认可的诚实替代。）

using Random, LinearAlgebra
using Printf
using KTrader

function relative_support_fixture(;seed=781,n=120,N=2)
    rng=MersenneTwister(seed); P=2length(KTrader.BANDS)*N
    Q=KTrader.relative_gauge(N); Pi=Q*Q'
    X=randn(rng,n,P)
    for c in KTrader.get_constraint_columns(N,length(KTrader.BANDS))
        X[:,c]=X[:,c]*Pi
    end
    A=zeros(N,P); A[:,1:N]=3.0Pi
    Y=X*A'+0.25randn(rng,n,N)*Pi
    (; X,Y,Q,N,n)
end

const WITNESS_LIAR=KTrader.RelativeSupportWitness(1,1,3)

function extract_field(msg, name)
    m=match(Regex("$name = ([0-9.eE+-]+|[-A-Za-z]+)"), msg)
    m === nothing ? "n/a" : m.captures[1]
end

f3=relative_support_fixture(N=3)
spec3=KTrader.ridge_spectrum(f3.X,f3.X'f3.X,f3.X'f3.Y)
YtY3=collect(f3.Y'f3.Y)
println("[probe] fixture ready: N=3, n=$(f3.n), size(X)=$(size(f3.X)) (对照测试 L50-52)")

# ---- 第一步：iters=2000 ----
println("[probe] step1: iters=2000 (test default was 200)")
r2000 = try
    KTrader.optimize_conditioned_eb(spec3,YtY3,f3.n;gauge=f3.Q,
        degenerate_witness=WITNESS_LIAR,return_certificate=true,iters=2000)
catch e
    global err2000 = sprint(showerror, e)
    nothing
end
if r2000 !== nothing
    c=r2000.certificate
    println("[probe] step1 RESULT: CONVERGED, iterations=$(r2000.iterations)")
    println("[probe]   certificate.valid=$(c.valid)  alpha=$(r2000.alpha)")
    println("[probe]   alpha cert: valid=$(c.alpha.valid) location=$(c.alpha.location) ",
            "slope=$(c.alpha.slope) rel_res=$(c.alpha.rel_res)")
    println("[probe]   cov cert: symmetry=$(c.covariance.symmetry) support=$(c.covariance.support) ",
            "min_eig=$(c.covariance.min_eigenvalue)")
    println("[probe]   cov cert: projected_residual=$(c.covariance.projected_residual) ",
            "active_directions=$(c.covariance.active_directions) free_rms=$(c.covariance.free_rms)")
    println("[probe]   evidence=$(c.evidence)")
    println("[probe] VERDICT: 在途慢收敛——200 迭代预算不足，2000 内收敛。修复方向=迭代预算。")
else
    println("[probe] step1 RESULT: NOT CONVERGED even at iters=2000")
    println("[probe]   throw原文: ", err2000[1:min(end,400)])
end

# ---- 第二步：阶梯观察（无论 step1 收敛与否都跑，取得轨迹表）----
println("[probe] step2: iters ladder (free_rms / evidence trajectory)")
println("[probe]   iters | converged | free_rms        | evidence")
for it in (200,400,800,1600,3200)
    try
        r=KTrader.optimize_conditioned_eb(spec3,YtY3,f3.n;gauge=f3.Q,
            degenerate_witness=WITNESS_LIAR,return_certificate=true,iters=it)
        c=r.certificate
        fr = isdefined(c.covariance,:free_rms) ? string(c.covariance.free_rms) : "n/a"
        ev = isdefined(c,:evidence) ? string(c.evidence) : "n/a"
        @printf("[probe]   %5d | CONVERGED(iter=%d) | %-15s | %s\n", it, r.iterations, fr, ev)
    catch e
        msg=sprint(showerror,e)
        fr=extract_field(msg,"free_rms")
        ev=extract_field(msg,"evidence")
        @printf("[probe]   %5d | THREW       | %-15s | %s\n", it, fr, ev)
    end
end
println("[probe] done.")
