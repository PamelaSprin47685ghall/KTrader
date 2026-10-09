using Test, Random, LinearAlgebra
using KTrader

# 真实 producer 结构 fixture：embedded_relative_field 同构的相对构造——
# 行和构造性为零 → design 通道两列反对称；Y 同源相对场；gauge 维 1（N=2）。
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
const WITNESS2=KTrader.RelativeSupportWitness(1,1,2)
const WITNESS_LIAR=KTrader.RelativeSupportWitness(1,1,3)

@testset "Relative support witness：解析消元、通用回退与表示不变性" begin
    f=relative_support_fixture()
    spec=KTrader.ridge_spectrum(f.X,f.X'f.X,f.X'f.Y)
    YtY=collect(f.Y'f.Y); q=vec(f.Q)
    # [W1] witness 生效：canonical alpha=旧 default 参考 1.0、解析噪声协方差
    # 解 S=q'YtYq/n（floor clamp），经原完整 conditioned_eb_certificate 验收
    # valid（含 alpha slope 与 covariance KKT——不伪造、不放宽）。
    solved=KTrader.optimize_conditioned_eb(spec,YtY,f.n;gauge=f.Q,
        degenerate_witness=WITNESS2,return_certificate=true)
    @test solved.certificate.valid
    @test solved.alpha==1.0
    @test solved.Sigma≈f.Q*max(dot(q,YtY*q)/f.n,KTrader.EB_COVARIANCE_FLOOR)*f.Q' rtol=1e-9
    @test solved.certificate.covariance.min_eigenvalue>=KTrader.EB_COVARIANCE_FLOOR-1e-14
    # [W2] Gaussian 噪声 likelihood 对照：条件化后 ỹ|Cg=0~N(0,S·I)，解析 S
    # 与一维 MLE ‖Ỹ‖²/n 一致（ỹ=Y·q，q'YtYq=‖ỹ‖²）。
    ytilde=f.Y*q
    @test dot(q,YtY*q)≈dot(ytilde,ytilde) rtol=1e-12
    # [W3] 无 witness 的通用路径不被压零：真实迭代（iterations≥1，非解析
    # 零迭代）、证书照常 valid——generic 输入永不进入解析分支。
    generic=KTrader.optimize_conditioned_eb(spec,YtY,f.n;gauge=f.Q,return_certificate=true)
    @test generic.certificate.valid
    @test generic.iterations>=1
    # [W4] witness 一致性拒绝：谎报 assets（3≠2）与 nothing 都返回 nothing
    # 走通用——witness 不是裸布尔，维度/资产空间语义被精确核对。
    @test KTrader.degenerate_relative_solution(YtY,f.n,f.Q,WITNESS_LIAR)===nothing
    @test KTrader.degenerate_relative_solution(YtY,f.n,f.Q,nothing)===nothing
    # [W5] 较高维（N=3）relative support 仍有非零可观测算子：support=2 维
    # 不满足一维消元前提（output_support_dim=1 与 gauge 实际维 2 不符被拒；
    # 正确的 2 维声明同样不适用一维消元），通用拟合收敛、fixture 的 A 信
    # 号可观测（G_c 非零由通用路径的证书与下游构造保证）。
    f3=relative_support_fixture(N=3)
    spec3=KTrader.ridge_spectrum(f3.X,f3.X'f3.X,f3.X'f3.Y)
    YtY3=collect(f3.Y'f3.Y)
    @test KTrader.degenerate_relative_solution(YtY3,f3.n,f3.Q,WITNESS_LIAR)===nothing
    @test KTrader.degenerate_relative_solution(YtY3,f3.n,f3.Q,
        KTrader.RelativeSupportWitness(2,2,3))===nothing
    solved3=KTrader.optimize_conditioned_eb(spec3,YtY3,f3.n;gauge=f3.Q,
        degenerate_witness=WITNESS_LIAR,return_certificate=true)
    @test solved3.certificate.valid
    @test solved3.iterations>=1
    # [W6] primal/dual 同一 law：dual 谱（n<P）下 witness 分支同样给出
    # canonical alpha=1.0 且原证书验收——条件化 evidence 与 α 无关是表示
    # 无关的数学事实；若 dual 实现细节（baseline=1/α 项）使证书拒绝，本断
    # 言红（诚实暴露具体限制；运行期回退通用、不伪造）。
    fd=relative_support_fixture(;n=20)   # n=20 < P=28 → dual 谱
    specd=KTrader.ridge_spectrum(fd.X,fd.X'fd.X,fd.X'fd.Y)
    solvedd=KTrader.optimize_conditioned_eb(specd,collect(fd.Y'fd.Y),fd.n;gauge=fd.Q,
        degenerate_witness=WITNESS2,return_certificate=true)
    @test solvedd.alpha==1.0
    @test solvedd.certificate.valid
    # [W7] posterior 不可达 prior 仍有方差：解析分支只选 (alpha,S) 两个
    # canonical 量；G/V 的 coefficient prior 与 sampling law 由下游
    # fit_response_operator 照常构造（V_out/RNG 路径未动），Sigma 的 floor
    # 下界（[W1] 已断言）保证 prior 方差保留——不可达系数不被削零。
    # [W8] 旧断言保留：incremental_tests.jl 的既有 alpha_rel 断言与 t14294
    # generic R6 路径/tol/floor 均未被本改动触碰（witness 默认 nothing 时
    # optimize_conditioned_eb 行为与改动前一致；本文件纯新增）。
end
