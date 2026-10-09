# =============================================================================
# KTraderGate0 PredictiveLaw 组合 — Step 13 测试（tiny 级）
# =============================================================================
#
# 测试对象：src/gate0/predictive.jl（八步链条组装 / 方差分解 / D-060 /
# 独立流契约）与 **posterior ↔ innovation 的接口契约**（本步核心新增
# ——两条独立开发线的真实边界）。
#
# =============================================================================
# 契约内容（钉死——任一侧 drift 出此 agreement，contract test 必须红）
# =============================================================================
#
# producer：src/gate0/oof.jl 的 `oof_residual_rows_full`（真 OOF，fold
#   网格）/ src/gate0/posterior.jl 的 `oof_residual_rows`（单 fold
#   in-sample 版——数值对照用，语义差异由其自身 docstring 钉死）。
# consumer：src/gate0/innovation.jl 的 `innovation_state` /
#   src/gate0/predictive.jl 的 `predictive_law`。
#
# crossing representation（producer → consumer 的交接物）：
#   1. ε̃ 矩阵：n×N，**asset 空间**（列 = active 域资产；producer 侧
#      E_active·r_mode 的输出形态），含 b₀ 扣除（DC 列在 B̂·x 内）；
#   2. 行 mask O_s：Bool^N（行 s 的观察资产集合——producer 不携带，
#      由调用方按观察事实并行提供；**mask 与残差行的观察事实必须一致**
#      ——mask 声称覆盖 ⇒ 该坐标有限）；
#   3. 行身份 u_s：**严格递增**（行序 = 目标日序）；
#   4. R_t：⊆ active 域索引集（1:N）；
#   5. E_active：N×N domain mode basis（两侧同一 active 域的坐标锚）。
#
# stable identity rule：行 s ↔ 目标日 u_s（ResidualOracle 的 ts_total
#   契约同源）；行集 J = {s : u_s ≤ t, O_s ⊇ R_t}（VI §1.3）。
#
# failure semantics（consumer 按契约拒绝，fail loudly）：
#   - 覆盖不足（|J| < 2）→ error "innovation coverage failure"（T4：
#     free 经 resolve_risk_domain 收缩 / locked fail）；
#   - 行身份非递增 → error；
#   - mask 与残差不一致（J 行 R 子向量 NaN）→ error；
#   - R_t 越界（⊄ 1:N）→ ArgumentError；
#   - 两侧 asset 空间维度不一致（N_a ≠ N）→ predictive_law
#     DimensionMismatch；
#   - posterior propriety 红 / s1 非正 / S<1 → 各自 fail loudly。
#
# 纪律：tiny 级（合成数据、固定 seed MersenneTwister、无 I/O、无回测）；
# 本任务不运行测试（Engineer 纪律——运行验证由 Manager 安排 DevOps 受控
# 执行，≤60s/RSS 护栏）；全部断言与回测收益/Sharpe 无关（SPEC §95 /
# 开发守则 §24）；正向契约测试**不 mock 任何一侧**——真实 producer
# machinery（fit_fold_posteriors 的 fold 独立 quadrature + 差分统计）
# 直接喂真实 consumer machinery（innovation_state 的行集/V/ℓ/q/z）。
#
# 加载链（依赖顺序——与各文件头声明的 include 顺序一致）：
# geometry → modes → response → posterior → oof → innovation → predictive。

using Test, Random, LinearAlgebra, Statistics

module Gate0PredictiveEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "response.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "posterior.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "oof.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "innovation.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "predictive.jl"))
end
using .Gate0PredictiveEnv

@testset "gate0 Step 13: PredictiveLaw composition" begin
    SEED = 20261013

    # ===================================================================
    # 契约正向：真实 producer（oof_residual_rows_full）→ 真实 consumer
    # （innovation_state）——两侧 machinery 真正相遇，零 mock。
    # DC 形态（P = 1+14N）：fit_fold_posteriors 走二维 quadrature——
    # 其结构正确性依赖 DC 形态（无 DC 时 evidence 与 u₀ 无关、二维
    # 积分不衰减——posterior.jl 的 fit_full_posterior 有一维 fallback
    # 而 oof.jl 没有；接口摩擦如实记录于收尾报告）。
    # ===================================================================
    @testset "contract positive: producer→consumer" begin
        N_ct = 3
        P_ct = 1 + 14 * N_ct                      # 43（DC 形态）
        n_ct = 66
        # hcat 修复（Wave 3 收尾轮）：原 [fill(1.0, n_ct)
        #  randn(...)] 的换行在 [] 字面量里是 vcat（行分隔）语义——
        # Vector 与 Matrix vcat 报 DimensionMismatch (1,42)。本意 DC 列
        # 拼接 = hcat（显式调用，避免换行/空格歧义）。
        X_ct = hcat(fill(1.0, n_ct),
                    randn(MersenneTwister(SEED + 20), n_ct, P_ct - 1))
        # fixture 信号修正（Wave 3 收尾轮）：原 Y_ct = 纯噪声——evidence
        # 的 α₀ 右尾平台高于峰（实测 161.5 vs 159.3），logf 边界衰减 =
        # u_span − (平台−峰) < u_span，tail 证书结构性红（积分本身良
        # 定、域外质量 ~1e-8 可忽略——证书判据对弱信号形态过严，见
        # tail_diag.log）。信号必须含 DC 列（列 1）：α₀ 只正则 DC 列，
        # 信号在动态列不改变 u₀ 方向的信息量。含 b₀ 的信号使 u₀ 方向
        # 有真峰（衰减 = u_span + (峰−平台) > u_span）。
        B_sig = 0.3 .* randn(MersenneTwister(SEED + 21), 5, N_ct)
        Y_ct = X_ct[:, 1:5] * B_sig .+ 0.05 .* randn(MersenneTwister(20261014), n_ct, N_ct)
        @test !all(iszero, vec(mean(Y_ct, dims = 1)))   # b₀ 信号健全（DC 系数非零）
        E_act = domain_mode_basis(N_ct)
        folds = fold_grid(n_ct; F_folds = 3)
        # quadrature 收敛配置调参（Wave 3 收尾轮，交付者自报「未经运行
        # 确认」——实测确认）：原 tol=1e-4/max_cells=1024 不可达（中点-
        # 梯形差估计器下实测收敛率 rel ∝ 1/cells：1024→9.8e-3、4096→
        # 2.4e-3；tol=1e-4 需 ~1e5 cells、60s 预算内不可行）。契约测试
        # 只断言结构性质（维度/有限性/within_computed），数值精度由
        # 端到端的点质量退化路径承担（不经 quadrature）——tol=5e-3/
        # max_cells=4096（实测 rel 2.4e-3，余量 2 倍，总时长 ~19s）。
        fp = fit_fold_posteriors(X_ct, Y_ct, folds; tol = 5e-3, max_cells = 4096)
        eps_tilde = oof_residual_rows_full(fp, X_ct, Y_ct, E_act)  # n×N asset 空间（真 OOF）
        @test size(eps_tilde) == (n_ct, N_ct)
        @test all(isfinite, eps_tilde)
        row_ids_ct = collect(200 .+ (1:n_ct))     # u_s 严格递增（行 s ↔ 目标日）
        t_ct = 200 + n_ct
        masks_ct = [trues(N_ct) for _ in 1:n_ct]
        R_ct = [1, 3]                              # R ⊂ active（资产 2 不进 risk 域）
        st_ct = innovation_state(eps_tilde, row_ids_ct, masks_ct, R_ct, E_act; t = t_ct)
        @test st_ct isa InnovationState
        @test st_ct.N_R == 2
        @test all(V -> size(V) == (2, 2), st_ct.V_t)
        @test isapprox(sum(st_ct.d_weights), 1.0; atol = 1e-12)
        @test length(st_ct.L_t) >= 1
        Zpool = z_pool_rows(st_ct, st_ct.d_nodes[1])
        @test size(Zpool, 1) == length(st_ct.L_t) && size(Zpool, 2) == 2
        # 组装层：真实 posterior + 真实 innovation state → predictive_law
        x_ct = [1.0; randn(MersenneTwister(SEED + 22), P_ct - 1)]
        res_ct = predictive_law(fp.posteriors[1], st_ct, x_ct;
                                s1 = fill(0.01, N_ct), E_active = E_act,
                                rng = MersenneTwister(SEED + 23), S = 64)
        @test res_ct isa PredictiveLawResult
        @test res_ct.R_t == R_ct
        @test size(res_ct.gross) == (64, 2)
        @test all(isfinite, res_ct.gross)
        # fold posterior 的 ν = n^(−f)+1−N = 44+1−3 = 42 > 2 → within 可算
        @test res_ct.within_computed && isempty(res_ct.not_computed)
    end

    # ===================================================================
    # 契约负向（mutation 式）：producer 侧的 plausible 不兼容输出 →
    # consumer 按契约拒绝（error）——任一侧单独 drift 立即红。
    # ===================================================================
    @testset "contract negative: mutation" begin
        N_ct = 3; n_ct = 66
        E_act = domain_mode_basis(N_ct)
        row_ids_ct = collect(200 .+ (1:n_ct)); t_ct = 200 + n_ct
        masks_ct = [trues(N_ct) for _ in 1:n_ct]
        R_ct = [1, 3]
        # 重用正向 fixture 的 eps_tilde（同 seed 重建）
        P_ct = 1 + 14 * N_ct
        X_ct = [fill(1.0, n_ct) randn(MersenneTwister(SEED + 20), n_ct, P_ct - 1)]
        # fixture 信号修正（同契约正向——含 DC 列；见正向注释）
        B_sig = 0.3 .* randn(MersenneTwister(SEED + 21), 5, N_ct)
        Y_ct = X_ct[:, 1:5] * B_sig .+ 0.05 .* randn(MersenneTwister(20261014), n_ct, N_ct)
        fp = fit_fold_posteriors(X_ct, Y_ct, fold_grid(n_ct; F_folds = 3);
                                 tol = 5e-3, max_cells = 4096)
        eps_tilde = oof_residual_rows_full(fp, X_ct, Y_ct, E_act)
        # (1) mask 与残差行不一致：mask 声称行 30 覆盖资产 3，残差该坐标 NaN
        eps_bad = copy(eps_tilde); eps_bad[30, 3] = NaN
        err_mask = try
            innovation_state(eps_bad, row_ids_ct, masks_ct, R_ct, E_act; t = t_ct)
            nothing
        catch e; e end
        @test err_mask isa ErrorException
        @test occursin("非有限", err_mask.msg) || occursin("finite", err_mask.msg)
        # (2) R_t 越界（不在 active 域 1:N 内）
        @test_throws ArgumentError innovation_state(eps_tilde, row_ids_ct, masks_ct,
                                                    [1, 4], E_act; t = t_ct)
        # (3) 行身份非递增
        ids_bad = copy(row_ids_ct); ids_bad[10] = ids_bad[9]
        err_ids = try
            innovation_state(eps_tilde, ids_bad, masks_ct, R_ct, E_act; t = t_ct)
            nothing
        catch e; e end
        @test err_ids isa ErrorException
        # (4) 两侧 asset 空间维度不一致（N_a ≠ N）——predictive_law 拦。
        #     st_ct 作用域修复（Wave 3 收尾轮）：交付者注释称「@testset
        #     body 内联展开、不引入新作用域——跨 testset 可见」——错误：
        #     Julia 的 @testset 引入新作用域，正向 testset 的 st_ct 不泄
        #     漏。此处用负向已重建的 eps_tilde 重建（同款参数）。
        st_ct = innovation_state(eps_tilde, row_ids_ct, masks_ct, R_ct, E_act;
                                 t = t_ct)
        N_nc = 2
        X_nc = randn(MersenneTwister(SEED + 24), 3, 2)
        Y_nc = 0.05 .* randn(MersenneTwister(SEED + 25), 3, N_nc)
        post_nc = fit_full_posterior(X_nc, Y_nc; prior = :point_mass, alpha_fixed = 0.5)
        @test_throws DimensionMismatch predictive_law(
            post_nc, st_ct, randn(MersenneTwister(SEED + 26), 2);
            s1 = fill(0.01, N_nc), E_active = domain_mode_basis(N_nc),
            rng = MersenneTwister(1), S = 8)
    end

    # ===================================================================
    # 端到端数值：合成小世界（已知 B*、已知 innovation 协方差）→
    # predictive_law 的 scenario 统计恢复（均值/方差对照，宽松容差——
    # scenario 是抽样）。点质量退化（alpha_fixed）加速；残差用
    # oof_residual_rows（单 fold in-sample 版——数值对照用途，其
    # docstring 钉死 in-sample 语义；OOF 语义归上方 contract test）。
    # ===================================================================
    @testset "end-to-end numeric recovery" begin
        N_e = 3; P_e = 4; n_e = 90                 # 小 P（数学层允许任意 P）
        rng_e = MersenneTwister(SEED + 30)
        X_e = randn(rng_e, n_e, P_e)
        B_star = 0.1 .* randn(rng_e, P_e, N_e)
        Sig = 0.01 .* Matrix{Float64}(I, N_e, N_e)
        Sig[1, 2] = Sig[2, 1] = 0.3 .* 0.01        # 已知 cross 结构
        L_sig = cholesky(Symmetric(Sig)).L
        # 行抽样形态修复（Wave 3 收尾轮）：原 L_sig * randn(n, N) 是
        # N×N 乘 n×N——维度不匹配。ε_s ~ N(0,Σ) 的行形态 = Z·Lᵀ
        # （每行 ε_sᵀ = L·z_s ⇒ 矩阵 E = Z·Lᵀ，n×N）。
        E_err = randn(rng_e, n_e, N_e) * L_sig'    # ε ~ N(0, Σ*)
        # 转置修正（Wave 3 收尾轮）：B_star 已是 P×N（4×3），X_e·B_star
        # 即 n×N；原 B_star'（N×P）与 X_e 的 4≠3 报 SYRK 维度错。
        Y_e = X_e * B_star .+ E_err                 # mode 空间（active 域）
        E_act_e = domain_mode_basis(N_e)
        post_e = fit_full_posterior(X_e, Y_e; prior = :point_mass, alpha_fixed = 0.1)
        eps_e = oof_residual_rows(post_e, E_act_e)   # n×N asset 空间（in-sample）
        row_ids_e = collect(300 .+ (1:n_e))
        R_e = [1, 2]                                 # R ⊂ active（资产 3 不进）
        st_e = innovation_state(eps_e, row_ids_e,
                                [trues(N_e) for _ in 1:n_e], R_e, E_act_e; t = 300 + n_e)
        x_t_e = randn(rng_e, P_e)
        s1_e = [0.01, 0.012, 0.02]
        S_mc = 6000
        res = predictive_law(post_e, st_e, x_t_e; s1 = s1_e, E_active = E_act_e,
                             rng = MersenneTwister(SEED + 31), S = S_mc)
        # --- R 子集执行路径：列集 = R_t（资产 3 构造性不进 scenario）---
        @test res.R_t == R_e
        @test size(res.gross) == (S_mc, 2)
        @test size(res.log_gross) == (S_mc, 2)
        # --- 方差分解两分量之和 ≈ 总 scenario 方差（MC 对照，宽松容差：
        #     差异来源 = M_z ≠ I 的经验形状（VI §5.4——报告口径 E[V_t]
        #     与 ε 完整二阶矩之差）+ MC 噪声 + t 尾部）---
        for a in 1:2
            v_mc = var(res.log_gross[:, a])
            v_dec = (res.var_epistemic[a, a] + res.var_aleatoric[a, a]) * s1_e[a]^2
            @test v_mc > 0
            @test isapprox(v_mc, v_dec; rtol = 0.35)
        end
        # epistemic = within + between（点质量单节点 → between = 0）
        @test maximum(abs.(res.var_between)) < 1e-15
        @test isapprox(res.var_epistemic, res.var_within; atol = 1e-14)
        # --- 均值恢复：E[log_gross]/s₁ ≈ μ_R + E[ε]（ε 均值 = z pool
        #     经验均值的 V^{1/2} 传播——精确手算，非近似）---
        eps_mean_R = zeros(2)
        for g in eachindex(st_e.d_weights)
            Zg = z_pool_rows(st_e, st_e.d_nodes[g])
            Fg = mp_sqrt_factors(st_e.V_t[g])
            eps_mean_R .+= st_e.d_weights[g] .* (Fg.half * vec(mean(Zg, dims = 1)))
        end
        for a in 1:2
            m_mc = mean(res.log_gross[:, a]) / s1_e[a]
            @test abs(m_mc - (res.mu_R[a] + eps_mean_R[a])) < 0.01
        end
        # μ_R 报告值与 predict_mu 的确定性投影一致（非 MC——解析断言）
        pmu = predict_mu(post_e, x_t_e)
        mu_R_hand = st_e.E_R' * (E_act_e * pmu.mu)[R_e]
        @test isapprox(res.mu_R, mu_R_hand; atol = 1e-14)
        # aleatoric = E_R·(Σ_g q_g·V_g)·E_Rᵀ（手算双构造）
        EV_hand = zeros(2, 2)
        for g in eachindex(st_e.d_weights)
            EV_hand .+= st_e.d_weights[g] .* st_e.V_t[g]
        end
        @test isapprox(res.var_aleatoric, st_e.E_R * EV_hand * st_e.E_R'; atol = 1e-14)
        # --- 同 seed 重放：μ draw / d 抽样 / 行抽样的独立流契约——
        #     同 seed 同输入逐位一致（SPEC §36 / 裁决 F）---
        r2 = predictive_law(post_e, st_e, x_t_e; s1 = s1_e, E_active = E_act_e,
                            rng = MersenneTwister(SEED + 31), S = S_mc)
        @test r2.gross == res.gross
        @test r2.log_gross == res.log_gross
        @test r2.var_epistemic == res.var_epistemic
    end

    # ===================================================================
    # D-060 接口负测试：返回对象不含 Σ_R 通道（字段层面）
    # ===================================================================
    @testset "D-060: no Sigma_R channel" begin
        fn = fieldnames(PredictiveLawResult)
        @test :Sigma_R ∉ fn
        @test :sigma_R ∉ fn
        varfields = filter(n -> startswith(string(n), "var_"), fn)
        @test Set(varfields) == Set([:var_epistemic, :var_aleatoric,
                                     :var_within, :var_between])
        # 方差分解只有两层来源（epistemic + aleatoric 同域可加）——
        # 无第三个「未来残差冲击」通道
        @test :var_epistemic ∈ fn && :var_aleatoric ∈ fn
    end

    # ===================================================================
    # NOT COMPUTED 路径：ν ≤ 2 时 within NaN → 报告标注（不显示 0）
    # ===================================================================
    @testset "NOT COMPUTED (nu <= 2)" begin
        # N=2、n=3 → ν = n+1−N = 2：t 矩二阶矩不存在
        rng_nc = MersenneTwister(SEED + 32)
        X_nc = randn(rng_nc, 3, 2)
        Y_nc = 0.05 .* randn(rng_nc, 3, 2)
        E_nc = domain_mode_basis(2)
        post_nc = fit_full_posterior(X_nc, Y_nc; prior = :point_mass, alpha_fixed = 0.5)
        @test post_nc.nu == 2
        eps_nc = oof_residual_rows(post_nc, E_nc)
        st_nc = innovation_state(eps_nc, collect(10 .+ (1:3)),
                                 [trues(2) for _ in 1:3], [1, 2], E_nc; t = 13)
        res_nc = predictive_law(post_nc, st_nc, randn(rng_nc, 2);
                                s1 = [0.01, 0.01], E_active = E_nc,
                                rng = MersenneTwister(SEED + 33), S = 50)
        @test !res_nc.within_computed
        @test res_nc.not_computed == [:within]
        @test any(isnan, res_nc.var_within)          # NaN 携带——不显示 0
        @test any(isnan, res_nc.var_epistemic)
        # aleatoric 仍可算（innovation 层与 ν 无关——两层独立）
        @test all(isfinite, res_nc.var_aleatoric)
        # scenario 本身可生成（t_2 抽样存在——只有二阶矩不存在）
        @test size(res_nc.gross) == (50, 2)
        @test all(isfinite, res_nc.gross)
    end

    # ===================================================================
    # fail-loudly 传导：propriety 红传导 / innovation 覆盖不足传导
    # （T4 语义——predictive_law 不吞错）/ 自身参数校验
    # ===================================================================
    @testset "fail-loudly propagation" begin
        rng = MersenneTwister(SEED + 34)
        # posterior propriety 红传导：n < N → fit 层 gate（真实调用链）
        err_prop = try
            fit_full_posterior(randn(rng, 2, 4), randn(rng, 2, 3);
                               prior = :point_mass, alpha_fixed = 0.5)
            nothing
        catch e; e end
        @test err_prop isa ErrorException
        @test occursin("posterior improper", err_prop.msg)
        # innovation 覆盖不足传导（T4）：R_t 资产全历史缺覆盖 → 构造层拦
        N_ct = 3; n_ct = 66
        E_act = domain_mode_basis(N_ct)
        P_ct = 1 + 14 * N_ct
        X_ct = [fill(1.0, n_ct) randn(MersenneTwister(SEED + 20), n_ct, P_ct - 1)]
        # fixture 信号修正（同契约正向——含 DC 列；见正向注释）
        B_sig = 0.3 .* randn(MersenneTwister(SEED + 21), 5, N_ct)
        Y_ct = X_ct[:, 1:5] * B_sig .+ 0.05 .* randn(MersenneTwister(20261014), n_ct, N_ct)
        fp = fit_fold_posteriors(X_ct, Y_ct, fold_grid(n_ct; F_folds = 3);
                                 tol = 5e-3, max_cells = 4096)
        eps_tilde = oof_residual_rows_full(fp, X_ct, Y_ct, E_act)
        masks_no1 = [begin m = trues(N_ct); m[1] = false; m end for _ in 1:n_ct]
        err_cov = try
            innovation_state(eps_tilde, collect(200 .+ (1:n_ct)), masks_no1,
                             [1, 3], E_act; t = 200 + n_ct)
            nothing
        catch e; e end
        @test err_cov isa ErrorException
        @test occursin("innovation coverage failure", err_cov.msg)
        # predictive_law 自身参数校验（不吞、不降级、不默认）
        N_e = 3; P_e = 4; n_e = 90
        rng_e = MersenneTwister(SEED + 30)
        X_e = randn(rng_e, n_e, P_e)
        # 转置修正（同端到端 testset：randn(P,N) 已是 P×N，直接乘）
        Y_e = X_e * (0.1 .* randn(rng_e, P_e, N_e)) .+
              0.01 .* randn(rng_e, n_e, N_e)
        E_act_e = domain_mode_basis(N_e)
        post_e = fit_full_posterior(X_e, Y_e; prior = :point_mass, alpha_fixed = 0.1)
        eps_e = oof_residual_rows(post_e, E_act_e)
        st_e = innovation_state(eps_e, collect(300 .+ (1:n_e)),
                                [trues(N_e) for _ in 1:n_e], [1, 2], E_act_e;
                                t = 300 + n_e)
        x_t_e = randn(rng_e, P_e)
        s1_e = [0.01, 0.012, 0.02]
        @test_throws ArgumentError predictive_law(post_e, st_e, x_t_e;
            s1 = s1_e, E_active = E_act_e, rng = MersenneTwister(1), S = 0)
        @test_throws DimensionMismatch predictive_law(post_e, st_e, x_t_e[1:3];
            s1 = s1_e, E_active = E_act_e, rng = MersenneTwister(1), S = 8)
        @test_throws DimensionMismatch predictive_law(post_e, st_e, x_t_e;
            s1 = [0.01, 0.01], E_active = E_act_e, rng = MersenneTwister(1), S = 8)
        # 类型名修正（Wave 3 收尾轮）：原 @test_throws DimensionError——
        # 拼写错误（无此类型，UndefVarError）；s1 非正检查实际抛
        # DomainError（predictive.jl:194）。
        @test_throws DomainError predictive_law(post_e, st_e, x_t_e;
            s1 = [0.01, -0.012, 0.02], E_active = E_act_e, rng = MersenneTwister(1), S = 8)
        @test_throws DimensionMismatch predictive_law(post_e, st_e, x_t_e;
            s1 = s1_e, E_active = domain_mode_basis(4), rng = MersenneTwister(1), S = 8)
    end
end
