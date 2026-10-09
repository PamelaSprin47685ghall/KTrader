# test/posterior_contract_tests.jl — 新增源码回归（本文件由本次任务新增，未改动任何现有文件）。
#
# DevOps 接入需求：将本文件名加入 test/runtests.jl 第 11 行的 extra 元组，与
# "conditioned_eb_tests.jl" 等并列即可；本文件不自行修改 runtests.jl。独立运行
# （julia --project test/posterior_contract_tests.jl）同样成立。
#
# 两个契约，全部是动态断言（需要真实运行才产生证据；本文件不宣称运行结果）：
#
# [契约 1] moving-fold boundary 下的 warm/cold 对照。
#   生产语义（predict.jl 310-342、incremental.jl 126-129/382-387）：fold 边界由
#   div(n,F) 决定，n 每日 +1，边界跨日移动；第 f 折的 EB warm start 是上一决策日
#   同折的 (alpha_macro, alpha_rel)（predict.jl 391 行回写 state.warm）。在移动日，
#   上一日折 f 的训练集（= 上一日全数据 − 上一日折 f eval）可以包含今日折 f 的
#   heldout 行：本 fixture 用 F=3、n_res 101→102（T=358→359），昨日折 1 train 含
#   今日折 1 heldout 行 34，昨日折 2 train 含今日折 2 heldout 行 67、68。
#   源码论证（非动态证据）：maximize_logalpha（response.jl 212-273）的候选集合
#   由全局网格的全部符号变化区间产生，warm 仅在落入某区间时把该区间已验证的
#   bracket 各收缩 1e-3，argmax 在全部候选上全局取优，浅平坦区（差 < 256·eps·scale）
#   锚定 log 网格几何中心——输出与 initial 无关，仅二分路径的浮点舍入可差；
#   optimize_conditioned_eb 的 Sigma 起点固定为 alpha=1 的参考 cache（response.jl
#   654-656 行），暖 alpha 不选择 covariance basin。因此 warm 只能作为数值起点，
#   最终解必须与 cold 在 KKT 收敛容差内一致。本测试把该论证钉成动态对照：
#   同一 bit 级今日 train Grams（inference_checkpoint 分叉 + 相同 advance 序列），
#   唯一差异是 alpha_initial（昨日 warm vs (1.0,1.0) 固定冷值）。
#
#   容差出处（不是"为绿而放水"）：两路各自收敛到 bounded_alpha_certificate /
#   covariance_certificate 的 KKT 停止容差（rel_res ≤ 1e-6·scale、free_rms ≤ 1e-6）
#   内的驻点，参数差的合法量级由此容差与 evidence 曲率决定；alpha 的 bracket
#   收敛 tol 为 1e-9（conditioned）/ 1e-12（matrix-normal）。若运行差异超出
#   下列容差，本测试必须红，且红的原因不是 tie-break：浅平坦时锚定几何中心，
#   warm/cold 应严格相等；非平坦时 argmax 选同一候选。超出容差的差异只能按
#   真分歧上报（例如 warm 意外进入 evidence 或候选集合）。
#
# [契约 2] 非 iid ScenarioQuadrature 下 lazy ResidualOracle view 与 dense view
#   的 scenario 与 Kelly 对照。generate_scenarios_v1（predict.jl 436-508）在
#   quadrature 模式下不消耗 rng（uniforms=nothing、无 rand/randn!），行选择与
#   own-fallback 完全由 rule 的确定性 uniform 值与观测 mask 决定；两 view 的
#   唯一差异是残差行的求值方式（oracle 的 fold-grouped GEMM vs dense 矩阵的
#   行索引），跨 view 一致性预期在 roundoff 尺度（求和顺序不同），同 view 同
#   rule 同 S 的 replay 预期 bit 级（oracle 无 cache、model 不被 mutate）。
#   独立 dense 真参考使用现有 dense_oof_residuals（residual_oracle.jl），本
#   文件不复刻任何新 oracle。反例 mutant（错 fold 划分、NaN→0）必须被抓。
#
#   own-support 边界的命中语义（本轮修正）：概率命中不是因果保证。fixture A
#   用确定性预检（pc_shared_rows，调生产原语 quadrature_uniform 与 observed）
#   逐 scenario 复算 pass 1 的 shared-row 选择，断言至少一个 scenario 落入
#   资产 3 的 unobserved 行——fallback 被真实进入是运行期验证过的事实，不是
#   分布论证。fixture B 构造为覆盖全部合法 rows 的未观测（资产 3 从 WARMUP
#   起消失，全部标签行 unobserved）：任何 shared row 都进入 own-fallback，
#   own 为空 → 抛错是构造性必然，与 rule/seed/S 无关。预检红时的处置是修
#   fixture 的覆盖（扩大缺失区间），不是更换 seed；fixture 退化触发 EB 失败
#   时的处置是精确修 fixture 的可拟合性，不是放宽证书。

using Test, Random, LinearAlgebra, Statistics, Distributions
using KTrader
const PC = KTrader

# ---- 局部 helper（前缀 pc_ 避免与 residual_oracle_tests.jl / flow 文件的
# ---- Main 命名空间 helper 冲突；runtests.jl 顺序 include 时安全）。

"""复刻 _fit_prepared_v1 的 OOF 折循环（同生产：full−fold Grams、
alpha_initial=(1.0,1.0) 冷值、need_uncertainty=false、N>2 时 witness=nothing），
产出 dense_oof_residuals 所需的 fold 模型。不复刻 oracle 本身。"""
function pc_refit_folds(prep; F_folds=3, ridge_alpha=nothing)
    stats = prep.stats
    fold_models = PC.ResponseOperator[]
    for fold in 1:F_folds
        fold_eval = stats.ranges[fold]
        S_xx_train = stats.xx === nothing ? nothing : stats.full_xx - stats.xx[fold]
        S_xy_train = stats.full_xy - stats.xy[fold]
        S_yy_train = stats.full_yy - stats.yy[fold]
        train_indices = [1:first(fold_eval)-1; last(fold_eval)+1:prep.n_res]
        train_ts = prep.ts_total[train_indices]
        train_design = length(train_indices) < prep.P_features ?
                        prep.X_rel_stacked[train_indices, :] : nothing
        resp_oof = PC.fit_response_operator(prep.B_m, fill(Float64[], prep.N), prep.m,
                                            prep.relative_embedding; ridge_alpha,
                                            ts = train_ts,
                                            S_xx_rel = S_xx_train, S_xy_rel = S_xy_train,
                                            S_yy_rel = S_yy_train, X_design = train_design,
                                            alpha_initial = (1.0, 1.0),
                                            need_uncertainty = false)
        push!(fold_models, resp_oof)
    end
    fold_models
end

"""同一 model，仅替换 res_history view（字段循环替换，避免手抄字段序）。"""
function pc_swap_res(model::PC.V1Model, res)
    args = ntuple(i -> fieldname(PC.V1Model, i) == :res_history ? res : getfield(model, i),
                  fieldcount(PC.V1Model))
    PC.V1Model(args...)
end

"""确定性预检：用生产原语逐 scenario 复算 pass 1 的 shared-row 选择。

ur 用生产 quadrature_uniform（predict.jl 412-421），row 公式照抄 predict.jl
474 行（min(floor(Int, ur*T)+1, T)）；观测状态对 oracle view 用生产 observed
（O(1) 查冻结 mask），对 dense view 用 isfinite。返回 (rows, hits)：hits 是
shared row 落入资产 j unobserved 行的 scenario 数。hits >= 1 证明 own-fallback
边界被真实进入——运行期确定性验证，不是分布论证。预检红（hits == 0）时的
处置是修 fixture 覆盖（扩大该资产缺失区间使 unobserved 更广），不是换 seed。"""
function pc_shared_rows(model, rule, S, j)
    res = model.res_history
    T = size(res, 1)
    N = length(model.active_indices)
    rows = [min(floor(Int, PC.quadrature_uniform(rule, s, N + 3) * T) + 1, T) for s in 1:S]
    isobs = row -> res isa PC.ResidualOracle ? PC.observed(res, row, j) : isfinite(res[row, j])
    hits = count(row -> !isobs(row), rows)
    (; rows, hits)
end

# ===========================================================================
# [契约 1] moving-fold boundary：warm（上一日训练含今日 heldout）vs cold
# ===========================================================================
@testset "moving-fold boundary: yesterday-train contains today's heldout, warm == cold" begin
    # T=358 → n_res=101（ranges [1:33,34:66,67:101]）；T=359 → n_res=102
    # （ranges [1:34,35:68,69:102]）。N=3 避开 N==2 的 degenerate witness
    # （witness 会把 alpha 钉为 1.0，使 warm/cold 对照平凡化）。
    rng = MersenneTwister(2041)
    T, N = 359, 3
    returns = 0.007 .* randn(rng, T, N)
    returns[280:290, :] .*= 18.0        # 一段强漂移，让 EB 解远离 1.0
    prices = exp.(cumsum(returns; dims = 1))

    # ---- 前提的显式动态验证：边界真的移动，且昨日折训练集含今日折 heldout ----
    prep_prev = PC._prepare_v1(prices[1:358, :]; F_folds = 3)
    prep_today = PC._prepare_v1(prices[1:359, :]; F_folds = 3)
    @test prep_prev.n_res == 101 && prep_today.n_res == 102
    @test prep_prev.stats.ranges != prep_today.stats.ranges
    @test prep_prev.stats.ranges == [1:33, 34:66, 67:101]
    @test prep_today.stats.ranges == [1:34, 35:68, 69:102]
    n_prev, n_today = prep_prev.n_res, prep_today.n_res
    # 昨日折 1 train = setdiff(1:101, 1:33) = 34:101，含今日折 1 eval 的行 34。
    @test !isempty(intersect(setdiff(1:n_prev, prep_prev.stats.ranges[1]),
                             prep_today.stats.ranges[1]))
    # 昨日折 2 train = 1:33 ∪ 67:101，含今日折 2 eval 的行 67、68。
    @test !isempty(intersect(setdiff(1:n_prev, prep_prev.stats.ranges[2]),
                             prep_today.stats.ranges[2]))
    @test intersect(setdiff(1:n_prev, prep_prev.stats.ranges[2]),
                    prep_today.stats.ranges[2]) == [67, 68]

    # ---- 真实引擎：昨日解 → warm 携带昨日 alpha → checkpoint 分叉 ----
    root = PC.initialize_inference(prices[1:358, :]; F_folds = 3)  # ridge_alpha=nothing → EB 路径
    m_prev = PC.solve_current!(root)
    warm_yesterday = copy(root.warm)      # [full, fold1, fold2, fold3] 的 (αm, αr)
    # warm 真实携带昨日折解（copyto! 精确拷贝，bit 级相等）。
    for f in 1:3
        fm = m_prev.res_history.fold_models[f]
        @test warm_yesterday[f+1] == (fm.alpha_macro, fm.alpha_rel)
    end
    # warm 必须是非平凡初值，否则 warm/cold 对照退化为自比。
    @test warm_yesterday[3] != (1.0, 1.0)

    left = PC.inference_checkpoint(root)   # warm 路径：保留昨日 warm
    right = PC.inference_checkpoint(root)  # cold 路径：唯一差异 = 重置初值
    fill!(right.warm, (1.0, 1.0))
    PC.advance_exact!(left, view(prices, 359, :))
    PC.advance_exact!(right, view(prices, 359, :))
    m_warm = PC.solve_current!(left)
    m_cold = PC.solve_current!(right)

    # ---- 引擎层对照：同 bit 级今日 Grams，唯一差异 alpha_initial ----
    # 容差出处：两路各自收敛到 KKT 停止容差（rel_res/free_rms ≤ 1e-6）内的驻点；
    # alpha bracket tol 1e-9（conditioned）。超出即真分歧，不是 tie-break
    # （浅平坦时锚定几何中心，两路应严格相等）。
    @test m_warm.resp.alpha_macro ≈ m_cold.resp.alpha_macro rtol = 1e-5
    @test m_warm.resp.alpha_rel ≈ m_cold.resp.alpha_rel rtol = 1e-5
    @test m_warm.resp.Sigma_rel ≈ m_cold.resp.Sigma_rel rtol = 3e-5
    @test m_warm.resp.G_c_mean ≈ m_cold.resp.G_c_mean atol = 1e-7
    @test m_warm.resp.G_macro ≈ m_cold.resp.G_macro atol = 1e-7
    @test m_warm.mu_pred ≈ m_cold.mu_pred atol = 1e-8 rtol = 1e-8
    for f in 1:3
        fm_w = m_warm.res_history.fold_models[f]
        fm_c = m_cold.res_history.fold_models[f]
        @test fm_w.alpha_macro ≈ fm_c.alpha_macro rtol = 1e-5
        @test fm_w.alpha_rel ≈ fm_c.alpha_rel rtol = 1e-5
        @test fm_w.Sigma_rel ≈ fm_c.Sigma_rel rtol = 3e-5
        @test fm_w.G_c_mean ≈ fm_c.G_c_mean atol = 1e-7
    end
    # 端到端：OOF 残差（参数差的传播，容差同量级放宽）。
    @test PC.materialize(m_warm.res_history) ≈ PC.materialize(m_cold.res_history) atol = 1e-6 rtol = 1e-6

    # ---- EB 直调层（batch 今日折 2 Grams，同 Grams 不同 initial）----
    # 引擎走 fast path（contracted Grams），直调用 batch Grams：两层互相独立，
    # 各自在"同 Grams、唯一差异 initial"下对照。warm 初值取昨日折 2 的真实解。
    r2 = prep_today.stats.ranges[2]                       # 35:68
    train_idx = [1:first(r2)-1; last(r2)+1:prep_today.n_res]
    n_train = length(train_idx)
    S_xx2 = prep_today.stats.full_xx - prep_today.stats.xx[2]
    S_xy2 = prep_today.stats.full_xy - prep_today.stats.xy[2]
    S_yy2 = prep_today.stats.full_yy - prep_today.stats.yy[2]
    design2 = prep_today.X_rel_stacked[train_idx, :]      # n_train×P；N=3 时 primal
    spec2 = PC.ridge_spectrum(design2, S_xx2, S_xy2)
    Q3 = PC.relative_gauge(3)

    sol_w = PC.optimize_conditioned_eb(spec2, S_yy2, n_train;
        initial = warm_yesterday[3][2], gauge = Q3, return_certificate = true)
    sol_c = PC.optimize_conditioned_eb(spec2, S_yy2, n_train;
        initial = 1.0, gauge = Q3, return_certificate = true)
    @test sol_w.certificate.valid && sol_c.certificate.valid
    @test sol_w.alpha ≈ sol_c.alpha rtol = 1e-5
    @test sol_w.Sigma ≈ sol_c.Sigma rtol = 3e-5
    @test sol_w.certificate.evidence ≈ sol_c.certificate.evidence atol = 1e-7
    # covariance 证书一致 = 两路证书都 valid 且 Sigma 一致（证书是 Sigma 的
    # 确定函数）；分项 KKT 残差同量级。
    @test sol_w.certificate.covariance.valid && sol_c.certificate.covariance.valid
    @test sol_w.certificate.covariance.free_rms <= 1e-6
    @test sol_c.certificate.covariance.free_rms <= 1e-6

    # macro 侧（matrix-normal EB）同构对照：昨日折 2 的 alpha_macro 作 warm。
    train_ts2 = prep_today.ts_total[train_idx]
    Xm = prep_today.B_m[train_ts2, :]
    ym = prep_today.m[train_ts2 .+ 1]
    evm = eigen(Symmetric(Xm' * Xm))
    vals_m = max.(evm.values, 0.0)
    Bm_ = evm.vectors' * reshape(Xm' * ym, :, 1)
    Syy_m = fill(dot(ym, ym), 1, 1)
    mn_w = PC.optimize_matrix_normal_eb(vals_m, Bm_, Syy_m, n_train;
        initial = warm_yesterday[3][1])
    mn_c = PC.optimize_matrix_normal_eb(vals_m, Bm_, Syy_m, n_train; initial = 1.0)
    @test mn_w.certificate.valid && mn_c.certificate.valid
    @test mn_w.alpha ≈ mn_c.alpha rtol = 1e-7
    @test mn_w.certificate.evidence ≈ mn_c.certificate.evidence atol = 1e-10
end

# ===========================================================================
# [契约 2] ScenarioQuadrature（非 iid）：lazy oracle view vs dense view
# ===========================================================================

# NaN→0 mutant wrapper（与 residual_oracle_tests.jl 的手法同构、不同名，
# 避免同一 Main 命名空间下的 struct 重定义）。
struct PCNanToZero <: AbstractMatrix{Float64}
    inner::PC.ResidualOracle
end
Base.size(o::PCNanToZero) = size(o.inner)
Base.IndexStyle(::Type{PCNanToZero}) = Base.IndexCartesian()
Base.getindex(o::PCNanToZero, i::Integer, j::Integer) =
    let v = o.inner[i, j]; isnan(v) ? 0.0 : v; end

@testset "ScenarioQuadrature lazy vs dense: same model, same rule, ragged + missing own support" begin
    rng = MersenneTwister(2035)
    T, N = 600, 3
    P = exp.(cumsum(0.01 .* randn(rng, T, N); dims = 1))

    # ---- fixture A：raw ragged（资产 3 前 300 行缺失、资产 2 有洞、一行全缺）----
    Pr = Matrix{Float64}(P)
    Pr[1:300, 3] .= NaN
    Pr[400:403, 2] .= NaN
    Pr[500, :] .= NaN

    model = PC.fit_v1(Pr; F_folds = 3)                    # 生产 lazy 路径
    prep = PC._prepare_v1(Pr; F_folds = 3)
    fold_models = pc_refit_folds(prep; F_folds = 3)
    truth = PC.dense_oof_residuals(prep, fold_models)     # 独立 dense 真参考（现有函数）
    active = model.active_indices
    j3 = findfirst(==(3), active)
    @test j3 !== nothing
    @test !isempty(model.own_res_rows[j3])                # ragged 下 fallback 支持非空
    # 资产 3 在 truth 前段确有 unobserved 行（fallback 的触发前提）。
    @test all(isnan, truth[1:43, j3])

    dense_model = pc_swap_res(model, Matrix{Float64}(truth))
    rule = PC.ScenarioQuadrature(2N + 3, MersenneTwister(2036))  # 同一 rule 对象，两 view 共用

    # ---- 确定性预检：own-fallback 边界被真实进入（非概率论证）----
    # 用生产原语（quadrature_uniform + observed）复算 pass 1 的 shared-row
    # 选择：至少一个 scenario 落入资产 3 的 unobserved 行。oracle 与 dense 的
    # 观测 mask 逐格一致（下一条断言），故该预检对两 view 同时成立。
    pre = pc_shared_rows(model, rule, 128, j3)
    @test pre.hits >= 1
    @test isfinite.(PC.materialize(model.res_history)) == isfinite.(truth)

    Xq_lazy = PC.generate_scenarios_v1(model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule)
    Xq_dense = PC.generate_scenarios_v1(dense_model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule)

    # quadrature 模式下 rng 不被消耗（uniforms=nothing），两 view 传同 seed
    # 只为强调唯一差异是 view；跨 view 一致性在 roundoff 尺度（求和顺序不同，
    # 与既有 flow 契约同语义，不宣称跨 view bit 级）。
    @test isnan.(Xq_lazy) == isnan.(Xq_dense)
    msk = .!isnan.(Xq_lazy)
    @test Xq_lazy[msk] ≈ Xq_dense[msk] atol = 1e-10 rtol = 1e-8
    # 同 view 同 rule 同 S 的 replay：bit 级（oracle 无 cache、model 不被 mutate）。
    @test isequal(PC.generate_scenarios_v1(model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule), Xq_lazy)
    @test isequal(PC.generate_scenarios_v1(dense_model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule), Xq_dense)

    # 预检已确定性验证 hits >= 1（fallback 被真实进入）；此处断言其后果：
    # 被进入的 own-fallback 产出有限值，无 NaN 泄漏。
    @test all(isfinite, Xq_lazy[:, active[j3]])
    @test all(isfinite, Xq_dense[:, active[j3]])

    # ---- Kelly 对照 ----
    free = trues(model.N_universe)
    w_lazy = PC.scenario_weights(Xq_lazy, active, free)
    w_dense = PC.scenario_weights(Xq_dense, active, free)
    @test sum(w_lazy) ≈ 1.0 atol = 1e-10
    @test sum(w_dense) ≈ 1.0 atol = 1e-10
    @test w_lazy ≈ w_dense atol = 1e-6 rtol = 1e-6
    c_lazy = PC.kelly_certificate(Xq_lazy[:, active], w_lazy[active])
    c_dense = PC.kelly_certificate(Xq_dense[:, active], w_dense[active])
    @test c_lazy.objective_gap <= 1e-8
    @test c_dense.objective_gap <= 1e-8
    @test c_lazy.feasibility <= 1e-10
    @test c_dense.feasibility <= 1e-10

    # ---- mutant 1：错 fold 划分（ranges 移一行，仍完整覆盖 1:n_res）----
    wrong_ranges = [1:113, 114:227, 228:343]
    wrong_oracle = PC.ResidualOracle(fold_models, wrong_ranges, prep.ts_total,
                                     prep.r, prep.s1, prep.s_perp, prep.B_m, prep.X_rel)
    wrong_model = pc_swap_res(model, wrong_oracle)   # own_res_rows 保持正确（与 fold 无关）
    Xq_wrong = PC.generate_scenarios_v1(wrong_model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule)
    @test !isapprox(Xq_wrong, Xq_lazy; atol = 1e-6, rtol = 1e-6)

    # ---- mutant 2：NaN→0（unobserved 残差当 0 值：跳过 fallback、值被改写）----
    n2z_model = pc_swap_res(model, PCNanToZero(model.res_history))
    Xq_n2z = PC.generate_scenarios_v1(n2z_model; S = 128,
        rng = MersenneTwister(2037), quadrature = rule)
    @test !isapprox(Xq_n2z, Xq_lazy; atol = 1e-6, rtol = 1e-6)

    # ---- fixture B：missing own support（active 但 own_res_rows 为空）----
    # 精确构造（可拟合 + 目标 support 空）：资产 3 在 1:255 有完整价格史
    # （ruler 正常幂律拟合、设计侧 X'X 的资产 3 列块在 WARMUP 后的窗口内
    # 有真实能量——Gram 不病态），从 256 起全部消失。由此：
    # (a) 资产 3 仍 active（r[254]=logP[255]-logP[254] 有限）；
    # (b) 全部 OOF 标签行（257:599）的资产 3 return 均为 NaN——r[k] 有限
    #     需要 P[k]、P[k+1] 都有限，而有限价格止于 255 → own[3] = ∅；
    # (c) 目标侧（标签 embedding）support 恰好为空——这是契约要的边界本身，
    #     由 EB floor 的单边 KKT 合法驻点接住（conditioned_eb_tests 的 floor
    #     激活用例已锁定该收敛路径），不是需要放宽证书的病态。
    # 构造性必命中：全部合法 rows 都未观测（下方预检断言逐行验证），任何
    # shared row 都进入 own-fallback；own 为空 → 抛错是必然，与 rule/seed/S
    # 无关。lazy view 与 dense view 的错误语义必须一致。
    # 失败处置纪律：若 fit 在 EB 处抛错，处置是继续精确修 fixture 的可拟合
    # 性（例如延长资产 3 的活跃段）或上报实现缺陷——不放宽证书、不换 seed。
    Pm = Matrix{Float64}(P)
    Pm[256:end, 3] .= NaN
    model_m = PC.fit_v1(Pm; F_folds = 3)              # fit 本身必须能完成
    @test isempty(model_m.own_res_rows[j3])
    # 确定性预检：资产 3 的全部合法 residual 行都未观测（构造性全行覆盖，
    # 非概率命中）——用生产 observed 原语逐行验证。
    @test all(idx -> !PC.observed(model_m.res_history, idx, j3),
              1:size(model_m.res_history, 1))
    prep_m = PC._prepare_v1(Pm; F_folds = 3)
    fold_models_m = pc_refit_folds(prep_m; F_folds = 3)
    truth_m = PC.dense_oof_residuals(prep_m, fold_models_m)
    @test all(isnan, truth_m[:, j3])                  # dense 参考同样整列未观测
    rule_m = PC.ScenarioQuadrature(2N + 3, MersenneTwister(2038))
    @test_throws ErrorException PC.generate_scenarios_v1(model_m; S = 64,
        rng = MersenneTwister(2039), quadrature = rule_m)
    dense_m = pc_swap_res(model_m, Matrix{Float64}(truth_m))
    @test_throws ErrorException PC.generate_scenarios_v1(dense_m; S = 64,
        rng = MersenneTwister(2039), quadrature = rule_m)
end
