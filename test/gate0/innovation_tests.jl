# =============================================================================
# KTraderGate0 vector innovation reference — Step 10/11/12 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/innovation.jl（InnovationState / V_t(d) / ℓ_d /
# q(d|H) / Moore-Penrose support 协议 / standardized empirical shape /
# joint row 规则 / T4 覆盖不足处理）。
#
# 覆盖组织：docs/GATE0_VECTOR_INNOVATION.md §9 实现者检查清单八类
# （1 propriety/support、2 floor→0 refinement、3 joint row 兼容性、
# 4 因果性、5 极限对照、6 协变性、7 shape 语义、8 接口）+ 施工图
# Step 10/11/12 段与终审裁决 G6 的补充（D-095 vector volatility world、
# stitching 反例、N_R=1 锚定分离报告、dummy 逐字节不变、T4 覆盖不足、
# cell mass 先验质量不漂移）。
#
# 纪律：
# - tiny 级（D-089 阶梯：static → tiny analytic → …）：合成数据、固定
#   seed（MersenneTwister(20261009)）、无外部依赖、无 I/O、无回测；
# - 本任务不运行测试（Engineer 不执行命令；运行验证由 Manager 安排
#   DevOps 受控执行，≤60s/RSS 护栏）；
# - 全部断言与回测收益/Sharpe 无关（SPEC §95 / 开发守则 §24）；
# - 加载方式：临时 module include 三个裸文件（geometry → modes →
#   innovation，依赖链顺序；主 module include 骨架归 DevOps 验证轮）。

using Test, Random, LinearAlgebra, Statistics

module Gate0InnovationEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "innovation.jl"))
end
using .Gate0InnovationEnv

# 测试内独立手算工具（naive 双构造 / 解析极限——与实现不同路径）：
# 手算 V：直接 kernel 加权二阶矩（用 export 的 frac_weights_gate0，
# 独立循环书写）。
function naive_V(epsR::Matrix{Float64}, uJ::Vector{Int}, d::Float64, tnow::Int)
    N_R = size(epsR, 2)
    w = frac_weights_gate0(d, tnow + 1 - uJ[1])
    V = zeros(N_R, N_R)
    den = 0.0
    for i in eachindex(uJ)
        τ = tnow + 1 - uJ[i]
        den += w[τ]
        for a in 1:N_R, b in 1:N_R
            V[a, b] += w[τ] * epsR[i, a] * epsR[i, b]
        end
    end
    V ./ den
end
# 样本峰度（四阶矩 / σ⁴，高斯 = 3；非 excess）。
function kurt(v::AbstractVector{Float64})
    m = mean(v)
    s2 = mean(abs2.(v .- m))
    s2 > 0 || return NaN
    mean((v .- m) .^ 4) / (s2 * s2)
end

@testset "gate0 Step 10-12: vector innovation reference" begin
    SEED = 20261009
    rng = MersenneTwister(SEED)

    # -------------------------------------------------------------------
    # 主 fixture：N_a=5、120 行 OOF 残差（asset 空间形态——含 b₀ 扣除
    # 语义由调用方保证，本测试模拟 response 层输出契约）、完整 panel。
    # row_ids = 目标日 257..376（严格递增）；t=376；R_t=[1,2,3,4]。
    # -------------------------------------------------------------------
    N_a = 5
    n_rows = 120
    row_ids = collect(257:376)
    t_dec = 376
    R_t = [1, 2, 3, 4]
    E_active = domain_mode_basis(N_a)
    eps_tilde = 0.01 .* randn(rng, n_rows, N_a)
    masks_full = [trues(N_a) for _ in 1:n_rows]
    st = innovation_state(eps_tilde, row_ids, masks_full, R_t, E_active; t = t_dec)

    # ===================================================================
    # 9.1 propriety / support（VI §9 检查 1）
    # ===================================================================
    @testset "9.1 propriety / support" begin
        # |J_t|=0 → error（文本含 innovation coverage failure）
        masks_no4 = [begin m = trues(N_a); m[4] = false; m end for _ in 1:n_rows]
        err0 = try
            innovation_state(eps_tilde, row_ids, masks_no4, R_t, E_active; t = t_dec)
            nothing
        catch e
            e
        end
        @test err0 isa ErrorException
        @test occursin("innovation coverage failure", err0.msg)

        # |J_t|=1（V 可定义但 |L|=0，ℓ 无信息）→ error（不得静默均匀先验）
        masks_one4 = [begin
                          m = trues(N_a)
                          m[4] = i == 60   # 仅行 60 覆盖资产 4
                          m
                      end for i in 1:n_rows]
        err1 = try
            innovation_state(eps_tilde, row_ids, masks_one4, R_t, E_active; t = t_dec)
            nothing
        catch e
            e
        end
        @test err1 isa ErrorException
        @test occursin("innovation coverage failure", err1.msg)

        # rank deficient：|J|=3 < N_R=4 → rank ≤ 3；null 方向不逆不注噪不造风险
        masks_rd = [begin
                        m = trues(N_a)
                        m[4] = i in (30, 60, 90)   # J = {30,60,90}
                        m
                    end for i in 1:n_rows]
        st_rd = innovation_state(eps_tilde, row_ids, masks_rd, R_t, E_active; t = t_dec)
        @test length(st_rd.J_t) == 3
        @test all(r -> r <= 3, st_rd.support_cert.ranks)          # rank ≤ |J|
        # 秩引理：不同 d 的 rank 相同（VI §2.3——rank 与 d 无关）
        @test length(unique(st_rd.support_cert.ranks)) == 1
        @test length(unique(st.support_cert.ranks)) == 1          # 完整 fixture 同
        # 无 δI 注入：最小特征值为数值零量级（非任何理论 floor）
        λmin = minimum(mp_sqrt_factors(st_rd.V_t[3]).eigenvalues)
        @test λmin < 1e-10
        # null 方向不逆：z ∈ range(V_{s-1})（pinv(V)V = P_s，P_s·z = z）
        Z_rd = z_pool_rows(st_rd, st_rd.d_nodes[3])
        for i in 1:size(Z_rd, 1)
            Vsm = naive_V(st_rd.eps_R[1:st_rd.L_t[i]-1, :],
                          st_rd.row_ids[st_rd.J_t][1:st_rd.L_t[i]-1],
                          st_rd.d_nodes[3], st_rd.row_ids[st_rd.J_t][st_rd.L_t[i]] - 1)
            @test isapprox(pinv(Vsm) * Vsm * Z_rd[i, :], Z_rd[i, :], atol = 1e-10)
        end
        # 预测创新在 null 方向 = 0：ε = V^{1/2}z ∈ range(V_t)
        r_draw = draw_innovation(st_rd, MersenneTwister(7))
        Vd = st_rd.V_t[findfirst(==(r_draw.d), st_rd.d_nodes)]
        @test isapprox(pinv(Vd) * Vd * r_draw.eps, r_draw.eps, atol = 1e-10)
    end

    # ===================================================================
    # 9.2 floor→0 refinement（VI §9 检查 2）
    # ===================================================================
    @testset "9.2 floor→0 refinement" begin
        floors = [1e-8, 5e-9, 1e-10, 1e-12]
        # 可逆 fixture：floor 减半序列下 z 收敛（正特征远大于 floor）
        Zs = [z_pool_rows(innovation_state(eps_tilde, row_ids, masks_full,
                                           R_t, E_active; t = t_dec, cls_floor = f),
                          0.4) for f in floors]
        for k in 2:length(floors)
            @test isapprox(Zs[k], Zs[1], atol = 1e-14)
        end
        # rank deficient fixture：null 特征（数值 ~1e-17）在全部 floor 下
        # 被分类为零——z 收敛（floor→0 refinement 不改变分类结果）
        masks_rd = [begin
                        m = trues(N_a)
                        m[4] = i in (30, 60, 90)
                        m
                    end for i in 1:n_rows]
        Zrds = [z_pool_rows(innovation_state(eps_tilde, row_ids, masks_rd,
                                             R_t, E_active; t = t_dec, cls_floor = f),
                            0.4) for f in floors]
        for k in 2:length(floors)
            @test isapprox(Zrds[k], Zrds[1], atol = 1e-14)
        end
        # V 的数学定义不含 δ：V_direct 无 floor 参数（构造性——签名层面）
        @test !occursin("cls_floor", string((Base.methods(Gate0InnovationEnv.V_direct))))
    end

    # ===================================================================
    # 9.3 joint row 兼容性 / mask 判定（VI §9 检查 3 + G6）
    # ===================================================================
    @testset "9.3 joint row / stitching 反例" begin
        # ragged 行（O_s ⊉ R_t）不进 J：行 50 缺资产 3
        masks_r = [begin m = trues(N_a); i == 50 && (m[3] = false); m end
                   for i in 1:n_rows]
        st_r = innovation_state(eps_tilde, row_ids, masks_r, R_t, E_active; t = t_dec)
        @test 50 ∉ st_r.J_t
        @test length(st_r.J_t) == n_rows - 1
        # 空观察行（O_s = ∅）不进任何统计（不是零残差）
        masks_e = deepcopy(masks_full)
        masks_e[70] .= falses(N_a)
        st_e = innovation_state(eps_tilde, row_ids, masks_e, R_t, E_active; t = t_dec)
        @test 70 ∉ st_e.J_t
        # R_t 收缩（剔除 free 资产）→ 行集恢复（D-056 出路可兑现）
        masks_one4 = [begin
                          m = trues(N_a)
                          m[4] = i in (30, 60, 90)
                          m
                      end for i in 1:n_rows]
        st_R4 = innovation_state(eps_tilde, row_ids, masks_one4, R_t, E_active; t = t_dec)
        st_R3 = innovation_state(eps_tilde, row_ids, masks_one4, [1, 2, 3],
                                 E_active; t = t_dec)
        @test length(st_R4.J_t) == 3
        @test length(st_R3.J_t) == n_rows          # 剔除资产 4 后全行恢复
        # stitching 拼接向量反例：pool 行 = 单一历史行的确定性变换
        # （集合精确对应 {V^{+1/2}ε_s}——拼接向量构造性不在 pool）
        Z = z_pool_rows(st, 0.4)
        @test size(Z, 1) == length(st.L_t)
        for i in 1:size(Z, 1)
            Vsm = naive_V(st.eps_R[1:st.L_t[i]-1, :],
                          st.row_ids[st.J_t][1:st.L_t[i]-1],
                          0.4, st.row_ids[st.J_t][st.L_t[i]] - 1)
            Fh = mp_sqrt_factors(Vsm)
            @test isapprox(Z[i, :], Fh.inv_half * st.eps_R[st.L_t[i], :], atol = 1e-12)
        end
        z_stitch = vcat(Z[2, 1:2], Z[3, 3:4])      # 手工跨行拼接
        @test all(i -> !isapprox(Z[i, :], z_stitch, atol = 1e-6), 1:size(Z, 1))
    end

    # ===================================================================
    # 9.4 因果性（VI §9 检查 4 / §7 三定理）
    # ===================================================================
    @testset "9.4 causality / replay" begin
        # 未来行注入（u_s > t）：构造性不进 J（V_t 逐字节不变）
        ids_f = vcat(row_ids, 400)
        eps_f = vcat(eps_tilde, 0.01 .* randn(rng, 1, N_a))
        masks_f = vcat(masks_full, [trues(N_a)])
        st_f = innovation_state(eps_f, ids_f, masks_f, R_t, E_active; t = t_dec)
        @test length(st_f.J_t) == length(st.J_t)
        @test st_f.eps_R == st.eps_R
        @test st_f.V_t == st.V_t
        # t 截断：t=300 → J 只含 u ≤ 300 的行（44 行）
        st_300 = innovation_state(eps_tilde, row_ids, masks_full, R_t, E_active; t = 300)
        @test length(st_300.J_t) == 300 - 257 + 1
        # V_direct 因果防御：u > 决策时刻 → error
        @test_throws ErrorException Gate0InnovationEnv.V_direct(
            st.eps_R, st.row_ids[st.J_t] .+ 0, 0.4, row_ids[1] - 1)
        # 同 seed 重放：d 抽样 + 行抽样 + ε 逐位一致（SPEC §36 / 裁决 F1）
        r1 = draw_innovation(st, MersenneTwister(42))
        r2 = draw_innovation(st, MersenneTwister(42))
        @test r1.eps == r2.eps
        @test r1.d == r2.d
        @test r1.z_row == r2.z_row
        @test r1.z_row_id == r2.z_row_id
    end

    # ===================================================================
    # 9.5 极限对照（VI §9 检查 5；N_R=1 锚定分离报告——D-059）
    # ===================================================================
    @testset "9.5 limit / N_R=1 / anchor removal" begin
        # d=1：joint 合法行等权二阶矩（解析可验）
        nJ = length(st.J_t)
        V_eq = (st.eps_R' * st.eps_R) ./ nJ
        @test isapprox(V_at(st, 1.0), V_eq, atol = 1e-12)
        # d→0⁺ 极限（Manager 裁决 Wave 3 执行轮：VI §2.5「调和衰减」声明
        # 是推导文档错误，勘误已由 Manager 登记）：w[τ] = k_d(τ) = π_{τ−1}、
        # π₀ = 1 恒定不随 d 变——d→0 时 τ=1（最近合法行）权重独占
        # （其余 w[τ] ∝ 1/Γ(d)·1/(τ−1) → 0），真极限 = 最近行外积。
        eps_last = st.eps_R[end, :]                 # J 的最后一行（τ = 1）
        @test isapprox(V_at(st, 1e-10), eps_last * eps_last', atol = 1e-12)
        # 权重侧证据：π₀（= w[1]）对其余权重和的比值随 d 减小爆炸性增长
        # （w[τ≥2] ∝ d/(τ−1)：d 缩小 100 倍 ⇒ 比值 ~×100）——「最近行
        # 独占」而非调和衰减。
        w_a = frac_weights_gate0(1e-8, nJ)
        w_b = frac_weights_gate0(1e-10, nJ)
        ra = w_a[1] / sum(w_a[2:end])
        rb = w_b[1] / sum(w_b[2:end])
        @test rb > 50 * ra
        # 完整 panel：分母 = 行集上 kernel 和（裁决书 D-045 字面形式成立
        # ——连续目标日使全部 lag 各出现一次，分母 = sum(w)）
        d0 = 0.6
        w = frac_weights_gate0(d0, t_dec + 1 - row_ids[1])
        V_hand = zeros(4, 4)
        den_hand = 0.0
        for i in 1:n_rows
            τ = t_dec + 1 - row_ids[i]
            den_hand += w[τ]
            V_hand .+= w[τ] .* (st.eps_R[i, :] * st.eps_R[i, :]')
        end
        V_hand ./= den_hand
        @test isapprox(V_at(st, d0), V_hand, atol = 1e-14)
        @test isapprox(den_hand, sum(frac_weights_gate0(d0, 120)), atol = 1e-12)
        # N_R=1 标量对照（kernel 部分）：E_R=[1] 恒等 → V = Σw ε²/Σw
        st1 = innovation_state(eps_tilde, row_ids, masks_full, [2], E_active; t = t_dec)
        @test st1.N_R == 1
        num = 0.0
        den = 0.0
        for i in 1:n_rows
            τ = t_dec + 1 - row_ids[i]
            num += w[τ] * st1.eps_R[i, 1]^2
            den += w[τ]
        end
        @test isapprox(V_at(st1, d0)[1, 1], num / den, atol = 1e-14)
        # 锚定分离报告（D-059）：v_bootstrap/v_forecasts 通道整体删除
        # ——构造性反断言（kernel 一致 ≠ 整体数值相等：旧线 scale_d =
        # sqrt(v/v_bootstrap) 通道在本层不存在，绝对量纲由 V^{1/2} 携带）
        fnames = fieldnames(InnovationState)
        @test :v_bootstrap ∉ fnames
        @test :v_forecasts ∉ fnames
    end

    # ===================================================================
    # 9.6 协变性（VI §9 检查 6 / §2.6）
    # ===================================================================
    @testset "9.6 covariance invariances" begin
        # w 作用域修复（Wave 3 验证轮）：原测试体引用 9.5 testset 的局部
        # w（跨 testset 变量泄漏——9.5 内定义、9.6 未定义即 UndefVarError）。
        # 本 testset 开头按同款 d0=0.6 重建。
        d0 = 0.6
        w = frac_weights_gate0(d0, t_dec + 1 - row_ids[1])
        # mode↔asset 一致（gauge 不变的可测形式）：E_R·V·E_Rᵀ = asset
        # 坐标加权二阶矩——任何正交 E_R' 给出同一 asset 对象（SPEC §59）
        W_asset = zeros(length(R_t), length(R_t))
        den_w = 0.0
        for i in 1:n_rows
            τ = t_dec + 1 - row_ids[i]
            e = eps_tilde[i, R_t]
            W_asset .+= w[τ] .* (e * e')
            den_w += w[τ]
        end
        W_asset ./= den_w
        @test isapprox(st.E_R * V_at(st, 0.6) * st.E_R', W_asset, atol = 1e-12)
        # 资产置换：universe 列置换 Π → V_asset 协变 Π·V·Πᵀ（SPEC §58）
        perm = [3, 1, 4, 2, 5]
        eps_p = eps_tilde[:, perm]
        masks_p = [m[perm] for m in masks_full]
        R_p = sort(perm[R_t])
        st_p = innovation_state(eps_p, row_ids, masks_p, R_p, E_active; t = t_dec)
        Va1 = st.E_R * V_at(st, 0.6) * st.E_R'
        Va2 = st_p.E_R * V_at(st_p, 0.6) * st_p.E_R'
        # 索引修复（Wave 3 验证轮）：原 Va1[perm, perm] 用 5 维 universe 置换
        # 索引 4×4 的 R 域 asset 矩阵（BoundsError）。R 域内的置换像 =
        # perm[R_t]（4 维：Va2 的 (k,l) = 资产 perm[R_t[k]] 与 perm[R_t[l]]
        # 的矩 = Va1[perm[R_t], perm[R_t]] 的对应元）。
        @test isapprox(Va2, Va1[perm[R_t], perm[R_t]], atol = 1e-12)
        # dummy 全 NaN 资产（不在 R_t）：V_t 逐字节不变（SPEC §60）
        eps_d = hcat(eps_tilde, fill(NaN, n_rows, 1))
        masks_d = [falses(N_a + 1) |> m -> begin m[1:N_a] .= masks_full[i]; m end
                   for i in 1:n_rows]
        st_d = innovation_state(eps_d, row_ids, masks_d, R_t,
                                domain_mode_basis(N_a + 1); t = t_dec)
        @test all(i -> st_d.V_t[i] == st.V_t[i], eachindex(st.d_nodes))
        r_d1 = draw_innovation(st, MersenneTwister(11))
        r_d2 = draw_innovation(st_d, MersenneTwister(11))
        @test r_d1.eps == r_d2.eps
    end

    # ===================================================================
    # 9.7 shape 语义（VI §9 检查 7 + 裁决 G6）
    # ===================================================================
    @testset "9.7 shape semantics" begin
        # M_z ≈ I（iid 高斯 + 可逆 V 的 fixture；诊断近似——VI §5.4：
        # 协议不假设 E[zzᵀ] = I，V_{s−1} 漂移使 M_z ≈ P_s 而非恒等）。
        # Manager 裁决（Wave 3 执行轮）：早期行近奇异 V_{s−1} 的 z 放大
        # （弱方向 1/√λ——D-049 无 burn 协议的真实数学行为，非缺陷）
        # 使全样本 M_z/kurt 重尾；**紧界断言在充分行数子集上验证**
        # （剔除前 max(20, 4·N_R) 行——早期 V_{s−1} 的近奇异窗口）；
        # VI 协议的 L 行集不排除早期行——保持原样。
        n_big = 400
        ids_big = collect(257:256 + n_big)
        eps_big = 0.01 .* randn(MersenneTwister(SEED + 1), n_big, N_a)
        masks_big = [trues(N_a) for _ in 1:n_big]
        st_big = innovation_state(eps_big, ids_big, masks_big, R_t, E_active;
                                  t = 256 + n_big)
        skip = max(20, 4 * length(R_t))
        Zsub = z_pool_rows(st_big, 0.4)[skip + 1:end, :]
        Mzsub = (Zsub' * Zsub) ./ size(Zsub, 1)
        @test all(diag(Mzsub) .> 0.7) && all(diag(Mzsub) .< 1.3)
        @test maximum(abs.(Mzsub - Diagonal(diag(Mzsub)))) < 0.2
        # tail 保留：重尾混合（10% ×10）→ z pool 峰度远超高斯；
        # 对照：纯高斯 fixture 峰度 ≈ 3（无 Gaussian 化、无 Student-t 参数）
        # ——同口径（充分行数子集）
        eps_tail = 0.01 .* randn(MersenneTwister(SEED + 2), n_big, N_a)
        heavy = rand(MersenneTwister(SEED + 3), n_big) .< 0.1
        eps_tail[heavy, :] .*= 10.0
        st_tail = innovation_state(eps_tail, ids_big,
                                   [trues(N_a) for _ in 1:n_big], R_t, E_active;
                                   t = 256 + n_big)
        Zt = z_pool_rows(st_tail, 0.4)[skip + 1:end, :]
        @test kurt(Zt[:, 1]) > 6
        @test kurt(Zt[:, 3]) > 6
        @test 2.0 < kurt(Zsub[:, 1]) < 4.5
        # cross-mode shock：联合行抽样的分量间经验相关保留（无拼接破坏）
        # ——mode 1/2 构造相关 0.8 的残差，draw 大样本协方差 ≈ 混合理论矩
        # E[V^{1/2} M_z V^{1/2}]（完整协议下联合结构保留的可测断言）
        eps_m = 0.01 .* randn(MersenneTwister(SEED + 4), n_big, 4)  # mode 行
        eps_m[:, 2] = 0.8 .* eps_m[:, 1] .+ 0.6 .* 0.01 .*
                      randn(MersenneTwister(SEED + 5), n_big)
        E_R4 = risk_domain_mode_basis(R_t)
        eps_cs = zeros(n_big, N_a)
        eps_cs[:, R_t] = (E_R4 * eps_m')'      # mode → asset（R 坐标嵌入）
        st_cs = innovation_state(eps_cs, ids_big, [trues(N_a) for _ in 1:n_big],
                                 R_t, E_active; t = 256 + n_big)
        theory = zeros(4, 4)
        for g in eachindex(st_cs.d_nodes)
            Fg = mp_sqrt_factors(st_cs.V_t[g])
            theory .+= st_cs.d_weights[g] .* (Fg.half * shape_moments(st_cs, st_cs.d_nodes[g]) * Fg.half)
        end
        rng_cs = MersenneTwister(SEED + 6)   # 共享 rng：每次重建同 seed 会使 4000 个 draw 全同
        draws = [draw_innovation(st_cs, rng_cs).eps for _ in 1:4000]
        # stack 形态修复（Wave 3 验证轮）：原 cov(hcat(draws...)) 把
        # 4000 个 4 维向量 hcat 成 4×4000——cov 按列得 4000×4000
        # （DimensionMismatch 实证）。stack(dims=1) 得 4000×4（观测×变量）
        # → cov/cor 为 4×4（posterior_tests 的 draw_mu 同款正确写法）。
        S_emp = cov(stack(draws; dims = 1))
        @test opnorm(S_emp .- theory, Inf) < 0.2 * opnorm(theory, Inf)
        # 相关结构本身非零（fixture 有效性：mode 1/2 相关传播到 ε）
        c_emp = cor(stack(draws; dims = 1))
        @test abs(c_emp[1, 2]) > 0.5
    end

    # ===================================================================
    # 9.8 ℓ_d 手算对照 / cell mass / d_weights（VI §9 检查 8 + Step 12）
    # ===================================================================
    @testset "9.8 quasi-likelihood / cell mass" begin
        # 小 fixture naive 双构造：ℓ_d 与独立路径（eigen + 手动 MP）一致
        n_sm = 8
        ids_sm = collect(257:264)
        eps_sm = 0.01 .* randn(MersenneTwister(SEED + 7), n_sm, N_a)
        masks_sm = [trues(N_a) for _ in 1:n_sm]
        st_sm = innovation_state(eps_sm, ids_sm, masks_sm, [1, 2], E_active; t = 264)
        uJ = st_sm.row_ids[st_sm.J_t]
        for d in (0.2, 0.6, 1.0)
            ll_naive = 0.0
            for i in 2:length(uJ)
                Vsm = naive_V(st_sm.eps_R[1:i-1, :], uJ[1:i-1], d, uJ[i] - 1)
                F = eigen(Symmetric(Vsm))
                fl = 2 * eps(Float64) * max(F.values[end], floatmin(Float64))
                λp = F.values[F.values .> fl]
                εv = st_sm.eps_R[i, :]
                quad = sum((F.values[j] > fl) ? (dot(F.vectors[:, j], εv)^2 / F.values[j]) : 0.0
                           for j in eachindex(F.values))
                ll_naive -= 0.5 * (sum(log, λp) + quad)
            end
            @test isapprox(quasi_loglik(st_sm.eps_R, uJ, d), ll_naive, atol = 1e-10)
        end
        # d_weights 归一（q_g ∝ exp(ℓ)·Δ_g，η≡1）
        @test isapprox(sum(st.d_weights), 1.0, atol = 1e-12)
        # cell mass 先验质量不漂移（SPEC §64）：全同行 fixture → ℓ_d 常数
        # （V_{s−1} = ε₀ε₀ᵀ 与权重无关）→ d_weights ∝ Δ_g
        eps_const = repeat(0.01 .* randn(MersenneTwister(SEED + 8), 1, N_a), n_sm, 1)
        st_c = innovation_state(eps_const, ids_sm, masks_sm, [1, 2], E_active; t = 264)
        @test isapprox(st_c.d_weights, st_c.d_cells, rtol = 1e-9)
        # 默认网格：节点 ∈ (0,1]、cell 覆盖 (0,1]、Σ = 1
        nodes0, cells0 = default_d_grid()
        @test all(0 .< nodes0 .<= 1)
        @test all(cells0 .> 0)
        @test isapprox(sum(cells0), 1.0, atol = 1e-12)
    end

    # ===================================================================
    # 9.9 覆盖不足 T4 判定次序（D-056 / 裁决 C3；Step 11 ragged 4-6）
    # ===================================================================
    @testset "9.9 coverage / T4" begin
        # 资产 5 仅行 60 覆盖
        masks_cov = [begin
                         m = trues(N_a)
                         m[5] = i == 60
                         m
                     end for i in 1:n_rows]
        # (a) free 覆盖不足 → 剔除（覆盖最少者优先）、R 收缩、行集恢复
        R_res, dropped = resolve_risk_domain(row_ids, masks_cov, [1, 2, 5], Int[], t_dec)
        @test R_res == [1, 2]
        @test dropped == [5]
        # (b) locked 覆盖不足 → fail loudly（文本含 innovation coverage failure）
        err_lock = try
            resolve_risk_domain(row_ids, masks_cov, Int[], [5], t_dec)
            nothing
        catch e
            e
        end
        @test err_lock isa ErrorException
        @test occursin("innovation coverage failure", err_lock.msg)
        # free 剔尽且 locked 空 → 空 R_t（cash 出路，D-056 出路 4）
        R_cash, dropped_cash = resolve_risk_domain(row_ids, masks_cov, [5], Int[], t_dec)
        @test isempty(R_cash)
        @test dropped_cash == [5]
        # coverage_report 诊断字段
        rep = coverage_report(row_ids, masks_cov, [1, 2, 5], t_dec)
        @test rep.n_joint == 1
        @test rep.per_asset_rows == [n_rows, n_rows, 1]
        @test rep.coverage_gap == [5]
        @test rep.sufficient == false
    end

    # ===================================================================
    # 9.10 D-095 vector volatility world（施工图 Step 10 测试）
    # ===================================================================
    @testset "9.10 D-095 vector volatility world" begin
        # 后半段 mode-2（relative 方向 q₁）波动 ×5：V_t 只放大该方向
        # （对照：旧 scalar law 会把 scale 乘全场——vector law 的结构性
        # 新增信息；VI §2.4 mode 风险轮动）
        n_v = 200
        ids_v = collect(257:256 + n_v)
        eps_v = 0.01 .* randn(MersenneTwister(SEED + 9), n_v, 4)
        eps_v[div(n_v, 2) + 1:end, 2] .*= 5.0
        E_R4 = risk_domain_mode_basis(R_t)
        eps_vasset = zeros(n_v, N_a)
        eps_vasset[:, R_t] = (E_R4 * eps_v')'
        st_v = innovation_state(eps_vasset, ids_v, [trues(N_a) for _ in 1:n_v],
                                R_t, E_active; t = 256 + n_v)
        Vv = V_at(st_v, 0.2)                     # 短记忆：后半段主导
        others = [Vv[1, 1], Vv[3, 3], Vv[4, 4]]
        @test Vv[2, 2] > 4 * maximum(others)
        @test all(others .< 2.0 * 1e-4)          # 其他方向 ≈ 原方差量级
    end

    # ===================================================================
    # 9.11 接口 fail loudly（VI §1.2 契约 5 / SPEC §56）
    # ===================================================================
    @testset "9.11 interface fail-loudly" begin
        # J 行的 R 子向量含 NaN → error（非 J 行的 NaN 由 mask 表达，合法）
        eps_nan = copy(eps_tilde)
        eps_nan[10, 3] = NaN
        @test_throws ErrorException innovation_state(eps_nan, row_ids, masks_full,
                                                      R_t, E_active; t = t_dec)
        @test innovation_state(eps_nan, row_ids,
                               [begin m = trues(N_a); i == 10 && (m[3] = false); m end
                                for i in 1:n_rows],
                               R_t, E_active; t = t_dec) isa InnovationState
        # 维度/参数错误
        @test_throws DimensionMismatch innovation_state(eps_tilde, row_ids, masks_full,
                                                         R_t, domain_mode_basis(4))
        masks_bad = [trues(N_a - 1) for _ in 1:n_rows]
        @test_throws DimensionMismatch innovation_state(eps_tilde, row_ids, masks_bad,
                                                         R_t, E_active)
        @test_throws ArgumentError innovation_state(eps_tilde, row_ids, masks_full,
                                                     [0, 2], E_active)
        @test_throws ArgumentError innovation_state(eps_tilde, row_ids, masks_full,
                                                     [1, 6], E_active)
        ids_bad = copy(row_ids); ids_bad[5] = ids_bad[4]
        @test_throws ErrorException innovation_state(eps_tilde, ids_bad, masks_full,
                                                     R_t, E_active)
        @test_throws DomainError innovation_state(eps_tilde, row_ids, masks_full,
                                                  R_t, E_active; d_nodes = [0.0, 0.5],
                                                  d_cells = [0.5, 0.5])
        @test_throws ErrorException innovation_state(eps_tilde, row_ids, masks_full,
                                                     R_t, E_active; d_nodes = [0.3, 0.7],
                                                     d_cells = [0.5, 0.4])
        @test_throws ErrorException innovation_state(eps_tilde, row_ids, masks_full,
                                                     Int[], E_active)
        # frac_weights 域：d ∉ (0,1] → DomainError
        @test_throws DomainError frac_weights_gate0(0.0, 5)

    # ===================================================================
    # 9.12 P0-2 adaptive continuous d quadrature（Gate-0 重开）
    # ===================================================================
    @testset "9.12 P0-2 adaptive d quadrature" begin
        # 初始网格改名：initial_mesh 是生产起点，default_d_grid 是兼容别名
        nodes0, cells0 = initial_mesh()
        @test nodes0 == [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        @test default_d_grid() == (nodes0, cells0)
        @test all(cells0 .> 0)
        @test isapprox(sum(cells0), 1.0, atol = 1e-12)

        # adaptive 路径（不传 d_nodes/d_cells）：节点数翻倍、Σcells=1、
        # 先验质量不漂移（SPEC §64：refinement 不改变 prior mass）
        st_adapt = innovation_state(eps_tilde, row_ids, masks_full, R_t,
                                    E_active; t = t_dec)
        @test length(st_adapt.d_nodes) >= length(nodes0)
        @test isapprox(sum(st_adapt.d_cells), 1.0, atol = 1e-9)
        @test isapprox(sum(st_adapt.d_weights), 1.0, atol = 1e-12)
        @test all(0 .< st_adapt.d_nodes .<= 1)

        # 与固定网格路径在相同输入上一致（adaptive 是细化，不是另一套数学）
        st_fixed = innovation_state(eps_tilde, row_ids, masks_full, R_t,
                                    E_active; t = t_dec,
                                    d_nodes = nodes0, d_cells = cells0)
        # 固定网格的节点是 adaptive 初始网格的子集——V_t 在相同节点上逐位一致
        for g in eachindex(nodes0)
            @test isapprox(st_adapt.V_t[findfirst(==(nodes0[g]), st_adapt.d_nodes)],
                           st_fixed.V_t[g], atol = 1e-12)
        end

        # refinement certificate 直接调用：默认 tol 下收敛（合成 iid 数据）
        qres = adaptive_d_quadrature(st_adapt.eps_R,
                                     st_adapt.row_ids[st_adapt.J_t], t_dec)
        @test qres.converged == true
        @test qres.n_levels >= 1
        @test length(qres.nodes) > length(nodes0)
        @test isapprox(sum(qres.weights), 1.0, atol = 1e-12)
        # E[V(d)] 与 st_adapt 的加权 V 一致（同一数学对象）
        E1 = zeros(4, 4)
        for g in eachindex(qres.weights)
            E1 .+= qres.weights[g] .* qres.V_t[g]
        end
        E2 = zeros(4, 4)
        for g in eachindex(st_adapt.d_weights)
            E2 .+= st_adapt.d_weights[g] .* st_adapt.V_t[g]
        end
        @test isapprox(opnorm(E1 - E2), 0.0, atol = 1e-6)

        # 预算耗尽 fail loudly（文本含 "Numerical integration did not converge"）
        err_budget = try
            adaptive_d_quadrature(st_adapt.eps_R,
                                  st_adapt.row_ids[st_adapt.J_t], t_dec;
                                  tol_d = 0.0, tol_d2 = 0.0, tol_V = 0.0,
                                  max_levels = 2)
            nothing
        catch e
            e
        end
        @test err_budget isa ErrorException
        @test occursin("Numerical integration did not converge", err_budget.msg)

        # 显式初始网格参数（非默认 initial_mesh）
        qres2 = adaptive_d_quadrature(st_adapt.eps_R,
                                      st_adapt.row_ids[st_adapt.J_t], t_dec;
                                      initial_nodes = nodes0, initial_cells = cells0)
        @test qres2.converged == true

        # 非法初始网格 fail loudly
        @test_throws DomainError adaptive_d_quadrature(
            st_adapt.eps_R, st_adapt.row_ids[st_adapt.J_t], t_dec;
            initial_nodes = [0.0, 0.5], initial_cells = [0.5, 0.5])
    end

    # ===================================================================
    # 9.13 P0-3 joint residual span 满秩 / fail-closed（Gate-0 重开）
    # ===================================================================
    @testset "9.13 P0-3 rank sufficiency / fail-closed" begin
        # rank_sufficient：满秩 fixture 为 true；rank 亏 fixture 为 false
        @test rank_sufficient(st.eps_R) == true
        masks_rd = [begin
                        m = trues(N_a)
                        m[4] = i in (30, 60, 90)
                        m
                    end for i in 1:n_rows]
        st_rd = innovation_state(eps_tilde, row_ids, masks_rd, R_t, E_active; t = t_dec)
        @test rank_sufficient(st_rd.eps_R) == false
        @test residual_rank(st_rd.eps_R) <= 3

        # 构造层满秩 gate：require_full_rank=true 时 rank 不足 → error
        err_rank = try
            innovation_state(eps_tilde, row_ids, masks_rd, R_t, E_active;
                             t = t_dec, require_full_rank = true)
            nothing
        catch e
            e
        end
        @test err_rank isa ErrorException
        @test occursin("innovation coverage failure", err_rank.msg)
        @test occursin("sample null space", err_rank.msg)

        # 满秩 fixture 在 require_full_rank=true 下正常构造（不误杀）
        st_full = innovation_state(eps_tilde, row_ids, masks_full, R_t,
                                   E_active; t = t_dec, require_full_rank = true)
        @test st_full isa InnovationState

        # resolve_risk_domain 满秩判据（提供 residual_rows）：
        # 资产 5 仅行 60 覆盖（rank 亏）→ 剔除 free 直至满秩
        masks_cov = [begin
                         m = trues(N_a)
                         m[5] = i == 60
                         m
                     end for i in 1:n_rows]
        # 构造残差：asset 空间形态（n_rows × N_a）
        eps_cov = copy(eps_tilde)
        # R=[1,2,5]：J 只有行 60 覆盖 5 → rank ≤ 2 < 3 → 剔除 5
        R_res, dropped = resolve_risk_domain(row_ids, masks_cov, [1, 2, 5],
                                             Int[], t_dec;
                                             residual_rows = eps_cov)
        @test R_res == [1, 2]
        @test dropped == [5]

        # locked 导致不可收缩（提供 residual_rows 时）→ fail loudly
        err_lock_rank = try
            resolve_risk_domain(row_ids, masks_cov, Int[], [5], t_dec;
                                residual_rows = eps_cov)
            nothing
        catch e
            e
        end
        @test err_lock_rank isa ErrorException
        @test occursin("innovation coverage failure", err_lock_rank.msg)
        @test occursin("sample null space", err_lock_rank.msg)

        # 不提供 residual_rows 时保持旧判据（向后兼容）
        R_legacy, dropped_legacy = resolve_risk_domain(row_ids, masks_cov,
                                                       [1, 2, 5], Int[], t_dec)
        @test R_legacy == [1, 2]
        @test dropped_legacy == [5]
    end

end # @testset "gate0 Step 10-12: vector innovation reference"

end # module Gate0InnovationTests
    