# =============================================================================
# KTraderGate0 统一 mode coordinate producer — Step 3 tiny 级测试
# =============================================================================
#
# 测试对象：src/gate0/geometry.jl（H7 复制纯函数）与 src/gate0/modes.jl
# （mode_field / asset_field_reconstruct / cumulative_path_coordinates /
# mode_basis_design / domain_mode_basis / risk_domain_mode_basis）。
#
# 纪律：
# - tiny 级（D-089 测试升级阶梯：static → tiny analytic → …）：合成数据、
#   固定 seed（MersenneTwister(20261009)）、无外部依赖、无 I/O；
# - 本任务不运行测试（Engineer 不执行命令；运行验证由 Manager 安排
#   DevOps 受控执行，≤60s/RSS 护栏）；
# - 断言对象全部与回测收益无关（开发守则 §24）；
# - 加载方式：经临时 module include 两个裸文件（主 module KTraderGate0
#   由后续施工步骤建立——裁决 H1/H2；此处不依赖旧线 KTrader）。

# 缺 using 修复（Wave 2 集成收口）：测试体 L117/L125 使用 Statistics 的
# mean（原 using 面遗漏——交付时未运行，运行验证暴露）。
using Test, Random, LinearAlgebra, Statistics

module Gate0ModesEnv
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "geometry.jl"))
    include(joinpath(@__DIR__, "..", "..", "src", "gate0", "modes.jl"))
end
using .Gate0ModesEnv

@testset "gate0 Step 3: mode coordinates" begin
    # -------------------------------------------------------------------
    # 合成数据（固定 seed；全观测 panel + 合成 s1）
    # -------------------------------------------------------------------
    SEED = 20261009
    rng = MersenneTwister(SEED)
    N = 5
    T = 300
    logp = 100.0 .+ cumsum(randn(rng, T, N); dims = 1)
    returns = diff(logp; dims = 1)          # (T-1) × N，全观测
    s1 = 1e-2 .+ 2e-2 .* rand(rng, N)       # 合成正有限一日 ruler
    E = domain_mode_basis(N)                # E_active（active 域级基，裁决 E1）
    r_row = collect(returns[100, :])        # 测试行（全观测）
    u_row = r_row ./ s1

    # -------------------------------------------------------------------
    # 0. 几何原语复制健全性（H7：复制不改数值语义）
    # -------------------------------------------------------------------
    @testset "geometry primitives (H7 copy)" begin
        Qg = relative_gauge(N)
        @test isapprox(Qg' * Qg, Matrix(I, N - 1, N - 1), atol = 1e-12)      # QᵀQ = I
        @test isapprox(Qg' * fill(1.0, N), zeros(N - 1), atol = 1e-12)       # Qᵀ𝟏 = 0
        @test size(relative_gauge(1)) == (1, 0)                              # N=1 退化
        @test isapprox(E' * E, Matrix(I, N, N), atol = 1e-12)                # E 正交
        @test E[:, 1] == fill(1.0 / sqrt(N), N)                              # e₀ 域级等权
        @test risk_domain_mode_basis(collect(1:3)) == domain_mode_basis(3)   # E_{R_t} 构造
        # principal_sqrt_root：唯一对称 PSD 主根（SVD 版复制）
        M4 = randn(rng, 4, 3)
        R4 = principal_sqrt_root(M4)
        @test isapprox(R4, R4', atol = 1e-12)
        @test isapprox(R4 * R4, M4 * M4', atol = 1e-10)
        # ruler：per-asset power-law 拟合（复制语义冒烟——形状/有限/正）
        s_rul = ruler(logp, fill(1, N))
        @test size(s_rul) == (N, length(TAUS))
        @test all(isfinite, s_rul)
        @test all(>(0), s_rul)
    end

    # -------------------------------------------------------------------
    # 1. reconstruct 精度（全观测行：asset → mode → asset 往返恒等）
    # -------------------------------------------------------------------
    @testset "reconstruct (full observation)" begin
        y = mode_field(r_row, trues(N), s1, E)
        rec = asset_field_reconstruct(y.y, E)
        @test isapprox(rec, u_row, atol = 1e-12)                    # 往返逐点一致
        @test isapprox(y.y[1], dot(E[:, 1], u_row), atol = 1e-12)   # 全观测行 m = e₀ᵀu
        @test isapprox(sum(y.e), 0.0, atol = 1e-12)                 # e ∈ 𝟏⊥（行和构造性为零）
        @test isapprox(E[:, 2:N] * y.y[2:N], y.e, atol = 1e-12)     # Q·q = e（无损）
        @test all(isfinite, y.y) && all(isfinite, y.e)
    end

    # -------------------------------------------------------------------
    # 2. permutation 协变（资产列置换 π；field 层面的 πᵀw 语义）
    # -------------------------------------------------------------------
    @testset "permutation covariance" begin
        y = mode_field(r_row, trues(N), s1, E)
        rec = asset_field_reconstruct(y.y, E)
        p = randperm(rng, N)
        y_p = mode_field(r_row[p], trues(N)[p], s1[p], E[p, :])
        rec_p = asset_field_reconstruct(y_p.y, E[p, :])
        @test isapprox(rec_p, rec[p], atol = 1e-12)   # asset 空间 gather 协变
        @test isapprox(y_p.y, y.y, atol = 1e-12)      # E 同步置换 ⇒ mode 表示不变
        @test isapprox(y_p.e, y.e[p], atol = 1e-12)   # embedded field 同步置换
    end

    # -------------------------------------------------------------------
    # 3. gauge 旋转不变（Q → QR；mode 表示变、asset 空间对象不变）
    # -------------------------------------------------------------------
    @testset "gauge rotation invariance" begin
        y = mode_field(r_row, trues(N), s1, E)
        rec = asset_field_reconstruct(y.y, E)
        Rrot = Matrix(qr(randn(rng, N - 1, N - 1)).Q)
        E_rot = hcat(E[:, 1], E[:, 2:N] * Rrot)
        y_rot = mode_field(r_row, trues(N), s1, E_rot)
        rec_rot = asset_field_reconstruct(y_rot.y, E_rot)
        @test isapprox(rec_rot, rec, atol = 1e-12)                     # asset 对象不变
        @test isapprox(y_rot.y[1], y.y[1], atol = 1e-12)               # m 不变
        @test isapprox(y_rot.y[2:N], Rrot' * y.y[2:N], atol = 1e-12)   # q 协变（Rᵀq）
    end

    # -------------------------------------------------------------------
    # 4. ragged：单日 missing / IPO / 行级归一 / 空行 / dummy（D-022/E1）
    # -------------------------------------------------------------------
    @testset "ragged embedding semantics" begin
        # (a) 单日 missing：缺失坐标 e=0、mask 保留、m_s 用当日 O_s 归一
        obs_miss = [trues(4)..., false]
        r_miss = [r_row[1:4]..., NaN]
        y_miss = mode_field(r_miss, obs_miss, s1, E)
        @test y_miss.observed == obs_miss                       # mask 保留（D-022）
        @test y_miss.e[5] == 0.0                                # 零嵌入（field 代数）
        u4 = u_row[1:4]
        @test isapprox(y_miss.e[1:4], u4 .- mean(u4), atol = 1e-12)
        @test isapprox(y_miss.y[1], sum(u4) / sqrt(4), atol = 1e-12)   # 行级归一 1/√|O_s|
        @test isapprox(sum(y_miss.e), 0.0, atol = 1e-12)
        @test isapprox(E[:, 2:N] * y_miss.y[2:N], y_miss.e, atol = 1e-12)  # Q·q = e 无损
        # (b) IPO：资产 5 前夜未观察、首日观察（field 层按 mask 语义）
        y_ipo1 = mode_field([r_row[1:4]..., NaN], [trues(4)..., false], s1, E)
        y_ipo2 = mode_field(r_row, trues(N), s1, E)
        @test y_ipo1.e[5] == 0.0                                # 上市前嵌入 0
        @test isapprox(y_ipo2.e[5], u_row[5] - mean(u_row), atol = 1e-12)  # 首日进入 field
        # (c) 行级归一系数：|O_s| 变化时 m_s = Σ_{O_s}u / √|O_s|
        for k in (2, 3, 5)
            obsk = falses(N); obsk[1:k] .= true
            rk = [obsk[j] ? r_row[j] : NaN for j in 1:N]
            yk = mode_field(rk, obsk, s1, E)
            @test isapprox(yk.y[1], sum(u_row[1:k]) / sqrt(k), atol = 1e-12)
            @test isapprox(sum(yk.e), 0.0, atol = 1e-12)
        end
        # (d) 空观察行：m=0、e=0（零嵌入语义；行集判定归消费层——VI §1.4）
        y_empty = mode_field(fill(NaN, N), falses(N), s1, E)
        @test y_empty.y[1] == 0.0
        @test all(iszero, y_empty.e)
        @test all(x -> !x, y_empty.observed)
    end

    @testset "dummy all-NaN asset (D-060 coordinate layer)" begin
        # 全 NaN 资产（永未观察）的 field 嵌入贡献严格为零：对剩余资产的
        # (m, e) 与 4 资产域完全一致。admission 层排除是 Step 1 职责（D-013）；
        # 坐标层验证的是零贡献不变性。注：含 dummy 的域级 e₀ 归一（1/√5）与
        # 4 资产域（1/√4）不同——asset 域重构的差异正是「域基 ≠ 行级归一」
        # （裁决 E1）的体现，也是 dummy 必须在 admission 层排除的原因。
        r_dummy = [r_row[1:4]..., NaN]
        y_dummy = mode_field(r_dummy, [trues(4)..., false], s1, E)
        y4 = mode_field(r_row[1:4], trues(4), s1[1:4], domain_mode_basis(4))
        @test y_dummy.e[5] == 0.0
        @test isapprox(y_dummy.e[1:4], y4.e, atol = 1e-12)
        @test isapprox(y_dummy.y[1], y4.y[1], atol = 1e-12)
    end

    # -------------------------------------------------------------------
    # 5. mode_basis_design：维度与布局（GATE0_RESPONSE_POSTERIOR §1 维度表）
    # -------------------------------------------------------------------
    @testset "mode_basis_design dimensions & layout" begin
        T_rows = size(returns, 1)
        m_series = Vector{Float64}(undef, T_rows)
        e_emb = Matrix{Float64}(undef, T_rows, N)
        for t in 1:T_rows
            yt = mode_field(collect(returns[t, :]), trues(N), s1, E)
            m_series[t] = yt.y[1]
            e_emb[t, :] .= yt.e
        end
        X_m, X_rel = cumulative_path_coordinates(m_series, e_emb)
        s_m = fast_s_m(X_m)                 # 9 维（|TAUS|），path_basis_1d 经 BANDCOL 分派
        s_perp = compute_s_perp(X_rel)      # 7 维（|BANDS|）per-band 标量 ruler
        Q = E[:, 2:N]
        design = mode_basis_design(X_m, X_rel, Q, s_m, s_perp)
        # 维度断言（维度表逐项：B_m 14；B_⊥ 14(N−1)；总输入 14N+1 含 DC 预留列 1）
        @test size(design.B_m) == (T_rows, 14)
        @test size(design.B_perp) == (T_rows, 14 * (N - 1))
        @test size(design.X_q) == (T_rows, N - 1)
        @test 1 + 14 + 14 * (N - 1) == 14 * N + 1
        @test 1 + size(design.B_m, 2) + size(design.B_perp, 2) == 14 * N + 1
        @test isapprox(design.B_m, path_basis_1d(X_m, s_m), atol = 1e-12)  # B_m 回归锚
        # B_⊥ 列布局抽样：(band, channel, gauge 坐标)——offset=(b−1)·2(N−1)，
        # Q 通道前 (N−1) 列、P 通道后 (N−1) 列
        Bk_cache = Dict{Int,Matrix{Float64}}()
        for k in (1, 2, 4), b in (1, 4, 7), ch in (1, 2)
            Bk = get!(Bk_cache, k) do
                path_basis_1d(collect(design.X_q[:, k]), s_perp)
            end
            dst = (b - 1) * 2 * (N - 1) + (ch - 1) * (N - 1) + k
            @test isapprox(design.B_perp[:, dst], view(Bk, :, 2 * (b - 1) + ch), atol = 1e-12)
        end
        # 累积坐标（SPEC §14：level 上累积）
        @test X_m[1] == m_series[1]
        @test isapprox(X_m[end], sum(m_series), atol = 1e-10)
        @test isapprox(X_rel[end, 1], sum(e_emb[:, 1]), atol = 1e-10)
    end

    # -------------------------------------------------------------------
    # 6. 线性交换性（QᵀX_rel = X_q：先投影再累积 = 先累积再投影；
    #    滤波 ∘ 投影 = 投影 ∘ 滤波——B_⊥ 构造语义）
    # -------------------------------------------------------------------
    @testset "linear exchangeability" begin
        T_rows = size(returns, 1)
        m_series = Vector{Float64}(undef, T_rows)
        e_emb = Matrix{Float64}(undef, T_rows, N)
        for t in 1:T_rows
            yt = mode_field(collect(returns[t, :]), trues(N), s1, E)
            m_series[t] = yt.y[1]
            e_emb[t, :] .= yt.e
        end
        X_m, X_rel = cumulative_path_coordinates(m_series, e_emb)
        s_perp = compute_s_perp(X_rel)
        Q = E[:, 2:N]
        design = mode_basis_design(X_m, X_rel, Q, fast_s_m(X_m), s_perp)
        # (i) 先投影再累积 == 先累积再投影
        X_q_first = cumsum(e_emb * Q; dims = 1)
        @test isapprox(design.X_q, X_q_first, atol = 1e-10)
        # (ii) 滤波与投影交换：B_⊥ 每 (band, channel) 块 == asset 坐标同块 × Q
        asset_B = zeros(T_rows, 2 * length(BANDS) * N)
        for j in 1:N
            Bj = path_basis_1d(collect(X_rel[:, j]), s_perp)
            for b in 1:length(BANDS), ch in 1:2
                asset_B[:, (b - 1) * 2 * N + (ch - 1) * N + j] .= view(Bj, :, 2 * (b - 1) + ch)
            end
        end
        for b in 1:length(BANDS), ch in 1:2
            perp_cols = (b - 1) * 2 * (N - 1) + (ch - 1) * (N - 1) .+ (1:(N - 1))
            asset_cols = (b - 1) * 2 * N + (ch - 1) * N .+ (1:N)
            @test isapprox(design.B_perp[:, perp_cols], asset_B[:, asset_cols] * Q, atol = 1e-10)
        end
    end

    # -------------------------------------------------------------------
    # 7. fail loudly（NaN/Inf/维度——SPEC §56）
    # -------------------------------------------------------------------
    @testset "fail loudly" begin
        y = mode_field(r_row, trues(N), s1, E)
        @test_throws DomainError mode_field([NaN, r_row[2:N]...], trues(N), s1, E)
        @test_throws DomainError mode_field([r_row[1:4]..., Inf], trues(N), s1, E)
        @test_throws DomainError mode_field(r_row, trues(N), [s1[1:N-1]..., 0.0], E)
        @test_throws DomainError mode_field(r_row, trues(N), [s1[1:N-1]..., NaN], E)
        @test_throws DomainError mode_field(r_row, trues(N), [-s1[1], s1[2:N]...], E)
        @test_throws DimensionMismatch mode_field(r_row, trues(N - 1), s1, E)
        @test_throws DimensionMismatch mode_field(r_row, trues(N), s1[1:N-1], E)
        @test_throws DimensionMismatch mode_field(r_row, trues(N), s1, domain_mode_basis(N + 1))
        @test_throws DimensionMismatch asset_field_reconstruct(y.y, domain_mode_basis(N + 1))
        @test_throws DimensionMismatch asset_field_reconstruct(y.y[1:N-1], E)
        @test_throws DomainError asset_field_reconstruct([y.y[1:N-1]..., NaN], E)
        @test_throws ArgumentError domain_mode_basis(0)
        # 序列层
        T_rows = size(returns, 1)
        m_series = Vector{Float64}(undef, T_rows)
        e_emb = Matrix{Float64}(undef, T_rows, N)
        for t in 1:T_rows
            yt = mode_field(collect(returns[t, :]), trues(N), s1, E)
            m_series[t] = yt.y[1]
            e_emb[t, :] .= yt.e
        end
        X_m, X_rel = cumulative_path_coordinates(m_series, e_emb)
        s_m = fast_s_m(X_m)
        s_perp = compute_s_perp(X_rel)
        bad_m = copy(m_series); bad_m[10] = NaN
        @test_throws DomainError cumulative_path_coordinates(bad_m, e_emb)
        bad_e = copy(e_emb); bad_e[20, 2] = Inf
        @test_throws DomainError cumulative_path_coordinates(m_series, bad_e)
        @test_throws DimensionMismatch cumulative_path_coordinates(m_series[1:end-1], e_emb)
        @test_throws DimensionMismatch mode_basis_design(X_m, X_rel, relative_gauge(N + 2), s_m, s_perp)
        @test_throws DimensionMismatch mode_basis_design(X_m, X_rel, E[:, 2:N], s_m[1:5], s_perp)
        @test_throws DimensionMismatch mode_basis_design(X_m, X_rel, E[:, 2:N], s_m, s_perp[1:3])
    end
end

# 断言计数（@test 逐条，供收尾报告核对）：
#   geometry primitives:      11
#   reconstruct:               5
#   permutation:               3
#   gauge rotation:            3
#   ragged:                   17   (a:6, b:2, c:3k×2=6, d:3)
#   dummy:                     3
#   dimensions & layout:      27   (5 维度 + 1 B_m 锚 + 18 布局抽样 + 3 累积)
#   exchangeability:          15   (1 交换 + 14 块)
#   fail loudly:              18
#   合计：                   102
