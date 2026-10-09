# test/history_cache_contract_tests.jl — SPEC §56 input/cache 一致性守卫的契约测试。
#
# 本文件由 cache 边界修复任务新增；配套源码改动为 src/numerics.jl（守卫
# verify_history_cache_prefix）、src/predict.jl（_prepare_v1 在首个 cache
# 读取前接线）与 src/incremental.jl（initialize_inference 在任何状态构建
# /advance 之前的统一守卫接线，替换旧的 post-advance logs-only 检查；
# advance_exact!/_prepare_current! 不读 cache——其 _prepare_v1 回退也不传
# cache，数学输入全部来自 advance 累积的 state，无需接线）。
#
# 接入方式：本文件经 test/runtests.jl 的标准 include 循环加载
# （ThreeContractGate 的 REQUIRED 登记已覆盖；不走 extra 元组登记）。
# 独立运行 (julia --project test/history_cache_contract_tests.jl) 同样
# 成立。本文件不宣称运行结果——全部断言是动态的，运行证据由 DevOps 侧
# 持有。
#
# 守卫契约：
#   [G1] 合法恒等与合法前缀：cache(panel) 对 panel 自身及任意前缀通过
#        （full-cache 供合法 prefix 复用）。
#   [G2] no-lookahead：守卫只读 1:T 行——cache 的未来行（log/return 载荷）
#        被 mutate 不影响当前 prefix 校验；未来首观测（行 > T）的元数据合法。
#   [G3] 异源拒绝：同 shape 不同价格（含纯价格 level 平移——收益近似不变、
#        logs 全变，证明必须比对 logs 而非只比 returns）、末行 finite→NaN、
#        NaN↔finite 双向，全部拒绝。
#   [G4] 突变拒绝：cache 构建后 panel 内部/末行突变、cache 载荷（logs/
#        returns）内部突变、cache 元数据（first_price/first_return）恶意
#        变异（含把确有 prefix 观测的列的首观测推到未来），全部拒绝。
#   [G5] 形状守卫：宽度不匹配、行数超限、手工构造的畸形 cache（returns
#        形状错 / 元数据长度错）在内容比对之前拒绝。
#   [W1] 接线：_prepare_v1 在任何 cache 字段被读之前调用守卫——
#        prepare_reference / fit_v1 / _prepare_v1 三个入口都不可绕过，
#        错误信息带 "input/cache mismatch"（区别于 F_folds/n_res 等数学
#        期校验错误，证明 fail-early、早于数学消费）。
#   [W2] 能红的反例：完整合法尺寸 panel + 异源 cache 的 prepare_reference /
#        fit_v1 必须抛错。守卫被移除时 _prepare_v1 会静默成功（SPEC §56
#        缺口复现：A 的训练数据 + B 的 alive mask 混用），本组断言转红。
#   [M1] 数学不变性：同 panel 无 cache 与有 cache 的 prepare 产出一致
#        （守卫不改变数学）；active 集合两路径一致（守卫的语义推论）。
#   [I1] initialize_inference 统一守卫接线（真实生产入口）：异源 cache 在
#        任何状态构建/advance 之前被共享守卫拒绝（错误信息含
#        "input/cache mismatch"——若接线回退到旧 post-advance logs-only
#        检查，信息断言转红）；cache returns 载荷变异与恶意元数据同样
#        拒绝（旧检查的覆盖缺口）；合法 full-cache/prefix 复用与未来后缀
#        ignore 在真实入口可用；合法 initialize 后 solve_current! 全链路
#        可用（_prepare_current!/_prepare_v1 均不消费 cache）。
#   [R*] ruler_stats 本源守卫（verify_ruler_stats_prefix，与 cache 守卫
#        同属 SPEC §56 一致性家族）：异源同形缓存（A/B 不同 log 振幅）、
#        正确 cache 配错误 stats、acc/cnt 有效读点变异（含 NaN/Inf）、
#        谎报 first-observation 行，全部在消费前 fail-loud 拒绝；更长
#        合法缓存的 future suffix 不读；合法同源 prepare 逐字段等价；
#        initialize_inference 既有同源验证不受影响。

using Test, Random
using KTrader
const HC = KTrader

# 最小 fixture：8×2 正价格、含一个 NaN 洞 [4,2]（守卫必须 NaN-aware）。
function hc_panel(; seed = 99)
    rng = MersenneTwister(seed)
    P = exp.(cumsum(0.01 .* randn(rng, 8, 2); dims = 1))
    P[4, 2] = NaN
    P
end

# 完整 fixture：300×2（WARMUP=256 → n_res=43，F_folds=3 可行），全 finite。
hc_big_panel() = exp.(cumsum(0.008 .* randn(MersenneTwister(100), 300, 2); dims = 1))

# 与 PriceHistoryCache 构造器相同派生量的显式重算（供手工构造恶意/畸形
# cache：载荷正确、元数据或形状被改）。
function hc_parts(P)
    T, N = size(P)
    logs = log.(P)
    rets = diff(logs; dims = 1)
    fp = [something(findfirst(isfinite, view(logs, :, j)), T + 1) for j in 1:N]
    fr = [something(findfirst(isfinite, view(rets, :, j)), T) for j in 1:N]
    ar = [count(isfinite, view(rets, t, :)) for t in 1:(T - 1)]
    isr = [cnt > 0 ? 1.0 / sqrt(cnt) : 0.0 for cnt in ar]
    (; logs, rets, fp, fr, ar, isr)
end

@testset "verify_history_cache_prefix: guard contract" begin
    A = hc_panel()
    T, N = size(A)

    # ---------- [G1] 合法恒等与合法前缀 ----------
    @test HC.verify_history_cache_prefix(A, HC.PriceHistoryCache(A)) === nothing
    for Tp in (1, 3, T)
        @test HC.verify_history_cache_prefix(view(A, 1:Tp, :),
                                             HC.PriceHistoryCache(A)) === nothing
    end

    # ---------- [G2] no-lookahead：未来行不读、不设限 ----------
    c = HC.PriceHistoryCache(A)
    c.log_prices[T, 1] += 0.5        # 未来行（prefix 只验证 1:5）
    c.returns[T - 1, 1] += 0.5       # 未来 return 行（prefix 只验证 1:4）
    @test HC.verify_history_cache_prefix(view(A, 1:5, :), c) === nothing
    # 未来首观测元数据合法：列 2 前 5 行价格置 NaN 后其 cache 的首价在行 6。
    A2 = Matrix{Float64}(A); A2[1:5, 2] .= NaN
    c2 = HC.PriceHistoryCache(A2)
    @test c2.first_price[2] == 6
    @test HC.verify_history_cache_prefix(view(A2, 1:5, :), c2) === nothing

    # ---------- [G3] 异源拒绝 ----------
    B = Matrix{Float64}(A); B[3, 1] *= 1.05          # 内部一行价格改动
    @test_throws ArgumentError HC.verify_history_cache_prefix(B, HC.PriceHistoryCache(A))
    # 纯价格 level 平移：returns 近似不变、logs 全变——只比 returns 的弱
    # 守卫会漏检；本守卫比 logs，必须拒绝。
    C = A .* 2.0
    @test_throws ArgumentError HC.verify_history_cache_prefix(C, HC.PriceHistoryCache(A))
    D = Matrix{Float64}(A); D[end, 2] = NaN          # 末行 finite→NaN
    @test_throws ArgumentError HC.verify_history_cache_prefix(D, HC.PriceHistoryCache(A))
    E = Matrix{Float64}(A); E[4, 2] = 1.3            # cache 侧 NaN 洞 vs finite
    @test_throws ArgumentError HC.verify_history_cache_prefix(E, HC.PriceHistoryCache(A))
    F = Matrix{Float64}(A); F[4, 1] = NaN            # cache 侧 finite vs NaN
    @test_throws ArgumentError HC.verify_history_cache_prefix(F, HC.PriceHistoryCache(A))

    # ---------- [G4] 突变拒绝 ----------
    Amut = Matrix{Float64}(A)                        # cache 建好后 panel 内部突变
    cm = HC.PriceHistoryCache(A)
    Amut[3, 1] *= 1.02
    @test_throws ArgumentError HC.verify_history_cache_prefix(Amut, cm)
    Amut2 = Matrix{Float64}(A); Amut2[end, 1] = NaN  # 末行突变
    @test_throws ArgumentError HC.verify_history_cache_prefix(Amut2, cm)
    clogs = HC.PriceHistoryCache(A)                  # cache logs 载荷内部突变
    clogs.log_prices[2, 1] += 1e-12
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, clogs)
    crets = HC.PriceHistoryCache(A)                  # cache returns 载荷内部突变
    crets.returns[2, 1] += 1e-12
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, crets)
    # 恶意元数据：正确载荷 + 错误 first_price / first_return（手工构造）。
    (; logs, rets, fp, fr, ar, isr) = hc_parts(A)
    bad_fp = HC.PriceHistoryCache(logs, rets, fp, fr, ar, isr)
    bad_fp.first_price[1] = 3                        # 真值 1（行 1 finite）
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, bad_fp)
    bad_fr = HC.PriceHistoryCache(logs, rets, fp, [2, 1], ar, isr)
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, bad_fr)
    # 把确有 prefix 观测的列的首观测推到未来，同样是元数据说谎。
    push_fr = HC.PriceHistoryCache(logs, rets, fp, fr, ar, isr)
    push_fr.first_return[1] = T                      # 列 1 真值 1
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, push_fr)
    push_fp = HC.PriceHistoryCache(logs, rets, fp, fr, ar, isr)
    push_fp.first_price[1] = T + 2
    @test_throws ArgumentError HC.verify_history_cache_prefix(A, push_fp)

    # ---------- [G5] 形状守卫（内容比对之前拒绝） ----------
    @test_throws ArgumentError HC.verify_history_cache_prefix(
        A, HC.PriceHistoryCache(view(A, 1:(T - 1), :)))            # 行数超限
    @test_throws ArgumentError HC.verify_history_cache_prefix(
        A, HC.PriceHistoryCache(A[:, 1:1]))                        # 宽度不匹配
    @test_throws ArgumentError HC.verify_history_cache_prefix(
        A, HC.PriceHistoryCache(logs, Matrix{Float64}(undef, T, N),
                                fp, fr, ar, isr))                  # returns 形状错
    @test_throws ArgumentError HC.verify_history_cache_prefix(
        A, HC.PriceHistoryCache(logs, rets, fp, [1, 2, 3], ar, isr))  # 元数据长度错
end

@testset "wiring: _prepare_v1 / prepare_reference / fit_v1 fail loud before math" begin
    A = hc_panel()
    cA = HC.PriceHistoryCache(A)

    # [W1] 小 panel + 异源 cache：守卫错误先于任何数学期校验（小 panel 的
    # F_folds/n_res 检查本会先/后抛错，但信息不同——用信息断言钉住来源）。
    B = Matrix{Float64}(A); B[3, 1] *= 1.05
    err = try
        HC._prepare_v1(B; F_folds = 3, history_cache = cA)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("input/cache mismatch", sprint(showerror, err))
    # 三个公开入口都不可绕过（同一 _prepare_v1 接线）。
    @test_throws ArgumentError HC.prepare_reference(B; F_folds = 3, history_cache = cA)
    @test_throws ArgumentError HC.fit_v1(B; F_folds = 3, history_cache = cA)
    err2 = try
        HC.prepare_reference(B; F_folds = 3, history_cache = cA)
        nothing
    catch e
        e
    end
    @test occursin("input/cache mismatch", sprint(showerror, err2))

    # [W2] 能红的反例：完整合法尺寸 panel + 异源 cache 必须抛错。守卫被
    # 移除时 prepare 会静默成功（A 训练 + B alive mask 混用），断言转红。
    Abig = hc_big_panel()
    cache_big = HC.PriceHistoryCache(Abig)
    Bbig = Abig .* 1.5                # 同 active、收益结构近似、价格不同
    @test_throws ArgumentError HC.prepare_reference(
        Bbig; F_folds = 3, history_cache = cache_big)
    @test_throws ArgumentError HC.fit_v1(
        Bbig; F_folds = 3, history_cache = cache_big)
    Btail = Matrix{Float64}(Abig); Btail[end, 2] = NaN   # 末行支持变化
    @test_throws ArgumentError HC.prepare_reference(
        Btail; F_folds = 3, history_cache = cache_big)
    # cache 载荷被破坏（returns 内部突变）同样在 prepare 前被拒。
    cbad = HC.PriceHistoryCache(Abig); cbad.returns[10, 1] += 1e-9
    @test_throws ArgumentError HC.prepare_reference(
        Abig; F_folds = 3, history_cache = cbad)
end

@testset "math unchanged: cache path ≡ no-cache path on a legal panel" begin
    # [M1] 合法 cache 不改变任何 prepare 输出（守卫只验证、不注入）。
    Abig = hc_big_panel()
    cache = HC.PriceHistoryCache(Abig)
    p_nc = HC.prepare_reference(Abig; F_folds = 3)
    p_c = HC.prepare_reference(Abig; F_folds = 3, history_cache = cache)
    @test p_c.T == p_nc.T == 300
    @test p_c.N == p_nc.N
    @test p_c.active_idx == p_nc.active_idx
    @test p_c.active_idx == HC.active_universe_indices(Abig)   # 两路径 active 一致
    @test p_c.s1 == p_nc.s1
    @test p_c.s_m == p_nc.s_m
    @test p_c.s_perp == p_nc.s_perp
    @test isequal(p_c.r, p_nc.r)
    @test isequal(p_c.m, p_nc.m)
    @test p_c.alive_now == p_nc.alive_now
    @test p_c.stats.ranges == p_nc.stats.ranges
    @test p_c.stats.full_xy ≈ p_nc.stats.full_xy rtol = 1e-12
    @test p_c.stats.full_yy ≈ p_nc.stats.full_yy rtol = 1e-12
    # 合法 prefix 复用：全长 cache 供前缀 prepare（no-lookahead 的正面）。
    p_pref = HC.prepare_reference(view(Abig, 1:280, :); F_folds = 3,
                                  history_cache = cache)
    @test p_pref.T == 280
    @test p_pref.active_idx == p_nc.active_idx
end

@testset "initialize_inference: unified guard at the real entry" begin
    A = hc_panel()
    # 异源 cache 在任何状态构建/advance 之前被共享守卫拒绝；错误信息来自
    # 统一守卫（区别于旧的 post-advance logs-only 信息——接线回退时转红）。
    B = Matrix{Float64}(A); B[3, 1] *= 1.05
    err = try
        HC.initialize_inference(B; F_folds = 3, history_cache = HC.PriceHistoryCache(A))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("input/cache mismatch", sprint(showerror, err))
    # returns 载荷变异（旧 logs-only 检查的覆盖缺口）同样拒绝。
    cret = HC.PriceHistoryCache(A); cret.returns[2, 1] += 1e-12
    @test_throws ArgumentError HC.initialize_inference(A; F_folds = 3, history_cache = cret)
    # 恶意元数据同样拒绝。
    (; logs, rets, fp, fr, ar, isr) = hc_parts(A)
    bad_fr = HC.PriceHistoryCache(logs, rets, fp, [2, 1], ar, isr)
    @test_throws ArgumentError HC.initialize_inference(A; F_folds = 3, history_cache = bad_fr)
    # 合法恒等通过；合法 full-cache 供 prefix 复用；未来后缀被 mutate 后
    # prefix initialize 仍通过（no-lookahead 的真实入口面）。
    st = HC.initialize_inference(A; F_folds = 3, history_cache = HC.PriceHistoryCache(A))
    @test st !== nothing
    Abig = hc_big_panel()
    cache_big = HC.PriceHistoryCache(Abig)
    st_pref = HC.initialize_inference(view(Abig, 1:280, :); F_folds = 3,
                                      history_cache = cache_big)
    @test st_pref !== nothing
    cache_big.log_prices[300, 1] += 0.5      # 未来行（prefix 只到 280）
    cache_big.returns[299, 1] += 0.5         # 未来 return 行
    st_fut = HC.initialize_inference(view(Abig, 1:280, :); F_folds = 3,
                                     history_cache = cache_big)
    @test st_fut !== nothing
    # 合法 prefix 的数学同一性：同一 prefix 有/无 cache 初始化后
    # prepare_incremental 产出的 PreparedProblem 逐字段等价——cache 只被
    # 守卫校验、从不注入数学，这是"同一 solver 输入"的直接证明（Prepared
    # 层等价即可，无需运行 solve/EB；_prepare_current! 的 fast 与 batch
    # fallback 均不读 cache，数学输入全部来自 advance 累积的 state）。
    st_c = HC.initialize_inference(Abig; F_folds = 3,
                                   history_cache = HC.PriceHistoryCache(Abig))
    st_nc = HC.initialize_inference(Abig; F_folds = 3)
    p_c = HC.prepare_incremental(st_c)
    p_nc = HC.prepare_incremental(st_nc)
    @test p_c.active_idx == p_nc.active_idx
    @test p_c.T == p_nc.T == 300
    @test p_c.N == p_nc.N
    @test p_c.ts_total == p_nc.ts_total
    @test p_c.n_res == p_nc.n_res
    @test p_c.P_features == p_nc.P_features
    @test p_c.F_folds == p_nc.F_folds
    @test p_c.s1 == p_nc.s1
    @test p_c.s_m == p_nc.s_m
    @test p_c.s_perp == p_nc.s_perp
    @test isequal(p_c.r, p_nc.r)
    @test isequal(p_c.m, p_nc.m)
    @test p_c.alive_now == p_nc.alive_now
    @test p_c.observed == p_nc.observed
    @test p_c.relative_embedding ≈ p_nc.relative_embedding rtol = 1e-12
    @test p_c.X_rel ≈ p_nc.X_rel rtol = 1e-12
    @test p_c.B_m ≈ p_nc.B_m rtol = 1e-12
    @test (p_c.X_rel_stacked === nothing) == (p_nc.X_rel_stacked === nothing)
    @test p_c.stats.ranges == p_nc.stats.ranges
    @test p_c.stats.full_xy ≈ p_nc.stats.full_xy rtol = 1e-12
    @test p_c.stats.full_yy ≈ p_nc.stats.full_yy rtol = 1e-12
    if p_c.stats.xx !== nothing
        @test p_nc.stats.xx !== nothing
        @test p_c.stats.full_xx ≈ p_nc.stats.full_xx rtol = 1e-12
        for f in eachindex(p_c.stats.xy)
            @test p_c.stats.xx[f] ≈ p_nc.stats.xx[f] rtol = 1e-12
        end
    else
        @test p_nc.stats.xx === nothing
    end
    for f in eachindex(p_c.stats.xy)
        @test p_c.stats.xy[f] ≈ p_nc.stats.xy[f] rtol = 1e-12
        @test p_c.stats.yy[f] ≈ p_nc.stats.yy[f] rtol = 1e-12
    end
    if p_c.macro_stats !== nothing
        @test p_nc.macro_stats !== nothing
        @test p_c.macro_stats.full.xy ≈ p_nc.macro_stats.full.xy rtol = 1e-12
        @test p_c.macro_stats.full.yy ≈ p_nc.macro_stats.full.yy rtol = 1e-12
        for f in eachindex(p_c.macro_stats.folds)
            @test p_c.macro_stats.folds[f].xy ≈ p_nc.macro_stats.folds[f].xy rtol = 1e-12
        end
    else
        @test p_nc.macro_stats === nothing
    end
end

@testset "ruler_stats prefix guard (verify_ruler_stats_prefix)" begin
    # ruler_stats 的本源语义：本历史 ruler 统计的 exact 加速缓存
    # （geometry.jl PrefixRulerStats）。异源同形缓存 / acc-cnt 有效
    # 读点变异（含 NaN/Inf 污染）/ 谎报首观测行，在消费前 fail-loud
    # 拒绝；更长合法缓存的未来 suffix 不读不设限；合法同源 prepare 逐
    # 字段等价（守卫只验证、不注入）。
    Abig = hc_big_panel()
    fbig = [something(findfirst(isfinite, view(Abig, :, j)), 1) for j in 1:2]
    stats_A = HC.build_prefix_ruler_stats(log.(Abig), fbig)

    # 异源 B：同形、每列 log 振幅不同（列 2 波动放大）。
    rngB = MersenneTwister(7)
    Bbig = exp.(cumsum([0.004 0.03] .* randn(rngB, 300, 2); dims = 1))
    fB = [something(findfirst(isfinite, view(Bbig, :, j)), 1) for j in 1:2]
    stats_B = HC.build_prefix_ruler_stats(log.(Bbig), fB)

    # 守卫本体：恒等通过、异源同形拒、谎报首观测行拒。
    @test HC.verify_ruler_stats_prefix(log.(Abig), stats_A, [1, 2], [1, 1]) === nothing
    @test_throws ArgumentError HC.verify_ruler_stats_prefix(
        log.(Abig), stats_B, [1, 2], [1, 1])
    @test_throws ArgumentError HC.verify_ruler_stats_prefix(
        log.(Abig), stats_A, [1, 2], [5, 1])

    # [R1] A/B 同形不同 log 振幅：prepare_reference / fit_v1 必拒——
    # 错误在 prepare 阶段抛出（守卫先于 ruler_from_stats 与任何 EB fit）。
    err = try
        HC.prepare_reference(Abig; F_folds = 3, ruler_stats = stats_B)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ruler_stats", sprint(showerror, err))
    @test_throws ArgumentError HC.fit_v1(Abig; F_folds = 3, ruler_stats = stats_B)
    @test_throws ArgumentError HC.path_kelly_v1(Abig; ridge_alpha = 1.0, ruler_stats = stats_B)

    # [R2] 正确 history_cache + 错误 ruler_stats 仍拒：cache 守卫通过
    # 不豁免 stats 的本源验证（两个守卫独立、同一语义家族）。
    cacheA = HC.PriceHistoryCache(Abig)
    @test_throws ArgumentError HC.prepare_reference(
        Abig; F_folds = 3, history_cache = cacheA, ruler_stats = stats_B)

    # [R3] acc/cnt 有效读点（T 行 active 列）变异必拒，含 NaN/Inf 污染。
    mut_acc = HC.build_prefix_ruler_stats(log.(Abig), fbig)
    mut_acc.acc[300, 1, 3] += 1e-9
    @test_throws ArgumentError HC.prepare_reference(Abig; F_folds = 3, ruler_stats = mut_acc)
    mut_nan = HC.build_prefix_ruler_stats(log.(Abig), fbig)
    mut_nan.acc[300, 2, 5] = NaN
    @test_throws ArgumentError HC.prepare_reference(Abig; F_folds = 3, ruler_stats = mut_nan)
    mut_inf = HC.build_prefix_ruler_stats(log.(Abig), fbig)
    mut_inf.acc[300, 1, 1] = Inf
    @test_throws ArgumentError HC.prepare_reference(Abig; F_folds = 3, ruler_stats = mut_inf)
    mut_cnt = HC.build_prefix_ruler_stats(log.(Abig), fbig)
    mut_cnt.cnt[300, 1, 4] += 1
    @test_throws ArgumentError HC.prepare_reference(Abig; F_folds = 3, ruler_stats = mut_cnt)

    # [R4] 更长合法缓存供前缀复用：future suffix（> T 的行）mutate 不
    # 影响前缀验证（守卫只读本 panel 的 T 行，不读不限制未来）。
    stats_full = HC.build_prefix_ruler_stats(log.(Abig), fbig)
    stats_full.acc[295, 1, 2] += 1.0
    stats_full.cnt[295, 2, 6] += 3
    p_pref = HC.prepare_reference(view(Abig, 1:280, :); F_folds = 3,
                                  ruler_stats = stats_full)
    @test p_pref.T == 280
    @test p_pref.active_idx == HC.active_universe_indices(view(Abig, 1:280, :))

    # [R5] 合法同源：有 stats 与无 stats 的 prepare 逐字段等价
    # （ruler_from_stats 与 ruler 同输入同拟合；守卫零注入）。
    p_s = HC.prepare_reference(Abig; F_folds = 3, ruler_stats = stats_A)
    p_ref = HC.prepare_reference(Abig; F_folds = 3)
    @test p_s.active_idx == p_ref.active_idx
    @test p_s.s1 == p_ref.s1
    @test p_s.s_m == p_ref.s_m
    @test p_s.s_perp == p_ref.s_perp
    @test isequal(p_s.m, p_ref.m)
    @test isequal(p_s.r, p_ref.r)
    @test p_s.alive_now == p_ref.alive_now
    @test p_s.stats.ranges == p_ref.stats.ranges
    @test p_s.stats.full_xy ≈ p_ref.stats.full_xy rtol = 1e-12
    @test p_s.stats.full_yy ≈ p_ref.stats.full_yy rtol = 1e-12
    for f in 1:3
        @test p_s.stats.xy[f] ≈ p_ref.stats.xy[f] rtol = 1e-12
        @test p_s.stats.yy[f] ≈ p_ref.stats.yy[f] rtol = 1e-12
    end

    # [R6] initialize_inference 的既有同源守卫不受影响：异源 stats 在
    # 任何 state 构建前被拒；合法同源照常初始化（现有行为防回归钉）。
    @test_throws ArgumentError HC.initialize_inference(
        Abig; F_folds = 3, ruler_stats = stats_B)
    st = HC.initialize_inference(Abig; F_folds = 3, ruler_stats = stats_A)
    @test st !== nothing
end

# ---------------------------------------------------------------------------
# [P*] prepare_reference 封闭公开 keyword 接口契约（2026-10-08 Manager 决定：
# 公开 reference 入口只接受显式合法 keyword 白名单 ridge_alpha / F_folds /
# ruler_stats / history_cache / alpha_initial / timing / workspace；三个
# internal 通道 ruler_override / scale_override / statistics_builder 属于
# incremental 引擎（incremental.jl _prepare_current!）的私有接线，经公开入口
# 传入时即使取 =nothing 也必须在 keyword dispatch 层拒绝——拒绝严格先于
# _prepare_v1 体内任何语句（workspace generation bump 是其第一条语句）、
# 先于任何 cache 读取、先于任何 builder 执行。这是白名单封闭，不是黑名单
# 过滤：不存在被静默丢弃的 keyword。
# 甄别力声明：本组反例在旧的 kwargs... 宽透传接口下必红——旧接口会静默
# 接受 ruler_override=nothing 并完整跑完 prepare（generation 被 +1、builder
# 被实际调用、调用不抛错），下列各条断言逐一翻转。正例钉死：封闭只是接口
# 接线，默认与显式合法调用的 PreparedProblem 逐字段不变；incremental 私有
# 接线（_prepare_v1 直传 internal 通道）不被误封。
# ---------------------------------------------------------------------------
@testset "prepare_reference: closed public keyword surface" begin
    Abig = hc_big_panel()

    # 不绑定具体异常类型（MethodError / "unsupported keyword argument" 的
    # 呈现随 Julia 版本而异）；判据是"调用必须失败"，由 [P4] 正例保证
    # 正常路径可用，故 refused 的红绿是干净的。
    refused(f) = (try; f(); false; catch; true; end)

    # ---------- [P1] 三个 internal 通道即使 =nothing 也被公开入口拒绝 ----------
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3, ruler_override = nothing))
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3, scale_override = nothing))
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3, statistics_builder = nothing))
    # 非 nothing 的 internal 通道同样拒绝（若实现退化为"剔除三个名字后
    # 静默转发"的黑名单过滤，本条与上三条都会红）。
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3, ruler_override = zeros(9, 2)))

    # ---------- [P2] 拒绝先于 workspace generation 变化 ----------
    # _prepare_v1 的第一条语句就是 generation += 1（失败/部分 prepare 也先
    # bump，见 prepare.jl 的 lease 契约）；拒绝必须发生在它之前——被拒调用
    # 后 generation 原地不动。旧宽接口下本调用静默成功并 bump → 双断言红。
    ws = HC.FitWorkspace()
    gen0 = ws.generation[]
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3, workspace = ws,
                                            ruler_override = nothing))
    @test ws.generation[] == gen0

    # ---------- [P3] 拒绝先于 builder 执行 ----------
    # builder 若被调用会先留下痕迹再抛错；拒绝必须先于任何 builder 代码
    # ——痕迹必须为空（旧宽接口下 builder 会被实际调用 → 本条红）。
    calls = Int[]
    probe_builder = (args...) -> (push!(calls, 1); error("builder must not run"))
    @test refused(() -> HC.prepare_reference(Abig; F_folds = 3,
                                             statistics_builder = probe_builder))
    @test isempty(calls)

    # ---------- [P4] 正例：默认 / 显式全合法 kwargs / 私有 _prepare_v1 ----------
    # 三者逐字段一致——封闭入口只是接线，不改任何默认数学。
    p_def = HC.prepare_reference(Abig)
    p_kw  = HC.prepare_reference(Abig; ridge_alpha = nothing, F_folds = 3,
                                 ruler_stats = nothing, history_cache = nothing,
                                 alpha_initial = nothing, timing = nothing,
                                 workspace = nothing)
    p_priv = HC._prepare_v1(Abig; F_folds = 3)
    for p in (p_kw, p_priv)
        @test p.active_idx == p_def.active_idx
        @test p.T == p_def.T && p.N == p_def.N && p.N_universe == p_def.N_universe
        @test p.s1 == p_def.s1
        @test p.s_m == p_def.s_m
        @test p.s_perp == p_def.s_perp
        @test isequal(p.m, p_def.m)
        @test isequal(p.r, p_def.r)
        @test p.relative_embedding == p_def.relative_embedding
        @test p.observed == p_def.observed
        @test p.X_rel == p_def.X_rel
        @test p.B_m == p_def.B_m
        @test p.alive_now == p_def.alive_now
        @test p.ts_total == p_def.ts_total
        @test p.n_res == p_def.n_res && p.P_features == p_def.P_features
        @test p.F_folds == p_def.F_folds
        @test p.X_rel_stacked == p_def.X_rel_stacked
        @test p.macro_stats === p_def.macro_stats
        @test p.ws_owner === p_def.ws_owner === nothing
        @test p.ws_generation == p_def.ws_generation == 0
        @test p.stats.ranges == p_def.stats.ranges
        @test p.stats.full_xy ≈ p_def.stats.full_xy rtol = 1e-12
        @test p.stats.full_yy ≈ p_def.stats.full_yy rtol = 1e-12
        @test isequal(p.stats.full_xx, p_def.stats.full_xx)
        for f in 1:3
            @test p.stats.xy[f] ≈ p_def.stats.xy[f] rtol = 1e-12
            @test p.stats.yy[f] ≈ p_def.stats.yy[f] rtol = 1e-12
            @test isequal(p.stats.xx[f], p_def.stats.xx[f])
        end
    end

    # ---------- [P5] 正例：incremental 私有接线不被误封 ----------
    # _prepare_v1 自身的 internal 通道照常可用（incremental.jl
    # _prepare_current! 的调用形态）；用同源重算值走三个通道，产出数学与
    # reference 一致——封闭只发生在公开入口，私有接线原样保留。
    logs = log.(Abig)
    f_first = [something(findfirst(isfinite, view(logs, :, j)), size(Abig, 1) + 1)
               for j in 1:size(Abig, 2)]
    s_raw = HC.ruler(logs, f_first)
    p_ro = HC._prepare_v1(Abig; F_folds = 3, ruler_override = s_raw)
    @test p_ro.s1 == p_def.s1

    p_sc = HC._prepare_v1(Abig; F_folds = 3,
                          scale_override = (; s_m = nothing, s_perp = p_def.s_perp))
    @test p_sc.s_perp == p_def.s_perp && p_sc.s_m == p_def.s_m

    built_rows = HC.build_X_rel_stacked(p_def.X_rel, p_def.s_perp, p_def.ts_total;
                                        workspace = nothing)
    builder = (X, Y, Bm, m, ts) ->
        (; stats = HC.fold_sufficient_statistics(built_rows, Y, 3; workspace = nothing),
           macro_stats = nothing)
    p_bu = HC._prepare_v1(Abig; F_folds = 3, statistics_builder = builder)
    @test p_bu.stats.ranges == p_def.stats.ranges
    @test p_bu.stats.full_xy ≈ p_def.stats.full_xy rtol = 1e-12
    @test p_bu.stats.full_yy ≈ p_def.stats.full_yy rtol = 1e-12
    for f in 1:3
        @test p_bu.stats.xy[f] ≈ p_def.stats.xy[f] rtol = 1e-12
        @test p_bu.stats.yy[f] ≈ p_def.stats.yy[f] rtol = 1e-12
    end
    # lazy design：builder 路径不物化 X_rel_stacked（solve 按需重建，与
    # incremental 引擎同一语义）；batch 参照路径物化。
    @test p_bu.X_rel_stacked === nothing
    @test p_def.X_rel_stacked !== nothing
end
