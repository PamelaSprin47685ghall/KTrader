using Test, Random, LinearAlgebra
using KTrader

# IPO Gram 预算账本契约（src/incremental.jl embed_active!）。
#
# 审计缺陷一（2026-10-07 第3任评审静态发现）：IPO 扩维时 embed_active!
# 直接重建 Gram 字典、绕开 budget_allows!，且不更新 gram_bytes——
# (1) 预算跨线时照样分配扩维矩阵；(2) 预算内时账本停留在旧 N 记账。
# 审计缺陷二（2026-10-08 DevOps 实跑暴露、静态复核确认）：advance_exact!
# 创建 RawInferenceCore 时丢弃 initialize_inference 校验过的
# gram_budget / materialize_row_limit——kwargs 是死参数，core 永远用
# 默认 512MiB/4096，预算跨线与 row-limit 路由从未被调用方配置触发。
# 本文件用独立核算（不信任 core.gram_bytes，直接遍历 Grams 数
# sum(length(matrix)*sizeof(Float64))）让两个旧缺陷必红。
#
# 时序语义（fixture 验收日）：IPO 资产首个 finite 价格日与首个有效
# 相邻 daily return 的 active day 恰差 1——active 扩维与 embed_active!
# 发生在首个有效 return 日（bar 361），预算跨线只能在该日验收，
# 不能在首价日（bar 360）验收。
#
# 数学不变量：释放路径（demote 到物化）与聚合收缩是同一数学的两种
# 求和顺序；stats 与 batch oracle（_prepare_v1）在 incremental_tests.jl
# 的同一容差内一致；:resource_budget 路由返回的恰是 batch oracle 的
# prepared 输入。账本只计量持有的 FoldGram payload（V+Xc+R2），不含
# rows/runs/zsum_hist，不是瞬时 workspace，更不是 OS RSS 上限。

function held_gram_payload_bytes(core)
    total = 0
    for g in values(core.grams)
        for M in g.V;  total += length(M) * sizeof(Float64); end
        for M in g.Xc; total += length(M) * sizeof(Float64); end
        total += length(g.R2) * sizeof(Float64)
    end
    total
end

function budget_assert_stats(state, prices)
    p = KTrader._prepare_current!(state)
    oracle = KTrader._prepare_v1(prices; F_folds = state.F_folds)
    @test p.active_idx == oracle.active_idx
    @test p.ts_total == oracle.ts_total
    for name in (:full_xy, :full_yy)
        @test getproperty(p.stats, name) ≈ getproperty(oracle.stats, name) atol=1e-9 rtol=1e-11
    end
    if p.stats.full_xx !== nothing
        @test p.stats.full_xx ≈ oracle.stats.full_xx atol=1e-9 rtol=1e-11
    end
    for f in 1:state.F_folds
        @test p.stats.xy[f] ≈ oracle.stats.xy[f] atol=1e-9 rtol=1e-11
        @test p.stats.yy[f] ≈ oracle.stats.yy[f] atol=1e-9 rtol=1e-11
        p.stats.xx === nothing ||
            (@test p.stats.xx[f] ≈ oracle.stats.xx[f] atol=1e-9 rtol=1e-11)
    end
    p
end

@testset "IPO gram budget ledger (embed_active! payload accounting)" begin
    rng = MersenneTwister(4242)
    returns = 0.007randn(rng, 680, 3)
    prices = exp.(cumsum(returns; dims = 1))
    # Asset 3: first finite PRICE at bar 360; first finite daily RETURN
    # (and the embed) at bar 361 — one day later by definition.
    ipo = copy(prices); ipo[1:359, 3] .= NaN

    N2 = KTrader.gram_bytes_for(2)
    N3 = KTrader.gram_bytes_for(3)
    # 同步 N=2 前缀：1 个 unique mask × 3 folds = 恰好 3 个 Grams 饱和
    # 3*N2；N2->N3 的 embed 目标占用是 3*N3。

    @testset "within budget: every advance keeps the ledger synced" begin
        # 6*N3 容纳 3 个 embed 后旧 Grams + 新 mask 的 fold3 Gram（见
        # 下方 4*N3 的推导）。
        state = initialize_inference(ipo[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 6 * N3)
        for t in 301:359
            advance_exact!(state, view(ipo, t, :))
        end
        core = state.core
        @test core.gram_bytes == 3 * N2                        # IPO 前账本
        @test core.gram_bytes == held_gram_payload_bytes(core)
        for t in 360:680
            advance_exact!(state, view(ipo, t, :))
            # 缺陷一在实际 embed 日（t=361）起红：embed 后账本停留在
            # 3*N2，而实际 payload 已是 newN 尺寸。
            @test state.core.gram_bytes == held_gram_payload_bytes(state.core)
        end
        core = state.core
        @test state.counters[:embeds]==1
        @test state.counters[:rebuilds] == 0                   # 扩维无 rebuild/回放
        # 4 个 Grams = 3 个 embed 后旧 Grams（[T,T,F] × 3 folds）+ 1 个
        # 新 mask fold3 Gram：transition 后 256 行 boundary 窗口内的
        # rows 全部物化，窗口外的新 rows（行号 360+）全部落在 fold3
        # （n=423 时 fold3 = 283:423）；fold1/fold2 没有任何 aggregated
        # 新 row，故不创建 Gram——Gram 由“该 fold 存在 aggregated 贡献”
        # 驱动，不为凑 key 而建。
        @test core.gram_bytes == 4 * N3
        for g in values(core.grams)
            @test size(first(g.V)) == (3, 3)                   # 扩维真实发生
            @test size(g.R2) == (3, 3)
        end
        @test !core.resource_limited                           # 预算内不误报
        budget_assert_stats(state, ipo)                        # exact stats
        @test state.last_reason === nothing
        @test state.counters[:fast] > 0
    end

    @testset "over budget at the embed: release, demote, exact, resumable" begin
        # N=2 阶段 3 个 Grams 恰好用满预算；N2->N3 embed 目标 3*N3 跨线。
        # （缺陷二旧代码下 core 用默认 512MiB，embed 永不跨线——本
        # testset 与 route testset 一起让该缺陷红。）
        state = initialize_inference(ipo[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 3 * N2)
        for t in 301:359
            advance_exact!(state, view(ipo, t, :))
        end
        core = state.core
        @test core.gram_bytes == 3 * N2
        @test core.gram_bytes == held_gram_payload_bytes(core)
        @test all(r -> r.aggregated, core.rows)   # IPO 前：同步 mask 全聚合
        # 首价日（bar 360）不是 embed 日：P_359 缺失使当日 return 仍为
        # NaN，active 不变、无扩维、无释放——预算跨线只能在首个有效
        # 相邻 daily return 日（bar 361）验收。
        @test advance_exact!(state, view(ipo, 360, :)) === state
        core = state.core
        @test state.counters[:embeds] == 0
        @test state.active == [1, 2]
        @test length(core.grams) == 3
        @test core.gram_bytes == 3 * N2
        @test all(r -> r.aggregated, core.rows)
        @test !core.resource_limited
        # bar 361：首个有效相邻 daily return——active 扩维、embed 跨线、
        # 释放。跨线 embed 是纯路由决策：不抛、无撕裂中间态，advance
        # 正常返回。
        @test advance_exact!(state, view(ipo, 361, :)) === state
        core = state.core
        @test state.counters[:embeds] == 1
        @test state.counters[:rebuilds] == 0
        # 释放语义：旧 Grams 全部回收、账本归零；旧 rows 全部 demote；
        # 当天的新 training row 本身是 boundary row（target 开新 run），
        # 同样物化——释放日不创建任何 Gram。
        @test isempty(core.grams)                # 缺陷一旧代码：照样扩维 → 红
        @test core.gram_bytes == 0
        @test core.gram_bytes == held_gram_payload_bytes(core)
        @test all(r -> !r.aggregated, core.rows)
        @test core.resource_limited
        @test state.active == [1, 2, 3]          # 坐标状态照常演进
        # 继续演进：账本始终同步、无 rebuild；boundary 窗口（256 行）
        # 之外的新行在释放出的预算内重新聚合——非永久降级。
        for t in 362:680
            advance_exact!(state, view(ipo, t, :))
            @test state.core.gram_bytes == held_gram_payload_bytes(state.core)
        end
        @test state.counters[:rebuilds] == 0
        @test any(r -> r.aggregated, core.rows)
        # demote 后的混合路径 stats 与 batch oracle 同容差（同一数学）。
        budget_assert_stats(state, ipo)
    end

    @testset "demoted mass routes to reference prepare via :resource_budget" begin
        # 缺陷二旧代码下 materialize_row_limit=10 从未到达 core（默认
        # 4096 吞掉路由）——本 testset 让该缺陷红。
        state = initialize_inference(ipo[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 3 * N2,
                                     materialize_row_limit = 10)
        for t in 301:680
            advance_exact!(state, view(ipo, t, :))
        end
        # demote + IPO boundary 使物化行数超过 limit → 路由到 batch
        # oracle（exact reference prepare），原因正确记录；路由后返回
        # 的 prepared 输入与 oracle 逐字段一致。
        budget_assert_stats(state, ipo)
        @test state.last_reason == :resource_budget
        @test get(state.fallback_reasons, :resource_budget, 0) > 0
    end

    @testset "multi-IPO and insertion reordering keep the ledger synced" begin
        multi = copy(prices)
        multi[1:399, 2] .= NaN    # asset 2 first price bar 400, embed bar 401
        multi[1:499, 3] .= NaN    # asset 3 first price bar 500, embed bar 501
        state = initialize_inference(multi[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 10^9)
        for t in 301:680
            advance_exact!(state, view(multi, t, :))
            state.counters[:embeds] > 0 &&
                (@test state.core.gram_bytes == held_gram_payload_bytes(state.core))
        end
        @test state.counters[:embeds] == 2
        @test state.counters[:rebuilds] == 0
        @test state.core.gram_bytes == held_gram_payload_bytes(state.core)
        budget_assert_stats(state, multi)

        # 插入重排：[1,3] -> [1,2,3]，旧坐标 2（资产 3）移位到 3。
        insert = copy(prices)
        insert[1:399, 3] .= NaN   # asset 3 first price bar 400, embed bar 401
        insert[1:449, 2] .= NaN   # asset 2 first price bar 450, embed bar 451
        state = initialize_inference(insert[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 10^9)
        for t in 301:680
            advance_exact!(state, view(insert, t, :))
        end
        @test state.active == [1, 2, 3]
        @test state.counters[:embeds] == 2
        @test state.counters[:rebuilds] == 0
        @test state.core.gram_bytes == held_gram_payload_bytes(state.core)
        budget_assert_stats(state, insert)
    end

    @testset "checkpoint carries an independent synchronized ledger" begin
        state = initialize_inference(ipo[1:300, :]; ridge_alpha = 1.0,
                                     gram_budget = 10^9)
        # advance 到实际 embed 日（bar 361）之后，checkpoint 才携带
        # embed 后的账本。
        for t in 301:361
            advance_exact!(state, view(ipo, t, :))
        end
        @test state.counters[:embeds] == 1
        cp = inference_checkpoint(state)
        @test cp.core.grams !== state.core.grams
        @test cp.core.gram_bytes == state.core.gram_bytes
        @test cp.core.gram_bytes == held_gram_payload_bytes(cp.core)
        for t in 362:380
            advance_exact!(state, view(ipo, t, :))
        end
        # 原状态继续演进不改变 checkpoint 的账本（deepcopy 独立）。
        @test cp.core.gram_bytes == held_gram_payload_bytes(cp.core)
    end
end
