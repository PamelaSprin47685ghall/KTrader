using Test, Random, LinearAlgebra
using KTrader

# timed 三分支一致透传 f 的返回值（快路径曾裸 return 丢值）。
@testset "timed return-value passthrough contract" begin
    # 标量
    @test KTrader.timed(() -> 42, nothing, :eigen) == 42
    t_fast = KTrader.DecisionTiming()  # sink=nothing → 快路径
    @test KTrader.timed(() -> 42, t_fast, :eigen) == 42
    t_obs = KTrader.DecisionTiming()
    t_obs.sink = (ev, b, s) -> nothing
    @test KTrader.timed(() -> 42, t_obs, :eigen) == 42
    # 元组
    @test KTrader.timed(() -> (1.5, 2.0), t_fast, :EB) == (1.5, 2.0)
    # NamedTuple（真实 ridge_spectrum 返回类型）
    nt = (values=[1.0, 2.0], basis=zeros(2, 2))
    @test KTrader.timed(() -> nt, t_fast, :eigen) === nt
    # 计时仍正确累加
    @test t_fast.seconds[findfirst(==(:eigen), KTrader.TIMING_BUCKETS)] >= 0
    # 主异常传播（不被 finally 吞掉）
    @test_throws ErrorException KTrader.timed(() -> error("boom"), t_fast, :eigen)
end
