using Test, Random, LinearAlgebra
using KTrader
# M1: probe entry points live in the dev namespace.
isdefined(@__MODULE__, :DevProbes) || include(joinpath(@__DIR__, "..", "dev", "probes.jl"))
using .DevProbes

# model-return API 微验证 + scenario/Kelly 接口串接（固化自 DevOps 受控
# 验证；小合成输入秒级，不假测性能）。R6 实际 BLAS6 companion 的最窄
# 门禁见 conditioned_eb_tests.jl 的 BLAS6 testset。
@testset "solve_stage_probe model-return API and downstream wiring" begin
    rng = MersenneTwister(11)
    signal = exp.(cumsum(0.01 .* randn(rng, 300, 2); dims = 1))
    prep = DevProbes.prepare_stage_probe(signal; t_star = 299, io = IOBuffer()).prep

    # 成功 → V1Model（非 nothing）；sink 不进 model
    res = DevProbes.solve_stage_probe(prep; t_star = 299, io = IOBuffer())
    @test res.model isa KTrader.V1Model
    @test res.succeeded == 1
    @test res.model !== res.timing            # sink/timing 不在 model 里

    # 失败 → model=nothing（内部 catch，不 throw）：污染 full_xy 为 NaN
    bad = deepcopy(prep)
    bad.stats.full_xy .= NaN
    res_bad = DevProbes.solve_stage_probe(bad; t_star = 299, io = IOBuffer())
    @test res_bad.model === nothing
    @test res_bad.succeeded == 0

    # 下游一次 S=10 scenario + Kelly 串接（仅接口）
    X = KTrader.generate_scenarios_v1(res.model; S = 10, rng = MersenneTwister(1))
    @test size(X) == (10, 2)
    w = KTrader.scenario_weights(X, res.model.active_indices,
                                 trues(length(res.model.active_indices)))
    @test all(w .>= 0)
    @test sum(w) <= 1 + 1e-8
end
