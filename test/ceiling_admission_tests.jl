using Test, Random, LinearAlgebra
using KTrader
# M1: probe entry points live in the dev namespace; KTrader no longer
# defines or exports any probe symbol. Guard against re-including the module
# when several consumer test files run in one standard test process (a bare
# `include` would REDEFINE DevProbes and leave earlier bare-name imports
# bound to the old Collector type).
isdefined(@__MODULE__, :DevProbes) || include(joinpath(@__DIR__, "..", "dev", "probes.jl"))
using .DevProbes

# Ceiling admission 门禁与 throwing-sink 的判别力测试。
# 判别力要求（不止 @test_throws ArgumentError——错误数据也可能抛同类）：
#   1) admission 拒绝必须有语义证据：错误消息含 admission 标记
#      （"rejected before any data" / "dual-mode"），且 io 日志零 stage 行、
#      timing 零计数——证明拒绝发生在阶段之前（零 prep / 零 fit），
#      prefix 未被触碰（正常路径的 history_rows==t 另行断言）。
#   2) throwing-sink 双失败：工作函数（primary）与 stop observer 同时
#      失败时，捕获的是**原 primary**（消息精确匹配），observer 错误
#      单独记录在 timing.sink_error，绝不互换或吞掉。
@testset "ceiling admission gates (discriminating)" begin
    rng = MersenneTwister(11)
    signal = exp.(cumsum(0.01 .* randn(rng, 300, 2); dims = 1))

    # --- 正常路径参照：prefix 完整、prepare 不隐含 solve ---
    log = IOBuffer()
    out = DevProbes.prepare_stage_probe(signal; t_star = 299, io = log)
    txt = String(take!(log))
    @test out.history_rows == 299          # 全 prefix 1:299，不被裁
    @test out.prep.n_res > 0
    sb = (:eigen, :EB, :condition, :fracFFT, :OOF_fit, :OOF_predict, :OOF_residual)
    @test sum(out.timing.seconds[findfirst(==(b), KTrader.TIMING_BUCKETS)] for b in sb) == 0.0
    @test occursin("start", txt) && occursin("end", txt)

    # --- admission 拒绝：消息语义 + 零阶段证据 ---
    function admission_rejected(f, marker)
        io = IOBuffer()
        err = try
            f(io)
            nothing
        catch e
            e
        end
        err === nothing && return (false, "no error", "")
        msg = sprint(showerror, err)
        stage_txt = String(take!(io))
        zero_stages = !occursin("start", stage_txt) && !occursin("end", stage_txt)
        (err isa ArgumentError && occursin(marker, msg) && zero_stages, msg, stage_txt)
    end

    # warmup=true：拒绝在数据触碰前（io 零 stage 行 = 零 prep/零 fit 证据）
    ok, msg, _ = admission_rejected(io -> DevProbes.solve_stage_probe(out.prep;
        t_star = 299, warmup = true, io = io), "rejected before any data")
    @test ok
    # dual-mode：单一 mode 语义
    ok2, msg2, _ = admission_rejected(io -> DevProbes.solve_stage_probe(out.prep;
        t_star = 299, mode = :both, io = io), "dual-mode")
    @test ok2
    # t_star 不匹配：prepared input 不被改写消费
    ok3, msg3, _ = admission_rejected(io -> DevProbes.solve_stage_probe(out.prep;
        t_star = 298, io = io), "does not match")
    @test ok3
    # F_folds 不匹配
    ok4, msg4, _ = admission_rejected(io -> DevProbes.solve_stage_probe(out.prep;
        t_star = 299, F_folds = 2, io = io), "F_folds")
    @test ok4

    # --- throwing-sink：双失败捕获原 primary ---
    t = KTrader.DecisionTiming()
    # start observer 正常、stop observer 抛错：bucket 成功 + observer 死
    t.sink = (ev, bucket, stamp) -> ev === :stop && throw(ErrorException("stop-obs-boom"))
    @test_throws ErrorException("stop-obs-boom") KTrader.timed(() -> 1.0, t, :gram)
    # 双失败：primary 与 observer 同时抛——必须捕获原 primary（精确消息），
    # observer 错误单独记录，绝不互换。
    t2 = KTrader.DecisionTiming()
    t2.sink = (ev, bucket, stamp) -> ev === :stop && throw(ErrorException("obs-boom"))
    captured = try
        KTrader.timed(() -> error("primary-boom"), t2, :gram)
        "NOTHROWN"
    catch e
        sprint(showerror, e)
    end
    @test captured == "primary-boom"                    # 原 primary，非任意异常
    @test t2.sink_error !== nothing
    @test occursin("obs-boom", t2.sink_error)           # observer 单独记录
    # sink_error 首错保留不被覆盖
    t3 = KTrader.DecisionTiming()
    t3.sink = (ev, bucket, stamp) -> throw(ErrorException("first-boom"))
    try
        KTrader.timed(() -> error("p3"), t3, :gram)
    catch
    end
    @test occursin("first-boom", t3.sink_error)
    # 通知开销排除：seconds 只计工作
    t4 = KTrader.DecisionTiming()
    notified = Ref(0)
    t4.sink = (ev, bucket, stamp) -> (notified[] += 1)
    KTrader.timed(() -> sum(sin.(1:1000)), t4, :gram)
    i_gram = findfirst(==(:gram), KTrader.TIMING_BUCKETS)
    @test t4.seconds[i_gram] > 0
    @test notified[] == 2                                # start+stop 恰好两次
    # 默认无 sink：字节等同快路径
    t5 = KTrader.DecisionTiming()
    KTrader.timed(() -> sum(sin.(1:1000)), t5, :gram)
    @test t5.sink === nothing && t5.sink_error === nothing
    @test t5.seconds[i_gram] > 0
end
