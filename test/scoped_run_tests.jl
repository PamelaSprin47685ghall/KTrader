using Test

# scoped_run 进程护栏测试。契约见 bin/scoped_run.sh 头部（第 1-7 条）。
# 经验来源：工具层"中止"后 Julia 残留 15 分钟的事故——本文件验证的是
# 真实资源后果，不是文本。
#
# ============================ 相入口契约 ====================================
# standalone 运行：julia test/scoped_run_tests.jl [--phase=<name>]
#   - 无参数 = ALL（全部相按序执行）；
#   - --phase=<name> 只跑一个相（唯一支持的参数形态；重复出现即硬拒）；
#   - 未知相 / 未知参数在任何进程 spawn 之前 error（硬拒）。
# 标准 runtests include 模式：始终 ALL——PROGRAM_FILE 不是本文件时不
# 解析 ARGS、不继承任何 partial 选择（include 方的参数属于 runtests，
# 与本文件无关）。
# 相清单与静态预算（单机估算、未实测；供 operator 规划 scoped deadline）：
#   selftest      ~15s  bash --self-test 全场景回归（场景 4 满宽限）
#   baseline      ~ 6s  G1（TERM-exit-0 锁死）/G2（>60s 拒启动）/G3（透传）
#   owner-term    ~2.5s 只 SIGTERM wrapper 自身
#   owner-int     ~2.5s 只 SIGINT  wrapper 自身
#   owner-hup     ~2.5s 只 SIGHUP  wrapper 自身
#   clock-mutant  ~ 8s  date-mutant 假墙钟下 deadline 失效（红）
#   clock-mono    ~ 3s  同假钟环境生产单调源照常 timeout（绿）
#   clock-fail    ~1.5s 受控坏 awk → 125 + MONO-CLOCK-FAILED
#   clock-saturate~3.5s %d 32 位饱和模拟 → 生产 %.0f 照常 timeout
#   clock-start   ~ 7s  start 坏钟（副本桩）：125 + 有界确认组净
#   clock-cleanup ~2.5s 收尾期坏钟（副本桩，rc 0/7 两例）：贯穿 125
#   clock-elapsed ~2.5s 仅最终 elapsed 坏（副本桩，rc 0/7 两例）：unknown
#   clock-transient~2.5s monitor 瞬态坏一次后恢复：sticky 仍 125
#   clock-window-poll~8s  窗口轮询坏钟+抗TERM live child：无钟回退125
#   clock-negctl  ~1.5s 吞码变异副本：同款 125 断言必然红（判别力）
#   command-admission~1s  --rss-guard 后无 command：spawn/log 前 RC2
#   ALL           ~75s + Julia 启动
#
# ========================== G1 持久契约（锁死） =============================
# TIMEOUT-EXCEEDED 行必须含 child_rc=0：这是子进程真实响应 TERM 后以
# exit 0 退出的持久证据（由 wrapper 的 wait 对 direct child 的自然收割
# 得到），而非被 KILL 杀成 137 再被 rc=124 掩盖的假绿。该断言位于
# baseline 相，与 !occursin("child_rc=137") 成对；任何实现演进不得删
# 除或放宽——这是 timeout-never-success 语义的可执行记忆。
#
# ============================ C1 证据质量 ====================================
# owner-* 相的证据链全部由本文件直接持有，不经任何临时 harness：
#   - direct Process：spawn_wrapper 返回 Julia 直属 Process，断言用的
#     exitcode 来自 wait(proc) 的自然父收割（超窗兜底 KILL 时 exitcode
#     变为 9 ≠ 128+sig，断言红——判别本身保留）；
#   - owned PGID 精确绑定：组 leader（setsid 后 PGID == leader PID）自
#     己把 $$ 写入 pidfile，harness 从文件读回——不 pgrep、不猜；
#   - 组外哨兵：独立 PID 的直属 sleep Process，取消后断言仍 Running；
#   - finally 兜底：按已绑定 PGID `kill -KILL -- -PGID`、kill 哨兵并
#     wait、kill+wait wrapper——不跨 shell、不留 stdout 持有者（wrapper
#     的 stdout/stderr 全部导入 devnull，日志真源是 wrapper 自己写的
#     log 文件）。
#
# ============================ 时钟口径 ======================================
# mono_ms 用 printf "%.0f"（awk 全程 double）：2^53 ms ≈ 9.0e15 ms ≈
# 约 28.5 万年——远超任何 runtime，无 32 位饱和路径（%d 在 mawk 类
# 实现饱和于 2^31-1 ms ≈ 24.8 天 uptime）。bin 头注释中"≈285 年"在
# Engineer 交付本文件时为数量级笔误（正确约 285,000 年）且当时 bin
# 冻结未改——该时点陈述保留为历史；此后 DevOps 已按普通注释修复
# 将 bin 头改为"≈28.5 万年"（不改逻辑），本注释不再跟踪 bin 当前
# 字节，以 bin 自身文件为准。
# 假墙钟（clock-mutant/clock-mono）：PATH 前置假 `date +%s`，内部以
# /proc/uptime 单调判定时刻（前 FAKE_HOLD_MS 返回真 epoch、之后回拨
# FAKE_BACK_S）——不真实修改机器时钟、不依赖 root/libfaketime。
# 受控故障钟（clock-fail）：PATH 前置计数转发假 awk（前 N 次转发真
# awk、之后失败）——注入只存在于该相进程的 PATH。
# 饱和判别（clock-saturate）：PATH 前置假 awk——程序含 %d 输出饱和
# 常数（不读输入文件）、否则转发真 awk。不动真实 /proc。

# ---- 相定义与入口解析（任何进程 spawn 前完成，未知相/参数硬拒）----
const PHASES = ["selftest", "baseline", "owner-term", "owner-int", "owner-hup",
                "clock-mutant", "clock-mono", "clock-fail", "clock-saturate",
                "clock-start", "clock-cleanup", "clock-elapsed",
                "clock-transient", "clock-window-poll", "clock-negctl",
                "command-admission"]

# standalone 判别：直接运行本文件时 PROGRAM_FILE == @__FILE__；
# include 模式（runtests.jl）PROGRAM_FILE 是入口文件 → 始终 ALL。
const STANDALONE = abspath(PROGRAM_FILE) == abspath(@__FILE__)
const PHASE = if STANDALONE
    # let 引入显式作用域：const 右侧 if 表达式内的顶层 soft-scope 赋值
    # 会创建 if 表达式的局部变量，for 作用域内读取即 UndefVarError（已
    # 由真实加载红证实）；let 局部变量对 for 内读写均词法可见。
    let phase_arg = nothing
        for a in ARGS
            if startswith(a, "--phase=")
                phase_arg === nothing ||
                    error("duplicate --phase argument; refusing before any process spawn")
                phase_arg = String(split(a, "="; limit=2)[2])
            else
                error("unknown argument '$a' (only --phase=<name> is supported); " *
                      "refusing before any process spawn")
            end
        end
        phase_arg === nothing ? "ALL" : phase_arg
    end
else
    "ALL"   # include 模式：始终全部，不继承 partial 选择
end
PHASE == "ALL" || PHASE in PHASES ||
    error("unknown phase '$PHASE' (valid: ALL, $(join(PHASES, ", "))); " *
          "refusing before any process spawn")

# ---- 公共 helper -----------------------------------------------------------
const script = joinpath(@__DIR__, "..", "bin", "scoped_run.sh")
isfile(script) || error("scoped_run.sh missing")

# spawn_wrapper — 以直属 Process 启动 wrapper，stdout/stderr 全部导入
# devnull（日志真源是 wrapper 写的 log 文件；harness 管道不参与 wrapper
# 生命周期，不遗留 stdout 持有者）。返回 Base.Process——direct wait
# 取自然退出码的唯一对象。
function spawn_wrapper(cmd::Cmd)
    return run(pipeline(cmd, stdout=devnull, stderr=devnull); wait=false)
end

# spawn_sentinel — 组外哨兵：独立 PID 的直属 sleep Process。
function spawn_sentinel()
    return run(pipeline(`sleep 60`, stdout=devnull, stderr=devnull); wait=false)
end

# wait_pidfile — 等组 leader 自记 PID（owned PGID 的绑定证据）出现；
# 超窗返回 0（断言将红，finally 跳过组清理）。
function wait_pidfile(pidf::AbstractString; window::Float64=5.0)
    t = time()
    while !isfile(pidf) && time() - t < window
        sleep(0.05)
    end
    return isfile(pidf) ? parse(Int, strip(read(pidf, String))) : 0
end

# wait_bounded — 有界等待 wrapper 自然退出；超窗 KILL 兜底后 wait
# （harness 自身绝不无界等待；KILL 后 exitcode≠自然值，断言红）。
function wait_bounded(proc; window::Float64)
    t = time()
    while process_running(proc) && time() - t < window
        sleep(0.1)
    end
    if process_running(proc)
        kill(proc, 9)
    end
    wait(proc)
end

# kill_group — finally 兜底：按已绑定 PGID 直接 KILL 整组
# （不 pgrep、不跨 shell）。
kill_group(pgid::Int) = run(ignorestatus(`kill -KILL -- -$(pgid)`))

# no_live_in_group — owned 组无 live 成员（zombie 由内核收，允许）。
function no_live_in_group(pgid::Int)
    psout = read(ignorestatus(`ps -o stat= -g $(pgid)`), String)
    n = count(l -> !isempty(strip(l)) && !startswith(strip(l), "Z"), split(psout, '\n'))
    return n == 0
end

# make_fake_date — 假墙钟 date（PATH 注入；内部 /proc/uptime 单调判定）。
function make_fake_date(dir::AbstractString)
    f = joinpath(dir, "date")
    write(f, """
#!/usr/bin/env bash
# fake date (test-only): +%s returns \$FAKE_EPOCH for the first
# \$FAKE_HOLD_MS (mono /proc/uptime), then \$FAKE_EPOCH-\$FAKE_BACK_S
# (simulated wall-clock rollback). Other invocations: error.
if [[ \${1:-} != "+%s" || \$# -ne 1 ]]; then
    echo "fake-date: only 'date +%s' supported" >&2; exit 3
fi
now=\$(awk '{printf "%d", int(\$1*1000)}' /proc/uptime)
if [[ ! -f "\$FAKE_STATE" ]]; then printf '%s' "\$now" > "\$FAKE_STATE"; fi
base=\$(cat "\$FAKE_STATE" 2>/dev/null || printf '%s' "\$now")
if (( now - base >= FAKE_HOLD_MS )); then
    echo \$(( FAKE_EPOCH - FAKE_BACK_S ))
else
    echo "\$FAKE_EPOCH"
fi
""")
    chmod(f, 0o755)
    return f
end

# make_mutant — 生产源复制到 dir、仅把 mono_ms 换成 date 计时（不覆写
# 生产）。目标必须与生产 mono_ms 完整函数体逐字一致；不命中即 error
# （fail-loudly，防 mutant==生产的假绿）。
function make_mutant(prod_script::AbstractString, dir::AbstractString)
    mono_line = "mono_ms() {\n    local v\n    v=\$(awk '{printf \"%.0f\", \$1*1000}' /proc/uptime 2>/dev/null) || return 1\n    [[ -n \$v && \$v =~ ^[0-9]+\$ ]] || return 1\n    (( v > 0 )) || return 1\n    echo \"\$v\"\n}"
    date_line = "mono_ms() {\n    echo \$(( \$(date +%s) * 1000 ))\n}"
    src = read(prod_script, String)
    m = replace(src, mono_line => date_line)
    occursin(date_line, m) && !occursin(mono_line, m) ||
        error("[clock-mutant] mutant generation failed: mono_ms drifted in production source")
    p = joinpath(dir, "mutant_scoped_run.sh")
    write(p, m)
    return p
end

# real_awk_path — 在 PATH 污染前解析真 awk 的绝对路径。
real_awk_path() = strip(read(`bash -c "command -v awk"`, String))

# ---- 相实现 ----------------------------------------------------------------

function phase_selftest()
    println("[phase=selftest] begin — scope: bash --self-test regression " *
            "(success/timeout/leader-exit-first/term-resistant + ALL PASS, no FAIL)")
    @testset "[phase=selftest] bash --self-test regression" begin
        out = read(`bash $script --self-test`, String)
        @test occursin("self-test success: PASS", out)
        @test occursin("self-test timeout: PASS", out)
        @test occursin("self-test leader-exit-first: PASS", out)
        @test occursin("self-test term-resistant: PASS", out)
        @test occursin("ALL PASS", out)
        @test !occursin("FAIL", out)
    end
    println("[phase=selftest] end")
end

function phase_baseline()
    println("[phase=baseline] begin — scope: G1 TERM-exit-0 LOCKED " *
            "(child_rc=0 present, child_rc=137 absent), G2 >60s refuse-to-start, G3 rc passthrough")
    @testset "[phase=baseline] G1/G2/G3 (locked contract)" begin
        # [G1] 持久契约：TERM 后 exit 0 的子进程绝不能让 deadline 违约
        # 报绿。child_rc=0 锁死反例为真（真实 TERM 路径，非 KILL 假绿）。
        tmp1 = joinpath(tempdir(), "scoped_g1_$(getpid()).log")
        cmd1 = `bash $script 15 $tmp1 --rss-guard=2048 bash -c 'trap "exit 0" TERM; sleep 30'`
        r1 = run(pipeline(ignorestatus(cmd1), devnull))
        @test r1.exitcode == 124                 # timeout => 124, never 0
        log1 = read(tmp1, String)
        @test occursin("TIMEOUT-EXCEEDED", log1)
        @test occursin("child_rc=0", log1)       # TERM answered with exit 0 (real trap path)
        @test !occursin("child_rc=137", log1)    # not KILLed into 137 then masked
        rm(tmp1; force=true)

        # [G2] >60s hard cap: the guard refuses to START (never silently runs).
        r2 = run(pipeline(ignorestatus(`bash $script 100 /dev/null true`), devnull))
        @test r2.exitcode != 0

        # [G3] normal exit passthrough unchanged (a real short command).
        tmp3 = joinpath(tempdir(), "scoped_g3_$(getpid()).log")
        r3 = run(pipeline(ignorestatus(`bash $script 20 $tmp3 --rss-guard=2048 true`), devnull))
        @test r3.exitcode == 0
        rm(tmp3; force=true)
    end
    println("[phase=baseline] end")
end

# phase_owner — 单信号 owner 取消相（每个信号一个独立相，不凑 batch）。
# 证据链：direct Process 自然父 wait 取 128+sig；owned PGID 由组 leader
# 自记 pidfile 精确绑定；组外哨兵独立 PID；finally 清子组与哨兵。
function phase_owner(sig::Int, name::String)
    ph = "owner-$(lowercase(name))"
    println("[phase=$(ph)] begin — scope: SIG$(name) to wrapper PID only; asserts " *
            "wrapper natural rc=128+$(sig) via direct wait, CANCELLED-BY-SIGNAL sig=$(name), " *
            "owned PGID (leader-recorded) has no live member, sentinel PID untouched, " *
            "no TIMEOUT (cancel at ~1.5s << hard)")
    @testset "[phase=$(ph)] owner cancel: SIG$(name) to wrapper only" begin
        tmp = joinpath(tempdir(), "scoped_owner_$(name)_$(getpid()).log")
        pidf = tmp * ".pgid"
        proc = nothing
        sentinel = nothing
        pgid = 0
        try
            # 组 leader 记录自身 PID（setsid 组的 PGID == leader PID）；
            # 组内留 sleep 300：取消若未清理组，它将长期存活（可观测）。
            inner = "echo \$\$ > $pidf; sleep 300"
            proc = spawn_wrapper(Cmd(["bash", script, "16", tmp, "bash", "-c", inner]))
            pgid = wait_pidfile(pidf)
            @test pgid != 0
            sleep(0.5)  # 调度保险（取消 handler 先于 spawn 已安装，契约第 5 条）
            sentinel = spawn_sentinel()
            kill(proc, sig)
            # 取消清理 ≤ 共享窗口 TERM_GRACE+REAP_BUDGET=10s；14s 未自然
            # 退则兜底 KILL（exitcode 变 9 ≠ 128+sig → 断言红）。
            wait_bounded(proc; window=14.0)
            @test proc.exitcode == 128 + sig           # 取消永不报绿
            log = isfile(tmp) ? read(tmp, String) : ""
            @test occursin("CANCELLED-BY-SIGNAL sig=$(name)", log)
            @test !occursin("TIMEOUT-EXCEEDED", log)   # 1.5s 取消，远未到 hard=6s
            @test no_live_in_group(pgid)
            @test process_running(sentinel)                 # 哨兵未受伤
        finally
            # harness 自清理：无论断言红绿，绝不留长时孤儿（旧行为回归时
            # 组内 sleep 300 由这里按已绑定 PGID 兜底 KILL）。
            if pgid != 0
                kill_group(pgid)
            end
            if sentinel !== nothing
                kill(sentinel, 9)
                wait(sentinel)
            end
            if proc !== nothing && process_running(proc)
                kill(proc, 9)
                wait(proc)
            end
            rm(tmp; force=true)
            rm(pidf; force=true)
        end
    end
    println("[phase=$(ph)] end")
end

# [clock-mutant] date 计时 mutant 在假墙钟下 deadline 失效（红）。
function phase_clock_mutant()
    println("[phase=clock-mutant] begin — scope: date-mutant (mono_ms→date) under " *
            "fake wall-clock rollback must FAIL its deadline — no TIMEOUT-EXCEEDED by " *
            "6s checkpoint, harness TERM finishes it (rc=143); owned group cleaned")
    @testset "[phase=clock-mutant] date-mutant red under fake rollback" begin
        fakebin = mktempdir()
        try
            make_fake_date(fakebin)
            mutant = make_mutant(script, fakebin)
            mlog = joinpath(fakebin, "c2.log")
            mpidf = joinpath(fakebin, "c2.pgid")
            state = joinpath(fakebin, "c2.state")
            proc = nothing
            pgid = 0
            try
                inner = "echo \$\$ > $mpidf; exec sleep 30"
                cmd = Cmd(["bash", mutant, "12", mlog, "bash", "-c", inner])
                env = Dict("PATH" => fakebin * ":" * ENV["PATH"],
                           "FAKE_STATE" => state,
                           "FAKE_EPOCH" => string(floor(Int, time())),
                           "FAKE_HOLD_MS" => "800",   # 0.8s 后墙钟回拨（早于 hard=2s）
                           "FAKE_BACK_S" => "3600")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(mpidf)
                @test pgid != 0
                # 6s 检查点：mutant 若计时正常，2s（hard）就该报 TIMEOUT；
                # 假墙钟回拨把它废掉——6s 时组仍在跑，由 harness TERM 收尾。
                sleep(6.0)
                kill(proc, 15)
                wait_bounded(proc; window=14.0)
                @test proc.exitcode == 143                    # 取消收尾，非 timeout 124
                log = isfile(mlog) ? read(mlog, String) : ""
                @test occursin("CANCELLED-BY-SIGNAL sig=TERM", log)
                @test !occursin("TIMEOUT-EXCEEDED", log)      # date 坏例核心：回拨废掉 deadline
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(fakebin; force=true, recursive=true)
        end
    end
    println("[phase=clock-mutant] end")
end

# [clock-mono] 同一假钟环境跑生产脚本：/proc/uptime 单调，hard 时刻照常
# TERM→124（绿）。若生产误用 date，0.3s 回拨即废掉 deadline → 红。
function phase_clock_mono()
    println("[phase=clock-mono] begin — scope: production script under the SAME fake " *
            "wall clock — /proc/uptime monotonic source fires at hard (2s) → rc=124 + " *
            "TIMEOUT-EXCEEDED, no CANCELLED; owned group cleaned")
    @testset "[phase=clock-mono] monotonic source green under fake rollback" begin
        fakebin = mktempdir()
        try
            make_fake_date(fakebin)
            slog = joinpath(fakebin, "c3.log")
            spidf = joinpath(fakebin, "c3.pgid")
            state = joinpath(fakebin, "c3.state")
            proc = nothing
            pgid = 0
            try
                inner = "echo \$\$ > $spidf; exec sleep 30"
                cmd = Cmd(["bash", script, "12", slog, "bash", "-c", inner])
                env = Dict("PATH" => fakebin * ":" * ENV["PATH"],
                           "FAKE_STATE" => state,
                           "FAKE_EPOCH" => string(floor(Int, time())),
                           "FAKE_HOLD_MS" => "300",   # 0.3s 后回拨：若生产误用 date，此处即废
                           "FAKE_BACK_S" => "3600")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(spidf)
                @test pgid != 0
                wait_bounded(proc; window=14.0)
                @test proc.exitcode == 124                    # 单调源不受假墙钟影响
                log = isfile(slog) ? read(slog, String) : ""
                @test occursin("TIMEOUT-EXCEEDED", log)
                @test !occursin("CANCELLED", log)
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(fakebin; force=true, recursive=true)
        end
    end
    println("[phase=clock-mono] end")
end

# [clock-fail] 受控时钟失败：坏 awk → 显式 fail-closed（125，不挂死）。
# 前 FAKE_AWK_OK_CALLS 次转发真 awk（start 成功），之后失败——
# phase=monitor 的 MONO-CLOCK-FAILED；收尾远快于 deadline。
function phase_clock_fail()
    println("[phase=clock-fail] begin — scope: counting fake awk (forwards first " *
            "FAKE_AWK_OK_CALLS then fails) → rc=125 + MONO-CLOCK-FAILED phase=monitor, " *
            "elapsed < deadline (fail-closed, not clock-dead hang); owned group cleaned")
    @testset "[phase=clock-fail] controlled clock failure → fail-closed 125" begin
        badbin = mktempdir()
        try
            real_awk = real_awk_path()
            @test !isempty(real_awk)   # REAL_AWK 解析失败则本相无法注入，红
            badawk = joinpath(badbin, "awk")
            write(badawk, """
#!/usr/bin/env bash
# fake awk (test-only): forwards the first \$FAKE_AWK_OK_CALLS invocations
# to the real awk (\$REAL_AWK, absolute path resolved before PATH pollution),
# then fails — a controlled mid-monitor mono_ms clock failure. No host
# clock change; the injection lives only in this scenario's PATH.
n=\$(cat "\$FAKE_AWK_STATE" 2>/dev/null || printf '0')
printf '%s' "\$(( n + 1 ))" > "\$FAKE_AWK_STATE"
if (( n < FAKE_AWK_OK_CALLS )); then
    exec "\$REAL_AWK" "\$@"
fi
exit 1
""")
            chmod(badawk, 0o755)
            clog = joinpath(badbin, "c4.log")
            cpidf = joinpath(badbin, "c4.pgid")
            awkstate = joinpath(badbin, "c4.awkcount")
            proc = nothing
            pgid = 0
            try
                t_begin = time()
                inner = "echo \$\$ > $cpidf; exec sleep 30"
                cmd = Cmd(["bash", script, "12", clog, "bash", "-c", inner])
                env = Dict("PATH" => badbin * ":" * ENV["PATH"],
                           "REAL_AWK" => real_awk,
                           "FAKE_AWK_STATE" => awkstate,
                           "FAKE_AWK_OK_CALLS" => "2")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(cpidf)
                @test pgid != 0
                # 无钟挂死则超窗 KILL → exitcode=9 ≠ 125 → 红。
                wait_bounded(proc; window=14.0)
                elapsed = time() - t_begin
                @test proc.exitcode == 125                    # 显式 fail-closed，非绿非挂
                @test elapsed < 12.0                          # 未跑满 deadline：不是永不超时
                log = isfile(clog) ? read(clog, String) : ""
                @test occursin("MONO-CLOCK-FAILED", log)
                @test occursin("phase=monitor", log)          # start 成功后监控期坏钟
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(badbin; force=true, recursive=true)
        end
    end
    println("[phase=clock-fail] end")
end

# [clock-saturate] %d 饱和判别：大 uptime 下 mono_ms 不得输出 32 位饱和
# 常数。假 awk：程序含 %d → 输出 2147483647（不读输入）；否则转发。
# 生产 %.0f → 正常计时 → 2s hard 真实超时 124（绿）；回退 %d →
# now-start 恒 0 → 永不超时 → 挂死 → TERM → 143 ≠ 124 → 红。
function phase_clock_saturate()
    println("[phase=clock-saturate] begin — scope: fake awk simulates mawk %d 32-bit " *
            "saturation (constant 2147483647) for %d programs, forwards %.0f — production " *
            "must fire at hard (2s) → rc=124 + TIMEOUT-EXCEEDED, no CANCELLED; " *
            "owned group cleaned; no real /proc touched by the saturation branch")
    @testset "[phase=clock-saturate] %.0f survives large-uptime saturation" begin
        satbin = mktempdir()
        try
            real_awk2 = real_awk_path()
            @test !isempty(real_awk2)
            satawk = joinpath(satbin, "awk")
            write(satawk, """
#!/usr/bin/env bash
# fake awk (test-only): programs using %d simulate 32-bit saturation
# (mawk-style constant 2147483647 — a positive integer that passes naive
# validation and freezes now-start at 0); anything else (e.g. %.0f)
# forwards to REAL_AWK. No host /proc modification: the saturation branch
# never reads input.
prog="\${1:-}"
if [[ "\$prog" == *'%d'* ]]; then
    echo 2147483647
    exit 0
fi
exec "\$REAL_AWK" "\$@"
""")
            chmod(satawk, 0o755)
            slog5 = joinpath(satbin, "c5.log")
            spid5 = joinpath(satbin, "c5.pgid")
            proc = nothing
            pgid = 0
            try
                inner = "echo \$\$ > $spid5; exec sleep 30"
                cmd = Cmd(["bash", script, "12", slog5, "bash", "-c", inner])
                env = Dict("PATH" => satbin * ":" * ENV["PATH"],
                           "REAL_AWK" => real_awk2)
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(spid5)
                @test pgid != 0
                t1 = time()
                while process_running(proc) && time() - t1 < 14
                    sleep(0.1)
                end
                if process_running(proc)
                    kill(proc, 15)   # 饱和挂死时给取消机会（正常路径不会走到这）
                end
                t2 = time()
                while process_running(proc) && time() - t2 < 12
                    sleep(0.1)
                end
                if process_running(proc)
                    kill(proc, 9)
                end
                wait(proc)
                @test proc.exitcode == 124                    # %.0f 正常计时 → 真实 timeout
                log = isfile(slog5) ? read(slog5, String) : ""
                @test occursin("TIMEOUT-EXCEEDED", log)
                @test !occursin("CANCELLED", log)             # 饱和挂死才会走到取消路径 → 红
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(satbin; force=true, recursive=true)
        end
    end
    println("[phase=clock-saturate] end")
end

# ---- 时钟边界副本相（clock-start / clock-cleanup / clock-elapsed /
# clock-transient / clock-negctl / command-admission）------------------------
# 晚期坏钟（monitor 钟好、收尾阶段才坏）无法用"前 N 次好、之后永久坏"
# 的 PATH 计数 awk 确定性构造：monitor 循环每 0.05-0.1s 读一次钟，永久
# 坏型注入必然先在 monitor 处触发，时序竞速不可消除。本组相改用纯测试
# 副本的时钟边界替换：每次 setup 从生产 bin/scoped_run.sh 原文读取生成
# 副本，仅把五个 mono_ms 调用边界（start / monitor / cleanup-window /
# 窗口轮询 / elapsed）替换为独立可控桩 clock_stub，其余逻辑——调用
# 顺序、rc 判定、清理生命周期——与生产同字节；任一替换锚点未命中
# （bin 结构变化）即生成失败、测试红。副本不是生产字节：这是声明过
# 的覆盖边界；生产侧判别力由本组负控相（clock-negctl 移除 fail-closed
# 判定后必须透传 child rc）与 PATH 注入相（clock-fail / clock-mutant /
# clock-saturate）共同承担。失败注入按边界名精确触发——不依赖 sleep
# 竞速或"恰好第 k 次循环"。
#
# 桩协议（环境变量，仅存在于该相副本进程；PATH 必须显式传递——桩
# fallback 转发真 mono_ms 需要 awk）：
#   FAKE_FAIL_START=1        start 边界失败
#   FAKE_FAIL_CLEANUP=1      begin_cleanup_window 边界失败（晚期坏钟）
#   FAKE_FAIL_ELAPSED=1      最终 elapsed 边界失败
#   FAKE_FAIL_WINDOWPOLL=1   cleanup_window_exceeded 窗口轮询边界失败
#   FAKE_FAIL_MONITOR_ONCE=1 monitor 边界失败一次后恢复（瞬态；计数经
#                           FAKE_STUB_STATE 文件传递，不依赖竞速）
#   strip_failclosed=true    生成时移除 run_scoped 尾部坏钟 125 判定
#                           （变异负控：重放旧缺陷形态——晚期坏钟透传）

function make_clock_stub_copy(dst::AbstractString; strip_failclosed::Bool=false)
    src = read(script, String)
    out = src
    reps = [
        ("if ! SCOPE_StartMs=\$(mono_ms); then" =>
         "if ! SCOPE_StartMs=\$(clock_stub start); then"),
        ("if ! now_ms=\$(mono_ms); then" =>
         "if ! now_ms=\$(clock_stub monitor); then"),
        ("    local now win abs\n    if ! now=\$(mono_ms); then" =>
         "    local now win abs\n    if ! now=\$(clock_stub cleanup-window); then"),
        ("    if ! now=\$(mono_ms); then\n        SCOPE_ClockFailed=1\n        SCOPE_CleanupDeadlineMs=0   # 坏钟：转无钟回退\n        [[ -n \$SCOPE_Log ]] && echo \"MONO-CLOCK-FAILED pgid=\$SCOPE_Pgid phase=window-poll\" >> \"\$SCOPE_Log\"" =>
         "    if ! now=\$(clock_stub window-poll); then\n        SCOPE_ClockFailed=1\n        SCOPE_CleanupDeadlineMs=0   # 坏钟：转无钟回退\n        [[ -n \$SCOPE_Log ]] && echo \"MONO-CLOCK-FAILED pgid=\$SCOPE_Pgid phase=window-poll\" >> \"\$SCOPE_Log\""),
        ("mono_elapsed_ms() {\n    local now\n    if ! now=\$(mono_ms); then" =>
         "mono_elapsed_ms() {\n    local now\n    if ! now=\$(clock_stub elapsed); then"),
    ]
    for (a, _) in reps
        occursin(a, out) || return false   # 锚点未命中：bin 结构已变，红
    end
    for (a, b) in reps
        out = replace(out, a => b)
    end
    occursin("trap scope_exit EXIT", out) || return false
    stub = raw"""
# ---- test-only clock stubs（注入边界；非生产字节）--------------------------
# clock_stub <phase>: 按环境变量决定该边界失败与否；否则转发真 mono_ms。
# 桩经命令替换被调用：失败以 return 1 传播，与生产的 if ! 检测同构。
clock_stub() {
    local phase=$1
    case $phase in
        start)
            [[ -n ${FAKE_FAIL_START:-} ]] && return 1
            ;;
        monitor)
            if [[ -n ${FAKE_FAIL_MONITOR_ONCE:-} ]]; then
                local n
                n=$(cat "$FAKE_STUB_STATE" 2>/dev/null || printf '0')
                printf '%s' "$(( n + 1 ))" > "$FAKE_STUB_STATE"
                (( n == 0 )) && return 1
            fi
            ;;
        cleanup-window)
            [[ -n ${FAKE_FAIL_CLEANUP:-} ]] && return 1
            ;;
        elapsed)
            [[ -n ${FAKE_FAIL_ELAPSED:-} ]] && return 1
            ;;
        window-poll)
            [[ -n ${FAKE_FAIL_WINDOWPOLL:-} ]] && return 1
            ;;
    esac
    mono_ms
}

trap scope_exit EXIT"""
    out = replace(out, "trap scope_exit EXIT" => stub)
    if strip_failclosed
        block = "    if (( SCOPE_ClockFailed )); then\n" *
                "        # 坏钟贯穿（契约第 7 条）：本 scope 内任何阶段——start / monitor\n" *
                "        # / cleanup-window / 窗口轮询 / 最终 elapsed——的 mono_ms 失败都\n" *
                "        # 使本次调用最终 fail-closed 125（sticky，瞬态恢复不洗绿）；未知\n" *
                "        # elapsed 报告 unknown，绝不以 0 伪充。时钟契约违约优先于\n" *
        "        # RSS/timeout 的 124。\n" *
                "        emit_result_line 125 \"\$elapsed_ms\"\n" *
                "        return 125\n" *
                "    fi\n"
        occursin(block, out) || return false
        out = replace(out, block => "")
    end
    write(dst, out)
    chmod(dst, 0o755)
    return true
end

function phase_clock_start()
    println("[phase=clock-start] begin — scope: stub-copy start-boundary clock failure → " *
            "fail-closed 125 with bounded leader confirmation and full group reap " *
            "(old defect: KILL delivered then Cleaned=1 with no confirmation); " *
            "pidfile PGID real and observable")
    @testset "[phase=clock-start] start clock failure → 125 + confirmed cleanup" begin
        tmp = mktempdir()
        try
            copy_sh = joinpath(tmp, "scoped_run_stub.sh")
            @test make_clock_stub_copy(copy_sh)
            clog = joinpath(tmp, "cs.log")
            cpidf = joinpath(tmp, "cs.pgid")
            proc = nothing
            pgid = 0
            try
                # child 抗 TERM（5s 宽限 >> echo 延迟）：pidfile 在 wrapper
                # 坏钟清理开始前必然落盘——无竞速；KILL 兜底后 owner 必须
                # 有界确认 leader 死亡并收割组（旧实现投递 KILL 即返回）。
                inner = "echo \$\$ > $cpidf; trap '' TERM; sleep 300"
                cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                env = Dict("PATH" => ENV["PATH"], "FAKE_FAIL_START" => "1")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(cpidf)
                @test pgid != 0
                wait_bounded(proc; window=14.0)
                @test proc.exitcode == 125
                log = isfile(clog) ? read(clog, String) : ""
                @test occursin("MONO-CLOCK-FAILED", log)
                @test occursin("phase=start", log)
                @test occursin("elapsed=unknown", log)   # StartMs 无效：不伪充 0
                @test !occursin("LEADER-STILL-ALIVE", log)    # 0-epoch 错位会提前返回 → 红
                @test !occursin("GROUP-ALIVE-UNREAPED", log)  # 清理确认必须真实完成
                @test no_live_in_group(pgid)              # 有界确认后组净
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(tmp; force=true, recursive=true)
        end
    end
    println("[phase=clock-start] end")
end

function phase_clock_cleanup()
    println("[phase=clock-cleanup] begin — scope: stub-copy cleanup-window boundary failure " *
            "(monitor healthy) → late-stage failure must carry through to 125 for BOTH " *
            "child rc sources — 0 (the old false-green proper: a successful command " *
            "washed to rc=0 + elapsed=0) and 7 (non-zero passthrough equally swallowed)")
    @testset "[phase=clock-cleanup] late cleanup-window clock failure → 125" begin
        for child_rc in (0, 7)
            tmp = mktempdir()
            try
                copy_sh = joinpath(tmp, "scoped_run_stub.sh")
                @test make_clock_stub_copy(copy_sh)
                clog = joinpath(tmp, "cc.log")
                cpidf = joinpath(tmp, "cc.pgid")
                proc = nothing
                pgid = 0
                try
                    # child 立即退出：monitor 桩恒好——失败只在
                    # begin_cleanup_window 边界精确触发，无竞速；旧实现
                    # 此处透传 child rc（rc=0 即假绿本尊）。
                    inner = "echo \$\$ > $cpidf; exit $child_rc"
                    cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                    env = Dict("PATH" => ENV["PATH"], "FAKE_FAIL_CLEANUP" => "1")
                    proc = spawn_wrapper(setenv(cmd, env))
                    pgid = wait_pidfile(cpidf)
                    @test pgid != 0
                    wait_bounded(proc; window=14.0)
                    @test proc.exitcode == 125
                    log = isfile(clog) ? read(clog, String) : ""
                    @test occursin("MONO-CLOCK-FAILED", log)
                    @test occursin("phase=cleanup-window", log)
                    @test occursin(r"elapsed=\d+s", log)     # elapsed 桩好：真实测量
                    @test no_live_in_group(pgid)
                finally
                    if pgid != 0
                        kill_group(pgid)
                    end
                    if proc !== nothing && process_running(proc)
                        kill(proc, 9)
                        wait(proc)
                    end
                end
            finally
                rm(tmp; force=true, recursive=true)
            end
        end
    end
    println("[phase=clock-cleanup] end")
end

function phase_clock_elapsed()
    println("[phase=clock-elapsed] begin — scope: stub-copy final-elapsed boundary failure " *
            "→ 125 + elapsed=unknown for BOTH child rc sources — 0 (old false-green " *
            "proper: success washed to rc=0 + elapsed=0) and 7 (non-zero swallowed)")
    @testset "[phase=clock-elapsed] final elapsed clock failure → 125 + unknown" begin
        for child_rc in (0, 7)
            tmp = mktempdir()
            try
                copy_sh = joinpath(tmp, "scoped_run_stub.sh")
                @test make_clock_stub_copy(copy_sh)
                clog = joinpath(tmp, "ce.log")
                cpidf = joinpath(tmp, "ce.pgid")
                proc = nothing
                pgid = 0
                try
                    inner = "echo \$\$ > $cpidf; exit $child_rc"
                    cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                    env = Dict("PATH" => ENV["PATH"], "FAKE_FAIL_ELAPSED" => "1")
                    proc = spawn_wrapper(setenv(cmd, env))
                    pgid = wait_pidfile(cpidf)
                    @test pgid != 0
                    wait_bounded(proc; window=14.0)
                    @test proc.exitcode == 125
                    log = isfile(clog) ? read(clog, String) : ""
                    @test occursin("MONO-CLOCK-FAILED", log)
                    @test occursin("phase=elapsed", log)
                    @test occursin("elapsed=unknown", log)  # 旧实现：elapsed=0s → 红
                    @test no_live_in_group(pgid)
                finally
                    if pgid != 0
                        kill_group(pgid)
                    end
                    if proc !== nothing && process_running(proc)
                        kill(proc, 9)
                        wait(proc)
                    end
                end
            finally
                rm(tmp; force=true, recursive=true)
            end
        end
    end
    println("[phase=clock-elapsed] end")
end

function phase_clock_transient()
    println("[phase=clock-transient] begin — scope: stub-copy transient monitor failure " *
            "(fails exactly once, then recovers) → sticky SCOPE_ClockFailed still " *
            "ends 125; recovered clock yields real measured elapsed (not unknown)")
    @testset "[phase=clock-transient] one-shot monitor clock failure → sticky 125" begin
        tmp = mktempdir()
        try
            copy_sh = joinpath(tmp, "scoped_run_stub.sh")
            @test make_clock_stub_copy(copy_sh)
            clog = joinpath(tmp, "ct.log")
            cpidf = joinpath(tmp, "ct.pgid")
            statef = joinpath(tmp, "ct.state")
            proc = nothing
            pgid = 0
            try
                # child 存活 1s >> 轮询间隔：monitor 第一次读钟必然发生且
                # 恰好失败一次（状态文件计数，无竞速）；此后恢复——
                # sticky 标志必须使最终 rc 仍为 125（恢复不洗绿）。
                inner = "echo \$\$ > $cpidf; sleep 1"
                cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                env = Dict("PATH" => ENV["PATH"],
                           "FAKE_FAIL_MONITOR_ONCE" => "1",
                           "FAKE_STUB_STATE" => statef)
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(cpidf)
                @test pgid != 0
                wait_bounded(proc; window=14.0)
                @test proc.exitcode == 125
                log = isfile(clog) ? read(clog, String) : ""
                @test occursin("MONO-CLOCK-FAILED", log)
                @test occursin("phase=monitor", log)
                @test occursin(r"elapsed=\d+s", log)   # 钟恢复：elapsed 真实测量
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(tmp; force=true, recursive=true)
        end
    end
    println("[phase=clock-transient] end")

end

function phase_clock_windowpoll()
    println("[phase=clock-window-poll] begin — scope: stub-copy cleanup_window_exceeded " *
            "poll-boundary failure with an owned live TERM-resistant child — the grace " *
            "loop MUST actually poll (group not empty, not skipped), window degrades " *
            "to shared no-clock rounds, 125 + phase=window-poll, group confirmed clean")
    @testset "[phase=clock-window-poll] window-poll clock failure during live cleanup → 125" begin
        tmp = mktempdir()
        try
            copy_sh = joinpath(tmp, "scoped_run_stub.sh")
            @test make_clock_stub_copy(copy_sh)
            clog = joinpath(tmp, "cw.log")
            cpidf = joinpath(tmp, "cw.pgid")
            proc = nothing
            pgid = 0
            try
                # ready 同步：child 自记 pidfile（wrapper 已过 start、
                # monitor 期钟健康）。child 抗 TERM 存活 → hard(2s) 超时后
                # TERM 宽限循环对 owned live descendant 必然实际轮询
                # cleanup_window_exceeded（组非空、不跳过）；第一次窗口
                # 轮询坏钟 → sticky 置位 + 无钟共享回退 + phase 标记，
                # KILL 兜底后 await/reap 真实确认组净。
                inner = "echo \$\$ > $cpidf; trap '' TERM; sleep 300"
                cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                env = Dict("PATH" => ENV["PATH"], "FAKE_FAIL_WINDOWPOLL" => "1")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(cpidf)
                @test pgid != 0
                wait_bounded(proc; window=14.0)
                @test proc.exitcode == 125
                log = isfile(clog) ? read(clog, String) : ""
                @test occursin("MONO-CLOCK-FAILED", log)
                @test occursin("phase=window-poll", log)
                @test !occursin("LEADER-STILL-ALIVE", log)
                @test !occursin("GROUP-ALIVE-UNREAPED", log)
                @test occursin(r"elapsed=\d+s", log)   # elapsed 边界好：真实测量
                @test no_live_in_group(pgid)
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(tmp; force=true, recursive=true)
        end
    end
    println("[phase=clock-window-poll] end")
end

function phase_clock_negctl()
    println("[phase=clock-negctl] begin — scope: mutation negative control — stub copy with " *
            "the run_scoped clock fail-closed block stripped (swallowed-clock mutant) — " *
            "the SAME 125 assertion used in clock-cleanup MUST go red on this mutant; " *
            "the expectation is not rewritten to the passthrough value — this proves " *
            "the rejection power of the assertion itself")
    @testset "[phase=clock-negctl] swallowed-clock mutant → same 125 assertion red" begin
        tmp = mktempdir()
        try
            copy_sh = joinpath(tmp, "scoped_run_mutant.sh")
            @test make_clock_stub_copy(copy_sh; strip_failclosed=true)
            clog = joinpath(tmp, "cn.log")
            cpidf = joinpath(tmp, "cn.pgid")
            proc = nothing
            pgid = 0
            try
                inner = "echo \$\$ > $cpidf; exit 7"
                cmd = Cmd(["bash", copy_sh, "12", clog, "bash", "-c", inner])
                env = Dict("PATH" => ENV["PATH"], "FAKE_FAIL_CLEANUP" => "1")
                proc = spawn_wrapper(setenv(cmd, env))
                pgid = wait_pidfile(cpidf)
                @test pgid != 0
                wait_bounded(proc; window=14.0)
                # 判别力元断言：clock-cleanup 相的同款断言（exitcode == 125）
                # 用在这个 swallowed-clock mutant 上必然红——mutant 透传
                # child rc（7 ≠ 125）。同一消费者、同一断言能拒绝被吞掉的
                # 坏钟错误码；此处不把期望"改成 7"，只证明拒绝能力。
                @test proc.exitcode != 125
                log = isfile(clog) ? read(clog, String) : ""
                @test occursin("phase=cleanup-window", log)  # mutant 上坏钟确实发生且被吞
                @test no_live_in_group(pgid)                  # 清理是生产字节：真实组净，非 stub 假净
            finally
                if pgid != 0
                    kill_group(pgid)
                end
                if proc !== nothing && process_running(proc)
                    kill(proc, 9)
                    wait(proc)
                end
            end
        finally
            rm(tmp; force=true, recursive=true)
        end
    end
    println("[phase=clock-negctl] end")
end

function phase_command_admission()
    println("[phase=command-admission] begin — scope: --rss-guard shifted away leaves no " *
            "command → RC2 refused before spawn AND before logfile creation")
    @testset "[phase=command-admission] no command after options → pre-spawn RC2" begin
        tmp = mktempdir()
        proc = nothing
        proc2 = nothing
        try
            clog = joinpath(tmp, "ca.log")
            proc = spawn_wrapper(Cmd(["bash", script, "12", clog, "--rss-guard=64"]))
            wait_bounded(proc; window=5.0)
            @test proc.exitcode == 2
            @test !isfile(clog)   # 拒绝发生在 logfile 建立之前
            # 对照：同参数形态带 command 时正常透传（既有行为不回归）。
            clog2 = joinpath(tmp, "ca2.log")
            proc2 = spawn_wrapper(Cmd(["bash", script, "12", clog2, "--rss-guard=64",
                                      "bash", "-c", "exit 7"]))
            wait_bounded(proc2; window=14.0)
            @test proc2.exitcode == 7
            @test isfile(clog2)
        finally
            for p in (proc, proc2)
                if p !== nothing && process_running(p)
                    kill(p, 9)
                    wait(p)
                end
            end
            rm(tmp; force=true, recursive=true)
        end
    end
    println("[phase=command-admission] end")
end
# ---- 相注册与执行 ----------------------------------------------------------
const PHASE_FUNCS = Dict{String,Function}(
    "selftest"       => phase_selftest,
    "baseline"       => phase_baseline,
    "owner-term"     => () -> phase_owner(15, "TERM"),
    "owner-int"      => () -> phase_owner(2, "INT"),
    "owner-hup"      => () -> phase_owner(1, "HUP"),
    "clock-mutant"   => phase_clock_mutant,
    "clock-mono"     => phase_clock_mono,
    "clock-fail"     => phase_clock_fail,
    "clock-saturate" => phase_clock_saturate,
    "clock-start"    => phase_clock_start,
    "clock-cleanup"  => phase_clock_cleanup,
    "clock-elapsed"  => phase_clock_elapsed,
    "clock-transient" => phase_clock_transient,
    "clock-window-poll" => phase_clock_windowpoll,
    "clock-negctl"   => phase_clock_negctl,
    "command-admission" => phase_command_admission,
)

function run_phase(p::String)
    PHASE_FUNCS[p]()
end

if PHASE == "ALL"
    println("[phase=ALL] running all phases: $(join(PHASES, ", "))")
    for p in PHASES
        run_phase(p)
    end
else
    run_phase(PHASE)
end
