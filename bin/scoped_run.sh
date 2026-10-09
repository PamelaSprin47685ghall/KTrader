#!/usr/bin/env bash
# scoped_run.sh — 仓库统一的 ≤60s scoped 进程执行护栏。
#
# 契约（每次调用的硬性保证，resource obligation 的完整 discharge）：
#   1. create/release 同 scope：命令经 setsid 成为新进程组；TERM 宽限后
#      KILL 只作用于该组（kill -- -PGID / 组内 PID），绝不触碰其他任务
#      的进程（不用全局 pkill/killall）。
#   2. 总 wall ≤ <deadline>：TERM 宽限（默认 5s）、leader 退出确认与组
#      收尾（合计默认预算 10s）都计入 deadline——终止/确认/收割三个
#      清理阶段共享同一次 acquire 的唯一绝对单调截止
#      SCOPE_CleanupDeadlineMs（timeout/RSS 路径恰为剩余 deadline；
#      取消/异常路径为发起时刻起 TERM_GRACE+REAP_BUDGET 且不超过
#      StartMs+deadline），绝不各自叠加独立预算。调用方的外层工具
#      deadline 必须严格大于本 deadline（如内 35 外 45），两者都 ≤60s，
#      保证 owner 走完清理路径后才被外层切断。
#   3. leader 退出（正常/TERM/错误码）不是终点：owner 继续检查组内成员
#      —— zombie（stat Z/X，不占 CPU、待内核/init reap）与 live 进程
#      区分；live 成员 KILL 并轮询确认退出；direct children 由 wait
#      收割。wait 收割之前先在共享清理截止内有界确认 leader 已退出
#      （KILL 是异步投递，不可中断睡眠的 leader 预算内可能不死）——
#      确认不了就不 wait。预算内清不干净才报 GROUP-ALIVE /
#      STILL-ALIVE / LEADER-STILL-ALIVE（退出码 125，fail-closed）——
#      那是未 discharge 的诚实报告，不是设计目标。
#   4. 日志直接写文件（无管道缓冲），退出前追加 scoped_run 结果行。
#   5. owner 自身取消：wrapper 收到 TERM/INT/HUP 时，对本次唯一 owned
#      组执行与 timeout 同一条清理路径（TERM→宽限→KILL→reap 确认），
#      以 128+signum 退出（HUP=129 / INT=130 / TERM=143），取消永不报
#      绿；组未清净时仍报 125（资源违约优先于取消码）。异常退出由
#      EXIT trap 兜底同一清理（清净时保留原退出码）。终止/清理流程中
#      的迟到与重入信号被并入既有流程，不另起第二次清理、不覆盖已定
#      结果（timeout 路径结果为 124，本就非绿）。取消 handler 在
#      spawn 之前安装——子命令经 fork+exec 获得默认信号 disposition
#      （只有 SIG_IGN 会跨 exec 继承，本脚本从不用 trap '' 挡窗口），
#      TERM 宽限对子命令保持真实语义。spawn 与登记之间到达的取消记
#      为 pending：handler 不退出、也不把信号改成 ignore（wrapper 保持
#      可取消），登记完成后立即按完整取消流程处理。
#   6. 不可捕捉边界（诚实声明，不伪造绝对自愈）：SIGKILL、SIGSTOP、
#      宿主断电/内核崩溃无法执行任何 trap——此时 owned 组会孤儿化，
#      唯一兜底是外层更大的 deadline 硬杀（调用纪律，见第 2 条），
#      本脚本对此不做承诺。TERM/HUP 的保证来自第 5 条的显式 trap；
#      INT 的 trap 仅当本 wrapper 自身启动时 INT disposition 为默认
#      （未被调用环境继承为 SIG_IGN）才生效：POSIX/bash 不允许 shell
#      对继承为忽略的信号重新设 trap——若本脚本从 INT 被忽略的环境
#      启动，INT 到达被丢弃：wrapper 存活、由监控循环按 deadline 收
#      尾（不孤儿化），但不会以 130 报告。本脚本不以未经证明的
#      Ctrl-C/终端分发行为作为 TERM 证据。
#   7. 计时一律使用 mono_ms（/proc/uptime，CLOCK_BOOTTIME 语义、含
#      suspend）——全脚本唯一时间源：墙钟回拨不放宽 deadline、不伪造
#      elapsed；不使用 date / $SECONDS / $EPOCHREALTIME 等墙钟源。
#      输出用 printf "%.0f"（全程 double，<2^53 ms ≈ 28.5 万年无饱和），
#      不用 "%d"：常见 awk 实现（如 mawk）的 %d 走 32 位转换，uptime
#      超约 24.8 天（>2^31 ms）时饱和为常数——正整数校验挡不住它，
#      now-start 恒 0 会使 deadline 永不触发。mono_ms 显式校验输出
#      （读取失败/空/非正整数一律视为坏钟）：坏钟走 fail-closed——
#      同一次 scope 内任何阶段（start / monitor / cleanup-window /
#      窗口轮询 / 最终 elapsed）的 mono_ms 失败都置位 sticky 的
#      SCOPE_ClockFailed，本次调用最终以 125 + MONO-CLOCK-FAILED
#      报告（瞬态失败后恢复也不洗绿）；未知 elapsed 报告为 unknown，
#      绝不以 0 伪充成功测量。已 acquire 的组照样清理——含 start 坏钟
#      路径的有界 leader 确认与组收割（与正常收尾同一生命周期契约，
#      确认不了如实报 125，绝不投递 KILL 即宣告已净）；无钟时清理退
#      化为三阶段共享的有限总轮次上界（不叠加独立预算）；绝不靠算术
#      退化碰巧非绿，绝不让无钟状态下的监控循环永不超时。
#      test/scoped_run_tests.jl [C2]（date-mutant 假墙钟）、[C4]（受控
#      坏 awk）、[C5]（%d 饱和模拟）分别验证此选择不可静默回退。
#
# 用法：scoped_run.sh <deadline_s> <logfile> [--rss-guard=<MiB>] <command> [args...]
#   --rss-guard=<MiB>  可选采样级 RSS kill 护栏：以 ≤100ms 间隔轮询进程组
#                      全体成员 VmRSS 之和（/proc/<pid>/status，真 RSS，
#                      非 VmPeak 虚拟内存），超阈值即 TERM→KILL 全组并
#                      以退出码 124 报告 RSS-EXCEEDED（含观测 peak 与超限
#                      值）。这是**采样级 kill 护栏**，不是 cgroup 严格
#                      瞬间限额——两次采样之间的短时尖峰可能不被观测。
#                      预算语义不变：监控计入 deadline；阈值只能收紧
#                      （≤2048MiB），非法参数启动前拒绝。
# 自测：scoped_run.sh --self-test（success/timeout/leader-exit-first/
#       term-resistant/error-exit/rss-exceeded/rss-ok 场景，总 ≤40s）。
#       owner 取消（TERM/INT/HUP 打 wrapper 自身）、假墙钟、受控时钟
#       失败与 %d 饱和在 test/scoped_run_tests.jl [C1]-[C5] 覆盖——
#       需要独立进程向 wrapper 发信号与 PATH 注入，self-test 函数内
#       无法自伤。
#
# 实测经验（写入纪律）：
#   - Julia 源码改动后首次运行含 ~25-30s 重编译，计时前先
#     `julia --project=. -e 'using KTrader'` 预热缓存。
#   - GROUP-ALIVE 曾在 Julia precompile 子进程瞬时残留时触发：本版本
#     的 leader 退出后 reap 会主动清理这类成员，125 仅在预算内确实
#     清不净时出现。

set -u

TERM_GRACE=5    # TERM 宽限秒数
REAP_BUDGET=5   # 清理（确认+收割）合计预算秒数；三阶段共享，不叠加
RSS_GUARD_MAX_MIB=2048   # 阈值只能收紧：>2048 启动前拒绝
RSS_INTERVAL=0.1         # 采样间隔 ≤100ms

# mono_ms — 单调毫秒时钟（/proc/uptime，CLOCK_BOOTTIME 语义含 suspend）。
# 全脚本唯一时间源（契约第 7 条）。printf 用 "%.0f" 而非 "%d"：常见
# awk 实现（mawk 等）的 %d 走 32 位转换，uptime 超约 24.8 天时饱和为
# 正整数常数（校验挡不住、deadline 永不触发）；%.0f 全程 double 精度。
# 显式 fail-closed：读取失败/空/非正整数一律 rc=1——调用方必须显式
# 处理，绝不落入隐式算术退化（无钟监控循环永不超时是最危险的退化
# 方向）。test [C2]/[C4]/[C5] 锁定此行为不可静默回退。
mono_ms() {
    local v
    v=$(awk '{printf "%.0f", $1*1000}' /proc/uptime 2>/dev/null) || return 1
    [[ -n $v && $v =~ ^[0-9]+$ ]] || return 1
    (( v > 0 )) || return 1
    echo "$v"
}

# ---- 单 scope 全局登记 ----------------------------------------------------
# bash 的 trap handler 在顶层执行，看不到 run_scoped 的局部变量；本脚本
# 一次只运行一个 owned scope（主路径单次调用；self-test 顺序复用），当前
# scope 的所有权事实登记于此：acquire（spawn）与 release（清理/取消/异常
# 退出）经由同一份登记闭合。SCOPE_Cleaned 是幂等标志——每份 ownership
# 只 discharge 一次。
SCOPE_Active=0       # 1 = 已 spawn，所有权成立
SCOPE_Pgid=0
SCOPE_Log=""
SCOPE_StartMs=0
SCOPE_Deadline=0
SCOPE_RssGuard=0
SCOPE_RssPeakKib=0
SCOPE_Killed=0       # 组被本脚本终止（timeout / rss / cancel / clock）
SCOPE_RssExceeded=0
SCOPE_Terminating=0  # 终止流程进行中：迟到/重入信号并入，不另起清理
SCOPE_Cleaned=1      # 1 = 无未清偿所有权（无活动 scope 或已 discharge）
SCOPE_ClockFailed=0  # 1 = 本 scope 内 mono_ms 曾失败（sticky：瞬态恢复
                     #     不洗绿；本次调用最终 fail-closed 125）
SCOPE_Cancel=""
SCOPE_Signum=0
SCOPE_PendingCancel=""    # acquire 之前到达的取消（登记完成后处理）
SCOPE_PendingSignum=0

# ---- 清理窗口（唯一绝对截止）----------------------------------------------
# 终止（TERM 宽限）、确认（await_leader_gone）、收割（reap_group）三个
# 阶段共享同一次 acquire 的唯一清理绝对截止——绝不各自开独立 5s 预算
# （防止 5+5+5 叠加越过 deadline，见契约第 2 条）：
#   timeout/RSS 路径：截止 = min(发起时刻+TERM_GRACE+REAP_BUDGET,
#                                StartMs+deadline) —— 恰为剩余 deadline；
#   取消/异常 EXIT 路径：同一 min 公式（取消时刻起算，且不越总截止）；
#   坏钟：CleanupDeadlineMs=0 → 无钟回退——清理循环退化为固定轮次
#   上界（量级同 TERM_GRACE+REAP_BUDGET），之后 fail-closed 125；
#   未设（-1）视为防御性立即超限（所有清理都应发生在窗口设定之后）；
#   SIGKILL/宿主断电等不可捕捉范围见契约第 6 条。
SCOPE_CleanupDeadlineMs=-1   # -1 = 未设；0 = 坏钟无钟回退
SCOPE_WindowSet=0
# 无钟回退的共享总努力上界：TERM 宽限 / leader 确认 / 组收割三个
# 阶段共用同一轮次预算（契约第 2 条——绝不各 phase 叠加独立预算）。
# 量级 (TERM_GRACE + REAP_BUDGET) * 10 轮 ≈ 10s 级有限总努力。
SCOPE_NoClockRounds=0
SCOPE_NoClockRoundBudget=$(( (TERM_GRACE + REAP_BUDGET) * 10 ))

# begin_cleanup_window — 设定本次 acquire 的唯一清理截止（幂等：只设
# 一次；由 terminate_owner_group 首次调用触发——所有终止路径必经）。
begin_cleanup_window() {
    (( SCOPE_WindowSet )) && return 0
    SCOPE_WindowSet=1
    local now win abs
    if ! now=$(mono_ms); then
        SCOPE_ClockFailed=1         # sticky（本函数被直接调用，父 shell
                                    # 内赋值真实生效，不经命令替换回传）
        SCOPE_CleanupDeadlineMs=0   # 坏钟：无钟回退
        [[ -n $SCOPE_Log ]] && echo "MONO-CLOCK-FAILED pgid=$SCOPE_Pgid phase=cleanup-window" >> "$SCOPE_Log"
        return 0
    fi
    win=$(( (TERM_GRACE + REAP_BUDGET) * 1000 ))
    SCOPE_CleanupDeadlineMs=$(( now + win ))
    if (( SCOPE_StartMs > 0 )); then
        # 有有效起点：不超过总截止（契约第 2 条 timeout/RSS 公式）。
        abs=$(( SCOPE_StartMs + SCOPE_Deadline * 1000 ))
        (( SCOPE_CleanupDeadlineMs > abs )) && SCOPE_CleanupDeadlineMs=$abs
    fi
    # StartMs==0（start 坏钟、无有效起点）：绝不把 0 当真实 epoch 与绝对
    # uptime 比较做钳制——否则 abs = deadline*1000 远小于 now、窗口立即
    # 超限，await/reap 的清理确认被跳过，退化为 KILL 投递即返回。此时
    # 窗口 = 发起时刻 + TERM_GRACE + REAP_BUDGET——与契约第 2 条取消/
    # 异常路径同一有界时窗公式（不新增 knob、不抬 deadline）。
    return 0
}

# cleanup_window_exceeded — 共享清理窗口是否已耗尽（返回 0=已超，调用
# 方 fail-closed 125）。无钟回退（CleanupDeadlineMs==0）用全局
# SCOPE_NoClockRounds 对 SCOPE_NoClockRoundBudget 的共享总上界——
# TERM 宽限 / leader 确认 / 组收割三阶段共用同一预算，调用方每轮
# bump 全局计数。窗口内 mono 失败时置位 SCOPE_ClockFailed 并转入
# 无钟回退，绝不无界等待。本函数被直接调用（非命令替换），函数体内
# 的全局赋值在父 shell 真实生效——坏钟事实不依赖子 shell 回传。
cleanup_window_exceeded() {
    local now
    if (( SCOPE_CleanupDeadlineMs < 0 )); then
        return 0
    fi
    if (( SCOPE_CleanupDeadlineMs == 0 )); then
        (( SCOPE_NoClockRounds >= SCOPE_NoClockRoundBudget )) && return 0
        return 1
    fi
    if ! now=$(mono_ms); then
        SCOPE_ClockFailed=1
        SCOPE_CleanupDeadlineMs=0   # 坏钟：转无钟回退
        [[ -n $SCOPE_Log ]] && echo "MONO-CLOCK-FAILED pgid=$SCOPE_Pgid phase=window-poll" >> "$SCOPE_Log"
        (( SCOPE_NoClockRounds >= SCOPE_NoClockRoundBudget )) && return 0
        return 1
    fi
    (( now >= SCOPE_CleanupDeadlineMs )) && return 0
    return 1
}

# mono_elapsed_ms — 自 SCOPE_StartMs 以来的单调毫秒。坏钟时写
# MONO-CLOCK-FAILED 行并输出 "unknown"、return 1——未知 elapsed
# 绝不以 0 伪充成功测量。本函数经命令替换调用（子 shell）：函数内
# 全局赋值无法传回父 scope，坏钟事实由调用方检测返回码置位
# SCOPE_ClockFailed——退出码经命令替换真实传播，不靠隐式状态。
mono_elapsed_ms() {
    local now
    if ! now=$(mono_ms); then
        [[ -n $SCOPE_Log ]] && echo "MONO-CLOCK-FAILED pgid=$SCOPE_Pgid phase=elapsed" >> "$SCOPE_Log"
        echo "unknown"
        return 1
    fi
    echo $(( now - SCOPE_StartMs ))
    return 0
}

# 组内 live 成员清理：区分 zombie/dead 与 live；live KILL 并轮询确认。
# 返回 0 = 组内无 live 成员（zombie 由内核/init 收，不占 CPU 不阻塞）。
# 预算 = 共享清理窗口（契约第 2/3 条）：有钟时窗口截止、无钟时全局
# 共享轮次上界，任一耗尽即 return 1（fail-closed），绝不无界 KILL 循环。
reap_group() {
    local pgid=$1
    local pid stat _rest
    while :; do
        local live=()
        while read -r pid stat _rest; do
            [[ -n ${pid:-} ]] || continue
            if [[ $stat != Z* && $stat != X* ]]; then
                live+=("$pid")
            fi
        done < <(ps -o pid=,stat= -g "$pgid" 2>/dev/null)
        (( ${#live[@]} == 0 )) && return 0
        kill -KILL "${live[@]}" 2>/dev/null
        SCOPE_NoClockRounds=$(( SCOPE_NoClockRounds + 1 ))   # 无钟共享预算
        cleanup_window_exceeded && return 1
        sleep 0.1
    done
    return 1
}

# await_leader_gone — 有界确认组 leader 已退出（进程消失或 zombie/dead）。
# KILL 是异步投递；不可中断睡眠（D 状态）的 leader 预算内可能不死。
# 返回 0 = 已退出（随后的 wait 只收割 zombie，立即返回）；
# 返回 1 = 共享清理窗口耗尽或无钟共享轮次上界——调用方必须
# fail-closed 125（LEADER-STILL-ALIVE / MONO-CLOCK-FAILED），绝不
# 进入无界 wait。
await_leader_gone() {
    local stat
    while :; do
        stat=$(ps -o stat= -p "$SCOPE_Pgid" 2>/dev/null | tr -d ' ')
        [[ -z $stat || $stat == Z* || $stat == X* ]] && return 0
        SCOPE_NoClockRounds=$(( SCOPE_NoClockRounds + 1 ))   # 无钟共享预算
        cleanup_window_exceeded && return 1
        sleep 0.05
    done
    return 1
}

# 进程组全体成员 VmRSS 之和（KiB）。读 /proc/<pid>/status（真 RSS，
# 非 VmPeak）。组内成员由 ps -g 枚举；瞬时消失的成员自然不计。
group_rss_kib() {
    local pgid=$1 total=0 pid rss
    for pid in $(ps -o pid= -g "$pgid" 2>/dev/null); do
        rss=$(grep -m1 '^VmRSS:' "/proc/$pid/status" 2>/dev/null | awk '{print $2}')
        [[ -n ${rss:-} ]] && total=$((total + rss))
    done
    echo "$total"
}

# terminate_owner_group — 对唯一 owned 组执行 TERM→宽限→KILL，并设定
# 共享清理窗口。timeout / RSS 超限 / owner 取消 / 异常 EXIT / 坏钟
# 五条路径共用同一终止数学（同一组、同一宽限、同一 KILL、同一窗口）。
# 幂等：对已死组无害；宽限在共享窗口耗尽时提前让位给 KILL/确认/收割。
terminate_owner_group() {
    (( SCOPE_Active )) || return 0
    begin_cleanup_window
    kill -TERM -- "-$SCOPE_Pgid" 2>/dev/null
    local _i
    for _i in $(seq 1 $((TERM_GRACE * 10))); do
        kill -0 "$SCOPE_Pgid" 2>/dev/null || break
        SCOPE_NoClockRounds=$(( SCOPE_NoClockRounds + 1 ))   # 无钟共享预算
        cleanup_window_exceeded && break   # 窗口/共享轮次耗尽：让出宽限
        sleep 0.1
    done
    kill -KILL -- "-$SCOPE_Pgid" 2>/dev/null
}

# emit_result_line — 统一结果尾行（契约第 4 条）。$1=rc；$2=elapsed_ms
# （毫秒数字）或 "unknown"（mono 坏钟——未知 elapsed 绝不以 0 伪充）。
emit_result_line() {
    local el
    if [[ $2 == unknown ]]; then el="unknown"; else el=$(( $2 / 1000 ))s; fi
    echo "scoped_run: rc=$1 elapsed=$el deadline=$SCOPE_Deadline killed=$SCOPE_Killed cancelled=${SCOPE_Cancel:-none} rss_guard=$SCOPE_RssGuard rss_peak=$(( SCOPE_RssPeakKib / 1024 ))MiB" >> "$SCOPE_Log"
}

# owner_cancel — wrapper 自身收到 TERM/INT/HUP（契约第 5 条）。
# $1=信号名 $2=signum。三个阶段：
#   ① 终止流程中（SCOPE_Terminating）：迟到取消并入（结果已非绿）；
#     trap 保持 handler，后续信号继续并入——不把信号改成 ignore；
#   ② acquire 之前（SCOPE_Active=0）：登记 pending 取消后返回——
#     wrapper 不退出、不把信号改成 ignore（保持可取消），
#     run_scoped 在登记完成后立即处理 pending；
#   ③ 已 acquire：完整取消——TERM→宽限→KILL→reap 确认，
#     exit 128+signum（取消永不报绿）；未清净 exit 125。
# 只清理本次唯一 owned 组，绝不触碰组外进程。
owner_cancel() {
    local name=$1 signum=$2
    if (( SCOPE_Terminating )); then
        return 0
    fi
    if (( ! SCOPE_Active )); then
        SCOPE_PendingCancel="$name"
        SCOPE_PendingSignum=$signum
        return 0
    fi
    trap '' TERM INT HUP   # 清理流程中的重入信号并入（wrapper 即将退出）
    SCOPE_Terminating=1
    SCOPE_Cancel="$name"; SCOPE_Signum=$signum; SCOPE_Killed=1
    terminate_owner_group
    local reaped=0
    reap_group "$SCOPE_Pgid" || reaped=1
    SCOPE_Cleaned=1
    local elapsed_ms
    if ! elapsed_ms=$(mono_elapsed_ms); then
        SCOPE_ClockFailed=1   # 退出码经命令替换真实传播（子 shell 赋值不可靠）
    fi
    local el_txt
    if [[ $elapsed_ms == unknown ]]; then el_txt="unknown"; else el_txt=$(( elapsed_ms / 1000 ))s; fi
    echo "CANCELLED-BY-SIGNAL sig=$SCOPE_Cancel($SCOPE_Signum) pgid=$SCOPE_Pgid elapsed=$el_txt deadline=$SCOPE_Deadline" >> "$SCOPE_Log"
    if (( reaped )); then
        echo "GROUP-ALIVE-UNREAPED pgid=$SCOPE_Pgid" >> "$SCOPE_Log"
        emit_result_line 125 "$elapsed_ms"
        exit 125   # 未清偿的资源违约优先于取消码；窗口耗尽即止，禁止无界 wait
    fi
    if (( SCOPE_ClockFailed )); then
        # 坏钟贯穿：取消路径同样 fail-closed 125（时钟契约违约优先于
        # 取消码；正常取消——钟好——仍为 128+signum，语义不变）。
        emit_result_line 125 "$elapsed_ms"
        exit 125
    fi
    emit_result_line $(( 128 + SCOPE_Signum )) "$elapsed_ms"
    exit $(( 128 + SCOPE_Signum ))
}

# scope_exit — EXIT trap 兜底（契约第 5 条异常退出路径）。
# 正常收尾已置 SCOPE_Cleaned：no-op 并保留原退出码。SIGKILL/SIGSTOP/
# 断电到不了这里（契约第 6 条，诚实边界）。
scope_exit() {
    trap '' TERM INT HUP
    (( SCOPE_Active )) || return 0
    (( SCOPE_Cleaned )) && return 0
    SCOPE_Terminating=1
    terminate_owner_group
    local reaped=0
    reap_group "$SCOPE_Pgid" || reaped=1
    SCOPE_Cleaned=1
    local el_se
    if ! el_se=$(mono_elapsed_ms); then
        SCOPE_ClockFailed=1
    fi
    echo "ABNORMAL-EXIT-SCOPE-CLEANED pgid=$SCOPE_Pgid reaped_ok=$(( 1 - reaped ))" >> "$SCOPE_Log"
    if (( reaped )); then
        echo "GROUP-ALIVE-UNREAPED pgid=$SCOPE_Pgid" >> "$SCOPE_Log"
        emit_result_line 125 "$el_se"
        exit 125
    fi
    if (( SCOPE_ClockFailed )); then
        emit_result_line 125 "$el_se"
        exit 125   # 坏钟贯穿：异常退出兜底同样 fail-closed
    fi
    # 清净：不显式 exit——保留脚本原有退出码
}

run_scoped() {
    local deadline=$1 logfile=$2; shift 2
    local rss_guard=0   # 0 = 未启用
    if [[ "${1:-}" == --rss-guard=* ]]; then
        rss_guard="${1#--rss-guard=}"
        shift
        if ! [[ $rss_guard =~ ^[0-9]+$ ]] || (( rss_guard < 1 )); then
            echo "--rss-guard must be a positive integer (MiB), got '$rss_guard'" >> "$logfile"
            return 2
        fi
        if (( rss_guard > RSS_GUARD_MAX_MIB )); then
            echo "--rss-guard $rss_guard exceeds the tighten-only cap ${RSS_GUARD_MAX_MIB}MiB" >> "$logfile"
            return 2
        fi
    fi
    if (( $# < 1 )); then
        # shift 掉 optional --rss-guard 后无 command：必须在 spawn 与
        # logfile 建立之前 RC2 拒绝（与顶层参数校验同一 fail-fast 语义；
        # 错误走 stderr，不截断/创建 logfile、不留半截 scope 状态）。
        echo "scoped_run: no command given after options" >&2
        return 2
    fi
    local hard_ms=$(( (deadline - TERM_GRACE - REAP_BUDGET) * 1000 ))
    : > "$logfile"

    # 重置单 scope 登记（self-test 在同进程顺序复用）。
    SCOPE_Active=0; SCOPE_Cleaned=1; SCOPE_Terminating=0
    SCOPE_Pgid=0; SCOPE_Log="$logfile"; SCOPE_StartMs=0
    SCOPE_Deadline=$deadline; SCOPE_RssGuard=$rss_guard
    SCOPE_RssPeakKib=0; SCOPE_Killed=0; SCOPE_RssExceeded=0
    SCOPE_Cancel=""; SCOPE_Signum=0
    SCOPE_PendingCancel=""; SCOPE_PendingSignum=0
    SCOPE_CleanupDeadlineMs=-1; SCOPE_WindowSet=0
    SCOPE_ClockFailed=0
    SCOPE_NoClockRounds=0

    # 取消 handler 在 spawn 之前安装（契约第 5 条）：子命令经 fork+exec
    # 获得默认信号 disposition——只有 SIG_IGN 会跨 exec 继承，本脚本
    # 不用 trap '' 挡窗口，TERM 宽限对子命令保持真实语义。spawn 与
    # 登记之间到达的取消由 handler 记为 pending（见 owner_cancel ②）。
    trap 'owner_cancel HUP 1' HUP
    trap 'owner_cancel INT 2' INT
    trap 'owner_cancel TERM 15' TERM
    setsid bash -c 'exec "$@"' _ "$@" >> "$logfile" 2>&1 &
    SCOPE_Pgid=$!
    SCOPE_Active=1; SCOPE_Cleaned=0
    if ! SCOPE_StartMs=$(mono_ms); then
        # 坏钟发生在 acquire 之后：组已存在，必须清理（fail-closed 125），
        # 绝不让监控循环在无钟状态下永不超时（契约第 7 条）。清理走与
        # 正常收尾同一生命周期契约：terminate（含共享清理窗口设定）后
        # 有界确认 leader 退出、再收割组——确认不了报 LEADER-STILL-ALIVE
        # 125、清不净报 GROUP-ALIVE-UNREAPED 125，绝不投递 KILL 即宣告
        # 已净；无钟时三阶段共享有限总轮次上界。StartMs 无效，elapsed
        # 如实报 unknown（即使钟瞬态恢复也不伪充测量）。
        SCOPE_StartMs=0
        SCOPE_ClockFailed=1; SCOPE_Terminating=1; SCOPE_Killed=1
        echo "MONO-CLOCK-FAILED pgid=$SCOPE_Pgid phase=start" >> "$logfile"
        terminate_owner_group
        local leader_gone=0
        await_leader_gone || leader_gone=1
        if (( leader_gone )); then
            echo "LEADER-STILL-ALIVE pgid=$SCOPE_Pgid" >> "$logfile"
            SCOPE_Cleaned=1
            emit_result_line 125 "unknown"
            return 125
        fi
        local reaped=0
        reap_group "$SCOPE_Pgid" || reaped=1
        SCOPE_Cleaned=1
        if (( reaped )); then
            echo "GROUP-ALIVE-UNREAPED pgid=$SCOPE_Pgid" >> "$logfile"
            emit_result_line 125 "unknown"
            return 125
        fi
        emit_result_line 125 "unknown"
        return 125
    fi
    # spawn 窗口内到达的取消已记为 pending：登记完成，立即按完整取消处理。
    if [[ -n $SCOPE_PendingCancel ]]; then
        owner_cancel "$SCOPE_PendingCancel" "$SCOPE_PendingSignum"
    fi

    local start_ms=$SCOPE_StartMs rss_peak=0 rss_now=0 now_ms=0
    while kill -0 "$SCOPE_Pgid" 2>/dev/null; do
        if ! now_ms=$(mono_ms); then
            # 坏钟（契约第 7 条）：终止组并 fail-closed，绝不无钟空转。
            # SCOPE_ClockFailed 是 scope 级 sticky 标志：瞬态失败后钟恢复
            # 也不洗绿（最终仍 125）。
            SCOPE_ClockFailed=1; SCOPE_Killed=1; SCOPE_Terminating=1
            terminate_owner_group
            echo "MONO-CLOCK-FAILED pgid=$SCOPE_Pgid phase=monitor" >> "$logfile"
            break
        fi
        if (( rss_guard > 0 )); then
            rss_now=$(group_rss_kib "$SCOPE_Pgid")
            (( rss_now > rss_peak )) && { rss_peak=$rss_now; SCOPE_RssPeakKib=$rss_peak; }
            if (( rss_now > rss_guard * 1024 )); then
                SCOPE_RssExceeded=1; SCOPE_Killed=1; SCOPE_Terminating=1
                terminate_owner_group
                break
            fi
        fi
        if (( now_ms - start_ms >= hard_ms )); then
            SCOPE_Killed=1; SCOPE_Terminating=1
            terminate_owner_group
            break
        fi
        if (( rss_guard > 0 )); then sleep 0.05; else sleep 0.1; fi
    done
    # 信号 trap 保持到函数结束（不卸载）：收尾期间到达的取消仍由
    # owner_cancel 处理（幂等），保证取消在任何时刻都不报绿。
    #
    # KILL 是异步投递、不可中断睡眠的 leader 预算内可能不死：先在共享
    # 清理窗口内有界确认 leader 已退出（消失/zombie），确认后才 wait
    # 收割；确认不了报 LEADER-STILL-ALIVE 125，绝不进入无界 wait
    # （契约第 3 条）。取消/异常 EXIT 路径不 wait direct child（zombie
    # 由 init 收），同受共享窗口约束——同义。
    local leader_gone=0
    await_leader_gone || leader_gone=1
    if (( leader_gone )); then
        SCOPE_Cleaned=1
        local el_lg
        if ! el_lg=$(mono_elapsed_ms); then SCOPE_ClockFailed=1; fi
        echo "LEADER-STILL-ALIVE pgid=$SCOPE_Pgid" >> "$logfile"
        emit_result_line 125 "$el_lg"
        return 125
    fi
    wait "$SCOPE_Pgid" 2>/dev/null
    local rc=$?
    # leader 退出后的组收尾：live 成员必须在本 scope 内清理（discharge，
    # 共享窗口内）。正常退出路径不经过 terminate_owner_group（leader
    # 自然死），清理窗口从未开启——SCOPE_CleanupDeadlineMs 仍为 -1，
    # cleanup_window_exceeded 会把"未设"视为立即超限，reap_group 首
    # 轮即失败（GROUP-ALIVE-UNREAPED 假红，self-test leader-exit-first
    # 场景实测复现）。此处显式开启（幂等）：以当前时刻起算共享窗口。
    begin_cleanup_window
    local reaped=0
    reap_group "$SCOPE_Pgid" || reaped=1
    SCOPE_Cleaned=1
    if (( reaped )); then
        echo "GROUP-ALIVE-UNREAPED pgid=$SCOPE_Pgid" >> "$logfile"
        return 125
    fi
    if kill -0 "$SCOPE_Pgid" 2>/dev/null; then
        echo "STILL-ALIVE pgid=$SCOPE_Pgid" >> "$logfile"
        return 125
    fi
    local elapsed_ms
    if ! elapsed_ms=$(mono_elapsed_ms); then
        SCOPE_ClockFailed=1   # 退出码经命令替换真实传播（子 shell 赋值不可靠）
    fi
    if (( SCOPE_ClockFailed )); then
        # 坏钟贯穿（契约第 7 条）：本 scope 内任何阶段——start / monitor
        # / cleanup-window / 窗口轮询 / 最终 elapsed——的 mono_ms 失败都
        # 使本次调用最终 fail-closed 125（sticky，瞬态恢复不洗绿）；未知
        # elapsed 报告 unknown，绝不以 0 伪充。时钟契约违约优先于
        # RSS/timeout 的 124。
        emit_result_line 125 "$elapsed_ms"
        return 125
    fi
    if (( SCOPE_RssExceeded )); then
        echo "RSS-EXCEEDED pgid=$SCOPE_Pgid guard=${rss_guard}MiB observed_peak=$((rss_peak / 1024))MiB" >> "$logfile"
        emit_result_line 124 "$elapsed_ms"
        return 124
    fi
    # TIMEOUT is never a success: a child whose TERM handler exits 0 must
    # not make a deadline breach report rc=0 (false green). killed=1 means WE
    # terminated the group for exceeding the deadline — report 124 regardless
    # of the child's own exit status. Normal (non-killed) exit codes still
    # pass through unchanged; RSS-EXCEEDED stays 124, unreaped stays 125.
    if (( SCOPE_Killed )); then
        echo "TIMEOUT-EXCEEDED pgid=$SCOPE_Pgid deadline=${deadline}s child_rc=$rc (timeout is never success)" >> "$logfile"
        emit_result_line 124 "$elapsed_ms"
        return 124
    fi
    emit_result_line "$rc" "$elapsed_ms"
    return "$rc"
}

self_test() {
    # 坏钟自检：mono_ms 不可用则整个 self-test 没有可靠计时，直接 FAIL。
    if ! mono_ms >/dev/null 2>&1; then
        echo "SELF-TEST FAIL: mono_ms unavailable (/proc/uptime or awk broken)"
        return 1
    fi
    local tmp=$(mktemp -d)
    local rc=0
    # 场景 1：success —— 短命令成功、日志落盘、组干净退出。
    run_scoped 15 "$tmp/ok.log" bash -c 'echo hello-from-child; sleep 0.2'
    local rc1=$?
    if [[ $rc1 -ne 0 || ! $(grep -c hello-from-child "$tmp/ok.log") -gt 0 \
          || $(grep -c "ALIVE" "$tmp/ok.log") -ne 0 ]]; then
        echo "SELF-TEST FAIL: success (rc=$rc1)"; cat "$tmp/ok.log"; rc=1
    else
        echo "self-test success: PASS"
    fi
    # 场景 2：timeout —— 孙进程 sleep 300，TERM（子进程为默认 disposition）
    # 立即生效、宽限提前结束，组干净终止。
    local t0=$(mono_ms)
    run_scoped 13 "$tmp/to.log" bash -c 'sleep 300 & sleep 300 & wait'
    local rc2=$? dt=$(( ($(mono_ms) - t0) / 1000 ))
    if [[ $dt -gt 13 || $(grep -c "ALIVE" "$tmp/to.log") -ne 0 ]]; then
        echo "SELF-TEST FAIL: timeout (dt=${dt}s rc=$rc2)"; cat "$tmp/to.log"; rc=1
    else
        echo "self-test timeout: PASS (dt=${dt}s, group reaped)"
    fi
    # 场景 3：leader 先退出、子进程继续 —— owner 必须在 leader 正常
    # 退出后清理组内孤儿（discharge，不留给 caller）。
    local t1=$(mono_ms)
    run_scoped 12 "$tmp/orphan.log" bash -c 'sleep 300 & sleep 300 & exit 0'
    local rc3=$? dt3=$(( ($(mono_ms) - t1) / 1000 ))
    if [[ $(grep -c "ALIVE" "$tmp/orphan.log") -ne 0 ]]; then
        echo "SELF-TEST FAIL: leader-exit-first"; cat "$tmp/orphan.log"; rc=1
    else
        echo "self-test leader-exit-first: PASS (dt=${dt3}s, orphans reaped in-scope)"
    fi
    # 场景 4：TERM 抗拒子进程 —— trap 忽略 TERM，KILL 兜底生效；
    # 同时验证组外无辜进程不被误杀。
    local bystander_pid
    sleep 60 & bystander_pid=$!   # 组外进程（本 shell 的子，不在 PGID）
    local t2=$(mono_ms)
    run_scoped 12 "$tmp/resist.log" bash -c 'trap "" TERM; sleep 300 & trap "" TERM; wait'
    local rc4=$? dt4=$(( ($(mono_ms) - t2) / 1000 ))
    local bystander_alive=0
    kill -0 "$bystander_pid" 2>/dev/null && bystander_alive=1
    kill -KILL "$bystander_pid" 2>/dev/null; wait "$bystander_pid" 2>/dev/null; true
    if [[ $(grep -c "ALIVE" "$tmp/resist.log") -ne 0 || $bystander_alive -ne 1 ]]; then
        echo "SELF-TEST FAIL: term-resistant (dt=${dt4}s bystander_alive=$bystander_alive)"; cat "$tmp/resist.log"; rc=1
    else
        echo "self-test term-resistant: PASS (dt=${dt4}s, bystander untouched)"
    fi
    # 场景 5：错误 exit —— rc 透传，组清理正常。
    run_scoped 25 "$tmp/err.log" bash -c 'exit 7'   # hard=deadline-10>0: exit 7 completes well within, rc passes through (the old deadline=10 made hard=0, misclassifying EVERY run as timeout and passing only by TERM/exit race)
    local rc5=$?
    if [[ $rc5 -ne 7 || $(grep -c "ALIVE" "$tmp/err.log") -ne 0 ]]; then
        echo "SELF-TEST FAIL: error-exit (rc=$rc5)"; rc=1
    else
        echo "self-test error-exit: PASS (rc=7 passed through)"
    fi
    # 场景 6：RSS 超限 kill 护栏（低成本反例：小阈值 + 小 buffer 持有）。
    # python 持续持有 ~40MiB，guard=20MiB 必须在采样点被观测并停组。
    if command -v python3 >/dev/null 2>&1; then
        run_scoped 15 "$tmp/rss.log" --rss-guard=20 python3 -c 'b=bytearray(40*1024*1024); import time; [time.sleep(0.1) for _ in range(100)]'
        local rc6=$?
        if [[ $rc6 -ne 124 ]] || ! grep -q "RSS-EXCEEDED" "$tmp/rss.log" \
           || [[ $(grep -c "ALIVE" "$tmp/rss.log") -ne 0 ]]; then
            echo "SELF-TEST FAIL: rss-exceeded (rc=$rc6)"; cat "$tmp/rss.log" 2>/dev/null; rc=1
        else
            echo "self-test rss-exceeded: PASS (rc=124, group stopped+reaped)"
        fi
    else
        echo "self-test rss-exceeded: SKIP (no python3)"
    fi
    # 场景 7：RSS 正常（guard 宽松）——命令成功退出、peak 被观测记录。
    if command -v python3 >/dev/null 2>&1; then
        run_scoped 15 "$tmp/rssok.log" --rss-guard=512 python3 -c 'b=bytearray(20*1024*1024); print("rss-ok-done")'
        local rc7=$?
        if [[ $rc7 -ne 0 ]] || ! grep -q "rss-ok-done" "$tmp/rssok.log" \
           || [[ $(grep -c "ALIVE" "$tmp/rssok.log") -ne 0 ]]; then
            echo "SELF-TEST FAIL: rss-ok (rc=$rc7)"; cat "$tmp/rssok.log" 2>/dev/null; rc=1
        else
            echo "self-test rss-ok: PASS (rc=0, peak observed: $(grep -o 'rss_peak=[0-9]*MiB' "$tmp/rssok.log"))"
        fi
    else
        echo "self-test rss-ok: SKIP (no python3)"
    fi
    # 场景 8：非法 guard 参数 —— 启动前拒绝（exit 2）。
    run_scoped 12 "$tmp/bad.log" --rss-guard=abc bash -c 'echo never-run'
    local rc8=$?
    if [[ $rc8 -ne 2 ]]; then
        echo "SELF-TEST FAIL: rss-bad-param (rc=$rc8)"; cat "$tmp/bad.log" 2>/dev/null; rc=1
    else
        echo "self-test rss-bad-param: PASS (rejected before start)"
    fi
    # 场景 9：guard 超上限（只许收紧）—— 启动前拒绝。
    run_scoped 12 "$tmp/big.log" --rss-guard=99999 bash -c 'echo never-run'
    local rc9=$?
    if [[ $rc9 -ne 2 ]]; then
        echo "SELF-TEST FAIL: rss-over-cap (rc=$rc9)"; cat "$tmp/big.log" 2>/dev/null; rc=1
    else
        echo "self-test rss-over-cap: PASS (tighten-only cap enforced)"
    fi
    rm -rf "$tmp"
    [[ $rc -eq 0 ]] && echo "scoped_run self-test: ALL PASS"
    return $rc
}

# EXIT trap 兜底挂载：参数校验/自测阶段 SCOPE_Active=0 或 Cleaned=1 →
# no-op；run_scoped 异常中断（如 set -u 触发）时清理 owned 组（第 5 条）。
trap scope_exit EXIT

if [[ "${1:-}" == "--self-test" ]]; then
    self_test
    exit $?
fi

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <deadline_s> <logfile> <command> [args...]  |  $0 --self-test" >&2
    exit 2
fi
# deadline 校验：宽限 + 收尾预算必须装得进 deadline；且硬上界 60s
#（mission 硬限——超限命令在启动前拒绝，不是运行后杀）。
if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "deadline must be a positive integer, got '$1'" >&2
    exit 2
fi
if (( $1 > 60 )); then
    echo "deadline $1 exceeds the 60s hard cap; refusing to start" >&2
    exit 2
fi
if (( $1 <= TERM_GRACE + REAP_BUDGET )); then
    echo "deadline must exceed TERM_GRACE+REAP_BUDGET=$((TERM_GRACE + REAP_BUDGET))s" >&2
    exit 2
fi
run_scoped "$@"
