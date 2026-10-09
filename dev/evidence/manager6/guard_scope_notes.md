# 第6任 guard scope 评审注记（2026-10-08）

任务性质：只读源码评审 + 本文档收口。本任未运行任何命令、测试、编译或
git 操作，未差遣 DevOps；manager3/4/5 的旧 log/summary/snapshot 与唯一
manifest 一字未改。本文件是评审事实与适用范围记录，不复制 SPEC、不构成
第二份运行真源——一切已发生的运行事实以各任 log 与 manifest 为准。

## 1. 评审事实（本任亲读源码所得）

### 1.1 prepare_reference 七 keyword 封闭与拒绝时序的精确口径

- 封闭成立：`prepare_reference`（src/prepare.jl:189-195）签名只声明
  ridge_alpha / F_folds / ruler_stats / history_cache / alpha_initial /
  timing / workspace；三个 internal 通道（ruler_override /
  scale_override / statistics_builder）只存在于 `_prepare_v1`
  （src/predict.jl:189-191）。`fit_v1`（predict.jl:328-331）、
  `path_kelly`（src/kelly.jl:173，ruler_stats 通道被守卫）、
  `prepare_incremental` → `_prepare_current!`（src/incremental.jl:899）
  均无 internal 注入面；`KTrader.jl` export 列表不含 `_prepare_v1`。
- **时序口径纠正（本任前次回报的过广句以此收窄）**：只有「未知
  keyword 的 Julia dispatch 拒绝」先于 `_prepare_v1` 的 generation
  bump（predict.jl:195）；合法 keyword 携带异源 cache / ruler_stats 的
  守卫拒绝发生在 generation bump **之后**、首次数学消费**之前**
  ——verify_history_cache_prefix 接线于 predict.jl:206（active-set
  决议之前），verify_ruler_stats_prefix 接线于 predict.jl:241
  （ruler_from_stats 之前）。并非所有错误都在副作用之前。
- failed / partial prepare 已经 bump generation、使同 workspace owner
  的旧 lease stale，是 prepare.jl:33-42 与 predict.jl:192-196 声明并
  由 test/prepared_problem_contract_tests.jl:151-161 断言（F_folds=10^6
  抛错的 prepare 也 bump）的**既有 lease 契约**，不是缺陷。
- Julia 无真隐私：qualified 调用 `KTrader._prepare_v1` 仍可达 internal
  通道——封闭是契约封闭而非物理封闭，与 manager5 声明的边界一致。

### 1.2 守卫面与消费面重合

- verify_history_cache_prefix（src/numerics.jl:199-264）逐项验证
  `_prepare_v1` 实际消费的四类事实：log_prices[1:T,:]（消费于
  predict.jl:216）、returns[1:T-1,:]（:248）、first_return（:207-208）、
  first_price（:219）；alive_returns / inv_sqrt_alive 只查长度
  （`_prepare_v1` 不读其值，声明与实现一致）。logs 逐 cell NaN-aware
  精确比较、returns 与已验 logs 的 diff 精确一致、首观元数据在已验
  prefix 上重算；T 之后行不读不限（no-lookahead）。
- verify_ruler_stats_prefix（numerics.jl:301-357）只验证 row T、active
  列、每 tau 的 acc/cnt——与 ruler_from_stats（geometry.jl:127-134，
  只读 row t、trust payload）的消费点逐点重合；cnt 硬等、acc
  rtol=64eps/atol=0、NaN/Inf 显式拒。
- 接受边界即数学不变面：消费统计恰好一致的「不同历史」被放行，但
  cache/ruler 的输出只由被验统计决定，接受面=与 no-cache reference 数
  学结果不变面（test/history_cache_contract_tests.jl 的
  「math unchanged: cache path ≡ no-cache path」压在此点）。

### 1.3 冻结与 lease（已核部分无漏洞）

- alive_now：构造器 owned copy（prepare.jl:160，BitVector/alias 输入
  也 copy），默认值 L131 构造时刻冻结；solve 的解构（predict.jl:
  359-361）与函数体（365-505）不引用 adj_act，L473 只读 prep.alive_now。
- ResidualOracle 构造全量冻结（residual_oracle.jl:110-112）：每个输入
  copy/Matrix/Vector 包裹，X_rel 缩减为新分配 prefix-sum 表
  （L101-109），不持任何 workspace 或 history-cache 引用。
- workspace slot 疑问**已关闭，不作为已证漏洞记录**（Manager
  2026-10-08 亲读 response.jl）：fit_response_operator 函数体不使用
  workspace；build_X helper 的 buffer 只有 :path_sums / :design；
  solve 内重建调用（predict.jl:417 build_X_rel_stacked(X_rel, s_perp,
  train_ts)）不传 workspace。与 prepare 阶段 fold_sufficient_statistics
  的 :fold_xx / :fold_xy / :fold_yy / :full_* slot（predict.jl:143-160）
  无重叠。此处仅留关闭事实。

### 1.4 测试区分（评审结论摘要）

- 真实行为断言（能区分坏行为）：history_cache_contract_tests.jl
  L104-152 异源 panel / panel 突变 / cache 载荷与元数据突变反例；
  L349-350 path_kelly_v1 公开 API 级红；L449 被拒调用后
  `ws.generation[] == gen0`；prepared_problem_contract_tests.jl
  L151-161 stale 拒绝与 failed-prepare 也 bump；prepared_lifecycle_
  tests.jl L34 / L63-66 owned mask 不随源移动。
- 静态反事实（如实标注、不冒充运行证据）：L443-444「旧宽接口下本
  调用会红」未实际运行旧源码（manager5 已自我纠正过同类措辞）；当前
  测试断言本身（新接口拒绝 + 零副作用）是真实的。
- 本任未运行任何测试；上述断言的运行结果以 manager3/4/5 log 为准。

## 2. SixResourceGuardAudit 源码交付与时间线（2026-10-08）

- 交付事实（本任文档 pass 亲读 bin/scoped_run.sh 全文 596 行核对）：
  源码 owner 已交付 scoped_run wrapper 的取消/EXIT/pending-acquire
  trap（owner_cancel L278-305、scope_exit L310-326、pending-acquire
  登记 L125-126 + L377-379；取消 handler 在 spawn 之前安装
  L359-361）；mono_ms 单调时钟（/proc/uptime，CLOCK_BOOTTIME 语义
  含 suspend，printf "%.0f" 全程 double，L98-104）与显式坏钟
  fail-closed 125（start L365-375 / monitor L383-389 /
  cleanup-window L148-151 / elapsed L184-192 四处；无钟回退为固定
  轮次上界，绝不无钟空转）；有界 leader 确认后才 wait
  （await_leader_gone L223-233，确认不了报 LEADER-STILL-ALIVE
  125，绝不进入无界 wait）；终止/确认/收割三个清理阶段共享同一次
  acquire 的唯一绝对截止 SCOPE_CleanupDeadlineMs
  （begin_cleanup_window L144-158：min(发起时刻+TERM_GRACE+
  REAP_BUDGET, StartMs+deadline)，幂等只设一次，绝不叠加独立
  5+5+5 预算）。本注记此前「owner 未交付」为交付时点时间线而非
  竞争 current；真运行完成状态待最终各相（运行最终结果由后续文档
  同步追加）。
- 首版缺陷时间线（经静态核对 source-written 已修；不声称每一个已
  运行）：SIG_IGN 跨 exec 继承（spawn 窗口挡信号会污染子命令、
  TERM 宽限失去真实语义——现 handler 先装 + pending 登记）；无界
  wait 与坏钟隐式算术（现显式 fail-closed 125 + 无钟固定轮次回
  退）；32 位 uptime 饱和（mawk 等 awk 实现的 %d 走 32 位转换、
  uptime 超约 24.8 天饱和为常数、now-start 恒 0 使 deadline 永不
  触发——现 %.0f 全程 double，<2^53 ms 无饱和，年数不复述）；
  独立清理预算叠加（现共享唯一绝对截止）。
- 不可捕捉界限（保留，不写「任何外部终止原子回收/自愈」）：
  SIGKILL、SIGSTOP、宿主断电/内核崩溃无法执行任何 trap——owned
  组会孤儿化，唯一兜底是外层更大 deadline 的硬杀（调用纪律：外层
  严格大于本 deadline、两者 ≤60s）；INT 的 trap 仅当 wrapper 自身
  启动时 INT disposition 为默认才生效（POSIX/bash 不允许对继承为
  SIG_IGN 的信号重设 trap）——wrapper 存活、由监控循环按 deadline
  收尾但不以 130 报告；不以未证明的 Ctrl-C/终端分发行为作 TERM
  证据。墙钟回拨：全脚本唯一时间源为 mono（不用 date / $SECONDS /
  $EPOCHREALTIME 等墙钟源），回拨不放宽 deadline、不伪造 elapsed。
- 具名 rationale（拒绝更廉价替代的理由；链接 scoped_run.sh 契约
  第 2/5/6/7 条，不复制全文、不为各 helper 立 ADR）：
  (a) 拒绝单靠外层 killPID：外层只杀 wrapper PID 时 setsid 组孤儿
      化、无人计时收尾——清理必须结构化在 owner 内（trap + 共享
      截止），外层硬超时只是不可捕捉路径的最后调用纪律；
  (b) 拒绝墙钟计时：date/$SECONDS 回拨会放宽 deadline / 伪造
      elapsed——单调 /proc/uptime 是唯一诚实时间源；
  (c) 拒绝忽略信号挡窗口：SIG_IGN 跨 exec 继承会污染子命令的
      TERM 宽限——handler 先装、pending 登记；
  (d) 拒绝延长 deadline 或把整锅运行当分相验证：60s 硬上界是
      mission 约束（超限启动前拒绝 exit 2），延长会把违约伪装成
      合法；一次整锅全绿不等于各失败类分别有红绿证据——独立
      --phase（test 正在增加、仍未交付、不预填）让每个失败类有
      自己的证据。

## 2.5 运行动态事实（当前仅此 scope；运行最终结果稍后追加）

- DevOps 前期（source 最新）：basic 真入口 0/7 精确透传、>60 RC2
  （60s 硬上界启动前拒绝）、G1 真实 TERM marker / child_rc=0 而
  wrapper RC=124（g1_term_exit0.log；timeout is never success）。
- 初期 TERM 临时 harness 的错误保留于 harness_fault_timeline.txt
  （本任亲读）：attempt1 的 wait-across-subshell 127 为 harness 伪
  影（harness 代码错误）、wrapper 自报 143 非自然观察的 parent
  wait status、pgrep 误配产生假 bystander dead；attempt2 的观测是
  外层 30s 超时与 wrapper 自报 2s rc=143 并存，harness 源码存在
  可静态指认的清理错误（杀 subshell PID 而非孙 sleep、stdout
  pipe 由孙进程持有）——「孤儿孙 sleep 持 stdout 导致外层等满
  30s」的因果链是推断：当时无 FD/native wait trace 直接计量，
  成因按 unknown 记（与 LLVM 超时口径一致：观测保留、不抹失
  败、不以后来干净运行反推）；owned pgid 无 live 成员（独立
  kill-0 验证）；干净 parent RC 与 post-cancel 哨兵已被 §2.6
  九窄相覆盖（本句旧「待测」口径降为时间线）。
- 真实标准入口 arch_admission.log（本任亲读）：五 testset 全 Pass
  （M1-A/B/C 3/3、M1-D 2/2、M1-E 2/2、bad fixtures 15/15、legal
  minimal 3/3），scoped_run 尾行 rc=0 elapsed=7s deadline=45
  killed=0 cancelled=none rss_guard=2048 rss_peak=841MiB；四
  runtime/Gate 对象（bin/scoped_run.sh、test/runtests.jl、
  test/contract_registry.jl、test/architecture_contract_tests.jl）
  前后 hash 固定。准入仅此 scope，不泛化为 whole M0/M1。
## 2.6 最终运行（2026-10-08 05:09 final3；本任文档 pass 亲读
final3_closure_summary.txt、final_runtime_snapshot.txt、
negative_control_corrected_record.txt、negative_control_hashes.txt、
arch_after_final2.log 与 negative_control_clock_mono.log 核对）

- 最终对象（完整 SHA 以实盘为准）：bin/scoped_run.sh
  c7048b3be064d5ef60903198a5e8938dca4a7cf4e70df968c323715462817446
  （cleanup-window fix）；test/scoped_run_tests.jl 最终版
  4b13b81fe3b54373e2acec4b4c093bb35d85659487339b9d7aa45223751adec6
  （comment timeline-anchored；05:05 的 final_runtime_snapshot 记
  的是其前一版 3a754edf let-scope + process_running）。补齐
  （2026-10-08 05:18 final3_runtime_snapshot.txt，本任亲读）：当
  前仓库直接 SHA 实测 bin c7048b3b / test 4b13b81f 与 final3
  as-run 一致，并绑定各 phase 原 log（final2_phase_* 九相）与
  两个真红记录的 full SHA；negroot 现存 badbin 为 date 注入
  态，%d 旧态 hash 在已存 negative_control_hashes.txt 中——此
  前「仅依 negroot 副本」的边界降为补齐前时点记录。当前资源
  状态引用 final3_resource_state.txt（本任亲读）实际内容：
  julia_procs=0、sleep_orphans=0、terminals_opened_this_route=0、
  HEAD=602b897、dirty=45、git_writes=none（本 route 从未
  commit/reset/clean）；runtests 186361d8、contract_registry
  d541110e 不变。
- 最终 std arch：arch_after_final2.log RC0，五 testset 全 Pass，
  scoped_run 尾行 rc=0 elapsed=7s deadline=45 killed=0
  cancelled=none rss_guard=2048 rss_peak=841MiB。
- 九相 53 项全 RC0（final3 摘要）：baseline 6/6、owner-term/
  int/hup 各 6/6、clock-mutant 5/5、clock-mono 5/5、clock-fail
  7/7、clock-saturate 6/6、selftest 6/6；unknown/duplicate 为预
  期 RC1 语义拒绝（phase 选择器 fail-closed），不是测试红。此
  前「parent RC / post-cancel 哨兵待测」的 current 已被九窄相覆
  盖；旧的「phase 未交付 / 待最终结果」措辞降为有时点的
  source-written 时间线。guard_closure_summary.txt 是过渡版记录
  （其 NOT-done 三项——bin 变更后 arch、最终字节九相重跑、真红
  负控——在 final3 版本全部完成），保留为时间线、不改旧 file。
- same-consumer 真注入负控（consumer 4b13b81f）：date→clock-mono
  （L100 注入 date 墙钟）自然 RC1 / killed=0 / 16s，本任终核亲读
  negative_control_date_real.log：2 Pass / 3 Fail（15.1s）——两个
  时钟目标红（exitcode 0 != 124、无 TIMEOUT-EXCEEDED）之外，第
  三个 no_live_in_group(pgid) 断言也红（断言时刻组内仍有 live
  成员）；原 corrected record 的「two target assertions」只指两
  个时钟目标，总红数以本 log 为准、共 3 红。%d→clock-saturate
  自然 RC1 / killed=0 / 三红（negative_control_final3.log）。生
  产 bin 对照绿（negative_control_clock_mono.log 亲读：monotonic
  source green under fake rollback 5/5、2.2s；clock-saturate
  6/6 RC0）——最终字节上的双向判别成立。措辞约束（具名更正）：
  16s 自然 RC1 与 exitcode 0 只证明断言红与无 TIMEOUT——原
  「sleep 30 正常结束」说法未证，exitcode 0 不是自然 wrapper
  成功证明；no_live_in_group 红的成因（组内 live 成员为何存
  在）未经 native 计量——不作 SIGKILL 机制断言，仅静态说明：
  harness 在断言之后清理（非源码或 fixture 盲区），断言时刻的
  组状态是观测事实；fake date hold 300ms 设计、真实注入与来源
  如实记录，DevOps 的内部路径解释不抄为确凿根因。
- 保留的真实失败（不洗为绿）：soft-scope / private process API /
  .status 相关早期红、normalleader cleanup window 修复前红、
  harness 错误（127 伪影 / 143 自报 / 30s 外层超时成因
  unknown）、负控注入错行（sed 命中注释 L47 而非 mono_ms L100）
  与一次负控外层 RC124。negative 最初 date RC0 是未注入的伪绿
  （注入工具错误，非 fixture 盲区）；此前「FAKE_HOLD 掩护」的
  解释已撤回，真注入后对照红已补齐。
- 证明范围与不夸大：runtime 证明范围是一个 required 文件
  （test/scoped_run_tests.jl）的分相运行，不是 21 REQUIRED 全
  跑；ALL 整跑明确不授权（授权边界、非本轮欠账）；模型/全
  suite/多日/GPU/吞吐全部保持暂停。不写任意取消原子自愈、不写
  跨机群字节保证、不写性能达标；stage scope / uncatchable
  signals / group 包含范围与外层独立硬超时的语义不夸大。
- 版本来源限定（具名：final3_as_run_binding_notes.txt，本任亲
  读；执行者亲历的命令顺序记录，非 log 回填）：05:05 的
  phase_run_summary / final_phase_*.log 属 BATCH-1（test
  3a754edf，comment edit 之前）；头注释 edit（仅 test L50-52 注
 释、零逻辑/断言改动，3a754edf→4b13b81f，bin c7048b3b 两批
  未变）之后为 BATCH-2（4b13b81f），实际顺序：hash 捕获
  （final2_objects.txt）→ arch_after_final2（g1）→ final2_
  phase 九相（g2）→ %d 真红（g3）→ date 真红（h1）→ closure
  summary → corrected record → final3_runtime_snapshot。诚实
  边界：per-phase 原 log 不内嵌源 hash（scoped_run log 行无
  hash 字段）——版本绑定靠批边界 hash 捕获文件与执行顺序记
  录，单 log 不能自证字节，绑定按顺序推断、如实声明，不称跨
  机器保证。clock-fail elapsed 0.3s 与 0.2s 分属 f3 / g2 两批
  原log、各取原文——是不同观察，不是同一数字的舍入；过渡
  summary 的数字不再转抄为最终参数。date 真红 2 Pass / 3 Fail
  与「sleep 30 正常结束」机制句撤回（与第三红矛盾、实际机制
  未再调查、原log 为准）已在上文记录；本条不改变既有三红/内
  部机制不推断口径，也不改变新版 snapshot 当前源与 4b13 匹配
  的既有引用。

## 3. 适用范围与不预填

- 前任窄绿仍有效（log 为准，非本任运行）：manager5 7-keyword 入口修
  订三 log——arch_gate.log RC0 7s、history_cache_ingress.log 210/210
  （含 closed-keyword-surface testset 85/85 真实负控 P1/P2）、
  primal_prep_ingress.log 46/46；21 REQUIRED 登记 ≠ 21 numeric 全跑。
- SixResourceGuardAudit 源码已交付并完成最终运行（见 §2 时间
  线、§2.5 动态事实与 §2.6 最终运行：九相 53 项 RC0 + same-
  consumer 真注入负控红 + 生产对照绿、最终 std arch RC0）；证明
  范围是一个 required 文件分相、非 21 全跑，ALL 整跑未授权。

## 4. 引用与边界

- 本文件被 AGENTS.md 第6任段与 dev/m1_typed_replay.md 的第6任引用段
  引用；三处口径一致，数值与运行事实不在本文件重复。
- 不改动 SPEC 数学与路线图；不覆盖任何历史事实；运行真源仍是各任
  log 与唯一 manifest（dev/m1_replay_manifest.toml）。
