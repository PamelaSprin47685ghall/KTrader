# manager7 evidence corrections — 第7任文档校准（2026-10-08）

性质：纯文档任务。本任未修改任何 src/bin/test/manifest 文件，未改动
manager3/4/5/6 旧 evidence、log、snapshot（含 guard_scope_notes.md——
其偏差由本文件具名纠正、原文一字不动），未执行任何命令/git/hash/
编译/测试，未差遣 DevOps。本文件是第7任勘误的唯一记录；
README.md、AGENTS.md、dev/m1_typed_replay.md 的 current 段引用本
文件、不复制全文。

## 1. scoped_run 坏钟/准入边界：撤回 manager6 的过广闭合口径

第6任 current 口径（guard_scope_notes.md §2.1 L89-90「start
L365-375 / monitor L383-389 / cleanup-window L148-151 / elapsed
L184-192 四处 fail-closed 125」；README.md Manager-6 段「explicit
bad-clock fail-closed 125 (no silent arithmetic degradation)」；
AGENTS.md 第6任段「start/monitor/cleanup-window/elapsed 四处
fail-closed」）声明过广。本任对 bin/scoped_run.sh 的静态核对确认：

1. 晚期坏钟只降级/0 占位：begin_cleanup_window 与
   mono_elapsed_ms 在 mono_ms 读取失败时分别 return 0 / echo 0
   后 return 0，不产生 125，也不阻断顶层控制流——收尾阶段坏钟
   的一次运行可以 rc=0 透传（假绿路径）。
2. start 坏钟无组净确认：初始时钟读取失败路径 terminate 后未
   确认 leader/owned live group 回收即 Cleaned=1 返回。
3. 空命令假绿：optional --rss-guard 之后无 command 的调用可被
   接纳（未在 spawn/建立 log 前拒绝）。
4. 测试覆盖缺口：现有 clock-fail 相只测 monitor 循环坏钟（该相
   的 125 成立且被 final2_phase_clock-fail.log 与 final3 九相内
   clock-mutant/mono/fail/saturate 各相 log 支撑）。

结论口径：

- 「四处 125 全部闭合 / 无静默降级已解决」撤回，不再作为可引用
  事实；成立的已测事实是 monitor 坏钟 fail-closed 125。
- 旧九相 53 项 RC0 与最终 std arch RC0 保留其已测范围有效——它们
  从未覆盖上述三条边界，故不被本勘误推翻，只是不再被引用为
  scoped_run 全局 fail-closed 的证明。
- 三条缺陷均为本任静态发现、未动态复现；源码 owner 的有界修复已
  另行托付、进行中，本文件不预填任何运行结果（不预填绿）。
- 静态结论的动态复现与修复后验证留给后续 DevOps 受控执行。

## 1a. 修复方向 rationale（决策记录；源码具体落点待 owner，本文件不造已交付、不造已绿）

链接既有权威（引用、不复制）：bin/scoped_run.sh 头部的 guard 执行
契约段（≤60s scoped 执行护栏与坏钟 fail-closed 契约）与 AGENTS
开发守则的「先证明再实现 / 失败不得认证成功 / 病态先止损」纪律。

约束（不可让步的边界）：

- 每命令 ≤60s / RSS2048 硬约束不变——提高 deadline/RSS 不是
  修复；
- 时钟提供者（/proc/uptime 的 mono_ms）在任意阶段、包括瞬态性
  失败，是合法输入空间的一部分，wrapper 必须在它失败时仍然
  正确；
- 故障不能认证成功：一次坏钟运行不得以 rc=0 或成功测量收尾；
- cleanup 必须可观察：终止/回收要么被有界确认，要么如实报告
  未净，不伪已净。

选定方向（供源码 owner 落地；此处只记录选择，不预填实现）：

- 任意阶段（含瞬态恢复后）的坏钟贯穿为同一次 scope 的非成功
  125——瞬态后恢复不得洗绿已被污染的本次运行；
- 耗时不可得时报告 unknown，绝不用 0 伪充有效测量；
- 清理确认在同一 scope 的有界预算内对 leader/owned live group
  做有界确认；确认不了则明确 125 并如实报告；
- optional --rss-guard 之后的空 command 在 spawn/建立 log 之前
  以 RC2 拒绝。

被拒绝的假修（各具名失败模式）：

- 只打印 MONO 错误却透传 rc=0——晚期坏钟假绿路径，即本任静态
  发现 1；
- 用 0 当有效耗时——把未知测量伪造成成功测量；
- KILL 投递即 Cleaned=1——投递不等于回收，即静态发现 2；
- 靠调用者自觉不传空命令、或以放宽 deadline/RSS「绕开」故障
  ——护栏失效路径原样保留，即假修。

后果与不可捕捉边界（诚实保留，不写「任何外部终止原子自愈」）：
SIGKILL/SIGSTOP/宿主断电/内核崩溃无法 trap，owned 组孤儿化仍只有
调用纪律的外层更大 deadline 硬杀兜底（两者均 ≤60s）。revisit 仅由
真实时钟/有界清理协议的新证据触发（例如证明某降级路径安全的
受控协议），公开重审、不偷偷放宽。

## 2. test/scoped_run_tests.jl full SHA 一字符转写差异

存盘字符串直接对比（本任未执行任何 hash 计算、不称「本任已重算
actual 字节」；差异纯粹是已落盘文本的表面不一致）：

- source-of-record（三处一致）：final2_objects.txt、
  final3_runtime_snapshot.txt（含其 /tmp/negroot 副本行）、
  negative_control_hashes.txt 均记
  4b13b81fe3b54373e2acec4b4c093bb35d85659487239b9d7aa45223751adec6
  （第 45-50 位为 87239）。
- 转写手误（两处）：guard_scope_notes.md:164 与 AGENTS.md 第6任段
  原文均写 …59487339…（87339）——单字符差异。
- 处置：guard_scope_notes.md 属 manager6 旧证据、原文不动，由本
  文件具名纠正；AGENTS.md 的引用已按 source-of-record 校正为
  87239 并括注指向本文件；README.md 与 dev/m1_typed_replay.md
  只用短前缀（4b13b81f / 4b13b81fe3b5… / 4b13…），不含 full
  hash，无转写错误、无需改动。
- 引用纪律：对该文件字节级对齐的后续核对以
  final3_runtime_snapshot.txt / final2_objects.txt 等存盘快照为
  准，不依赖任何 markdown 内硬编码 hash 字符串。前任某 Engineer
  曾把该差异的核对误归属为「委任者亲跑」——该归属已撤回，本任
  不传递该说法。

## 3. 口径不变（全部保留）

21 REQUIRED 登记 ≠ 21 numeric 全跑；full suite / ALL 整跑 / 多日 /
N65 真实 fit / GPU / 吞吐全部暂停（用户暂停，不是本轮可强行执行
的欠账）；每命令 ≤60s / RSS2048 硬约束；用户 dirty 工作树保护
（不 commit/reset/clean）；panel provenance unknown。规范依据链接
SPEC §56 fail-loudly 与 §51 逐字段验收（引用规范、不复制全文）。
本文件不改路线图、不添加性能下界、不预填任何运行结果。
