# 证据引用校准 — manager5（2026-10-08）

本任为纯文档与证据引用校准：未运行任何命令/git/编译/测试，未差遣
DevOps；未修改 manager3/manager4 的任何旧证据（summary、
corrections、日志、snapshot、manifest）、源码与测试。本文件只纠正
已记录陈述的证据权重与归属，不改动任何原始记录；历史红与时间线
原样保留。

本任逐行读毕的原文（非抽样）：
dev/evidence/manager3/standard_entry_4way_summary.txt（11 行）、
std_entry_minimal.log（8 行）、std_entry_nested.log（8 行）、
std_entry_bad.log（29 行）、std_entry_empty.log（60 行）、
dev/evidence/manager4/final_summary.txt（10 行）、
dev/evidence/manager4/final_snapshot.txt（18 行）、
dev/evidence/manager4/evidence_corrections.md（108 行）、
README.md 与 AGENTS.md、dev/m1_typed_replay.md 的相关 current 段。

## 1. standard_entry_4way_summary.txt 的 guard 配置转写偏差

被纠正的记录（原文保留、一字不改，manager3/standard_entry_4way_
summary.txt 第 2 行）：

    cmd: julia --startup-file=no --project=. test/runtests.jl --architecture-only --architecture-root <ROOT> (scoped_run 25s, RSS2048, no production KTrader load)

四份 log 原文的 scoped_run 尾行（本项的唯一事实来源）：

- std_entry_minimal.log:8 — `rc=0 elapsed=1s deadline=12 killed=0 rss_guard=0 rss_peak=0MiB`
- std_entry_nested.log:8 — `rc=0 elapsed=1s deadline=12 killed=0 rss_guard=0 rss_peak=0MiB`
- std_entry_bad.log:29 — `rc=1 elapsed=3s deadline=25 killed=0 rss_guard=0 rss_peak=0MiB`
- std_entry_empty.log:60 — `rc=1 elapsed=3s deadline=25 killed=0 rss_guard=0 rss_peak=0MiB`

偏差（summary 对 log 的错误转写，共三处）：

1. RSS2048 不成立：四 log 均 `rss_guard=0`、`rss_peak=0MiB`——
   guard 未生效/未测量。RSS 合规不能从该 summary 或这四份 log
   推出。
2. "25s" 对 minimal/nested 不成立：它们的 `deadline=12`（bad/empty
   为 25，与 summary 一致）。
3. summary 第 3/4 行的 "3s"（minimal/nested）与 log 内
   `elapsed=1s` 不一致（3s 只与 bad/empty 的 elapsed 一致；口径
   混淆）。

不受影响的事实（证据仍在）：四向红绿 verdict 本身——minimal/nested
RC0、bad/empty RC1、killed=0 自然退出、violations 内容、
fixture-mode 声明——由四份 log 原文直接支撑，且四 log 的 sha256
绑定于 summary 第 8–11 行。guard 转写偏差不推翻红绿行为；推翻的
只是「这四向运行受 RSS2048/25s guard 保护」这一配置层陈述。

引用口径：可称「红绿行为真实、log 原文支撑」；不得称「RSS2048
合规」，亦不得对 minimal/nested 称「25s deadline 下运行」。

## 2. manager4/final_summary.txt 字段归属偏差

被纠正的引用（原文保留、一字不改）：manager4/
evidence_corrections.md 第 67–69 行、README.md（run-scope
calibrations 段）、dev/m1_typed_replay.md（FINAL RUN SYNC 段）、
AGENTS.md 第4任段——四处均把 process 0 / HEAD 602b897 / dirty 45 /
no commit-reset-clean 归给 dev/evidence/manager4/final_summary.txt。

核对事实：manager4/final_summary.txt 全文 10 行，内容为六项运行
记录加红历史与 NOT-done 清单，不含上述任何字段；
manager4/final_snapshot.txt 全文 18 行，为 16 个对象的逐路径
sha256，亦不含上述任何字段。

三类陈述的区分（本任口径纪律）：

- log 原文：上述字段在 manager4 两个具名文件的原文中均未出现——
  这是本任逐行核对后的事实。
- 前任回报：该组字段来自前任的回报文本（被四处 current 文档转写
  为「final_summary.txt 记录了这些」）。前任回报本身未被推翻——
  进程退出码 0、HEAD、脏文件数完全可能真实；但它在本任核对范围
  内无落盘证据支撑。
- 尚未知：该组字段是否存在于其他落盘文件，本任未穷尽全仓库
  搜索，不下肯定或否定结论；引用时必须标注「前任回报、落盘
  未见」。

已落盘的修正：README.md、dev/m1_typed_replay.md 的相应引用已改
为「figures carried in the prior report, not in the file itself」；
AGENTS.md 第5任段同步。manager4/evidence_corrections.md 为旧证据
（前任交付物），一字不改，其偏差由本文件具名纠正。

本任 current 资源/Git 状态的落盘依据（2026-10-08 后续 DevOps
观察，本任文档 pass 亲读）：dev/evidence/manager5/preflight.txt
——julia_procs=0（ps 全列、无本路线残留、无击杀）、scoped_run
契约完整（60s 硬上限、RSS 2048 tighten-only、超时返回 124）、
HEAD 602b897 / dirty 45 / 未 commit-reset-clean，文件自声明
「observed NOW (not reconstructed from manager4 history)」——
仅代表本任时点、不倒推 manager4；manager4 历史口径不变（前任
回报、两具名文件未见）。

## 3. ReferenceIngressRepair — 源码已交付、运行已验证窄绿

状态（最终同步 2026-10-08：源码 owner 静态交付 + DevOps 受控运行
+ 本任文档 pass 亲读 manager5 三份 log 与 ingress_repair_summary.txt
核准，落盘与回报一致、无差异）：

- 源码交付（源码 owner，静态）：prepare_reference 封闭显式
  7-keyword 白名单（ridge_alpha / F_folds / ruler_stats /
  history_cache / alpha_initial / timing / workspace，默认值逐一
  不变），Julia keyword dispatch 在函数体前拒绝一切其它关键字
  ——三个 internal override（ruler_override / scale_override /
  statistics_builder）即使 =nothing 也拒，拒绝先于 workspace
  generation 变化、cache 读取与 builder 执行；_prepare_v1 私有
  路径与 incremental 的 state-owned override 接线未改
  （predict.jl 仅接口注释）；新增 testset 落在现有 REQUIRED 文件
  test/history_cache_contract_tests.jl。
- 运行验证（DevOps 受控、本任亲读 log 核准）：先真实架构门
  arch_gate.log RC0（elapsed 7s、killed=0、rss_guard=2048、
  peak 840MiB、五 Test Summary 全 Pass）；后
  history_cache_ingress.log RC0（elapsed 28s、killed=0、guard
  2048、peak 1039MiB）六 testset 210/210 = 24+9+15+45+32+85，
  其中新增「prepare_reference: closed public keyword surface」
  testset 85/85（含 P1/P2 真实负控）；primal_prep_ingress.log
  RC0（elapsed 16s、killed=0、guard 2048、peak 1029MiB）46/46，
  私有 incremental route 保留。四对象验证前后 SHA256 固定于
  ingress_repair_summary.txt（src/prepare.jl、src/predict.jl、
  两 test 文件）；无自修、无 tol/guard 放宽、existing 单线程
  纯语义配置（非吞吐口径）。
- 措辞校准（本任钉死）：否定注入的实测语义是「新接口下无效
  调用抛错且 workspace/builder 副作用为零」；「旧宽接口下这些
  断言会红」属于静态反事实——未实际运行旧源码或 mutant，不得
  写成旧接口红运行。
- 边界：不称「全部外部必须只走 prepare_reference」——fit_v1/
  solve 正常公开链仍合法，封闭的只是公开 reference 入口的
  internal override 面。

以下为该修复的设计决策记录（本任落盘时点写就；交付与运行事实
见上）：

Context 约束：`prepare_reference` 是 SPEC §46 的 reference
prepare——executable specification，最简单、从完整前缀重算、是
所有 accelerator 的对照基准。本任核准时点的实现（src/prepare.jl:169）
为 `prepare_reference(adj; kwargs...) = _prepare_v1(adj; kwargs...)`：
公开入口把任意 kwargs 裸透传给内部 `_prepare_v1`，公开契约由内部
实现隐式定义——这正是被本修复替换的形态（交付与运行事实见本节
开头）。

选择：显式公有 kwargs 白名单。公开入口逐名列出它接受并转发的
公有参数；白名单之外的 kwarg 一律报错。

拒绝的替代方案及理由：

- kwargs 裸透传（现状）：内部参数的增删会静默改变公开 API 的
  实际行为；拼错或越界的 kwarg 被静默接受或忽略；公开入口无法
  在代码与文档中声明自己的契约。违反 SPEC §56（fail loudly）。
- 三名黑名单（只拒绝三个已知内部名）：黑名单永远落后于新内部
  名，fail-open——内部新增参数会静默漏进公开入口，边界随时间
  腐化。
- 调用者自觉（约定 caller 不传内部参数）：无可机器验证的边界；
  违约不可观察，等于没有边界。

兼容后果：白名单落地后，此前被静默吞掉的越界 kwargs 将显式报错
——这是有意的破坏性收紧；依赖透传的调用方需改用显式列出的公有
参数。对合法调用，reference 的数学输出不变（SPEC §55：同数学
目标）。

新证据触发重审：出现合法需要新公有参数的证据（新的公有
prepare 选项）时，把该名字显式加入白名单并同步文档；不回退到
透传。规范链接：SPEC §46（reference prepare 语义）、§55
（fallback/同数学目标）、§56（fail loudly）——链接不复制，不另
起 ADR。

## 4. 口径钉死（与既有记录一致，集中声明）

- 21 REQUIRED 登记 + 文件门通过 ≠ 21 套 numeric 测试全部运行；
  本任（第五任验证批）实际运行的 numeric 恰为两文件：
  test/history_cache_contract_tests.jl 与
  test/incremental_primal_prep_tests.jl（另有一条架构门命令）。
  无 N65 fit、无 artifact 重放、无多日/fullsuite/GPU/bench。
- full suite / 多日 / 宏基准 / GPU / 吞吐继续暂停。
- 每条命令 ≤60 s、RSS2048 硬约束不变。
- 9×65 残差为样本行保真，不称全历史 T×N；浮点 tol 等价非跨机
  字节恒等；panel provenance unknown。
- 本任全部结论为单机文档核对；manager3/4 旧证据、日志、
  snapshot、manifest、源码、测试一字未改。
