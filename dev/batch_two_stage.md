# 批量单日决策：两段式调度 runbook（BATCH-2STAGE-1）

**状态：工程脚手架（dev/），不改数学/门禁/容差；数学对象与证书口径均不动。**

## 背景

过门配置（mu_qmc=true + chisq_qmc=true + max=131072）的完整单日链 85-95s，超过
单命令有效窗口（~48s）。分段：prep 28.6s + kelly 40.8s（已验证收敛：A=6.26e-5、
三项证书全过，证据 CQ_*）。两段均在窗口内。

## 每日两段（示例）

    # 段 A：prep 捕获（t=330；正常退出 rc=0）
    GATE0_DATA_DIR=<repo>/data julia --startup-file=no --project=. \
      dev/batch_t_capture.jl 330 dev/batch_out

    # 段 B：离线 kelly 求解（过门配置）
    # 默认 min=65536 / max=131072 / mu_qmc=true / chisq=true
    julia --startup-file=no --project=. \
      dev/batch_t_solve.jl dev/batch_out/batch_t330_prep.bin

## 60 日循环（示意；每段独立命令，≤60s 护栏内）

    for t in $(seq 330 389); do
      julia --startup-file=no --project=. dev/batch_t_capture.jl $t dev/batch_out || exit 1
      julia --startup-file=no --project=. dev/batch_t_solve.jl dev/batch_out/batch_t${t}_prep.bin || exit 2
    done

## 退出码（独立可辨，不得合并）

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数错误 |
| 3 | 段 B 自检失败（--selftest） |
| 10 | 段 A：prep 构造失败 |
| 11 | 段 A：落盘失败 |
| 20 | 段 B：读取/契约校验失败（缺字段、版本不匹配、残缺文件） |
| 21 | 段 B：kelly 求解失败（含 D-067 fail-loud） |
| 22 | 段 B：落盘失败 |
| 23 | 段 B：解归一性检查失败 |

## 产物契约

- 段 A 产物：batch_t<T>_prep.bin（NamedTuple，version=batch-prep-v1；含 post/st/xt/
  s1/E_active/locked/budget/t/seed/held_mode/rule_kwargs）。临时文件 + 原子重命名：
  要么完整要么不存在。
- 段 B 产物：batch_t<T>_solve.txt（w、三项证书、M、耗时、norm_err）。同样原子写入。
- 读取端显式校验字段/版本/held_mode，不匹配即 rc=20（不得静默接受）。

## 等价性近似（登记）

min=65536 跳过 64..65536 的中间层；t=330 已知该区间全部不过。若某日中间层可早停，
min=65536 会跳过该早停——首版以深段判定运行，min 可配置（GATE0_BATCH_MIN）。

## 边界

- held 首版固定 zeros（跨日持仓推进登记为后续）。
- 段 A 与 driver.jl 步骤 0-11 同式（复刻）；driver 变更须同步本脚本。
- 60-day 为批量负载：逐日两段、串行调度；不得把单日计时外推为长回测耗时。

## 三段式拆分（当段 B 超过 ~48s 窗口时，B1'/B2'/B3'）

段 B 的 50-58s 组成可分：w_65536 与 w_131072 独立计算；A/B 是后验统计量
（用 131072 或 audit 场景矩阵评估）。逐点等价性见 dev/batch_t_solve3.jl 头注释
（rule 派生 / solve_layer / A/B 公式与 adaptive_scenario_kelly 逐点复刻）。

    # B1'：w_65536（gen + cash_kelly；seg1 落盘）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b1 dev/batch_out/batch_t330_prep.bin
    # B2'：w_131072（seg2 落盘；含 X_2M 缓存，N_R=1 时 ~1MB）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2 dev/batch_out/batch_t330_prep.bin
    # B3'：拼 A/B_opt/B_audit + 证书 + 归一（秒级）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b3 dev/batch_out/batch_t330_prep.bin

判据：三段拼的 A 逐位复现 CQ 探针的 6.2585e-5；原 batch_t_solve.jl（整段 B）
保留为对拍基准。退出码：31/32 B1'/B2' 求解失败；33 B3' 统计/归一失败；
41/42/43 各段落盘失败；其余（0/2/3/20）同批量纪律。

### 细粒度六段（当三段式的某段仍超窗时，如 t=331 的 B2' 两次超窗）

    # b1a：gen(65536) -> X1；b1b：X1 -> seg1（cash_kelly）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b1a dev/batch_out/batch_t331_prep.bin
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b1b dev/batch_out/batch_t331_prep.bin
    # b2a：gen(131072) -> X2；b2b：X2 -> seg2（cash_kelly）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2a dev/batch_out/batch_t331_prep.bin
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b dev/batch_out/batch_t331_prep.bin
    # b3：拼装（与三段式共用；seg1/seg2 结构一致）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b3 dev/batch_out/batch_t331_prep.bin

等价性：b1a+b1b ≡ b1、b2a+b2b ≡ b2（同 gen、同 cash_kelly，仅改变在
哪条命令里算）；每段目标 ≤25s。追加码：34 b1a gen；35 b1b kelly；
36 b2a gen；37 b2b kelly；44/45 X 产物落盘。

### b2b 出口链拆分（B2B-SPLIT-1/2/3；主求解 / 候选构建 / tie-break 分段）

起因与演进：b2b 出口链先出窗（FB 诊断：主求解进窗、出口链出窗）→ SPLIT-1
（prelude / finish 两段）；finish 段再拆 → SPLIT-2（b2b2 候选构建 + b2b3
tie-break）；prelude 段再拆 → SPLIT-3（b2b1a prep + b2b1b solve）。与
dev/batch_t_solve3.jl 实际实现对齐：

    # b2b1：主求解（_cash_kelly_prelude）→ pre2 中间产物（kind=:pre）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b1 dev/batch_out/batch_t332_prep.bin
    # b2b2：候选构建段（_cash_kelly_finish_main，SPLIT-2）→ mid2 中间产物（kind=:mid）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b2 dev/batch_out/batch_t332_prep.bin
    # b2b3：tie-break 段（_cash_kelly_finish_tiebreak，读 mid2）→ seg2
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b3 dev/batch_out/batch_t332_prep.bin
    # SPLIT-3（b2b1 仍超窗时，prelude 再拆）：
    # b2b1a：prep 段（_cash_kelly_prelude_prep）→ prep1 中间产物（kind=:prep1）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b1a dev/batch_out/batch_t332_prep.bin
    # b2b1b：主求解段（_cash_kelly_prelude_solve，读 prep1）→ pre2（与 b2b1 同格式）
    julia --startup-file=no --project=. dev/batch_t_solve3.jl b2b1b dev/batch_out/batch_t332_prep.bin

等价性：prelude + finish_main + finish_tiebreak 组合 = cash_kelly 逐点一致
（kelly_cash_tests 的 B2B-SPLIT-1 testset 直接对拍）；各段产出的 seg2 与 b2b 的
seg2 结构一致（t=330 的既有 seg2 可对拍）。退出码：38 b2b1 / 39 b2b2 / 40 b2b3
段失败；50 b2b1a / 51 b2b1b 段失败；46 b2b1 / 48 b2b2 / 47 b2b3 / 52 b2b1a /
53 b2b1b 落盘失败。