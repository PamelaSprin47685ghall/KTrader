# Changelog

## Gate-0 纠偏工作线：P0 修复、数值治理与 A 门闭合（2026-10-10 至 2026-10-11）

依据 AGENTS.md 第 13–16 任段与 docs/GATE0_EXIT_CHECKLIST.md 的 2026-10-11
状态同步；以下为本工作线新增记录（下方历史条目未做改动）。

- **Gate-0 重开与 P0 修复**：2.0.0 数学地位撤销、Gate 0 重开（2026-10-09，
  见下方条目）；九项 P0（P0-1 strict prequential / P0-2 连续 d / P0-3 innovation
  满秩 / P0-4 locked 无回落 / P0-5 locked 组装统一 / P0-6 四证书 / P0-7 严格
  数值合同 / P0-8 proper HalfCauchy-τ（`_WEAKINFO` 删除）/ P0-9 canonical
  tie-break）逐项落地；静态核对表 dev/gate0_static_audit_2026.md。
- **求积与几何修复**：2D 后验求积升级为张量 G-L 自适应规则（收敛指数 2.014、
  1929 cells 达 1e-6；docs/ADJUDICATION_T327_POSTERIOR_QUAD.md）；ruler 短历史
  回退改为 τ=1 RMS 平坦外推（docs/RULER_SHORT_HISTORY_FALLBACK.md）。
- **kelly 数值治理链**：条件行缩放（span>4.5e15 阈值；
  docs/KELLY_NUMERICAL_ROW_SCALING.md）、polish 轮数修复、M2 出口证书化、
  设计 A 双路径取优、TIE-FAST-1 全 cash 快速路径、prequential 成本加速
  （docs/PREQUENTIAL_COST_ACCELERATION.md）。
- **A 门闭合**：过门配置（GATE0_MU_QMC=true + GATE0_CHISQ_QMC=true +
  GATE0_MAX_SCENARIOS=131072）下 t=330–346 十六日十二过四超（未收敛日
  336/340/343/344；fail-loud）；首次过门 t=330/331。证据
  archive/evidence/gate0_multiday_run_20261010/（AGENTS.md 第 15/16 任段）。
- **SPLIT 链与批量调度**：b2b 出口链三级拆分 B2B-SPLIT-1/2/3（逐点等价；
  kelly_cash 26/26）；九段批量调度稳定（零重试、最大段 40s）。
- **未收敛日决策包**：b/c/d 三路径整理并交 owner/SPEC
  （docs/KELLY_NUMERICAL_ROW_SCALING.md §9.13；d 已由 S10 诊断否定——本链
  n=1 无零列）。

测试口径：test/gate0/ 全量 984 项全绿（2026-10-10 口径；
archive/evidence/gate0_final_verify_20261010/）；其后 kelly_cash 新增
B2B-SPLIT-1 testset（终态 26/26）；受影响模块（quadrature/driver/backtest
场景 1–8/kelly_cash）已于 2026-10-11 字节受控回归通过
（`archive/evidence/gate0_regress_20261011/`）；其余组引用既有全绿
（`archive/evidence/gate0_final_verify_20261010/`）。

## 树结构调整：脚手架归入 archive/（2026-10-10）

依据用户当日指令（记录于 AGENTS.md 第 12 任段）：全部开发用脚手架正规化
或归入 `archive/`，被忽略文件退出 git 索引，历史不动，入口收敛。

- 移动（内容字节未改，只改路径）：31 个轮次/研究证据目录与 `fit_local_reuse_*.log`
  由 `dev/evidence/` 入 `archive/evidence/`；25 个一次性脚本由 `dev/` 入
  `archive/scripts/`；两条研究线入 `archive/research/`；两份交付注记入 `archive/notes/`。
- 保留在活动树（仍被发布/测试/工具链消费）：`dev/probes.jl`、`dev/m1_artifact_replay.jl`
  及其 manifest 与注记、`dev/cpu20_acceptance.jl`、`dev/hip_core_*`，以及发布凭据链
  `dev/evidence/{cpu20_acceptance, earlier_closure, final_2_0_0, release_freeze}_20261009/`。
- git：根 `.gitignore` 统一全部规则并新增 `archive/`；8 个嵌套 `.gitignore` 删除；
  `archive/` 下全部路径与此前跟踪的归档文件从索引剥离（`git rm --cached`，728 项），
  工作树与 git 历史、tag `v2.0.0` 不变。取回：`git show 8d54f0d:<原路径>`。
- 文档：README 新增「仓库结构」节；docs/ 与被改路径的文件引用同步改写；
  AGENTS.md 以追加段记录（旧规范文本一字未改）。
- 发布凭据链未受损（被钉路径全部在保留集合内）；`bin/verify_release.jl` 与
  `dev/cpu20_acceptance.jl` 对 2.0.0 快照的预期拦截点不变，详见 AGENTS.md 第 12 任段。

## 状态变更：撤销 2.0.0 Final 地位，Gate 0 重开（2026-10-09）

依据 AGENTS.md 的 Gate-0 裁决书（D-003、D-096、D-097、§57）：2.0.0 保留
为历史发布物，原字节、原 hash、原时间线不变（D-002）；但其「数学闭合」
声明已被 Gate-0 审计撤销，不再是当前推荐规范。当前开发状态改为
KTrader 2.0-RC / Gate 0 Reopened（版本线记号 2.0-RC-G0）。RELEASE.toml
的 status 同步改为 "2.0-RC-G0"，[acceptance] 历史验收凭据原样保留；
README.md 与 RELEASE_CPU_2_0.md 头部已新增 Gate-0 状态声明，旧表述按
发布时点时间线理解。本条为新增记录，下方既有条目未做任何改动。

## 2.0.0 Final — 2026-10-09

CPU release. GPU development belongs to 2.1; no GPU dependency is loaded by
the production package. The original 1.0 mathematical definitions and
public `*_v1` interfaces remain compatible.

### Delivered

- One typed prepare/solve boundary, independent full and OOF fits, exact
  incremental preparation with reference fallback and explicit ownership.
- Fit-local geometry, alpha-cache and work-array reuse; unused gradient
  elimination; compact scalar tensors and lower-allocation residual paths.
- Dense free-scenario packing for certified Kelly, and removal of redundant
  full-prefix caches from the backtest without weakening public cache guards.
- Producer exceptions reach the backtest caller; an existing certified
  terminal candidate covers a previously omitted failed-ray branch.

### Finalization

Package version is 2.0.0. Accepted CPU implementation, mathematical tests
and dependency lock remain byte-identical to the final engineering audit.
The release has a standalone integrity verifier and an explicit CPU source
archive allowlist. Private market data, model snapshots, generated binaries,
GPU prototypes and historical development reports are not shipped in that archive.

### Validation scope

41 standard files / 22 groups, plus independent model and process replay of
the earlier full-history eight-day window. This is grouped coverage, not a
single-process full-suite run. The numerical/data/execution group used O1;
other groups and production measurements used the default optimization level.
The scope is the documented local CPU profile, not convergence on all data,
proof of maximum speed, or safe memory headroom for arbitrary parallelism.

Existing log failures, timeouts and source snapshots remain historical
records. A later passing check does not rewrite an earlier command outcome.
