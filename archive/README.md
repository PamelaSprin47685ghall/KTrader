# archive/ — 归档区（2026-10-10 树结构调整）

本目录存放**已离开活动树**的开发脚手架：轮次运行日志、一次性微基准脚本、
研究线与交付注记。移动时内容**一字未改**；这里保留它们是为了可 grep、
可追溯与反例/历史引用，不是为了发布。

git 策略：`archive/` 整体被根 `.gitignore` 忽略，且已从 git 索引剥离
（`git rm --cached`）。历史与 tag 原样保留；要取回任一文件的被跟踪版本：

```sh
git show 8d54f0d:dev/<原相对路径>      # 例如 git show 8d54f0d:dev/probes.jl
```

（`8d54f0d` 是结构调整前的最后一个提交。）

## 目录

| 目录 | 内容 | 原路径 |
|---|---|---|
| `evidence/` | 31 个轮次/研究证据目录与 `fit_local_reuse_*.log`（manager3–13、gate0_step16、gate0_wave2–4、batch_memory、conditioned_cpu、local_panel、oof_memory、release_batch/followthrough/qualification/work、response_deadwork、scalar_tensor、support201、theory_incremental） | `dev/evidence/…` |
| `scripts/` | 25 个一次性微基准 / 探针 / 发布辅助脚本（`*_micro.jl`、`local_panel_*.jl`、`release_*.jl`、`earlier_*.jl` 等） | `dev/*.jl` |
| `research/` | `support201/`（2.0.1 候选求解器研究线）与 `theory_incremental_20261009/`（增量求解推导） | `dev/support201`、`dev/theory_incremental_20261009` |
| `notes/` | 两份交付注记（`fit_local_reuse_20261008.md`、`response_deadwork_20261008.md`） | `dev/*.md` |

## 什么没有进 archive（仍留在活动树，因为它们被发布/测试/工具链消费）

- `dev/probes.jl`、`dev/m1_artifact_replay.jl`、`dev/m1_replay_manifest.toml`、`dev/m1_typed_replay.md`
- `dev/cpu20_acceptance.jl`、`dev/hip_core_micro.cpp`、`dev/hip_core_input.jl`
- `dev/evidence/{cpu20_acceptance, earlier_closure, final_2_0_0, release_freeze}_20261009/`：2.0.0 发布凭据链，被 `RELEASE.toml` / `bin/verify_release.jl` 钉死

## 已知偏差（如实记录）

`dev/evidence/earlier_closure_20261009/qualification/qualification_snapshot.toml`
的 `[sources]` 仍写着 `dev/*.jl` 旧路径（该 25 个脚本现居 `archive/scripts/`）。
该快照是历史记录、不随结构改写，因此依赖它的 `dev/cpu20_acceptance.jl`
与 `bin/verify_release.jl` 在现状树上按设计拦截 2.0.0 快照身份。要恢复可运行，
把 `archive/scripts/*.jl` 移回 `dev/` 即可（字节未变）。这与第 10 任交接
登记 f「新凭据链生成归重新发布流程」一致。

## 检索建议

```sh
rg -n "关键词" archive/                 # 全归档检索
rg -n "关键词" archive/evidence/manager10/   # 特定轮次
```
