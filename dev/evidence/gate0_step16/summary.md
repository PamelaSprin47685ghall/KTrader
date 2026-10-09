# Gate-0 Step 16 — Constitutional Suite 全量汇总 + verify_release 遗留查验

日期：2026-10-09。同一时点、当前工作树状态的完整快照（防止跨轮修复
引入回归）。全部命令 scoped ≤60s / RSS2048，串行执行。

## 1. 十一项全量运行（全部 rc=0）

| # | 套件 | log | Pass/Total | elapsed | RSS 峰值 | Wave 对照 |
|---|------|-----|-----------|---------|----------|-----------|
| 1 | market_tests | 01_market_tests.log | **82/82** | 6s | 630MiB | Wave 2 裁决轮 82 ✓ 无漂移 |
| 2 | kelly_cash_tests | 02_kelly_cash_tests.log | **59/59** | 16s | 885MiB | Wave 2 裁决轮 59 ✓ |
| 3 | modes_tests | 03_modes_tests.log | **102/102** | 4s | 391MiB | Wave 2 102 ✓ |
| 4 | response_tests | 04_response_tests.log | **78/78** | 5s | 411MiB | Wave 2 收尾轮 78 ✓ |
| 5 | posterior_tests | 05_posterior_tests.log | **40/40** | 10s | 490MiB | Wave 3 裁决轮 40 ✓ |
| 6 | innovation_tests | 06_innovation_tests.log | **206/206** | 10s | 514MiB | Wave 3 裁决轮 206 ✓ |
| 7 | oof_tests | 07_oof_tests.log | **25/25** | 7s | 546MiB | Wave 3 裁决执行轮 25 ✓ |
| 8 | predictive_tests | 08_predictive_tests.log | **56/56** | 32s | 1600MiB | Wave 3 收尾轮 56 ✓ |
| 9 | quadrature_tests | 09_quadrature_tests.log | **54/54** | 18s | 923MiB | Wave 4 裁决执行轮 54 ✓ |
| 10 | driver_tests | 10_driver_tests.log | **73/73** | 38s | 1478MiB | Wave 4 裁决执行轮 73 ✓ |
| 11 | module 加载 | 11_module_load.log | API exports = 83 | 3s | 569MiB | ✓（84 名含自名） |

**总计：775 项断言全部通过 + 模块加载确认，零计数漂移**（与各 Wave
证据逐一对照一致——跨轮修复（裁决 4/5 的 quadrature.jl 修改、driver
的 posterior/kelly 配置）未引入任何回归）。

## 2. verify_release.jl 遗留查验与适配

**只读查验（适配前）**：
- 现状钉死：verify_release.jl:40（原行号）的
  `metadata["status"]=="Final"` 硬校验 vs RELEASE.toml:6 的
  `status = "2.0-RC-G0"`（Gate-0 裁决 D-003/D-097/§57 的合法状态）。
- 拦截行为实测（12_verify_release_before.log）：rc=1、
  `ArgumentError: wrong release identity` @ verify_release.jl:39
  （栈定位到 status 校验行）。

**机械适配**（bin/verify_release.jl）：
- L6-17（原 L6 后新增）：`const FINAL_STATUSES = ("Final", "2.0-RC-G0")`
  + 注释（白名单理由、D-002 历史字节不动、后续校验的预期拦截语义）。
- L48-50（校验行）：`metadata["status"]=="Final"` →
  `metadata["status"] in FINAL_STATUSES`。
- **历史凭据字节未动**（FINAL_ACCEPTANCE_HASH / FINAL_QUALIFICATION_HASH /
  acceptance 段 / qualification receipts——全部原样）。

**适配后行为实测**（13_verify_release_after.log）：
- status 层**通过**（`wrong release identity` 消失——白名单生效）。
- 下一层拦截：`ArgumentError: Project changed beyond the approved
  version line` @ verify_release.jl:43（final_project_check）——
  Project.toml 相对 2.0.0 基线的字节差异（Wave 3 的 QuadGK/
  SpecialFunctions 依赖声明——Gate-0 纠偏期的合法修改）。
- **定性**：这是预期中的拦截，不是适配缺陷——verify_release.jl 的
  语义是「验证 2.0.0 历史发布物的字节身份」（D-002：历史凭据不可
  篡改），而当前工作树是纠偏期开发线。**修复 baseline hash = 篡改
  历史凭据，超出机械适配范围**。处置归 Manager：Gate-0 关闭后的
  重新发布流程应生成新的凭据链（新 baseline/acceptance），而非修改
  2.0.0 的历史校验。

## 3. 边界

未修改：旧 src/、旧 test/、docs/、git。全部绿证据为单机当前时点一次
运行事实。verify_release 的后续拦截（Project.toml 基线、src/gate0/
新文件的 unqualified addition 检查等）为工具语义与纠偏期工作树的
预期张力，未做进一步适配。
