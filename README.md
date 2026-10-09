# KTrader 2.0.0 Final — CPU

价格历史 → 因果多尺度特征 → 独立 OOF 后验 → 收益场景 → 原始对数 Kelly → 执行。

2.0.0 是 CPU 正式版本。保留原 1.0 数学定义和已有 `*_v1` API；GPU 属于
[2.1](ROADMAP_2_1.md)，不是本版本依赖或发布门槛。

## 安装与版本校验

验证环境为 Julia 1.12.7、OpenBLAS、Linux。使用随包提供的 Manifest 锁定依赖。
以下安装命令会按需要下载依赖，不会下载行情或连接交易账户：

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. bin/verify_release.jl
julia --startup-file=no --project=. -e 'using KTrader; println(Base.pkgversion(KTrader))'
```

版本应为 `2.0.0`。校验器核对已验收的 CPU 源码、测试、锁定依赖和验收凭据，
并验证此次版本号变更是唯一 Project 元数据变化。它不跑拟合、不加载 GPU，
也不把哈希校验冒充新的数值测试。

## 使用

`data/close.csv`、`data/adj.csv` 使用 `date` 列加资产列；未观察价格用 `NaN`。
行情、账户信息和已拟合模型不随源码包发布。

```julia
using KTrader, Random, LinearAlgebra

BLAS.set_num_threads(6)
bars = load_bars("data")
prices = signal_prices(bars)   # 缺失交易日保持 NaN，不用前向填充价格推断
prepared = prepare_reference(prices; F_folds=3)
model = solve(prepared)
scenarios = generate_scenarios_v1(model; S=300, rng=MersenneTwister(1))
weights = KTrader.scenario_weights(
    scenarios, model.active_indices, bars.bar[end, :], nothing)
```

完整回测命令：

```sh
DATE_TASKS=1 BLAS_THREADS=6 ENGINE=batch SCENARIOS=300 \
  julia -t 1 --project=. bin/backtest.jl 2026-09-22
```

起始日期由本地数据决定；不要把短窗口计时外推为长回测耗时。本版本的已验证
资源配置是一日期任务／BLAS6／默认 GC。双任务曾接近 2 GiB 护栏，未作为
稳定内存配置推荐，也没有修改原 CLI 默认值。

## 数学与实现边界

`prepare_reference → PreparedProblem → solve` 是主路径。增量准备最终也进入
同一个求解器；资源预算只选择准备路径，不改变结果。full 和每个 OOF 折
独立拟合。所有返回仍须原 alpha、协方差和 Kelly 证书通过，失败即报错。

2.0 优化了拟合内几何和工作数组复用、按需梯度、同 alpha 证书计算、稀疏
观察掩码遍历、增量列拷贝和 Kelly 矩阵布局。没有缩历史、截秩、减折或
降精度。`*_v1` 名称描述数学/API 兼容性，不是残留的包版本号。

核心层次：`data / geometry / numerics / response / residual_oracle / prepare /
predict / incremental / kelly / backtest / broker / live`。生产模块不加载
`dev/` 中的诊断或 GPU 原型。执行入口需要账户授权，本次发布不会调用它们。

## 验证与已知限制

已验收 CPU 源码的 41 个标准文件、22 个分组全部通过；两个不重叠的 N65
八日窗口已验证。较早窗口的独立逐日模型和两次进程输出比较，仓位 L1 和
收益差异均为 0。详细证据与边界见 [CPU 2.0 交付](RELEASE_CPU_2_0.md)。

测试入口和有界执行示例：

```sh
julia --project=. test/runtests.jl
bash bin/scoped_run.sh 45 /tmp/ktrader-architecture.log --rss-guard=2048 \
  julia --startup-file=no --project=. test/runtests.jl --architecture-only
```

历史完整覆盖采用分组运行，不是单进程全套；数值／数据／执行模拟组使用
`-O1` 控制测试编译成本，其余组与正式性能测量使用默认优化级别。局部验证
不保证任意历史都收敛，也不证明 CPU 已无优化空间。42 days/s 的性能目标
尚未达到。源码校验和测试不替代部署前对自有数据与执行环境的验证。

## 发布内容

[RELEASE.toml](RELEASE.toml) 固定 Final 版本与验收来源；
[CHANGELOG.md](CHANGELOG.md) 汇总变化；[ROADMAP_2_1.md](ROADMAP_2_1.md) 记录 GPU 范围。
源码归档包含完整测试和必要开发测试库，不包含行情、拟合快照、GPU 二进制
或逐轮开发历史。仓库中的原始证据保留，旧 README 位于
`dev/evidence/final_2_0_0_20261009/README.pre-final.md`。

本仓库未配置远端；此版本的交付是本地发布提交、`v2.0.0` 标签和源码归档，
不表示已上传 GitHub 或 Julia 注册表。许可范围不因本次封版而另行扩大。
