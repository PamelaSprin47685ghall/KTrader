# 2026-10-09：CPU 2.0 验收闭合，GPU 划到 2.1

本轮按用户的新范围停止扩展 GPU 工作。AGENTS.md 的版本速查、§80.4 和
mission 说明已同步；没有修改 CPU 数学、容差或既有失败记录。

生产代码、测试、Project.toml、Manifest.toml 全部保持上一轮已验证的字节。
本轮没有再跑所有拟合和 22 个测试分组，而是校验其唯一快照及各组真实凭据，
再完成此前未执行的输出比对。

## 本轮实际执行

`verify1_cpu20.log`：24 项逐日独立模型比较通过，8 天每日仓位 L1 均为 0。
命令 RC0、13 秒、RSS 峰值1146MiB。

`verify2_cpu20.log`：24 项逐日比较和4项两次进程结果比较通过；每日仓位
L1 和两次收益差异均为0。命令 RC0、14秒、峰值1142MiB。

两份窗口 SHA256 未变，验证程序本身仍是上一轮快照绑定的文件。
第一份原命令的124是完成数值工作后发生的后处理超时，保持原值。
新的验证命令通过不倒改它的历史退出码。第二份生产命令仍为原RC0。

新汇总器 `dev/cpu20_acceptance.jl` 对41标准文件/22分组凭据、当前代码、
输入、保存的完整窗口及独立模型作交叉核验，不运行拟合或GPU。
21项汇总器检查通过，其中包含坏凭据、错快照和缺失条目的反例。
这些21项不是模型数值回归，不能加到原套件计数里制造更大的通过数。

首次实际结果在 `audit.log`，RC0、5秒、峰值447MiB：

```
CPU20_ACCEPTANCE=PASS groups=22/22 standard_files=41 earlier_replays=2/2
independent_weight_L1=0.0 return_max=0.0 GPU_RELEASE=2.1 GPU_BLOCKER=false published=false
```

输出 `acceptance.toml` 的首次SHA256：
`dd1fe993b2ad5020e46db8f634fdb9254b816d114ce9dd7c38a64fb188c820df`。
后续 `status` 模式重新核对实际文件与输出，不仅仅读取这个PASS字符串。
该独立复核也实际RC0：`status.log`，21项汇总器检查通过，5秒、峰值448MiB。
验收TOML的SHA256与首次输出一致。末次进程检查没有Julia或HIP微基准进程
（ps退出1表示没有匹配项），没有遗留本轮计算任务。

## CPU 2.0 的边界没有被隐去

数值定义仍为1.0，batch/增量共用原求解器和证书。验收配置是本地 N65、
完整历史、F3/S300/default EB、Float64、单任务/BLAS6/default GC。
最近八日与旧认证输出相同，较早八日独立模型及独立进程比较现已闭合。
没有声称42days/s mission达标、任意日期必定收敛或CPU绝无优化空间。
双任务曾贴近2GiB护栏，不把它列为已证明有稳定内存余量的默认配置。

标准文件全范围为分组验证，不是单进程全套；数值/数据/执行模拟组-O1，
其余组和正式性能测量默认优化级别。这些历史配置在凭据中保持原样。

本地工程交付状态与公开版本发布分开：`cpu_engineering_acceptance_passed`
为true，`published`为false；Project.toml的包版本仍0.1.0，未commit/tag/
publish/deploy。AGENTS原本就区分概念里程碑与Git tag，本轮没有用改版本号
替代验收。根目录 RELEASE_CPU_2_0.md 和 ROADMAP_2_1.md 是交付入口。

本轮GPU既没有运行，也没有装依赖或改动设备设置；已有HIP原语及其日志只作
2.1开发输入，不进入CPU验收条件。
