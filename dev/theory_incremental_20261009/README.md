# 增量求解研究，非2.0生产路径

主文档：[THEORY.md](THEORY.md)。它给出证明、条件、反例、复杂度和数据核验。

核心结论不是“热启动一下”：

1. 历史重标存在精确mask交换子结构，但必须保留满秩的ridge先验缺陷。
2. 原conditioned evidence的协方差Hessian是Sylvester基算子加至多106维
   矩修正，联合alpha的Newton块消元为107维，混合导数成本没有被省略。
3. 实际最慢第三折的19个floor方向来自构造性的训练支撑缺失：数据谱可缩
   到630维、协方差未知块45维，全部未识别先验/trace条件均值仍解析保留。

`identities.jl`只实现研究恒等式与内点Newton方向，没有接受规则，不可直接
用作生产求解器。`checks.jl`是小矩阵/反例检查；`support_face_audit.jl`
用本地已保存模型验证实际第三折结构；`local_structure.jl`为较宽的只读
准备/谱诊断。都不调用fit_v1/solve/optimize_conditioned_eb/backtest。

示例（仓库根目录，原资源护栏）：

```sh
bash bin/scoped_run.sh 45 /tmp/ktrader-theory.log --rss-guard=2048 \
  env JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no --compiled-modules=existing --project=. \
  dev/theory_incremental_20261009/checks.jl
```

下一项主工作：实现构造支撑面上的原证书候选、精确alpha混合导数与
跨日校正；以原选择语义和heldout隔离验收。没有证明通用ragged+重标
问题每日全链O(P²)，没有测得新的wall-time收益。2.0源码与标签均不变。
