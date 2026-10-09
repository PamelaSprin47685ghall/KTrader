# 主要后验计算的跨日增量化：可证明的结构与尚未跨过的门槛

2026-10-09。数学对象仍为 Path Kelly 1.0；不修改2.0 Final的生产代码、
依赖、测试、版本号或标签。本文是推导，不是已上线的增量求解器，也不声称
首次提出 Woodbury、Riccati、Schur 补或预测—校正这些既有工具。

本文区分：代数恒等式、带条件的复杂度结论、已核对的实际数据、待完成的
算法验收。数值小实验只能查错，不能代替证明或完整后验回归。

## 0. 记号与问题位置

N=资产数，C=14=七频带×两通道，P=CN，d=N−1，n=某个full/fold训练行数。
约束数m=C=14；对称协方差有d(d+1)/2个坐标，N65时为2080。
原生产求解器并没有构造2080×2080的稠密牛顿矩阵；它用Riccati/不动点及
线搜索。因此不能拿“避免了原本不存在的2080维分解”宣称测得加速。

主路径证据：src/response.jl中的conditioned_cached_state、
conditioned_constraint_jacobian、conditioned_eb_certificate；
src/predict.jl的full/OOF求解；src/incremental.jl的原始收益/mask因子。
下面的Sigma均写在global zero-sum gauge内，故为d×d正定矩阵S。

## 1. 尺度变化应先搬运坐标，但不能漏掉先验

观察mask为m，c=1'm>0，P_m=diag(m)−mm'/c。原始收益z保留缺失mask，
写成zero-filled向量只是原field的代数表示。旧尺度D0、新尺度D1均为正
对角矩阵，L=D1 D0^{-1}，u=D0z。

**命题1（精确交换子）。**

    P_m L − L P_m = [(Lm)m' − m(Lm)']/c,
    e_new = L e_old + [(Lm)(m'u) − m((Lm)'u)]/c.

证明：diag(m)与L交换，展开两个中心化项即可。因此每个mask的残差落在
span{m,Lm}，秩至多2。空mask两边都为0。

对任意线性因果滤波h_{i,s,c}，令旧、新通道逆尺度为b0_c,b1_c。
在同一批历史行上，有

    X_new = X_old A + U V',
    A = blockdiag_c[(b1_c/b0_c)L].

若出现K个不同mask，令M=[m1,...,mK]，那么可取

    rank(UV') <= q_x <= C rank([M,LM]) <= C min(N,2K).

无需将任意小奇异值剪掉。这个界来自列空间的构造性因子，而非数值秩猜测。
单一固定mask下，在其zero-sum基底中可进一步减少到每通道一个公共分量。

**旧相对场本身并不足以搬运。**u和u+a1有相同旧相对场，但当L不均匀时，
P L u和P L(u+a1)不同。必须保留原始统计或丢失的公共分量，不能只缓存旧G/Sigma。

**命题2（Gram搬运）。** 若X_new=X_old A+UV'，则

    X_new'X_new − A'X_old'X_old A = Z K Z',
    Z=[A'X_old'U, V],   K=[0 I; I U'U].

证明：展开乘积的两个交叉项和UV'自身的二次项。故这一差的秩至多2q_x。
新加入/从某折训练集中删掉r_row行，再加一个秩至多r_row的带符号更新。

连续F折在n增加1时，只有floor(n/F)变化时边界才移动。每个训练补集至多
变化2F−2行（F>=2，已有非空折）；F3时至多4行。这是固定当前特征坐标下
的组合事实，不意味着每天全部历史特征只变4行。

**不能漏掉ridge项。** 令Aold_precision=X_old'X_old+alpha0 I，则

    Anew_precision − A' Aold_precision A
      = low_rank_term + alpha1 I − alpha0 A'A.

最后一项一般满秩；ambient坐标下它是对角的，但不是低秩。
把旧先验随A搬走、又把新先验当成alphaI，会改变模型。只有A'A等于一个
标量单位阵且alpha相应一致等特殊情形，才可消掉该项。
因此“搬运后低秩”本身不足以宣布全部线性代数O(P²)。

当实际更新确为秩r且SPD合法时，维护逆/分解、行列式可以使用
Woodbury/Cholesky update-downdate，典型dense代价O(P²r+r³)，不是每次O(P³)。
对于alpha变化、满秩非均匀重标和许多mask，这个条件必须重新证明，不能省略。

### 当前N65诊断

2026-09-10→09-11，两次prepare和已保存后验，零重新拟合。
原始mask有60种，mask列空间秩60，[M,LM]秩64；上述一般界为896，已接近P910。
搬运后的共同历史design残差相对Frobenius为9.908870969e-5，按E'E谱估计、
相对奇异阈值1e-6的数值秩503。该计数不是精确rank，也不与上一轮对另一个
矩阵的867/910按同一门槛比较。它说明：不能把真实数据当作一两阶更新。

## 2. 固定alpha后，协方差Hessian有精确的106维修正

从原生产公式取负evidence，丢掉独立于S的常数：

    f(S) = f0(S) + phi(A(S)),
    f0(S) = n/2 logdet(S) + 1/2 tr(R S^{-1}),
    A(S) = [svec(M(S)); tr(S)],
    phi(M,t) = 1/2 logdet(M) + 1/2 h'M^{-1}h − m/2 log(t).

M_rs(S)=<H_rs,S>；H_rs是原对称Jacobian core，加上dual baseline的对角项。
svec使用Frobenius正交坐标：对角元素原值，非对角乘sqrt(2)。因此A输出
q=m(m+1)/2+1=106个数。它是低维统计映射，不表示每个H_rs矩阵低秩。

**命题3（精确Hessian分解）。**

    Hess f = H0 + A* G A,   G=Hess phi,
    rank(A* G A) <= q.

证明：A在S上是线性的，二阶链式法则没有额外的A二阶项。
令V=M^{-1}，z=Vh，则

    grad_M phi = (V−zz')/2,
    Hess_M phi[E] = (−VEV + VEzz' + zz'EV)/2,
    Hess_tt phi = m/(2t²),   Hess_Mt phi=0.

这正对应生产代码的K=M^{-1}−zz'及J=A* K，不是新目标。

**命题4（H0逆是Sylvester算子，不是d²维稠密求逆）。**
令S=LL'，对称分解L^{-1}R L^{-T}=U diag(lambda)U'，T=LU。
对任意对称Y，若所有lambda_i+lambda_j−n不为0，则

    H0^{-1}[Y] = T [ (T'YT)_ij / ((lambda_i+lambda_j−n)/2) ] T'.

证明：先求

    H0[H] = −n/2 S^{-1}H S^{-1}
          +1/2 S^{-1}H S^{-1}R S^{-1}
          +1/2 S^{-1}R S^{-1}H S^{-1},

再用H=L Z L'作合同变换，得到
(Z Q+Q Z−nZ)/2；在Q的特征基中逐元素可逆。
若min(lambda)>n/2，则H0正定，这是研究原型明确检查的准入条件；
它不是Hess f整体正定、约束合法或牛顿步应被接受的证明。

**命题5（106维Schur方程）。** 原牛顿方程

    (H0 + A* G A) Delta = −g

等价于先求

    W=A H0^{-1} A*,   b=A H0^{-1}g,
    (I+WG)y=−b,
    Delta=−H0^{-1}(g+A*Gy).

只要求相关基算子与Schur系统可逆，不要求G可逆；不能用G^{-1}版本
隐含排除退化方向。解出后仍检查原牛顿线性残差及原非线性证书。

构造W可在同一特征基中完成：令F[:,a]=vec(T'A_a T)，D_ij=(lambda_i+
lambda_j−n)/2，则W=F' diag(vec(D)^{-1}) F。
每次精确校正的代价为O(q d³+q²d²+q³)，内存O(qd²+q²)，不会形成
2080×2080矩阵。这是一个可实现的精确牛顿方向，不是“拿近似梯度少跑几步”。

局部曲率也有小系统判据：当H0正定时，Hess f正定等价于
I_q+W^{1/2}GW^{1/2}正定（多余零秩方向只给特征值1）。这是由
H0^{-1/2}A*GAH0^{-1/2}的非零谱与W^{1/2}GW^{1/2}相同得到。
它能检查当前点，不自动证明整片邻域正定；分支唯一性仍需要在邻域内的
一致下界。不能把“基Hessian正定”偷换成“原问题强凸”。

### alpha联合校正：线性系统107维，但导数不是免费的

令a=logalpha，u=f_Sa，h_aa=f_aa，g_a=f_a。u和h_aa必须来自当前完整evidence，
包括alpha依赖的R/h/H_rs，不能把它们冻结成0。令v=A H0^{-1}u，则联合
牛顿方程可消元为

    [ I+WG        v                  ] [y      ] = [−b                      ]
    [ −v'G   h_aa−<u,H0^{-1}u>       ] [Delta_a]   [−g_a+<u,H0^{-1}g>       ].

然后Delta_S=−H0^{-1}(g+A*Gy+u Delta_a)。该系统为107维。
原型只测试这个消元恒等式；尚未交付生产alpha混合导数或原全区间候选选择。
计算alpha相关谱函数、对数行列式和导数的成本没有被这个维数计数消除。

## 3. 最慢第三折的19个floor方向：可构造的支撑，而非数值剪秩

### 3.1 从mask建图，得到与当前尺度无关的支撑

对参与训练目标或因果特征历史的每个mask，将同时观察的资产连成图。
因为range(P_m)=span{e_i−e_j : i,j在mask内}，所有这些投影的像空间之和正好是
各连通分量内和为0的空间。实际收益行处于这个空间，不要求其数值张成整个
空间。设其正交基为Z，维数r；该空间由mask构造，
与收益取值及每日尺度无关，不通过“小特征值阈值”猜测。

若图有b个连通分量（包括孤立资产），r=N−b。整个global relative空间
还有a=d−r个分量公共方向。记P_A=P_global−ZZ'。
所有训练X每个通道和Y均处于Z：

    E=I_C kron Z,   X=X_R E',   Y=Y_R Z'.

取覆盖全部相关历史的mask图可以是保守上界；额外历史最多减少压缩收益，
不能漏掉实际特征所需的mask。代码诊断对第三折使用完整训练前缀的图。

末折训练集是递增的前缀，因此其图只加边；资产全集固定时，总共至多N−1
次连通分量合并。这只界定了结构支撑改变的次数，不界定所有数值floor
切换或异常日期的次数。其他折有训练行删除，不能照搬这个单调计数。

### 3.2 似然特征维数可以减少，先验不能删

**命题6（保留完整ridge prior的结构消元）。**

    V=(X'X+alpha I_P)^{-1}
      =E (X_R'X_R+alpha I_Cr)^{-1} E' + alpha^{-1}(I_P−EE').

证明：在range(E)和其正交补上分别作用即可。
因此需要分解的数据矩阵可以是Cr×Cr，而全部未被数据识别的先验方差
alpha^{-1}仍存在。不是把posterior rank改成Cr。

对logdet有

    logdet(I_P+X'X/alpha)=logdet(I_Cr+X_R'X_R/alpha).

但是原evidence前面的输出维数系数仍然是d/2，**不能改成r/2**。未观测
输出的零场分量仍属于当前模型的似然定义，不因计算消元而消失。

该压缩只作用于某个fold的后验线性代数。当前整段输入所确定的ruler、
频带尺度、global active universe和fold布局都保留。特别是compute_s_perp
中的原N/(N−1)归一化，不能因为数据谱缩到Cr维就改成r/(r−1)。若改掉
这些常数或只用训练子集重新定义尺度，就不是这里证明的同一个模型。

### 3.3 原floor面上的完整目标与KKT可提升

考虑候选Sigma=Z B Z'+delta P_A，B>=delta I_r，delta为原floor。
这里是“构造并验证一个原约束面”，不是先验断言全局最优一定在这个面。
在图支撑下R、输入V各通道块都与该分解相容，故

    M(B)=M_R(B)+(a delta/alpha) I_m,
    t(B)=tr(B)+a delta.

固定alpha时，完整evidence（保留所有与alpha有关的项）为

    ell = −d/2 logdet(I_Cr+X_R'X_R/alpha)
          −n/2 [logdet(B)+a log(delta)] −1/2 tr(B^{-1}R_R)
          −1/2 logdet(M) −1/2 h'M^{-1}h
          +m/2 log((tr(B)+a delta)/alpha).

M,t现在是B的仿射映射；第2节Hessian低维修正仍成立，常量偏移不产生二阶项。
alpha联合导数必须同时对a delta/alpha求导。

设K=M^{-1}−M^{-1}hh'M^{-1}，原evidence在A子空间的梯度为gamma I_a：

    gamma = −n/(2delta) − tr(K)/(2alpha) + m/[2(tr(B)+a delta)].

**命题7（KKT提升）。** 若B通过其原floor的KKT，gamma<=0，且上述支撑
恒等式成立，则完整Sigma通过原数学的KKT：自由块条件由B给出，Z/A交叉
梯度恒为0，A块满足最大化问题的单边条件。不需要把数值找到的19个特征向量
冻结下来，也不省掉最后的原生产证书。
若gamma>0，或支撑不成立，必须退出此面而非强行接受。KKT不是全局最优
或与cold参考选中同一驻点的充分证明，原分支选择问题仍须单独解决。

**最容易漏掉的项：未识别系数的均值不一定是0。**
令lambda=M^{-1}h。每个通道的条件均值在A×A块为

    P_A G_c P_A = −(delta/alpha) lambda_c P_A.

它是trace-neutral约束把修正分配到未识别方向的结果。若把这些方向的
系数直接删掉或置0，heldout中的新资产预测会改变。先验协方差也必须保留：
完整条件协方差仍由Omega−Omega C'(C Omega C')^{-1}C Omega计算或等价
因子计算；不同支撑块会经14个trace约束产生条件相关，不能只留对角方差。

### 3.4 不满足结构时，简单floor裁剪确实错误

令S0=diag(0.5,2)，K=[[0.2,0.1],[0.1,0.2]]，R=S0+S0 K S0。
对f(S)=(logdetS+tr(RS^{-1})+tr(KS))/2，S0为驻点。
取floor=1并将其谱裁剪到diag(1,2)，则grad f的交叉项为0.025，不满足
active/free旋转条件。这已作为明确反例测试，不能把无约束Riccati解
逐特征值clip当作一般带floor问题的解。

没有构造支撑可用时，需解真正的cone KKT，例如投影残差

    R_eta(S)=S−Pi_floor(S−eta grad f(S)).

在投影谱值不等于阈值处，DPi由原谱的差商给出；其Jacobian仍是
“I−DPi+eta DPi H0”加秩<=106的修正。然而前一个基算子目前没有一般廉价
逆公式，不能据此宣称所有floor状态也已压成106维求解。

## 4. 跨日校正为何有望减少迭代，以及它的严格前提

令F_t(theta)=0表示同一个被选中的、正则的KKT分支。在一个公共邻域内，
假设Jacobian可逆且逆范数<=B，Jacobian的Lipschitz常数<=L，昨日点在
今日问题上的残差<=c epsilon_t。以今日Jacobian作一次Newton校正，则

    ||F_{t+1}(theta+Delta)|| <= (L B² c²/2) epsilon_t².

证明：Delta=−J^{-1}F，Taylor展开的一阶项抵消，余项用L||Delta||²/2界定。
同样条件持续成立时，多次校正具有二次误差递推。
若正规化充分统计的日变化是O(1/T)，且上述常数、解分支和约束间隙稳定，
一次校正残差可到O(T^{-2})；这是带条件的结论，不是从“新增一天”自动推出。

当前模型还会遇到新资产、mask连通性改变、ruler有效tau门槛变化、极端新
观测、alpha边界、非结构floor激活和近退化Hessian。需要按实际事件率计算

    E[C_day] = C_statistics + C_factor_updates
               + E[K] C_correction + C_original_certificate
               + p_rebuild C_reference_rebuild.

这个账不能漏掉对数行列式、alpha混合导数、原分支选择及恢复路径，也不能
把首次构造、分解刷新和失败回退藏到计时之外。重建概率目前没有数据证明。

### OOF隔离不是“昨天的东西就安全”

今日heldout可能昨天属于训练集。精确统计的加减会移除其直接数据贡献，
但用旧后验作起点仍可能影响非凸求解选中了哪个驻点。原KKT通过不能排除
这种路径依赖。因此增量状态只能提出候选，不自动决定今日答案。
需要证明目标分支在一个与今日数据绑定的区域中唯一并和参考一致，或保留
可验证的参考选解机制。定期跑一次cold只能做经验诊断，不能证明中间日均
无泄漏。必须保留原heldout扰动和moving-boundary对照。

## 5. 真正可下的复杂度结论

1. 固定合法坐标/先验且仅r_row行变化：矩阵分解更新从O(P³)变为
   O(P² r_row+r_row³)。这个条件在一般每日重标情况下不自动满足。
2. 构造图支撑r<N：第三折数据谱可从CN维缩到Cr维，原未识别先验解析保留；
   floor面经原KKT检查后协方差计算用r而不是d。结构保持期间可以跨日维护。
3. 正则协方差校正：用106维Schur系统（alpha联合107维）求精确牛顿方向。
   代价O(qr³+q²r²+q³)，与当前不动点的迭代次数比较后才谈实际收益。
4. 尚未证明一般ragged/nonuniform-rescaling条件下，每个完整决策日统一
   从O(P³)降到O(P²)。alpha/logdet、数据搬运秩和分支等价仍是必要工作。

下一步应把“构造支撑面+精确Schur校正”实现成dev中的候选器，先对最慢
第三折作原证书、原选解和heldout隔离验证；在没有这些证据前，不接入生产。
增量因子/对数行列式是第二条并行的数学工作，不用一个warm-start开关代替。

## 6. 已执行检查与研究边界

`identities.jl`提供矩阵恒等式、固定alpha的负evidence梯度/Hessian和Schur
方向原型；联合alpha原型只验证块消元，未提供生产混合导数。
`checks.jl`以小矩阵核对原生产梯度、有限差分Hessian、直接稠密Newton
方程、局部平方误差阶、mask搬运、fold变化、错误floor裁剪反例，以及
结构支撑的完整evidence/非零null均值。没有重新拟合或跑回测。
`local_structure.jl`只对两个认证日期作prepare/谱/已存后验结构检查。

原命题是精确实数代数；浮点实现仍须保持结构零项与参考误差门槛。图支撑
不能用小特征值替代；零方向被解析积分/保留，不是删除系数先验。
数值检查与日志只是实现查错，不增加Final的原测试计数、不构成测速或发布。

### 已取得的真实支撑核验

`support_face_audit.jl`只重算2026-09-10第三折的准备/谱/收缩，不拟合后验。
mask图给出r=45、a=19，故数据谱维数C r=630，而原为C N=910。
该结果来自观察图，不来自协方差的“小特征值”判断。

| 检查 | 实际值 |
|---|---:|
| 训练特征对构造支撑的相对残差 | 3.5555445214e-13 |
| 训练目标对构造支撑的相对残差 | 3.8825469034e-16 |
| 已存Sigma的跨支撑块相对残差 | 4.5503192738e-16 |
| 未识别块与原delta P_A的绝对残差 | 3.3374716550e-16 |
| 在45维支撑中，白化H0最小系数除以n | 0.4956486950 |
| 未识别方向原evidence梯度gamma | −4.6789999989e11 |
| 原条件均值未识别块的最大范数 | 2.2625971184e-9 |
| 上述均值与解析公式的最大绝对差 | 1.7392963309e-17 |

因此这个已存第三折确实落在命题7的候选面上，gamma有明确负裕量；
在完整64维空间白化H0最小系数约−0.5n，而移除解析保留的null块后为
正0.49565n。这里只说明基算子恢复稳定正性，没有计算或宣称整个联合
Hessian强凸，更没有运行一个新的后验求解器。

数据谱稠密分解的尺寸立方比(630/910)^3约0.332，协方差矩阵立方比
(45/64)^3约0.348。这是相应kernel的尺寸推算，不是已测端到端加速。
额外280个输入方向的先验、19个输出方向的floor噪声及trace修正仍解析保留。
此单折检查通过8项断言，命令RC0、10秒、峰值1104MiB；完整输出见
`dev/evidence/theory_incremental_20261009/support_face.log`。

一个扩大的结构扫描曾超时（support_structure.log，RC124、35秒），
原失败保留；它不是该8项支撑核验的通过来源。已完成的小矩阵恒等式脚本
验证了710项，其中多数是不同mask/维数的重复代数查错，不当作710个
独立定理或新增Final数值验收。其原生产梯度最大相对差约8.18e-16，
有限差分Hessian误差约1.32e-9，Schur与直接Newton解的归一差约2.57e-18。

## 参考来源与贡献边界

低秩因子更新采用既有工具，不声称重新发明：Davis & Hager,
Multiple-Rank Modifications of a Sparse Cholesky Factorization,
SIAM J. Matrix Anal. Appl. 22 (2001), DOI10.1137/S0895479899357346。
官方出版页：https://epubs.siam.org/doi/10.1137/S0895479899357346

预测—校正的一般背景：Simonetto et al., A Class of Prediction-Correction
Methods for Time-Varying Convex Optimization，https://arxiv.org/abs/1509.05196；
Simonetto & Dall'Anese, Prediction-Correction Algorithms for Time-Varying
Constrained Optimization，https://arxiv.org/abs/1611.03681。
它们的凸性/正则性假设不能不加验证地套到本项目EB。

本文针对本项目推导的是：mask交换子及ridge缺陷的完整分解、原conditioned
evidence的106/107维Schur结构、以及mask图支撑下保留全部先验的floor面
消元与KKT提升。没有宣称这些组合在文献中从未出现，亦未把推导当成已
完成的端到端增量算法。
