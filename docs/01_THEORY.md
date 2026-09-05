# 01｜理论规格：有限步路由事件，而不是必然缺失的经验梯度

状态：设计规格与条件性推导。基础恒等式不是原创定理；算法重要性、新颖性和真实模型收益待检验。历史数值见 legacy，不作为这里任何泛化保证。

## 1. 唯一主问题

给定当前可部署量化矩阵 Q0 与同格式合法候选集合 C（含 Q0），能否比固定分支局部模型更准确地选出使参考损失下降的候选，并且额外计算值得？

量化格点固定时 Q=sC_int，C_int 是受位宽约束的整数码。连续候选坐标只用于求导，不是部署参数；每个最终端点必须能被原格式表示。一个候选同时作用于所有 token。

## 2. 模块、形状与参数依赖

以列向量为数学约定：a_n 属于 R^m，b_n 与 h_n 属于 R^d，Q 属于 R^(d×m)。

h_n(Q)=b_n+Qa_n。

N(h)=D_gamma h/rho(h)，rho(h)=sqrt(||h||²/d+epsilon)>0。

无偏置线性 router R 属于 R^(E×d)，r(h)=RN(h)，S(h)=TopK(r(h))，|S|=k。

定义指定集合的完整分支：

f_S(h)=h+sum_{i in S} alpha_{i,S}(h) E_i(N(h))。

真实模块 F(h)=f_{S(h)}(h)。alpha 必须使用真实实现：既支持全 E 专家 softmax 后直接取 top-k，也支持取 top-k 后重新归一化。强制集合只是固定 indices，不固定候选终点上的 gate 数值。

参考 y*_n 固定。损失 L(Q)=mean_n [(F(h_n(Q))-y*_n)^T M(F(h_n(Q))-y*_n)]，M 半正定。v0 取 M=I/d，明确总损失已经对 token 和隐藏维度平均。理论公式使用任意 M；不同实现的 sum/mean 不得混用。

仅改变本层 expert 权重不会改变同一层先发生的 routing；它可能改变下游 routing。本层理论起点可以是 W_O -> residual -> RMSNorm -> router。扩展到专家权重后若中间有 attention，就不能沿用 h(Q)=b+Qa 的精确线性成本。

## 3. 经验目标与总体目标不能混淆

F_theta(x)=1[x>theta]，参考 1[x>0]。连续 x~U[-1,1] 时 L_pop(theta)=|theta|/2；theta>0 的总体导数是 1/2。

对有限固定样本 {x_n}，L_N(theta) 在 theta 不越过样本的位置之间局部常数，导数为 0。这并非“自动微分错了”。对一般 finite calibration，远离全部路由边界时，固定分支导数可等于真实经验导数。

量化关心的是有限 Q0 -> Q1，局部导数正确也不能保证跨分支的有限步排序正确。不要把总体分布的边界积分机械加到经验梯度上。

## 4. 当前分支的局部信息为什么不够

固定 S 的 Jacobian 含 norm、active experts、gate 连续变化和 residual：

J_N(h)=D_gamma[I/rho - hh^T/(d rho³)]。

J_fS(h)=I+sum_i [alpha_i J_Ei(N(h)) J_N(h)+E_i(N(h)) (grad_h alpha_i)^T]。

它不包含未执行分支的任意功能值。构造两个系统：当前分支均输出 1，target=0，router/margin 与当前分支所有局部导数相同；未激活分支分别输出 0 和 2。切换损失分别为 -1 和 +3。

因此，仅观察当前分支局部导数与路由信息，无法普遍确定切换损失符号。这是受限观察接口的必要信息反例，不是拥有完整模型的全局不可能性。替代分支的功能信息必须来自实际计算、经验证代理或额外假设。

## 5. 精确路由几何的适用范围

由于 rho(h)>0 且 router 无偏置：TopK(RN(h))=TopK(RD_gamma h)。

设 C_E=I-11^T/E，A=C_E R D_gamma。给所有 logits 加相同常数不改排序，所以 S(h) 只依赖 Ah，rank(A)<=E-1。

若 A delta=0，则 S(h+delta)=S(h)（采用相同 tie 规则）。但不能推出功能不变：专家输入改变，而且 1/rho 可能改变 gate 温度。需要负对照证明这个区别。

令 w_i 为 RD_gamma 的第 i 行。当前集合 S 的区域由 (w_i-w_j)^T h>0 对所有 i in S,j not in S 定义，是凸多面锥。

沿 h(t)=h0+t delta，m_ij=(w_i-w_j)^T h0，v_ij=(w_i-w_j)^T delta。严格无 tie 时：

t_exit=min_{i in S,j not in S,v_ij<0} m_ij/(-v_ij)，空集合取 infinity。

t_exit>1 表示此有限步不退出当前路由区域。不能只查当前第 k 与 k+1 专家；更靠后的专家可能有更大的上升速度。无需显式构造 d×d 零空间矩阵来判断路由；直接更新 E 维 logits 即可。

边界 t=1、接近 tie、浮点排序不一致单独记录，不能默默删样本。toy 规定 stable descending sort、相同分数优先较小 expert id；真实适配器复制原实现的行为，并将 tie 不可重复性作为数值现象。

router bias、grouped top-k、capacity drops、stochastic routing 或依赖 batch 的调度不在该精确结论里。基础终点分解可重新定义在完整确定性模块上，但 token 局部性和事件稀疏性不再自动成立。

## 6. 有限步精确分解

h0_n=h_n(Q0)，h1_n=h_n(Q1)，S0_n=S(h0_n)，S1_n=S(h1_n)。

Delta L=mean_n [ell(f_S1(h1),y*)-ell(f_S0(h0),y*)]。

加减在新输入 h1 上强制旧集合的损失：

Delta L_fixed=mean_n [ell(f_S0(h1),y*)-ell(f_S0(h0),y*)]。

J_event=mean_n [ell(f_S1(h1),y*)-ell(f_S0(h1),y*)]。

于是 Delta L=Delta L_fixed+J_event，逐 token 和聚合均严格成立。不需要连续密度、高斯噪声、核带宽或只跨一次边界。

若 S1_n=S0_n，则该 token 的事件项为 0。注意集合相同但 gate 权重改变仍属于 fixed 部分。

令 residual e=f_S0(h1)-y*，jump d=f_S1(h1)-f_S0(h1)，则：

J_n=2 e^T M d+d^T M d。

交叉项不能删。e=1,d=-0.8,M=1 时 J=-0.96。事件不是非负惩罚，也不要求恢复全精度专家身份。

f_S0(h1)、f_S1(h1) 必须重算候选输入处的完整聚合；仅比较旧输入处两个裸 expert 输出不是此恒等式。

## 7. 离散候选、局部导数与误差

Q(z)=Q0+sum_{j=1}^p z_j D_j，z_j in {0,1}。D_j 是合法且组合后仍合法的码修改；不重叠坐标是最简单的保证。z=0 必须保留。v0 选取最接近舍入中点的 p 个坐标，不能查看损失后挑坐标。

用连续扩展定义固定 S0 的 L_fixed(z)。令 g=grad L_fixed(0)，H=Hessian L_fixed(0)。g 通常不为 0。

平方损失完整 Hessian：H=2 mean_n [J_n^T M J_n+sum_a (M e_n)_a Hessian_z f_{n,a}]。

Gauss-Newton 只取第一项；当前残差非零时二者一般不同。H 可以不半正定。禁止将 PSD 化、damping、对角近似或 GN 偷换成 exact H。

若固定分支 Hessian 在全部候选线段上 beta-Lipschitz：

|Delta L_fixed(z)-g^T z-0.5 z^T H z|<=beta ||z||³/6。

证明：沿 tz 积分 Hessian，余项上界 beta ||z||³ integral_0^1 t(1-t)dt。前提包含分支延拓的光滑性，例如 toy 使用 SiLU；跨 ReLU kink 时需修改结论。

评分 s(z)=g^T z+0.5 z^T H z+J_event(z)。精确事件校正后，上界仍为 beta ||z||³/6。不是已知数值证书；真实 beta 尚未估计。

使用近似 g_hat、H_hat、J_hat 时还要加 ||g_hat-g||||z||+0.5||H_hat-H||op||z||²+|J_hat-J|。候选步大、上游非线性或数值近 tie 时不能声称上述局部误差必小。

## 8. 候选选择遗憾与可获得空间

C 包含 Q0。定义机会 V=L(Q0)-min_C L(Q)>=0。评分方法选中 Q_m，其 regret R_m=L(Q_m)-min_C L(Q)>=0。

若 |s(Q)-Delta L(Q)|<=epsilon 对所有候选成立，则 argmin_s 的 R<=2epsilon：真实损失 <= 自己评分+epsilon <= oracle 评分+epsilon <= oracle真实损失+2epsilon。

这是基础近似优化结论，不是独立创新。实验首先看 V 与 R；相关系数只是辅助。V 很小时 normalized regret 不稳定，应报告 null 并给出原始分母，而不是强行除零。

校准 oracle 与留出 oracle 分开：方法只能在 calibration 选择一次 Q_m；holdout 只评估，不重新选。可报告相对 holdout hindsight oracle 的 regret，但必须标成诊断上限，不能称部署方法。

## 9. Shared-weight 可达性

两个 token margin m1(u)=0.1+u，m2(u)=0.2+u。使 token2 翻转需要 u<-0.2，此时 token1 也翻转。“只翻第二个”不可达。

真实路由序列由同一个 z 通过 A h_n(Q(z)) 决定。逐 token 任意选 expert 的 oracle 不是可部署量化收益；不得作为方法成绩。

## 10. 计算与推广边界

事件实现对每个事件 token 计算 S0 union S1，至多 2k 个专家；全 E logits 的 softmax 仍保留。统计实际 token-expert 调用和 wall time，不把随机 toy 的 dense oracle 当稀疏实现。

非线性上游时需要计算真实 h1 或另有误差界；不能沿用线性 W_O 场景的成本。多层事件不能在无假设下相加；下游状态改变、attention 跨 token 耦合和门控相互作用都需要重新测量。

局部 MSE 改善不保证全模型 NLL/准确率。fake quantization 不保证 packed low-bit 内核的数值或速度。以上边界都是后续关卡，而非计划中自动解决的事项。
