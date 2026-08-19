# DriftTTT：两个核心算法及计算过程

## 1. 先用两句话说明方法

**TTT-MoE**：把一个 TTT 快速 MLP 的容量分给短期、长期两个专家；短期专家看逐点变化，长期专家看分段均值，最后由门控按位置融合。

**CB-DTS**：训练时不再只监督窗口最后一个点，而是由已知端点标签恢复窗口内的目标轨迹；先在每个 cycle 内平均误差，再让各 cycle 等权参与损失。

完整方法是 **Transformer 主干 + TTT-MoE + CB-DTS**。前者改变序列混合层，后者只改变训练损失。CB-DTS 在验证和测试时关闭，因此不会增加部署参数量和推理计算量。

## 2. 从一个 batch 开始

### 2.1 输入变量与维度

设一个 batch 含有 \(B\) 个滑动窗口，每个窗口最多包含 \(L\) 个时间点，输入特征数为 \(F\)。模型接收：

| 符号 | 代码中的含义 | 维度 |
|---|---|---:|
| \(X\) | 运行条件和传感器特征 `features` | \([B,L,F]\) |
| \(A\) | 有效位置掩码 `mask`，有效为 1 | \([B,L]\) |
| \(C\) | 每个位置的 flight cycle ID `cycle_ids` | \([B,L]\) |
| \(y\) | 每个窗口末端的归一化 RUL 标签 `target` | \([B]\) |

以项目常用设置为例：\(B=64\)、\(L=512\)、不输入 cycle 特征时 \(F=18\)。模型维度 \(D=128\)，注意力头数 \(H=4\)，所以每个头的维度 \(d=D/H=32\)。表中的字母表示一般情况，算法不依赖 batch 必须等于 64。

这里要区分两件事：cycle ID 可以不拼入 \(X\)，但仍保留在 \(C\) 中作为元数据。TTT-MoE 用它避免长期分段跨越 flight cycle；CB-DTS 用它恢复训练轨迹。模型不会把 cycle ID 当作连续数值特征直接预测 RUL。

### 2.2 主干的输入和输出

输入先经过线性投影和位置编码：

\[
Z^{(0)}=\operatorname{Linear}(X)+\operatorname{PE},
\qquad Z^{(0)}\in\mathbb{R}^{B\times L\times D}.
\]

随后通过 4 个 Transformer Block。每个 Block 中的序列混合器可以替换为 Attention、标准 TTT 或 TTT-MoE，其余前馈网络和残差结构保持一致。最后，共享回归头对每个位置都能产生一个标量：

\[
\hat Y=\operatorname{Head}(Z),
\qquad \hat Y\in\mathbb{R}^{B\times L}.
\]

设第 \(b\) 个窗口的最后一个有效位置为 \(\tau_b\)，实际用于 RUL 评估的端点预测是 \(\hat y_b=\hat Y_{b,\tau_b}\)，所以端点预测维度为 \([B]\)。普通训练只使用这 \(B\) 个数；CB-DTS 训练还会使用整张 \([B,L]\) 的预测。

## 3. 创新一：TTT-MoE 的逐步计算

### 3.1 它解决什么问题

工业时序同时包含快速工况波动和缓慢状态变化。标准 TTT 用同一个快速 MLP、同一种更新频率处理全部变化。TTT-MoE 不再要求一个专家同时兼顾两种尺度，而是在基本不扩大快速状态预算的前提下将它分为短期和长期两部分。

### 3.2 第一步：生成每个头的 Q、K、V

对某一 Transformer Block 的输入 \(Z\in\mathbb{R}^{B\times L\times D}\) 做一次线性投影并拆成 \(H\) 个头：

\[
Q,K,V=\operatorname{split}(W_{qkv}Z),
\qquad Q,K,V\in\mathbb{R}^{B\times L\times H\times d}.
\]

在当前配置中，维度是 \([64,512,4,32]\)。\(Q\) 和 \(K\) 会按最后一维进行归一化。

### 3.3 第二步：把序列拆成“长期均值”和“短期偏差”

长期分支把相邻的 \(P\) 个有效时间点组成一个 segment，当前 \(P=64\)。遇到新的 flight cycle 时，segment 立即重新开始。设第 \(b\) 个样本最终得到 \(J_b\) 个 segment，第 \(j\) 个 segment 的位置集合为 \(G_{b,j}\)。以 \(Q\) 为例，长期表示就是段内均值：

\[
\bar Q_{b,j}=\frac{1}{|G_{b,j}|}\sum_{t\in G_{b,j}}Q_{b,t}.
\]

\(\bar K\) 和 \(\bar V\) 同样计算，形状都是 \([B,J,H,d]\)，其中 \(J=\max_b J_b\)。如果一个 512 点窗口没有额外 cycle 边界，它会得到 \(512/64=8\) 个 segment。

再把每个 segment 均值广播回所属时间点，记为 \(\bar Q^{\uparrow}\)。短期分支使用原值减去段均值：

\[
Q_s=Q-\bar Q^{\uparrow},\quad
K_s=K-\bar K^{\uparrow},\quad
V_s=V-\bar V^{\uparrow}.
\]

因此，长期分支处理 \((\bar Q,\bar K,\bar V)\)，关注段与段之间的慢变化；短期分支处理 \((Q_s,K_s,V_s)\)，关注段内的快速偏差。

### 3.4 第三步：两个专家各自做一次无标签 TTT 更新

每个专家都是一个两层快速 MLP：

\[
f_\theta(U)=W_2\,\sigma(W_1U+b_1)+b_2.
\]

对每个头，输入和输出维度均为 \(d\)。短期专家的隐藏维度为 \(M_s\)，所以 \(W_1^s\in\mathbb{R}^{d\times M_s}\)、\(W_2^s\in\mathbb{R}^{M_s\times d}\)；长期专家只需将 \(M_s\) 换成 \(M_l\)。代码使用同坐标重构任务：输入是 \(K\)，目标是 \(V-K\)。单个专家的内部损失可简写为：

\[
L_{\mathrm{TTT}}(\theta)
=\operatorname{mean}\left\|f_\theta(K)-(V-K)\right\|^2.
\]

然后在当前样本上做内部梯度更新：

\[
\theta' = \theta-\alpha\nabla_\theta L_{\mathrm{TTT}}.
\]

更新后，用同一专家处理 query 并得到自适应残差 \(r=f_{\theta'}(Q)\)。短、长期专家分别代入自己的数据：

\[
r_s=f_{\theta'_s}(Q_s),\qquad
\bar r_l=f_{\theta'_l}(\bar Q).
\]

短期专家每 16 个原始时间点更新一次，内部学习率为 0.1；长期专家每 4 个 segment 更新一次，内部学习率为 0.025。长期输出 \(\bar r_l\in\mathbb{R}^{B\times J\times H\times d}\) 会广播为 \(r_l\in\mathbb{R}^{B\times L\times H\times d}\)。这些内部更新只依赖当前输入的 \(K,V\)，不使用 RUL 标签。

快速 MLP 的总隐藏维度为 \(M=2d=64\)。当前按约 70%/30% 划分为：

\[
M_s=45,\qquad M_l=19,
\]

而不是创建两个各有 64 维的完整专家。这就是“固定快速状态预算”的含义。

### 3.5 第四步：门控融合两个专家

模型为每个样本、时间点和头计算一个短期权重：

\[
g=\operatorname{sigmoid}\left(\frac{Q\cdot w_g}{\sqrt d}+b_g\right),
\qquad g\in\mathbb{R}^{B\times L\times H}.
\]

设短期容量占比 \(p=M_s/M\)，\(\operatorname{Center}(\cdot)\) 表示在有效时间点上减去均值。代码中的实际融合为：

\[
R=\frac{g}{p}r_s+
\operatorname{Center}\left(\frac{1-g}{1-p}r_l\right),
\qquad H_{\mathrm{mix}}=Q+R.
\]

中心化使长期专家主要表达“相对慢变化”，避免整体抬高或压低健康基线。除以 \(p\) 和 \(1-p\) 是为了补偿两个专家的容量比例，使门控初始化时不会无意改变残差尺度。

如果窗口中不足两个长期 segment，长期分支没有可比较的慢变化，代码会关闭其贡献并只保留短期分支。最后将 \([B,L,H,d]\) 合并回 \([B,L,D]\)，再做输出投影。

### 3.6 一个 batch 经过 TTT-MoE 后发生了什么

对 \([64,512,128]\) 的 Block 输入，计算路径可以直接概括为：

```text
[64, 512, 128]
  → Q/K/V 各为 [64, 512, 4, 32]
  → 长期分段均值：约 [64, 8, 4, 32]
  → 短期专家在 512 点上更新，长期专家在约 8 个 segment 上更新
  → 两个输出都恢复到 [64, 512, 4, 32]
  → 门控逐位置融合
  → 合并并投影为 [64, 512, 128]
```

训练时，外层反向传播仍会学习 Q/K/V 投影、快速 MLP 的初始参数、门控和整个预测网络；测试时，每个样本从已学到的初始快速状态出发，仅利用自己的输入完成 TTT 内部更新，不需要 RUL 标签。

## 4. 创新二：CB-DTS 的逐步计算

CB-DTS 全称为 **Cycle-Balanced Dense Trajectory Supervision**，即按 cycle 均衡的稠密轨迹监督。它不改变 TTT-MoE 的前向结构，只改变训练时如何使用标签。

### 4.1 第一步：计算普通端点损失

一个 batch 有端点预测 \(\hat y\in\mathbb{R}^{B}\) 和端点标签 \(y\in\mathbb{R}^{B}\)。普通端点 MSE 为：

\[
L_{\mathrm{end}}=\frac{1}{B}\sum_{b=1}^{B}(\hat y_b-y_b)^2.
\]

这只监督了每个窗口最后一个有效位置，batch 中虽然有 \(B\times L\) 个时序表示，却只有 \(B\) 个直接任务监督信号。

### 4.2 第二步：由端点标签恢复窗口内标签

N-CMAPSS 的 RUL 以剩余 flight cycle 数计量。设第 \(b\) 个窗口末端的 cycle 为 \(c_b^{\mathrm{end}}\)，位置 \(t\) 的 cycle 为 \(C_{b,t}\)，RUL 截断上限为 \(R=125\)。已知归一化端点标签 \(y_b\) 后，该位置的归一化标签可以直接恢复为：

\[
Y_{b,t}=\min\left(1,\ y_b+\frac{c_b^{\mathrm{end}}-C_{b,t}}{R}\right).
\]

恢复后的稠密标签 \(Y\) 维度为 \([B,L]\)，与共享回归头输出的 \(\hat Y\in\mathbb{R}^{B\times L}\) 一一对应。这里没有额外模型，也没有生成伪标签；它只利用训练集已有端点标签和 cycle 间的确定关系。

一个最简单的数值例子：若窗口末端归一化 RUL 为 0.20，末端 cycle 为 80，较早位置属于 cycle 75，则：

\[
Y=\min(1,\ 0.20+(80-75)/125)=0.24.
\]

同一个 flight cycle 内的所有采样点拥有相同的 RUL 标签。

### 4.3 第三步：先在 cycle 内平均

不同 flight cycle 的采样点数量可能相差很大。若直接平均全部时间点，点数多的 cycle 会自然获得更大权重。设样本 \(b\) 中属于 cycle \(c\) 的有效位置集合为 \(S_{b,c}\)，先计算该 cycle 的平均误差：

\[
E_{b,c}=\frac{1}{|S_{b,c}|}
\sum_{t\in S_{b,c}}(\hat Y_{b,t}-Y_{b,t})^2.
\]

例如一个窗口含两个 cycle，分别有 100 和 20 个采样点。普通逐点 MSE 的权重是 5:1；经过这一步后，先各自得到一个 \(E_{b,c}\)，长 cycle 不会仅因采样点多而支配训练。

### 4.4 第四步：再对 cycle、样本和 batch 平均

设样本 \(b\) 中出现的 cycle 集合为 \(\mathcal C_b\)，该样本的轨迹损失为：

\[
L_{\mathrm{traj}}^{(b)}=
\frac{1}{|\mathcal C_b|}\sum_{c\in\mathcal C_b}E_{b,c}.
\]

最后对 batch 中的样本等权平均：

\[
L_{\mathrm{CB}}=\frac{1}{B}\sum_{b=1}^{B}L_{\mathrm{traj}}^{(b)}.
\]

因此，代码中的聚合顺序非常明确：**时间点误差 → cycle 内平均 → cycle 间平均 → batch 样本平均**。

### 4.5 第五步：与端点任务合并

最终训练目标只有两个部分：

\[
L=\frac{L_{\mathrm{end}}+\lambda L_{\mathrm{CB}}}{1+\lambda},
\qquad \lambda=0.5.
\]

分母 \(1+\lambda\) 让总损失的量级较稳定。\(\lambda=0.5\) 表示端点任务仍是主要目标，稠密轨迹只作为辅助监督，不替代最终端点预测。

### 4.6 训练、验证和测试的区别

| 阶段 | 模型输出的使用方式 | 是否计算 CB-DTS | 是否把真实 RUL 输入模型 |
|---|---|---|---|
| 训练 | 使用整段 \([B,L]\) 预测和端点 \([B]\) 预测 | 是 | 否，标签只用于算损失 |
| 验证 | 只使用端点 \([B]\) 预测选择检查点 | 否 | 否 |
| 测试 | 只使用端点 \([B]\) 预测计算指标 | 否 | 否 |

所以，CB-DTS 是一种**训练阶段的任务对齐损失**：它让整段 TTT 表征都接收到与实际预测目标一致的梯度。它不是测试时读取标签的在线学习，也不是标准的元学习算法。

### 4.7 可推广的部分是什么

CB-DTS 在本项目中的具体实现使用 RUL 与 cycle 的关系，但它的核心并不限定为 RUL：只要某个工业任务能根据端点标签和已知过程规律恢复中间目标，就可以写成

\[
Y_{b,t}=\Phi\bigl(y_b,\ s_b^{\mathrm{end}}-s_{b,t}\bigr),
\]

其中 \(s\) 是阶段、里程、循环次数或时间等过程坐标，\(\Phi\) 是该任务已知的目标演化规则。之后仍按“组内平均、组间平均”计算辅助损失。N-CMAPSS 中取 \(s=C\)，\(\Phi(y,\Delta c)=\min(1,y+\Delta c/R)\)，分组就是 flight cycle。如果中间目标不能由可靠规律确定，则不应强行使用该稠密监督。

## 5. 两项创新如何配合

在一个训练 batch 中，两种“学习”发生在不同层次：

1. **TTT 内部更新**：每个样本在前向计算时，用无标签重构任务更新短、长期快速 MLP，产生适应该样本的时序表示。
2. **模型外部训练**：用端点损失和 CB-DTS 对整个网络正常反向传播，学习一个更适合实际 RUL 目标的初始化、门控和回归头。RUL 标签只出现在这一层，不进入 TTT 内部重构公式。

可以将完整训练过程写成下面的简化伪代码：

```text
输入一个 batch：X [B,L,F]、mask [B,L]、cycle [B,L]、端点标签 y [B]

for 每个 Transformer Block:
    Q, K, V = 投影并拆成多头
    短期数据 = 原始 Q/K/V - 所属 segment 均值
    长期数据 = 每个 segment 的 Q/K/V 均值
    用 (K, V-K) 分别更新短期和长期快速 MLP
    用更新后的 MLP 处理 Q，得到两个残差
    用门控融合两个残差

hat_Y = 共享回归头输出整段预测 [B,L]
hat_y = 取每个样本最后一个有效位置 [B]
L_end = 端点 MSE
Y_dense = 由 y 和 cycle 恢复的训练轨迹 [B,L]
L_CB = 按“点 → cycle → 样本 → batch”计算的轨迹 MSE
L = (L_end + 0.5 * L_CB) / 1.5
对 L 反向传播并更新模型参数
```

两项创新的边界也很清楚：TTT-MoE 负责“怎样从输入中提取多尺度、自适应表示”，CB-DTS 负责“训练时怎样让这些表示对齐实际任务”。没有额外预训练阶段，也没有额外教师模型。
