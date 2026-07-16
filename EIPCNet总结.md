# EIPCNet 论文总结与可借鉴内容

## 1. 论文信息

- 论文题目：Enhanced Edge Information and Prototype Constrained Clustering for SAR Change Detection
- 作者：Bin Cui、Yao Peng、Yonghong Zhang、Hujun Yin、Hong Fang、Shanchuan Guo、Peijun Du
- 期刊：IEEE Transactions on Geoscience and Remote Sensing（TGRS）
- 年份：2024
- DOI：[10.1109/TGRS.2024.3367970](https://doi.org/10.1109/TGRS.2024.3367970)
- 方法简称：EIPCNet
- 本地论文：`D:\lunwen\single-sar\Enhanced_Edge_Information_and_Prototype_Constrained_Clustering_for_SAR_Change_Detection.pdf`

## 2. 论文核心思想

EIPCNet 是一种无监督 SAR 变化检测框架，主要解决以下问题：

1. SAR 斑点噪声会干扰双时相差异信息。
2. 变化区域边缘容易模糊，导致漏检和误检。
3. 无监督聚类生成的伪标签存在类别不平衡。
4. 伪标签中不可避免地包含错误标签。

完整流程为：

```text
双时相 SAR 影像
    ↓
GNR 梯度邻域比率差异图
    ↓
PCH-FLICM 原型约束层次聚类
    ↓
筛选高置信变化与未变化伪标签
    ↓
构造双时相双通道 Patch
    ↓
训练 CBNTNet
    ↓
输出完整变化检测图
```

该方法并不是将多个数据集合并训练一个通用模型，而是针对每一组双时相 SAR 场景分别生成伪标签、训练网络并预测。Ground Truth 只用于最终指标计算，不参与网络训练。

## 3. 论文使用的数据集

| 数据集 | 传感器与时间 | 图像尺寸 | 主要变化 | 最优 Patch |
|---|---|---:|---|---:|
| Daxing-I | Gaofen-3；2017 年 4 月、2018 年 5 月 | 400×400 | 建筑建设与拆除，变化区域较少 | 5×5 |
| Daxing-II | Gaofen-3；2017 年 4 月、2018 年 5 月 | 215×248 | 城市建设、裸地与人工建筑转换 | 5×5 |
| Coast | Radarsat-2；2008 年 6 月、2009 年 6 月 | 280×450 | 黄河三角洲海岸和水陆边界变化 | 7×7 |
| Yellow River | Radarsat-2；2008 年 6 月、2009 年 6 月 | 444×291 | 水体及岸边变化 | 5×5 |
| Ottawa | Radarsat-1；1997 年 5 月、8 月 | 350×290 | 洪水消退引起的后向散射变化 | 5×5 |

当前已有数据可以对应论文中的三组：

```text
Coast：img1.bmp、img2.bmp、img_rf.bmp
Yellow River：Yellow River1.bmp、Yellow River2.bmp、Yellow River_rf.bmp
Ottawa：1997.05.bmp、1997.08.bmp、reference1.bmp
```

目前缺少 Daxing-I 和 Daxing-II，因此可以先复现论文五组实验中的三组。

## 4. 可以直接借鉴的内容

| 论文方法 | 解决的问题 | 在 ScratchFormer-SAR 中的借鉴方式 |
|---|---|---|
| GNR 差异图 | 斑点噪声和变化边缘模糊 | 作为辅助差异分支或 MoE 路由先验 |
| PCH-FLICM 伪标签 | 缺少大量人工标注 | 用于无监督预训练或半监督训练 |
| 高置信样本筛选 | 聚类伪标签含有错误 | 只使用高置信变化和未变化区域训练 |
| 类别平衡 Focal Loss | 变化像素远少于未变化像素 | 替换普通交叉熵或与 Dice 联合 |
| MAE 抗噪损失 | 伪标签存在噪声 | 降低错误标签导致的过拟合 |
| 双时相 Patch 输入 | 完整场景数量少 | 从完整影像构造局部训练样本 |
| Patch 尺寸分析 | 感受野与噪声抑制存在冲突 | 对 ScratchFormer 比较 128 和 256 输入 |
| 噪声与边缘消融 | 证明方法具有 SAR 针对性 | 增加斑点噪声鲁棒性和边界精度实验 |

## 5. GNR 差异图的借鉴方式

### 5.1 GNR 的作用

普通像素差或对数比率容易受到斑点噪声影响。GNR 同时使用双时相强度、局部邻域和梯度信息，在抑制孤立噪声的同时保留变化区域边界。

对于当前研究，GNR 不建议直接替代 ScratchFormer 的深层特征，而应作为辅助先验。

### 5.2 作为辅助输入

可以将输入从单纯的双时相影像扩展为：

```text
输入 1：前时相单通道 SAR
输入 2：后时相单通道 SAR
辅助输入：GNR 差异图
```

GNR 经过轻量卷积编码后，与 CEFF 输出或解码器特征进行融合。

### 5.3 作为 MoE 路由先验

GNR 更适合用于控制专家选择。建议路由器同时使用深层差异特征和 GNR：

```text
D_feature = abs(F_pre - F_post)

RouterInput = concat(
    GAP(D_feature),
    GMP(D_feature),
    GAP(GNR_feature)
)

RoutingWeight = TopK-Softmax(RouterInput)
```

其中：

- GAP 表示全局平均池化。
- GMP 表示全局最大池化。
- GNR 提供显式边缘和差异强度先验。
- TopK-Softmax 只激活少量专家，降低计算量和过拟合。

## 6. 推荐的 GNR 引导 SAR-SSMoE ScratchFormer

建议整体结构为：

```text
前时相 SAR ──→ 共享 ScratchFormer 编码器 ──┐
                                              ├→ CEFF → SAR-SSMoE → 解码器 → 变化图
后时相 SAR ──→ 共享 ScratchFormer 编码器 ──┘              ↑
                                                             │
前后时相 ──→ GNR 差异图 ──→ 轻量特征编码 ──→ MoE 路由先验 ──┘
```

建议将新模块命名为：

```text
GNR-guided SAR-SSMoE
Gradient Neighborhood Ratio-guided
SAR Speckle-aware Shared Sparse Mixture of Experts
```

### 6.1 专家设计

| 专家 | 推荐结构 | 主要作用 |
|---|---|---|
| 局部纹理专家 | 深度可分离 3×3 卷积 | 检测小范围变化和局部纹理 |
| 去斑专家 | 深度可分离 5×5 卷积 | 建模较大邻域并抑制斑点干扰 |
| 上下文专家 | 空洞 3×3 卷积 | 检测大范围结构变化 |
| 可选边缘专家 | Sobel/高通特征加轻量卷积 | 强化变化边界 |
| 共享分支 | Identity 或 1×1 卷积 | 保留通用变化信息 |

### 6.2 推荐放置位置

按以下顺序逐步实验：

1. 只在最终多尺度融合特征后加入一个 MoE。
2. 在四个 CEFF 输出后使用跨尺度共享 MoE。
3. 在 CEFF 和最终融合位置同时使用共享 MoE。
4. 输入端 MoE 仅作为消融，不作为默认结构。

不建议一开始同时使用输入端 MoE、四个尺度 MoE 和最终融合 MoE，因为当前数据量很小，容易造成过拟合。

## 7. PCH-FLICM 伪标签的借鉴方式

### 7.1 伪标签分类

可以根据 GNR 差异图和聚类结果生成三类训练标签：

```text
高置信变化像素     → 标签 1
高置信未变化像素   → 标签 0
中间或不确定像素   → ignore_index=255
```

不确定边界不参与初始训练，可以减少错误伪标签对模型的干扰。

### 7.2 推荐训练流程

```text
训练域双时相 SAR
    ↓
计算 GNR 差异图
    ↓
聚类并筛选高置信伪标签
    ↓
使用伪标签预训练 ScratchFormer-SAR
    ↓
使用真实标签微调
```

该流程可以缓解真实标注和训练场景数量不足的问题。

### 7.3 数据泄漏限制

在跨数据域实验中，伪标签只能由训练域影像生成。测试域影像不能参与伪标签预训练，否则主实验将从严格跨域设置变成传导式设置。

如果确实使用测试域无标签影像，必须单独标记为：

```text
Transductive Setting
```

不能与严格跨域结果放在同一设置下比较。

## 8. 类别平衡与抗噪损失

### 8.1 监督基线损失

真实标签训练时推荐：

```text
L_seg = L_weighted_CE + L_Dice
```

Weighted CE 处理类别不平衡，Dice 直接优化变化区域重叠。

### 8.2 边缘损失

借鉴论文强化边缘的思想，可以由 Ground Truth 生成边缘图，并增加：

```text
L_edge = BCE(PredictedEdge, GroundTruthEdge)
```

监督模型总损失可以写为：

```text
L = L_weighted_CE + L_Dice + lambda_edge * L_edge
```

### 8.3 伪标签抗噪损失

使用伪标签时，可借鉴论文将类别平衡 Focal Loss 与 MAE 结合：

```text
L_pseudo = lambda_focal * L_CBAFL
         + lambda_mae * L_MAE
```

论文使用：

```text
lambda_focal = 0.7
lambda_mae = 0.3
gamma = 2
```

这些参数可以作为初始值，但最终必须根据训练域验证集选择。

### 8.4 完整 MoE 损失

建议最终模型使用：

```text
L_total = L_weighted_CE
        + L_Dice
        + lambda_edge * L_edge
        + lambda_balance * L_router_balance
        + lambda_sym * L_temporal_consistency
```

使用伪标签预训练时，再加入 MAE 抗噪项。

推荐初始权重：

```text
lambda_edge = 0.1
lambda_balance = 0.01
lambda_sym = 0.1
lambda_mae = 0.3
```

## 9. Patch 思想如何迁移到 ScratchFormer

EIPCNet 使用 5×5 或 7×7 的像素中心 Patch，是一个局部分类网络。ScratchFormer 是像素级分割网络，不能直接照搬这种 Patch 尺寸。

建议在 ScratchFormer 中比较：

```text
128×128，stride=64
256×256，stride=128 或 64
```

测试阶段使用滑窗预测并融合重叠区域 Logits，恢复完整变化图。

论文 Patch 尺寸实验可借鉴的本质是：

- 感受野过小容易受斑点噪声干扰。
- 感受野过大容易混合变化和未变化区域。
- 应通过消融找到适合当前网络和数据的输入范围。

## 10. 推荐实验矩阵

### 10.1 主实验

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | 单通道 ScratchFormer | SAR 基线 |
| B1 | ScratchFormer + Weighted CE + Dice | 排除损失函数影响 |
| E1 | B1 + GNR 辅助分支 | 验证 GNR 差异先验 |
| E2 | B1 + 普通 MoE | 验证通用 MoE 收益 |
| E3 | B1 + GNR 引导 MoE | 验证 GNR 路由作用 |
| E4 | E3 + 路由均衡 | 防止专家塌缩 |
| E5 | E4 + 时相交换一致性 | 最终监督模型 |
| E6 | E5 + 伪标签预训练 | 完整小样本模型 |

### 10.2 GNR 消融

| 设置 | 说明 |
|---|---|
| 无差异先验 | 仅使用深层特征 |
| Log-ratio | 传统比率差异图 |
| GNR | 论文提出的边缘增强差异图 |
| GNR 直接融合 | 将 GNR 与解码特征拼接 |
| GNR 引导路由 | GNR 仅参与专家权重计算 |

### 10.3 伪标签消融

| 设置 | 说明 |
|---|---|
| 仅真实标签 | 标准监督基线 |
| 仅伪标签 | 检验无监督训练能力 |
| 伪标签预训练后微调 | 推荐方案 |
| 无置信度筛选 | 所有伪标签参与训练 |
| 有置信度筛选 | 不确定区域设为 ignore |
| Focal Loss | 只处理类别不平衡 |
| Focal + MAE | 同时处理不平衡和标签噪声 |

### 10.4 MoE 消融

| 消融项 | 推荐配置 |
|---|---|
| 专家数量 | 2、3、4 |
| 路由方式 | Dense、Top-1、Top-2 |
| 路由输入 | 特征差异、GNR、特征差异+GNR |
| MoE 位置 | 最终融合、CEFF、多尺度全部位置 |
| 专家参数 | 各尺度独立、跨尺度共享 |
| 均衡损失 | 关闭、开启 |

## 11. SAR 专项验证

为了证明改进来自 SAR 特性而不是单纯增加参数，应增加以下实验：

### 11.1 斑点噪声鲁棒性

在测试影像上增加不同强度的乘性 Gamma 斑点噪声：

```text
等效视数 L = 1、2、4、8
```

比较基础 ScratchFormer、普通 MoE 和 GNR 引导 MoE 的 F1、IoU 下降幅度。

### 11.2 边界指标

除 F1、IoU 和 Kappa 外，可以增加 Boundary F1 或 Boundary IoU，验证 GNR 是否真正改善变化区域边缘。

### 11.3 路由可视化

建议展示：

- 不同数据集上的专家平均权重。
- 变化区域和背景区域的专家选择差异。
- 不同斑点噪声强度下的路由变化。
- 是否发生某个专家长期独占的路由塌缩。

## 12. 最值得形成论文创新的方向

最推荐的论文核心创新为：

> 使用 GNR 差异先验引导 ScratchFormer 多尺度变化特征的共享稀疏 MoE 路由，使不同专家分别建模局部变化、斑点噪声和大范围上下文，并通过路由均衡及时相一致性提高小样本跨场景泛化能力。

该方向与 EIPCNet 的区别为：

| EIPCNet | 拟研究方法 |
|---|---|
| GNR 用于生成差异图和伪标签 | GNR 用于指导 Transformer/MoE 路由 |
| 每个场景单独训练 | 强调训练域到未知测试域的泛化 |
| 小 Patch 分类网络 | ScratchFormer 像素级分割网络 |
| 普通卷积注意力网络 | 多尺度 Transformer 与共享稀疏专家 |
| 主要处理伪标签不平衡和噪声 | 同时处理斑点噪声、稀疏变化和专家路由 |

## 13. 创新边界与引用要求

以下内容来自 EIPCNet 论文，不能作为自己的原创贡献：

- GNR 差异图公式。
- PCH-FLICM 原型约束层次聚类。
- CBAFL 与 MAE 的组合损失思想。
- EIPCNet 的高置信伪标签筛选流程。

使用这些模块时必须引用原论文。

可以形成自身创新的内容包括：

- 将 GNR 用于 ScratchFormer 的 MoE 动态路由。
- 面向单极化 SAR 的专家结构设计。
- 跨尺度共享专家和 Top-k 稀疏路由。
- GNR、深层差异特征和路由器的融合方式。
- 路由均衡及时相交换一致性约束。
- 严格无数据泄漏的跨数据域实验协议。

## 14. 推荐实施顺序

1. 完成单通道 ScratchFormer 基线。
2. 完成 Weighted CE + Dice，固定统一损失基线。
3. 实现 GNR 计算并保存差异图。
4. 测试 GNR 直接融合是否有效。
5. 实现仅位于最终融合层的 GNR 引导 MoE。
6. 实现 CEFF 多尺度共享专家和 Top-2 路由。
7. 加入路由均衡及时相交换一致性。
8. 实现高置信伪标签生成和预训练。
9. 完成 GNR、MoE、损失和伪标签消融。
10. 完成斑点噪声、边界精度和路由可视化实验。

## 15. 总结

EIPCNet 对当前研究最有价值的不是其小型分类网络本身，而是以下三个可迁移思想：

1. 使用显式 SAR 差异先验强化边缘并抑制斑点噪声。
2. 通过高置信伪标签利用少量完整场景中的大量无标签像素。
3. 同时处理类别不平衡和错误伪标签，提升小样本训练稳定性。

建议以 GNR 引导的 SAR-SSMoE 作为主创新，以伪标签预训练、抗噪损失和边缘监督作为支撑策略。所有新增模块都应逐项消融，并与普通 MoE、无 GNR 的 MoE 和单通道 ScratchFormer 基线进行公平比较。
