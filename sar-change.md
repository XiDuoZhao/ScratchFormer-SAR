# 基于 ScratchFormer 的小样本单极化 SAR 变化检测实验方案

## 1. 研究目标

本研究以 ScratchFormer 为基础框架，将原始光学遥感变化检测模型适配到单极化 SAR 变化检测任务，并针对现有数据量小、斑点噪声强、变化类别不平衡和跨场景泛化困难等问题，引入轻量级 Mixture of Experts（MoE）模块。

建议的研究主题为：

> 面向小样本单极化 SAR 变化检测的斑点噪声感知共享稀疏 MoE-ScratchFormer

研究重点不是简单地在网络中堆叠 MoE，而是验证以下问题：

1. ScratchFormer 是否能够从光学变化检测有效迁移到单通道 SAR 变化检测。
2. SAR 特有的多尺度纹理和斑点噪声是否适合由不同专家分别建模。
3. 共享稀疏 MoE 是否能在少量训练场景下提高泛化能力，同时控制参数量和过拟合。
4. 改进模型在不同 SAR 数据域、不同噪声强度下是否均能取得稳定提升。

## 2. 当前数据核验结果

数据位于：

```text
D:\lunwen\DataSets\SAR影像\sar
```

现有 27 幅 BMP 可以组成 9 组双时相影像及标签，但其中只有 7 组适合用于单极化 SAR 主实验。

| 场景 | 前时相 | 后时相 | 标签 | 尺寸 | 变化比例 | 用途 |
|---|---|---|---|---:|---:|---|
| Ottawa | `1997.05.bmp` | `1997.08.bmp` | `reference1.bmp` | 290×350 | 15.812% | SAR 主实验 |
| Bern | `1999.04.bmp` | `1999.05.bmp` | `reference.bmp` | 301×301 | 1.275% | SAR 主实验 |
| Farmland-B | `im11.bmp` | `im22.bmp` | `im33_rf.bmp` | 257×289 | 18.085% | SAR 主实验 |
| Coastline | `img1.bmp` | `img2.bmp` | `img_rf.bmp` | 450×280 | 1.070% | SAR 主实验 |
| Farmland-A | `img11.bmp` | `img22.bmp` | `img33_rf.bmp` | 306×291 | 5.918% | SAR 主实验 |
| San Francisco | `san_1.bmp` | `san_2.bmp` | `san_gt.bmp` | 256×256 | 7.149% | SAR 主实验 |
| Inland River | `Yellow River1.bmp` | `Yellow River2.bmp` | `Yellow River_rf.bmp` | 291×444 | 3.293% | SAR 主实验 |
| Sardinia | `im1.bmp` | `im2.bmp` | `im3.bmp` | 412×300 | 6.170% | Landsat 光学数据，不进入 SAR 主实验 |
| 疑似 Mexico | `2002.5.bmp` | `2005.4.bmp` | `reference2.bmp` | 512×512 | 9.761% | 光学数据，不进入 SAR 主实验 |

7 组 SAR 数据合计约 676160 个像素，其中变化像素约占 6.83%。不同场景的变化比例从 1.07% 到 18.09%，类别不平衡和场景差异都比较明显。

所有以 RGB 格式保存的 SAR 影像，其三个颜色通道实际完全相同，因此可以无损转换为单通道灰度图，并使用：

```text
--img_mode L --input_nc 1
```

## 3. 数据规模限制与实验定位

这些数据是少量完整场景，而不是大量相互独立的训练样本。即使通过滑窗裁剪得到数百或数千个 Patch，相邻 Patch 仍然高度相关，不能将其描述为大规模数据集。

以 `128×128`、步长 64 裁剪，7 组 SAR 数据只能形成约 129 个重叠基础 Patch。因此，本研究应定位为：

- 小样本单极化 SAR 变化检测。
- 跨场景或跨数据域泛化。
- 参数高效、噪声鲁棒的 MoE 改进。
- 不宣称面向大规模 SAR 数据的通用 SOTA。

## 4. 数据预处理方案

### 4.1 标签二值化

统一使用以下规则：

```python
label = (label >= 128).astype(np.uint8)
```

不建议继续使用：

```python
label = label // 255
```

原因是 `reference2.bmp` 含有 84 个灰度值，直接整除只会将像素值恰好等于 255 的位置识别为变化区域，会丢失 18175 个应属于变化区域的像素。虽然该组数据不进入 SAR 主实验，统一阈值化仍能提高数据处理的稳健性。

### 4.2 SAR 输入处理

主实验采用单通道输入，推荐比较以下三种输入策略：

1. 原始 8 位强度图归一化，作为基础方案。
2. `log1p` 对数变换后进行百分位归一化。
3. 对数变换加轻量 Lee 或中值去斑，作为预处理消融实验。

建议使用每幅影像的 1% 和 99% 分位数进行鲁棒归一化，避免极端强散射点主导数值范围。去斑不应默认启用，因为过强滤波可能破坏变化边缘。

### 4.3 裁块与拼接

推荐配置：

```text
Patch 大小：128×128
训练步长：64
测试步长：64
```

ScratchFormer 最深层下采样倍率约为 32。输入为 128 时，最深层仍有约 `4×4` 的空间特征；输入为 64 时仅剩约 `2×2`，不利于深层空间建模。

训练阶段应先按空间区域划分训练、验证和测试范围，再在各范围内部裁块。不能先裁出全部重叠 Patch 后随机打乱划分，否则相邻区域会同时出现在训练集和测试集，造成数据泄漏。

测试阶段使用重叠滑窗预测，将重叠区域的 Logits 取平均后恢复完整变化图。可使用 Hann 或高斯权重降低 Patch 边缘拼接痕迹。

### 4.4 数据增强

建议使用：

- 同步水平翻转和垂直翻转。
- 同步旋转 90°、180°、270°。
- 随机交换前后时相，标签保持不变。
- 轻微灰度增益和 Gamma 变化。
- 可控的乘性 Gamma 斑点噪声增强。

不使用 RGB ColorJitter。当前代码中的随机模糊应作为独立消融项，不建议默认开启。

### 4.5 采样策略

建议先均匀选择场景，再从该场景中进行类别平衡采样，避免尺寸较大的场景支配训练过程。

可将 Patch 分为：

- 正样本 Patch：变化像素比例不低于 1%。
- 背景或困难负样本 Patch：变化像素比例低于 1%。

训练时按照约 1:1 采样两类 Patch，但验证和测试阶段必须保持真实类别分布。

## 5. 无数据泄漏的划分协议

Farmland-A、Farmland-B、Coastline 和 Inland River 来源于同一组黄河口双时相 SAR 大图，应绑定为同一个数据域。不能使用其中三个训练、另一个测试，否则会造成同源数据泄漏。

主实验划分为四个数据域：

1. Ottawa。
2. Bern。
3. San Francisco。
4. Yellow River，包括 Farmland-A、Farmland-B、Coastline 和 Inland River。

### 5.1 主协议：四折留一数据域测试

每一折完整保留一个数据域作为测试集，其余三个数据域用于训练。训练数据域内部划出约 15%～20% 的非重叠空间块作为验证集，并丢弃跨越划分边界的 Patch。

| 折次 | 测试域 | 训练域 |
|---|---|---|
| Fold 1 | Ottawa | Bern、San Francisco、Yellow River |
| Fold 2 | Bern | Ottawa、San Francisco、Yellow River |
| Fold 3 | San Francisco | Ottawa、Bern、Yellow River |
| Fold 4 | Yellow River 全部子场景 | Ottawa、Bern、San Francisco |

该协议用于检验模型能否泛化到从未见过的 SAR 地区和成像条件，应作为论文最主要的结果。

### 5.2 辅助协议：场景内空间块交叉验证

可以对每个完整场景执行空间块交叉验证，用于展示模型在同一数据域中的上限性能。该结果必须与跨域实验分表报告，不能混合平均。

### 5.3 随机种子

每个模型、每一折至少运行 3 个随机种子，例如：

```text
0、42、3407
```

最终报告四折和多个随机种子的均值与标准差，避免只选择一次最好结果。

## 6. 模型设计

### 6.1 SAR ScratchFormer 基线

保留原始 ScratchFormer 的主要结构：

- 双时相共享参数的孪生编码器。
- 四阶段层次化特征提取。
- Shuffled Sparse Attention（SSA）。
- Change-Enhanced Feature Fusion（CEFF）。
- 多尺度预测和最终解码器。

基线只进行必要的 SAR 适配：

```text
输入通道：3 -> 1
图像模式：RGB -> L
关闭颜色扰动
保持编码器、CEFF 和解码器结构不变
关闭 MoE
```

该模型记为 `ScratchFormer-SAR-Baseline`。

### 6.2 当前 MoE 原型的作用与问题

仓库当前的 `SARMoEBlock` 包含三个卷积专家，并在输入端、四个 CEFF 输出以及最终融合特征上使用 MoE。它可以作为第一版原型和消融对照，但不建议直接作为论文最终方法，主要原因包括：

- 输入端只有一个通道，三个专家的表达空间有限。
- 路由器主要依赖全局平均池化，稀疏变化可能被背景平均掉。
- 所有专家均被 Dense Softmax 混合，并不是真正的稀疏 Top-k 路由。
- 没有路由均衡损失，可能出现专家塌缩。
- 五个尺度分别使用独立专家，增加参数和过拟合风险。
- 专家使用 BatchNorm，小 Batch 下统计量可能不稳定。

### 6.3 建议的 SAR-SSMoE

最终方法建议命名为：

```text
SAR-SSMoE：SAR Speckle-aware Shared Sparse Mixture of Experts
```

专家设计建议如下：

| 专家 | 结构 | 主要作用 |
|---|---|---|
| Expert 1 | 深度可分离 `3×3` 卷积 | 提取局部纹理和小变化 |
| Expert 2 | 深度可分离 `5×5` 卷积 | 建模斑点噪声和较大邻域 |
| Expert 3 | 空洞 `3×3` 卷积 | 提取大范围上下文和结构变化 |
| 可选共享分支 | Identity 或轻量 `1×1` 卷积 | 保留通用变化信息 |

路由器不直接使用单幅影像，而是基于双时相差异特征：

```text
d = abs(f_pre - f_post)
router_input = concat(GAP(d), GMP(d))
```

其中 GAP 为全局平均池化，GMP 为全局最大池化。最大池化可以增强路由器对稀疏变化区域的敏感性。

建议使用 Top-2 路由，仅激活权重最大的两个专家。四个 CEFF 尺度可以共享同一组专家，仅保留尺度相关路由器或尺度嵌入，从而减少参数量并促进跨尺度知识共享。

优先测试以下放置方式：

1. 只在最终多尺度融合特征后加入一个 MoE。
2. 在四个 CEFF 输出后使用共享 MoE。
3. 在四个 CEFF 输出和最终融合位置同时使用共享 MoE。
4. 输入端 MoE 仅作为消融，不作为默认结构。

### 6.4 归一化方式

MoE 专家内部优先使用 GroupNorm 或 LayerNorm，避免 BatchNorm 在 batch size 为 4～8 时产生不稳定统计量。

### 6.5 时相交换一致性

标准二值变化检测对时间顺序应基本对称，即：

```text
Model(A, B) ≈ Model(B, A)
```

可以对预测概率和路由权重增加一致性约束，使模型减少对时相顺序的偶然依赖，并提高小样本训练稳定性。

## 7. 损失函数

SAR 数据变化比例较低，不建议仅使用普通交叉熵。主损失建议为：

```text
L_seg = L_weighted_CE + L_Dice
```

完整 MoE 模型的损失为：

```text
L_total = L_seg
        + lambda_balance * L_router_balance
        + lambda_sym * L_temporal_consistency
```

初始建议：

```text
lambda_balance = 0.01
lambda_sym = 0.1
```

这些权重只能根据验证集选择，不能根据测试集结果调整。所有模型对比应使用相同的分割损失，避免把损失函数提升误认为 MoE 模块提升。

## 8. 推荐训练参数

| 参数 | 建议值 |
|---|---|
| 输入尺寸 | 128×128 |
| Batch size | 4 或 8 |
| 有效 Batch size | 通过梯度累积达到 16 |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-2` |
| 最大 Epoch | 200 |
| Warmup | 5～10 Epoch |
| 学习率策略 | Cosine decay |
| Early stopping | 30 Epoch |
| 混合精度 | 开启 AMP |
| 随机种子 | 至少 3 个 |

原论文参数 `lr=4.1e-4、batch size=16、epoch=300` 应保留用于 LEVIR-CD 复现。SAR 实验的数据规模和 Batch size 不同，应单独使用上述小样本配置，并在论文中说明原因。

## 9. 实验矩阵

### 9.1 基础对比实验

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | Log-ratio + Otsu 或 FCM | 传统无监督基线 |
| B1 | FC-Siam-diff | 轻量 CNN 孪生基线 |
| B2 | ScratchFormer 单通道基线 | 验证基础框架 |
| B3 | ScratchFormer + 当前 Dense MoE | 验证现有 MoE 原型 |
| M1 | ScratchFormer + 最终融合 Shared MoE | 验证单位置 MoE |
| M2 | ScratchFormer + 多尺度 Shared MoE | 验证多尺度共享专家 |
| M3 | M2 + Top-2 + 路由均衡 | 验证稀疏路由和防塌缩 |
| M4 | M3 + 时相交换一致性 | 最终完整模型 |

### 9.2 MoE 消融实验

| 消融项 | 可选配置 |
|---|---|
| 专家数量 | 2、3、4 |
| 路由方式 | Dense、Top-1、Top-2 |
| MoE 位置 | 输入端、最终融合、CEFF 多尺度、全部位置 |
| 专家共享 | 各尺度独立、跨尺度共享 |
| 归一化 | BatchNorm、GroupNorm、LayerNorm |
| 路由输入 | GAP、GAP+GMP、差异特征统计 |
| 均衡损失 | 关闭、开启 |
| 时相一致性 | 关闭、开启 |

### 9.3 SAR 输入与增强消融

| 消融项 | 配置 |
|---|---|
| 输入变换 | 原始强度、Log、Log+去斑 |
| Patch 大小 | 128、256 |
| 时间交换 | 关闭、开启 |
| 斑点增强 | 关闭、开启 |
| 随机模糊 | 关闭、开启 |

### 9.4 可选自监督实验

如果监督训练仍然明显过拟合，可以只使用当前训练折中的无标签 SAR Patch 进行掩码重建或对比学习预训练，再进行监督微调。

严禁在自监督预训练阶段使用测试数据域的影像，否则主协议将变成跨域数据可见的传导式实验。若确实使用测试域无标签影像，必须单独标注为 Transductive Setting，不能与严格跨域结果混合。

## 10. 评价指标

主要指标：

- 变化类 F1。
- 变化类 IoU。
- Precision。
- Recall。
- Kappa。
- Matthews Correlation Coefficient（MCC）。

辅助指标：

- Overall Accuracy（OA）。
- False Positive 和 False Negative。

由于变化像素比例较低，OA 很容易被大量未变化像素抬高，不能作为主要结论依据。

结果应报告：

- 每个测试数据域的独立结果。
- 四折平均值和标准差。
- 多随机种子的平均值和标准差。
- 模型参数量、FLOPs、GPU 显存和单幅推理时间。
- 最优阈值只能由验证集确定，不能在测试标签上调节。

可以进一步使用配对 Wilcoxon 检验或 Bootstrap 置信区间，验证改进是否稳定而非随机波动。

## 11. SAR 噪声鲁棒性实验

为了证明模型确实针对 SAR，而不是依靠增加参数获得提升，可以在测试影像上添加不同强度的乘性斑点噪声，例如等效视数：

```text
L = 1、2、4、8
```

比较不同模型在噪声增强前后的 F1 和 IoU 下降幅度。若 SAR-SSMoE 的下降幅度小于基础 ScratchFormer，可以为斑点噪声专家的有效性提供直接证据。

同时应可视化：

- 不同场景的专家路由权重。
- 不同噪声强度下的专家选择变化。
- MoE 前后的特征响应图。
- 典型正确检测、漏检和误检区域。

## 12. 当前代码正式实验前需要完成的事项

### 12.1 数据配置

当前 `utils.resolve_data_config()` 会先调用 `DataConfig().get_data_config(data_name)`，因此即使提供新的 `--data_root`，未定义的 `data_name` 仍可能提前报错。

需要选择以下一种方式：

1. 在 `data_config.py` 中新增 SAR 数据配置。
2. 修改 `resolve_data_config()`，当提供 `data_root` 时允许直接创建自定义配置。

### 12.2 数据集构建脚本

需要新增预处理脚本，将原始 7 组 SAR 场景转换为代码要求的结构：

```text
SAR-CD/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

脚本需要负责：

- 场景配对。
- 单通道转换。
- 标签阈值化。
- 按空间区域划分数据。
- 生成 Patch 和文件清单。
- 保存场景名称与坐标，供完整影像拼接和数据泄漏检查使用。

### 12.3 数据加载器

需要调整：

- 标签使用 `>=128` 二值化。
- 训练集 `shuffle=True`。
- 验证集和测试集 `shuffle=False`。
- SAR 默认关闭 ColorJitter。
- 随机模糊改为可配置参数。
- 支持场景均衡和变化 Patch 均衡采样。

### 12.4 MoE 模型

需要为命令行增加可消融配置，例如：

```text
--moe_position final|multiscale|all
--moe_num_experts 3
--moe_top_k 2
--moe_shared
--moe_norm group
--moe_balance_weight 0.01
--temporal_consistency_weight 0.1
```

训练日志中应保存每个专家的平均路由概率和使用次数，用于判断是否发生专家塌缩。

## 13. 推荐实施顺序

1. 在 LEVIR-CD 上复现原始 ScratchFormer，确认代码和环境正确。
2. 完成 7 组 SAR 数据的身份确认、单通道转换、空间划分和裁块。
3. 运行传统方法、FC-Siam-diff 和单通道 ScratchFormer 基线。
4. 运行当前 Dense MoE 原型，判断 MoE 是否具有初步收益。
5. 实现只位于最终融合层的 Shared MoE。
6. 实现多尺度共享专家和 Top-2 路由。
7. 加入路由均衡损失和时相交换一致性。
8. 完成四折跨域实验和 3 个随机种子重复实验。
9. 完成专家数量、放置位置、路由方式和损失项消融。
10. 完成斑点噪声鲁棒性、复杂度和可视化实验。
11. 若监督结果仍不稳定，再增加严格限制于训练域的自监督预训练。

## 14. 预期论文贡献

论文可以围绕以下三点组织：

1. 将 ScratchFormer 系统适配到单极化 SAR，并建立严格按数据域划分的小样本评估协议。
2. 提出面向斑点噪声、局部纹理和大范围上下文的共享稀疏 MoE，减少独立专家带来的参数和过拟合。
3. 使用路由均衡及时相交换一致性增强小样本训练稳定性，并通过跨域、噪声鲁棒性和专家可视化验证其有效性。

仅添加普通 MoE 很难形成充分创新。最终方法必须将 MoE 的专家设计、路由依据、参数共享和 SAR 噪声特性建立明确联系。

## 15. 风险与结论边界

- 数据域数量只有 4 个，统计结论必须使用多随机种子和置信区间。
- Yellow River 的四个子场景不能被当成四个完全独立的数据域。
- 相邻重叠 Patch 不能随机分到训练集和测试集。
- 不应根据测试结果选择模型位置、专家数量或损失权重。
- 不应只报告最好一次结果。
- 不应使用 OA 作为主要结论。
- 不应将少量场景裁出的 Patch 描述成大规模 SAR 数据集。

在遵守以上实验协议的前提下，这些数据可以支撑一篇以“小样本、单极化 SAR、共享稀疏 MoE、跨场景泛化”为主题的小论文，但不足以单独支撑大规模 SAR 通用模型或全面 SOTA 的结论。

## 16. 相关工作

- [ScratchFormer 官方仓库](https://github.com/mustansarfiaz/ScratchFormer)
- [当前 ScratchFormer-SAR 仓库](https://github.com/XiDuoZhao/ScratchFormer-SAR)
- [Convolution and Attention Mixer for SAR Change Detection](https://arxiv.org/abs/2309.12010)
- [M2CD: Optical-SAR Change Detection with Mixture of Experts](https://arxiv.org/abs/2503.19406)
- [SSLChange: Self-supervised Change Detection](https://arxiv.org/abs/2405.18224)
- [Yellow River SAR 数据集及不同子场景说明](https://www.mdpi.com/1424-8220/21/24/8290)
- [SAR 变化检测数据集与多方向注意力方法](https://www.mdpi.com/2072-4292/16/19/3590)

