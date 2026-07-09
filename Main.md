# ScratchFormer 复现与 SAR-MoE 改进实验路线

## 1. 论文框架总结

论文《Remote Sensing Change Detection With Transformers Trained From Scratch》提出了一个面向遥感变化检测的 Siamese Transformer 框架，模型名称为 ScratchFormer。该方法的核心目标是解决传统 Transformer 变化检测模型依赖预训练的问题，使模型可以直接在目标变化检测数据集上从头训练。

传统 Transformer 变化检测方法通常需要 ImageNet 预训练，或者先在其他变化检测数据集上预训练，再迁移到目标数据集。论文认为，标准 self-attention 在小规模变化检测数据集上难以充分学习图像结构和变化区域的归纳偏置，因此提出了更适合变化检测任务的注意力机制和双时相融合模块。

ScratchFormer 的整体结构如下：

```text
变化前图像 Ipre
变化后图像 Ipost
        ↓
共享权重的 Siamese 编码器
        ↓
四个尺度的双时相特征
        ↓
CEFF 变化增强特征融合
        ↓
多尺度解码器
        ↓
二值变化图 M
```

模型主要由三部分组成：

- Siamese hierarchical encoder：双分支共享编码器，分别提取变化前和变化后图像的多尺度特征。
- Shuffled Sparse Attention, SSA：编码器中的核心注意力模块，通过数据相关的稀疏采样和特征重排，让模型关注稀疏但重要的变化区域。
- Change-Enhanced Feature Fusion, CEFF：双时相特征融合模块，通过逐通道重加权增强真实语义变化，抑制阴影、光照、季节变化等伪变化。

论文中的编码器共有 4 个阶段，每个阶段的 block 数量为：

```text
[3, 3, 9, 3]
```

对应的多尺度特征分辨率大致为：

```text
H/4, H/8, H/16, H/32
```

对应通道数为：

```text
[64, 128, 320, 512]
```

论文的主要创新点可以概括为：

- 使用 SSA 替代标准 dense self-attention，使 Transformer 更容易从头训练。
- 使用 CEFF 替代简单相加、相减或拼接，使双时相特征融合更关注真实变化。
- 在多个遥感变化检测数据集上证明 ScratchFormer 不依赖预训练也可以取得较强性能。

## 2. 论文复现流程

复现论文实验时，建议先复现光学遥感变化检测数据集上的 baseline，不要直接进入 SAR 改造。这样可以确认代码、数据、训练参数和评估流程都是可靠的。

### 2.1 数据集准备

当前代码要求的数据集结构如下：

```text
数据集根目录
├─A
├─B
├─label
└─list
   ├─train.txt
   ├─val.txt
   └─test.txt
```

其中：

- `A`：变化前图像。
- `B`：变化后图像。
- `label`：二值变化标签。
- `list/train.txt`：训练集文件名列表。
- `list/val.txt`：验证集文件名列表。
- `list/test.txt`：测试集文件名列表。

当前 LEVIR-CD 数据集已整理到：

```text
./datasets/CD/LEVIR-CD-256/
```

`data_config.py` 中 LEVIR 的路径应保持为：

```python
self.root_dir = './datasets/CD/LEVIR-CD-256/'
```

### 2.2 论文训练参数

论文主要训练设置如下：

```text
输入尺寸: 256 x 256
输入通道: 3
类别数: 2
batch size: 16
优化器: AdamW
学习率: 4.1e-4
weight decay: 0.01
betas: (0.9, 0.999)
训练轮次: 300
损失函数: pixel-wise cross entropy
学习率策略: linear decay
```

复现时需要特别注意：

- 论文输入尺寸是 `256`，而代码中曾经默认使用过 `512`，复现时应显式指定 `--img_size 256`。
- 复现论文 baseline 时不要开启 `--use_moe`。
- 复现光学数据集时使用 `--img_mode RGB` 和 `--input_nc 3`。

### 2.3 推荐复现命令

训练 LEVIR-CD：

```bash
python main_cd.py --data_name LEVIR --img_size 256 --batch_size 16 --lr 0.00041 --max_epochs 300 --optimizer adamw --loss ce --lr_policy linear --img_mode RGB --input_nc 3 --n_class 2
```

单独测试：

```bash
python eval_cd.py --data_name LEVIR --img_size 256 --batch_size 1 --img_mode RGB --input_nc 3 --n_class 2 --checkpoint_name best_ckpt.pt
```

### 2.4 复现时需要记录的信息

每次实验建议记录：

- 数据集名称。
- 数据集路径。
- 输入尺寸。
- batch size。
- 学习率。
- 优化器。
- 训练轮次。
- 是否使用预训练。
- 是否开启 MoE。
- 最佳验证轮次。
- 测试集 F1、IoU、OA、Precision、Recall。
- 训练总耗时和测试总耗时。

当前代码已经将训练和测试日志写入：

```text
checkpoints/项目名称/Logs/
```

每次训练和测试都会生成独立日志文件。

## 3. 复现后如何迁移到 SAR 变化检测

完成论文 baseline 复现后，可以将 ScratchFormer 作为 SAR 变化检测的基础框架。迁移的核心思想是：保留 ScratchFormer 的 Siamese 编码器、SSA 注意力和 CEFF 变化增强融合结构，只针对 SAR 的输入模态和噪声特性做适配。

### 3.1 SAR 数据特点

单极化 SAR 与光学图像存在明显差异：

- SAR 通常是单通道灰度图，而不是 RGB 三通道图像。
- SAR 图像存在 speckle 斑点噪声。
- SAR 的灰度值反映后向散射强度，与光学颜色语义不同。
- 变化区域可能边界碎裂、纹理复杂。
- 不同地物的散射变化模式差异明显。

因此，不能简单地把 SAR 图像强制转为 RGB 后使用光学变化检测流程。

### 3.2 当前代码中的 SAR 适配

当前代码已经支持单极化 SAR 输入，主要改动包括：

- `datasets/CD_dataset.py`：新增 `img_mode` 参数，支持 `RGB` 和 `L` 两种读取方式。
- `datasets/data_utils.py`：归一化方式改为按实际通道数自适应生成 mean 和 std。
- `utils.py`：将 `img_mode` 传递到数据集，并在 SAR 模式下关闭颜色扰动。
- `main_cd.py` 和 `eval_cd.py`：新增 `--img_mode` 和 `--input_nc` 参数。
- `models/networks.py`：模型构造时显式传入输入通道数。

SAR baseline 训练命令示例：

```bash
python main_cd.py --data_name YOUR_SAR_DATA --data_root YOUR_SAR_ROOT --img_mode L --input_nc 1 --img_size 256 --n_class 2
```

SAR baseline 测试命令示例：

```bash
python eval_cd.py --data_name YOUR_SAR_DATA --data_root YOUR_SAR_ROOT --img_mode L --input_nc 1 --img_size 256 --n_class 2 --checkpoint_name best_ckpt.pt
```

### 3.3 SAR 迁移实验顺序

建议按以下顺序开展：

1. 复现光学 LEVIR-CD baseline，确认代码可靠。
2. 准备 SAR 数据集，并整理为 `A/B/label/list` 结构。
3. 使用 `--img_mode L --input_nc 1` 跑通 ScratchFormer-SAR baseline。
4. 记录 SAR baseline 的 F1、IoU、OA、Precision、Recall。
5. 在 baseline 上逐步加入 SAR 专用改进模块。
6. 做消融实验，验证每个模块的贡献。

## 4. 如何在框架上新增 MoE 模块

在 ScratchFormer 上新增 MoE 模块时，建议不要破坏原有 SSA 编码器和 CEFF 融合逻辑。更稳妥的方式是在 CEFF 后或多尺度融合后增加轻量级 MoE 特征增强模块，使模型可以根据 SAR 图像的不同纹理、噪声和变化模式自适应选择专家。

### 4.1 MoE 设计动机

SAR 变化检测中，不同区域的变化模式差异较大：

- 小目标变化依赖局部边缘和细节。
- 大范围变化依赖更大的上下文。
- speckle 噪声容易造成局部误检。
- 建筑、水体、植被等地物的散射变化模式不同。

单一卷积核或单一注意力分支难以同时适配这些情况。MoE 通过多个专家分支建模不同特征模式，再由门控网络根据输入自适应分配专家权重，因此适合用于 SAR 变化检测。

### 4.2 推荐 MoE 插入位置

推荐的整体结构如下：

```text
变化前 SAR 图像
变化后 SAR 图像
        ↓
可选 Input SAR-MoE
        ↓
共享 ScratchFormer 编码器
        ↓
四尺度双时相特征
        ↓
CEFF 变化增强融合
        ↓
Scale-wise SAR-MoE
        ↓
多尺度特征拼接融合
        ↓
Global SAR-MoE
        ↓
Decoder
        ↓
变化图
```

建议优先尝试三个位置：

- 输入端 MoE：用于 SAR 模态适配和低层纹理增强。
- CEFF 后 MoE：用于增强每个尺度的变化特征。
- 多尺度融合后 MoE：用于整合不同尺度的变化模式。

其中最推荐的位置是 CEFF 后，因为 CEFF 后的特征已经是双时相变化增强特征，MoE 在这里更容易直接作用于变化检测目标。

### 4.3 当前代码中的 SAR-MoE 原型

当前代码已经加入了一个可切换的 `SARMoEBlock`，位于 `models/scratch_former.py`。

该模块包含三个专家：

```text
Expert 1: 3x3 depthwise convolution
Expert 2: 5x5 depthwise convolution
Expert 3: dilated 3x3 depthwise convolution
```

三个专家分别负责：

- `3x3` 专家：建模局部边缘和细小变化。
- `5x5` 专家：建模更平滑的局部上下文。
- `dilated 3x3` 专家：建模更大感受野的区域变化。

门控网络结构为：

```text
GAP
1x1 Conv
GELU
1x1 Conv
Softmax
```

MoE 输出形式为：

```text
Y = X + GELU(sum(g_i * Expert_i(X)))
```

其中 `g_i` 是门控网络预测的专家权重。

### 4.4 SAR-MoE 使用方式

训练 SAR baseline：

```bash
python main_cd.py --data_name YOUR_SAR_DATA --data_root YOUR_SAR_ROOT --img_mode L --input_nc 1 --img_size 256 --n_class 2
```

训练 SAR-MoE：

```bash
python main_cd.py --data_name YOUR_SAR_DATA --data_root YOUR_SAR_ROOT --img_mode L --input_nc 1 --img_size 256 --n_class 2 --use_moe
```

测试 SAR-MoE：

```bash
python eval_cd.py --data_name YOUR_SAR_DATA --data_root YOUR_SAR_ROOT --img_mode L --input_nc 1 --img_size 256 --n_class 2 --use_moe --checkpoint_name best_ckpt.pt
```

### 4.5 建议消融实验

为了支撑小论文，建议至少做以下实验：

- ScratchFormer-SAR baseline。
- ScratchFormer-SAR + Input MoE。
- ScratchFormer-SAR + Scale-wise MoE。
- ScratchFormer-SAR + Global MoE。
- ScratchFormer-SAR + Scale-wise MoE + Global MoE。
- ScratchFormer-SAR + Input MoE + Scale-wise MoE + Global MoE。

如果实验资源有限，最少保留：

- Baseline。
- Baseline + Scale-wise MoE。
- Baseline + Scale-wise MoE + Global MoE。
- Full SAR-MoE。

### 4.6 评价指标

SAR 变化检测中，变化类通常比整体准确率更重要。建议重点报告：

- F1。
- IoU。
- OA。
- Precision。
- Recall。
- F1_1。
- IoU_1。

其中 `F1_1` 和 `IoU_1` 更能反映变化类检测能力。

## 5. 推荐论文实验叙事

后续小论文可以按照以下逻辑组织：

```text
1. ScratchFormer 在光学遥感变化检测中表现良好，但原始模型主要面向 RGB 光学图像。
2. 单极化 SAR 图像具有 speckle 噪声、灰度散射统计和复杂纹理变化。
3. 直接迁移 ScratchFormer 到 SAR 后，仍存在误检、漏检和边界碎裂问题。
4. 因此提出 SAR-MoE 模块，在变化增强特征上进行多专家自适应增强。
5. 多专家分别关注局部细节、区域上下文和大感受野变化。
6. 门控机制根据输入特征自适应选择专家权重。
7. 在多个 SAR 变化检测数据集上验证方法有效性。
```

可以将方法命名为：

```text
SAR-MoE ScratchFormer
```

或者：

```text
SAR-aware Mixture-of-Experts ScratchFormer
```

## 6. 后续可继续增强的方向

在 SAR-MoE 基础上，还可以继续探索：

- SAR 专用预处理：对数变换、dB 变换、分位数裁剪归一化。
- SAR 专用增强：speckle 噪声增强、局部对比度扰动。
- 损失函数增强：CE + Dice、Focal + IoU、类别不平衡加权损失。
- MoE 可解释性：可视化不同专家的门控权重。
- 专家数量消融：2 个、3 个、4 个专家对比。
- 插入位置消融：输入端、CEFF 后、多尺度融合后。

## 7. 当前阶段建议

当前最稳妥的实验路线是：

```text
先复现论文 LEVIR-CD baseline
        ↓
确认 ScratchFormer 光学实验可跑通
        ↓
迁移到单极化 SAR baseline
        ↓
加入 SAR-MoE
        ↓
做消融实验和对比实验
        ↓
整理小论文实验结果
```

这条路线可以保证后续 SAR 改进实验有清晰的 baseline，也方便在论文中说明每一步改动的必要性和有效性。
