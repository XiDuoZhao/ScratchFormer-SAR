# ScratchFormer 在 WHU-CD 上的复现记录

## 当前状态

已完成从官方 WHU-CD 原始影像构建 `256 x 256` 图块、300 轮训练和最终测试。训练从随机初始化开始，未使用预训练权重或 SAR MoE 模块。

## 数据流程

原始 WHU-CD 数据包含 2012 年与 2016 年的 RGB 航片及官方变化标签。数据处理脚本将其裁为不重叠的 `256 x 256` 图块，并生成项目使用的 `A/`、`B/`、`label/`、`list/` 目录结构。

| 划分 | 图块数 |
| --- | ---: |
| train | 5947 |
| val | 743 |
| test | 744 |

图块数量和输入分辨率与论文一致。论文及官方 ScratchFormer 仓库未提供逐图块的公开划分名单；本次使用固定随机种子从官方原始数据生成相同数量的互不重叠划分。因此结果可用于流程与方法复现，但测试样本归属无法证明与作者实验逐文件一致。

## 本次运行配置

```text
project_name: whu_scratchformer_300
data_root: ./datasets/CD/WHU-CD-256-patches
data_name: WHU
img_size: 256
img_mode: RGB
input_nc: 3
batch_size: 16
optimizer: AdamW
weight_decay: 0.01
betas: (0.9, 0.999)
learning_rate: 0.00041
max_epochs: 300
lr_policy: linear
use_moe: false
pretrain: none
```

训练耗时为 17 小时 15 分 53 秒，最佳验证检查点出现在第 278 轮。

## 结果对比

| 指标 | 论文结果 | 复现结果 | 差值（复现结果 - 论文结果） |
| --- | ---: | ---: | ---: |
| 变化类 F1（`F1_1`） | 91.87% | 89.62% | -2.25 个百分点 |
| 总体精度（`acc`） | 99.37% | 99.12% | -0.25 个百分点 |
| 变化类 IoU（`iou_1`） | 84.97% | 81.18% | -3.79 个百分点 |

测试结果来自 `checkpoints/whu_scratchformer_300/scores_dict.npy` 和最终评估日志。请使用变化类 `F1_1` 与 `iou_1` 对比论文，不能用平均 F1（`mf1`）替代论文的变化类 F1。

## 复现结论

在官方 WHU-CD 原始数据、论文规定的 256 分辨率、图块数量及训练超参数下，本次实验完成了 ScratchFormer 的 WHU-CD 复现。结果的 OA 与论文接近，变化类 F1 和 IoU 低于论文。除单张 RTX 4090 D 与论文 4 张 A100 的硬件差异外，重新生成的随机图块划分也可能带来性能差异；若需要严格的基准数值对比，应取得作者使用的官方 256 图块名单并固定随机种子进行多次重复实验。

与 `reports/levir_scratchformer_300.md` 记录的 LEVIR-CD 结果合并，本项目已完成 LEVIR-CD 与 WHU-CD 两个数据集的 ScratchFormer 复现。
