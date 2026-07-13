# ScratchFormer 在 LEVIR-CD 上的复现记录

## 当前状态

已完成严格数据流程下的 300 轮训练和测试评估。训练在第 172 轮结束后因终端关闭中断一次，随后从 `last_ckpt.pt` 的第 173 轮继续；模型、优化器和学习率调度器状态均从检查点恢复。

## 论文参考结果

- 论文：[Remote Sensing Change Detection With Transformers Trained from Scratch](https://arxiv.org/abs/2304.06710)
- 数据集：LEVIR-CD
- 训练设置：训练 300 轮，使用 AdamW 优化器，初始学习率为 `4.1e-4`，批大小为 16。
- 论文报告指标：变化类 F1、总体精度（OA）和变化类 IoU。

| 指标 | 论文结果 | 复现结果 | 差值（复现结果 - 论文结果） |
| --- | ---: | ---: | ---: |
| 变化类 F1（`F1_1`） | 91.68% | 91.17% | -0.51 个百分点 |
| 总体精度（`acc`） | 99.16% | 99.12% | -0.04 个百分点 |
| 变化类 IoU（`iou_1`） | 84.63% | 83.77% | -0.86 个百分点 |

## 本次运行配置

```text
project_name: levir_scratchformer_300
data_root: ./datasets/CD/LEVIR-CD-256-patches
data_name: LEVIR
img_mode: RGB
input_nc: 3
batch_size: 16
optimizer: AdamW
learning_rate: 0.00041
max_epochs: 300
use_moe: false
```

## 结果来源

本次测试指标读取自：

```text
checkpoints/levir_scratchformer_300/scores_dict.npy
checkpoints/levir_scratchformer_300/Logs/eval_*.txt
```

请使用变化类的 `F1_1` 和 `iou_1` 与论文比较。项目日志中的平均 F1（`mf1`）不能直接与论文报告的变化类 F1 比较。

## 可比性检查

论文直接以 `256 x 256` 图像对输入模型。论文将 637 张 `1024 x 1024` 原始图像裁成不重叠的 `256 x 256` 图块，训练、验证、测试集分别包含 `7120`、`1024`、`2048` 个样本。本次正式实验使用同一批原图生成了对应数量的非重叠图块，数据划分、输入分辨率与论文保持一致。此前使用 `445/64/128` 张整图缩放的实验已废弃。

## 复现结论

此前未按裁块流程进行的结果已废弃，不纳入论文对比。本次训练使用 `datasets/CD/LEVIR-CD-256-patches/`，其训练、验证、测试图块数与论文的 `7120/1024/2048` 一致。

本次复现结果与论文高度接近：变化类 F1 低 `0.51` 个百分点，变化类 IoU 低 `0.86` 个百分点，OA 仅低 `0.04` 个百分点。因此，在当前单张 RTX 4090 D 硬件和一次随机训练的条件下，可以认为 ScratchFormer 在 LEVIR-CD 上的论文结果已成功复现。为评估结果波动，后续可固定随机种子并重复多次训练，报告均值与标准差。
