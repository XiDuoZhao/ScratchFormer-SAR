# ScratchFormer 在 LEVIR-CD 上的复现记录

## 当前状态

已完成 300 轮训练和测试集评估。最佳验证平均 F1（`mf1`）为 `0.91609`，出现在第 295 轮。

## 论文参考结果

- 论文：[Remote Sensing Change Detection With Transformers Trained from Scratch](https://arxiv.org/abs/2304.06710)
- 数据集：LEVIR-CD
- 训练设置：训练 300 轮，使用 AdamW 优化器，初始学习率为 `4.1e-4`，批大小为 16。
- 论文报告指标：变化类 F1、总体精度（OA）和变化类 IoU。

| 指标 | 论文结果 | 复现结果 | 差值（复现结果 - 论文结果） |
| --- | ---: | ---: | ---: |
| 变化类 F1（`F1_1`） | 91.68% | 83.02% | -8.66 个百分点 |
| 总体精度（`acc`） | 99.16% | 98.34% | -0.82 个百分点 |
| 变化类 IoU（`iou_1`） | 84.63% | 70.98% | -13.65 个百分点 |

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

论文直接以 `256 x 256` 图像对输入模型。论文将 637 张 `1024 x 1024` 原始图像裁成不重叠的 `256 x 256` 图块，训练、验证、测试集分别包含 `7120`、`1024`、`2048` 个样本。当前实验使用 `445/64/128` 张原始图像，并在读取时缩放到 `256 x 256`，因此数据样本数量、空间内容和预处理方式均与论文不一致。

## 复现结论

本次运行完成了训练、验证和测试流程，但没有达到论文报告的 LEVIR-CD 结果。变化类 F1 比论文低 `8.66` 个百分点，变化类 IoU 低 `13.65` 个百分点；OA 差距较小，但该指标受未变化像素占比影响较大，不能据此认定变化检测性能得到复现。

该结论目前应视为“同一训练轮数下的运行结果”，而不是严格的论文复现结论。`--img_size 256` 与论文模型输入分辨率一致；已确认的主要差异是数据没有按论文方式裁成非重叠的 `256 x 256` 图块。此外，单卡 RTX 4090 D 与论文使用的 4 张 A100、未固定随机种子以及本项目已有的代码修改，都可能造成性能差异。后续应先对齐论文的图块数据集与划分，再用多个随机种子重复实验。
