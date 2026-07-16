# 小场景单极化 SAR 变化检测数据集处理总结

## 1. 处理目标

本次处理将收集到的小场景双时相 SAR 数据统一为 ScratchFormer 可读取的
`A/B/label/list` 格式，并按数据域构建四折留一域测试协议。处理过程中保留了
场景来源、尺寸、标签统计、图块坐标和划分区域等元数据，同时加入自动泄漏检查。

当前结果用于小场景、单通道 SAR 变化检测实验。数据集处理已经完成，但正式评估
仍需根据坐标元数据将测试图块预测拼接回完整场景，再计算场景级指标。

## 2. 数据域与场景

标准化后共有 4 个数据域、7 个场景：

| 数据域 | 场景 | 尺寸（宽×高） | 变化像素 | 变化比例 | 外部测试折 |
|---|---|---:|---:|---:|---:|
| Ottawa | Ottawa | 290×350 | 16,049 | 15.8118% | Fold 1 |
| Bern | Bern | 301×301 | 1,155 | 1.2748% | Fold 2 |
| SanFrancisco | SanFrancisco | 256×256 | 4,685 | 7.1487% | Fold 3 |
| YellowRiver | Yellow-A | 257×289 | 13,432 | 18.0846% | Fold 4 |
| YellowRiver | Yellow-B | 450×280 | 1,348 | 1.0698% | Fold 4 |
| YellowRiver | Yellow-C | 291×444 | 4,255 | 3.2932% | Fold 4 |
| YellowRiver | Yellow-D | 306×291 | 5,270 | 5.9183% | Fold 4 |

Yellow-A 至 Yellow-D 来自同一 YellowRiver 数据域，因此四个场景始终作为整体进入
训练域或测试域，不允许被拆到同一折的训练集和测试集中。YellowRiver 文件与
[INLPG 数据目录](https://github.com/yulisun/INLPG/tree/master/datasets)中的对应文件
经过像素级比对一致。Ottawa、Bern 和 SanFrancisco 当前记录为本地收集版本，论文中
使用这些数据时仍需补充其原始发布论文、传感器、极化方式和获取链接。

完整场景清单保存在：

```text
datasets/CD/SAR-CD-128/domains/manifest.csv
```

## 3. 场景和数据域标准化

标准化脚本为：

```text
prepare_sar_domains.py
```

脚本执行以下处理：

1. 根据预先核对的文件名建立 A 时相、B 时相和真值标签的对应关系。
2. 将图像统一读取为二维灰度数组。若源文件以三通道保存，则要求三个通道完全一致，
   防止将真实 RGB 图像误当作单通道 SAR。
3. 检查同一场景的 A、B 和标签尺寸完全一致。
4. 将 A、B 保存为 8 位单通道 PNG。
5. 使用阈值 `128` 将标签二值化，并保存为只包含 `0` 和 `255` 的单通道 PNG。
6. 为每个源文件计算 SHA-256，用于数据来源和内容一致性追踪。
7. 统计每个场景的尺寸、变化像素数、总像素数和变化比例。

标准化场景目录如下：

```text
datasets/CD/SAR-CD-128/domains/
├── manifest.csv
├── Ottawa/
│   ├── domain.json
│   └── scenes/Ottawa/{A.png,B.png,label.png,metadata.json}
├── Bern/
├── SanFrancisco/
└── YellowRiver/
    └── scenes/{Yellow-A,Yellow-B,Yellow-C,Yellow-D}/
```

`domain.json` 记录该域包含的场景和外部测试折，`metadata.json` 记录单个场景的尺寸、
标签统计、源文件名、文件哈希和来源说明。

## 4. 四折跨域实验协议

四折采用 leave-one-domain-out（留一数据域测试）协议：

| Fold | 训练与验证域 | 完全留出的测试域 | Train 图块 | Val 图块 | Test 图块 |
|---|---|---|---:|---:|---:|
| Fold 1 | Bern、SanFrancisco、YellowRiver | Ottawa | 71 | 25 | 20 |
| Fold 2 | Ottawa、SanFrancisco、YellowRiver | Bern | 71 | 25 | 16 |
| Fold 3 | Ottawa、Bern、YellowRiver | SanFrancisco | 80 | 28 | 9 |
| Fold 4 | Ottawa、Bern、SanFrancisco | YellowRiver | 27 | 9 | 84 |

图块参数：

- 图块尺寸：`128×128`
- 滑窗步长：`64`
- 输入通道：单通道灰度
- 标签值：磁盘中为 `0/255`，加载时通过 `label_transform=norm` 转换为 `0/1`

训练域中的每个场景先按空间位置划分为互不相交的 `2×2` 四个象限。选择一个包含
变化像素、且变化比例最接近完整场景的象限作为验证区域，其余三个象限作为训练区域。
图块必须完整位于所属区域内，因此训练图块与验证图块不会跨越区域边界。

测试域不参与训练或模型选择。测试时在完整场景上按步长 64 滑窗，右边缘和下边缘
不足一个常规步长时，将最后一个窗口对齐到场景边缘，保证每个原始像素至少被一个
测试图块覆盖。

Fold 4 的训练图块明显少于其他折，是因为 YellowRiver 含 4 个场景并被整体留作测试，
属于严格跨域划分产生的数据不平衡，不是图块遗漏。

## 5. 可训练目录与坐标元数据

四折生成脚本为 `prepare_sar_folds.py`，输出目录为：

```text
datasets/CD/SAR-CD-128/folds/
├── folds_manifest.json
├── fold_1/
│   ├── A/
│   ├── B/
│   ├── label/
│   ├── list/{train.txt,val.txt,test.txt}
│   ├── metadata.csv
│   └── protocol.json
├── fold_2/
├── fold_3/
└── fold_4/
```

`A/`、`B/` 和 `label/` 中同名文件组成一个训练样本。`list/` 中的三个清单可以被
现有 `CDDataset` 直接读取。

每个 `metadata.csv` 包含以下关键信息：

- 图块名、Fold、split、数据域和场景名；
- 图块在原始场景中的 `left/top/right/bottom` 坐标；
- 训练或验证区域在原始场景中的边界；
- 原始场景宽度和高度；
- 图块变化像素数、总像素数和变化比例。

这些坐标既用于检查训练/验证空间隔离，也用于后续将重叠测试窗口的概率或 logits
累加并平均，恢复完整场景预测图。

## 6. 泄漏与完整性检查

独立检查脚本 `verify_sar_folds.py` 已对全部四折执行并通过，检查内容包括：

1. 必须存在且仅存在 Fold 1 至 Fold 4。
2. `train.txt`、`val.txt`、`test.txt` 不含重复文件名，且与 `metadata.csv` 完全一致。
3. 每折测试集只包含协议指定的留出域，测试域不得进入训练集或验证集。
4. 训练域集合和样本数量必须与 `protocol.json` 一致。
5. A/B 必须为 `128×128` 单通道图像，标签只能包含 `0/255`。
6. 对 A、B、标签联合计算内容哈希，禁止相同内容跨 train/val/test 出现。
7. 同一场景的训练图块与验证图块在原图坐标上不得相交。
8. 留出测试场景的全部像素必须被测试窗口覆盖。

此外，已使用项目现有 `CDDataset` 对 `4 folds × 3 splits` 进行读取测试。A/B 张量尺寸
均为 `1×128×128`，标签张量为 `1×128×128`，标签转换结果只包含 `0/1`。

## 7. 复现数据处理

从原始文件重新生成标准化数据域：

```bash
.conda/bin/python prepare_sar_domains.py \
  --source datasets/CD/sar \
  --output datasets/CD/SAR-CD-128/domains
```

从标准化数据域生成四折图块：

```bash
.conda/bin/python prepare_sar_folds.py
```

目标目录已存在并确认需要重新生成时：

```bash
.conda/bin/python prepare_sar_folds.py --overwrite
```

单独执行完整性和泄漏检查：

```bash
.conda/bin/python verify_sar_folds.py \
  --folds-root datasets/CD/SAR-CD-128/folds
```

训练时每折应分别指定对应的数据根目录，并使用以下 SAR 输入参数：

```text
--data_name SAR
--data_root ./datasets/CD/SAR-CD-128/folds/fold_N
--img_size 128
--img_mode L
--input_nc 1
```

四折必须分别训练独立模型，禁止在一个 Fold 上训练后直接将其权重作为另一个 Fold 的
正式结果。模型选择只能使用该 Fold 的验证集，留出测试域只能用于最终测试。

## 8. 统一类别不平衡策略

四折训练集的类别统计如下。正样本图块指至少包含一个变化像素的图块：

| Fold | 训练图块 | 正样本图块 | 纯背景图块 | 变化像素比例 | 背景权重 | 变化权重 |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 | 71 | 46 | 25 | 4.8440% | 0.368176 | 1.631824 |
| Fold 2 | 71 | 54 | 17 | 7.3959% | 0.440674 | 1.559326 |
| Fold 3 | 80 | 55 | 25 | 6.5461% | 0.418550 | 1.581450 |
| Fold 4 | 27 | 19 | 8 | 9.0895% | 0.480475 | 1.519525 |

含变化图块占每折训练集的 64.8% 至 76.1%，因此主要问题是图块内部的像素级不平衡，
并非缺少含变化图块。统一策略确定为：

1. 不进行正样本图块过采样。每个 epoch 使用全部训练图块一次并随机打乱，避免重复放大
   高度重叠的小场景样本。
2. 类别频率只扫描当前 Fold 的 `train.txt` 及对应标签，不读取验证集或测试集标签，也不
   使用随机增强后的标签统计。
3. 类别权重使用平方根逆频率并归一化到均值为 1：
   `w_c = (1 / sqrt(f_c)) / mean(1 / sqrt(f))`。相比直接逆频率，该权重不容易过度放大
   少量变化标注误差。
4. 主实验统一使用 `0.5 × Weighted Cross Entropy + 0.5 × Foreground Soft Dice`。
5. 所有 ScratchFormer-SAR、MoE 和消融模型都使用相同损失；最佳检查点统一按验证集的
   变化类 `F1_1` 选择，而不是容易被背景类别主导的 Accuracy。
6. 验证集和测试集保持原始自然分布，不进行过采样、类别重加权或阈值调优。

训练命令中应统一加入：

```text
--loss ce_dice
--selection_metric F1_1
```

训练日志和检查点会保存当前 Fold 的训练像素计数、实际类别权重、损失名称和选模指标。
为了说明类别不平衡处理的影响，可以额外报告一次原始 `--loss ce` 结果作为损失消融，
但 MoE 与非 MoE 的主对比必须共同使用 `ce_dice`，不能改变其中一方的损失设置。

## 9. 可复现随机种子与验证区域评估

命令行参数 `--seed` 同时控制 Python、NumPy、PyTorch、全部 CUDA 设备、DataLoader
generator 和 worker 随机状态，并关闭 cuDNN benchmark、启用确定性 cuDNN 算法。
训练检查点还会保存这些随机状态；从 `last_ckpt.pt` 恢复时，同时恢复训练和验证
DataLoader 状态，使继续训练的样本顺序与随机增强序列保持连续。

训练 DataLoader 使用 `shuffle=True`，验证和测试 DataLoader 固定使用 `shuffle=False`。
训练与验证使用相互独立的 generator，验证过程不会推进训练集的随机状态。

`--val_eval_mode` 支持以下模式：

- `auto`：默认模式；数据根目录存在 `metadata.csv` 时使用空间拼接，否则使用图块指标；
- `scene`：强制使用坐标拼接，缺少元数据时直接报错；
- `patch`：保留原始图块级验证方式。

SAR 四折目录均包含坐标元数据，因此默认 `auto` 会在每轮验证时按 `split=val` 恢复每个
场景的验证象限。重叠窗口的 softmax 概率先求平均，再计算场景级指标。指定
`--selection_metric F1_1` 时，先在每个数据域内汇总验证场景，再对训练域等权平均，
最佳检查点实际依据 `domain_macro_f1` 选择。这样既避免重叠像素在图块级混淆矩阵中
被重复计权，也避免拥有四个场景的 YellowRiver 在选模时获得过高权重。

每轮验证只在内存中完成拼接；只有产生新的最佳检查点时，才将对应验证区域保存到：

```text
checkpoints/<project_name>/validation_scene_evaluation/
```

正式实验应固定三个随机种子，例如 `0/1/2`，并将 Fold 和 seed 同时写入项目名：

```text
sar_baseline_fold1_seed0
sar_baseline_fold1_seed1
sar_baseline_fold1_seed2
```

不同种子必须使用独立项目目录，不能让一个种子的实验自动加载另一个种子的
`last_ckpt.pt`。主实验命令统一包含：

```text
--seed 0
--val_eval_mode auto
--loss ce_dice
--selection_metric F1_1
```

## 10. 完整场景拼接与评估

完整场景评估器已经在 `misc/scene_evaluation.py` 中实现，并接入
`models/evaluator.py`。评估过程如下：

1. 读取当前 Fold 的 `metadata.csv`，只使用其中 `split=test` 的记录。
2. 对每个测试图块的模型 logits 计算 softmax 概率。
3. 按 `domain/scene` 和原图坐标累加概率，重叠位置除以覆盖次数得到平均概率。
4. 对平均概率执行 argmax，恢复完整场景二值预测。
5. 同时恢复原尺寸 A、B 和真值图，并检查重叠标签是否一致、是否存在未覆盖像素。
6. 计算逐场景、逐域和像素汇总指标。

已有最佳检查点时，可以只运行测试与整图评估，不重新训练：

```bash
.conda/bin/python main_cd.py \
  --mode test \
  --gpu_ids 0 \
  --data_name SAR \
  --data_root ./datasets/CD/SAR-CD-128/folds/fold_1 \
  --img_size 128 \
  --img_mode L \
  --input_nc 1 \
  --batch_size 16 \
  --seed 0 \
  --embed_dim 256 \
  --n_class 2 \
  --project_name sar_fold1_baseline \
  --checkpoint_name best_ckpt.pt \
  --scene_eval
```

默认输出到：

```text
checkpoints/sar_fold1_baseline/scene_evaluation/
├── scene_metrics.csv
├── summary.json
└── <domain>/<scene>/
    ├── time1.png
    ├── time2.png
    ├── ground_truth.png
    ├── prediction.png
    ├── change_probability.png
    ├── overview.png
    └── metrics.json
```

其中 `scene_metrics.csv` 保存每个场景的 TN、FP、FN、TP、Accuracy、Precision、Recall、
Specificity、F1、IoU、Kappa、MCC、mIoU 和 mF1。`summary.json` 同时提供：

- `scene_macro_mean`：每个场景等权重的宏平均，适合作为小场景跨域实验主结果；
- `domain_macro_mean`：先汇总域内场景、再对数据域等权平均，用于跨域验证选模；
- `pixel_global`：合并全部场景混淆矩阵后的像素汇总结果；
- `domains`：逐数据域的场景宏平均和像素汇总结果。

已使用人工构造的重叠窗口和四折真实图块执行理想预测回放。Fold 1 至 Fold 4 均能
恢复正确数量和尺寸的完整场景，理想预测的 F1 和 IoU 均为 1。该测试只验证拼接和
指标代码正确，不代表模型已经获得相应实验结果。

## 11. 当前限制与下一步

1. 当前数据规模很小，且不同域的变化比例差异明显，不能只报告总体 Accuracy。
2. 建议报告每场景 Precision、Recall、F1、IoU、Kappa 和 MCC，再计算域级及四折宏平均。
3. 每个 Fold 建议至少运行 3 个随机种子，报告均值和标准差。
4. 完整场景评估已经实现，但仍需通过 Fold 1 冒烟训练验证真实模型检查点的端到端流程。
5. Ottawa、Bern、SanFrancisco 的原始出处与传感器信息仍需补齐，避免论文数据来源说明
   不完整。
6. `datasets/CD/` 已被 `.gitignore` 忽略，GitHub 中应保存处理脚本、协议说明和实验结果，
   不直接上传数据文件。
