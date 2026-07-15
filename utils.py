import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import cv2

import data_config
from datasets.CD_dataset import CDDataset


def resolve_data_config(data_name, data_root=''):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    # 允许通过命令行直接覆盖数据根目录，方便切换不同 SAR 数据集而不反复改源码
    if data_root:
        dataConfig.root_dir = data_root
    return dataConfig


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset', img_mode='RGB', data_root=''):
    dataConfig = resolve_data_config(data_name, data_root=data_root)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform,
                                 # 将输入模式向下传递到数据集，供 RGB / 单通道 SAR 共用
                                 img_mode=img_mode,
                                 # 仅 RGB 训练阶段开启颜色扰动，避免破坏 SAR 灰度统计特性
                                 random_color_tf=(img_mode == 'RGB' and is_train))
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    data_root = getattr(args, 'data_root', '')
    dataConfig = resolve_data_config(data_name, data_root=data_root)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        # 未显式指定时保持原项目 RGB 行为，指定为 L 时用于单极化 SAR
        img_mode = getattr(args, 'img_mode', 'RGB')
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform,
                                 img_mode=img_mode,
                                 random_color_tf=(img_mode == 'RGB'))
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform,
                                 img_mode=img_mode,
                                 # 验证阶段关闭颜色增强，保证评估稳定
                                 random_color_tf=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def stack_labeled_visualizations(panels):
    """Stack visualizations with a title bar identifying each panel."""
    title_height = 32
    labeled_panels = []
    for title, panel in panels:
        panel = np.asarray(panel)
        if panel.ndim == 2:
            panel = np.repeat(panel[..., None], 3, axis=2)
        title_bar = np.full((title_height, panel.shape[1], 3), 0.08, dtype=np.float32)
        cv2.putText(title_bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (1.0, 1.0, 1.0), 1, cv2.LINE_AA)
        labeled_panels.extend((title_bar, panel))
    return np.concatenate(labeled_panels, axis=0)


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
