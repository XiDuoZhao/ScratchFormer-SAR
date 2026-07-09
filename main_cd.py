from argparse import ArgumentParser
import torch
from models.trainer import *
import os

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  # 测试阶段也需要沿用训练时的读图模式，确保 SAR 以单通道方式送入网络
                                  split='test', img_mode=args.img_mode, data_root=args.data_root)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='./scratchformer', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='./vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CDD', type=str)
    parser.add_argument('--data_root', default='', type=str, help='optional custom dataset root directory')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=512, type=int)
    # img_mode=RGB 对应原始光学遥感流程；img_mode=L 用于单极化 SAR 单通道输入
    parser.add_argument('--img_mode', default='RGB', type=str, help='RGB for optical images, L for single-channel SAR')
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    # input_nc 用于显式指定每个时相图像的通道数，单极化 SAR 应设置为 1
    parser.add_argument('--input_nc', default=3, type=int, help='number of channels for each input image')
    parser.add_argument('--use_moe', action='store_true', help='enable SAR-oriented MoE adapter and fusion blocks')
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=bool)
    parser.add_argument('--multi_scale_infer', default=False, type=bool)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])
    parser.add_argument('--net_G', default='ScratchFormer', type=str, help='ScratchFormer')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.00041, type=float)
    parser.add_argument('--max_epochs', default=300, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str, help='linear | step')
    parser.add_argument('--lr_decay_iters', default=[100], type=int)
    
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
