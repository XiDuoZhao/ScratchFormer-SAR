from argparse import ArgumentParser
import torch
from models.evaluator import *

print(torch.cuda.is_available())


"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='scratchformer', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='./checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CDD', type=str)
    parser.add_argument('--data_root', default='', type=str, help='optional custom dataset root directory')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=512, type=int)
    # 与训练阶段保持一致：RGB 用于光学数据，L 用于单极化 SAR
    parser.add_argument('--img_mode', default='RGB', type=str, help='RGB for optical images, L for single-channel SAR')

    # model
    # 评估时也必须指定与训练相同的输入通道数，单极化 SAR 通常为 1
    parser.add_argument('--input_nc', default=3, type=int, help='number of channels for each input image')
    parser.add_argument('--use_moe', action='store_true', help='enable SAR-oriented MoE adapter and fusion blocks')
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='ScratchFormer', type=str, help='ScratchFormer')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  # 评估数据读取模式需要与训练完全一致
                                  split=args.split, img_mode=args.img_mode, data_root=args.data_root)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()
