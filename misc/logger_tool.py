import sys
import time

ARG_NAME_MAP = {
    'gpu_ids': 'GPU编号',
    'project_name': '项目名称',
    'checkpoint_root': '检查点根目录',
    'checkpoints_root': '检查点根目录',
    'checkpoint_dir': '检查点目录',
    'vis_root': '可视化根目录',
    'vis_dir': '可视化目录',
    'mode': '运行模式',
    'num_workers': '数据加载线程数',
    'seed': '随机种子',
    'dataset': '数据集类型',
    'data_name': '数据集名称',
    'data_root': '数据集路径',
    'batch_size': '批大小',
    'split': '训练划分',
    'split_val': '验证划分',
    'img_size': '输入图像尺寸',
    'img_mode': '图像读取模式',
    'shuffle_AB': '是否打乱AB时相',
    'scene_eval': '是否启用完整场景评估',
    'scene_metadata': '场景坐标元数据',
    'scene_eval_output': '完整场景评估输出目录',
    'val_eval_mode': '验证评估模式',
    'input_nc': '输入通道数',
    'n_class': '类别数',
    'embed_dim': '嵌入维度',
    'pretrain': '预训练权重路径',
    'multi_scale_train': '是否多尺度训练',
    'multi_scale_infer': '是否多尺度推理',
    'multi_pred_weights': '多尺度预测权重',
    'net_G': '生成网络名称',
    'loss': '损失函数',
    'selection_metric': '最佳模型选择指标',
    'optimizer': '优化器',
    'lr': '学习率',
    'max_epochs': '训练轮次',
    'lr_policy': '学习率策略',
    'lr_decay_iters': '学习率衰减步',
    'checkpoint_name': '检查点文件名',
    'print_models': '是否打印模型',
    'use_moe': '是否启用MoE',
}


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            key_name = ARG_NAME_MAP.get(k, k)
            message += '%s: %s ' % (key_name, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out
