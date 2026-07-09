
import os
from PIL import Image
import numpy as np

from torch.utils import data

from datasets.data_utils import CDDataAugmentation


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.jpg' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True,
                 img_mode='RGB', random_color_tf=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  #train | train_aug | val
        # 读取模式由上层指定：光学图像通常为 RGB，单极化 SAR 使用 L
        self.img_mode = img_mode
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                # 颜色增强仅对 RGB 数据开启，SAR 数据应关闭
                random_color_tf=random_color_tf
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                is_evaluation=True
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        # 统一通过 img_mode 控制输入通道数，避免代码默认把单通道 SAR 强制转成 RGB
        img = np.asarray(Image.open(A_path).convert(self.img_mode))
        img_B = np.asarray(Image.open(B_path).convert(self.img_mode))

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, img_mode='RGB', random_color_tf=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor, img_mode=img_mode,
                                        random_color_tf=random_color_tf)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        # 标签任务主链路同样使用可配置读图模式，以支持单通道 SAR 变化检测
        img = np.asarray(Image.open(A_path).convert(self.img_mode))
        img_B = np.asarray(Image.open(B_path).convert(self.img_mode))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)
        # if you are getting error because of dim mismatch ad [:,:,0] at the end
        
        if self.label_transform == 'norm':
            label = label // 255
        
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        # print(label.max())
        
        return {'name': name, 'A': img, 'B': img_B, 'L': label}
