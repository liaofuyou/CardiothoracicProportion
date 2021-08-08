import os

import cv2
import torch
from torch.utils.data import Dataset


class ChestHeartDataset(Dataset):
    def __init__(self, img_ids, data_dir, transform=None):
        self.img_ids = img_ids
        self.transform = transform

        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.img_ext = ".jpg"
        self.mask_ext = ".png"
        self.num_classes = 3

        self.fucking = True

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]

        # 原图
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # 标签
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 图片
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float)

        # mask
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask, {'img_id': img_id}

        # if self.fucking:
        #     import matplotlib.pyplot as plt
        #     print(img.shape, mask.shape)
        #     print("img_id", img_id)
        #     # plt.imshow(img)
        #     plt.imshow(mask)
        #     plt.show()
        #     self.fucking = False

