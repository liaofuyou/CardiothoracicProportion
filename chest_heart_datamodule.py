import os
from glob import glob
from typing import Optional

import albumentations as transforms
from albumentations.core.composition import Compose, OneOf
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from chest_heart_dataset import ChestHeartDataset


class ChestHeartDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "dataset/",
            num_workers: int = 1,
            batch_size: int = 10,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...

        self.prepare_data()
        self.setup()

    @property
    def num_classes(self):
        return 3

    def setup(self, stage: Optional[str] = None):
        # Data loading code
        img_ids = glob(os.path.join(self.data_dir, 'images', '*.jpg'))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        # 分数据
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=41)

        # 数据增强：
        train_transform = self.get_transforms(True)
        val_transform = self.get_transforms(False)

        self.dataset_train = ChestHeartDataset(train_img_ids, self.data_dir, train_transform)
        self.dataset_val = ChestHeartDataset(val_img_ids, self.data_dir, val_transform)

    @staticmethod
    def get_transforms(train=False):
        # 数据增强：

        train_transform = Compose([
            transforms.RandomRotate90(),
            OneOf([
                transforms.HueSaturationValue(),
                transforms.RandomBrightness(),
                transforms.RandomContrast(),
            ], p=1),  # 按照归一化的概率选择执行哪一个
            transforms.Resize(512, 512),
            transforms.Normalize()
        ])

        val_transform = Compose([
            transforms.Resize(512, 512),
            transforms.Normalize()
        ])

        return train_transform if train else val_transform

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader


if __name__ == '__main__':

    dm = ChestHeartDataModule()
    dataloader = dm.train_dataloader()

    for x, y, _ in dataloader:
        print(1, x.shape, y.shape)
        break
