import ssl

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import IoU
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from chest_heart_datamodule import ChestHeartDataModule

ssl._create_default_https_context = ssl._create_unverified_context


class ChestHeartModule(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.net = deeplabv3_mobilenet_v3_large(pretrained=False,
                                                num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.metric_train = IoU(num_classes=num_classes)
        self.metric_val = IoU(num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        # 过模型
        output = self(x)["out"]
        loss = self.criterion(output, y)

        # "准确率"
        acc = self.metric_train(output, y)
        print("train acc", acc)
        # self.log("Acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        # 过模型
        output = self(x)["out"]
        loss = self.criterion(output, y)

        # "准确率"
        acc = self.metric_val(output, y)
        print("val acc", acc)
        # self.log("Acc", acc, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        opt = Adam(self.net.parameters(), lr=self.learning_rate)
        sch = CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


def cli_main():
    # tb_logger = pl_loggers.TensorBoardLogger("logs/")

    # init model
    model = ChestHeartModule(num_classes=3)

    # init data
    dm = ChestHeartDataModule()

    # train
    trainer = pl.Trainer(max_epochs=10,
                         val_check_interval=0.25,
                         # logger=tb_logger,
                         fast_dev_run=True)
    trainer.fit(model, dm)


if __name__ == '__main__':
    cli_main()
