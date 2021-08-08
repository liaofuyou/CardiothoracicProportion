import albumentations as transforms
import cv2
import matplotlib.pyplot as plt
import torch.utils.data

from chest_heart_datamodule import ChestHeartDataModule
from compute_cardio_thoracic_proportion import compute_cardio_thoracic_proportion
from train import ChestHeartModule


def predict(model, image_path):
    transform = ChestHeartDataModule.get_transforms(False)

    img = cv2.imread(image_path)
    img = transform(image=img)['image']

    model.eval()
    img = torch.unsqueeze(img, dim=0)
    output = model(img)["out"]
    output = torch.argmax(output, dim=1)
    output = output[0, :, :]

    return output.detach().numpy()


def tmp_compute():
    im_path = "dataset/images/ID_0a0adf93f.jpg"
    mask_path = "dataset/masks/ID_0ab3c2234.png"
    image = cv2.imread(im_path)
    segmentation = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 用来展示画图用
    resize = transforms.Resize(512, 512)
    augmented = resize(image=image, mask=segmentation)
    image = augmented['image']
    segmentation = augmented['mask']

    compute_cardio_thoracic_proportion(image, segmentation, im_path)


def cli_main():
    model = ChestHeartModule(num_classes=3)  # .load_from_checkpoint("path/to/checkpoint_file.ckpt")
    out = predict(model, "dataset/images/ID_0a4f3c934.jpg")

    # print(out)
    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    cli_main()
