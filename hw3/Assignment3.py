import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):
        rgb_img = cv.cvtColor(torch_img, cv.COLOR_BGR2RGB)
        return torch.from_numpy(rgb_img).to(torch.float32)

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0
        return bright_img

    def saturation_arithmetic(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        wide_img = torch.from_numpy(rgb_img).to(torch.int16)
        saturated_img = torch.clamp(wide_img + 100, 0, 255).to(torch.uint8)
        return saturated_img

    def add_noise(self, torch_img):
        noise = torch.randn_like(torch_img) * 100.0
        noisy_img = torch_img + noise

        noisy_img = torch.clamp(noisy_img / 255.0, 0.0, 1.0)
        return noisy_img.to(torch.float32)

    def normalization_image(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb_img).to(torch.float64)

        channel_mean = image.mean(dim=(0, 1))
        channel_std = image.std(dim=(0, 1))
        image_norm = (image - channel_mean) / channel_std
        return image_norm

    def Imagenet_norm(self, img):
        means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        deviations = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb_img).to(torch.float64) / 255.0
        ImageNet_norm = torch.clamp((image - means) / deviations, 0.0, 1.0)
        return ImageNet_norm

    def dimension_rearrange(self, img):
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb_img).to(torch.float32)
        rearrange = image.permute(2, 0, 1).unsqueeze(0)
        return rearrange

    def stride(self, img):
        image = torch.from_numpy(img).to(torch.float32)
        image = image.unsqueeze(0).unsqueeze(0)
        scharr_x = torch.tensor(
            [[-3.0, 0.0, 3.0],
             [-10.0, 0.0, 10.0],
             [-3.0, 0.0, 3.0]],
            dtype=torch.float32
        ).flip(0, 1).unsqueeze(0).unsqueeze(0)
        stride_imgage = torch.nn.functional.conv2d(
            image, scharr_x, stride=2, padding=1
        ).squeeze(0).squeeze(0)
        return stride_imgage


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    img_cat = cv.imread("cat_eye.jpg", cv.IMREAD_GRAYSCALE)
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    stride_imgage = assign.stride(img_cat)
