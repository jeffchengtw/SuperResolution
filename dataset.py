import io
import os

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
import preprocess

class ImageDataset(Dataset):
    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
            ])
        else:
            self.hr_transforms = transforms.CenterCrop(image_size)

        self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC, antialias=True)

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        image = Image.open(self.filenames[batch_index])

        # Transform image
        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = preprocess.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = preprocess.image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.filenames)

