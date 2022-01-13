from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode
from PIL import Image
import matplotlib.pyplot as plt
import torch
import preprocess
image_size = 75
upscale_factor=3

image = Image.open('data/train/12003.png')

pool_transform = transforms.Compose([transforms.CenterCrop(image_size),
                            transforms.ToTensor()]) 
lr_transforms = transforms.Resize((25,25), interpolation=IMode.BICUBIC, antialias=True)

lr_image = lr_transforms(image)
lr_tensor = preprocess.image2tensor(lr_image, range_norm=False, half=False)
lr_img = preprocess.tensor2image(lr_tensor, range_norm=False, half=False)
sourceImg = pool_transform(image)
cropImg = torch.nn.MaxPool2d(3)(sourceImg)

res = preprocess.tensor2image(cropImg, range_norm=False, half=False)

plt.subplot(1, 2, 1)
plt.title('Bicubic')
plt.imshow(lr_img)

plt.subplot(1, 2, 2)
plt.title('MaxPool2d')
plt.imshow(res)
plt.show()