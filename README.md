# HW4 Image super resolution

# Introduction
The highly challenging task of estimating a highresolution (HR) image from its low-resolution (LR) counterpart is referred to as super-resolution (SR). SR
received substantial attention from within the computer vision research community and has a wide range of applications.
![image](https://github.com/jeffchengtw/HW4/blob/main/fig/breif.PNG)

# ENV
python 3.6.10 
prtorch 1.10.1
cuda 10.2
cudnn 7.0.4

# Data Folder
```
├── train
│   ├── 2092.png
│   ├── 8049.png
├   ├──  ...
├── valid                   
│   ├── 24004.png
│   ├── 65019.png      
├   ├──  ...             
└── test
│   ├── 00.png
│   ├── 01.png      
├   ├──  ...     
```
# Training
```
python trian.py
```
# Test
```
python infer.py
```
# low-resolution image preprocessing
Using random cropping and converting to tensor, I randomly crop the image to a size of (75 x 75) and use it as a high-pixel image, and then resize it to (25 x 25) by 
Bicubic as a low-resolution image. I have also tried downsampling with maxpooling, but the training results is not good.<br>
![image](https://github.com/jeffchengtw/HW4/blob/main/fig/preprocess.PNG)

# Architecture
1) Residual Blocks<br>
Based on this article[1]. There is a change: use Parametric ReLU instead of 
ReLU to help it adaptively learn some of the negative coefficients.<br>
2) Upsampling Block<br>
This CNN framework refers to this paper[2], which is used to extract features 
and perform super-resolution processing on low-resolution images. However, 
the previous super-resolution method needs to upsample the low-resolution 
image to the size of the high-resolution image, and then use the filter to 
perform bilinear interpolation, which is easy to fall into local optimum and 
requires a large amount of calculation. This paper proposes a sub-pixel 
convolution layer that learns a set of upsampling filters instead of a single 
upsampling filter to obtain high-resolution images from low-resolution 
feature maps.<br>
![image](https://github.com/jeffchengtw/HW4/blob/main/fig/structure.PNG)
