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
