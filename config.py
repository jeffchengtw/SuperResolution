# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import torch
from torch.backends import cudnn as cudnn

torch.manual_seed(0)
device = torch.device("cuda", 0)
cudnn.benchmark = True
upscale_factor = 3
train_mode = 'normal_precision'
mode = "train_srresnet"
exp_name = "train_srresnet_normal_precision"

model_path = f"results/{exp_name}/g-last.pth"
train_image_dir = "data/train"
valid_image_dir = "data/valid"
image_size = 75
batch_size = 128
num_workers = 0
resume = True
strict = False
start_epoch = 0
resume_weight = ""
epochs = 30000
model_lr = 1e-4
model_betas = (0.9, 0.999)
print_frequency = 1000


lr_dir = f"data/test"
sr_dir = f"results/test/{exp_name}"
model_path = f"results/{exp_name}/g-best.pth"
