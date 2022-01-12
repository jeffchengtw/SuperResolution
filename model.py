import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

class ResidualConvBlock(nn.Module):
 
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)
        out = torch.add(out, identity)

        return out

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # First conv layer.
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # Features trunk blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer.
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64 *3**2, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(3),
            nn.PReLU(),
        )
        
        # Output layer.
        self.conv_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)