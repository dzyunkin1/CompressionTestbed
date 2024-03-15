import torch
import torch.nn as nn
from models.BasicBlock import BasicBlock
from typing import List


class ResNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        num_blocks: List[int],
        num_classes: int = 10,
        in_channels: int = 1,
    ) -> None:
        """
        Initializes a ResNet instance.

        Parameters:
            block (BasicBlock): The block class to use (e.g., BasicBlock).
            num_blocks (List[int]): List containing the number of blocks for each layer.
            num_classes (int): Number of output classes. Defaults to 10.
            in_channels (int): Number of input channels. Defaults to 1 for grayscale images.
        """
        super(ResNet, self).__init__()
        self.in_planes: int = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: BasicBlock, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """
        Helper method to create a layer of blocks.

        Parameters:
            block (BasicBlock): The block class to use (e.g., BasicBlock).
            planes (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Sequential layer of blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
