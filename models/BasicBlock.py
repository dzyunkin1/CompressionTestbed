import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1) -> None:
        """
        Initializes a BasicBlock instance.

        Parameters:
            input_dim (int): Number of input channels.
            output_dim (int): Number of output channels.
            stride (int): Stride for the first convolutional layer. Defaults to 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(output_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != self.expansion * output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_dim,
                    self.expansion * output_dim,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out
