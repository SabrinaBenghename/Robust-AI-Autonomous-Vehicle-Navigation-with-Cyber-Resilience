import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class WRFEB(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = DepthwiseSeparableConv(
            channels,
            channels
        )

        self.conv2 = DepthwiseSeparableConv(
            channels,
            channels
        )

        self.attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        attention = self.attention(x)

        x = x * attention

        x = x + identity

        return x