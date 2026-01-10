import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding = 1, groups = in_channels, bias = False)
        self.separable = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace = True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu6(out)
        out = self.separable(out)
        out = self.bn2(out)
        out = self.relu6(out)

        return out

class MobileNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(MobileNet, self).__init__()

        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace = True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 16, 1),
            DepthwiseSeparableConv(16, 32, 1),
            DepthwiseSeparableConv(32, 64, 2),
            DepthwiseSeparableConv(64, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x