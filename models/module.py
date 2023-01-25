import torch
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv3x3 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv3x3(x)
        attention = self.glob_avg_pool(feature)
        attention = self.conv1x1(attention)
        attention = self.bn(attention)
        attention = self.sigmoid(attention)
        output = torch.mul(feature, attention)
        return output


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv_cat = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_spatial, feature_context):
        feature_cat = torch.cat([feature_spatial, feature_context], dim=1)
        feature = self.conv_cat(feature_cat)
        attention = self.glob_avg_pool(feature)
        attention = self.relu(self.conv1(attention))
        attention = self.sigmoid(self.conv2(attention))
        feature_attention = torch.mul(feature, attention)
        output = feature_attention + feature
        return output


class AddBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, stride):
        super(AddBottleneck, self).__init__()
        self.blocks = nn.ModuleList([])
        self.stride = stride

        if stride == 2:
            self.layer = nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1,
                          groups=out_channels // 2, bias=False),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        for i in range(num_block):
            if i == 0:
                self.blocks.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0))
            elif i == 1 and num_block == 2:
                self.blocks.append(ConvBNReLU(out_channels // 2, out_channels // 2))
            elif i == 1 and num_block > 2:
                self.blocks.append(ConvBNReLU(out_channels // 2, out_channels // 4))
            elif i < num_block - 1:
                self.blocks.append(ConvBNReLU(out_channels // (2 ** i), out_channels // (2 ** (i + 1))))
            else:
                self.blocks.append(ConvBNReLU(out_channels // (2 ** i), out_channels // (2 ** i)))

    def forward(self, x):
        outputs = []
        output = x

        for i, block in enumerate(self.blocks):
            if i == 0 and self.stride == 2:
                output = self.layer(block(output))
            else:
                output = block(output)
            outputs.append(output)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(outputs, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, stride):
        super(CatBottleneck, self).__init__()
        self.blocks = nn.ModuleList([])
        self.stride = stride

        if stride == 2:
            self.layer = nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1,
                          groups=out_channels // 2, bias=False),
                nn.BatchNorm2d(out_channels // 2)
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(num_block):
            if i == 0:
                self.blocks.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0))
            elif i == 1 and num_block == 2:
                self.blocks.append(ConvBNReLU(out_channels // 2, out_channels // 2))
            elif i == 1 and num_block > 2:
                self.blocks.append(ConvBNReLU(out_channels // 2, out_channels // 4))
            elif i < num_block - 1:
                self.blocks.append(ConvBNReLU(out_channels // (2 ** i), out_channels // (2 ** (i + 1))))
            else:
                self.blocks.append(ConvBNReLU(out_channels // (2 ** i), out_channels // (2 ** i)))

    def forward(self, x):
        outputs = []
        output1 = self.blocks[0](x)

        for i, block in enumerate(self.blocks[1:]):
            if i == 0:
                if self.stride == 2:
                    output = block(self.layer(output1))
                else:
                    output = block(output1)
            else:
                output = block(output)
            outputs.append(output)

        if self.stride == 2:
            output1 = self.skip(output1)
        outputs.insert(0, output1)

        return torch.cat(outputs, dim=1)


class OutputModule(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super(OutputModule, self).__init__()
        self.layer = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)
