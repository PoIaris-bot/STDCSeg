import torch.nn.functional as F
from torch import nn
from models.module import (ConvBNReLU, CatBottleneck, AddBottleneck, FeatureFusionModule, OutputModule,
                           AttentionRefinementModule)


# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self, base=64, num_layers=(2, 2, 2), num_block=4, block_type='cat', use_conv_last=False):
        super(STDCNet813, self).__init__()
        self.use_conv_last = use_conv_last
        self.conv_last = ConvBNReLU(base * 16, max(1024, base * 16), 1, 1, 0)
        if block_type == 'cat':
            block = CatBottleneck
        elif block_type == 'add':
            block = AddBottleneck

        layers = [
            ConvBNReLU(3, base // 2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(base // 2, base, kernel_size=3, stride=2, padding=1)
        ]
        for i, num_layer in enumerate(num_layers):
            for j in range(num_layer):
                if i == 0 and j == 0:
                    layers.append(block(base, base * 4, num_block, 2))
                elif j == 0:
                    layers.append(block(base * (2 ** (i + 1)), base * (2 ** (i + 2)), num_block, 2))
                else:
                    layers.append(block(base * (2 ** (i + 2)), base * (2 ** (i + 2)), num_block, 1))
        self.layers = nn.Sequential(*layers)
        self.layer2 = nn.Sequential(self.layers[:1])
        self.layer4 = nn.Sequential(self.layers[1:2])
        self.layer8 = nn.Sequential(self.layers[2:4])
        self.layer16 = nn.Sequential(self.layers[4:6])
        self.layer32 = nn.Sequential(self.layers[6:])

    def forward(self, x):
        feature2 = self.layer2(x)
        feature4 = self.layer4(feature2)
        feature8 = self.layer8(feature4)
        feature16 = self.layer16(feature8)
        feature32 = self.layer32(feature16)
        if self.use_conv_last:
            feature32 = self.conv_last(feature32)

        return feature2, feature4, feature8, feature16, feature32


# STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, num_layers=(4, 5, 3), num_block=4, block_type='cat', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        self.use_conv_last = use_conv_last
        self.conv_last = ConvBNReLU(base * 16, max(1024, base * 16), 1, 1, 0)
        if block_type == 'cat':
            block = CatBottleneck
        elif block_type == 'add':
            block = AddBottleneck

        layers = [
            ConvBNReLU(3, base // 2, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(base // 2, base, kernel_size=3, stride=2, padding=1)
        ]
        for i, num_layer in enumerate(num_layers):
            for j in range(num_layer):
                if i == 0 and j == 0:
                    layers.append(block(base, base * 4, num_block, 2))
                elif j == 0:
                    layers.append(block(base * (2 ** (i + 1)), base * (2 ** (i + 2)), num_block, 2))
                else:
                    layers.append(block(base * (2 ** (i + 2)), base * (2 ** (i + 2)), num_block, 1))
        self.layers = nn.Sequential(*layers)
        self.layer2 = nn.Sequential(self.layers[:1])
        self.layer4 = nn.Sequential(self.layers[1:2])
        self.layer8 = nn.Sequential(self.layers[2:6])
        self.layer16 = nn.Sequential(self.layers[6:11])
        self.layer32 = nn.Sequential(self.layers[11:])

    def forward(self, x):
        feature2 = self.layer2(x)
        feature4 = self.layer4(feature2)
        feature8 = self.layer8(feature4)
        feature16 = self.layer16(feature8)
        feature32 = self.layer32(feature16)
        if self.use_conv_last:
            feature32 = self.conv_last(feature32)

        return feature2, feature4, feature8, feature16, feature32


class STDCSeg(nn.Module):
    def __init__(self, backbone, use_boundary2=False, use_boundary4=False, use_boundary8=False,
                 use_conv_last=False):
        super(STDCSeg, self).__init__()
        self.use_boundary2 = use_boundary2
        self.use_boundary4 = use_boundary4
        self.use_boundary8 = use_boundary8

        self.cp = ContextPath(backbone, use_conv_last=use_conv_last)
        conv_out_in_channels = 128
        sp2_in_channels = 32
        sp4_in_channels = 64
        sp8_in_channels = 256
        in_channels = sp8_in_channels + conv_out_in_channels

        self.ffm = FeatureFusionModule(in_channels, 256)
        self.conv_out = OutputModule(256, 256, 1)
        self.conv_out16 = OutputModule(conv_out_in_channels, 64, 1)
        self.conv_out32 = OutputModule(conv_out_in_channels, 64, 1)

        self.conv_out_sp8 = OutputModule(sp8_in_channels, 64, 1)
        self.conv_out_sp4 = OutputModule(sp4_in_channels, 64, 1)
        self.conv_out_sp2 = OutputModule(sp2_in_channels, 64, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        feature2, feature4, feature8, feature16_up, feature32_up = self.cp(x)

        feature_out_sp2 = self.conv_out_sp2(feature2)
        feature_out_sp4 = self.conv_out_sp4(feature4)
        feature_out_sp8 = self.conv_out_sp8(feature8)

        feature_fuse = self.ffm(feature8, feature16_up)

        feature_out = self.conv_out(feature_fuse)
        feature_out16 = self.conv_out16(feature16_up)
        feature_out32 = self.conv_out32(feature32_up)

        feature_out = F.interpolate(feature_out, (h, w), mode='bilinear', align_corners=True)
        feature_out16 = F.interpolate(feature_out16, (h, w), mode='bilinear', align_corners=True)
        feature_out32 = F.interpolate(feature_out32, (h, w), mode='bilinear', align_corners=True)

        if self.use_boundary2 and self.use_boundary4 and self.use_boundary8:
            return feature_out, feature_out16, feature_out32, feature_out_sp2, feature_out_sp4, feature_out_sp8

        if (not self.use_boundary2) and self.use_boundary4 and self.use_boundary8:
            return feature_out, feature_out16, feature_out32, feature_out_sp4, feature_out_sp8

        if (not self.use_boundary2) and (not self.use_boundary4) and self.use_boundary8:
            return feature_out, feature_out16, feature_out32, feature_out_sp8

        if (not self.use_boundary2) and (not self.use_boundary4) and (not self.use_boundary8):
            return feature_out, feature_out16, feature_out32


class ContextPath(nn.Module):
    def __init__(self, backbone='STDCNet813', use_conv_last=False):
        super(ContextPath, self).__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet813':
            self.backbone = STDCNet813(use_conv_last=use_conv_last)
        else:  # STDCNet1446
            self.backbone = STDCNet1446(use_conv_last=use_conv_last)
        self.arm16 = AttentionRefinementModule(512, 128)
        self.arm32 = AttentionRefinementModule(1024, 128)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_avg = ConvBNReLU(1024, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feature2, feature4, feature8, feature16, feature32 = self.backbone(x)
        h8, w8 = feature8.shape[2:]
        h16, w16 = feature16.shape[2:]
        h32, w32 = feature32.shape[2:]

        feature32_avg = self.glob_avg_pool(feature32)
        feature32_avg = self.conv_avg(feature32_avg)
        feature32_avg = F.interpolate(feature32_avg, (h32, w32), mode='nearest')

        feature32_arm = self.arm32(feature32)
        feature32_sum = feature32_arm + feature32_avg
        feature32_up = F.interpolate(feature32_sum, (h16, w16), mode='nearest')
        feature32_up = self.conv_head32(feature32_up)

        feature16_arm = self.arm16(feature16)
        feature16_sum = feature16_arm + feature32_up
        feature16_up = F.interpolate(feature16_sum, (h8, w8), mode='nearest')
        feature16_up = self.conv_head16(feature16_up)

        return feature2, feature4, feature8, feature16_up, feature32_up
