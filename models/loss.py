import torch
import torch.nn.functional as F
from torch import nn


def dice_loss_func(prediction, target):
    smooth = 1.
    n = prediction.size(0)
    prediction_flat = prediction.view(n, -1)
    target_flat = target.view(n, -1)
    intersection = (prediction_flat * target_flat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (prediction_flat.sum(1) + target_flat.sum(1) + smooth))
    return loss.mean()


class DetailAggregateLoss(nn.Module):
    def __init__(self, device):
        super(DetailAggregateLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32
        ).reshape(1, 1, 3, 3).requires_grad_(False).to(device)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor(
            [[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32
        ).reshape(1, 3, 1, 1)).to(device)

    def forward(self, boundary, mask):
        boundary_target = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, padding=1)
        boundary_target = boundary_target.clamp(min=0)
        boundary_target[boundary_target > 0.1] = 1
        boundary_target[boundary_target <= 0.1] = 0

        boundary_target2 = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, stride=2, padding=1)
        boundary_target2 = boundary_target2.clamp(min=0)
        boundary_target2_up = F.interpolate(boundary_target2, boundary_target.shape[2:], mode='nearest')
        boundary_target2_up[boundary_target2_up > 0.1] = 1
        boundary_target2_up[boundary_target2_up <= 0.1] = 0

        boundary_target4 = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, stride=4, padding=1)
        boundary_target4 = boundary_target4.clamp(min=0)
        boundary_target4_up = F.interpolate(boundary_target4, boundary_target.shape[2:], mode='nearest')
        boundary_target4_up[boundary_target4_up > 0.1] = 1
        boundary_target4_up[boundary_target4_up <= 0.1] = 0

        boudary_target_pyramid = torch.stack((boundary_target, boundary_target2_up, boundary_target4_up), dim=1)

        boudary_target_pyramid = boudary_target_pyramid.squeeze(2)
        boudary_target_pyramid = F.conv2d(boudary_target_pyramid, self.fuse_kernel)
        boudary_target_pyramid[boudary_target_pyramid > 0.1] = 1
        boudary_target_pyramid[boudary_target_pyramid <= 0.1] = 0

        if boundary.shape[-1] != boundary_target.shape[-1]:
            boundary = F.interpolate(boundary, boundary_target.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy(boundary, boudary_target_pyramid)
        dice_loss = dice_loss_func(boundary, boudary_target_pyramid)
        return bce_loss, dice_loss
