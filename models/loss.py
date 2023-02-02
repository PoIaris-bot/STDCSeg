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
    def __init__(self):
        super(DetailAggregateLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32
        ).reshape(1, 1, 3, 3).requires_grad_(False)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor(
            [[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32
        ).reshape(1, 3, 1, 1))

    def forward(self, boundary, mask):
        if self.laplacian_kernel.device != boundary.device:
            self.laplacian_kernel = self.laplacian_kernel.to(boundary.device)
            self.fuse_kernel = self.fuse_kernel.to(boundary.device)

        boundary_target0 = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, padding=1)
        boundary_target0 = boundary_target0.clamp(min=0)
        boundary_target0[boundary_target0 > 0.1] = 1
        boundary_target0[boundary_target0 <= 0.1] = 0

        boundary_target2 = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, stride=2, padding=1)
        boundary_target2 = boundary_target2.clamp(min=0)
        boundary_target2_up = F.interpolate(boundary_target2, boundary_target0.shape[2:], mode='nearest')
        boundary_target2_up[boundary_target2_up > 0.1] = 1
        boundary_target2_up[boundary_target2_up <= 0.1] = 0

        boundary_target4 = F.conv2d(mask.unsqueeze(1), self.laplacian_kernel, stride=4, padding=1)
        boundary_target4 = boundary_target4.clamp(min=0)
        boundary_target4_up = F.interpolate(boundary_target4, boundary_target0.shape[2:], mode='nearest')
        boundary_target4_up[boundary_target4_up > 0.1] = 1
        boundary_target4_up[boundary_target4_up <= 0.1] = 0

        boudary_target = torch.stack((boundary_target0, boundary_target2_up, boundary_target4_up), dim=1)

        boudary_target = boudary_target.squeeze(2)
        boudary_target = F.conv2d(boudary_target, self.fuse_kernel)
        boudary_target[boudary_target > 0.1] = 1
        boudary_target[boudary_target <= 0.1] = 0

        if boundary.shape[-1] != boundary_target0.shape[-1]:
            boundary = F.interpolate(boundary, boundary_target0.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy(boundary, boudary_target)
        dice_loss = dice_loss_func(boundary, boudary_target)
        return bce_loss, dice_loss


class OhemBCELoss(nn.Module):
    def __init__(self, thresh, n_min):
        super(OhemBCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float32))
        self.n_min = n_min
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, output, target):
        loss = self.criterion(output, target).view(-1)
        loss, _ = torch.sort(loss, descending=True)

        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
