import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from models.loss import DetailAggregateLoss
from models.network import STDCSeg
from utils.dataset import KeyholeDataset
from utils.general import MetricMonitor, increment_path
from utils.transform import train_transform, val_transform


def create_model(backbone, weights, device, use_boundary2, use_boundary4, use_boundary8, use_conv_last):
    model = STDCSeg(backbone, use_boundary2, use_boundary4, use_boundary8, use_conv_last)
    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
    return model.to(device)


def train(train_loader, model, bce_loss_func0, bce_loss_func16, bce_loss_func32, boundary_loss_func, optimizer, epoch,
          epochs, device, use_boundary2, use_boundary4, use_boundary8):
    metric_monitor = MetricMonitor(float_precision=5)
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_boundary2 and use_boundary4 and use_boundary8:
            out, out16, out32, detail2, detail4, detail8 = model(images)
        if (not use_boundary2) and use_boundary4 and use_boundary8:
            out, out16, out32, detail4, detail8 = model(images)
        if (not use_boundary2) and (not use_boundary4) and use_boundary8:
            out, out16, out32, detail8 = model(images)
        if (not use_boundary2) and (not use_boundary4) and (not use_boundary8):
            out, out16, out32 = model(images)

        bce_loss0 = bce_loss_func0(out.squeeze(1), target)
        bce_loss16 = bce_loss_func16(out16.squeeze(1), target)
        bce_loss32 = bce_loss_func32(out32.squeeze(1), target)
        bce_loss = bce_loss0 + bce_loss16 + bce_loss32
        metric_monitor.update('Segmentation Loss', bce_loss.item())

        boundary_loss = 0.
        boundary_loss = 0.
        if use_boundary2:
            boundary_bce_loss2, boundary_dice_loss2 = boundary_loss_func(detail2, target)
            boundary_loss += boundary_bce_loss2 + boundary_dice_loss2
        if use_boundary4:
            boundary_bce_loss4, boundary_dice_loss4 = boundary_loss_func(detail4, target)
            boundary_loss += boundary_bce_loss4 + boundary_dice_loss4
        if use_boundary8:
            boundary_bce_loss8, boundary_dice_loss8 = boundary_loss_func(detail8, target)
            boundary_loss += boundary_bce_loss8 + boundary_dice_loss8
        metric_monitor.update('Detail Loss', boundary_loss.item())

        loss = bce_loss + boundary_loss
        metric_monitor.update('Total Loss', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            'Epoch: {epoch}/{epochs}. Train.      {metric_monitor}'.format(
                epoch=epoch, epochs=epochs, metric_monitor=metric_monitor
            )
        )


def validate(val_loader, model, bce_loss_func0, bce_loss_func16, bce_loss_func32, boundary_loss_func, epoch, epochs,
             device, use_boundary2, use_boundary4, use_boundary8):
    metric_monitor = MetricMonitor(float_precision=5)
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if use_boundary2 and use_boundary4 and use_boundary8:
                out, out16, out32, detail2, detail4, detail8 = model(images)
            if (not use_boundary2) and use_boundary4 and use_boundary8:
                out, out16, out32, detail4, detail8 = model(images)
            if (not use_boundary2) and (not use_boundary4) and use_boundary8:
                out, out16, out32, detail8 = model(images)
            if (not use_boundary2) and (not use_boundary4) and (not use_boundary8):
                out, out16, out32 = model(images)

            bce_loss0 = bce_loss_func0(out.squeeze(1), target)
            bce_loss16 = bce_loss_func16(out16.squeeze(1), target)
            bce_loss32 = bce_loss_func32(out32.squeeze(1), target)
            bce_loss = bce_loss0 + bce_loss16 + bce_loss32
            metric_monitor.update('Segmentation Loss', bce_loss.item())

            boundary_loss = 0.
            boundary_loss = 0.
            if use_boundary2:
                boundary_bce_loss2, boundary_dice_loss2 = boundary_loss_func(detail2, target)
                boundary_loss += boundary_bce_loss2 + boundary_dice_loss2
            if use_boundary4:
                boundary_bce_loss4, boundary_dice_loss4 = boundary_loss_func(detail4, target)
                boundary_loss += boundary_bce_loss4 + boundary_dice_loss4
            if use_boundary8:
                boundary_bce_loss8, boundary_dice_loss8 = boundary_loss_func(detail8, target)
                boundary_loss += boundary_bce_loss8 + boundary_dice_loss8
            metric_monitor.update('Detail Loss', boundary_loss.item())

            loss = bce_loss + boundary_loss
            metric_monitor.update('Total Loss', loss.item())
            stream.set_description(
                'Epoch: {epoch}/{epochs}. Validation. {metric_monitor}'.format(
                    epoch=epoch, epochs=epochs, metric_monitor=metric_monitor
                )
            )
    avg_loss = metric_monitor.metrics['Total Loss']['avg']
    return avg_loss


def train_and_validate(model, train_dataset, val_dataset, device, batch_size, num_workers, epochs, use_boundary2,
                       use_boundary4, use_boundary8, save_dir):
    train_loader = DataLoader(
        KeyholeDataset(train_dataset, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        KeyholeDataset(val_dataset, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    bce_loss_func0 = nn.BCELoss().to(device)
    bce_loss_func16 = nn.BCELoss().to(device)
    bce_loss_func32 = nn.BCELoss().to(device)
    boundary_loss_func = DetailAggregateLoss(device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    min_loss = np.inf
    for epoch in range(1, epochs + 1):
        train(train_loader, model, bce_loss_func0, bce_loss_func16, bce_loss_func32, boundary_loss_func, optimizer,
              epoch, epochs, device, use_boundary2, use_boundary4, use_boundary8)
        torch.save(model.state_dict(), f'{save_dir}/last.pth')

        loss = validate(val_loader, model, bce_loss_func0, bce_loss_func16, bce_loss_func32, boundary_loss_func, epoch,
                        epochs, device, use_boundary2, use_boundary4, use_boundary8)
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), f'{save_dir}/best.pth')
    print(f'\nResults saved to {save_dir}')


def run(
        backbone='STDC813',
        weights='',
        train_dataset='datasets/train',
        val_dataset='datasets/val',
        device='cuda',
        batch_size=4,
        num_workers=4,
        epochs=50,
        use_boundary2=False,
        use_boundary4=False,
        use_boundary8=True,
        use_conv_last=False
):
    save_dir = increment_path('runs/train/exp')
    os.makedirs(save_dir, exist_ok=True)

    model = create_model(backbone, weights, device, use_boundary2, use_boundary4, use_boundary8, use_conv_last)
    train_and_validate(model, train_dataset, val_dataset, device, batch_size, num_workers, epochs, use_boundary2,
                       use_boundary4, use_boundary8, save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, help='backbone', default='STDC813')
    parser.add_argument('--train-dataset', type=str, help='training datasets', default='datasets/train')
    parser.add_argument('--val-dataset', type=str, help='validation datasets', default='datasets/val')
    parser.add_argument('--weights', help='weight path', default='')
    parser.add_argument('--batch-size', type=int, help='batch size', default=4)
    parser.add_argument('--num-workers', type=int, help='number of workers', default=4)
    parser.add_argument('--device', type=str, help='device', default='cuda')
    parser.add_argument('--epochs', type=int, help='number of epochs', default=50)
    parser.add_argument('--use-boundary2', type=bool, help='use boundary 2 or not', default=False)
    parser.add_argument('--use-boundary4', type=bool, help='use boundary 4 or not', default=False)
    parser.add_argument('--use-boundary8', type=bool, help='use boundary 8 or not', default=True)
    parser.add_argument('--use-conv-last', type=bool, help='use last conv layer or not', default=False)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
