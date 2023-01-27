import os
import cv2
import time
import torch
import argparse
from math import sqrt
from pathlib import Path
from models.network import STDCSeg
from utils.general import increment_path, localization
from utils.transform import resize_image, resize_mask, threshold


def create_model(backbone, weights, device):
    print(f'Loading {backbone}Seg...')
    model = STDCSeg(backbone)
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
        print('Successfully loaded weights\n')
    else:
        raise SystemExit('Failed to load weights')
    return model.eval().to(device)


def log(backbone, weights, save_dir, avg_error, max_error, max_error_image_path, avg_infer_time, error_leq1p_ratio,
        error_leq3p_ratio, error_leq5p_ratio):
    text = \
        'backbone: {}\n' \
        'weights: {}\n' \
        'average error: {:.4f} p\n' \
        'maximum error: {:.4f} p\n' \
        'test image with the maximum error: {}\n' \
        'average inference time: {:.2f} ms\n' \
        'percentage of images with error less equal than 1 pixel: {:.2f} %\n' \
        'percentage of images with error less equal than 3 pixels: {:.2f} %\n' \
        'percentage of images with error less equal than 5 pixels: {:.2f} %\n' \
        'Results saved to {}'.format(
            backbone, Path(weights).absolute(), avg_error, max_error, Path(max_error_image_path).absolute(),
            avg_infer_time, error_leq1p_ratio, error_leq3p_ratio, error_leq5p_ratio, save_dir
        )

    with open(f'{save_dir}/log.txt', 'w') as f:
        f.write(text)
    print(text)


def test(backbone, weights, test_dataset, device, save_dir):
    model = create_model(backbone, weights, device)
    image_directory = os.path.join(test_dataset, 'JPEGImages')
    mask_directory = os.path.join(test_dataset, 'SegmentationClass')
    image_filenames = os.listdir(image_directory)

    avg_error = 0
    max_error = 0
    max_error_image_path = ''

    error_leq1p_count = 0
    error_leq3p_count = 0
    error_leq5p_count = 0

    avg_infer_time = 0
    with torch.no_grad():
        for image_filename in image_filenames:
            image_path = os.path.join(image_directory, image_filename)
            mask_path = os.path.join(mask_directory, image_filename)
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = threshold(mask)

            input_image = torch.unsqueeze(resize_image(image), dim=0).to(device)
            start = time.time()
            predicted_masks = model(input_image)
            end = time.time()
            avg_infer_time += end - start

            predicted_mask = predicted_masks[0].squeeze().cpu().detach().numpy() * 255
            predicted_mask = threshold(predicted_mask.astype('uint8'))
            predicted_mask = resize_mask(image, predicted_mask)

            (x, y), predicted_contours = localization(predicted_mask)
            (x0, y0), contours = localization(mask)

            cv2.drawContours(image, predicted_contours, -1, (0, 0, 255), 3)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            cv2.circle(image, (x, y), 3, (0, 255, 0), 2)
            cv2.circle(image, (x0, y0), 3, (0, 0, 255), 2)
            cv2.imwrite(f'{save_dir}/{image_filename}', image)

            error = sqrt((x - x0) ** 2 + (y - y0) ** 2)
            error_leq1p_count += 1 if error <= 1 else 0
            error_leq3p_count += 1 if error <= 3 else 0
            error_leq5p_count += 1 if error <= 5 else 0

            avg_error += error
            max_error_image_path = image_path if error > max_error else max_error_image_path
            max_error = error if error > max_error else max_error
    log(backbone, weights, save_dir, avg_error / len(image_filenames), max_error, max_error_image_path,
        avg_infer_time / len(image_filenames) * 1000, error_leq1p_count / len(image_filenames) * 100,
        error_leq3p_count / len(image_filenames) * 100, error_leq5p_count / len(image_filenames) * 100)


def run(backbone, weights, test_dataset, device):
    save_dir = increment_path('runs/test/exp')
    os.makedirs(save_dir)

    test(backbone, weights, test_dataset, device, save_dir)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, help='backbone', default='STDC813')
    parser.add_argument('--weights', type=str, help='model path', default='weights/STDC813Seg.pth')
    parser.add_argument('--test-dataset', type=str, help='test datasets', default='datasets/test')
    parser.add_argument('--device', type=str, help='device', default='cuda')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
