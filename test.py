import os
import cv2
import time
import torch
import argparse
from math import sqrt
from models.network import STDCSeg
from utils.general import increment_path, threshold, localization
from utils.transform import resize_image, resize_mask


def create_model(backbone, weights, device):
    model = STDCSeg(backbone)
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights))
        print('Successfully loaded weights\n')
    else:
        raise SystemExit('Failed to load weights')
    return model.eval().to(device)


def test(model, test_dataset, device, save_dir):
    image_directory = os.path.join(test_dataset, 'JPEGImages')
    mask_directory = os.path.join(test_dataset, 'SegmentationClass')
    image_filenames = os.listdir(image_directory)

    avg_error = 0
    max_error = 0
    max_error_image_name = ''

    error_leq1p_count = 0
    error_leq3p_count = 0
    error_leq5p_count = 0

    avg_infer_time = 0
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
        max_error_image_name = image_filename if error > max_error else max_error_image_name
        max_error = error if error > max_error else max_error
    print(f'average error: {avg_error / len(image_filenames)} maximum error: {max_error}')
    print(f'average inference time: {avg_infer_time / len(image_filenames)} s')
    print(f'percentage of images with error less equal than 1 pixel: {error_leq1p_count / len(image_filenames)}')
    print(f'percentage of images with error less equal than 3 pixels: {error_leq3p_count / len(image_filenames)}')
    print(f'percentage of images with error less equal than 5 pixels: {error_leq5p_count / len(image_filenames)}')
    print(f'test image with the maximum error: {max_error_image_name}')
    print(f'\nResults saved to {save_dir}')


def run(backbone, weights, test_dataset, device):
    save_dir = increment_path('runs/test/exp')
    os.makedirs(save_dir)

    model = create_model(backbone, weights, device)
    test(model, test_dataset, device, save_dir)


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
