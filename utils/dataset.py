import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.transform import threshold


class KeyholeDataset(Dataset):
    def __init__(self, dataset_directory, transform=None):
        self.dataset_directory = dataset_directory
        self.image_directory = os.path.join(dataset_directory, 'JPEGImages')
        self.mask_directory = os.path.join(dataset_directory, 'SegmentationClass')
        self.image_filenames = os.listdir(self.image_directory)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image = cv2.imread(os.path.join(self.image_directory, image_filename))
        mask = cv2.imread(os.path.join(self.mask_directory, image_filename), cv2.IMREAD_GRAYSCALE)
        mask = threshold(mask).astype(np.float32) / 255

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
