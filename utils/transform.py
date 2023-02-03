import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

mean = (0.2518, 0.2756, 0.3137)
std = (0.1822, 0.1999, 0.2314)
size = (256, 256)

train_transform = A.Compose(
    [
        A.Resize(*size),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [A.Resize(*size), A.Normalize(mean=mean, std=std), ToTensorV2()]
)


def resize_image(image):
    transform = A.Compose(
        [A.Resize(*size), A.Normalize(mean=mean, std=std), ToTensorV2()]
    )
    transformed = transform(image=image)
    return transformed['image']


def resize_mask(image, mask):
    transform = A.Resize(image.shape[0], image.shape[1])
    transformed = transform(image=image, mask=mask)
    return transformed['mask']


def threshold(mask):
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask
