import torchvision.transforms as T
from torchvision.transforms import RandAugment


def build_bf_color_transform():
    return T.Compose([
        T.RandomChoice([
            T.RandomPosterize(bits=3, p=1.0),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
            T.RandomSolarize(threshold=100, p=1.0),
        ], p=[0.4, 0.2, 0.4]),
        T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
        RandAugment(num_ops=2, magnitude=9),
    ])


def build_fl_color_transform():
    return T.Compose([
        T.ColorJitter(brightness=0.8, contrast=0.8),
        T.GaussianBlur(kernel_size=5, sigma=(0.3, 3.2)),
    ])


def build_shared_geo_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
    ])


def build_eval_transform():
    return None
