import torchvision.transforms as T
from PIL import Image
import torch


def build_bf_color_transform():
    """BF photometric (PIL -> PIL). Strong stain/illumination jitter to break
    the patient-identity shortcut. Hue is 0 (BF stain shifts shouldn't move
    on the colour wheel)."""
    return T.Compose([
        T.RandomChoice([
            T.RandomPosterize(bits=3, p=1.0),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
            T.RandomSolarize(threshold=100, p=1.0),
        ], p=[0.4, 0.2, 0.4]),
        T.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.0),
    ])


def build_fl_color_transform(fl_channels=3):
    """FL photometric (PIL -> PIL). ColorJitter is only valid on RGB so it's
    skipped when FL is RGBA (4-channel). The tensor-space GaussianBlur is
    applied later, inside the dataset, after Normalize."""
    if fl_channels == 3:
        return T.Compose([
            T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
        ])
    return T.Compose([])  # no-op for 4-channel


def build_shared_geo_transform():
    """Geometric stage operating on the concatenated BF+FL tensor so the two
    modalities receive an identical spatial transform. Cells have no
    canonical orientation so rotation/affine/erasing are label-preserving."""
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08),
                       scale=(0.9, 1.1), shear=8),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


class SimCLRAugmentation:
    """Produces one augmented (bf, fl) pair with shared geometry,
    independent color. (Used by the SimCLR pretraining path.)"""

    def __init__(self):
        self.geo = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(90),
        ])
        self.bf_color = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.0),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
        self.fl_color = T.Compose([
            T.ColorJitter(brightness=0.8, contrast=0.8),
            T.GaussianBlur(kernel_size=5, sigma=(0.3, 3.2)),
        ])
        self.to_tensor = T.ToTensor()

    def __call__(self, bf_np, fl_np):
        bf = Image.fromarray(bf_np)
        fl = Image.fromarray(fl_np)

        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        bf = self.geo(bf)
        torch.manual_seed(seed)
        fl = self.geo(fl)

        bf = self.bf_color(bf)
        fl = self.fl_color(fl)

        return self.to_tensor(bf), self.to_tensor(fl)
