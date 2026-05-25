import torchvision.transforms as T
from PIL import Image
import torch

def build_bf_color_transform():
    return T.Compose([
        T.RandomChoice([
            T.RandomPosterize(bits=3, p=1.0),
            T.GaussianBlur(kernel_size=5, sigma=1.5),
            T.RandomSolarize(threshold=100, p=1.0),
        ], p=[0.4, 0.2, 0.4]),
        T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
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
    ])


class SimCLRAugmentation:
    """Produces one augmented (bf, fl) pair with shared geometry,
    independent color."""
    
    def __init__(self):
        self.geo = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(90),
        ])
        self.bf_color = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.1),
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
        
        # Shared geometric (same random seed for both)
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        bf = self.geo(bf)
        torch.manual_seed(seed)
        fl = self.geo(fl)
        
        # Independent color
        bf = self.bf_color(bf)
        fl = self.fl_color(fl)
        
        return self.to_tensor(bf), self.to_tensor(fl)