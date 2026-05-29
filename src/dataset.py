import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import numpy as np


class OralCancerDataset(Dataset):
    """Returns (x, label, fname) where x is the BF|FL concatenated tensor
    (channels = 3 + fl_channels). Per-modality photometric jitter is applied
    in PIL space; then Resize -> ToTensor -> Normalize -> (FL tensor blur),
    then BF and FL are concatenated and a single geometric transform is run
    on the joined tensor so the two modalities stay spatially aligned.
    """

    def __init__(self, filenames, labels, bf_dir, fl_dir,
                 size=128, crop_size=None,
                 bf_mean=None, bf_std=None, fl_mean=None, fl_std=None,
                 fl_channels=3,
                 bf_color=None, fl_color=None, geo_transform=None,
                 fl_tensor_blur=True):
        self.filenames = filenames
        self.labels = labels
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.size = size
        self.fl_channels = fl_channels
        self.fl_mode = 'RGBA' if fl_channels == 4 else 'RGB'

        self.bf_color = bf_color
        self.fl_color = fl_color
        self.geo_transform = geo_transform
        # Final center crop runs on the concatenated tensor (after geometry) so
        # rotation/affine black corners are cropped away and BF/FL stay aligned.
        self.crop = T.CenterCrop(crop_size) if crop_size else None

        bf_steps = [T.Resize((size, size), antialias=True), T.ToTensor()]
        if bf_mean is not None and bf_std is not None:
            bf_steps.append(T.Normalize(bf_mean, bf_std))
        self.bf_post = T.Compose(bf_steps)

        fl_steps = [T.Resize((size, size), antialias=True), T.ToTensor()]
        if fl_tensor_blur:
            fl_steps.append(T.GaussianBlur(kernel_size=5, sigma=(0.3, 3.2)))
        if fl_mean is not None and fl_std is not None:
            fl_steps.append(T.Normalize(fl_mean, fl_std))
        self.fl_post = T.Compose(fl_steps)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        bf = Image.open(os.path.join(self.bf_dir, fname)).convert('RGB')
        fl = Image.open(os.path.join(self.fl_dir, fname)).convert(self.fl_mode)

        if self.bf_color is not None:
            bf = self.bf_color(bf)
        if self.fl_color is not None:
            fl = self.fl_color(fl)

        bf = self.bf_post(bf)
        fl = self.fl_post(fl)

        x = torch.cat([bf, fl], dim=0)

        if self.geo_transform is not None:
            x = self.geo_transform(x)

        if self.crop is not None:
            x = self.crop(x)

        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)
        return x, label, fname


class SimCLRDataset(Dataset):
    """Returns two augmented views of the same BF+FL pair."""

    def __init__(self, filenames, bf_dir, fl_dir, transform):
        self.filenames = filenames
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        bf = Image.open(os.path.join(self.bf_dir, fname)).convert('RGB')
        fl = Image.open(os.path.join(self.fl_dir, fname)).convert('RGB')

        bf_np = np.array(bf)
        fl_np = np.array(fl)

        bf1, fl1 = self.transform(bf_np.copy(), fl_np.copy())
        x1 = torch.cat([bf1, fl1], dim=0)

        bf2, fl2 = self.transform(bf_np.copy(), fl_np.copy())
        x2 = torch.cat([bf2, fl2], dim=0)

        return x1, x2

    def __len__(self):
        return len(self.filenames)
