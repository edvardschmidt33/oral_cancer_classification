import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import numpy as np


class OralCancerDataset(Dataset):
    """Returns (bf_tensor, fl_tensor, label, filename)."""
    def __init__(self, filenames, labels, bf_dir, fl_dir,
                 bf_transform=None, fl_transform=None, geo_transform=None):
        self.filenames = filenames
        self.labels = labels
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.bf_transform = bf_transform
        self.fl_transform = fl_transform
        self.geo_transform = geo_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        bf = Image.open(os.path.join(self.bf_dir, fname)).convert('RGB')
        fl = Image.open(os.path.join(self.fl_dir, fname)).convert('RGB')

        if self.bf_transform: bf = self.bf_transform(bf)
        if self.fl_transform: fl = self.fl_transform(fl)

        if self.geo_transform:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed); bf = self.geo_transform(bf)
            torch.manual_seed(seed); fl = self.geo_transform(fl)

        bf = T.ToTensor()(bf) if not isinstance(bf, torch.Tensor) else bf
        fl = T.ToTensor()(fl) if not isinstance(fl, torch.Tensor) else fl

        label = torch.tensor(self.labels[index], dtype=torch.float32) if self.labels is not None else -1
        return bf, fl, label, fname

class SimCLRDataset(Dataset):
    """Returns two augmented views of the same BF+FL pair."""
    
    def __init__(self, filenames, bf_dir, fl_dir, transform):
        self.filenames = filenames
        self.bf_dir = bf_dir
        self.fl_dir = fl_dir
        self.transform = transform  # stochastic — gives different output each call
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        bf = Image.open(os.path.join(self.bf_dir, fname)).convert('RGB')
        fl = Image.open(os.path.join(self.fl_dir, fname)).convert('RGB')
        
        # Stack to 6 channels (as numpy for transforms, or apply separately)
        bf_np = np.array(bf)
        fl_np = np.array(fl)
        
        # View 1: random augmentation
        bf1, fl1 = self.transform(bf_np.copy(), fl_np.copy())
        x1 = torch.cat([bf1, fl1], dim=0)  # [6, 128, 128]
        
        # View 2: different random augmentation of same cell
        bf2, fl2 = self.transform(bf_np.copy(), fl_np.copy())
        x2 = torch.cat([bf2, fl2], dim=0)  # [6, 128, 128]
        
        return x1, x2
    
    def __len__(self):
        return len(self.filenames)