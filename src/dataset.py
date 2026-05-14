import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch


class OralCancerDataset(Dataset):
    """Loads paired BF + FL images and returns a 6-channel tensor."""
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
        x = torch.cat([bf, fl], dim=0)

        label = torch.tensor(self.labels[index], dtype=torch.float32) if self.labels is not None else -1
        return x, label, fname
