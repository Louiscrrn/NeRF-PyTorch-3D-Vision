import numpy as np
import torch
from torch.utils.data import Dataset

class LegoDataset(Dataset):
    """
    Dataset PyTorch minimal pour Tiny NeRF (lego).
    - split='train' : 100 premières vues (RGB)
    - split='test'  : vue de holdout (index configurable)
    Renvoie un dict: {image: (H,W,3) float32, pose: (4,4) float32, focal: float32}
    """
    def __init__(self, npz_path: str, split: str = "train", holdout_index: int = 101, keep_rgb_channels: int = 3):
        super().__init__()
        data = np.load(npz_path)
        images = data["images"]            # (N, H, W, 4) ou (N, H, W, 3)
        poses  = data["poses"]             # (N, 4, 4)
        focal  = data["focal"]             # scalaire

        if keep_rgb_channels is not None:
            images = images[..., :keep_rgb_channels]

        self.H, self.W = images.shape[1:3]
        self.focal = float(focal)

        if split == "train":
            # setup fidèle au notebook tiny-nerf
            self.images = images[:100].astype(np.float32)
            self.poses  = poses[:100].astype(np.float32)
        elif split in ("test", "val"):
            # un seul exemple holdout
            self.images = images[holdout_index:holdout_index+1, ...].astype(np.float32)
            self.poses  = poses[holdout_index:holdout_index+1, ...].astype(np.float32)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.split = split

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img  = torch.from_numpy(self.images[idx])      # (H, W, 3) float32
        pose = torch.from_numpy(self.poses[idx])       # (4, 4) float32
        focal = torch.tensor(self.focal, dtype=torch.float32)
        sample = {
            "image": img,          # (H,W,3) float32
            "pose": pose,          # (4,4)  float32
            "focal": focal,        # ()     float32
            "H": self.H,
            "W": self.W,
        }
        return sample
