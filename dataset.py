import numpy as np
import torch
from torch.utils.data import Dataset

def get_rays(H, W, focal, c2w):
        i, j = torch.meshgrid(
            torch.linspace(0, W - 1, W),
            torch.linspace(0, H - 1, H),
            indexing='xy'
        )
        dirs = torch.stack([(i - W * 0.5) / focal,
                            -(j - H * 0.5) / focal,
                            -torch.ones_like(i)], dim=-1)

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        return rays_o, rays_d


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
        images = data["images"]            
        poses  = data["poses"]             
        focal  = data["focal"]             

        if keep_rgb_channels is not None:
            images = images[..., :keep_rgb_channels]

        self.H, self.W = images.shape[1:3]
        self.focal = float(focal)

        if split == "train":

            self.images = images[:100].astype(np.float32)
            self.poses  = poses[:100].astype(np.float32)
        elif split in ("test", "val"):

            self.images = images[holdout_index:holdout_index+1, ...].astype(np.float32)
            self.poses  = poses[holdout_index:holdout_index+1, ...].astype(np.float32)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if split == "train":
            self.all_rays_o, self.all_rays_d, self.all_targets = self._precompute_all_rays(
                images=self.images,
                poses=self.poses,
                H=self.H,
                W=self.W,
                focal=self.focal,
                get_rays_fn=get_rays 
            )

        self.split = split

    def __len__(self):
        if self.split == "train":
            return self.all_targets.shape[0]
        else:
            return self.images.shape[0]


    def _precompute_all_rays(self, images, poses, H, W, focal, get_rays_fn):
        all_rays_o, all_rays_d, all_targets = [], [], []

        for i in range(len(images)):
            rays_o, rays_d = get_rays_fn(H, W, focal, torch.from_numpy(poses[i]))
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

            img_tensor = images[i]
            if isinstance(img_tensor, np.ndarray):
                img_tensor = torch.from_numpy(img_tensor)
            all_targets.append(img_tensor)

        all_rays_o = torch.cat([r.view(-1, 3) for r in all_rays_o], dim=0)
        all_rays_d = torch.cat([r.view(-1, 3) for r in all_rays_d], dim=0)
        all_targets = torch.cat([t.view(-1, 3) for t in all_targets], dim=0)

        return all_rays_o, all_rays_d, all_targets

    def __getitem__(self, idx):
        if self.split == "train":
            return {
                "rays_o": self.all_rays_o[idx],
                "rays_d": self.all_rays_d[idx],
                "target_s": self.all_targets[idx],
            }
        else:
            pose = torch.from_numpy(self.poses[idx])
            target_img_rgb = torch.from_numpy(self.images[idx])
            rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose)
            return {
                "rays_o": rays_o,                 
                "rays_d": rays_d,                
                "target_rgbs": target_img_rgb.reshape(-1, 3),
                "H": self.H,
                "W": self.W,
                "c2w": pose                       
            }

if __name__ == "__main__":
    
    data_path = "data/tiny_nerf_data.npz"
                 
    # -- Train split -- 
    train_dataset = LegoDataset(data_path, split="train")

    print(f"Dataset length {len(train_dataset)} rays")
    print(f"Image size: {train_dataset.H}x{train_dataset.W}")
    print(f"Focal: {train_dataset.focal:.2f}")

    train_sample = train_dataset[0]
    print("Ray origin shape :", train_sample["rays_o"].shape)
    print("Ray direction shape :", train_sample["rays_d"].shape)
    print("Target RGB shape :", train_sample["target_s"].shape)


    # -- Val split -- 
    val_dataset = LegoDataset(data_path, split="val")
    
    print(f"Dataset length: {len(val_dataset)} images")
    print(f"Image size: {val_dataset.H}x{val_dataset.W}")
    print(f"Focal: {val_dataset.focal:.2f}")

    val_sample = val_dataset[0]
    print("Ray origins shape :", val_sample["rays_o"].shape)
    print("Ray directions shape :", val_sample["rays_d"].shape)
    print("Target RGBs shape :", val_sample["target_rgbs"].shape)
    print("Pose shape (c2w):", val_sample["c2w"].shape)
    print("Image dims:", val_sample["H"], "x", val_sample["W"])