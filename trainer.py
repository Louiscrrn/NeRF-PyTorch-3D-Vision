import os
import torch
import imageio
import numpy as np
from tqdm import trange
from rendering import render_full_image, render_batch_of_rays
from sampling import sample, dep_to_pos
from metrics import psnr_metric 
from torchmetrics.functional.image import structural_similarity_index_measure

class Trainer:
    """
    Classe d'entraînement pour NeRF, utilisant des DataLoader PyTorch.
    Un batch correspond à une ou plusieurs images complètes.
    """
    def __init__(self, model, optimizer, scheduler, pos_encoder, dir_encoder, device="cuda"):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder

        self.device = device

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_psnr": [],
            "val_psnr": [],
            "train_ssim": [],
            "val_ssim": [],
            "lr": []
        }

    def fit(self, train_ds, val_ds, epoch=10000, batch_size=1024, 
            near=2.0, far=6.0, n_samples=64, 
            white_bkgd=False, chunk_size=4096, step_validation=500, logdir="logs"):
        """
        Train the NeRF model on the dataset.
        """
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, "val_images"), exist_ok=True)

        all_rays_o = train_ds.all_rays_o.to(self.device)
        all_rays_d = train_ds.all_rays_d.to(self.device)
        all_rgb_gt = train_ds.all_targets.to(self.device)

        # Training loop
        pbar = trange(epoch, desc="Training")
        for i in pbar:
            ray_indices = torch.randint(0, all_rays_o.shape[0], (batch_size,), device=self.device)
            rays_o = all_rays_o[ray_indices]
            rays_d = all_rays_d[ray_indices]
            target_rgb = all_rgb_gt[ray_indices]

            self.model.train()
            rendered_rgb, depth_map, acc_map, weights = render_batch_of_rays(
                model=self.model,
                pos_enc_input=self.pos_encoder,
                pos_enc_dir=self.dir_encoder,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                N_samples=n_samples,
                white_bkgd=white_bkgd,
                device=self.device,
                chunk_size=chunk_size,
            )

            loss = torch.mean((rendered_rgb - target_rgb) ** 2)
            psnr = psnr_metric(target_rgb, rendered_rgb)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.history["epoch"].append(i)
            self.history["train_loss"].append(loss.item())
            self.history["train_psnr"].append(psnr.item())
            self.history["lr"].append(current_lr)

            pbar.set_description(
                f"[Iter {i}] Loss: {loss.item():.4f} | PSNR: {psnr.item():.2f} | LR: {current_lr:.6f}"
            )

            self.model.eval()
            with torch.no_grad():
                sample = val_ds[0]
                rendered_val = render_full_image(
                        model=self.model,
                        encoder_input=self.pos_encoder,
                        encoder_dir=self.dir_encoder,
                        H=val_ds.H,
                        W=val_ds.W,
                        focal=val_ds.focal,
                        c2w=sample["c2w"],
                        near=near,
                        far=far,
                        n_samples=n_samples,
                        white_bkgd=white_bkgd,
                        device=self.device,
                        chunk_size=chunk_size,
                    )

            target_img_val = sample["target_rgbs"].reshape(val_ds.H, val_ds.W, 3).to(self.device)
            val_loss = torch.mean((rendered_val - target_img_val) ** 2)
            psnr_val = psnr_metric(target_img_val, rendered_val)
            ssim_val = structural_similarity_index_measure(
                        rendered_val.permute(2, 0, 1).unsqueeze(0),
                        target_img_val.permute(2, 0, 1).unsqueeze(0)
                    )

            if (i + 1) % step_validation == 0:   
                img_np = (rendered_val.detach().cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(logdir, "val_images", f"iter_{i:06d}.png"), img_np)
                print(f"Saved validation image at iteration {i}")
                
                print(f"Validation Metrics — Iter {i}")
                print(f"Val Loss: {val_loss.item():.4f} | PSNR: {psnr_val.item():.2f} | SSIM: {ssim_val.item():.4f}")

            self.history["val_loss"].append(val_loss.item())
            self.history["val_psnr"].append(psnr_val.item())
            self.history["val_ssim"].append(ssim_val.item())

            if (i + 1) % 2000 == 0:
                ckpt_path = os.path.join(logdir, f"checkpoint_{i:06d}.tar")
                torch.save({
                    "iteration": i,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                }, ckpt_path)
                print(f"Checkpoint saved at {ckpt_path}")

        hist_path = os.path.join(logdir, "training_history.npy")
        np.save(hist_path, self.history)
        print(f"\nTraining complete. History saved at {hist_path}")
        
        print("Training complete")
        return
