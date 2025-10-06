# trainer.py
from typing import Optional, Dict, Any
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from rendering import get_rays, render_rays


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """Compute PSNR = -10 * log10(MSE). Assumes normalized RGB in [0,1]."""
    return -10.0 * torch.log10(mse.clamp(min=1e-12))


class Trainer:
    """
    Classe d'entraînement pour NeRF, utilisant des DataLoader PyTorch.
    Un batch correspond à une ou plusieurs images complètes.
    """
    def __init__(self, model: torch.nn.Module, cfg: Dict[str, Any], device: str = "cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg["train"]["lr"]))
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # Dossier de sortie
        self.out_dir = cfg["train"].get("save_dir", "outputs")
        os.makedirs(self.out_dir, exist_ok=True)
        self.save_every = int(cfg["train"].get("save_every", 200))

    # ------------------------------------------------------
    # 1. Forward pass sur un batch (image, pose, focal)
    # ------------------------------------------------------
    def _one_forward(self, image, pose, focal, H, W, rand=True):
        image = image.to(self.device, dtype=torch.float32)
        pose  = pose.to(self.device,  dtype=torch.float32)
        focal = float(focal.item() if isinstance(focal, torch.Tensor) else focal)

        rays_o, rays_d = get_rays(H, W, focal, pose, device=self.device)
        rgb, depth, acc = render_rays(
            self.model, rays_o, rays_d,
            near=float(self.cfg["render"]["near"]),
            far=float(self.cfg["render"]["far"]),
            N_samples=int(self.cfg["train"]["n_samples"]),
            L_embed=int(self.cfg["model"]["L_embed"]),
            rand=rand,
            device=self.device
        )
        loss_mse = F.mse_loss(rgb, image)
        loss_ssim = 1.0 - self.ssim_metric(
            rgb.permute(2,0,1).unsqueeze(0),  # (1,3,H,W)
            image.permute(2,0,1).unsqueeze(0)
        )
        return loss_mse, loss_ssim, rgb

    # ------------------------------------------------------
    # 2. Boucle d'entraînement + validation
    # ------------------------------------------------------
    def fit(self, train_loader, val_loader: Optional[torch.utils.data.DataLoader] = None):
        n_iters = int(self.cfg["train"]["n_iters"])
        i_plot  = int(self.cfg["train"]["i_plot"])

        t0 = time.time()
        n_epochs = int(np.ceil(n_iters / len(train_loader)))
        iteration = 0

        print(f"Starting training for {n_epochs} epochs ({n_iters} iterations total)...\n")

        for epoch in range(1, n_epochs + 1):
            # --------------------------------------------------
            # Phase TRAIN
            # --------------------------------------------------
            self.model.train()
            train_mse, train_ssim = [], []

            for batch in train_loader:
                if iteration >= n_iters:
                    break

                loss_mse, loss_ssim, _ = self._one_forward(
                    image=batch["image"].squeeze(0),
                    pose=batch["pose"].squeeze(0),
                    focal=batch["focal"],
                    H=batch["H"],
                    W=batch["W"],
                    rand=True
                )

                loss_total = loss_mse + loss_ssim * 0.1  # pondération optionnelle

                self.opt.zero_grad()
                loss_total.backward()
                self.opt.step()

                train_mse.append(loss_mse.item())
                train_ssim.append(1 - loss_ssim.item())

                # -------------------------------
                # Sauvegarde périodique d'image
                # -------------------------------
                if (iteration + 1) % self.save_every == 0 and val_loader is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_batch = next(iter(val_loader))
                        _, _, val_rgb = self._one_forward(
                            image=val_batch["image"].squeeze(0),
                            pose=val_batch["pose"].squeeze(0),
                            focal=val_batch["focal"],
                            H=val_batch["H"],
                            W=val_batch["W"],
                            rand=False
                        )
                        img_save = val_rgb.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                        save_path = os.path.join(self.out_dir, f"render_iter_{iteration+1:04d}.png")
                        save_image(img_save, save_path)
                        print(f"🖼️  Saved render at iteration {iteration+1} -> {save_path}")

                iteration += 1

            # stats entraînement
            mean_mse = np.mean(train_mse)
            mean_ssim = np.mean(train_ssim)
            mean_psnr = psnr_from_mse(torch.tensor(mean_mse)).item()

            # --------------------------------------------------
            # Phase VALIDATION
            # --------------------------------------------------
            val_mse, val_ssim, val_psnr = [], [], []

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        loss_mse, loss_ssim, _ = self._one_forward(
                            image=val_batch["image"].squeeze(0),
                            pose=val_batch["pose"].squeeze(0),
                            focal=val_batch["focal"],
                            H=val_batch["H"],
                            W=val_batch["W"],
                            rand=False
                        )
                        val_mse.append(loss_mse.item())
                        val_ssim.append(1 - loss_ssim.item())
                        val_psnr.append(psnr_from_mse(loss_mse).item())

                mean_val_mse = np.mean(val_mse)
                mean_val_ssim = np.mean(val_ssim)
                mean_val_psnr = np.mean(val_psnr)

                print(f"[Epoch {epoch:03d}] "
                      f"Train: MSE={mean_mse:.6f}, SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f} | "
                      f"Val: MSE={mean_val_mse:.6f}, SSIM={mean_val_ssim:.4f}, PSNR={mean_val_psnr:.2f}")
            else:
                print(f"[Epoch {epoch:03d}] "
                      f"Train: MSE={mean_mse:.6f}, SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}")

        print("\nTraining complete.")
        return {"psnr": [], "iters": []}

