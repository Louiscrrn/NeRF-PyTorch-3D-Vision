# main.py
import yaml
import torch
from torch.utils.data import DataLoader
from dataset import LegoDataset
from model import NeRFModel
from trainer import Trainer


def main():
    # -----------------------------------------------------
    # Chargement de la configuration
    # -----------------------------------------------------
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]
    if not torch.cuda.is_available():
        device = "cpu"
        print("⚠️  CUDA non disponible — entraînement sur CPU.\n")

    print("=== TinyNeRF Training ===")
    print(f"Device: {device}")
    print(f"Data: {cfg['data_path']}")
    print("---------------------------\n")

    # -----------------------------------------------------
    # Chargement du dataset et des DataLoaders
    # -----------------------------------------------------
    train_ds = LegoDataset(cfg["data_path"], split="train")
    val_ds   = LegoDataset(cfg["data_path"], split="test")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}\n")

    # -----------------------------------------------------
    # Initialisation du modèle et du trainer
    # -----------------------------------------------------
    model = NeRFModel(**cfg["model"]).to(device)
    trainer = Trainer(model, cfg, device=device)

    # -----------------------------------------------------
    # Lancement de l'entraînement
    # -----------------------------------------------------
    history = trainer.fit(train_loader, val_loader)

    # -----------------------------------------------------
    # Résumé final
    # -----------------------------------------------------
    print("\n=== Entraînement terminé ===")
    if "psnr" in history and len(history["psnr"]) > 0:
        print(f"Meilleur PSNR (val) : {max(history['psnr']):.2f} dB")
    print(f"Images sauvegardées dans : {cfg['train'].get('save_dir', 'outputs/')}\n")
    print("✅ Done.")


if __name__ == "__main__":
    main()
