import yaml
import torch
from dataset import LegoDataset
from model import NeRF, PositionalEncoding
from trainer import Trainer
import torch.optim as optim


def main():
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]
   
    print("=== NeRF Training ===")
    print(f"Device: {device}")
    print(f"Data: {cfg['data_path']}")
    print("---------------------------\n")

    # -- Dataset loading --
    train_ds = LegoDataset(cfg["data_path"], split="train")
    val_ds   = LegoDataset(cfg["data_path"], split="val")
    print(f"Train samples: {len(train_ds)} | Validation samples: {len(val_ds)}\n")

    # -- Model loading --
    cfg_model = cfg["model"]
    pos_encoder = PositionalEncoding(num_freqs=int(cfg_model["L_position"]))
    dir_encoder = PositionalEncoding(num_freqs=int(cfg_model["L_direction"]))
    model = NeRF(pos_input_size=pos_encoder.output_dims, pos_dir_size=dir_encoder.output_dims).to(device)
    
    # -- Trainer loading --
    cfg_train = cfg["train"]
    optimizer=optim.Adam(model.parameters(),lr=float(cfg_train['lr']))
    scheduler=optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(cfg_train['lr_decay_rate']**(1/cfg_train['lr_decay_steps']))
    )
    trainer = Trainer(model, optimizer, scheduler, pos_encoder, dir_encoder, device=device)

    # -- Training --
    render_cfg = cfg['render']
    history = trainer.fit(train_ds, val_ds, epoch=int(cfg_train["epoch"]), batch_size=int(cfg_train["batch_size"]),
                            near=float(render_cfg['near']), far=float(render_cfg['far']), n_samples=int(render_cfg['n_samples']),
                             chunk_size=int(render_cfg['chunk_size']), logdir=cfg_train['logs_dir'], step_validation=int(cfg_train["step_validation"]) )



if __name__ == "__main__":
    main()
