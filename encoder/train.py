import dataclasses
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HMDataset
from model import FashionEncoder


@dataclass
class Config:
    image_root:      str   = ".."
    articles_csv:    str   = "../articles.csv"
    embedding_dim:   int   = 512
    freeze_backbone: bool  = True
    batch_size:      int   = 32
    epochs:          int   = 50
    lr:              float = 1e-4
    margin:          float = 0.2
    num_workers:     int   = 0
    seed:            int   = 42
    checkpoint_dir:  str   = "../checkpoints"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device  = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    dataset = HMDataset(articles_csv=cfg.articles_csv, image_root=cfg.image_root, seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = FashionEncoder(embedding_dim=cfg.embedding_dim, freeze_backbone=cfg.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    criterion = nn.TripletMarginLoss(margin=cfg.margin, p=2, reduction="mean")
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, pos, neg) in enumerate(dataloader):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(anchor), model(pos), model(neg))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{cfg.epochs} | Batch {batch_idx}/{len(dataloader)} | Loss {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{cfg.epochs} | Avg Loss {avg_loss:.4f}")

        ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": dataclasses.asdict(cfg),
            },
            ckpt_path,
        )

    torch.save(
        {"model_state_dict": model.state_dict(), "config": dataclasses.asdict(cfg)},
        os.path.join(cfg.checkpoint_dir, "model_final.pt"),
    )
    print(f"Saved: {os.path.join(cfg.checkpoint_dir, 'model_final.pt')}")


if __name__ == "__main__":
    train(Config())
