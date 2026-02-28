import os
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
import torchvision.transforms as T


@dataclass
class TrainConfig:
    data_root: str = "outputs/frames_faces"  # expects subfolders real_### and fake_###
    out_dir: str = "outputs/models_cnn"
    img_size: int = 224
    batch_size: int = 32
    num_epochs: int = 4
    lr: float = 1e-3
    seed: int = 42
    train_ratio: float = 0.8
    max_imgs_per_tag: int = 80  # cap per video/tag to reduce imbalance
    num_workers: int = 0        # macOS + PIL is happier with 0/2; keep 0 to avoid issues


class FaceCropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.float32)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(cfg: TrainConfig):
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing: {root}")

    samples = []
    tag_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for tag_dir in tag_dirs:
        tag = tag_dir.name.lower()
        if tag.startswith("real_"):
            label = 0
        elif tag.startswith("fake_"):
            label = 1
        else:
            continue

        imgs = sorted(tag_dir.glob("*.jpg"))
        if len(imgs) == 0:
            continue

        # cap per tag to avoid one long video dominating
        if cfg.max_imgs_per_tag and len(imgs) > cfg.max_imgs_per_tag:
            imgs = random.sample(imgs, cfg.max_imgs_per_tag)

        for img in imgs:
            samples.append((str(img), label))

    random.shuffle(samples)
    return samples


def split_train_val(samples, train_ratio=0.8):
    n = len(samples)
    n_train = int(n * train_ratio)
    train = samples[:n_train]
    val = samples[n_train:]
    return train, val


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    samples = build_samples(cfg)
    if len(samples) < 50:
        raise RuntimeError(f"Too few samples found: {len(samples)}. Do you have real_*/fake_* crops?")

    train_samples, val_samples = split_train_val(samples, cfg.train_ratio)

    print(f"Total samples: {len(samples)} | train: {len(train_samples)} | val: {len(val_samples)}")
    print("Example:", train_samples[0])

    tfm = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_train = FaceCropDataset(train_samples, tfm)
    ds_val = FaceCropDataset(val_samples, tfm)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = get_device()
    print("Device:", device)

    # EfficientNet backbone
    backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)  # outputs features
    feat_dim = backbone.num_features
    backbone.to(device).eval()

    # freeze backbone for linear probe
    for p in backbone.parameters():
        p.requires_grad = False

    # binary head
    head = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1)  # logits
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(head.parameters(), lr=cfg.lr)

    def run_epoch(train=True):
        head.train(train)
        total_loss, correct, total = 0.0, 0, 0

        loader = dl_train if train else dl_val
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            with torch.no_grad():
                feats = backbone(x)

            logits = head(feats)
            loss = criterion(logits, y)

            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)

        return total_loss / len(loader.dataset), correct / total

    best_val_acc = -1.0
    best_path = Path(cfg.out_dir) / "efficientnet_b0_linearprobe.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc = run_epoch(train=True)
        va_loss, va_acc = run_epoch(train=False)
        print(f"Epoch {epoch}/{cfg.num_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "backbone_name": "efficientnet_b0",
                "feat_dim": feat_dim,
                "head_state": head.state_dict(),
                "img_size": cfg.img_size,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }, best_path)
            print("✅ saved:", best_path)

    print("\nBEST val acc:", round(best_val_acc, 3))
    print("Model saved at:", best_path)
    print("\nNext: we will use this as CNN Model C in your video-level pipeline.")


if __name__ == "__main__":
    main()