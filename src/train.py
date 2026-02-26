import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

try:
    from .datasets import build_train_val_datasets  # type: ignore[attr-defined]
    from .model import TinyCNN  # type: ignore[attr-defined]
except ImportError:
    from datasets import build_train_val_datasets
    from model import TinyCNN


def train(
    out_path: str = "artifacts/model.pt",
    dataset: str = "synthetic",
    data_dir: str = "artifacts/data",
    download: bool = True,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_classes: int = 10,
    train_samples: int | None = None,
    val_samples: int | None = None,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("artifacts", exist_ok=True)

    train_ds, val_ds, resolved_num_classes = build_train_val_datasets(
        dataset=dataset,
        num_classes=num_classes,
        seed=seed,
        train_samples=train_samples,
        val_samples=val_samples,
        data_dir=data_dir,
        download=download,
    )

    model = TinyCNN(num_classes=resolved_num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_acc = correct / total
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

    payload = {
        "state_dict": model.state_dict(),
        "num_classes": resolved_num_classes,
        "seed": seed,
        "dataset": dataset,
        "data_dir": data_dir,
    }
    torch.save(payload, out_path)
    print(f"Saved checkpoint to {out_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyCNN on synthetic data or CIFAR-10.")
    parser.add_argument("--out", default="artifacts/model.pt", help="Output checkpoint path.")
    parser.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        out_path=args.out,
        dataset=args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )
