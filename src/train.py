import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .model import TinyCNN  # type: ignore[attr-defined]
except ImportError:
    from model import TinyCNN


def build_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_samples, 3, 32, 32)).astype(np.float32)

    # Deterministic labels based on pooled channel means and simple thresholds.
    ch_means = x.mean(axis=(2, 3))
    scores = (
        1.8 * ch_means[:, 0]
        - 1.2 * ch_means[:, 1]
        + 0.7 * ch_means[:, 2]
        + 0.3 * (ch_means[:, 0] * ch_means[:, 2])
    )
    bins = np.quantile(scores, np.linspace(0.0, 1.0, num_classes + 1))
    y = np.clip(np.digitize(scores, bins[1:-1], right=False), 0, num_classes - 1)

    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))


def train(
    out_path: str = "artifacts/model.pt",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_classes: int = 10,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("artifacts", exist_ok=True)
    model = TinyCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = build_synthetic_dataset(num_samples=4096, num_classes=num_classes, seed=seed)
    val_ds = build_synthetic_dataset(num_samples=1024, num_classes=num_classes, seed=seed + 1)

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
        "num_classes": num_classes,
        "seed": seed,
    }
    torch.save(payload, out_path)
    print(f"Saved checkpoint to {out_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyCNN on synthetic data.")
    parser.add_argument("--out", default="artifacts/model.pt", help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        seed=args.seed,
    )
