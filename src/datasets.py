import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

try:
    import torchvision
    import torchvision.transforms as T
except ImportError:  # pragma: no cover - handled at runtime for real-data mode
    torchvision = None
    T = None


DatasetKind = str


def build_synthetic_dataset(
    num_samples: int,
    num_classes: int,
    seed: int,
) -> TensorDataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_samples, 3, 32, 32)).astype(np.float32)

    # Deterministic labels from channel statistics.
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


def _subset_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False).tolist()
    return Subset(dataset, indices)


def _build_cifar10_dataset(
    data_dir: str,
    train: bool,
    download: bool,
) -> Dataset:
    if torchvision is None or T is None:
        raise RuntimeError("torchvision is required for --dataset cifar10. Install torchvision first.")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if train:
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    else:
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    return torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform,
    )


def build_train_val_datasets(
    dataset: DatasetKind,
    num_classes: int,
    seed: int,
    train_samples: int | None,
    val_samples: int | None,
    data_dir: str,
    download: bool,
) -> tuple[Dataset, Dataset, int]:
    if dataset == "synthetic":
        train_n = 4096 if train_samples is None else train_samples
        val_n = 1024 if val_samples is None else val_samples
        train_ds = build_synthetic_dataset(train_n, num_classes, seed)
        val_ds = build_synthetic_dataset(val_n, num_classes, seed + 1)
        return train_ds, val_ds, num_classes

    if dataset == "cifar10":
        train_ds = _build_cifar10_dataset(data_dir=data_dir, train=True, download=download)
        val_ds = _build_cifar10_dataset(data_dir=data_dir, train=False, download=download)
        train_ds = _subset_dataset(train_ds, train_samples, seed)
        val_ds = _subset_dataset(val_ds, val_samples, seed + 1)
        return train_ds, val_ds, 10

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_eval_dataset(
    dataset: DatasetKind,
    num_classes: int,
    seed: int,
    val_samples: int | None,
    data_dir: str,
    download: bool,
) -> Dataset:
    if dataset == "synthetic":
        val_n = 1024 if val_samples is None else val_samples
        return build_synthetic_dataset(val_n, num_classes, seed)

    if dataset == "cifar10":
        val_ds = _build_cifar10_dataset(data_dir=data_dir, train=False, download=download)
        return _subset_dataset(val_ds, val_samples, seed)

    raise ValueError(f"Unsupported dataset: {dataset}")
