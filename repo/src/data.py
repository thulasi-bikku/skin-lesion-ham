"""Data loading, lesion-level splitting and image transforms for HAM10000."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .common import CODE_TO_IDX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def index_images(data_dir):
    """Map image_id -> file path for every .jpg below data_dir."""
    paths = {p.stem: str(p) for p in Path(data_dir).rglob("*.jpg")}
    if not paths:
        raise FileNotFoundError(f"No .jpg files found below {data_dir}")
    return paths


def load_metadata(data_dir):
    meta = pd.read_csv(Path(data_dir) / "HAM10000_metadata.csv")
    paths = index_images(data_dir)
    meta = meta[meta["image_id"].isin(paths)].copy()
    meta["path"] = meta["image_id"].map(paths)
    meta["label"] = meta["dx"].map(CODE_TO_IDX)
    return meta.reset_index(drop=True)


def make_splits(meta, seed=42, n_folds=10, test_folds=2, val_folds=1):
    """Stratified, lesion-grouped split into train/val/test (about 70/10/20).

    HAM10000 contains several images of the same lesion. Grouping by
    lesion_id keeps all images of a lesion in a single partition and prevents
    near-duplicate leakage between training and evaluation data.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(meta), -1)
    for k, (_, idx) in enumerate(sgkf.split(meta, meta["label"], groups=meta["lesion_id"])):
        fold[idx] = k
    split = np.where(fold < test_folds, "test",
                     np.where(fold < test_folds + val_folds, "val", "train"))
    out = meta.copy()
    out["split"] = split
    return out


def build_transforms(size, train):
    from torchvision import transforms as T
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


class LesionDataset:
    """Minimal map-style dataset (compatible with torch.utils.data.DataLoader)."""

    def __init__(self, frame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]
