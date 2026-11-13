from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from .constants import FRESHNESS, FRESHNESS_TO_ID

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_ffe(root: str | Path) -> pd.DataFrame:
    """Index FFE folders of the form '<species> - <freshness>'."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for label in FRESHNESS:
            suffix = " - " + label
            if not folder.name.casefold().endswith(suffix.casefold()):
                continue
            species = folder.name[:-len(suffix)].strip()
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                    rows.append({"path": path.relative_to(root).as_posix(),
                                 "species": species, "freshness": label,
                                 "group": path.stem})
            break
    if not rows:
        raise ValueError("No FFE images found in species - freshness folders")
    frame = pd.DataFrame(rows)
    validate_manifest(frame)
    return frame


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"path": str, "species": str, "freshness": str})
    if "group" not in frame:
        frame["group"] = frame["path"]
    validate_manifest(frame)
    return frame


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {"path", "species", "freshness"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Manifest is empty or has missing values")
    invalid = sorted(set(frame["freshness"]) - set(FRESHNESS))
    if invalid:
        raise ValueError(f"Unknown freshness labels {invalid}; use {FRESHNESS}")
    if frame["path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image paths")


def species_map(frame: pd.DataFrame) -> dict[str, int]:
    return {str(s): i for i, s in enumerate(sorted(frame["species"].unique()))}


def stratified_split(frame: pd.DataFrame, seed: int, ratios=(0.7, 0.15, 0.15)):
    if not np.isclose(sum(ratios), 1):
        raise ValueError("Split ratios must sum to 1")
    strata = frame["species"].astype(str) + "|" + frame["freshness"]
    if strata.value_counts().min() < 3:
        strata = frame["freshness"]
    train, rest = train_test_split(frame, test_size=1.0-ratios[0],
                                   random_state=seed, stratify=strata)
    rest_strata = rest["species"].astype(str) + "|" + rest["freshness"]
    if rest_strata.value_counts().min() < 2:
        rest_strata = rest["freshness"]
    val, test = train_test_split(rest,
        test_size=ratios[2] / (ratios[1] + ratios[2]),
        random_state=seed, stratify=rest_strata)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def grouped_day_split(frame: pd.DataFrame, seed: int):
    """Split distinct acquisition days 70:15:15; days, not specimens, are groups."""
    if "day" not in frame:
        raise ValueError("DaFiF manifest requires the 'day' acquisition-day column")
    groups = frame["day"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    tr, rem = next(splitter.split(frame, groups=groups))
    train = frame.iloc[tr].reset_index(drop=True)
    remainder = frame.iloc[rem].reset_index(drop=True)
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    va, te = next(splitter2.split(remainder, groups=remainder["day"].astype(str)))
    return train, remainder.iloc[va].reset_index(drop=True), remainder.iloc[te].reset_index(drop=True)


def image_transforms(train: bool, image_size: int = 224):
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(image_size, scale=(0.70, 1.00)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(12),
            T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((256, 256)), T.CenterCrop(image_size), T.ToTensor(), norm])


class FishDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: str | Path,
                 species_to_id: Mapping[str, int], train: bool = False,
                 image_size: int = 224):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root).expanduser().resolve()
        self.species_to_id = dict(species_to_id)
        self.transform = image_transforms(train=train, image_size=image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx):
        item = self.frame.iloc[idx]
        path = Path(item["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as opened:
            image = self.transform(opened.convert("RGB"))
        return {
            "image": image,
            "freshness": torch.tensor(FRESHNESS_TO_ID[str(item["freshness"]).casefold()], dtype=torch.long),
            "species": torch.tensor(self.species_to_id.get(str(item["species"]), -1), dtype=torch.long),
            "path": str(item["path"]),
            "species_name": str(item["species"]),
        }


def loader(frame, root, s_map, batch_size, workers, train=False, image_size=224):
    return DataLoader(FishDataset(frame, root, s_map, train=train, image_size=image_size),
        batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
