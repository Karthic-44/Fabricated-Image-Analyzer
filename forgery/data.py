import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .preprocess import to_array

SOURCE_ID = re.compile(r"([a-z]{3})_?(\d{5})")
JPEG_SUFFIXES = {".jpg", ".jpeg"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_dir(root, name):
    matches = [p for p in Path(root).rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"no {name} folder under {root}")
    return min(matches, key=lambda p: len(p.parts))


def group_key(path):
    ids = SOURCE_ID.findall(path.stem.lower())
    if not ids:
        return path.stem
    cat, num = ids[0]
    return cat + num


def quant_mean(path):
    with Image.open(path) as im:
        tables = getattr(im, "quantization", None)
    if not tables:
        return np.nan
    return float(np.mean(tables[0]))


def assign_splits(df, seed, val_frac, test_frac):
    split = np.full(len(df), "train", dtype=object)
    outer = GroupShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    _, hold_idx = next(outer.split(df, groups=df["group"]))
    hold = df.iloc[hold_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    val_pos, test_pos = next(inner.split(hold, groups=hold["group"]))
    split[hold_idx[val_pos]] = "val"
    split[hold_idx[test_pos]] = "test"
    return split


def build_manifest(root, seed=42, val_frac=0.15, test_frac=0.15):
    rows = []
    for folder_name, label in (("Au", 0), ("Tp", 1)):
        folder = find_dir(root, folder_name)
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in JPEG_SUFFIXES:
                rows.append({"path": str(p), "label": label, "group": group_key(p)})
    df = pd.DataFrame(rows)
    df["qmean"] = [quant_mean(p) for p in df["path"]]
    df["split"] = assign_splits(df, seed, val_frac, test_frac)
    return df


def _load(job):
    path, mode, size, quality = job
    with Image.open(path) as im:
        return to_array(im, mode, size, quality)


def load_arrays(df, mode, size, quality, cache_dir="cache"):
    target = Path(cache_dir) / f"{mode}_{size}_q{quality}.npz"
    paths = df["path"].to_numpy().astype(str)
    if target.exists():
        saved = np.load(target)
        if np.array_equal(saved["paths"], paths):
            return saved["x"]
    jobs = [(p, mode, size, quality) for p in paths]
    x = np.empty((len(jobs), size, size, 3), dtype=np.uint8)
    with ProcessPoolExecutor() as pool:
        for i, arr in enumerate(tqdm(pool.map(_load, jobs, chunksize=32), total=len(jobs), desc=f"caching {mode}")):
            x[i] = arr
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, paths=paths, x=x)
    return x


def prepare_batch(x, device, train=False):
    x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255
    if train:
        flip = torch.rand(x.size(0), device=device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    return (x - MEAN.to(device)) / STD.to(device)
