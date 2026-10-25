import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from forgery.data import build_manifest, load_arrays, prepare_batch
from forgery.metrics import best_threshold, summarize
from forgery.models import build_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/CASIA2")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--input", default="ela", choices=["ela", "rgb"])
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_manifest(args):
    path = Path(args.manifest)
    if path.exists():
        return
    root = Path(args.dataset)
    if not root.exists():
        sys.exit(f"Dataset not found at {root.resolve()}. Extract CASIA v2 there and run this again.")
    df = build_manifest(root, seed=args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"manifest written: {len(df)} images")


def make_loader(x, y, batch_size, shuffle, device):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=device.type == "cuda")


def amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            with amp(device):
                logits = model(prepare_batch(xb, device)).squeeze(1)
            probs.append(torch.sigmoid(logits.float()).cpu())
            ys.append(yb)
    return torch.cat(ys).numpy().astype(int), torch.cat(probs).numpy()


def export_model(name):
    result = subprocess.run([sys.executable, "-m", "forgery.export_onnx", name])
    if result.returncode != 0:
        print(f"Export failed. The trained model is safe in runs/{name}.")
        print(f"After fixing the error above, run: python -m forgery.export_onnx {name}")
        return False
    return True


def main():
    os.chdir(Path(__file__).resolve().parent)
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("warning: no GPU found, training will be very slow")

    ensure_manifest(args)
    df = pd.read_csv(args.manifest)
    x = load_arrays(df, args.input, args.size, args.quality)
    y = df["label"].to_numpy()
    idx = {s: np.flatnonzero(df["split"] == s) for s in ("train", "val", "test")}
    train_dl = make_loader(x[idx["train"]], y[idx["train"]], args.batch_size, True, device)
    val_dl = make_loader(x[idx["val"]], y[idx["val"]], args.batch_size, False, device)
    test_dl = make_loader(x[idx["test"]], y[idx["test"]], args.batch_size, False, device)

    model = build_model(args.model, args.size, pretrained=not args.from_scratch).to(device)
    pos = int(y[idx["train"]].sum())
    neg = len(idx["train"]) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, dtype=torch.float32, device=device))

    lr = args.lr or (1e-5 if args.model == "small_cnn" else 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    name = args.name or f"{args.model}_{args.input}_{args.size}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.pt"

    history = []
    best_auc, best_epoch, stale = 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            yb = yb.to(device)
            with amp(device):
                logits = model(prepare_batch(xb, device, train=True)).squeeze(1)
            loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        sched.step()

        yv, pv = predict(model, val_dl, device)
        val = summarize(yv, pv)
        train_loss = total / len(idx["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val["auc"], "val_f1": val["f1"], "val_acc": val["accuracy"]})
        print(f"epoch {epoch:02d}  loss {train_loss:.4f}  val auc {val['auc']:.4f}  f1 {val['f1']:.4f}  {time.time() - start:.0f}s")

        if val["auc"] > best_auc:
            best_auc, best_epoch, stale = val["auc"], epoch, 0
            torch.save({"state_dict": model.state_dict()}, ckpt)
        else:
            stale += 1
            if stale >= args.patience:
                print("early stop")
                break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    yv, pv = predict(model, val_dl, device)
    threshold = best_threshold(yv, pv)
    yt, pt = predict(model, test_dl, device)
    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "val": summarize(yv, pv, threshold),
        "test": summarize(yt, pt, threshold),
        "test_at_0.5": summarize(yt, pt, 0.5),
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": args.model,
            "size": args.size,
            "input": args.input,
            "quality": args.quality,
            "threshold": threshold,
        },
        ckpt,
    )
    print(json.dumps(results["test"], indent=2))
    print(f"Model saved to {ckpt}")

    if not args.no_export and export_model(name):
        print("Exported to the models folder. Start the API with: python -m uvicorn api.main:app")


if __name__ == "__main__":
    main()
