import argparse
from pathlib import Path

from .data import build_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out", default="data/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_manifest(args.root, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} jpeg images, {df['group'].nunique()} groups")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(df.groupby("label")["qmean"].describe()[["count", "mean", "50%"]])


if __name__ == "__main__":
    main()
