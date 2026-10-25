import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    ckpt = torch.load(Path("runs") / args.run / "model.pt", map_location="cpu")
    size = ckpt["size"]
    model = build_model(ckpt["model"], size, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, size, size),
        str(target),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    config = {key: ckpt[key] for key in ("model", "size", "input", "quality", "threshold")}
    (out / "model.json").write_text(json.dumps(config, indent=2))

    x = np.random.rand(2, 3, size, size).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    print(f"max abs diff vs torch: {np.abs(expected - got).max():.2e}")


if __name__ == "__main__":
    main()
