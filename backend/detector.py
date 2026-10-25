import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from forgery.ela import ela_image
from forgery.preprocess import resize_array, to_input


class Detector:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "model.json").read_text())
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img):
        ela = ela_image(img, self.config["quality"])
        source = ela if self.config["input"] == "ela" else img.convert("RGB")
        x = to_input(resize_array(source, self.config["size"]))
        logit = float(self.session.run(None, {self.input_name: x})[0].reshape(-1)[0])
        prob = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        return float(prob), ela
