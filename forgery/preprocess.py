import numpy as np
from PIL import Image

from .ela import ela_image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_array(img, size):
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def to_array(img, mode, size, quality):
    img = ela_image(img, quality) if mode == "ela" else img.convert("RGB")
    return resize_array(img, size)


def to_input(arr):
    x = (arr.astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]
