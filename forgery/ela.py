import io

from PIL import Image, ImageChops


def ela_image(img, quality=90):
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(rgb, resaved)
    peak = max(hi for _, hi in diff.getextrema()) or 1
    lut = [min(255, round(v * 255 / peak)) for v in range(256)]
    return diff.point(lut * 3)
