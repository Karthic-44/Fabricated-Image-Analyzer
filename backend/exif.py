from PIL import ExifTags

from .schemas import ExifInfo

EDITORS = ("photoshop", "gimp", "lightroom", "canva", "snapseed", "picsart", "pixelmator", "affinity", "paint.net")


def clean(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    value = str(value).strip("\x00 ").strip()
    return value or None


def read_exif(img):
    raw = img.getexif()
    tags = {ExifTags.TAGS.get(key, key): value for key, value in raw.items()}
    make, model = clean(tags.get("Make")), clean(tags.get("Model"))
    camera = " ".join(part for part in (make, model) if part) or None
    software = clean(tags.get("Software"))
    editor = next((name for name in EDITORS if software and name in software.lower()), None)
    return ExifInfo(
        camera=camera,
        software=software,
        taken_at=clean(tags.get("DateTime")),
        has_gps=bool(raw.get_ifd(ExifTags.IFD.GPSInfo)),
        editor_detected=editor,
    )
