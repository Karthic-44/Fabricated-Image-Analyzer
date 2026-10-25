from typing import Literal

from pydantic import BaseModel


class ExifInfo(BaseModel):
    camera: str | None = None
    software: str | None = None
    taken_at: str | None = None
    has_gps: bool = False
    editor_detected: str | None = None


class AnalyzeResponse(BaseModel):
    label: Literal["tampered", "authentic"]
    probability: float
    threshold: float
    reliable: bool
    warnings: list[str]
    ela_png_base64: str
    exif: ExifInfo
    width: int
    height: int
    format: str
    inference_ms: float
