import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .detector import Detector
from .exif import read_exif
from .schemas import AnalyzeResponse

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
PREVIEW_SIDE = 1024

Image.MAX_IMAGE_PIXELS = 40_000_000


@asynccontextmanager
async def lifespan(app):
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    app.state.detector = Detector(model_dir) if (model_dir / "model.onnx").exists() else None
    yield


app = FastAPI(title="Fabricated Image Analyzer", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request):
    detector = request.app.state.detector
    return {"status": "ok", "model_loaded": detector is not None, "model": detector.config if detector else None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: Request, file: UploadFile = File(...)):
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(503, "No model is loaded on the server")

    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "The file could not be read as an image")
    if img.format not in ALLOWED_FORMATS:
        raise HTTPException(415, "Only JPEG, PNG and WebP images are supported")

    exif = read_exif(img)
    started = time.perf_counter()
    probability, ela = detector.predict(img)
    elapsed_ms = (time.perf_counter() - started) * 1000

    uses_ela = detector.config["input"] == "ela"
    reliable = img.format == "JPEG" or not uses_ela
    warnings = []
    if not reliable:
        warnings.append(f"{img.format} files carry no JPEG compression history, so this score is not reliable")

    ela.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    buf = io.BytesIO()
    ela.save(buf, "PNG")

    threshold = detector.config["threshold"]
    return AnalyzeResponse(
        label="tampered" if probability >= threshold else "authentic",
        probability=probability,
        threshold=threshold,
        reliable=reliable,
        warnings=warnings,
        ela_png_base64=base64.b64encode(buf.getvalue()).decode(),
        exif=exif,
        width=img.width,
        height=img.height,
        format=img.format,
        inference_ms=round(elapsed_ms, 1),
    )
