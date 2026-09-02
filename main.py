"""FastAPI service for the chicken disease classifier.

INFO

    GET  /            interactive test page
    GET  /docs        Swagger UI
    GET  /health      liveness + readiness (is the model actually loaded?)
    GET  /metadata    model card: threshold, metrics, preprocessing contract
    POST /predict     one image -> label + probability
    POST /predict/batch  up to 20 images
"""
from __future__ import annotations

import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api.inference import (ChickenDiseaseClassifier, InvalidImageError,
                           ModelNotLoadedError)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

ARTIFACTS = pathlib.Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS / os.getenv("MODEL_FILE", "chicken_model.keras")
METADATA_PATH = ARTIFACTS / os.getenv("METADATA_FILE", "metadata.json")
STATIC_DIR = pathlib.Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_BATCH = 20
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}

classifier = ChickenDiseaseClassifier(MODEL_PATH, METADATA_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request.

    A load failure does NOT crash the service: /health then reports the reason,
    which is far easier to debug than a container that exits on boot.
    """
    logger.info("loading model from %s", MODEL_PATH)
    classifier.load()
    if not classifier.is_ready:
        logger.warning("service is up but NOT ready: %s", classifier.load_error)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Chicken Disease Classifier",
    description=(
        "Binary screening of poultry fecal images: Healthy vs Diseased "
        "(Coccidiosis, Salmonella, or New Castle Disease). "
        "A triage aid, not a diagnosis — laboratory testing remains ground truth."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)



class PredictionResponse(BaseModel):
    label: str = Field(..., description="Healthy or Diseased")
    label_index: int = Field(..., description="0 = Healthy, 1 = Diseased")
    probability_diseased: float = Field(..., description="Raw sigmoid score in [0,1]")
    threshold: float = Field(..., description="Operating threshold used for this call")
    confidence: float = Field(..., description="Normalised distance from the threshold")
    inference_ms: float
    filename: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    detail: str | None = None



@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request, exc: ModelNotLoadedError):
    # 503, not 500: the service is fine, the model just isn't there yet.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request, exc: InvalidImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _read_valid_image(file: UploadFile) -> bytes:
    """Validate content type and size, and return the bytes.

    Size is checked after reading because UploadFile does not expose a reliable
    length up front; 10 MB is small enough that buffering is not a concern.
    """
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Send a JPEG, PNG, BMP or WebP.",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )
    return raw


@app.get("/", include_in_schema=False)
async def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"detail": "Test page not found. Try /docs."}, status_code=404)
    return FileResponse(page)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Readiness probe. Returns 200 with model_loaded=false when the weights are
    missing, so an orchestrator can tell 'starting' apart from 'broken'."""
    return HealthResponse(
        status="ok" if classifier.is_ready else "degraded",
        model_loaded=classifier.is_ready,
        model_name=str(classifier.metadata.get("model_name", "unknown")),
        detail=classifier.load_error,
    )


@app.get("/metadata", tags=["ops"])
async def metadata():
    """The model card: tuned threshold, test metrics, per-class breakdown, and
    the preprocessing contract clients must not violate."""
    return classifier.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Fecal image: JPEG, PNG, BMP or WebP"),
    threshold: float | None = Query(
        None, ge=0.0, le=1.0,
        description=(
            "Override the tuned operating threshold. Lower it to catch more sick "
            "birds at the cost of more false alarms."
        ),
    ),
):
    raw = await _read_valid_image(file)
    result = classifier.predict(raw, threshold)
    return PredictionResponse(**result.as_dict(), filename=file.filename)


@app.post("/predict/batch", tags=["inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description=f"Up to {MAX_BATCH} images"),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
):
    """Per-file errors are reported inline rather than failing the whole batch —
    one corrupt photo shouldn't discard the other nineteen."""
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Send at most {MAX_BATCH} images per request.")

    results, errors = [], []
    for file in files:
        try:
            raw = await _read_valid_image(file)
            results.append({**classifier.predict(raw, threshold).as_dict(), "filename": file.filename})
        except (HTTPException, InvalidImageError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            errors.append({"filename": file.filename, "error": detail})

    diseased = sum(r["label_index"] for r in results)
    return {
        "results": results,
        "errors": errors,
        "summary": {
            "processed": len(results),
            "failed": len(errors),
            "diseased": diseased,
            "healthy": len(results) - diseased,
        },
    }
