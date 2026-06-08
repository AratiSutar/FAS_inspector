import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.ocr.engine import run_ocr
from app.detector.shape_detector import run_detection
from app.verifier.cross_verifier import verify
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FAS Inspector Hercules starting up...")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="FAS Inspector Hercules",
    description="Cast-iron part OCR + shape verification pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "service": "fas-inspector-hercules"}


@app.get("/variants", tags=["system"])
async def list_variants():
    from config.variant_loader import VARIANT_MAP
    return {
        k: {
            "display_name": v["display_name"],
            "ocr_tokens": v["ocr_tokens"],
            "shape_class": v["shape_class"],
        }
        for k, v in VARIANT_MAP["variants"].items()
    }


@app.post("/inspect", tags=["inspection"])
async def inspect(file: UploadFile = File(...)):
    import asyncio

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {settings.max_image_size_mb} MB)."
        )

    t0 = time.perf_counter()

    ocr_task = asyncio.to_thread(run_ocr, image_bytes)
    det_task = asyncio.to_thread(run_detection, image_bytes)
    results = await asyncio.gather(ocr_task, det_task, return_exceptions=True)

    ocr_result, det_result = results[0], results[1]

    if isinstance(ocr_result, Exception):
        logger.exception("OCR failed: %s", ocr_result)
        raise HTTPException(status_code=500, detail="OCR processing failed.")

    if isinstance(det_result, FileNotFoundError):
        raise HTTPException(status_code=503, detail="Model not found. Train the detector first.")

    if isinstance(det_result, Exception):
        logger.exception("Detection failed: %s", det_result)
        raise HTTPException(status_code=500, detail="Detection processing failed.")

    verification = verify(ocr_result, det_result)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        "filename": file.filename,
        "processing_ms": elapsed_ms,
        "verification": verification.to_dict(),
        "ocr": ocr_result.to_dict(),
        "detection": det_result.to_dict(),
    }

    return JSONResponse(content=response, status_code=200)