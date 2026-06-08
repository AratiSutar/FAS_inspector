import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from config.settings import settings
from app.ocr.engine import OCRResult
from app.detector.shape_detector import DetectionResult

logger = logging.getLogger(__name__)

try:
    with open(settings.variant_map_path) as f:
        _VARIANT_MAP = json.load(f)
except FileNotFoundError:
    raise RuntimeError(
        f"variant_map.json not found at {settings.variant_map_path}. "
        f"Make sure config/variant_map.json exists."
    )


class VerificationStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    NO_CODE = "NO_CODE"
    NO_SHAPE = "NO_SHAPE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


OCR_MIN_CONF = 0.55
DETECTOR_MIN_CONF = 0.45


@dataclass
class VerificationResult:
    status: VerificationStatus
    ocr_variant: Optional[str]
    detected_shape: Optional[str]
    ocr_confidence: float
    detector_confidence: float
    message: str
    raw_ocr_texts: list[str] = field(default_factory=list)
    all_detections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "ocr_variant": self.ocr_variant,
            "detected_shape": self.detected_shape,
            "ocr_confidence": round(self.ocr_confidence, 4),
            "detector_confidence": round(self.detector_confidence, 4),
            "message": self.message,
            "raw_ocr_texts": self.raw_ocr_texts,
            "all_detections": self.all_detections,
        }

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.MATCH


def verify(ocr_result: OCRResult, det_result: DetectionResult) -> VerificationResult:
    ocr_variant = ocr_result.matched_variant
    ocr_conf = ocr_result.confidence
    det_class = det_result.predicted_class
    det_conf = det_result.confidence

    if det_class is None:
        return VerificationResult(
            status=VerificationStatus.NO_SHAPE,
            ocr_variant=ocr_variant,
            detected_shape=None,
            ocr_confidence=ocr_conf,
            detector_confidence=det_conf,
            message="No part shape detected. Check image framing and lighting.",
            raw_ocr_texts=ocr_result.raw_texts,
            all_detections=det_result.all_predictions,
        )

    if ocr_variant is None:
        return VerificationResult(
            status=VerificationStatus.NO_CODE,
            ocr_variant=None,
            detected_shape=det_class,
            ocr_confidence=0.0,
            detector_confidence=det_conf,
            message=f"Shape detected as {det_class} but no variant code found by OCR. "
                    f"Raw texts: {ocr_result.raw_texts}",
            raw_ocr_texts=ocr_result.raw_texts,
            all_detections=det_result.all_predictions,
        )

    if ocr_conf < OCR_MIN_CONF or det_conf < DETECTOR_MIN_CONF:
        low = []
        if ocr_conf < OCR_MIN_CONF:
            low.append(f"OCR ({ocr_conf:.2f} < {OCR_MIN_CONF})")
        if det_conf < DETECTOR_MIN_CONF:
            low.append(f"detector ({det_conf:.2f} < {DETECTOR_MIN_CONF})")
        return VerificationResult(
            status=VerificationStatus.LOW_CONFIDENCE,
            ocr_variant=ocr_variant,
            detected_shape=det_class,
            ocr_confidence=ocr_conf,
            detector_confidence=det_conf,
            message=f"Low confidence in: {', '.join(low)}. Manual review required.",
            raw_ocr_texts=ocr_result.raw_texts,
            all_detections=det_result.all_predictions,
        )

    variant_info = _VARIANT_MAP["variants"].get(ocr_variant)
    if variant_info is None:
        return VerificationResult(
            status=VerificationStatus.NO_CODE,
            ocr_variant=ocr_variant,
            detected_shape=det_class,
            ocr_confidence=ocr_conf,
            detector_confidence=det_conf,
            message=f"OCR token '{ocr_variant}' not found in variant map.",
            raw_ocr_texts=ocr_result.raw_texts,
            all_detections=det_result.all_predictions,
        )
    expected_shape = variant_info["shape_class"]

    if expected_shape == det_class:
        status = VerificationStatus.MATCH
        msg = (f"OK — Code '{ocr_result.matched_token}' maps to {ocr_variant} "
               f"and shape confirmed as {det_class}.")
    else:
        status = VerificationStatus.MISMATCH
        msg = (f"MISMATCH — Code '{ocr_result.matched_token}' expects shape "
               f"{expected_shape} but detector found {det_class}. "
               f"Possible mis-cast or wrong part on line.")

    return VerificationResult(
        status=status,
        ocr_variant=ocr_variant,
        detected_shape=det_class,
        ocr_confidence=ocr_conf,
        detector_confidence=det_conf,
        message=msg,
        raw_ocr_texts=ocr_result.raw_texts,
        all_detections=det_result.all_predictions,
    )