import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from app.ocr.preprocessing import load_image, preprocess_for_detector

logger = logging.getLogger(__name__)

CLASS_NAMES = ["SU61855", "SU61856", "SU65551RAW", "SU66072RAW"]

_yolo_model = None


def _get_model():
    global _yolo_model
    if _yolo_model is None:
        model_path = settings.model_dir / settings.yolo_model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}.\n"
                f"Train it first with: python scripts/train_detector.py"
            )
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(str(model_path))
            logger.info("YOLOv8 model loaded from %s", model_path)
        except ImportError:
            raise RuntimeError("Run: pip install ultralytics")
    return _yolo_model


class DetectionResult:
    def __init__(self, predicted_class: Optional[str], confidence: float,
                 bbox: Optional[list], all_predictions: list[dict]):
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.bbox = bbox
        self.all_predictions = all_predictions

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "detector_confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "all_predictions": self.all_predictions,
        }

    def __repr__(self):
        return f"DetectionResult(class={self.predicted_class}, conf={self.confidence:.2f})"


def run_detection(source) -> DetectionResult:
    img = load_image(source)
    processed = preprocess_for_detector(img)

    model = _get_model()
    results = model.predict(
        processed,
        conf=settings.detector_conf_threshold,
        iou=settings.detector_iou_threshold,
        verbose=False,
    )

    all_preds: list[dict] = []
    best_class = None
    best_conf = 0.0
    best_bbox = None

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

            all_preds.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in xyxy],
            })

            if conf > best_conf:
                best_conf = conf
                best_class = cls_name
                best_bbox = [round(v, 1) for v in xyxy]

    logger.debug("Detection result: class=%s conf=%.2f", best_class, best_conf)

    return DetectionResult(
        predicted_class=best_class,
        confidence=best_conf,
        bbox=best_bbox,
        all_predictions=all_preds,
    )