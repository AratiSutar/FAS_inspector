import re
import json
import logging
from typing import Optional

from config.settings import settings
from app.ocr.preprocessing import preprocess_for_ocr, load_image

logger = logging.getLogger(__name__)

from config.variant_loader import VARIANT_MAP as _VARIANT_MAP

_TOKEN_RULES: list[tuple[re.Pattern, str, int]] = []
for variant_key, info in _VARIANT_MAP["variants"].items():
    for token in info["ocr_tokens"]:
        priority = len(token)
        pattern = re.compile(re.escape(token), re.IGNORECASE)
        _TOKEN_RULES.append((pattern, variant_key, priority))

_TOKEN_RULES.sort(key=lambda x: x[2], reverse=True)

_paddle_ocr = None


def _get_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(
                use_angle_cls=settings.ocr_use_angle_cls,
                lang=settings.ocr_lang,
                use_gpu=settings.ocr_use_gpu,
            )
            logger.info("PaddleOCR initialized")
        except ImportError:
            raise RuntimeError("Run: pip install paddleocr paddlepaddle")
    return _paddle_ocr


class OCRResult:
    def __init__(self, raw_texts: list[str], matched_variant: Optional[str],
                 matched_token: Optional[str], confidence: float):
        self.raw_texts = raw_texts
        self.matched_variant = matched_variant
        self.matched_token = matched_token
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "raw_texts": self.raw_texts,
            "matched_variant": self.matched_variant,
            "matched_token": self.matched_token,
            "ocr_confidence": round(self.confidence, 4),
        }

    def __repr__(self):
        return f"OCRResult(variant={self.matched_variant}, token={self.matched_token}, conf={self.confidence:.2f})"


def run_ocr(source) -> OCRResult:
    img = load_image(source)
    processed = preprocess_for_ocr(img)

    ocr = _get_ocr()
    result = ocr.ocr(processed, cls=settings.ocr_use_angle_cls)

    raw_texts: list[str] = []
    confidences: dict[str, float] = {}

    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2:
                text, conf = line[1]
                raw_texts.append(text)
                confidences[text] = float(conf)

    logger.debug("OCR raw texts: %s", raw_texts)

    best_variant = None
    best_token = None
    best_conf = 0.0
    best_priority = -1

    full_text = " ".join(raw_texts).upper()

    for pattern, variant_key, priority in _TOKEN_RULES:
        if priority <= best_priority:
            continue
        m = pattern.search(full_text)
        if m:
            token_conf = _find_confidence(m.group(0), raw_texts, confidences)
            best_variant = variant_key
            best_token = m.group(0)
            best_priority = priority
            best_conf = token_conf

    return OCRResult(
        raw_texts=raw_texts,
        matched_variant=best_variant,
        matched_token=best_token,
        confidence=best_conf,
    )


def _find_confidence(matched: str, raw_texts: list[str],
                     confidences: dict[str, float]) -> float:
    matched_upper = matched.upper()
    for text in raw_texts:
        if matched_upper in text.upper():
            return confidences.get(text, 0.0)
    return 0.0