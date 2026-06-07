import cv2
import numpy as np
from pathlib import Path
from config.settings import settings


def load_image(source) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot load image: {source}")
        return img
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes")
        return img
    elif isinstance(source, np.ndarray):
        return source.copy()
    raise TypeError(f"Unsupported image source type: {type(source)}")


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=settings.clahe_clip_limit,
        tileGridSize=(settings.clahe_tile_grid, settings.clahe_tile_grid),
    )
    return clahe.apply(gray)


def apply_sobel(gray: np.ndarray) -> np.ndarray:
    ksize = settings.sobel_ksize
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sx**2 + sy**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray)
    sobel = apply_sobel(clahe)
    blended = cv2.addWeighted(clahe, 0.7, sobel, 0.3, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


def preprocess_for_detector(img: np.ndarray) -> np.ndarray:
    target = settings.detector_imgsz
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target - new_h
    pad_w = target - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded