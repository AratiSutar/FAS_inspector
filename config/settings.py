from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Paths
    variant_map_path: Path = BASE_DIR / "config" / "variant_map.json"
    model_dir: Path = BASE_DIR / "models"
    yolo_model_name: str = "shape_detector.pt"

    # OCR
    ocr_lang: str = "en"
    ocr_use_angle_cls: bool = True
    ocr_use_gpu: bool = False

    # Detector
    detector_conf_threshold: float = 0.45
    detector_iou_threshold: float = 0.45
    detector_imgsz: int = 640

    # Preprocessing
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: int = 8
    sobel_ksize: int = 3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    max_image_size_mb: int = 20

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()