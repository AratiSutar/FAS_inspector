"""
Train YOLOv8 shape detector on FAS Inspector dataset.
Run this on Google Colab with GPU.

Steps:
1. Upload your dataset to Google Drive
2. Open this script in Colab
3. Run it
4. Download models/shape_detector.pt
5. Place it in your local models/ folder
"""

import sys
from pathlib import Path

DATA_YAML = Path("datasets/fas_parts/data.yaml")
EPOCHS = 80
BATCH = 16
IMGSZ = 640
PROJECT = "runs/train"
RUN_NAME = "fas_shape_v1"


def train():
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run prepare_dataset.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    print(f"Training for {EPOCHS} epochs...")
    print(f"Dataset: {DATA_YAML}")
    print(f"Image size: {IMGSZ}px  Batch: {BATCH}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        patience=20,
        # Augmentation for industrial lighting variance
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        flipud=0.2,
        fliplr=0.5,
    )

    best_weights = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path("models/shape_detector.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\nBest weights saved to: {dest}")
        print("Download this file and place it in your local models/ folder.")
    else:
        print(f"Weights not found at {best_weights}")

    return results


if __name__ == "__main__":
    train()