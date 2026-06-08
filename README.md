# FAS Inspector Hercules

Industrial computer vision system for detecting embossed part codes on cast iron components and verifying them against expected shapes.

## What it does

1. **OCR** — extracts embossed punch code from part image (e.g. `SUG1856`, `DSU61855`)
2. **Shape detection** — classifies physical part shape using YOLOv8 (4 classes)
3. **Verification** — cross-checks OCR code against detected shape

## Part variants

| Variant | OCR tokens | Images |
|---|---|---|
| SU65551RAW | SU65551, 65551 | 33 |
| SU66072RAW | SUG6072, FAN | 47 |
| SU61856 | SUG1856, 1856 | 20 |
| SU61855 | DSU61855, 61855 | 85 |

## Project structure
FAS_inspector/
├── app/
│   ├── ocr/
│   │   ├── preprocessing.py   # CLAHE + Sobel enhancement
│   │   └── engine.py          # PaddleOCR + regex matcher
│   ├── detector/
│   │   └── shape_detector.py  # YOLOv8 inference
│   ├── verifier/
│   │   └── cross_verifier.py  # MATCH / MISMATCH logic
│   └── api/
│       └── server.py          # FastAPI endpoints
├── config/
│   ├── settings.py            # All tunable parameters
│   ├── variant_map.json       # Code → shape mapping
│   └── variant_loader.py      # Centralized map loader
├── models/
│   └── shape_detector.pt      # YOLOv8 weights (after training)
├── scripts/
│   ├── prepare_dataset.py     # Convert labels to YOLO format
│   └── train_detector.py      # Train YOLOv8 on Colab
├── tests/
│   └── test_verifier.py       # Unit + API integration tests
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── main.py
└── requirements.txt

## Local setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Training (Google Colab)

1. Upload your 185 images to Google Drive
2. Run `scripts/prepare_dataset.py` to organize into YOLO format
3. Upload dataset to Colab
4. Run `scripts/train_detector.py`
5. Download `models/shape_detector.pt`
6. Place it in your local `models/` folder

## Run the API server

```bash
python main.py
```

Server starts at `http://localhost:8000`

## API usage

### Inspect a part image

```bash
curl -X POST http://localhost:8000/inspect \
  -F "file=@/path/to/part_image.jpg"
```

### Response

```json
{
  "filename": "part_image.jpg",
  "processing_ms": 312.4,
  "verification": {
    "status": "MATCH",
    "ocr_variant": "SU61855",
    "detected_shape": "SU61855",
    "ocr_confidence": 0.9123,
    "detector_confidence": 0.8741,
    "message": "OK — Code 'DSU61855' maps to SU61855 and shape confirmed as SU61855."
  }
}
```

### Verification statuses

| Status | Meaning |
|---|---|
| `MATCH` | Code and shape agree ✓ |
| `MISMATCH` | Code and shape disagree ✗ |
| `NO_CODE` | Shape detected but no valid OCR code found |
| `NO_SHAPE` | OCR found code but no shape detected |
| `LOW_CONFIDENCE` | Below threshold, manual review needed |

### Other endpoints

```bash
# Health check
curl http://localhost:8000/health

# List known variants
curl http://localhost:8000/variants
```

## Docker

```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

## CI/CD

GitHub Actions runs on every push to `main`:
1. Runs all tests
2. Builds Docker image

## Configuration

All parameters in `config/settings.py`, overridable via `.env`:

```env
OCR_USE_GPU=false
DETECTOR_CONF_THRESHOLD=0.45
CLAHE_CLIP_LIMIT=3.0
API_PORT=8000
```

## Run tests

```bash
pytest tests/ -v
```