"""
Prepare YOLO dataset from your labeled images.

Usage:
  python scripts/prepare_dataset.py \
      --images_dir /path/to/your/images \
      --labels_dir /path/to/label_studio_export \
      --out_dir datasets/fas_parts \
      --val_split 0.15
"""

import argparse
import json
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["SU61855", "SU61856", "SU65551RAW", "SU66072RAW"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}
MERGE_REPORT = Path("config/_label_merge_report_20260517T170952Z.json")


def load_merge_report() -> dict[str, str]:
    if not MERGE_REPORT.exists():
        return {}
    with open(MERGE_REPORT) as f:
        data = json.load(f)
    return {item["stem"]: item["variant"] for item in data.get("details", [])}


def split_and_copy(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    val_split: float,
    seed: int = 42,
):
    stem_to_variant = load_merge_report()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    all_images = [p for p in images_dir.iterdir()
                  if p.suffix.lower() in image_exts]
    random.seed(seed)
    random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_split))
    val_set = set(p.stem for p in all_images[:n_val])

    counts = {"train": 0, "val": 0, "skipped": 0}

    for img_path in all_images:
        stem = img_path.stem
        split = "val" if stem in val_set else "train"

        img_dest = out_dir / "images" / split / img_path.name
        lbl_dest = out_dir / "labels" / split / (stem + ".txt")

        img_dest.parent.mkdir(parents=True, exist_ok=True)
        lbl_dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, img_dest)

        src_label = labels_dir / (stem + ".txt")
        if src_label.exists():
            shutil.copy(src_label, lbl_dest)
        elif stem in stem_to_variant:
            variant = stem_to_variant[stem]
            cls_idx = CLASS_INDEX.get(variant)
            if cls_idx is None:
                counts["skipped"] += 1
                continue
            lbl_dest.write_text(f"{cls_idx} 0.5 0.5 1.0 1.0\n")
        else:
            counts["skipped"] += 1
            continue

        counts[split] += 1

    print(f"Dataset prepared in {out_dir}")
    print(f"  Train: {counts['train']}  Val: {counts['val']}  "
          f"Skipped: {counts['skipped']}")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(f"""path: {out_dir.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
{chr(10).join(f'  {i}: {n}' for i, n in enumerate(CLASS_NAMES))}
""")
    print(f"  data.yaml written to {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", default="datasets/fas_parts")
    parser.add_argument("--val_split", type=float, default=0.15)
    args = parser.parse_args()

    split_and_copy(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        out_dir=Path(args.out_dir),
        val_split=args.val_split,
    )