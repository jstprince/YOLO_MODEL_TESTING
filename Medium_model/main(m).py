#Langston Pearson 

#Main script for testing the medium sized yolo model

"""
Object detection with YOLOv8 medium (yolov8m) over a whole dataset (coco128
by default) instead of a single image.

Same script as the nano/small versions, with the model swapped to
yolov8m.pt. The "m" (medium) variant has ~25.9M parameters — noticeably
more accurate than small, at the cost of a larger download and slower
runtime, especially on CPU.

Setup
-----
    pip install ultralytics

The model weights (~52 MB) and the coco128 dataset (~7 MB, 128 images)
download automatically on first run, so you need an internet connection
the first time. After that it runs offline.

Usage
-----
    python yolo_medium_batch_detect.py                  # runs on coco128 (val split)
    python yolo_medium_batch_detect.py path/to/folder    # runs on your own folder of images
    python yolo_medium_batch_detect.py path/to/data.yaml # runs on a different dataset config
"""

import sys
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

MODEL_WEIGHTS = "yolov8m.pt"  # "m" = medium (25.9M params)

# coco128's own config file — Ultralytics resolves this name automatically
# and downloads the images/labels if they aren't already cached locally.
DEFAULT_DATASET = "coco128.yaml"

CONFIDENCE_THRESHOLD = 0.25  # ignore detections below this confidence


def resolve_source(arg):
    """
    Figures out what kind of source was passed in:
    - a .yaml dataset config (like coco128.yaml) -> pull the image folder from it
    - a folder path -> use directly
    - anything else -> pass straight through to YOLO (it also accepts globs, videos, etc.)
    """
    if arg.endswith(".yaml"):
        from ultralytics.data.utils import check_det_dataset
        dataset_info = check_det_dataset(arg)
        return dataset_info.get("val") or dataset_info.get("train")
    return arg


def main():
    source_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    image_source = resolve_source(source_arg)

    model = YOLO(MODEL_WEIGHTS)

    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Running detection on: {image_source}\n")

    results = model.predict(
        source=image_source,
        conf=CONFIDENCE_THRESHOLD,
        save=True,           # writes annotated copies to runs/detect/predict/
        project="runs",
        name="detect_medium",
        exist_ok=True,
        stream=True,          # process one image at a time instead of loading them all into memory
        verbose=False,
    )

    total_images = 0
    total_detections = 0
    class_counts = Counter()
    images_with_no_detections = []

    for result in results:
        total_images += 1
        image_name = Path(result.path).name
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            images_with_no_detections.append(image_name)
            continue

        total_detections += len(boxes)
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            class_counts[label] += 1

        top_labels = Counter(
            model.names[int(b.cls[0])] for b in boxes
        )
        summary = ", ".join(f"{n}x {lbl}" for lbl, n in top_labels.most_common(4))
        print(f"{image_name:<25} {len(boxes):>3} object(s) — {summary}")

    print("\n" + "=" * 60)
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Processed {total_images} images, {total_detections} total detections")
    print(f"Images with zero detections: {len(images_with_no_detections)}")

    print("\nDetections by class (all images combined):")
    for label, count in class_counts.most_common():
        print(f"  {label:<15} {count}")

    if images_with_no_detections:
        print(f"\nNo detections found in: {', '.join(images_with_no_detections[:10])}"
              + (" ..." if len(images_with_no_detections) > 10 else ""))

    print(f"\nAnnotated images saved to: {Path('runs/detect_medium').resolve()}")


if __name__ == "__main__":
    main()