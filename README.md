# YOLO Model Comparison Project

This repository contains three YOLOv8 detection scripts that test the same dataset workflow with different model sizes:

- Nano model (`Nano_model/main.py`)
- Small model (`Small model/main(s).py`)
- Medium model (`Medium_model/main(m).py`)

Each script loads a YOLOv8 model, runs inference on the COCO128 dataset by default, and saves annotated output images under a `runs/` folder.

## Project structure

- `Nano_model/` — YOLOv8n example
- `Small model/` — YOLOv8s example
- `Medium_model/` — YOLOv8m example

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the models

```bash
python "Nano_model/main.py"
python "Small model/main(s).py"
python "Medium_model/main(m).py"
```

## Notes

- Model weights and dataset files are intentionally ignored by Git to keep the repository lightweight.
- The first run may download the YOLO weights and COCO128 dataset automatically.
- Annotated output images are saved to each model folder under `runs/`.
