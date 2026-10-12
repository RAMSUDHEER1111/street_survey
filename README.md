# Production Backup — Inference Check Pipeline

This folder contains the complete, standalone production snapshot of the 360° Village Street-Survey Detection Pipeline.

## Files Included

| File | Type | Description |
|---|---|---|
| `main.py` | Python Script | CLI orchestrator (360° perspective view extraction, GPX interpolation, EXIF handling, KPI generation) |
| `inference_engine.py` | Python Module | Core inference engine (`ConvNeXtRoadClassifier`, `YoloInferenceEngine`, view annotation) |
| `convnext_road_model_v2.pth` | Weights (~111 MB) | ConvNeXt-Tiny model for road surface classification (`tar`, `concrete`, `dirt`, `noise`) |
| `detector.pt` | Weights (~20 MB) | YOLO model for solid waste detection (`trash`, `dump`) |
| `segmentor.pt` | Weights (~23 MB) | YOLO instance segmentation model for road damage and drainage (`open_drainage`, `closed_drainage`) |
| `requirements.txt` | Dependencies | Environment dependencies (`torch`, `torchvision`, `ultralytics`, `opencv-python`, etc.) |

## Quick Run Example

```bash
python main.py --input "path/to/frames" --output "path/to/output" --views penta --fps 2.0 --village "my_village"
```
