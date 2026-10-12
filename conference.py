"""
Combined Inference — Classification + Detection + Segmentation
===============================================================
Runs three models over a folder of images and writes ONE annotated image
per input, plus an optional JSON per image.

  1. ROAD TYPE   MobileNetV2 (.pt)  → dirt / concrete / tar
                 Whole-image classification (matches the training val
                 transform: Resize 224x224 + ImageNet normalization).
  2. DETECTION   YOLO               → trash, dump
                 road_damage is DROPPED by name — segmentation owns it, so
                 keeping it here would double-annotate the same object.
                 Works whether you pass a 2-class or 3-class detector.
  3. SEGMENTATION YOLO-seg          → open_drainage, road_damage

Class ownership is exclusive, so nothing is annotated twice.

Usage:
    Set the paths ONCE in the CONFIG block below, then simply:

        python combined_inference_3model.py

    Every config value can still be overridden on the command line:

        python combined_inference_3model.py --folder other/images --json
        python combined_inference_3model.py --no_road --seg_conf 0.15

Requires: torch torchvision ultralytics opencv-python
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════
# CONFIG — edit these paths once, then just run the script.
# Any CLI flag still overrides the value set here.
# ════════════════════════════════════════════════════════════

ROAD_MODEL = r"inference_check/mobilenet_classification.pt"
DET_MODEL  = r"inference_check/object_detection.pt"
SEG_MODEL  = r"inference_check/segment.pt"

INPUT_FOLDER = r"inference_check/test_images"
OUTPUT_FOLDER = r"inference_check/annotated"

# Fallback class order, used only if the checkpoint has no class_names.
# MUST match training order (ImageFolder sorts alphabetically).
ROAD_CLASSES = "concrete,dirt,tar"

DET_CONF = 0.25
SEG_CONF = 0.25
WRITE_JSON = False

# ════════════════════════════════════════════════════════════

# Detection classes that this script must NOT draw (owned by another model)
DROP_DETECTION = {"road_damage"}

# BGR colors, keyed by display name
COLORS = {
    "trash":           (0, 0, 255),      # red
    "dump":            (0, 165, 255),    # orange
    "open_drainage":   (0, 255, 0),      # green
    "road_damage":     (255, 0, 0),      # blue
}
DEFAULT_COLOR = (200, 200, 200)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════
# 1. Road-type classifier (MobileNetV2)
# ════════════════════════════════════════════════════════════

class RoadClassifier:
    def __init__(self, ckpt_path: str, fallback_classes):
        obj = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        # Accept the three shapes a .pt can arrive in
        if isinstance(obj, nn.Module):                      # whole model
            self.model = obj.to(DEVICE)
            self.classes = fallback_classes
        else:
            if isinstance(obj, dict) and "model_state" in obj:
                state = obj["model_state"]
                self.classes = obj.get("class_names") or fallback_classes
            elif isinstance(obj, dict) and "state_dict" in obj:
                state = obj["state_dict"]
                self.classes = obj.get("class_names") or fallback_classes
            else:                                            # raw state_dict
                state = obj
                self.classes = fallback_classes
            m = models.mobilenet_v2(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(nn.Dropout(p=0.3),
                                         nn.Linear(in_f, len(self.classes)))
            missing, unexpected = m.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"  ⚠ classifier state_dict mismatch — missing {len(missing)}, "
                      f"unexpected {len(unexpected)} keys. Check --road_classes "
                      f"matches how it was trained.")
            self.model = m.to(DEVICE)

        self.model.eval()
        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[Load] road classifier: {ckpt_path}")
        print(f"       classes: {self.classes}")

    @torch.no_grad()
    def predict(self, pil_img: Image.Image):
        x = self.tf(pil_img).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        i = int(probs.argmax())
        return (self.classes[i], float(probs[i]),
                {c: round(float(p), 4) for c, p in zip(self.classes, probs)})


# ════════════════════════════════════════════════════════════
# 2 & 3. YOLO detection + segmentation
# ════════════════════════════════════════════════════════════

def load_yolo(path, label):
    m = YOLO(path)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    print(f"[Load] {label}: {path}")
    print(f"       classes: {names}")
    return m, names


def run_detection(model, names, img_path, conf, class_conf):
    """Boxes, excluding any class in DROP_DETECTION."""
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    out = []
    for box in res.boxes:
        name = names.get(int(box.cls[0]), f"id{int(box.cls[0])}")
        if name in DROP_DETECTION:
            continue                       # owned by the segmentation model
        c = float(box.conf[0])
        if c < class_conf.get(name, conf):
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append({"name": name, "conf": c, "box": [x1, y1, x2, y2]})
    return out


def run_segmentation(model, names, img_path, conf, class_conf):
    res = model.predict(str(img_path), conf=conf, verbose=False)[0]
    out = []
    if res.masks is None or res.masks.xy is None:
        return out, res.orig_img
    for poly, box in zip(res.masks.xy, res.boxes):
        if len(poly) < 3:
            continue
        name = names.get(int(box.cls[0]), f"id{int(box.cls[0])}")
        c = float(box.conf[0])
        if c < class_conf.get(name, conf):
            continue
        out.append({"name": name, "conf": c,
                    "polygon": np.array(poly, dtype=np.int32)})
    return out, res.orig_img


# ════════════════════════════════════════════════════════════
# Drawing
# ════════════════════════════════════════════════════════════

def label_box(canvas, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(canvas, (x, max(y - th - 8, 0)),
                  (x + tw + 4, max(y, th + 8)), color, -1)
    cv2.putText(canvas, text, (x + 2, max(y - 5, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def annotate(img, road, dets, segs):
    canvas = img.copy()

    # segmentation first (filled), so boxes stay readable on top
    for s in segs:
        color = COLORS.get(s["name"], DEFAULT_COLOR)
        layer = canvas.copy()
        cv2.fillPoly(layer, [s["polygon"]], color)
        canvas = cv2.addWeighted(layer, 0.40, canvas, 0.60, 0)
        cv2.polylines(canvas, [s["polygon"]], True, color, 2)
        x, y = int(s["polygon"][:, 0].min()), int(s["polygon"][:, 1].min())
        label_box(canvas, f"{s['name']} {s['conf']:.2f}", x, y, color)

    for d in dets:
        color = COLORS.get(d["name"], DEFAULT_COLOR)
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_box(canvas, f"{d['name']} {d['conf']:.2f}", x1, y1, color)

    # road-type banner
    bar = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
    if road:
        txt = f"ROAD: {road[0].upper()}  ({road[1]*100:.0f}%)"
        cv2.putText(bar, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2)
    n_txt = f"{len(dets)} detections   {len(segs)} masks"
    cv2.putText(bar, n_txt, (canvas.shape[1] - 330, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, canvas])


# ════════════════════════════════════════════════════════════

def check_paths(args):
    """Fail early with a clear message rather than deep inside a model load."""
    problems = []
    if not Path(args.folder).is_dir():
        problems.append(f"  input folder not found : {args.folder}")
    for label, path in (("detection model", args.det_model),
                        ("segmentation model", args.seg_model)):
        if not Path(path).is_file():
            problems.append(f"  {label:<19}not found : {path}")
    if args.road_model and not Path(args.road_model).is_file():
        problems.append(f"  road model         not found : {args.road_model}")
    if problems:
        raise SystemExit("[ERROR] fix these paths (edit the CONFIG block at the "
                         "top of this script, or pass the matching CLI flag):\n"
                         + "\n".join(problems))


def main(args):
    check_paths(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "json"
    if args.json:
        json_dir.mkdir(parents=True, exist_ok=True)

    class_conf = {}
    for kv in args.class_conf:
        k, v = kv.split("=")
        class_conf[k.strip()] = float(v)

    road_clf = None
    if args.road_model:
        road_clf = RoadClassifier(args.road_model,
                                  [c.strip() for c in args.road_classes.split(",")])
    det_model, det_names = load_yolo(args.det_model, "detection")
    seg_model, seg_names = load_yolo(args.seg_model, "segmentation")

    dropped = [n for n in det_names.values() if n in DROP_DETECTION]
    if dropped:
        print(f"\n[Note] detection classes dropped (owned by segmentation): {dropped}")

    folder = Path(args.folder)
    targets = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"\n[Found] {len(targets)} images  |  det conf {args.det_conf}  "
          f"seg conf {args.seg_conf}\n")

    det_ct, seg_ct, road_ct = Counter(), Counter(), Counter()
    blank = 0

    for img_path in targets:
        segs, img = run_segmentation(seg_model, seg_names, img_path,
                                     args.seg_conf, class_conf)
        dets = run_detection(det_model, det_names, img_path,
                             args.det_conf, class_conf)

        road = None
        if road_clf:
            road = road_clf.predict(Image.open(img_path).convert("RGB"))
            road_ct[road[0]] += 1

        for d in dets:
            det_ct[d["name"]] += 1
        for s in segs:
            seg_ct[s["name"]] += 1
        if not dets and not segs:
            blank += 1

        canvas = annotate(img, road, dets, segs)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), canvas)

        if args.json:
            h, w = img.shape[:2]
            rec = {
                "image": img_path.name,
                "image_size": {"width": w, "height": h},
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "road_type": ({"class": road[0], "confidence": round(road[1], 4),
                               "probabilities": road[2]} if road else None),
                "detections": [{"class": d["name"],
                                "confidence": round(d["conf"], 4),
                                "bbox_xyxy": d["box"]} for d in dets],
                "segmentations": [{"class": s["name"],
                                   "confidence": round(s["conf"], 4),
                                   "polygon_xy": s["polygon"].tolist()}
                                  for s in segs],
            }
            (json_dir / f"{img_path.stem}.json").write_text(json.dumps(rec, indent=2))

        print(f"  {img_path.name}: "
              + (f"road={road[0]}  " if road else "")
              + f"{len(dets)} det, {len(segs)} seg")

    print("\n" + "=" * 52)
    print(f"  images processed     : {len(targets)}")
    print(f"  with nothing detected: {blank}")
    if road_ct:
        print(f"\n  road type:")
        for k, v in road_ct.most_common():
            print(f"    {k:<18} {v}")
    print(f"\n  detections (boxes):")
    for k, v in det_ct.most_common() or [("(none)", 0)]:
        print(f"    {k:<18} {v}")
    print(f"  segmentations (masks):")
    for k, v in seg_ct.most_common() or [("(none)", 0)]:
        print(f"    {k:<18} {v}")
    print(f"\n  annotated images → {out_dir}")
    if args.json:
        print(f"  json             → {json_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3-model combined inference")
    p.add_argument("--folder", default=INPUT_FOLDER,
                   help="Folder of input images (default from CONFIG block)")
    p.add_argument("--road_model", default=ROAD_MODEL,
                   help="MobileNetV2 .pt (default from CONFIG block)")
    p.add_argument("--no_road", action="store_true",
                   help="Skip road classification entirely")
    p.add_argument("--road_classes", default=ROAD_CLASSES,
                   help="Fallback class order if the checkpoint has no class_names. "
                        "MUST match training order (ImageFolder = alphabetical).")
    p.add_argument("--det_model", default=DET_MODEL,
                   help="YOLO detection best.pt (default from CONFIG block)")
    p.add_argument("--seg_model", default=SEG_MODEL,
                   help="YOLO-seg best.pt (default from CONFIG block)")
    p.add_argument("--out", default=OUTPUT_FOLDER)
    p.add_argument("--det_conf", type=float, default=DET_CONF)
    p.add_argument("--seg_conf", type=float, default=SEG_CONF)
    p.add_argument("--class_conf", nargs="*", default=[],
                   help="Per-class overrides, e.g. road_damage=0.15 dump=0.3")
    p.add_argument("--json", action="store_true", default=WRITE_JSON,
                   help="Also write per-image JSON")
    a = p.parse_args()
    if a.no_road:
        a.road_model = None
    main(a)