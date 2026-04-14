import os
import cv2
import pickle
import numpy as np
import torch
import tempfile
from ultralytics import YOLO

IMG_SIZE = 512

def load_model():
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "model.pkl")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    tmp_det = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp_seg = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)

    torch.save(bundle["det"], tmp_det.name)
    torch.save(bundle["seg"], tmp_seg.name)

    det_model = YOLO(tmp_det.name)
    seg_model = YOLO(tmp_seg.name)

    return {
        "det": det_model,
        "seg": seg_model
    }


def is_box_outside_mask(box, mask):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(IMG_SIZE, x2), min(IMG_SIZE, y2)
    region = mask[y1:y2, x1:x2]
    return np.sum(region) == 0


def predict(model, image_path, return_image=False):

    det_model = model["det"]
    seg_model = model["seg"]

    img = cv2.imread(image_path)
    if img is None:
        return 0.0, None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    annotated = img.copy()

    # ---------------- DETECTION ----------------
    det_results = det_model(img, conf=0.45)[0]

    bin_boxes = []
    trash_boxes = []

    if det_results.boxes is not None:
        for box, cls in zip(det_results.boxes.xyxy, det_results.boxes.cls):
            label = det_model.names[int(cls)]
            box = box.cpu().numpy()

            x1, y1, x2, y2 = map(int, box)

            if label == "bin":
                bin_boxes.append(box)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(annotated, "BIN", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            elif label == "trash":
                trash_boxes.append(box)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(annotated, "TRASH", (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    if len(bin_boxes) == 0:
        return 0.0, annotated

    # ---------------- SEGMENTATION ----------------
    seg_results = seg_model(img, conf=0.25)[0]

    opening_mask = None

    if seg_results.masks is not None:
        masks = seg_results.masks.data.cpu().numpy()
        combined_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        for mask in masks:
            mask = (mask > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)

        opening_mask = combined_mask

        # overlay mask (green)
        colored_mask = np.zeros_like(annotated)
        colored_mask[:,:,1] = opening_mask * 255
        annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.4, 0)

    # ---------------- LOGIC ----------------
    result = 0.0

    if opening_mask is None:
        if len(trash_boxes) > 0:
            result = 1.0

    elif len(trash_boxes) > 0:
        for box in trash_boxes:
            if is_box_outside_mask(box, opening_mask):
                result = 1.0
                break

    return result, annotated