from __future__ import annotations

import cv2
import numpy as np

from utils.config import IMG_SIZE, SEG_SIZE

def run_segmentation(seg_model, tmp_path: str) -> np.ndarray:
    try:
        in_shape = seg_model.input_shape
        seg_h = int(in_shape[1]) if in_shape[1] else SEG_SIZE
        seg_w = int(in_shape[2]) if in_shape[2] else SEG_SIZE
        seg_ch = int(in_shape[3]) if in_shape[3] else 1
    except Exception:
        seg_h, seg_w, seg_ch = (SEG_SIZE, SEG_SIZE, 3)
    img_bgr = cv2.imread(tmp_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image for segmentation: {tmp_path}")
    if seg_ch == 1:
        gray = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (seg_w, seg_h))
        seg_in = np.expand_dims(gray / 255.0, (0, -1))
    else:
        rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (seg_w, seg_h))
        seg_in = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)
    raw = seg_model(seg_in, training=False)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    out = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
    if out.ndim == 4:
        out = out[0]
    if out.ndim == 3:
        out = out[:, :, 0]
    mask = (out > 0.5).astype(np.uint8)
    return cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
