from __future__ import annotations

import numpy as np

from utils.config import GATE_THRESHOLDS

def quality_metrics(image_np: np.ndarray) -> dict:
    image = np.asarray(image_np, dtype=np.float32)
    gray = image.mean(axis=2)
    gy, gx = np.gradient(gray)
    blur = float(np.var(np.abs(gx) + np.abs(gy)))
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    quality_score = 0.25 + 0.3 * min(blur / 150.0, 1.0) + 0.2 * (1.0 - min(abs(brightness - 127.5) / 127.5, 1.0)) + 0.25 * min(contrast / 64.0, 1.0)
    return {'quality_score': round(quality_score, 4), 'blur_metric': round(blur, 4), 'brightness_metric': round(brightness, 4), 'contrast_metric': round(contrast, 4), 'usable_for_analysis': bool(quality_score >= GATE_THRESHOLDS['quality_min'])}
