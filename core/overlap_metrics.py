from __future__ import annotations

import numpy as np

def overlap_metrics(heatmap: np.ndarray, seg_mask: np.ndarray, heat_threshold: float=0.6) -> dict:
    heat_region = (heatmap >= heat_threshold).astype(np.uint8)
    lesion = (seg_mask > 0).astype(np.uint8)
    inter = int(np.logical_and(heat_region, lesion).sum())
    union = int(np.logical_or(heat_region, lesion).sum())
    hp = int(heat_region.sum())
    outside = max(hp - inter, 0)
    iou = inter / union if union else 0.0
    inside = inter / hp if hp else 0.0
    outside_pct = outside / hp if hp else 0.0
    consistency = float(iou * 0.6 + inside * 0.4)
    status = 'Validated' if consistency >= 0.55 else 'Review' if consistency >= 0.3 else 'Rejected'
    return {'overlap_score': round(float(iou), 4), 'attention_inside_lesion_percent': round(float(inside * 100.0), 2), 'outside_attention_percent': round(float(outside_pct * 100.0), 2), 'explainability_consistency': round(consistency, 4), 'validation_status': status}
