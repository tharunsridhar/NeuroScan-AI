from __future__ import annotations

import math
import cv2
import numpy as np

from utils.config import MM2_CM2, PIX_MM

def estimate_size(mask: np.ndarray) -> dict:
    tp = int(np.sum(mask))
    tot = mask.shape[0] * mask.shape[1]
    amm2 = tp * PIX_MM ** 2
    acm2 = amm2 * MM2_CM2
    dcm = 2 * math.sqrt(amm2 / math.pi) / 10.0 if tp else 0.0
    pct = tp / max(tot * 0.6, 1) * 100.0
    vol = acm2 * 0.5
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        bbox = {'x': x, 'y': y, 'w': w, 'h': h, 'width_cm': round(w * PIX_MM / 10.0, 2), 'height_cm': round(h * PIX_MM / 10.0, 2)}
    return {'tumor_pixels': tp, 'area_mm2': round(amm2, 2), 'area_cm2': round(acm2, 4), 'diameter_cm': round(dcm, 3), 'tumor_percent': round(pct, 2), 'volume_cm3': round(vol, 3), 'bbox': bbox}


def analyze_shape(mask: np.ndarray) -> dict | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    lg = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(lg)
    peri = cv2.arcLength(lg, True)
    if area == 0 or peri == 0:
        return None
    irr = min(round(1.0 - 4 * math.pi * area / peri ** 2, 3), 1.0)
    comp = round(peri ** 2 / (4 * math.pi * area), 3)
    hull = cv2.convexHull(lg)
    conv = round(area / cv2.contourArea(hull), 3) if cv2.contourArea(hull) > 0 else 1.0
    ecc = 0.0
    if len(lg) >= 5:
        el = cv2.fitEllipse(lg)
        a, b = (el[1][0] / 2, el[1][1] / 2)
        if max(a, b) > 0:
            ecc = round(math.sqrt(1.0 - (min(a, b) / max(a, b)) ** 2), 3)
    bdef = 'Poorly defined' if irr > 0.5 else 'Moderately defined' if irr > 0.3 else 'Well-defined'
    rough = 'High' if irr > 0.5 else 'Moderate' if irr > 0.3 else 'Low'
    return {'irregularity': irr, 'compactness': comp, 'convexity': conv, 'eccentricity': ecc, 'border_def': bdef, 'roughness': rough}


def mass_effect(mask: np.ndarray) -> dict:
    h, w = mask.shape
    mid = w // 2
    lt = int(np.sum(mask[:, :mid]))
    rt = int(np.sum(mask[:, mid:]))
    tot = lt + rt
    lat = 'None'
    smm = 0.0
    if tot > 0:
        r = lt / float(tot)
        lat = 'Left hemisphere' if r > 0.6 else 'Right hemisphere' if r < 0.4 else 'Bilateral/Midline'
        smm = round(abs(lt - rt) * PIX_MM ** 2 / max(h * PIX_MM, 1e-06), 2)
    comp = 'Moderate' if smm > 5 else 'Mild' if smm > 2 else 'None'
    return {'laterality': lat, 'shift_mm': smm, 'compression': comp, 'sulcal': 'Present' if smm > 2 else 'Absent'}
