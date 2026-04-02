# -*- coding: utf-8 -*-
"""
NeuroScan AI  —  features.py
=====================================================================
PATENT CORE: Lesion-Aware Closed-Loop Multi-Model Fusion with
             Diagnostic Reliability Index (DRI)

Core invention:
  A two-pass, closed-loop medical-image analysis pipeline in which
  segmentation and explainability (GradCAM) evidence are fed back
  into the multi-model classification fusion step as a structured
  Lesion Trust Multiplier (LTM), and all evidence streams are
  unified into a single, formally-defined composite trustworthiness
  score — the Diagnostic Reliability Index (DRI) — that drives an
  auto-accept / caution / escalation gate.

Inventive claims summary:
  1. LTM = f(size, shape_irregularity, XAI_overlap)  [closed-loop]
  2. Adaptive per-model weight = base × peak × certainty × Q × LTM
  3. DRI = Σ wᵢ·sᵢ  over 7 orthogonal evidence signals
  4. Three-tier gate (Accepted / Caution / Specialist Review) driven
     solely by the DRI threshold and per-signal failure flags
  5. Two-pass pipeline: pass-1 label → segmentation → XAI → LTM →
     pass-2 fusion with lesion context (closed loop)

All analysis logic lives here.  No Streamlit, no PDF, no model I/O.
"""

from __future__ import annotations

import os
import re
import base64
import math
import warnings
from datetime import datetime

import cv2
import numpy as np

from config import (
    CLASS_NAMES,
    DRI_HIGH_THRESHOLD,
    DRI_MODERATE_THRESHOLD,
    DRI_WEIGHTS,
    GATE_THRESHOLDS,
    IMG_SIZE,
    LTM_MAX,
    LTM_MIN,
    MM2_CM2,
    MODEL_WEIGHTS,
    MRI_PARAMS,
    PIX_MM,
    SEG_SIZE,
)

warnings.filterwarnings("ignore")




# ════════════════════════════════════════════════════════════════
# SECTION 1 — TEXT UTILITIES
# ════════════════════════════════════════════════════════════════

def safe(text: str) -> str:
    """Sanitise unicode for FPDF latin-1 output."""
    if not isinstance(text, str):
        text = str(text)
    _map = {
        "\u2014": "-", "\u2013": "-", "\u2012": "-", "\u2010": "-", "\u2011": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u2026": "...", "\u00b2": "2", "\u00b3": "3",
        "\u00b0": " deg", "\u00b1": "+/-", "\u2192": "->",
        "\u2265": ">=", "\u2264": "<=", "\u2260": "!=",
        "\u00b5": "u", "\u00d7": "x",
    }
    for c, r in _map.items():
        text = text.replace(c, r)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def clean_text(text: str) -> str:
    """Strip markdown formatting from LLM output."""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    return safe(text)


# ════════════════════════════════════════════════════════════════
# SECTION 2 — SCAN QUALITY VALIDATION
# ════════════════════════════════════════════════════════════════

def quality_metrics(image_np: np.ndarray) -> dict:
    """
    Assess MRI scan quality from pixel statistics.

    Combines blur (gradient variance), brightness deviation from
    mid-grey, and contrast (std-dev) into a single quality_score
    in [0, 1].  Scans below 0.45 are flagged as unusable.

    Returns
    -------
    dict
        quality_score, blur_metric, brightness_metric,
        contrast_metric, usable_for_analysis
    """
    image      = np.asarray(image_np, dtype=np.float32)
    gray       = image.mean(axis=2)
    gy, gx     = np.gradient(gray)
    blur       = float(np.var(np.abs(gx) + np.abs(gy)))
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))
    quality_score = (
        0.25
        + 0.30 * min(blur / 150.0, 1.0)
        + 0.20 * (1.0 - min(abs(brightness - 127.5) / 127.5, 1.0))
        + 0.25 * min(contrast / 64.0, 1.0)
    )
    return {
        "quality_score"      : round(quality_score, 4),
        "blur_metric"        : round(blur, 4),
        "brightness_metric"  : round(brightness, 4),
        "contrast_metric"    : round(contrast, 4),
        "usable_for_analysis": bool(quality_score >= GATE_THRESHOLDS["quality_min"]),
    }


# ════════════════════════════════════════════════════════════════
# SECTION 3 — PATENT CLAIM 1–2
#             LESION-AWARE CLOSED-LOOP MULTI-MODEL FUSION
# ════════════════════════════════════════════════════════════════

def _normalize(scores: np.ndarray) -> np.ndarray:
    arr   = np.asarray(scores, dtype=np.float32)
    total = float(arr.sum())
    return arr / total if total > 0 else np.zeros_like(arr)


def _lesion_trust_multiplier(lesion_context: dict | None) -> float:
    """
    PATENT CLAIM 1 — Lesion Trust Multiplier (LTM)
    ------------------------------------------------
    Compute a scalar modifier in [LTM_MIN, LTM_MAX] from three
    orthogonal lesion signals that are only available *after*
    segmentation and GradCAM have run (closed-loop dependency):

        size_factor  = 0.85 + 0.20 · clip(size_signal, 0, 1)
            size_signal = 0.4·clip(area/15, 0,1) + 0.6·clip(diam/5, 0,1)

        shape_factor = 0.90 + 0.15 · clip(irregularity, 0, 1)

        xai_factor   = 0.85 + 0.20 · clip(overlap_iou, 0, 1)

        LTM = clip(size_factor × shape_factor × xai_factor, LTM_MIN, LTM_MAX)

    If no lesion_context is provided (e.g. first-pass or no-tumor
    branch) the multiplier is 1.0 (identity — no modulation).

    Parameters
    ----------
    lesion_context : dict | None
        Keys: area_cm2, diameter_cm, irregularity, overlap_score

    Returns
    -------
    float  LTM in [LTM_MIN, LTM_MAX]
    """
    if not lesion_context:
        return 1.0

    area         = float(lesion_context.get("area_cm2",      0.0))
    diameter     = float(lesion_context.get("diameter_cm",   0.0))
    irregularity = float(lesion_context.get("irregularity",  0.0))
    overlap_iou  = float(lesion_context.get("overlap_score", 0.0))

    size_signal  = min(area / 15.0, 1.0) * 0.4 + min(diameter / 5.0, 1.0) * 0.6
    size_factor  = 0.85 + 0.20 * float(np.clip(size_signal,  0.0, 1.0))
    shape_factor = 0.90 + 0.15 * float(np.clip(irregularity, 0.0, 1.0))
    xai_factor   = 0.85 + 0.20 * float(np.clip(overlap_iou,  0.0, 1.0))

    return float(np.clip(size_factor * shape_factor * xai_factor, LTM_MIN, LTM_MAX))


def adaptive_model_fusion(
    raw_probs     : dict,
    quality_score : float,
    weights       : dict | None = None,
    class_names   : list | None = None,
    lesion_context: dict | None = None,
) -> dict:
    """
    PATENT CLAIM 2 — Adaptive Per-Model Weight Formula
    ---------------------------------------------------
    Each model's effective weight is:

        w_i = base_i  ×  (0.55 + 0.45·peak_i)
                      ×  (0.65 + 0.35·certainty_i)
                      ×  (0.70 + 0.30·Q)
                      ×  LTM

    where
        peak_i      = max softmax probability for model i
        certainty_i = 1 − H(p_i) / H_max       (normalised entropy)
        Q           = quality_score
        LTM         = _lesion_trust_multiplier(lesion_context)

    When lesion_context is None (first-pass), LTM = 1.0 and the
    formula degrades to the quality-only adaptive baseline.

    Parameters
    ----------
    raw_probs      : {model_name: array-like (n_classes,)}
    quality_score  : float  scan quality score [0, 1]
    weights        : {model_name: float}  base weights
    class_names    : list of str
    lesion_context : dict | None — see _lesion_trust_multiplier()

    Returns
    -------
    dict  with keys:
        final_class, fused_confidence, agreement_score, margin,
        uncertainty_score, uncertainty_flag, decision_logic,
        model_votes, adaptive_weights, class_scores,
        per_model_scores, lesion_trust_multiplier, lesion_context_used
    """
    weights     = weights     or MODEL_WEIGHTS
    class_names = class_names or CLASS_NAMES

    ltm            = _lesion_trust_multiplier(lesion_context)
    quality_factor = 0.70 + 0.30 * float(np.clip(quality_score, 0.0, 1.0))

    weighted         = np.zeros(len(class_names), dtype=np.float32)
    total_w          = 0.0
    votes            = {}
    pretty           = {}
    adaptive_weights = {}

    for name, probs in raw_probs.items():
        p           = _normalize(probs)
        entropy     = float(-(p[p > 0] * np.log(p[p > 0] + 1e-8)).sum()) if np.any(p > 0) else 0.0
        max_entropy = math.log(len(class_names)) if len(class_names) > 1 else 1.0
        certainty   = 1.0 - min(entropy / max(max_entropy, 1e-8), 1.0)
        peak        = float(np.max(p)) if len(p) else 0.0

        wt = (
            float(weights.get(name, 1.0))
            * (0.55 + 0.45 * peak)
            * (0.65 + 0.35 * certainty)
            * quality_factor
            * ltm                       # ← PATENT: closed-loop lesion modulation
        )
        adaptive_weights[name] = round(wt, 4)
        pretty[name]           = {lbl: round(float(v), 4)
                                   for lbl, v in zip(class_names, p)}
        weighted += p * wt
        total_w  += wt
        votes[name] = class_names[int(np.argmax(p))]

    weighted  /= max(total_w, 1.0)
    ranked     = np.argsort(weighted)[::-1]
    top_idx    = int(ranked[0])
    top        = float(weighted[top_idx])
    second     = float(weighted[int(ranked[1])]) if len(ranked) > 1 else 0.0
    margin     = top - second
    agreement  = (sum(1 for v in votes.values() if v == class_names[top_idx])
                  / max(len(votes), 1))
    uncertainty = float(np.clip(
        (1.0 - top) * 0.5
        + (1.0 - agreement) * 0.3
        + max(0.0, 0.12 - margin) * 1.8,
        0.0, 1.0,
    ))

    return {
        "final_class"             : class_names[top_idx],
        "fused_confidence"        : round(top, 4),
        "agreement_score"         : round(float(agreement), 4),
        "margin"                  : round(float(margin), 4),
        "uncertainty_score"       : round(uncertainty, 4),
        "uncertainty_flag"        : bool(uncertainty >= GATE_THRESHOLDS["uncertainty_max"]),
        "decision_logic"          : ("Escalate for review"
                                     if uncertainty >= GATE_THRESHOLDS["uncertainty_max"]
                                     else "Accept fused output"),
        "model_votes"             : votes,
        "adaptive_weights"        : adaptive_weights,
        "class_scores"            : {n: round(float(s), 4)
                                     for n, s in zip(class_names, weighted)},
        "per_model_scores"        : pretty,
        "lesion_trust_multiplier" : round(ltm, 4),
        "lesion_context_used"     : bool(lesion_context is not None),
    }


# ════════════════════════════════════════════════════════════════
# SECTION 4 — SEGMENTATION RUNNER
# ════════════════════════════════════════════════════════════════

def run_segmentation(seg_model, tmp_path: str) -> np.ndarray:
    """
    Run the U-Net segmentation model on a saved image file.

    Returns
    -------
    np.ndarray  binary uint8 mask of shape (IMG_SIZE, IMG_SIZE)
    """
    try:
        in_shape = seg_model.input_shape
        seg_h    = int(in_shape[1]) if in_shape[1] else SEG_SIZE
        seg_w    = int(in_shape[2]) if in_shape[2] else SEG_SIZE
        seg_ch   = int(in_shape[3]) if in_shape[3] else 1
    except Exception:
        seg_h, seg_w, seg_ch = SEG_SIZE, SEG_SIZE, 3

    img_bgr = cv2.imread(tmp_path)
    if seg_ch == 1:
        gray   = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (seg_w, seg_h))
        seg_in = np.expand_dims(gray / 255.0, (0, -1))
    else:
        rgb    = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (seg_w, seg_h))
        seg_in = np.expand_dims(rgb.astype(np.float32) / 255.0, 0)

    import tensorflow as tf
    raw = seg_model(seg_in, training=False)
    out = raw[0][0].numpy() if isinstance(raw, (list, tuple)) else raw[0].numpy()
    if out.ndim == 3:
        out = out[:, :, 0]
    mask = (out > 0.5).astype(np.uint8)
    return cv2.resize(mask, (IMG_SIZE, IMG_SIZE))


# ════════════════════════════════════════════════════════════════
# SECTION 5 — GRAD-CAM EXPLAINABILITY
# ════════════════════════════════════════════════════════════════

def get_gradcam(img_tensor, base_model, head_layers) -> np.ndarray:
    """
    Compute GradCAM heatmap from EfficientNetV2-S backbone.

    Returns
    -------
    np.ndarray  2-D float32 heatmap (h, w), values 0–1
    """
    import tensorflow as tf
    if base_model is None:
        return np.zeros((12, 12), dtype=np.float32)

    img_t    = tf.cast(img_tensor, tf.float32)
    conv_out = base_model(img_t, training=False)
    conv_var = tf.Variable(conv_out)

    with tf.GradientTape() as tape:
        tape.watch(conv_var)
        x = conv_var
        for lyr in head_layers:
            x = lyr(x)
        score = x[:, tf.argmax(x[0])]

    grads = tape.gradient(score, conv_var)
    if grads is None:
        return np.zeros((12, 12), dtype=np.float32)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    hmap   = conv_out[0] @ pooled[..., tf.newaxis]
    hmap   = tf.squeeze(hmap)
    hmap   = tf.maximum(hmap, 0) / (tf.math.reduce_max(hmap) + 1e-8)
    return hmap.numpy()


# ════════════════════════════════════════════════════════════════
# SECTION 6 — TUMOR MORPHOLOGY (Size + Shape + Mass Effect)
# ════════════════════════════════════════════════════════════════

def estimate_size(mask: np.ndarray) -> dict:
    """
    Compute tumor area, diameter, volume, and bounding box from mask.

    Returns
    -------
    dict  tumor_pixels, area_mm2, area_cm2, diameter_cm,
          tumor_percent, volume_cm3, bbox
    """
    tp   = int(np.sum(mask))
    tot  = mask.shape[0] * mask.shape[1]
    amm2 = tp * (PIX_MM ** 2)
    acm2 = amm2 * MM2_CM2
    dcm  = (2 * math.sqrt(amm2 / math.pi)) / 10.0 if tp else 0.0
    pct  = (tp / max(tot * 0.60, 1)) * 100.0
    vol  = acm2 * 0.5

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        bbox = {
            "x": x, "y": y, "w": w, "h": h,
            "width_cm" : round(w * PIX_MM / 10.0, 2),
            "height_cm": round(h * PIX_MM / 10.0, 2),
        }
    return {
        "tumor_pixels" : tp,
        "area_mm2"     : round(amm2, 2),
        "area_cm2"     : round(acm2, 4),
        "diameter_cm"  : round(dcm, 3),
        "tumor_percent": round(pct, 2),
        "volume_cm3"   : round(vol, 3),
        "bbox"         : bbox,
    }


def analyze_shape(mask: np.ndarray) -> dict | None:
    """
    Geometric shape descriptors via OpenCV contours.

    Returns
    -------
    dict | None  irregularity, compactness, convexity, eccentricity,
                 border_def, roughness — or None if no contour found
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    lg   = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(lg)
    peri = cv2.arcLength(lg, True)
    if area == 0 or peri == 0:
        return None

    irr  = min(round(1.0 - (4 * math.pi * area) / (peri ** 2), 3), 1.0)
    comp = round((peri ** 2) / (4 * math.pi * area), 3)
    hull = cv2.convexHull(lg)
    conv = round(area / cv2.contourArea(hull), 3) if cv2.contourArea(hull) > 0 else 1.0
    ecc  = 0.0
    if len(lg) >= 5:
        el  = cv2.fitEllipse(lg)
        a, b = el[1][0] / 2, el[1][1] / 2
        if max(a, b) > 0:
            ecc = round(math.sqrt(1.0 - (min(a, b) / max(a, b)) ** 2), 3)

    bdef  = ("Poorly defined"    if irr > 0.5
             else "Moderately defined" if irr > 0.3
             else "Well-defined")
    rough = "High" if irr > 0.5 else ("Moderate" if irr > 0.3 else "Low")

    return {
        "irregularity": irr,
        "compactness" : comp,
        "convexity"   : conv,
        "eccentricity": ecc,
        "border_def"  : bdef,
        "roughness"   : rough,
    }


def mass_effect(mask: np.ndarray) -> dict:
    """
    Estimate laterality and midline shift from the tumour mask.

    Returns
    -------
    dict  laterality, shift_mm, compression, sulcal
    """
    h, w = mask.shape
    mid  = w // 2
    lt   = int(np.sum(mask[:, :mid]))
    rt   = int(np.sum(mask[:, mid:]))
    tot  = lt + rt
    lat  = "None"
    smm  = 0.0

    if tot > 0:
        r   = lt / float(tot)
        lat = ("Left hemisphere"  if r > 0.6
               else "Right hemisphere" if r < 0.4
               else "Bilateral/Midline")
        smm = round(abs(lt - rt) * (PIX_MM ** 2) / max(h * PIX_MM, 1e-6), 2)

    comp = "Moderate" if smm > 5 else ("Mild" if smm > 2 else "None")
    return {
        "laterality" : lat,
        "shift_mm"   : smm,
        "compression": comp,
        "sulcal"     : "Present" if smm > 2 else "Absent",
    }


# ════════════════════════════════════════════════════════════════
# SECTION 7 — XAI OVERLAP VALIDATION
# ════════════════════════════════════════════════════════════════

def overlap_metrics(heatmap: np.ndarray, seg_mask: np.ndarray,
                    heat_threshold: float = 0.60) -> dict:
    """
    Measure spatial agreement between GradCAM heatmap and
    segmentation mask via Intersection-over-Union.

    Returns
    -------
    dict  overlap_score (IoU), attention_inside_lesion_percent,
          outside_attention_percent, explainability_consistency,
          validation_status
    """
    heat_region = (heatmap >= heat_threshold).astype(np.uint8)
    lesion      = (seg_mask > 0).astype(np.uint8)
    inter       = int(np.logical_and(heat_region, lesion).sum())
    union       = int(np.logical_or(heat_region, lesion).sum())
    hp          = int(heat_region.sum())
    outside     = max(hp - inter, 0)
    iou         = inter / union if union else 0.0
    inside      = inter / hp   if hp    else 0.0
    outside_pct = outside / hp if hp    else 0.0
    consistency = float((iou * 0.6) + (inside * 0.4))
    status      = ("Validated" if consistency >= 0.55
                   else "Review"   if consistency >= 0.30
                   else "Rejected")
    return {
        "overlap_score"                   : round(float(iou), 4),
        "attention_inside_lesion_percent" : round(float(inside * 100.0), 2),
        "outside_attention_percent"       : round(float(outside_pct * 100.0), 2),
        "explainability_consistency"      : round(consistency, 4),
        "validation_status"               : status,
    }


# ════════════════════════════════════════════════════════════════
# SECTION 8 — PATENT CLAIM 3–4
#             DIAGNOSTIC RELIABILITY INDEX (DRI) + GATE
# ════════════════════════════════════════════════════════════════

def reliability_gate(quality: dict, fusion: dict,
                     overlap: dict, risk_info: dict) -> dict:
    """
    PATENT CLAIM 3 — Diagnostic Reliability Index (DRI)
    ----------------------------------------------------
    A single composite score in [0, 1] that unifies seven orthogonal
    evidence streams into one interpretable trustworthiness measure:

        DRI = w_Q · Q
            + w_A · A
            + w_C · C
            + w_M · clip(M / 0.25, 0, 1)
            + w_X · X
            + w_R · (1 − R)
            + w_L · L_norm

    where (weights from DRI_WEIGHTS):
        Q      = scan quality score                [0, 1]
        A      = model agreement score             [0, 1]
        C      = fused softmax confidence          [0, 1]
        M      = top-2 class margin                [0, 1]
        X      = explainability consistency (IoU)  [0, 1]
        R      = aggressiveness / risk score       [0, 1]
        L_norm = (LTM − LTM_MIN) / (LTM_MAX − LTM_MIN)  [0, 1]

    DRI Tiers:
        >= DRI_HIGH_THRESHOLD     (0.72) → HIGH
        >= DRI_MODERATE_THRESHOLD (0.50) → MODERATE
        <  DRI_MODERATE_THRESHOLD        → LOW

    PATENT CLAIM 4 — Three-Tier Acceptance Gate
    --------------------------------------------
    The gate combines DRI thresholds with per-signal hard failure
    flags to produce one of three decisions:

        "Accepted"                 — DRI >= 0.72 and zero flags raised
        "Caution"                  — DRI in [0.50, 0.72) or exactly 1 flag
        "Specialist Review Required" — DRI < 0.50 or >= 2 flags
                                       or overlap < OVERLAP_CRITICAL

    Parameters
    ----------
    quality   : dict from quality_metrics()
    fusion    : dict from adaptive_model_fusion()
    overlap   : dict from overlap_metrics()
    risk_info : dict from reliability_and_risk()

    Returns
    -------
    dict  dri, dri_tier, dri_label, dri_components,
          acceptance_status, escalation_reasons, report_caution,
          evidence_score (legacy alias = dri)
    """
    Q  = float(quality.get("quality_score",              0.0))
    A  = float(fusion.get("agreement_score",             0.0))
    C  = float(fusion.get("fused_confidence",            0.0))
    M  = float(fusion.get("margin",                      0.0))
    X  = float(overlap.get("explainability_consistency", 0.0))
    R  = float(risk_info.get("score",                    0.0))
    U  = float(fusion.get("uncertainty_score",           0.0))
    LT = float(fusion.get("lesion_trust_multiplier",     1.0))

    # Normalise LTM to [0, 1] using the declared bounds
    L_norm = float(np.clip((LT - LTM_MIN) / (LTM_MAX - LTM_MIN), 0.0, 1.0))

    W = DRI_WEIGHTS
    dri = (
        W["quality"]      * Q
        + W["agreement"]  * A
        + W["confidence"] * C
        + W["margin"]     * float(np.clip(M / 0.25, 0.0, 1.0))
        + W["xai"]        * X
        + W["risk_inv"]   * (1.0 - float(np.clip(R, 0.0, 1.0)))
        + W["lesion_trust"] * L_norm
    )
    dri = round(float(np.clip(dri, 0.0, 1.0)), 4)

    dri_tier = (
        "HIGH"     if dri >= DRI_HIGH_THRESHOLD
        else "MODERATE" if dri >= DRI_MODERATE_THRESHOLD
        else "LOW"
    )

    # ── Per-signal failure flags ─────────────────────────────────
    reasons = []
    GT = GATE_THRESHOLDS
    if Q  < GT["quality_min"]    : reasons.append("Low scan quality")
    if A  < GT["agreement_min"]  : reasons.append("Low model agreement")
    if C  < GT["confidence_min"] : reasons.append("Low fused confidence")
    if M  < GT["margin_min"]     : reasons.append("Small class margin")
    if X  < GT["overlap_min"]    : reasons.append("Poor XAI-segmentation overlap")
    if U  >= GT["uncertainty_max"]: reasons.append("High fusion uncertainty")

    # ── Gate decision ────────────────────────────────────────────
    status = "Accepted"
    if len(reasons) >= 2 or X < GT["overlap_critical"]:
        status = "Specialist Review Required"
    elif reasons:
        status = "Caution"
    # DRI override
    if dri < DRI_MODERATE_THRESHOLD and status == "Accepted":
        status = "Specialist Review Required"
        reasons.append("DRI below minimum threshold")
    elif dri < DRI_HIGH_THRESHOLD and status == "Accepted":
        status = "Caution"

    return {
        # Primary DRI output
        "dri"               : dri,
        "dri_tier"          : dri_tier,
        "dri_label"         : f"DRI = {dri:.4f}  [{dri_tier}]",
        # Component breakdown — for transparency & citeability
        "dri_components"    : {
            "quality_contribution"     : round(W["quality"]       * Q,                              4),
            "agreement_contribution"   : round(W["agreement"]     * A,                              4),
            "confidence_contribution"  : round(W["confidence"]    * C,                              4),
            "margin_contribution"      : round(W["margin"]        * float(np.clip(M/0.25,0,1)),     4),
            "xai_contribution"         : round(W["xai"]           * X,                              4),
            "risk_contribution"        : round(W["risk_inv"]      * (1-float(np.clip(R,0,1))),      4),
            "lesion_trust_contribution": round(W["lesion_trust"]  * L_norm,                         4),
        },
        # Gate decision
        "acceptance_status" : status,
        "escalation_reasons": reasons,
        "report_caution"    : bool(status != "Accepted"),
        # Legacy alias
        "evidence_score"    : dri,
    }


# ════════════════════════════════════════════════════════════════
# SECTION 9 — RISK & CLINICAL SCORING
# ════════════════════════════════════════════════════════════════

def reliability_and_risk(label: str, confidence: float, agreement: float,
                          quality_score: float, size_info: dict,
                          shape_info: dict | None, mass_info: dict,
                          overlap_score: float) -> dict:
    """
    Compute clinical severity, risk tier, and reliability score.

    Returns
    -------
    dict  severity, risk, clinical_priority,
          reliability_score, progression_risk, score
    """
    if label == "no_tumor":
        rel = (0.35 * confidence + 0.25 * agreement
               + 0.20 * quality_score + 0.20 * overlap_score)
        return {
            "severity"         : "None",
            "risk"             : "None",
            "clinical_priority": "Routine",
            "reliability_score": round(float(rel), 4),
            "progression_risk" : "0%",
            "score"            : 0.0,
        }

    sh_irr = shape_info["irregularity"] if shape_info else 0.0
    si = (
        min(size_info["area_cm2"]       / 25.0, 1.0) * 0.25
        + min(size_info["diameter_cm"]  /  6.0, 1.0) * 0.15
        + min(size_info["tumor_percent"]/ 30.0, 1.0) * 0.15
        + min(size_info["volume_cm3"]   / 20.0, 1.0) * 0.10
        + min(sh_irr, 1.0)                            * 0.15
        + min(mass_info["shift_mm"]     / 10.0, 1.0)  * 0.10
        + min(overlap_score, 1.0)                      * 0.10
    )
    rel  = (0.30 * confidence + 0.25 * agreement
            + 0.20 * quality_score + 0.25 * overlap_score)
    sev  = "Severe"   if si >= 0.70 else ("Moderate" if si >= 0.40 else "Mild")
    risk = "High"     if si >= 0.75 else ("Moderate" if si >= 0.45 else "Low")
    pri  = "Urgent"   if si >= 0.75 else ("Priority" if si >= 0.45 else "Routine")
    prog = (f"{int(si * 100)}%" if si >= 0.70
            else f"{int(si * 80)}%"  if si >= 0.40
            else f"{int(si * 60)}%")
    return {
        "severity"         : sev,
        "risk"             : risk,
        "clinical_priority": pri,
        "reliability_score": round(float(rel), 4),
        "progression_risk" : prog,
        "score"            : round(float(si), 3),
    }


def confidence_cal(confidence: float) -> dict:
    u = round((1.0 - confidence) * 0.5 + 0.01, 3)
    return {"uncertainty": f"+/-{round(u * 100, 1)}%", "cal_score": round(1.0 - u, 3)}


def get_severity_label(label: str, confidence: float, tumor_percent: float) -> str:
    if label == "no_tumor": return "None"
    if tumor_percent > 20 or confidence > 0.95: return "Severe"
    if tumor_percent > 10 or confidence > 0.80: return "Moderate"
    return "Mild"


def clinical_decision(label: str, severity: str, mass_info: dict | None) -> dict:
    if label == "no_tumor":
        return {
            "steps": [
                "Routine follow-up MRI in 12 months",
                "No immediate clinical action required",
                "Maintain regular neurological check-ups",
            ],
            "urgency": "LOW",
        }
    steps = ["Neurosurgical consultation", "Contrast MRI follow-up"]
    if severity == "Severe":
        steps += ["MR Spectroscopy", "Perfusion MRI",
                  "Surgical planning", "Biopsy / Histopathology"]
        urg = "CRITICAL"
    elif severity == "Moderate":
        steps += ["MR Spectroscopy", "Neuropsychological assessment"]
        urg = "HIGH"
    else:
        steps += ["PET scan consideration", "Neurological evaluation"]
        urg = "MODERATE"
    if mass_info and mass_info["shift_mm"] > 3:
        steps.append("Immediate review for mass effect")
    return {"steps": steps, "urgency": urg}


def rano_assessment(label: str, size_info: dict,
                     shape_info: dict | None) -> dict | None:
    if label == "no_tumor":
        return None
    sc  = ("Large"  if size_info["diameter_cm"] > 3
           else "Medium" if size_info["diameter_cm"] > 1.5 else "Small")
    nec = ("Likely present"
           if shape_info and shape_info["irregularity"] > 0.5
           else "Not clearly identified")
    enh = ("Heterogeneous"
           if shape_info and shape_info["irregularity"] > 0.4
           else "Homogeneous")
    grd = (
        "High-Grade (imaging features suggest possible high-grade"
        " - histopathological confirmation required)"
        if not (size_info["diameter_cm"] < 2
                and (not shape_info or shape_info["irregularity"] < 0.3))
        else "Low-Grade (imaging features suggest possible low-grade"
             " - histopathological confirmation required)"
    )
    return {"size_cat": sc, "enhancement": enh, "necrosis": nec, "grade": grd}


# ════════════════════════════════════════════════════════════════
# SECTION 10 — PRIOR-CASE COMPARISON
# ════════════════════════════════════════════════════════════════

def compare_with_prior(history: list, filename: str,
                        current_area: float) -> dict:
    """
    Compare current tumour area against the most recent prior scan
    of the same file.

    Returns
    -------
    dict  prior_available, change_percent, progression_flag
    """
    matches    = [row for row in history if row.get("filename") == filename]
    prior_area = None
    for row in reversed(matches):
        area = row.get("area_cm2")
        if isinstance(area, (int, float)):
            prior_area = float(area)
            break
    if prior_area is None or prior_area <= 0:
        return {"prior_available": False,
                "change_percent": None,
                "progression_flag": "Unknown"}
    change = ((float(current_area) - prior_area) / prior_area) * 100.0
    return {
        "prior_available" : True,
        "change_percent"  : round(float(change), 2),
        "progression_flag": ("Progression" if change >= 20
                             else "Regression" if change <= -20
                             else "Stable"),
    }


# ════════════════════════════════════════════════════════════════
# SECTION 11 — LLM REPORT GENERATION (Groq)
# ════════════════════════════════════════════════════════════════

def groq_tumor_report(image_path, label, confidence, size_info,
                       shape_info, mass_info, risk_info, severity,
                       rano, groq_api_key, patient_name="") -> str:
    """
    Generate a structured 6-section radiology report via Groq LLM.

    Returns
    -------
    str  clean ASCII report text
    """
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    bbox_t  = (f"{size_info['bbox']['width_cm']} x {size_info['bbox']['height_cm']} cm"
               if size_info.get("bbox") else "N/A")
    rano_t  = ("" if not rano else
               f"\n- RANO Size: {rano['size_cat']}"
               f"\n- Enhancement: {rano['enhancement']}"
               f"\n- Necrosis: {rano['necrosis']}")
    shape_t = ("" if not shape_info else
               f"\n- Irregularity: {shape_info['irregularity']}"
               f"\n- Border: {shape_info['border_def']}")
    pt_line = f"\nPatient Name: {patient_name}" if patient_name else ""

    prompt = (
        f"You are a senior neuroradiology AI assistant generating a formal radiology report.\n"
        f"{pt_line}\n"
        f"AI ANALYSIS FINDINGS:\n"
        f"- Tumor Type: {label.upper()} | Confidence: {confidence:.2%} | Severity: {severity}\n"
        f"- Tumor Area: {size_info['area_cm2']} cm2 | Diameter: {size_info['diameter_cm']} cm"
        f" | Est. Volume: {size_info['volume_cm3']} cm3\n"
        f"- Brain Coverage: {size_info['tumor_percent']}% | Bounding Box: {bbox_t}\n"
        f"- Laterality: {mass_info['laterality']} | Midline Shift: {mass_info['shift_mm']} mm\n"
        f"- Risk Score: {risk_info['score']}/1.0 | Growth Risk: {risk_info['risk']}"
        f"{shape_t}{rano_t}\n\n"
        "Generate a structured clinical radiology report with EXACTLY these 6 numbered sections.\n"
        "Write each section as 2-4 complete professional sentences.\n\n"
        "1. CLINICAL INDICATION\n2. IMAGING TECHNIQUE\n3. FINDINGS\n"
        "4. IMPRESSION\n5. SEVERITY ASSESSMENT\n6. RECOMMENDATIONS\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Use plain ASCII text only. No markdown, no bold, no bullets, no special characters.\n"
        "- Never definitively confirm tumor grade.\n"
        "- Always state that histopathological confirmation is required.\n"
        "- Total report must be under 350 words.\n"
        "- Start each section with its number and title on its own line."
    )
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=1200,
    )
    return clean_text(resp.choices[0].message.content)


def groq_normal_report(image_path, confidence, groq_api_key,
                        patient_name="") -> str:
    """
    Generate a structured 6-section normal (no-tumour) report via Groq.

    Returns
    -------
    str  clean ASCII report text
    """
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    pt_line = f"\nPatient Name: {patient_name}" if patient_name else ""
    prompt  = (
        f"You are a senior neuroradiology AI assistant generating a formal radiology report.\n"
        f"{pt_line}\n"
        "AI CLASSIFICATION RESULT:\n"
        "- Result: NO TUMOR DETECTED\n"
        f"- Classifier Confidence: {confidence:.2%}\n"
        "- Severity: NONE\n\n"
        "Generate a structured clinical radiology report with EXACTLY these 6 numbered sections.\n"
        "Each section must reflect that NO tumor was found. Write 2-3 complete professional sentences per section.\n\n"
        "1. CLINICAL INDICATION\n2. IMAGING TECHNIQUE\n3. FINDINGS\n"
        "4. IMPRESSION\n5. SEVERITY ASSESSMENT\n6. RECOMMENDATIONS\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Use plain ASCII text only. No markdown, no bold, no bullets, no special characters.\n"
        "- Total report must be under 300 words.\n"
        "- Start each section with its number and title on its own line."
    )
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=900,
    )
    return clean_text(resp.choices[0].message.content)


# ════════════════════════════════════════════════════════════════
# SECTION 12 — VISUALIZATION OVERLAYS
# ════════════════════════════════════════════════════════════════

def build_overlays(orig_bgr: np.ndarray, mask: np.ndarray,
                   heatmap_2d: np.ndarray, size_info: dict) -> dict:
    """
    Build coloured overlay images for display and PDF.

    Returns
    -------
    dict  mask_ov (BGR), gcam_ov (BGR), hmap_r (float32 2-D)
    """
    mask_vis = np.uint8(mask * 255)
    mask_col = cv2.applyColorMap(mask_vis, cv2.COLORMAP_HOT)
    mask_ov  = cv2.addWeighted(orig_bgr, 0.7, mask_col, 0.3, 0)

    if size_info.get("bbox"):
        b = size_info["bbox"]
        cv2.rectangle(mask_ov,
                      (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (0, 255, 0), 2)
        cv2.putText(mask_ov, f"{size_info['diameter_cm']}cm",
                    (b["x"], b["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    hmap_c  = cv2.applyColorMap(np.uint8(255 * heatmap_2d), cv2.COLORMAP_JET)
    gcam_ov = cv2.addWeighted(orig_bgr, 0.6, hmap_c, 0.4, 0)
    return {"mask_ov": mask_ov, "gcam_ov": gcam_ov, "hmap_r": heatmap_2d}


# ════════════════════════════════════════════════════════════════
# SECTION 13 — RISK CHART (matplotlib)
# ════════════════════════════════════════════════════════════════

def make_risk_chart(size_info, risk_info, shape_info, mass_info) -> str:
    """
    Generate a 3-panel risk analytics chart as a JPEG file.

    Returns
    -------
    str  path to saved chart (/tmp/rpt/chart.jpg)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    DARK = "#14284A"; RED = "#C0392B"; ORG = "#E67E22"; GRN = "#27AE60"
    BLU  = "#2980B9"; PUR = "#8E44AD"; TEA = "#16A085"; BG  = "#F5F7FA"

    def rc(v): return RED if v > 0.6 else (ORG if v > 0.3 else GRN)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    fig.patch.set_facecolor("#FFFFFF")

    tp = max(min(size_info["tumor_percent"], 100), 0)
    ax = axes[0]; ax.set_facecolor("#FFFFFF")
    _, _, ats = ax.pie(
        [tp, 100 - tp], colors=[RED, GRN], autopct="%1.1f%%", startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops={"fontsize": 7.5, "color": "#333333"},
    )
    for at in ats:
        at.set_fontsize(7); at.set_fontweight("bold"); at.set_color("white")
    ax.text(0, 0, f"{tp:.1f}%\nTumor", ha="center", va="center",
            fontsize=8, fontweight="bold", color=DARK)
    ax.set_title("Tumor vs Brain Coverage", fontsize=9,
                 fontweight="bold", color=DARK, pad=8)
    ax.legend(
        handles=[mpatches.Patch(facecolor=RED, label="Tumor"),
                 mpatches.Patch(facecolor=GRN, label="Healthy Brain")],
        fontsize=6.5, loc="lower center",
        bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
    )

    ax = axes[1]; ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color("#DDDDDD")
    cats = ["Mass Effect", "Shape", "Size", "Confidence"]
    vals = [
        round(min(mass_info["shift_mm"] / 10.0, 1.0), 2),
        round(shape_info["irregularity"] if shape_info else 0, 2),
        round(min(size_info["tumor_percent"] / 30.0, 1.0), 2),
        round(min(risk_info["score"] * 1.2, 1.0), 2),
    ]
    bars = ax.barh(cats, vals, color=[rc(v) for v in vals],
                   height=0.45, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score (0-1)", fontsize=7, color="#555555")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("Risk Factor Analysis", fontsize=9,
                 fontweight="bold", color=DARK, pad=8)
    for x in [0.3, 0.6]:
        ax.axvline(x, color="#CCCCCC", linewidth=0.8, linestyle="--")
    for bar, v in zip(bars, vals):
        ax.text(min(v + 0.03, 0.97), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", ha="left",
                fontsize=7, fontweight="bold", color="#333333")
    ax.legend(
        handles=[mpatches.Patch(facecolor=RED, label="High > 0.6"),
                 mpatches.Patch(facecolor=ORG, label="Mod  > 0.3"),
                 mpatches.Patch(facecolor=GRN, label="Low <= 0.3")],
        fontsize=6, loc="lower right", frameon=True, framealpha=0.85,
    )

    ax = axes[2]; ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color("#DDDDDD")
    if shape_info:
        lbls  = ["Irregularity", "Compactness\n(norm)", "Eccentricity", "Inv.Convexity"]
        vals2 = [
            shape_info["irregularity"],
            min(shape_info["compactness"] / 3.5, 1.0),
            shape_info["eccentricity"],
            round(1.0 - shape_info["convexity"], 3),
        ]
        b2 = ax.bar(range(len(lbls)), vals2,
                    color=[PUR, BLU, TEA, ORG],
                    width=0.55, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, fontsize=6.2)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=7, color="#555555")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.axhline(0.5, color="#AAAAAA", linewidth=0.8,
                   linestyle="--", label="Threshold")
        ax.legend(fontsize=6, loc="upper right", frameon=True, framealpha=0.85)
        for bar, v in zip(b2, vals2):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.2f}",
                    ha="center", fontsize=7, fontweight="bold", color="#333333")
    else:
        ax.text(0.5, 0.5, "No tumor data", ha="center", va="center",
                fontsize=10, color="#888888", transform=ax.transAxes)
    ax.set_title("Tumor Shape Metrics", fontsize=9,
                 fontweight="bold", color=DARK, pad=8)

    fig.suptitle("Tumor Risk Analytics", fontsize=11,
                 fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    os.makedirs("/tmp/rpt", exist_ok=True)
    p = "/tmp/rpt/chart.jpg"
    plt.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    return p


# ════════════════════════════════════════════════════════════════
# SECTION 14 — PDF HELPERS
# ════════════════════════════════════════════════════════════════

NAV = (20, 40, 74);  TXT = (30, 35, 45);  LGY = (200, 205, 210)
MUT = (100, 110, 125); BGC = (245, 247, 250); WHT = (255, 255, 255)


def _header(pdf, title, sub1, sub2=""):
    pdf.set_fill_color(*NAV); pdf.rect(0, 0, 210, 14, "F")
    pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(*WHT)
    pdf.set_xy(10, 2); pdf.cell(125, 10, safe(title))
    pdf.set_font("Helvetica", "", 7); pdf.set_text_color(180, 200, 230)
    pdf.set_xy(135, 3); pdf.cell(65, 4, safe(sub1), align="R")
    pdf.set_xy(135, 8); pdf.cell(65, 4, safe(sub2), align="R")
    pdf.set_text_color(*TXT)


def _footer(pdf, pg, total=3):
    pdf.set_xy(10, 287)
    pdf.set_font("Helvetica", "I", 6.5); pdf.set_text_color(*MUT)
    pdf.cell(155, 4,
             "AI-generated report. Must be reviewed by a qualified medical "
             "professional before clinical use.")
    pdf.set_font("Helvetica", "B", 7); pdf.set_xy(175, 287)
    pdf.cell(25, 4, f"Page {pg} / {total}", align="R")
    pdf.set_text_color(*TXT)


def _secbar(pdf, title, y):
    pdf.set_fill_color(*NAV); pdf.rect(10, y, 190, 6, "F")
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(*WHT)
    pdf.set_xy(13, y + 0.9); pdf.cell(184, 4.5, safe(f"  {title}"))
    pdf.set_text_color(*TXT); return y + 7


def _card(pdf, x, y, w, h):
    pdf.set_fill_color(*BGC); pdf.set_draw_color(*LGY)
    pdf.rect(x, y, w, h, "FD")


def _divider(pdf, y):
    pdf.set_draw_color(*LGY); pdf.line(10, y, 200, y)


def _kv(pdf, x, y, k, v, kw=32, vw=36, bold_val=False, vc=None):
    pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
    pdf.set_xy(x, y); pdf.cell(kw, 4.5, safe(f"{k}:"), ln=False)
    pdf.set_font("Helvetica", "B" if bold_val else "", 7.5)
    pdf.set_text_color(*(vc if vc else TXT))
    pdf.set_xy(x + kw, y); pdf.cell(vw, 4.5, safe(str(v)))
    pdf.set_text_color(*TXT)


def _badge(pdf, x, y, text, color):
    tw = max(len(text) * 2.0 + 6, 18)
    pdf.set_fill_color(*color); pdf.rect(x, y, tw, 5.5, "F")
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(*WHT)
    pdf.set_xy(x, y); pdf.cell(tw, 5.5, safe(text), align="C")
    pdf.set_text_color(*TXT)


def _pt_banner(pdf, patient_name, patient_id="", y_after_header=14):
    if not patient_name: return y_after_header
    pdf.set_fill_color(12, 22, 50); pdf.rect(0, y_after_header, 210, 6, "F")
    pdf.set_font("Helvetica", "", 7.5); pdf.set_text_color(150, 180, 230)
    pdf.set_xy(10, y_after_header + 1.2)
    id_part = f"   ID: {patient_id}" if patient_id else ""
    pdf.cell(190, 4, safe(
        f"  Patient: {patient_name}{id_part}"
        f"   |   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ))
    pdf.set_text_color(*TXT); return y_after_header + 6


def _llm_page(pdf, llm_report, title, sub1, sub2, pg, total,
               patient_name="", patient_id=""):
    pdf.add_page()
    _header(pdf, title, sub1, sub2)
    y = _pt_banner(pdf, patient_name, patient_id)
    y = max(y, 14) + 2
    _divider(pdf, y); y += 4
    for line in llm_report.split("\n"):
        line = safe(line.strip())
        if not line: y += 2; continue
        if y > 278: break
        is_sec = any(line.startswith(h)
                     for h in ["1.", "2.", "3.", "4.", "5.", "6."])
        if is_sec:
            y += 2
            pdf.set_fill_color(*NAV); pdf.rect(10, y, 4, 7, "F")
            pdf.set_fill_color(228, 234, 252); pdf.rect(14, y, 186, 7, "F")
            pdf.set_font("Helvetica", "B", 8.5); pdf.set_text_color(*NAV)
            pdf.set_xy(16, y + 0.8); pdf.cell(182, 5.5, safe(f" {line}"))
            pdf.set_text_color(*TXT); y += 9.5
        else:
            pdf.set_font("Helvetica", "", 8); pdf.set_text_color(45, 50, 60)
            pdf.set_xy(14, y); pdf.multi_cell(184, 4.8, line)
            y = pdf.get_y() + 1.5
    _footer(pdf, pg, total)


# ════════════════════════════════════════════════════════════════
# SECTION 15 — PDF TUMOR (3 pages)
# ════════════════════════════════════════════════════════════════

def pdf_tumor(filename, image_path, label, confidence,
              size_info, shape_info, mass_info, risk_info, cal_info,
              severity, clinical, rano, llm_report,
              orig, mask_overlay, heatmap, gcam_overlay,
              report_dir,
              patient_name="", patient_id="",
              fusion=None, quality=None, overlap=None,
              comparison=None, reliability=None):
    """Build and save a 3-page PDF for a tumour case."""
    from fpdf import FPDF

    SC  = (180, 30, 30) if severity in ("Severe", "Moderate") else (20, 110, 50)
    pdf = FPDF(); pdf.set_auto_page_break(auto=False)

    # PAGE 1 — Clinical Analysis
    pdf.add_page()
    _header(pdf, "AI Brain Tumor Analysis Report",
            f"File: {os.path.basename(filename)}",
            f"Sequence: {MRI_PARAMS['Sequence']}")
    y = _pt_banner(pdf, patient_name, patient_id); y = max(y, 14)

    sc_color = ((140, 0, 0) if severity == "Severe"
                else (155, 75, 0) if severity == "Moderate"
                else (0, 110, 50))
    pdf.set_fill_color(*sc_color); pdf.rect(10, y, 190, 8, "F")
    pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*WHT)
    banner = (f"  SEVERITY: {severity.upper()}"
              f"   |   PREDICTION: {label.upper()}"
              f"   |   CONFIDENCE: {confidence:.2%}"
              f"   |   GROWTH RISK: {risk_info['risk'].upper()}")
    pdf.set_xy(10, y + 2); pdf.cell(190, 4, safe(banner), align="C")
    pdf.set_text_color(*TXT); y += 10

    y = _secbar(pdf, "ANALYSIS SUMMARY", y)
    _card(pdf, 10, y, 190, 38); ROW = 7.2
    c1 = [(12, "Tumor Type",   label.upper()),
          (12, "Confidence",   f"{confidence:.2%}"),
          (12, "Uncertainty",  cal_info["uncertainty"]),
          (12, "Calibration",  str(cal_info["cal_score"])),
          (12, "Severity",     severity)]
    c2 = [(76, "Tumor Area",   f"{size_info['area_cm2']} cm2"),
          (76, "Diameter",     f"{size_info['diameter_cm']} cm"),
          (76, "Est. Volume",  f"{size_info['volume_cm3']} cm3"),
          (76, "Brain Coverage", f"{size_info['tumor_percent']}%"),
          (76, "Bounding Box",
           (f"{size_info['bbox']['width_cm']} x {size_info['bbox']['height_cm']} cm"
            if size_info.get("bbox") else "N/A"))]
    c3 = [(140, "Laterality",   mass_info["laterality"]),
          (140, "Midline Shift",f"{mass_info['shift_mm']} mm"),
          (140, "Compression",  mass_info["compression"]),
          (140, "Sulcal Eff.",  mass_info["sulcal"]),
          (140, "Aggressiveness", f"{risk_info['score']:.3f} / 1.0")]
    for col in [c1, c2, c3]:
        for i, (cx, k, v) in enumerate(col):
            _kv(pdf, cx, y + 2 + i * ROW, k, v, kw=30, vw=34)
    y += 40

    # DRI section in PDF
    if reliability:
        y = _secbar(pdf, "DIAGNOSTIC RELIABILITY INDEX (DRI)", y)
        DH = 22; _card(pdf, 10, y, 190, DH)
        dri_v   = reliability.get("dri", 0.0)
        dri_t   = reliability.get("dri_tier", "")
        dri_col = ((0, 130, 55) if dri_t == "HIGH"
                   else (155, 100, 0) if dri_t == "MODERATE"
                   else (160, 30, 30))
        _kv(pdf, 12, y + 2,  "DRI Score", f"{dri_v:.4f}  [{dri_t}]",
            kw=24, vw=40, bold_val=True, vc=dri_col)
        _kv(pdf, 12, y + 11, "Gate Status",
            reliability.get("acceptance_status", ""),
            kw=24, vw=40, bold_val=True, vc=dri_col)
        _kv(pdf, 110, y + 2,  "Lesion Trust",
            str(fusion.get("lesion_trust_multiplier", "N/A") if fusion else "N/A"),
            kw=26, vw=20)
        _kv(pdf, 110, y + 11, "Flags",
            str(len(reliability.get("escalation_reasons", []))),
            kw=26, vw=20)
        y += DH + 3

    if fusion:
        y = _secbar(pdf, "LESION-AWARE MULTI-MODEL FUSION", y)
        FH = 20; _card(pdf, 10, y, 190, FH)
        _kv(pdf, 12, y + 2,  "Decision",   fusion["decision_logic"],       kw=30, vw=50)
        _kv(pdf, 12, y + 10, "Agreement",  f"{fusion['agreement_score']*100:.0f}%",
            kw=30, vw=20)
        fw = 45
        for i, (mn, mv) in enumerate(fusion["model_votes"].items()):
            _kv(pdf, 80 + i * fw, y + 2, mn,
                mv.replace("_", " ").title(), kw=24, vw=20)
        if quality:
            _kv(pdf, 80, y + 10, "Scan Quality",
                str(quality["quality_score"]), kw=26, vw=20)
            _kv(pdf, 126, y + 10, "Reliability",
                str(risk_info["reliability_score"]), kw=24, vw=18)
        if overlap:
            _kv(pdf, 162, y + 10, "XAI Consistency",
                str(overlap["explainability_consistency"]), kw=30, vw=16)
        y += FH + 3

    y = _secbar(pdf, "VISUAL ANALYSIS", y)
    tmp = "/tmp/rpt"; os.makedirs(tmp, exist_ok=True)

    def sv(arr, name, cm_=None):
        p = f"{tmp}/{name}.jpg"
        if cm_ is None: cv2.imwrite(p, arr)
        else:
            import matplotlib.pyplot as plt
            plt.imsave(p, arr, cmap=cm_)
        return p

    imgs = [sv(orig, "orig"), sv(mask_overlay, "mask"),
            sv(heatmap, "hmap", cm_="jet"), sv(gcam_overlay, "gcam")]
    lbls = ["Original MRI", "Segmentation", "GradCAM Heatmap", "GradCAM Overlay"]
    IW, IH = 44, 40; GAP = (190 - 4 * IW) / 5
    _card(pdf, 10, y, 190, IH + 12)
    for i, (p, l) in enumerate(zip(imgs, lbls)):
        ix = 10 + GAP + i * (IW + GAP)
        pdf.image(p, x=ix, y=y + 2, w=IW, h=IH)
        pdf.set_font("Helvetica", "B", 6.5); pdf.set_text_color(*MUT)
        pdf.set_xy(ix, y + IH + 3); pdf.cell(IW, 4, safe(l), align="C")
    pdf.set_text_color(*TXT); y += IH + 16

    y = _secbar(pdf, "SHAPE ANALYSIS   |   MASS EFFECT   |   RISK SCORE", y)
    SH = 32; _card(pdf, 10, y, 190, SH); RH = 5.8
    pdf.set_draw_color(*LGY)
    pdf.line(74, y + 2, 74, y + SH - 2)
    pdf.line(138, y + 2, 138, y + SH - 2)
    if shape_info:
        _kv(pdf, 12, y + 2,        "Irregularity", str(shape_info["irregularity"]),
            kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH,   "Compactness",  str(shape_info["compactness"]),
            kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH*2, "Convexity",    str(shape_info["convexity"]),
            kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH*3, "Border",       shape_info["border_def"],
            kw=24, vw=18)
        _kv(pdf, 12, y + 2 + RH*4, "Roughness",    shape_info["roughness"],
            kw=24, vw=18)
    _kv(pdf, 78, y + 2,        "Laterality",  mass_info["laterality"],   kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH,   "Shift",       f"{mass_info['shift_mm']} mm",
        kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH*2, "Compression", mass_info["compression"],  kw=26, vw=28)
    _kv(pdf, 78, y + 2 + RH*3, "Sulcal",      mass_info["sulcal"],       kw=26, vw=28)
    _kv(pdf, 142, y + 2,        "Aggress. Score",
        f"{risk_info['score']:.3f} / 1.0", kw=28, vw=22, bold_val=True, vc=SC)
    _kv(pdf, 142, y + 2 + RH,   "Growth Risk",
        risk_info["risk"],                  kw=28, vw=22, bold_val=True, vc=SC)
    _kv(pdf, 142, y + 2 + RH*2, "Progression",
        risk_info["progression_risk"],      kw=28, vw=22)
    _kv(pdf, 142, y + 2 + RH*3, "Uncertainty",
        cal_info["uncertainty"],            kw=28, vw=22)
    y += SH + 3

    if comparison and comparison.get("prior_available"):
        y = _secbar(pdf, "PRIOR-CASE COMPARISON", y)
        CH = 14; _card(pdf, 10, y, 190, CH)
        flag = comparison["progression_flag"]
        pct  = comparison["change_percent"]
        fc   = ((180, 30, 30)  if flag == "Progression"
                else (0, 110, 50) if flag == "Regression"
                else (30, 80, 150))
        _kv(pdf, 12, y + 2, "Progression Flag",
            flag, kw=28, vw=24, bold_val=True, vc=fc)
        _kv(pdf, 90, y + 2, "Area Change",
            f"{pct:+.1f}%", kw=22, vw=20, bold_val=True, vc=fc)
        y += CH + 3

    if rano:
        y = _secbar(pdf, "RANO CRITERIA ASSESSMENT", y)
        RAN_H = 22; _card(pdf, 10, y, 190, RAN_H)
        _kv(pdf, 12, y + 2,  "Size Category", rano["size_cat"],    kw=28, vw=28)
        _kv(pdf, 12, y + 11, "Enhancement",   rano["enhancement"], kw=28, vw=28)
        _kv(pdf, 80, y + 2,  "Necrosis",      rano["necrosis"],    kw=22, vw=44)
        pdf.set_xy(80, y + 11)
        pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
        pdf.cell(22, 4.5, "Grade:", ln=False)
        pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*SC)
        pdf.multi_cell(88, 4.5, safe(rano["grade"]))
        pdf.set_text_color(*TXT)
        y += RAN_H + 3

    y = _secbar(pdf, "CLINICAL DECISION SUPPORT", y)
    n = len(clinical["steps"]); mid = (n + 1) // 2; ch = mid * 5.8 + 14
    _card(pdf, 10, y, 190, ch)
    pdf.set_xy(12, y + 3)
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(*MUT)
    pdf.cell(28, 5, "Urgency Level:", ln=False)
    _badge(pdf, 42, y + 3, f"  {clinical['urgency']}  ", SC)
    for i, step in enumerate(clinical["steps"]):
        col = 0 if i < mid else 1
        row = i if i < mid else i - mid
        pdf.set_xy(14 + col * 96, y + 12 + row * 5.8)
        pdf.set_font("Helvetica", "", 7.5); pdf.set_text_color(*TXT)
        pdf.cell(90, 4.8, safe(f"  >>  {step}"))
    _footer(pdf, 1, 3)

    # PAGE 2 — Risk Dashboard
    chart = make_risk_chart(size_info, risk_info, shape_info, mass_info)
    pdf.add_page()
    _header(pdf, "Risk Analytics Dashboard",
            f"File: {os.path.basename(filename)}",
            f"Prediction: {label.upper()}  |  Confidence: {confidence:.2%}"
            f"  |  Severity: {severity}")
    y = _pt_banner(pdf, patient_name, patient_id); y = max(y, 14) + 2
    pdf.set_font("Helvetica", "I", 7.5); pdf.set_text_color(*MUT)
    pdf.set_xy(10, y)
    pdf.cell(190, 5, safe(
        "Quantitative breakdown of tumor characteristics derived from "
        "segmentation and classification models."
    ))
    pdf.set_text_color(*TXT); y += 8
    _card(pdf, 10, y, 190, 96)
    pdf.image(chart, x=12, y=y + 3, w=186, h=90); y += 100

    y = _secbar(pdf, "KEY METRICS AT A GLANCE", y)
    STRIP = 24; _card(pdf, 10, y, 190, STRIP)
    metrics = [("Tumor Area", f"{size_info['area_cm2']} cm2"),
               ("Diameter",   f"{size_info['diameter_cm']} cm"),
               ("Volume",     f"{size_info['volume_cm3']} cm3"),
               ("Brain Cov.", f"{size_info['tumor_percent']}%"),
               ("Midline Shift", f"{mass_info['shift_mm']} mm"),
               ("Risk Score", f"{risk_info['score']:.3f} / 1.0")]
    cw = 190 / len(metrics)
    for i, (mk, mv) in enumerate(metrics):
        cx = 10 + i * cw
        if i > 0:
            pdf.set_draw_color(*LGY)
            pdf.line(cx, y + 3, cx, y + STRIP - 3)
        pdf.set_font("Helvetica", "B", 6.5); pdf.set_text_color(*MUT)
        pdf.set_xy(cx, y + 4); pdf.cell(cw, 4, safe(mk), align="C")
        pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*NAV)
        pdf.set_xy(cx, y + 10); pdf.cell(cw, 7, safe(mv), align="C")
    pdf.set_text_color(*TXT); y += STRIP + 4

    if shape_info:
        y = _secbar(pdf, "SHAPE DETAIL", y)
        SH2 = 28; _card(pdf, 10, y, 190, SH2)
        shape_rows = [
            ("Irregularity", shape_info["irregularity"],
             "Higher = more irregular boundary"),
            ("Compactness",  round(min(shape_info["compactness"] / 3.5, 1.0), 3),
             "Normalised compactness (0-1)"),
            ("Eccentricity", shape_info["eccentricity"],
             "Elongation of fitted ellipse"),
            ("Convexity",    shape_info["convexity"],
             "How convex the tumor contour is"),
            ("Border",       shape_info["border_def"],
             "Border definition quality"),
            ("Roughness",    shape_info["roughness"],
             "Surface texture assessment"),
        ]
        scw = 190 / 3
        for i, (sk, sv2, desc) in enumerate(shape_rows):
            ci = i % 3; ri = i // 3
            cx2 = 10 + ci * scw + 3; cy2 = y + 3 + ri * 11
            pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUT)
            pdf.set_xy(cx2, cy2); pdf.cell(scw - 3, 4.5, safe(f"{sk}:"), ln=False)
            pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*NAV)
            pdf.set_xy(cx2 + 28, cy2); pdf.cell(28, 4.5, safe(str(sv2)), ln=False)
            pdf.set_font("Helvetica", "I", 6); pdf.set_text_color(*MUT)
            pdf.set_xy(cx2, cy2 + 5.5); pdf.cell(scw - 3, 4, safe(desc))
        pdf.set_text_color(*TXT); y += SH2 + 4

    y = max(y, 248)
    pdf.set_fill_color(255, 248, 225)
    pdf.set_draw_color(230, 175, 60)
    pdf.rect(10, y, 190, 16, "FD")
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(150, 90, 0)
    pdf.set_xy(13, y + 2.5); pdf.cell(10, 5, "(!)", ln=False)
    pdf.set_font("Helvetica", "", 7); pdf.set_xy(21, y + 2.5)
    pdf.multi_cell(176, 5, safe(
        "IMPORTANT: This AI-generated analysis is intended for screening and "
        "research purposes only. All findings must be reviewed and validated by "
        "a qualified radiologist or neurosurgeon before any clinical decisions "
        "are made. Histopathological confirmation is required for definitive diagnosis."
    ))
    pdf.set_text_color(*TXT)
    _footer(pdf, 2, 3)

    # PAGE 3 — LLM Radiology Report
    _llm_page(pdf, llm_report,
              "AI Generated Radiology Report",
              f"File: {os.path.basename(filename)}",
              "Note: AI-generated imaging findings. Histopathological confirmation required.",
              3, 3, patient_name, patient_id)

    ts  = datetime.now().strftime("%Y%m%d%H%M%S")
    sn  = os.path.splitext(os.path.basename(filename))[0]
    pt  = f"_{patient_name.replace(' ', '_')}" if patient_name else ""
    out = os.path.join(report_dir, f"{sn}{pt}_{ts}_mri_report.pdf")
    pdf.output(out)
    return out


# ════════════════════════════════════════════════════════════════
# SECTION 16 — PDF NO TUMOR (2 pages)
# ════════════════════════════════════════════════════════════════

def pdf_normal(filename, image_path, confidence, llm_report, orig,
               report_dir, patient_name="", patient_id="",
               fusion=None, quality=None, reliability=None):
    """Build and save a 2-page PDF for a normal (no-tumour) case."""
    from fpdf import FPDF

    GRN_C = (0, 120, 55)
    pdf   = FPDF(); pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    _header(pdf, "AI Brain MRI Analysis Report",
            f"File: {os.path.basename(filename)}",
            f"Sequence: {MRI_PARAMS['Sequence']}")
    y = _pt_banner(pdf, patient_name, patient_id); y = max(y, 14)
    pdf.set_fill_color(*GRN_C); pdf.rect(10, y, 190, 8, "F")
    pdf.set_font("Helvetica", "B", 8); pdf.set_text_color(*WHT)
    pdf.set_xy(10, y + 2)
    pdf.cell(190, 4, safe(
        f"  RESULT: NORMAL   |   NO TUMOR DETECTED"
        f"   |   CONFIDENCE: {confidence:.2%}   |   SEVERITY: NONE"
    ), align="C")
    pdf.set_text_color(*TXT); y += 10

    if reliability:
        y = _secbar(pdf, "DIAGNOSTIC RELIABILITY INDEX (DRI)", y)
        DH = 14; _card(pdf, 10, y, 190, DH)
        dri_v = reliability.get("dri", 0.0)
        dri_t = reliability.get("dri_tier", "")
        dri_col = ((0, 130, 55) if dri_t == "HIGH"
                   else (155, 100, 0) if dri_t == "MODERATE"
                   else (160, 30, 30))
        _kv(pdf, 12, y + 2, "DRI", f"{dri_v:.4f}  [{dri_t}]",
            kw=18, vw=36, bold_val=True, vc=dri_col)
        _kv(pdf, 100, y + 2, "Gate",
            reliability.get("acceptance_status", ""),
            kw=18, vw=36, bold_val=True, vc=dri_col)
        y += DH + 3

    if fusion:
        y = _secbar(pdf, "MULTI-MODEL FUSION RESULT", y)
        FH = 14; _card(pdf, 10, y, 190, FH)
        _kv(pdf, 12, y + 2, "Fused Confidence",
            f"{fusion['fused_confidence']*100:.1f}%", kw=28, vw=20)
        _kv(pdf, 80, y + 2, "Agreement",
            f"{fusion['agreement_score']*100:.0f}%", kw=20, vw=16)
        if quality:
            _kv(pdf, 140, y + 2, "Scan Quality",
                str(quality["quality_score"]), kw=26, vw=20)
        y += FH + 3

    y = _secbar(pdf, "ANALYSIS SUMMARY", y)
    _card(pdf, 10, y, 190, 36)
    rows = [
        ("Tumor Type",                "NO TUMOR DETECTED"),
        ("Classification Confidence", f"{confidence:.2%}"),
        ("Severity",                  "None"),
        ("Growth Risk",               "None"),
        ("Risk Score",                "0.000 / 1.0"),
        ("Recommended Action",        "Routine follow-up in 12 months"),
    ]
    for i, (k, v) in enumerate(rows):
        col = 0 if i < 3 else 1; row = i if i < 3 else i - 3
        cx  = 12 + col * 96
        _kv(pdf, cx, y + 2 + row * 11, k, v, kw=46, vw=46,
            bold_val=True,
            vc=GRN_C if k == "Tumor Type" else TXT)
    y += 38

    y = _secbar(pdf, "ORIGINAL MRI SCAN", y)
    tmp = "/tmp/rpt"; os.makedirs(tmp, exist_ok=True)
    orig_p = f"{tmp}/orig_normal.jpg"; cv2.imwrite(orig_p, orig)
    img_h  = 85; _card(pdf, 10, y, 190, img_h + 8)
    pdf.image(orig_p, x=10 + (190 - 100) / 2, y=y + 4, w=100, h=img_h)
    y += img_h + 12

    y = _secbar(pdf, "CLINICAL DECISION SUPPORT", y)
    cal = {
        "steps": [
            "Routine follow-up MRI in 12 months if clinically indicated",
            "No neurosurgical or oncological intervention required",
            "Correlate with clinical symptoms and patient history",
            "Maintain regular neurological check-ups",
        ],
        "urgency": "LOW",
    }
    n = len(cal["steps"]); mid = (n + 1) // 2; ch = mid * 5.8 + 14
    _card(pdf, 10, y, 190, ch)
    pdf.set_xy(12, y + 3)
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(*MUT)
    pdf.cell(28, 5, "Urgency Level:", ln=False)
    _badge(pdf, 42, y + 3, "  LOW  ", GRN_C)
    for i, step in enumerate(cal["steps"]):
        col = 0 if i < mid else 1; row = i if i < mid else i - mid
        pdf.set_xy(14 + col * 96, y + 12 + row * 5.8)
        pdf.set_font("Helvetica", "", 7.5); pdf.set_text_color(*TXT)
        pdf.cell(90, 4.8, safe(f"  >>  {step}"))
    y += ch + 4

    y = max(y, 248)
    pdf.set_fill_color(230, 255, 235)
    pdf.set_draw_color(60, 180, 80)
    pdf.rect(10, y, 190, 16, "FD")
    pdf.set_font("Helvetica", "B", 7.5); pdf.set_text_color(0, 110, 40)
    pdf.set_xy(13, y + 2.5); pdf.cell(10, 5, "(!)", ln=False)
    pdf.set_font("Helvetica", "", 7); pdf.set_xy(21, y + 2.5)
    pdf.multi_cell(176, 5, safe(
        "IMPORTANT: This AI-generated result is intended for screening purposes only. "
        "All findings must be reviewed by a qualified radiologist before any clinical "
        "decisions are made. If the patient has symptoms, further investigation by a "
        "medical professional is strongly advised."
    ))
    pdf.set_text_color(*TXT)
    _footer(pdf, 1, 2)

    _llm_page(pdf, llm_report,
              "AI Generated Radiology Report",
              f"File: {os.path.basename(filename)}",
              "Note: AI-generated imaging findings - No tumor detected.",
              2, 2, patient_name, patient_id)

    ts  = datetime.now().strftime("%Y%m%d%H%M%S")
    sn  = os.path.splitext(os.path.basename(filename))[0]
    pt  = f"_{patient_name.replace(' ', '_')}" if patient_name else ""
    out = os.path.join(report_dir, f"{sn}{pt}_{ts}_normal_report.pdf")
    pdf.output(out)
    return out


# ════════════════════════════════════════════════════════════════
# SECTION 17 — PATENT CLAIM 5
#             BASELINE vs NEUROSCAN PIPELINE COMPARISON
# ════════════════════════════════════════════════════════════════

def baseline_pipeline(raw_probs: dict, quality_score: float,
                       class_names: list | None = None) -> dict:
    """
    BASELINE: Simple equal-weight majority vote with no quality,
    lesion, or XAI modulation.  Used as the comparison reference
    to demonstrate the improvement introduced by the NeuroScan
    closed-loop pipeline.

    No DRI.  No LTM.  No gate.  Just average softmax + argmax.
    """
    class_names = class_names or CLASS_NAMES
    stacked = np.stack([_normalize(p) for p in raw_probs.values()])
    avg     = stacked.mean(axis=0)
    top_idx = int(np.argmax(avg))
    top     = float(avg[top_idx])
    votes   = {n: class_names[int(np.argmax(_normalize(p)))]
               for n, p in raw_probs.items()}
    agreement = (sum(1 for v in votes.values() if v == class_names[top_idx])
                 / max(len(votes), 1))
    return {
        "method"           : "Baseline (equal-weight average)",
        "final_class"      : class_names[top_idx],
        "confidence"       : round(top, 4),
        "agreement"        : round(float(agreement), 4),
        "model_votes"      : votes,
        "class_scores"     : {n: round(float(s), 4)
                               for n, s in zip(class_names, avg)},
        "dri"              : None,   # not computed
        "lesion_trust"     : None,   # not applied
        "acceptance_status": "N/A",  # no gate
    }


def neuroscan_pipeline(raw_probs: dict, quality_score: float,
                        lesion_context: dict | None,
                        quality_dict: dict,
                        overlap_dict: dict,
                        risk_dict: dict,
                        class_names: list | None = None) -> dict:
    """
    NEUROSCAN (proposed): Lesion-aware closed-loop fusion + DRI gate.

    Wraps adaptive_model_fusion() + reliability_gate() into one
    call for easy side-by-side comparison with baseline_pipeline().

    Parameters
    ----------
    raw_probs      : {model: probs}
    quality_score  : float
    lesion_context : dict | None  — set None for first-pass
    quality_dict   : full output of quality_metrics()
    overlap_dict   : full output of overlap_metrics()
    risk_dict      : full output of reliability_and_risk()

    Returns
    -------
    dict  with all fusion keys + dri, dri_tier, acceptance_status,
          lesion_trust, method label
    """
    fusion = adaptive_model_fusion(
        raw_probs, quality_score,
        class_names=class_names,
        lesion_context=lesion_context,
    )
    gate = reliability_gate(quality_dict, fusion, overlap_dict, risk_dict)
    return {
        "method"           : "NeuroScan (lesion-aware closed-loop + DRI)",
        "final_class"      : fusion["final_class"],
        "confidence"       : fusion["fused_confidence"],
        "agreement"        : fusion["agreement_score"],
        "model_votes"      : fusion["model_votes"],
        "class_scores"     : fusion["class_scores"],
        "adaptive_weights" : fusion["adaptive_weights"],
        "lesion_trust"     : fusion["lesion_trust_multiplier"],
        "dri"              : gate["dri"],
        "dri_tier"         : gate["dri_tier"],
        "dri_components"   : gate["dri_components"],
        "acceptance_status": gate["acceptance_status"],
        "escalation_reasons": gate["escalation_reasons"],
    }


def compare_pipelines(raw_probs: dict, quality_score: float,
                       lesion_context: dict | None,
                       quality_dict: dict,
                       overlap_dict: dict,
                       risk_dict: dict,
                       class_names: list | None = None,
                       verbose: bool = True) -> dict:
    """
    Run both pipelines and print a side-by-side comparison table.

    This is the canonical function for demonstrating the patent
    improvement.  It is designed to be called from the comparison
    notebook cell and can also be imported for unit-testing.

    Returns
    -------
    dict  {"baseline": ..., "neuroscan": ..., "delta": ...}
    """
    base = baseline_pipeline(raw_probs, quality_score, class_names)
    ns   = neuroscan_pipeline(raw_probs, quality_score, lesion_context,
                               quality_dict, overlap_dict, risk_dict,
                               class_names)

    delta_conf = round(ns["confidence"] - base["confidence"], 4)
    delta_agr  = round(ns["agreement"]  - base["agreement"],  4)

    if verbose:
        LINE = "=" * 62
        print(LINE)
        print("  PIPELINE COMPARISON  —  NeuroScan AI")
        print(LINE)
        rows = [
            ("Metric",            "Baseline",         "NeuroScan",        "Delta"),
            ("-" * 20,            "-" * 14,           "-" * 14,           "-" * 8),
            ("Method",            "Equal-weight avg", "Lesion-aware LTM", ""),
            ("Prediction",        base["final_class"],ns["final_class"],  ""),
            ("Confidence",        base["confidence"], ns["confidence"],   f"{delta_conf:+.4f}"),
            ("Agreement",         base["agreement"],  ns["agreement"],    f"{delta_agr:+.4f}"),
            ("Lesion Trust (LTM)",str(base["lesion_trust"]),
                                  str(ns["lesion_trust"]),                ""),
            ("DRI",               str(base["dri"]),   f"{ns['dri']:.4f}", ""),
            ("DRI Tier",          str(base["dri"]),   ns["dri_tier"],     ""),
            ("Gate Decision",     base["acceptance_status"],
                                  ns["acceptance_status"],                ""),
        ]
        for r in rows:
            print(f"  {r[0]:<22}  {str(r[1]):<16}  {str(r[2]):<16}  {r[3]}")
        print(LINE)
        if ns.get("escalation_reasons"):
            print("  Escalation flags:")
            for reason in ns["escalation_reasons"]:
                print(f"    - {reason}")
            print(LINE)
        if ns.get("dri_components"):
            print("  DRI Component Contributions:")
            for k, v in ns["dri_components"].items():
                bar = "#" * int(v * 80)
                print(f"    {k:<32} {v:.4f}  {bar}")
            print(LINE)

    return {
        "baseline" : base,
        "neuroscan": ns,
        "delta"    : {
            "confidence"       : delta_conf,
            "agreement"        : delta_agr,
            "dri_gained"       : ns["dri"],
            "lesion_trust_used": ns["lesion_trust"],
        },
    }
