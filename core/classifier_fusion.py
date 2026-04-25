from __future__ import annotations

import numpy as np

from utils.config import CLASS_NAMES, MODEL_WEIGHTS, LTM_MAX, LTM_MIN, get_model_specs


def _normalize(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    total = float(arr.sum())
    return arr / total if total > 0 else np.zeros_like(arr)


def _lesion_trust_multiplier(lesion_context: dict | None) -> float:
    if not lesion_context:
        return 1.0
    area = float(lesion_context.get('area_cm2', 0.0))
    diameter = float(lesion_context.get('diameter_cm', 0.0))
    irregularity = float(lesion_context.get('irregularity', 0.0))
    overlap_iou = float(lesion_context.get('overlap_score', 0.0))
    size_signal = min(area / 15.0, 1.0) * 0.4 + min(diameter / 5.0, 1.0) * 0.6
    size_factor = 0.85 + 0.20 * float(np.clip(size_signal, 0.0, 1.0))
    shape_factor = 0.90 + 0.15 * float(np.clip(irregularity, 0.0, 1.0))
    xai_factor = 0.85 + 0.20 * float(np.clip(overlap_iou, 0.0, 1.0))
    return float(np.clip(size_factor * shape_factor * xai_factor, LTM_MIN, LTM_MAX))


def adaptive_model_fusion(raw_probs: dict, quality_score: float, weights: dict | None = None, class_names: list | None = None, lesion_context: dict | None = None) -> dict:
    class_names = class_names or CLASS_NAMES
    weights = weights or MODEL_WEIGHTS
    lesion_trust = _lesion_trust_multiplier(lesion_context)
    fused_scores = np.zeros(len(class_names), dtype=np.float32)
    effective_weights = {}
    model_votes = {}
    for model_name, probs in raw_probs.items():
        probs = _normalize(probs)
        peak = float(np.max(probs))
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))))
        entropy_max = float(np.log(len(probs))) if len(probs) > 1 else 1.0
        certainty = 1.0 - entropy / max(entropy_max, 1e-8)
        base_weight = float(weights.get(model_name, 1.0))
        effective_weight = base_weight * (0.55 + 0.45 * peak) * (0.65 + 0.35 * certainty) * (0.70 + 0.30 * quality_score) * lesion_trust
        fused_scores += probs * effective_weight
        effective_weights[model_name] = round(effective_weight, 4)
        model_votes[model_name] = class_names[int(np.argmax(probs))]
    fused_scores = _normalize(fused_scores)
    ranked = np.argsort(fused_scores)[::-1]
    top_idx = int(ranked[0])
    second = float(fused_scores[int(ranked[1])]) if len(ranked) > 1 else 0.0
    margin = float(fused_scores[top_idx]) - second
    uncertainty_score = 1.0 - margin
    agreement = sum(1 for vote in model_votes.values() if vote == class_names[top_idx]) / max(len(model_votes), 1)
    return {
        'final_class': class_names[top_idx],
        'fused_confidence': round(float(fused_scores[top_idx]), 4),
        'agreement_score': round(float(agreement), 4),
        'uncertainty_flag': bool(margin < 0.08),
        'margin': round(float(margin), 4),
        'uncertainty_score': round(float(uncertainty_score), 4),
        'decision_logic': 'Escalate for review' if margin < 0.08 else 'Accept fused output',
        'model_votes': model_votes,
        'class_scores': {name: round(float(score), 4) for name, score in zip(class_names, fused_scores)},
        'effective_weights': effective_weights,
        'lesion_trust_multiplier': round(float(lesion_trust), 4),
    }


def preprocess_for_model(image_pil, model_name):
    model_specs = get_model_specs()
    arr = np.array(image_pil.convert('RGB').resize((384, 384)), dtype=np.float32)
    preprocess = model_specs[model_name]['preprocess']
    try:
        arr = preprocess(arr)
    except Exception:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def normalize_scores(scores):
    arr = np.asarray(scores, dtype=np.float32)
    total = float(arr.sum())
    return arr / total if total > 0 else np.zeros_like(arr)


def classify_image(classifiers, image_pil):
    raw = {}
    pretty = {}
    model_specs = get_model_specs()
    for name, model in classifiers.items():
        x = preprocess_for_model(image_pil, name)
        probs = normalize_scores(model.predict(x, verbose=0)[0])
        raw[name] = probs
        pretty[name] = {label: round(float(score), 4) for label, score in zip(CLASS_NAMES, probs)}
    weighted = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    total_weight = 0.0
    votes = {}
    for name, probs in raw.items():
        weight = model_specs[name]['weight']
        weighted += probs * weight
        total_weight += weight
        votes[name] = CLASS_NAMES[int(np.argmax(probs))]
    weighted /= max(total_weight, 1.0)
    ranked = np.argsort(weighted)[::-1]
    top_idx = int(ranked[0])
    top = float(weighted[top_idx])
    second = float(weighted[int(ranked[1])]) if len(ranked) > 1 else 0.0
    margin = top - second
    uncertainty_score = 1.0 - margin
    agreement = sum(1 for vote in votes.values() if vote == CLASS_NAMES[top_idx]) / max(len(votes), 1)
    uncertainty_flag = bool(margin < 0.08)
    return {
        'final_class': CLASS_NAMES[top_idx],
        'fused_confidence': round(top, 4),
        'agreement_score': round(float(agreement), 4),
        'uncertainty_flag': uncertainty_flag,
        'margin': round(float(margin), 4),
        'uncertainty_score': round(float(uncertainty_score), 4),
        'decision_logic': 'Escalate for review' if uncertainty_flag else 'Accept fused output',
        'model_votes': votes,
        'class_scores': {name: round(float(score), 4) for name, score in zip(CLASS_NAMES, weighted)},
        'raw_probs': raw,
    }, pretty

