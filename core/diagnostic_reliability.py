from __future__ import annotations

import numpy as np

from utils.config import CLASS_NAMES, DRI_HIGH_THRESHOLD, DRI_MODERATE_THRESHOLD, DRI_WEIGHTS, GATE_THRESHOLDS, LTM_MAX, LTM_MIN
from core.classifier_fusion import _normalize, adaptive_model_fusion

def reliability_gate(quality: dict, fusion: dict, overlap: dict, risk_info: dict) -> dict:
    Q = float(quality.get('quality_score', 0.0))
    A = float(fusion.get('agreement_score', 0.0))
    C = float(fusion.get('fused_confidence', 0.0))
    M = float(fusion.get('margin', 0.0))
    X = float(overlap.get('explainability_consistency', 0.0))
    R = float(risk_info.get('score', 0.0))
    U = float(fusion.get('uncertainty_score', 0.0))
    LT = float(fusion.get('lesion_trust_multiplier', 1.0))
    L_norm = float(np.clip((LT - LTM_MIN) / (LTM_MAX - LTM_MIN), 0.0, 1.0))
    W = DRI_WEIGHTS
    dri = W['quality'] * Q + W['agreement'] * A + W['confidence'] * C + W['margin'] * float(np.clip(M / 0.25, 0.0, 1.0)) + W['xai'] * X + W['risk_inv'] * (1.0 - float(np.clip(R, 0.0, 1.0))) + W['lesion_trust'] * L_norm
    dri = round(float(np.clip(dri, 0.0, 1.0)), 4)
    dri_tier = 'HIGH' if dri >= DRI_HIGH_THRESHOLD else 'MODERATE' if dri >= DRI_MODERATE_THRESHOLD else 'LOW'
    reasons = []
    GT = GATE_THRESHOLDS
    if Q < GT['quality_min']:
        reasons.append('Low scan quality')
    if A < GT['agreement_min']:
        reasons.append('Low model agreement')
    if C < GT['confidence_min']:
        reasons.append('Low fused confidence')
    if M < GT['margin_min']:
        reasons.append('Small class margin')
    if X < GT['overlap_min']:
        reasons.append('Poor XAI-segmentation overlap')
    if U >= GT['uncertainty_max']:
        reasons.append('High fusion uncertainty')
    status = 'Accepted'
    if len(reasons) >= 2 or X < GT['overlap_critical']:
        status = 'Specialist Review Required'
    elif reasons:
        status = 'Caution'
    if dri < DRI_MODERATE_THRESHOLD and status == 'Accepted':
        status = 'Specialist Review Required'
        reasons.append('DRI below minimum threshold')
    elif dri < DRI_HIGH_THRESHOLD and status == 'Accepted':
        status = 'Caution'
    return {'dri': dri, 'dri_tier': dri_tier, 'dri_label': f'DRI = {dri:.4f}  [{dri_tier}]', 'dri_components': {'quality_contribution': round(W['quality'] * Q, 4), 'agreement_contribution': round(W['agreement'] * A, 4), 'confidence_contribution': round(W['confidence'] * C, 4), 'margin_contribution': round(W['margin'] * float(np.clip(M / 0.25, 0, 1)), 4), 'xai_contribution': round(W['xai'] * X, 4), 'risk_contribution': round(W['risk_inv'] * (1 - float(np.clip(R, 0, 1))), 4), 'lesion_trust_contribution': round(W['lesion_trust'] * L_norm, 4)}, 'acceptance_status': status, 'escalation_reasons': reasons, 'report_caution': bool(status != 'Accepted'), 'evidence_score': dri}


def baseline_pipeline(raw_probs: dict, quality_score: float, class_names: list | None=None) -> dict:
    class_names = class_names or CLASS_NAMES
    stacked = np.stack([_normalize(p) for p in raw_probs.values()])
    avg = stacked.mean(axis=0)
    top_idx = int(np.argmax(avg))
    top = float(avg[top_idx])
    votes = {n: class_names[int(np.argmax(_normalize(p)))] for n, p in raw_probs.items()}
    agreement = sum((1 for v in votes.values() if v == class_names[top_idx])) / max(len(votes), 1)
    return {'method': 'Baseline (equal-weight average)', 'final_class': class_names[top_idx], 'confidence': round(top, 4), 'agreement': round(float(agreement), 4), 'model_votes': votes, 'class_scores': {n: round(float(s), 4) for n, s in zip(class_names, avg)}, 'dri': None, 'lesion_trust': None, 'acceptance_status': 'N/A'}


def neuroscan_pipeline(raw_probs: dict, quality_score: float, lesion_context: dict | None, quality_dict: dict, overlap_dict: dict, risk_dict: dict, class_names: list | None=None) -> dict:
    fusion = adaptive_model_fusion(raw_probs, quality_score, class_names=class_names, lesion_context=lesion_context)
    gate = reliability_gate(quality_dict, fusion, overlap_dict, risk_dict)
    return {'method': 'NeuroScan (lesion-aware closed-loop + DRI)', 'final_class': fusion['final_class'], 'confidence': fusion['fused_confidence'], 'agreement': fusion['agreement_score'], 'model_votes': fusion['model_votes'], 'class_scores': fusion['class_scores'], 'effective_weights': fusion['effective_weights'], 'lesion_trust': fusion['lesion_trust_multiplier'], 'dri': gate['dri'], 'dri_tier': gate['dri_tier'], 'dri_components': gate['dri_components'], 'acceptance_status': gate['acceptance_status'], 'escalation_reasons': gate['escalation_reasons']}


def compare_pipelines(raw_probs: dict, quality_score: float, lesion_context: dict | None, quality_dict: dict, overlap_dict: dict, risk_dict: dict, class_names: list | None=None, verbose: bool=True) -> dict:
    base = baseline_pipeline(raw_probs, quality_score, class_names)
    ns = neuroscan_pipeline(raw_probs, quality_score, lesion_context, quality_dict, overlap_dict, risk_dict, class_names)
    delta_conf = round(ns['confidence'] - base['confidence'], 4)
    delta_agr = round(ns['agreement'] - base['agreement'], 4)
    if verbose:
        LINE = '=' * 62
        print(LINE)
        print('  PIPELINE COMPARISON  —  NeuroScan AI')
        print(LINE)
        rows = [('Metric', 'Baseline', 'NeuroScan', 'Delta'), ('-' * 20, '-' * 14, '-' * 14, '-' * 8), ('Method', 'Equal-weight avg', 'Lesion-aware LTM', ''), ('Prediction', base['final_class'], ns['final_class'], ''), ('Confidence', base['confidence'], ns['confidence'], f'{delta_conf:+.4f}'), ('Agreement', base['agreement'], ns['agreement'], f'{delta_agr:+.4f}'), ('Lesion Trust (LTM)', str(base['lesion_trust']), str(ns['lesion_trust']), ''), ('DRI', str(base['dri']), f"{ns['dri']:.4f}", ''), ('DRI Tier', str(base['dri']), ns['dri_tier'], ''), ('Gate Decision', base['acceptance_status'], ns['acceptance_status'], '')]
        for r in rows:
            print(f'  {r[0]:<22}  {str(r[1]):<16}  {str(r[2]):<16}  {r[3]}')
        print(LINE)
        if ns.get('escalation_reasons'):
            print('  Escalation flags:')
            for reason in ns['escalation_reasons']:
                print(f'    - {reason}')
            print(LINE)
        if ns.get('dri_components'):
            print('  DRI Component Contributions:')
            for k, v in ns['dri_components'].items():
                bar = '#' * int(v * 80)
                print(f'    {k:<32} {v:.4f}  {bar}')
            print(LINE)
    return {'baseline': base, 'neuroscan': ns, 'delta': {'confidence': delta_conf, 'agreement': delta_agr, 'dri_gained': ns['dri'], 'lesion_trust_used': ns['lesion_trust']}}
