from __future__ import annotations

import numpy as np

def reliability_and_risk(label: str, confidence: float, agreement: float, quality_score: float, size_info: dict, shape_info: dict | None, mass_info: dict, overlap_score: float) -> dict:
    if label == 'no_tumor':
        rel = 0.35 * confidence + 0.25 * agreement + 0.2 * quality_score + 0.2 * overlap_score
        return {'severity': 'None', 'risk': 'None', 'clinical_priority': 'Routine', 'reliability_score': round(float(rel), 4), 'progression_risk': '0%', 'score': 0.0}
    sh_irr = shape_info['irregularity'] if shape_info else 0.0
    si = min(size_info['area_cm2'] / 25.0, 1.0) * 0.25 + min(size_info['diameter_cm'] / 6.0, 1.0) * 0.15 + min(size_info['tumor_percent'] / 30.0, 1.0) * 0.15 + min(size_info['volume_cm3'] / 20.0, 1.0) * 0.1 + min(sh_irr, 1.0) * 0.15 + min(mass_info['shift_mm'] / 10.0, 1.0) * 0.1 + min(overlap_score, 1.0) * 0.1
    rel = 0.3 * confidence + 0.25 * agreement + 0.2 * quality_score + 0.25 * overlap_score
    sev = 'Severe' if si >= 0.7 else 'Moderate' if si >= 0.4 else 'Mild'
    risk = 'High' if si >= 0.75 else 'Moderate' if si >= 0.45 else 'Low'
    pri = 'Urgent' if si >= 0.75 else 'Priority' if si >= 0.45 else 'Routine'
    prog = f'{int(si * 100)}%' if si >= 0.7 else f'{int(si * 80)}%' if si >= 0.4 else f'{int(si * 60)}%'
    return {'severity': sev, 'risk': risk, 'clinical_priority': pri, 'reliability_score': round(float(rel), 4), 'progression_risk': prog, 'score': round(float(si), 3)}


def confidence_cal(confidence: float) -> dict:
    u = round((1.0 - confidence) * 0.5 + 0.01, 3)
    return {'uncertainty': f'+/-{round(u * 100, 1)}%', 'cal_score': round(1.0 - u, 3)}


def get_severity_label(label: str, confidence: float, tumor_percent: float) -> str:
    if label == 'no_tumor':
        return 'None'
    if tumor_percent > 20 or confidence > 0.95:
        return 'Severe'
    if tumor_percent > 10 or confidence > 0.8:
        return 'Moderate'
    return 'Mild'


def clinical_decision(label: str, severity: str, mass_info: dict | None) -> dict:
    if label == 'no_tumor':
        return {'steps': ['Routine follow-up MRI in 12 months', 'No immediate clinical action required', 'Maintain regular neurological check-ups'], 'urgency': 'LOW'}
    steps = ['Neurosurgical consultation', 'Contrast MRI follow-up']
    if severity == 'Severe':
        steps += ['MR Spectroscopy', 'Perfusion MRI', 'Surgical planning', 'Biopsy / Histopathology']
        urg = 'CRITICAL'
    elif severity == 'Moderate':
        steps += ['MR Spectroscopy', 'Neuropsychological assessment']
        urg = 'HIGH'
    else:
        steps += ['PET scan consideration', 'Neurological evaluation']
        urg = 'MODERATE'
    if mass_info and mass_info['shift_mm'] > 3:
        steps.append('Immediate review for mass effect')
    return {'steps': steps, 'urgency': urg}


def rano_assessment(label: str, size_info: dict, shape_info: dict | None) -> dict | None:
    if label == 'no_tumor':
        return None
    sc = 'Large' if size_info['diameter_cm'] > 3 else 'Medium' if size_info['diameter_cm'] > 1.5 else 'Small'
    nec = 'Likely present' if shape_info and shape_info['irregularity'] > 0.5 else 'Not clearly identified'
    enh = 'Heterogeneous' if shape_info and shape_info['irregularity'] > 0.4 else 'Homogeneous'
    grd = 'Higher-Risk Imaging Pattern (imaging features suggest a higher-risk pattern - histopathological confirmation required)' if not (size_info['diameter_cm'] < 2 and (not shape_info or shape_info['irregularity'] < 0.3)) else 'Lower-Risk Imaging Pattern (imaging features suggest a lower-risk pattern - histopathological confirmation required)'
    return {'size_cat': sc, 'enhancement': enh, 'necrosis': nec, 'grade': grd}
