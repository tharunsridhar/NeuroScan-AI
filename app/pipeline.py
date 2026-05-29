from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from PIL import Image

from core.classifier_fusion import adaptive_model_fusion, classify_image
from core.diagnostic_reliability import reliability_gate
from core.gradcam import get_gradcam
from core.morphology import analyze_shape, estimate_size, mass_effect
from core.overlap_metrics import overlap_metrics
from core.quality_metrics import quality_metrics
from core.risk_engine import clinical_decision, confidence_cal, rano_assessment, reliability_and_risk
from core.segmentation import run_segmentation
from reporting.llm_report_generator import groq_normal_report, groq_tumor_report
from reporting.pdf_report_generator import build_overlays, pdf_normal, pdf_tumor
from utils.config import GROQ_API_KEY, IMG_SIZE, REPORT_DIR, load_models
from utils.history_manager import compare_with_prior, load_history, save_history


@lru_cache(maxsize=1)
def get_loaded_models():
    return load_models()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "raw_probs"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _save_history(history, source_name, patient_name, label, confidence, severity, area_cm2, risk, reliability):
    save_history(
        {
            "id": f"#MRI-{len(history)+1:03d}",
            "filename": source_name,
            "patient": patient_name or "-",
            "label": label,
            "confidence": f"{confidence*100:.1f}%",
            "severity": severity,
            "area_cm2": area_cm2,
            "risk": risk,
            "reliability": reliability,
            "date": datetime.now().strftime("%b %d, %Y %H:%M"),
        }
    )


def _fallback_normal_report(confidence: float, patient_name: str = "") -> str:
    patient = f" for {patient_name}" if patient_name else ""
    return (
        "1. CLINICAL INDICATION\n"
        f"AI screening analysis{patient} was requested for the supplied brain MRI image.\n\n"
        "2. IMAGING TECHNIQUE\n"
        "Single-image MRI decision-support analysis was performed with multi-model classification.\n\n"
        "3. FINDINGS\n"
        f"No tumor class was selected by the classifier with {confidence:.2%} confidence.\n\n"
        "4. IMPRESSION\n"
        "No intracranial tumor is detected by this screening model.\n\n"
        "5. RISK STRATIFICATION\n"
        "The automated risk level is low, subject to clinical correlation and radiologist review.\n\n"
        "6. RECOMMENDATIONS\n"
        "Use this result only as decision support. A qualified clinician should review the image and symptoms."
    )


def _fallback_tumor_report(label: str, confidence: float, size_info: dict, severity: str, patient_name: str = "") -> str:
    patient = f" for {patient_name}" if patient_name else ""
    return (
        "1. CLINICAL INDICATION\n"
        f"AI screening analysis{patient} was requested for the supplied brain MRI image.\n\n"
        "2. IMAGING TECHNIQUE\n"
        "Single-image MRI decision-support analysis was performed with classification, segmentation, and Grad-CAM.\n\n"
        "3. FINDINGS\n"
        f"The model predicts {label.replace('_', ' ')} with {confidence:.2%} confidence. "
        f"Estimated area is {size_info.get('area_cm2')} cm2 and diameter is {size_info.get('diameter_cm')} cm.\n\n"
        "4. IMPRESSION\n"
        f"The automated severity estimate is {severity}. This is not a definitive diagnosis.\n\n"
        "5. RISK STRATIFICATION\n"
        "Risk should be interpreted with clinical context, image quality, and specialist review.\n\n"
        "6. RECOMMENDATIONS\n"
        "Radiology review and histopathological confirmation are required before clinical decisions."
    )


def _normal_report(image_path: str, confidence: float, patient_name: str) -> str:
    if GROQ_API_KEY:
        return groq_normal_report(image_path, confidence, GROQ_API_KEY, patient_name)
    return _fallback_normal_report(confidence, patient_name)


def _tumor_report(image_path: str, label: str, confidence: float, size_info: dict, shape_info: dict | None, mass_info: dict, risk_info: dict, severity: str, rano: dict | None, patient_name: str) -> str:
    if GROQ_API_KEY:
        return groq_tumor_report(image_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, GROQ_API_KEY, patient_name)
    return _fallback_tumor_report(label, confidence, size_info, severity, patient_name)


def analyze_mri(image_pil: Image.Image, source_name: str, tmp_path: str, patient_name: str = "", patient_id: str = "") -> dict:
    image_np = np.array(image_pil.convert("RGB"))
    quality = quality_metrics(image_np)

    classifiers, seg_model, base_model, head_layers = get_loaded_models()
    fusion, model_scores = classify_image(classifiers, image_pil)
    label = fusion["final_class"]
    confidence = fusion["fused_confidence"]

    history = load_history()
    orig = cv2.imread(tmp_path)
    if orig is None:
        raise ValueError("Could not read uploaded image with OpenCV")
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

    if label == "no_tumor":
        llm = _normal_report(tmp_path, confidence, patient_name)
        pdf_path = pdf_normal(source_name, tmp_path, confidence, llm, orig, REPORT_DIR, patient_name=patient_name, patient_id=patient_id, fusion=fusion, quality=quality)
        _save_history(history, source_name, patient_name, "No Tumor", confidence, "None", "N/A", "None", quality["quality_score"])
        return _jsonable(
            {
                "no_tumor": True,
                "label": label,
                "confidence": confidence,
                "severity": "None",
                "filename": source_name,
                "patient_name": patient_name,
                "patient_id": patient_id,
                "report": llm,
                "pdf_path": str(pdf_path),
                "pdf_file": os.path.basename(str(pdf_path)),
                "fusion": fusion,
                "quality": quality,
                "model_scores": model_scores,
                "clinical": {"urgency": "LOW", "steps": ["Routine follow-up in 12 months", "No immediate action required"]},
            }
        )

    mask = run_segmentation(seg_model, tmp_path)
    import tensorflow as tf

    input_arr = np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    input_tensor = tf.keras.applications.efficientnet_v2.preprocess_input(input_arr)
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_tensor, 0))
    heatmap = get_gradcam(input_tensor, base_model, head_layers)

    size_info = estimate_size(mask)
    shape_info = analyze_shape(mask)
    mass_info = mass_effect(mask)
    hmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    overlap = overlap_metrics(hmap_resized, mask)
    lesion_context = {
        "area_cm2": size_info["area_cm2"],
        "diameter_cm": size_info["diameter_cm"],
        "irregularity": 0.0 if shape_info is None else shape_info["irregularity"],
        "overlap_score": overlap["overlap_score"],
    }
    adaptive_fusion = adaptive_model_fusion(fusion["raw_probs"], quality["quality_score"], lesion_context=lesion_context)
    label = adaptive_fusion["final_class"]
    confidence = adaptive_fusion["fused_confidence"]
    risk_info = reliability_and_risk(label, confidence, adaptive_fusion["agreement_score"], quality["quality_score"], size_info, shape_info, mass_info, overlap["overlap_score"])
    gate_info = reliability_gate(quality, adaptive_fusion, overlap, risk_info)
    cal_info = confidence_cal(confidence)
    severity = risk_info["severity"]
    clinical = clinical_decision(label, severity, mass_info)
    rano = rano_assessment(label, size_info, shape_info)
    comparison = compare_with_prior(history, source_name, size_info["area_cm2"])
    overlays = build_overlays(orig, mask, hmap_resized, size_info)
    llm = _tumor_report(tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, patient_name)
    pdf_path = pdf_tumor(source_name, tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, cal_info, severity, clinical, rano, llm, orig, overlays["mask_ov"], overlays["hmap_r"], overlays["gcam_ov"], REPORT_DIR, patient_name, patient_id, adaptive_fusion, quality, overlap, comparison, gate_info)
    _save_history(history, source_name, patient_name, label.capitalize(), confidence, severity, size_info["area_cm2"], risk_info["risk"], risk_info["reliability_score"])

    return _jsonable(
        {
            "no_tumor": False,
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "filename": source_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "size_info": size_info,
            "shape_info": shape_info,
            "mass_info": mass_info,
            "risk_info": risk_info,
            "cal_info": cal_info,
            "clinical": clinical,
            "rano": rano,
            "report": llm,
            "pdf_path": str(pdf_path),
            "pdf_file": os.path.basename(str(pdf_path)),
            "fusion": adaptive_fusion,
            "quality": quality,
            "overlap": overlap,
            "gate_info": gate_info,
            "comparison": comparison,
            "model_scores": model_scores,
        }
    )


def save_upload_to_temp(contents: bytes, filename: str) -> str:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(contents)
        return tmp_file.name
