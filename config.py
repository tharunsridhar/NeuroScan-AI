from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
REPORT_DIR = DOCS_DIR / "sample_outputs"
HISTORY_FILE = REPORT_DIR / "case_history.json"
SAMPLE_DIR = BASE_DIR / "sample_data"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
NGROK_TOKEN = os.getenv("NGROK_TOKEN", "")
PORT = int(os.getenv("PORT", "8501"))

IMG_SIZE = 384
SEG_SIZE = 256
PIX_MM = 0.5
MM2_CM2 = 0.01

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

MODEL_WEIGHTS = {
    "EfficientNetV2-S": 1.2,
    "MobileNetV3": 1.0,
    "ConvNeXt Tiny": 1.1,
}

MODEL_FILES = {
    "EfficientNetV2-S": "class_Tumor_v2s_clean.keras",
    "MobileNetV3": "class_Tumor_mobilenet_v3.keras",
    "ConvNeXt Tiny": "class_Tumor_convnext_tiny_tumor.keras",
}

SEG_MODEL_FILE = "Segmentation_brisc_effunet.keras"

MRI_PARAMS = {
    "Sequence": "T1-Weighted Contrast Enhanced",
    "Resolution": "384 x 384 px",
    "Slice Thickness": "5.0 mm",
    "Field Strength": "1.5 T",
    "Voxel Size": "0.5 x 0.5 x 5.0 mm",
}

DRI_WEIGHTS = {
    "quality": 0.18,
    "agreement": 0.20,
    "confidence": 0.20,
    "margin": 0.12,
    "xai": 0.18,
    "risk_inv": 0.05,
    "lesion_trust": 0.07,
}

DRI_HIGH_THRESHOLD = 0.72
DRI_MODERATE_THRESHOLD = 0.50

LTM_MIN = 0.75
LTM_MAX = 1.30

GATE_THRESHOLDS = {
    "quality_min": 0.45,
    "agreement_min": 0.50,
    "confidence_min": 0.60,
    "margin_min": 0.08,
    "overlap_min": 0.30,
    "overlap_critical": 0.20,
    "uncertainty_max": 0.45,
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)
