from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT
MODELS_DIR = Path(os.getenv("NEUROSCAN_MODELS_DIR", PROJECT_ROOT / "MODEL"))
if not MODELS_DIR.exists():
    legacy_models_dir = PROJECT_ROOT / "Models"
    if legacy_models_dir.exists():
        MODELS_DIR = legacy_models_dir
REPORT_DIR = PROJECT_ROOT / "Reports"
HISTORY_DIR = PROJECT_ROOT / "History"
HISTORY_FILE = HISTORY_DIR / "case_history.json"
SAMPLE_DIR = PROJECT_ROOT / "Test Data"
ENV_FILE = REPO_ROOT / ".env"
LEGACY_ENV_FILE = REPO_ROOT / "env.txt"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 384
SEG_SIZE = 256
PIX_MM = 0.5
MM2_CM2 = 0.01
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
MRI_PARAMS = {
    "Sequence": "T1-Weighted Contrast Enhanced",
    "Resolution": "384 x 384 px",
    "Slice Thickness": "5.0 mm",
    "Field Strength": "1.5 T",
    "Voxel Size": "0.5 x 0.5 x 5.0 mm",
}
MODEL_WEIGHTS = {
    "EfficientNetV2-S": 1.2,
    "MobileNetV3": 1.0,
    "ConvNeXt Tiny": 1.1,
}
SEG_MODEL_PATH = MODELS_DIR / "Segmentation_brisc_effunet.keras"
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


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    data: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")
    return data


ENV = load_env()
if not ENV and LEGACY_ENV_FILE.exists():
    ENV = load_env(LEGACY_ENV_FILE)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", ENV.get("GROQ_API_KEY", ""))


def get_tensorflow():
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(0)
    return tf


def get_model_specs():
    tf = get_tensorflow()
    return {
        "EfficientNetV2-S": {
            "path": MODELS_DIR / "class_Tumor_v2s_clean.keras",
            "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
            "weight": MODEL_WEIGHTS["EfficientNetV2-S"],
        },
        "MobileNetV3": {
            "path": MODELS_DIR / "class_Tumor_mobilenet_v3.keras",
            "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
            "weight": MODEL_WEIGHTS["MobileNetV3"],
        },
        "ConvNeXt Tiny": {
            "path": MODELS_DIR / "class_Tumor_convnext_tiny_tumor.keras",
            "preprocess": tf.keras.applications.convnext.preprocess_input,
            "weight": MODEL_WEIGHTS["ConvNeXt Tiny"],
        },
    }


def load_models():
    tf = get_tensorflow()
    model_specs = get_model_specs()
    classifiers = {}
    for name, spec in model_specs.items():
        classifiers[name] = tf.keras.models.load_model(str(spec["path"]), compile=False)
    seg_model = tf.keras.models.load_model(str(SEG_MODEL_PATH), compile=False)
    base_model = None
    head_layers = []
    if "EfficientNetV2-S" in classifiers:
        classifier_model = classifiers["EfficientNetV2-S"]
        try:
            base_model = classifier_model.get_layer("efficientnetv2-s")
            hit = False
            for layer in classifier_model.layers:
                if layer.name == "efficientnetv2-s":
                    hit = True
                    continue
                if hit:
                    head_layers.append(layer)
        except Exception:
            base_model = None
            head_layers = []
    return classifiers, seg_model, base_model, head_layers
