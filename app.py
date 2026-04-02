
# -*- coding: utf-8 -*-
"""
NeuroScan AI  -  Brain Tumor MRI Analysis System
app.py  -  Streamlit UI + model loading only.
All analysis logic lives in features.py.
"""

import os, warnings, json, sys, tempfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


import cv2
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_prep

from config import (
    BASE_DIR,
    CLASS_NAMES,
    GROQ_API_KEY,
    HISTORY_FILE,
    IMG_SIZE,
    MODEL_FILES,
    MODEL_WEIGHTS,
    MODELS_DIR,
    MRI_PARAMS,
    REPORT_DIR,
    SAMPLE_DIR,
    SEG_MODEL_FILE,
    SEG_SIZE,
)
from features import (
    quality_metrics,
    adaptive_model_fusion,
    run_segmentation,
    get_gradcam,
    estimate_size,
    analyze_shape,
    mass_effect,
    overlap_metrics,
    reliability_and_risk,
    reliability_gate,
    confidence_cal,
    get_severity_label,
    clinical_decision,
    rano_assessment,
    compare_with_prior,
    groq_tumor_report,
    groq_normal_report,
    build_overlays,
    pdf_tumor,
    pdf_normal,
    safe,
    clean_text,
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE CONFIG
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="ðŸ§ ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# THEME CSS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0c12;color:#c9cdd6}
[data-testid="stHeader"]{background:#0d0f18;border-bottom:1px solid #1a1f2e}
[data-testid="stSidebar"]{background:#0d0f18!important;border-right:1px solid #1a1f2e}
.block-container{padding:1.5rem 2rem 2rem}
div[data-testid="metric-container"]{background:#0d0f18;border:1px solid #1a1f2e;border-radius:12px;padding:14px 18px}
div[data-testid="metric-container"] label{color:#3d6fa5!important;font-size:11px;text-transform:uppercase;letter-spacing:.1em}
div[data-testid="stMetricValue"]{color:#5ea8ff;font-size:22px;font-weight:700}
[data-testid="stFileUploadDropzone"]{background:#0b1220!important;border:1.5px dashed #1e3050!important;border-radius:12px!important}
.stButton>button{background:linear-gradient(135deg,#1a3a70,#0f2050)!important;color:#7ab8ff!important;border:1px solid #1e4080!important;border-radius:10px!important;font-weight:600!important;width:100%;transition:all .2s}
.stButton>button:hover{background:linear-gradient(135deg,#1f4585,#142860)!important;border-color:#4a9eff!important}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#0f2a10,#082010)!important;color:#4dc86f!important;border:1px solid #1a4020!important;border-radius:10px!important;font-weight:600!important;width:100%}
.streamlit-expanderHeader{background:#0d0f18!important;color:#c9cdd6!important;border:1px solid #1a1f2e!important;border-radius:10px!important}
.streamlit-expanderContent{background:#0d0f18!important;border:1px solid #1a1f2e!important;border-radius:0 0 10px 10px!important}
hr{border-color:#1a1f2e!important}
[data-testid="stDataFrame"]{border:1px solid #1a1f2e;border-radius:12px}
[data-testid="stAlert"]{background:#0d0f18!important;border:1px solid #1a1f2e!important;border-radius:10px!important}
div[data-testid="stSuccess"]{border-left:3px solid #4dc86f!important}
div[data-testid="stError"]{border-left:3px solid #e8604a!important}
div[data-testid="stWarning"]{border-left:3px solid #d4a843!important}
div[data-testid="stInfo"]{border-left:3px solid #4a9eff!important}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] label{color:#7a8299}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#4a6fa5}
[data-testid="stImage"] img{border-radius:10px;border:1px solid #1a1f2e}
h1{color:#e2e5ed!important;font-weight:700!important}
h2,h3{color:#c8cfe0!important}
p,li{color:#8090aa}
.tag-red  {background:#2a1010;color:#e87060;border:1px solid #3a1818;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.tag-amber{background:#221a08;color:#d4a843;border:1px solid #3a2c0a;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.tag-green{background:#0f2a10;color:#4dc86f;border:1px solid #1a4020;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.tag-blue {background:#0d1a2a;color:#4a9eff;border:1px solid #1a2e4a;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.dri-high  {background:#0a1f0a;border:1px solid #1a4020;border-radius:10px;padding:10px 16px}
.dri-mod   {background:#1a1500;border:1px solid #3a2c0a;border-radius:10px;padding:10px 16px}
.dri-low   {background:#1f0a0a;border:1px solid #3a1818;border-radius:10px;padding:10px 16px}
</style>
""", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CONFIG
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "EfficientNetV2-S": {
        "path"      : MODELS_DIR / MODEL_FILES["EfficientNetV2-S"],
        "preprocess": tf.keras.applications.efficientnet_v2.preprocess_input,
        "weight"    : MODEL_WEIGHTS["EfficientNetV2-S"],
    },
    "MobileNetV3": {
        "path"      : MODELS_DIR / MODEL_FILES["MobileNetV3"],
        "preprocess": tf.keras.applications.mobilenet_v3.preprocess_input,
        "weight"    : MODEL_WEIGHTS["MobileNetV3"],
    },
    "ConvNeXt Tiny": {
        "path"      : MODELS_DIR / MODEL_FILES["ConvNeXt Tiny"],
        "preprocess": tf.keras.applications.convnext.preprocess_input,
        "weight"    : MODEL_WEIGHTS["ConvNeXt Tiny"],
    },
}
SEG_MODEL_PATH = MODELS_DIR / SEG_MODEL_FILE

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODEL LOADING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    classifiers = {}
    for name, spec in MODEL_SPECS.items():
        try:
            classifiers[name] = tf.keras.models.load_model(str(spec["path"]), compile=False)
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    seg_model = tf.keras.models.load_model(str(SEG_MODEL_PATH), compile=False)
    base_model, head_layers = None, []
    if "EfficientNetV2-S" in classifiers:
        cm = classifiers["EfficientNetV2-S"]
        try:
            base_model = cm.get_layer("efficientnetv2-s")
            hit = False
            for layer in cm.layers:
                if layer.name == "efficientnetv2-s": hit = True; continue
                if hit: head_layers.append(layer)
        except Exception:
            pass
    return classifiers, seg_model, base_model, head_layers


def preprocess_for_model(image_pil, model_name):
    arr  = np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    prep = MODEL_SPECS[model_name]["preprocess"]
    try:    arr = prep(arr)
    except: arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image(classifiers, image_pil):
    """Run all classifiers, return raw_probs dict only (no fusion yet)."""
    raw = {}
    for name, model in classifiers.items():
        x = preprocess_for_model(image_pil, name)
        raw[name] = model.predict(x, verbose=0)[0]
    return raw


def build_report_payload(label, confidence, size_info, shape_info,
                          mass_info, risk_info, comparison):
    """Build structured payload dict for PDF/report consumption."""
    return {
        "label"      : label,
        "confidence" : confidence,
        "size_info"  : size_info,
        "shape_info" : shape_info,
        "mass_info"  : mass_info,
        "risk_info"  : risk_info,
        "comparison" : comparison,
    }

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HISTORY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def load_history():
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except: return []
    return []

def save_history(record):
    h = load_history(); h.insert(0, record)
    HISTORY_FILE.write_text(json.dumps(h[:100], indent=2), encoding="utf-8")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DRI DISPLAY HELPER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def render_dri_card(reliability):
    """Render a styled Diagnostic Reliability Index card."""
    dri      = reliability.get("dri", reliability.get("evidence_score", 0.0))
    tier     = reliability.get("dri_tier", "LOW")
    label    = reliability.get("dri_label", f"DRI = {dri:.4f}")
    status   = reliability.get("acceptance_status", "Caution")
    reasons  = reliability.get("escalation_reasons", [])
    comps    = reliability.get("dri_components", {})

    tier_css = {"HIGH": "dri-high", "MODERATE": "dri-mod", "LOW": "dri-low"}
    tier_col = {"HIGH": "#4dc86f",  "MODERATE": "#d4a843",  "LOW": "#e87060"}
    status_col = ("#4dc86f" if status == "Accepted"
                  else "#d4a843" if status == "Caution"
                  else "#e87060")

    st.markdown(f"""
    <div class='{tier_css.get(tier, "dri-low")}' style='margin-bottom:12px'>
      <div style='font-size:11px;color:#5a6a80;text-transform:uppercase;letter-spacing:.1em'>
        Diagnostic Reliability Index</div>
      <div style='font-size:26px;font-weight:700;color:{tier_col.get(tier,"#e87060")}'>
        {dri:.4f} &nbsp;<span style='font-size:13px'>[{tier}]</span></div>
      <div style='font-size:11px;color:{status_col};margin-top:4px'>
        Status: <b>{status}</b></div>
    </div>
    """, unsafe_allow_html=True)

    if comps:
        with st.expander("DRI Component Breakdown", expanded=False):
            comp_labels = {
                "quality_contribution"     : "Scan Quality",
                "agreement_contribution"   : "Model Agreement",
                "confidence_contribution"  : "Fused Confidence",
                "margin_contribution"      : "Class Margin",
                "xai_contribution"         : "XAI Consistency",
                "risk_contribution"        : "Risk (inverted)",
                "lesion_trust_contribution": "Lesion Trust",
            }
            rows = [{"Component": comp_labels.get(k, k),
                     "Score Contribution": round(v, 4)}
                    for k, v in comps.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if reasons:
        with st.expander(f"Escalation Flags ({len(reasons)})", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SIDEBAR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with st.sidebar:
    st.markdown("## ðŸ§  NeuroScan AI")
    st.caption("Brain Tumor MRI Analysis System")
    st.markdown("---")
    page = st.radio("Navigate",
                    ["ðŸ¥ Dashboard", "ðŸ“‹ Patient History", "â„¹ï¸ Model Info"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### AI Pipeline")
    st.markdown("""<div style='font-size:12px;line-height:2.2'>
    <span style='color:#4dc86f'>âœ“</span> Scan Quality Validation<br>
    <span style='color:#4dc86f'>âœ“</span> Multi-Model Fusion (3 models)<br>
    <span style='color:#4dc86f'>âœ“</span> Case-Specific Dynamic Weighting<br>
    <span style='color:#4dc86f'>âœ“</span> Lesion-Aware Fusion (closed-loop)<br>
    <span style='color:#4dc86f'>âœ“</span> EfficientNet U-Net Segmentation<br>
    <span style='color:#4dc86f'>âœ“</span> Grad-CAM Explainability<br>
    <span style='color:#4dc86f'>âœ“</span> Tumor Size &amp; Shape Analysis<br>
    <span style='color:#4dc86f'>âœ“</span> XAI Overlap Validation<br>
    <span style='color:#4dc86f'>âœ“</span> Diagnostic Reliability Index (DRI)<br>
    <span style='color:#4dc86f'>âœ“</span> Auto Rejection / Escalation Gate<br>
    <span style='color:#4dc86f'>âœ“</span> Prior-Case Comparison<br>
    <span style='color:#4dc86f'>âœ“</span> Groq Llama-4 Radiology Report<br>
    <span style='color:#4dc86f'>âœ“</span> 3-Page PDF Export (Tumor)<br>
    <span style='color:#4dc86f'>âœ“</span> 2-Page PDF Export (Normal)
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("For research & screening only. Not for clinical diagnosis.")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: DASHBOARD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "ðŸ¥ Dashboard":
    st.title("ðŸ§  NeuroScan AI - Doctor Dashboard")
    st.caption("Upload a brain MRI scan to run the full AI analysis pipeline.")
    st.markdown("---")
    col_L, col_R = st.columns([1, 1.7], gap="large")

    with col_L:
        st.subheader("ðŸ“¤ Scan Upload")
        pm1, pm2 = st.columns(2)
        with pm1: patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe", key="pname")
        with pm2: patient_id   = st.text_input("Patient ID",   placeholder="e.g. MRI-001",  key="pid")

        uploaded = st.file_uploader("Drop MRI image here",
                                    type=["png", "jpg", "jpeg", "bmp"],
                                    label_visibility="collapsed")

        sample_names = []
        if SAMPLE_DIR.exists():
            sample_names = sorted([p.name for p in SAMPLE_DIR.glob("*.jpg")])
        sample_choice = st.selectbox("Or choose a sample image", ["None"] + sample_names)

        image_pil = None; source_name = None; tmp_path = None
        if uploaded is not None:
            image_pil   = Image.open(uploaded); source_name = uploaded.name
            suffix      = "." + uploaded.name.split(".")[-1]
            tf_         = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tf_.write(uploaded.getvalue()); tf_.close(); tmp_path = tf_.name
        elif sample_choice != "None" and SAMPLE_DIR.exists():
            image_pil   = Image.open(SAMPLE_DIR / sample_choice); source_name = sample_choice
            tmp_path    = str(SAMPLE_DIR / sample_choice)

        if image_pil is not None:
            st.image(image_pil, caption="Input MRI", use_container_width=True)
        run_btn = st.button("â–¶  Run Full Analysis", disabled=image_pil is None)

        if image_pil is not None and run_btn:
            prog = st.progress(0); status_msg = st.empty()

            # â”€â”€ STEP 0: Scan quality â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            status_msg.info("Step 0 / 7 - Validating scan quality...")
            image_np = np.array(image_pil.convert("RGB"))
            quality  = quality_metrics(image_np)
            if not quality["usable_for_analysis"]:
                st.warning(f"Low quality scan (score={quality['quality_score']}). Results may be unreliable.")
            prog.progress(8)

            # â”€â”€ STEP 1: Classification (raw probs only â€” no fusion yet) â”€â”€â”€â”€â”€â”€
            status_msg.info("Step 1 / 7 - Running classification models...")
            classifiers, seg_model, base_m, heads = load_models()
            raw_probs = classify_image(classifiers, image_pil)
            prog.progress(20)

            history = load_history()
            orig    = cv2.resize(cv2.imread(tmp_path), (IMG_SIZE, IMG_SIZE))

            # â”€â”€ NO TUMOR PATH (quality-only fusion since no lesion) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # First-pass fusion to detect no_tumor before spending time on seg
            fusion_pass1 = adaptive_model_fusion(raw_probs, quality["quality_score"])
            label_pass1  = fusion_pass1["final_class"]

            if label_pass1 == "no_tumor":
                status_msg.info("No tumor detected - generating normal report...")
                # For no-tumor, use the pass-1 fusion (no lesion context needed)
                fusion = fusion_pass1
                label      = fusion["final_class"]
                confidence = fusion["fused_confidence"]
                if fusion["uncertainty_flag"]:
                    st.warning(f"Uncertain prediction - {fusion['decision_logic']}")

                # Reliability gate (no lesion signals available)
                reliability = reliability_gate(
                    quality, fusion,
                    {"explainability_consistency": 1.0, "overlap_score": 1.0},
                    {"score": 0.0},
                )

                llm      = groq_normal_report(tmp_path, confidence, GROQ_API_KEY, patient_name)
                pdf_path = pdf_normal(source_name, tmp_path, confidence, llm, orig,
                                      str(REPORT_DIR), patient_name, patient_id, fusion, quality)
                prog.progress(100); status_msg.success("Analysis complete - Normal MRI!")
                st.session_state["result"] = {
                    "no_tumor": True, "label": label, "confidence": confidence,
                    "severity": "None", "filename": source_name,
                    "patient_name": patient_name, "patient_id": patient_id,
                    "llm": llm, "pdf_path": pdf_path, "orig": orig,
                    "fusion": fusion, "quality": quality,
                    "model_scores": {mn: dict(zip(CLASS_NAMES, p.tolist()))
                                     for mn, p in raw_probs.items()},
                    "reliability_gate": reliability,
                    "clinical": clinical_decision(label, "None", None),
                }
                save_history({
                    "id": f"#MRI-{len(history)+1:03d}", "filename": source_name,
                    "patient": patient_name or "-", "label": "No Tumor",
                    "confidence": f"{confidence*100:.1f}%", "severity": "None",
                    "area_cm2": "N/A", "risk": "None",
                    "reliability": quality["quality_score"],
                    "dri": reliability.get("dri", "N/A"),
                    "date": datetime.now().strftime("%b %d, %Y %H:%M"),
                })

            # â”€â”€ TUMOR PATH â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            else:
                status_msg.info("Step 2 / 7 - Segmenting tumor region...")
                mask = run_segmentation(seg_model, tmp_path); prog.progress(32)

                status_msg.info("Step 3 / 7 - Generating Grad-CAM heatmap...")
                ic = effv2_prep(np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32))
                it = tf.convert_to_tensor(np.expand_dims(ic, 0))
                heatmap = get_gradcam(it, base_m, heads); prog.progress(46)

                status_msg.info("Step 4 / 7 - Computing lesion metrics & overlap...")
                s_info   = estimate_size(mask)
                sh_info  = analyze_shape(mask)
                m_info   = mass_effect(mask)
                hmap_r   = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                overlap  = overlap_metrics(hmap_r, mask)
                prog.progress(58)

                # â”€â”€ STEP 5: Lesion-aware second-pass fusion (closed-loop) â”€â”€â”€â”€â”€â”€
                status_msg.info("Step 5 / 7 - Lesion-aware model fusion (closed-loop)...")
                lesion_ctx = {
                    "area_cm2"     : s_info["area_cm2"],
                    "diameter_cm"  : s_info["diameter_cm"],
                    "irregularity" : sh_info["irregularity"] if sh_info else 0.0,
                    "overlap_score": overlap["overlap_score"],
                }
                fusion = adaptive_model_fusion(
                    raw_probs, quality["quality_score"],
                    lesion_context=lesion_ctx,
                )
                label      = fusion["final_class"]
                confidence = fusion["fused_confidence"]
                if fusion["uncertainty_flag"]:
                    st.warning(f"Uncertain prediction - models disagree. {fusion['decision_logic']}")
                prog.progress(67)

                r_info   = reliability_and_risk(
                    label, confidence, fusion["agreement_score"],
                    quality["quality_score"], s_info, sh_info, m_info,
                    overlap["overlap_score"],
                )
                c_info   = confidence_cal(confidence)
                severity = r_info["severity"]
                clin     = clinical_decision(label, severity, m_info)
                rano     = rano_assessment(label, s_info, sh_info)
                comparison = compare_with_prior(history, source_name, s_info["area_cm2"])

                # â”€â”€ DRI â€” Diagnostic Reliability Index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                reliability = reliability_gate(quality, fusion, overlap, r_info)
                # Escalation: if gate says review, prepend to clinical steps
                if reliability["acceptance_status"] != "Accepted":
                    clin = {
                        "steps": ["[DRI GATE] Specialist review required before relying on automated result"]
                                  + clin["steps"],
                        "urgency": "HIGH" if clin["urgency"] == "MODERATE" else clin["urgency"],
                    }

                overlays = build_overlays(orig, mask, hmap_r, s_info)
                prog.progress(75)

                status_msg.info("Step 6 / 7 - Generating LLM radiology report...")
                llm = groq_tumor_report(tmp_path, label, confidence, s_info, sh_info,
                                        m_info, r_info, severity, rano, GROQ_API_KEY, patient_name)
                prog.progress(88)

                status_msg.info("Step 7 / 7 - Building PDF report...")
                pdf_path = pdf_tumor(
                    source_name, tmp_path, label, confidence,
                    s_info, sh_info, m_info, r_info, c_info,
                    severity, clin, rano, llm,
                    orig, overlays["mask_ov"], overlays["hmap_r"], overlays["gcam_ov"],
                    str(REPORT_DIR), patient_name, patient_id,
                    fusion, quality, overlap, comparison,
                )
                prog.progress(100); status_msg.success("Analysis complete!")

                st.session_state["result"] = {
                    "no_tumor": False, "label": label, "confidence": confidence,
                    "severity": severity, "filename": source_name,
                    "patient_name": patient_name, "patient_id": patient_id,
                    "size_info": s_info, "shape_info": sh_info, "mass_info": m_info,
                    "risk_info": r_info, "cal_info": c_info, "clinical": clin, "rano": rano,
                    "llm": llm, "pdf_path": pdf_path,
                    "orig": orig, "mask_ov": overlays["mask_ov"],
                    "hmap_r": overlays["hmap_r"], "gcam_ov": overlays["gcam_ov"],
                    "fusion": fusion, "quality": quality, "overlap": overlap,
                    "comparison": comparison,
                    "model_scores": {mn: dict(zip(CLASS_NAMES, p.tolist()))
                                     for mn, p in raw_probs.items()},
                    "reliability_gate": reliability,
                    "lesion_ctx": lesion_ctx,
                }
                save_history({
                    "id": f"#MRI-{len(history)+1:03d}", "filename": source_name,
                    "patient": patient_name or "-", "label": label.capitalize(),
                    "confidence": f"{confidence*100:.1f}%", "severity": severity,
                    "area_cm2": s_info["area_cm2"], "risk": r_info["risk"],
                    "reliability": r_info["reliability_score"],
                    "dri": reliability.get("dri", "N/A"),
                    "dri_tier": reliability.get("dri_tier", ""),
                    "lesion_trust": fusion.get("lesion_trust_multiplier", "N/A"),
                    "date": datetime.now().strftime("%b %d, %Y %H:%M"),
                })

        # â”€â”€ RESULT PANEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if "result" in st.session_state:
            res = st.session_state["result"]
            st.markdown("---"); st.subheader("ðŸ”¬ Detection Result")
            sev  = res["severity"]
            tag  = "tag-red" if sev == "Severe" else ("tag-amber" if sev == "Moderate" else "tag-green")
            icon = "ðŸ”´" if sev == "Severe" else ("ðŸŸ¡" if sev == "Moderate" else "ðŸŸ¢")
            fusion_r     = res.get("fusion", {})
            decision_txt = fusion_r.get("decision_logic", "")
            pt_disp      = (f"<div style='font-size:11px;color:#5a6a80;margin-top:2px'>"
                            f"Patient: {res.get('patient_name','-')} | ID: {res.get('patient_id','-')}</div>"
                            if res.get("patient_name") or res.get("patient_id") else "")
            st.markdown(f"""
            <div style='background:#0d0f18;border:1px solid #1a1f2e;border-radius:12px;
                        padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:14px'>
              <div style='font-size:30px'>{icon}</div>
              <div>
                <div style='font-size:18px;font-weight:700;color:#e2e5ed'>
                  {res['label'].replace('_',' ').capitalize()}</div>
                <div style='font-size:11px;color:#4a6a8a'>Lesion-Aware Fusion | Confidence: {res['confidence']*100:.1f}%</div>
                <div style='font-size:11px;color:#3a5a70'>{decision_txt}</div>
                {pt_disp}
              </div>
              <div style='margin-left:auto'><span class='{tag}'>{sev.upper()}</span></div>
            </div>""", unsafe_allow_html=True)

            # DRI card (shown for all cases)
            if res.get("reliability_gate"):
                render_dri_card(res["reliability_gate"])

            if res.get("quality"):
                q = res["quality"]
                qc = st.columns(3)
                qc[0].metric("Scan Quality", str(q["quality_score"]))
                qc[1].metric("Blur Score",   str(q["blur_metric"]))
                qc[2].metric("Contrast",     str(q["contrast_metric"]))

            if res["no_tumor"]:
                st.success("No intracranial mass or tumor identified. Brain appears normal.")
                c1, c2 = st.columns(2)
                with c1: st.metric("Confidence", f"{res['confidence']*100:.1f}%")
                with c2: st.metric("Severity", "None")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Confidence",  f"{res['confidence']*100:.1f}%")
                    st.metric("Tumor Area",  f"{res['size_info']['area_cm2']} cmÂ²")
                    st.metric("Risk Score",  f"{res['risk_info']['score']:.3f} / 1.0")
                with c2:
                    st.metric("Diameter",    f"{res['size_info']['diameter_cm']} cm")
                    st.metric("Volume",      f"{res['size_info']['volume_cm3']} cmÂ³")
                    lt = res.get("fusion", {}).get("lesion_trust_multiplier", "N/A")
                    st.metric("Lesion Trust", str(lt))

                if res.get("fusion"):
                    st.markdown("**Model Votes (Lesion-Aware):**")
                    fv = res["fusion"]["model_votes"]
                    vote_cols = st.columns(len(fv))
                    for i, (mn, mv) in enumerate(fv.items()):
                        vote_cols[i].metric(mn, mv.replace("_", " ").title())

                comp = res.get("comparison", {})
                if comp.get("prior_available"):
                    flag = comp["progression_flag"]; pct = comp["change_percent"]
                    col  = "ðŸ”´" if flag == "Progression" else ("ðŸŸ¢" if flag == "Regression" else "ðŸŸ¡")
                    st.info(f"{col} **Prior Case Found** - {flag} | Area change: {pct:+.1f}%")

            st.markdown("---"); st.subheader("ðŸ“‹ Clinical Decision")
            urg   = res["clinical"]["urgency"]
            steps = res["clinical"]["steps"]
            urg_col = ("#e8604a" if urg == "CRITICAL"
                       else "#d4a843" if urg in ("HIGH", "MODERATE") else "#4dc86f")
            st.markdown(f"**Urgency:** <span style='color:{urg_col};font-weight:700'>{urg}</span>",
                        unsafe_allow_html=True)
            for step in steps: st.markdown(f"-> {step}")

            st.markdown("---"); st.subheader("ðŸ“„ PDF Report")
            pdf_p = res.get("pdf_path", "")
            if pdf_p and os.path.exists(pdf_p):
                with open(pdf_p, "rb") as f:
                    pt_t = (f"_{res.get('patient_name','').replace(' ','_')}"
                            if res.get("patient_name") else "")
                    st.download_button(
                        "â¬‡  Download Full PDF Report", data=f,
                        file_name=f"neuroscan{pt_t}_{res['filename']}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )

    # â”€â”€ RIGHT PANEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with col_R:
        if "result" in st.session_state:
            res  = st.session_state["result"]
            tabs = st.tabs(["ðŸ“Š Visuals", "ðŸ“ Report", "ðŸ”¬ Model Details",
                             "ðŸ” DRI Analysis", "âš™ï¸ Features"])

            with tabs[0]:
                if res["no_tumor"]:
                    st.subheader("Uploaded Scan")
                    st.image(cv2.cvtColor(res["orig"], cv2.COLOR_BGR2RGB),
                             caption="Original MRI - No abnormality detected",
                             use_container_width=True)
                    st.info("No tumor detected - segmentation and GradCAM not generated for normal scans.")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.cvtColor(res["orig"], cv2.COLOR_BGR2RGB),
                                 caption="Original MRI", use_container_width=True)
                        st.image(res["hmap_r"], caption="Grad-CAM Heatmap",
                                 use_container_width=True, clamp=True)
                    with c2:
                        st.image(cv2.cvtColor(res["mask_ov"], cv2.COLOR_BGR2RGB),
                                 caption=f"Segmentation | Ã˜ {res['size_info']['diameter_cm']} cm",
                                 use_container_width=True)
                        st.image(cv2.cvtColor(res["gcam_ov"], cv2.COLOR_BGR2RGB),
                                 caption="Grad-CAM Overlay", use_container_width=True)
                    if res.get("overlap"):
                        ov = res["overlap"]
                        oc1, oc2, oc3 = st.columns(3)
                        oc1.metric("Overlap (IoU)",     str(ov["overlap_score"]))
                        oc2.metric("Attn. in Lesion",   f"{ov['attention_inside_lesion_percent']}%")
                        oc3.metric("Exp. Consistency",  str(ov["explainability_consistency"]))

            with tabs[1]:
                with st.expander("View Full Radiology Report", expanded=True):
                    for line in res["llm"].split("\n"):
                        line = line.strip()
                        if not line: continue
                        if any(line.startswith(h) for h in ["1.", "2.", "3.", "4.", "5.", "6."]):
                            st.markdown(f"**{line}**")
                        else:
                            st.write(line)
                if not res["no_tumor"] and res.get("rano"):
                    st.markdown("---"); st.subheader("ðŸ¥ RANO Assessment")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Size Category", res["rano"]["size_cat"])
                        st.metric("Enhancement",   res["rano"]["enhancement"])
                    with c2:
                        st.metric("Necrosis", res["rano"]["necrosis"])
                    st.caption(res["rano"]["grade"])

            with tabs[2]:
                if res.get("model_scores"):
                    st.subheader("Per-Model Class Scores")
                    rows = []
                    for mn, scores in res["model_scores"].items():
                        row = {"Model": mn}; row.update(scores); rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if res.get("fusion"):
                    f = res["fusion"]
                    st.subheader("Lesion-Aware Fusion Summary")
                    st.write({
                        "Agreement"             : f["agreement_score"],
                        "Uncertainty Flag"      : f["uncertainty_flag"],
                        "Decision Logic"        : f["decision_logic"],
                        "Lesion Trust Mult."    : f.get("lesion_trust_multiplier", "N/A"),
                        "Lesion Context Used"   : f.get("lesion_context_used", False),
                        "Fused Class Scores"    : f["class_scores"],
                    })
                    if f.get("adaptive_weights"):
                        st.subheader("Per-Model Adaptive Weights")
                        aw = f["adaptive_weights"]
                        wc = st.columns(len(aw))
                        for i, (mn, wv) in enumerate(aw.items()):
                            wc[i].metric(mn, str(wv))
                if res.get("lesion_ctx"):
                    st.subheader("Lesion Context Passed to Fusion")
                    st.json(res["lesion_ctx"])

            with tabs[3]:
                st.subheader("ðŸ” Diagnostic Reliability Index (DRI)")
                st.caption("The DRI is the single composite trustworthiness score for the full pipeline output.")
                if res.get("reliability_gate"):
                    rel = res["reliability_gate"]
                    render_dri_card(rel)
                    st.markdown("---")
                    st.markdown("**DRI Formula:**")
                    st.code(
                        "DRI = 0.18Â·Quality + 0.20Â·Agreement + 0.20Â·Confidence\n"
                        "    + 0.12Â·Margin   + 0.18Â·XAI_Consistency\n"
                        "    + 0.05Â·(1âˆ’Risk) + 0.07Â·LesionTrust\n\n"
                        "Tiers:  HIGH â‰¥ 0.72  |  MODERATE 0.50â€“0.71  |  LOW < 0.50\n"
                        "Gate:   Accepted | Caution | Specialist Review Required",
                        language="text"
                    )
                    st.markdown("---")
                    st.subheader("Acceptance Gate")
                    status    = rel.get("acceptance_status", "")
                    sc        = ("#4dc86f" if status == "Accepted"
                                 else "#d4a843" if status == "Caution"
                                 else "#e87060")
                    st.markdown(f"<div style='font-size:18px;color:{sc};font-weight:700'>{status}</div>",
                                unsafe_allow_html=True)
                    reasons = rel.get("escalation_reasons", [])
                    if reasons:
                        st.markdown("**Flags raised:**")
                        for r in reasons: st.markdown(f"- {r}")
                    else:
                        st.success("No flags raised â€” all signals within acceptable bounds.")

            with tabs[4]:
                feature_rows = [
                    {"#": "1",  "Feature": "Scan Quality Validation",
                     "Value": res.get("quality", {}).get("quality_score", "N/A"),
                     "Status": "âœ… Active"},
                    {"#": "2",  "Feature": "Multi-Model Fusion (3 models)",
                     "Value": res.get("fusion", {}).get("decision_logic", "N/A"),
                     "Status": "âœ… Active"},
                    {"#": "3",  "Feature": "Case-Specific Dynamic Weighting",
                     "Value": str(res.get("fusion", {}).get("adaptive_weights", "N/A")),
                     "Status": "âœ… Active"},
                    {"#": "4",  "Feature": "Lesion-Aware Fusion (closed-loop)",
                     "Value": str(res.get("fusion", {}).get("lesion_trust_multiplier", "N/A (no tumor)")),
                     "Status": "âœ… Active" if not res["no_tumor"] else "âž– N/A"},
                    {"#": "5",  "Feature": "Tumor Segmentation (U-Net)",
                     "Value": "Yes" if not res["no_tumor"] else "N/A (no tumor)",
                     "Status": "âœ… Active" if not res["no_tumor"] else "âž– N/A"},
                    {"#": "6",  "Feature": "Grad-CAM Explainability",
                     "Value": "Generated" if not res["no_tumor"] else "N/A",
                     "Status": "âœ… Active" if not res["no_tumor"] else "âž– N/A"},
                    {"#": "7",  "Feature": "Tumor Size Measurement",
                     "Value": (f"{res['size_info']['area_cm2']} cm2"
                               if not res["no_tumor"] else "N/A"),
                     "Status": "âœ… Active" if not res["no_tumor"] else "âž– N/A"},
                    {"#": "8",  "Feature": "Shape & Irregularity Analysis",
                     "Value": (str(res["shape_info"]["irregularity"])
                               if (not res["no_tumor"] and res.get("shape_info")) else "N/A"),
                     "Status": "âœ… Active" if (not res["no_tumor"] and res.get("shape_info")) else "âž– N/A"},
                    {"#": "9",  "Feature": "XAI Overlap Validation (IoU)",
                     "Value": str(res.get("overlap", {}).get("explainability_consistency", "N/A")),
                     "Status": "âœ… Active" if not res["no_tumor"] else "âž– N/A"},
                    {"#": "10", "Feature": "Diagnostic Reliability Index (DRI)",
                     "Value": res.get("reliability_gate", {}).get("dri_label", "N/A"),
                     "Status": "âœ… Active"},
                    {"#": "11", "Feature": "Auto Rejection / Escalation Gate",
                     "Value": res.get("reliability_gate", {}).get("acceptance_status", "N/A"),
                     "Status": "âœ… Active"},
                    {"#": "12", "Feature": "Severity & Risk Scoring",
                     "Value": f"{res['severity']} / {res.get('risk_info',{}).get('risk','None')}",
                     "Status": "âœ… Active"},
                    {"#": "13", "Feature": "Prior-Case Comparison",
                     "Value": res.get("comparison", {}).get("progression_flag", "N/A"),
                     "Status": "âœ… Active" if res.get("comparison", {}).get("prior_available") else "âž– No prior"},
                    {"#": "14", "Feature": "Groq Llama-4 Report Generation",
                     "Value": "Enabled" if GROQ_API_KEY else "Key missing",
                     "Status": "âœ… Active" if GROQ_API_KEY else "âŒ Missing key"},
                    {"#": "15", "Feature": "Doctor Dashboard History",
                     "Value": len(load_history()),
                     "Status": "âœ… Active"},
                ]
                st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style='background:#0d0f18;border:1px dashed #1a2a3a;border-radius:14px;
                        padding:60px 40px;text-align:center;margin-top:40px'>
              <div style='font-size:48px;margin-bottom:12px'>ðŸ§ </div>
              <div style='font-size:16px;color:#4a6a8a;font-weight:600'>Upload an MRI and click Run Analysis</div>
              <div style='font-size:13px;color:#2a3a50;margin-top:6px'>Results will appear here</div>
            </div>""", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: PATIENT HISTORY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“‹ Patient History":
    st.title("ðŸ“‹ Patient History")
    st.caption("All cases analyzed - persisted on Google Drive.")
    st.markdown("---")
    history = load_history()
    if not history:
        st.info("No cases analyzed yet. Go to the Dashboard to analyze an MRI scan.")
    else:
        df = pd.DataFrame(history)
        rename = {"id": "Case ID", "filename": "File", "patient": "Patient",
                  "label": "Tumor Type", "confidence": "Confidence", "severity": "Severity",
                  "area_cm2": "Area (cm2)", "risk": "Risk", "reliability": "Reliability",
                  "dri": "DRI", "dri_tier": "DRI Tier",
                  "lesion_trust": "Lesion Trust", "date": "Date"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Cases", len(history))
        with c2: st.metric("Severe", sum(1 for h in history if h.get("severity") == "Severe"))
        with c3: st.metric("Tumor Detected",
                            sum(1 for h in history
                                if h.get("label", "").lower() not in ["no_tumor", "no tumor", "-"]))
        with c4: st.metric("Normal Scans",
                            sum(1 for h in history
                                if h.get("label", "").lower() in ["no_tumor", "no tumor"]))

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE: MODEL INFO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "â„¹ï¸ Model Info":
    st.title("â„¹ï¸ Model Information"); st.markdown("---")
    st.subheader("Classification Models (Lesion-Aware Fusion)")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Primary", "EfficientNetV2-S")
        st.caption("Weight: 1.2 | Classes: 4 | Input: 384x384")
    with mc2:
        st.metric("Secondary", "MobileNetV3")
        st.caption("Weight: 1.0 | Classes: 4 | Input: 384x384")
    with mc3:
        st.metric("Tertiary", "ConvNeXt Tiny")
        st.caption("Weight: 1.1 | Classes: 4 | Input: 384x384")
    st.markdown("---")
    st.subheader("Lesion-Aware Fusion Pipeline")
    st.markdown("""
    **Two-pass closed-loop design:**
    1. **Pass 1** â€” Quality-only fusion for initial label detection
    2. **Segmentation + GradCAM** run on the initial label
    3. **Pass 2** â€” Full lesion-aware re-fusion using size, shape irregularity, and XAI overlap
       as a *Lesion Trust Multiplier* that modulates each model's adaptive weight

    The Lesion Trust Multiplier combines:
    - **Size factor** (0.85â€“1.05): larger lesions increase trust
    - **Shape factor** (0.90â€“1.05): higher irregularity rewards confident models
    - **XAI factor** (0.85â€“1.05): strong GradCAM-segmentation IoU = spatially grounded prediction

    Final multiplier range: **[0.75 â€“ 1.30]**
    """)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Segmentation")
        st.metric("Architecture", "EfficientNet U-Net")
        st.metric("Input", f"{SEG_SIZE}x{SEG_SIZE}"); st.metric("Output", "Binary Mask")
        st.caption("Pixel-level tumor region segmentation")
    with c2:
        st.subheader("Report Generation")
        st.metric("Explainability", "Grad-CAM")
        st.metric("LLM", "Llama-4 Scout (Groq)"); st.metric("PDF", "2-3 Pages")
        st.caption("Full radiology report with risk analytics")
    st.markdown("---")
    st.subheader("Diagnostic Reliability Index (DRI)")
    st.code(
        "DRI = 0.18Â·Q  + 0.20Â·A  + 0.20Â·C  + 0.12Â·M  + 0.18Â·X  + 0.05Â·(1-R)  + 0.07Â·L\n\n"
        "Q = Scan quality          A = Model agreement     C = Fused confidence\n"
        "M = Class margin (norm)   X = XAI consistency     R = Risk score\n"
        "L = Lesion trust (norm)\n\n"
        "Tiers:  HIGH >= 0.72  |  MODERATE 0.50-0.71  |  LOW < 0.50\n"
        "Gate:   Accepted | Caution | Specialist Review Required",
        language="text"
    )
    st.markdown("---"); st.subheader("MRI Acquisition Parameters")
    for k, v in MRI_PARAMS.items():
        st.write(f"**{k}:** {v}")
    st.markdown("---")
    st.warning("This system is for research and screening purposes only. "
               "All AI outputs must be reviewed by a qualified radiologist or neurosurgeon "
               "before any clinical decisions are made.")

