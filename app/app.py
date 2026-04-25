from __future__ import annotations

import os
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess

from core.classifier_fusion import adaptive_model_fusion, classify_image
from core.gradcam import get_gradcam
from core.morphology import analyze_shape, estimate_size, mass_effect
from core.overlap_metrics import overlap_metrics
from core.diagnostic_reliability import reliability_gate
from core.quality_metrics import quality_metrics
from core.risk_engine import clinical_decision, confidence_cal, rano_assessment, reliability_and_risk
from core.segmentation import run_segmentation
from reporting.llm_report_generator import groq_normal_report, groq_tumor_report
from reporting.pdf_report_generator import build_overlays, pdf_normal, pdf_tumor
from utils.config import GROQ_API_KEY, IMG_SIZE, MRI_PARAMS, REPORT_DIR, SAMPLE_DIR, SEG_SIZE, load_models
from utils.history_manager import compare_with_prior, load_history, save_history

APP_CSS = """
<style>
[data-testid="stAppViewContainer"]{background:#0a0c12;color:#c9cdd6}
[data-testid="stHeader"]{background:#0d0f18;border-bottom:1px solid #1a1f2e}
[data-testid="stSidebar"]{background:#0d0f18!important;border-right:1px solid #1a1f2e}
.block-container{padding:1.5rem 2rem 2rem}
div[data-testid="metric-container"]{background:#0d0f18;border:1px solid #1a1f2e;border-radius:12px;padding:14px 18px}
div[data-testid="metric-container"] label{color:#3d6fa5!important;font-size:11px;text-transform:uppercase;letter-spacing:.1em}
div[data-testid="stMetricValue"]{color:#5ea8ff;font-size:22px;font-weight:700}
[data-testid="stFileUploadDropzone"]{background:#0b1220!important;border:1.5px dashed #1e3050!important;border-radius:12px!important}
.stButton>button{background:linear-gradient(135deg,#1a3a70,#0f2050)!important;color:#7ab8ff!important;border:1px solid #1e4080!important;border-radius:10px!important;font-weight:600!important;width:100%}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#0f2a10,#082010)!important;color:#4dc86f!important;border:1px solid #1a4020!important;border-radius:10px!important;font-weight:600!important;width:100%}
.streamlit-expanderHeader{background:#0d0f18!important;color:#c9cdd6!important;border:1px solid #1a1f2e!important;border-radius:10px!important}
.streamlit-expanderContent{background:#0d0f18!important;border:1px solid #1a1f2e!important;border-radius:0 0 10px 10px!important}
hr{border-color:#1a1f2e!important}
[data-testid="stDataFrame"]{border:1px solid #1a1f2e;border-radius:12px}
[data-testid="stAlert"]{background:#0d0f18!important;border:1px solid #1a1f2e!important;border-radius:10px!important}
h1{color:#e2e5ed!important;font-weight:700!important}
h2,h3{color:#c8cfe0!important}
p,li{color:#8090aa}
.tag-red{background:#2a1010;color:#e87060;border:1px solid #3a1818;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.tag-amber{background:#221a08;color:#d4a843;border:1px solid #3a2c0a;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
.tag-green{background:#0f2a10;color:#4dc86f;border:1px solid #1a4020;padding:3px 10px;border-radius:5px;font-size:11px;font-weight:700}
</style>
"""

st.set_page_config(page_title="NeuroScan AI", page_icon="NS", layout="wide", initial_sidebar_state="expanded")
st.markdown(APP_CSS, unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## NeuroScan AI")
        st.caption("Brain Tumor MRI Analysis System")
        st.markdown("---")
        page = st.radio("Navigate", ["Dashboard", "Patient History", "Model Info"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### AI Pipeline")
        st.markdown(
            """<div style='font-size:12px;line-height:2.2'>
            Scan Quality Validation<br>
            Multi-Model Fusion (3 models)<br>
            Confidence Decision Logic<br>
            EfficientNet U-Net Segmentation<br>
            Grad-CAM Explainability<br>
            Tumor Size and Shape Analysis<br>
            Overlap and Reliability Scoring<br>
            Prior-Case Comparison<br>
            Groq Llama-4 Imaging Summary<br>
            PDF Export
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("For research and screening only. This is a decision-support tool, not a standalone diagnostic system.")
        return page


def choose_image():
    patient_col, id_col = st.columns(2)
    with patient_col:
        patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe", key="pname")
    with id_col:
        patient_id = st.text_input("Patient ID", placeholder="e.g. MRI-001", key="pid")

    uploaded = st.file_uploader("Drop MRI image here", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
    sample_names = sorted(p.name for p in SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
    sample_choice = st.selectbox("Or choose a sample image", ["None"] + sample_names)

    image_pil = None
    source_name = None
    tmp_path = None
    if uploaded is not None:
        image_pil = Image.open(uploaded)
        source_name = uploaded.name
        suffix = "." + uploaded.name.split(".")[-1]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded.getvalue())
        tmp_file.close()
        tmp_path = tmp_file.name
    elif sample_choice != "None" and SAMPLE_DIR.exists():
        image_pil = Image.open(SAMPLE_DIR / sample_choice)
        source_name = sample_choice
        tmp_path = str(SAMPLE_DIR / sample_choice)
    return patient_name, patient_id, image_pil, source_name, tmp_path


def save_result_history(history, source_name, patient_name, label, confidence, severity, area_cm2, risk, reliability):
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


def run_analysis(image_pil, source_name, tmp_path, patient_name, patient_id):
    progress = st.progress(0)
    status = st.empty()

    status.info("Step 0 / 6 - Validating scan quality")
    image_np = np.array(image_pil.convert("RGB"))
    quality = quality_metrics(image_np)
    if not quality["usable_for_analysis"]:
        st.warning(f"Low quality scan (score={quality['quality_score']}). Results may be unreliable.")
    progress.progress(10)

    status.info("Step 1 / 6 - Running multi-model fusion")
    classifiers, seg_model, base_model, head_layers = load_models()
    fusion, model_scores = classify_image(classifiers, image_pil)
    label = fusion["final_class"]
    confidence = fusion["fused_confidence"]
    if fusion["uncertainty_flag"]:
        st.warning(f"Uncertain prediction - models disagree. Decision: {fusion['decision_logic']}")
    progress.progress(25)

    history = load_history()
    orig = cv2.resize(cv2.imread(tmp_path), (IMG_SIZE, IMG_SIZE))

    if label == "no_tumor":
        status.info("No tumor detected - generating normal report")
        llm = groq_normal_report(tmp_path, confidence, GROQ_API_KEY, patient_name)
        pdf_path = pdf_normal(source_name, tmp_path, confidence, llm, orig, REPORT_DIR, patient_name=patient_name, patient_id=patient_id, fusion=fusion, quality=quality)
        progress.progress(100)
        status.success("Analysis complete - Normal MRI")
        st.session_state["result"] = {
            "no_tumor": True,
            "label": label,
            "confidence": confidence,
            "severity": "None",
            "filename": source_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "llm": llm,
            "pdf_path": pdf_path,
            "orig": orig,
            "fusion": fusion,
            "quality": quality,
            "model_scores": model_scores,
            "clinical": {"urgency": "LOW", "steps": ["Routine follow-up in 12 months", "No immediate action required"]},
        }
        save_result_history(history, source_name, patient_name, "No Tumor", confidence, "None", "N/A", "None", quality["quality_score"])
        return

    status.info("Step 2 / 6 - Segmenting tumor region")
    mask = run_segmentation(seg_model, tmp_path)
    progress.progress(40)

    status.info("Step 3 / 6 - Generating Grad-CAM heatmap")
    input_tensor = effv2_preprocess(np.array(image_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32))
    input_tensor = tf.convert_to_tensor(np.expand_dims(input_tensor, 0))
    heatmap = get_gradcam(input_tensor, base_model, head_layers)
    progress.progress(55)

    status.info("Step 4 / 6 - Computing metrics and overlap")
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
    progress.progress(70)

    status.info("Step 5 / 6 - Generating LLM imaging summary")
    llm = groq_tumor_report(tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, severity, rano, GROQ_API_KEY, patient_name)
    progress.progress(85)

    status.info("Step 6 / 6 - Building PDF report")
    pdf_path = pdf_tumor(source_name, tmp_path, label, confidence, size_info, shape_info, mass_info, risk_info, cal_info, severity, clinical, rano, llm, orig, overlays["mask_ov"], overlays["hmap_r"], overlays["gcam_ov"], REPORT_DIR, patient_name, patient_id, fusion, quality, overlap, comparison)
    progress.progress(100)
    status.success("Analysis complete")
    st.session_state["result"] = {
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
        "llm": llm,
        "pdf_path": pdf_path,
        "orig": orig,
        "mask_ov": overlays["mask_ov"],
        "hmap_r": overlays["hmap_r"],
        "gcam_ov": overlays["gcam_ov"],
        "fusion": adaptive_fusion,
        "quality": quality,
        "overlap": overlap,
        "gate_info": gate_info,
        "comparison": comparison,
        "model_scores": model_scores,
    }
    save_result_history(history, source_name, patient_name, label.capitalize(), confidence, severity, size_info["area_cm2"], risk_info["risk"], risk_info["reliability_score"])


def render_result_panel(result):
    st.markdown("---")
    st.subheader("Detection Result")
    severity = result["severity"]
    tag = "tag-red" if severity == "Severe" else "tag-amber" if severity == "Moderate" else "tag-green"
    st.markdown(
        f"""
        <div style='background:#0d0f18;border:1px solid #1a1f2e;border-radius:12px;padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:14px'>
          <div>
            <div style='font-size:18px;font-weight:700;color:#e2e5ed'>{result['label'].replace('_', ' ').capitalize()}</div>
            <div style='font-size:11px;color:#4a6a8a'>Multi-Model Fusion | Confidence: {result['confidence']*100:.1f}%</div>
            <div style='font-size:11px;color:#3a5a70'>{result.get('fusion', {}).get('decision_logic', '')}</div>
          </div>
          <div style='margin-left:auto'><span class='{tag}'>{severity.upper()}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result.get("quality"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Scan Quality", str(result["quality"]["quality_score"]))
        c2.metric("Blur Score", str(result["quality"]["blur_metric"]))
        c3.metric("Contrast", str(result["quality"]["contrast_metric"]))
    if result["no_tumor"]:
        st.success("No intracranial mass or tumor identified. Brain appears normal.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Tumor Area", f"{result['size_info']['area_cm2']} cm2")
        c1.metric("Risk Score", f"{result['risk_info']['score']:.3f} / 1.0")
        c2.metric("Diameter", f"{result['size_info']['diameter_cm']} cm")
        c2.metric("Volume", f"{result['size_info']['volume_cm3']} cm3")
        if result.get("comparison", {}).get("prior_available"):
            st.info(f"Prior Case Found - {result['comparison']['progression_flag']} | Area change: {result['comparison']['change_percent']:+.1f}%")
    st.markdown("---")
    st.subheader("Decision Support")
    urgency = result["clinical"]["urgency"]
    urgency_color = "#e8604a" if urgency == "CRITICAL" else "#d4a843" if urgency in ("HIGH", "MODERATE") else "#4dc86f"
    st.markdown(f"**Urgency:** <span style='color:{urgency_color};font-weight:700'>{urgency}</span>", unsafe_allow_html=True)
    for step in result["clinical"]["steps"]:
        st.markdown(f"- {step}")
    st.markdown("---")
    st.subheader("PDF Report")
    pdf_path = result.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as handle:
            st.download_button("Download Full PDF Report", data=handle, file_name=f"neuroscan_{result['filename']}_{datetime.now().strftime('%Y%m%d%H%M')}.pdf", mime="application/pdf", use_container_width=True)


def render_visual_tabs(result):
    tabs = st.tabs(["Visuals", "Report", "Model Details", "Pipeline Features"])
    with tabs[0]:
        if result["no_tumor"]:
            st.image(cv2.cvtColor(result["orig"], cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
            st.info("No tumor detected. Segmentation and Grad-CAM are skipped for normal scans.")
        else:
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(result["orig"], cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
            c1.image(result["hmap_r"], caption="Grad-CAM Heatmap", use_container_width=True, clamp=True)
            c2.image(cv2.cvtColor(result["mask_ov"], cv2.COLOR_BGR2RGB), caption=f"Segmentation | {result['size_info']['diameter_cm']} cm", use_container_width=True)
            c2.image(cv2.cvtColor(result["gcam_ov"], cv2.COLOR_BGR2RGB), caption="Grad-CAM Overlay", use_container_width=True)
    with tabs[1]:
        with st.expander("View Full Imaging Summary", expanded=True):
            for line in result["llm"].splitlines():
                if line.strip():
                    st.write(line.strip())
    with tabs[2]:
        rows = []
        for model_name, scores in result.get("model_scores", {}).items():
            row = {"Model": model_name}
            row.update(scores)
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if result.get("fusion"):
            st.write({
                "Agreement": result["fusion"]["agreement_score"],
                "Uncertainty Flag": result["fusion"]["uncertainty_flag"],
                "Decision Logic": result["fusion"]["decision_logic"],
                "Fused Class Scores": result["fusion"]["class_scores"],
            })
    with tabs[3]:
        feature_rows = [
            {"Feature": "Scan Quality Validation", "Value": result.get("quality", {}).get("quality_score", "N/A")},
            {"Feature": "Multi-Model Fusion", "Value": result.get("fusion", {}).get("decision_logic", "N/A")},
            {"Feature": "Tumor Segmentation", "Value": "Yes" if not result["no_tumor"] else "No tumor"},
            {"Feature": "Reliability Score", "Value": result.get("risk_info", {}).get("reliability_score", "N/A")},
            {"Feature": "Groq Report Generation", "Value": "Enabled" if GROQ_API_KEY else "Key missing"},
            {"Feature": "History Records", "Value": len(load_history())},
        ]
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)


def render_dashboard():
    st.title("NeuroScan AI - Screening and Decision Support Dashboard")
    st.caption("Upload a brain MRI scan to run the full AI analysis pipeline.")
    st.markdown("---")
    col_left, col_right = st.columns([1, 1.7], gap="large")
    with col_left:
        st.subheader("Scan Upload")
        patient_name, patient_id, image_pil, source_name, tmp_path = choose_image()
        if image_pil is not None:
            st.image(image_pil, caption="Input MRI", use_container_width=True)
        run_button = st.button("Run Full Analysis", disabled=image_pil is None)
        if image_pil is not None and run_button:
            run_analysis(image_pil, source_name, tmp_path, patient_name, patient_id)
        if "result" in st.session_state:
            render_result_panel(st.session_state["result"])
    with col_right:
        if "result" in st.session_state:
            render_visual_tabs(st.session_state["result"])
        else:
            st.markdown("<div style='background:#0d0f18;border:1px dashed #1a2a3a;border-radius:14px;padding:60px 40px;text-align:center;margin-top:40px'><div style='font-size:16px;color:#4a6a8a;font-weight:600'>Upload an MRI and click Run Analysis</div><div style='font-size:13px;color:#2a3a50;margin-top:6px'>Results will appear here</div></div>", unsafe_allow_html=True)


def render_history():
    st.title("Patient History")
    st.caption("All cases analyzed and persisted locally.")
    st.markdown("---")
    history = load_history()
    if not history:
        st.info("No cases analyzed yet. Go to the Dashboard to analyze an MRI scan.")
        return
    dataframe = pd.DataFrame(history)
    dataframe = dataframe.rename(columns={
        "id": "Case ID",
        "filename": "File",
        "patient": "Patient",
        "label": "Tumor Type",
        "confidence": "Confidence",
        "severity": "Severity",
        "area_cm2": "Area (cm2)",
        "risk": "Risk",
        "reliability": "Reliability",
        "date": "Date",
    })
    st.dataframe(dataframe, use_container_width=True, hide_index=True)


def render_model_info():
    st.title("Model Information")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Primary", "EfficientNetV2-S")
    c1.caption("Weight: 1.2 | Classes: 4 | Input: 384x384")
    c2.metric("Secondary", "MobileNetV3")
    c2.caption("Weight: 1.0 | Classes: 4 | Input: 384x384")
    c3.metric("Tertiary", "ConvNeXt Tiny")
    c3.caption("Weight: 1.1 | Classes: 4 | Input: 384x384")
    st.markdown("---")
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Segmentation")
        st.metric("Architecture", "EfficientNet U-Net")
        st.metric("Input", f"{SEG_SIZE}x{SEG_SIZE}")
        st.metric("Output", "Binary Mask")
    with m2:
        st.subheader("Report Generation")
        st.metric("Explainability", "Grad-CAM")
        st.metric("LLM", "Llama-4 Scout (Groq)")
        st.metric("PDF", "2-3 Pages")
    st.markdown("---")
    for key, value in MRI_PARAMS.items():
        st.write(f"**{key}:** {value}")
    st.warning("This system is for research and screening purposes only. All outputs must be reviewed by a qualified radiologist or neurosurgeon before clinical use.")


def main():
    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard()
    elif page == "Patient History":
        render_history()
    else:
        render_model_info()


if __name__ == "__main__":
    main()

