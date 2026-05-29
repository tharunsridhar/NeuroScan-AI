from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.pipeline import analyze_mri, save_upload_to_temp
from app.schemas import (
    AnalyzeResponse,
    ApiError,
    HealthResponse,
    HistoryResponse,
    ModelInfoResponse,
    ReadinessResponse,
    ReportsResponse,
)
from utils.config import CLASS_NAMES, IMG_SIZE, MRI_PARAMS, MODELS_DIR, REPORT_DIR, SEG_MODEL_PATH, SEG_SIZE
from utils.history_manager import load_history

APP_NAME = "NeuroScan AI API"
APP_VERSION = "1.0.0"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_MB = int(os.getenv("NEUROSCAN_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


app = FastAPI(
    title=APP_NAME,
    description=(
        "Production FastAPI service for brain MRI screening, tumor classification, "
        "segmentation, explainability, reliability scoring, report generation, and PDF export."
    ),
    version=APP_VERSION,
    contact={"name": "NeuroScan AI"},
    license_info={"name": "Research and screening use only"},
    responses={400: {"model": ApiError}, 413: {"model": ApiError}, 500: {"model": ApiError}},
)
app.state.started_at = datetime.now(timezone.utc)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env("NEUROSCAN_CORS_ORIGINS", "*"),
    allow_credentials=os.getenv("NEUROSCAN_CORS_ORIGINS", "*") != "*",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - started:.4f}"
    response.headers["X-Service"] = APP_NAME
    return response


@app.exception_handler(UnidentifiedImageError)
async def image_error_handler(_: Request, exc: UnidentifiedImageError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": f"Invalid image file: {exc}"})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


def _model_files() -> dict[str, Path]:
    return {
        "EfficientNetV2-S": MODELS_DIR / "class_Tumor_v2s_clean.keras",
        "MobileNetV3": MODELS_DIR / "class_Tumor_mobilenet_v3.keras",
        "ConvNeXt Tiny": MODELS_DIR / "class_Tumor_convnext_tiny_tumor.keras",
        "Segmentation": SEG_MODEL_PATH,
    }


def _missing_model_files() -> list[str]:
    return [str(path) for path in _model_files().values() if not Path(path).exists()]


def _report_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "url": f"/api/reports/{path.name}",
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _resolve_report(filename: str) -> Path:
    clean_name = Path(filename).name
    path = (REPORT_DIR / clean_name).resolve()
    report_root = REPORT_DIR.resolve()
    if report_root not in path.parents or not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return path


def _index_html() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>NeuroScan AI API</title>
        <style>
          :root {
            color-scheme: dark;
            --bg: #080b10;
            --panel: #10151d;
            --panel-2: #151c26;
            --line: #273241;
            --line-soft: #1d2632;
            --text: #edf2f7;
            --muted: #9aa8b8;
            --soft: #6f7f91;
            --blue: #58a6ff;
            --green: #51d88a;
            --amber: #f1b75f;
            --red: #ff6b6b;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
          }
          body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
              linear-gradient(180deg, rgba(24, 34, 48, .94), rgba(8, 11, 16, 1) 46%),
              repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 96px);
          }
          main { position: relative; max-width: 1220px; margin: 0 auto; padding: 28px 18px 42px; }
          header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 22px;
          }
          h1 { margin: 0; font-size: clamp(28px, 4vw, 46px); letter-spacing: 0; line-height: 1.05; }
          h2 { margin: 0 0 14px; font-size: 17px; letter-spacing: 0; }
          p { margin: 8px 0 0; color: var(--muted); line-height: 1.55; max-width: 720px; }
          a { color: var(--blue); text-decoration: none; }
          a:hover { text-decoration: underline; }
          .nav { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
          .nav a, .ghost {
            display: inline-flex;
            align-items: center;
            min-height: 36px;
            padding: 8px 12px;
            border: 1px solid var(--line);
            border-radius: 6px;
            background: rgba(16, 21, 29, .78);
            color: var(--text);
            font-size: 13px;
            font-weight: 650;
          }
          .shell {
            display: grid;
            grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
            gap: 18px;
            align-items: start;
          }
          .panel {
            background: rgba(16, 21, 29, .92);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 48px rgba(0,0,0,.28);
          }
          .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 16px 18px;
            border-bottom: 1px solid var(--line-soft);
          }
          .panel-body { padding: 18px; }
          label { display: block; margin: 16px 0 7px; color: var(--muted); font-size: 13px; font-weight: 650; }
          input {
            width: 100%;
            min-height: 42px;
            padding: 10px 12px;
            border-radius: 6px;
            border: 1px solid #313d4d;
            background: #0b1017;
            color: var(--text);
            outline: none;
          }
          input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(88,166,255,.13); }
          input[type="file"] { padding: 9px; }
          button {
            width: 100%;
            min-height: 44px;
            margin-top: 18px;
            border: 1px solid #2d6caa;
            border-radius: 6px;
            background: #14508a;
            color: #f4f9ff;
            font-weight: 760;
            cursor: pointer;
          }
          button:disabled { opacity: .62; cursor: wait; }
          .status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 12px 14px;
            border-radius: 6px;
            border: 1px solid var(--line-soft);
            background: #0b1017;
            color: var(--muted);
            font-size: 13px;
          }
          .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--soft); display: inline-block; margin-right: 8px; }
          .dot.ok { background: var(--green); }
          .dot.busy { background: var(--amber); }
          .dot.err { background: var(--red); }
          .metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
          .metric {
            min-height: 82px;
            padding: 13px;
            background: var(--panel-2);
            border: 1px solid var(--line-soft);
            border-radius: 8px;
          }
          .metric span { display: block; color: var(--muted); font-size: 12px; font-weight: 700; }
          .metric strong { display: block; margin-top: 8px; font-size: clamp(18px, 3vw, 24px); overflow-wrap: anywhere; }
          .metric.ok strong { color: var(--green); }
          .metric.warn strong { color: var(--amber); }
          .metric.bad strong { color: var(--red); }
          .summary {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 10px;
          }
          .kv {
            padding: 11px 12px;
            border: 1px solid var(--line-soft);
            border-radius: 6px;
            background: #0c121a;
          }
          .kv span { display: block; color: var(--soft); font-size: 12px; margin-bottom: 4px; }
          .kv strong { display: block; overflow-wrap: anywhere; }
          .report {
            margin-top: 14px;
            padding: 14px;
            border-radius: 8px;
            border: 1px solid var(--line-soft);
            background: #0c121a;
            white-space: pre-wrap;
            color: #dbe6f3;
            line-height: 1.48;
            max-height: 300px;
            overflow: auto;
          }
          .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
          .actions a { min-height: 36px; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); font-weight: 700; font-size: 13px; }
          details { margin-top: 14px; border: 1px solid var(--line-soft); border-radius: 8px; background: #0c121a; }
          summary { cursor: pointer; padding: 12px 14px; color: var(--muted); font-weight: 700; }
          pre { margin: 0; padding: 0 14px 14px; overflow: auto; white-space: pre-wrap; color: #c9d6e6; font-size: 12px; line-height: 1.45; }
          .empty { color: var(--muted); padding: 34px 18px; text-align: center; border: 1px dashed #334052; border-radius: 8px; background: #0b1017; }
          @media (max-width: 900px) {
            header { align-items: flex-start; flex-direction: column; }
            .nav { justify-content: flex-start; }
            .shell { grid-template-columns: 1fr; }
            .metrics, .summary { grid-template-columns: 1fr; }
          }
        </style>
      </head>
      <body>
        <main>
          <header>
            <div>
              <h1>NeuroScan AI</h1>
              <p>Brain MRI screening with classification, segmentation, reliability scoring, reporting, and PDF export.</p>
            </div>
            <nav class="nav">
              <a href="/docs">API Docs</a>
              <a href="/ready">Readiness</a>
              <a href="/api/reports">Reports</a>
              <a href="/api/history">History</a>
            </nav>
          </header>

          <section class="shell">
            <form id="scan-form" class="panel">
              <div class="panel-head">
                <h2>New Analysis</h2>
                <a class="ghost" href="/api/model-info">Model Info</a>
              </div>
              <div class="panel-body">
                <div class="status" id="status"><span><i class="dot"></i>Waiting for MRI upload</span><span>Max 25 MB</span></div>
                <label for="patient_name">Patient Name</label>
                <input id="patient_name" name="patient_name" placeholder="Optional">
                <label for="patient_id">Patient ID</label>
                <input id="patient_id" name="patient_id" placeholder="Optional">
                <label for="file">MRI Image</label>
                <input id="file" name="file" type="file" accept=".png,.jpg,.jpeg,.bmp" required>
                <button id="submit" type="submit">Run Analysis</button>
              </div>
            </form>

            <section class="panel">
              <div class="panel-head">
                <h2>Result</h2>
                <span class="ghost" id="result-state">No case loaded</span>
              </div>
              <div class="panel-body" id="result">
                <div class="empty">Upload an MRI image and run analysis. The first request loads the model files and can take longer.</div>
              </div>
            </section>
          </section>
        </main>
        <script>
          const form = document.getElementById("scan-form");
          const result = document.getElementById("result");
          const statusBox = document.getElementById("status");
          const state = document.getElementById("result-state");
          const submit = document.getElementById("submit");

          function escapeHtml(value) {
            return String(value ?? "").replace(/[&<>"']/g, (char) => ({
              "&": "&amp;",
              "<": "&lt;",
              ">": "&gt;",
              "\"": "&quot;",
              "'": "&#039;"
            })[char]);
          }

          function setStatus(kind, text, detail = "") {
            const cls = kind === "ok" ? "ok" : kind === "busy" ? "busy" : kind === "err" ? "err" : "";
            statusBox.innerHTML = `<span><i class="dot ${cls}"></i>${escapeHtml(text)}</span><span>${escapeHtml(detail)}</span>`;
          }

          function metric(label, value, tone = "") {
            return `<div class="metric ${tone}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
          }

          function kv(label, value) {
            return `<div class="kv"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value || "N/A")}</strong></div>`;
          }

          function renderResult(data) {
            const confidence = typeof data.confidence === "number" ? `${(data.confidence * 100).toFixed(1)}%` : "N/A";
            const severityTone = data.severity === "Severe" ? "bad" : data.severity === "Moderate" ? "warn" : "ok";
            const pdf = data.pdf_url ? `<a href="${escapeHtml(data.pdf_url)}" target="_blank" rel="noreferrer">Open PDF Report</a>` : "";
            const urgency = data.clinical && data.clinical.urgency ? data.clinical.urgency : "N/A";
            const area = data.size_info && data.size_info.area_cm2 ? `${data.size_info.area_cm2} cm2` : data.no_tumor ? "N/A" : "Pending";
            const diameter = data.size_info && data.size_info.diameter_cm ? `${data.size_info.diameter_cm} cm` : data.no_tumor ? "N/A" : "Pending";
            state.textContent = "Analysis complete";
            result.innerHTML = `
              <div class="metrics">
                ${metric("Prediction", String(data.label || "").replaceAll("_", " "), data.no_tumor ? "ok" : severityTone)}
                ${metric("Confidence", confidence)}
                ${metric("Severity", data.severity || "N/A", severityTone)}
              </div>
              <div class="summary">
                ${kv("Patient", data.patient_name || "-")}
                ${kv("Patient ID", data.patient_id || "-")}
                ${kv("Urgency", urgency)}
                ${kv("Scan Quality", data.quality ? data.quality.quality_score : "N/A")}
                ${kv("Tumor Area", area)}
                ${kv("Diameter", diameter)}
              </div>
              <div class="actions">
                ${pdf}
                <a href="/api/reports" target="_blank" rel="noreferrer">All Reports</a>
                <a href="/docs" target="_blank" rel="noreferrer">API Docs</a>
              </div>
              <div class="report">${escapeHtml(data.report || "No report text returned.")}</div>
              <details>
                <summary>Raw JSON</summary>
                <pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>
              </details>
            `;
          }

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            submit.disabled = true;
            state.textContent = "Running";
            setStatus("busy", "Running analysis", "Model inference");
            result.innerHTML = `<div class="empty">Processing MRI scan. Keep this tab open while the API runs the pipeline.</div>`;
            try {
              const response = await fetch("/api/analyze", { method: "POST", body: new FormData(form) });
              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || "Analysis failed");
              }
              setStatus("ok", "Analysis complete", data.pdf_file || "JSON ready");
              renderResult(data);
            } catch (error) {
              state.textContent = "Error";
              setStatus("err", "Analysis failed", "Check details");
              result.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
            } finally {
              submit.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return _index_html()


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return {"status": "ok", "service": APP_NAME, "version": APP_VERSION}


@app.get("/ready", response_model=ReadinessResponse, tags=["system"])
def ready():
    missing = _missing_model_files()
    return {"status": "ready" if not missing else "not_ready", "models_available": not missing, "missing_models": missing}


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(
    file: UploadFile = File(..., description="Brain MRI image. Supported: PNG, JPG, JPEG, BMP."),
    patient_name: str = Form("", max_length=120),
    patient_id: str = Form("", max_length=80),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file name supplied")

    source_name = Path(file.filename).name
    suffix = Path(source_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a PNG, JPG, JPEG, or BMP image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File is larger than {MAX_UPLOAD_MB} MB")

    tmp_path = save_upload_to_temp(contents, source_name)
    try:
        image = Image.open(tmp_path)
        image.verify()
        image = Image.open(tmp_path)
        result = await run_in_threadpool(analyze_mri, image, source_name, tmp_path, patient_name.strip(), patient_id.strip())
        if result.get("pdf_file"):
            result["pdf_url"] = f"/api/reports/{result['pdf_file']}"
        return result
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/history", response_model=HistoryResponse, tags=["history"])
def history(limit: int = 100):
    bounded_limit = max(1, min(limit, 500))
    items = load_history()[:bounded_limit]
    return {"count": len(items), "items": items}


@app.get("/api/reports", response_model=ReportsResponse, tags=["reports"])
def list_reports(limit: int = 100):
    bounded_limit = max(1, min(limit, 500))
    reports = sorted(REPORT_DIR.glob("*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)[:bounded_limit]
    items = [_report_payload(path) for path in reports]
    return {"count": len(items), "items": items}


@app.get("/api/reports/{filename}", tags=["reports"])
def get_report(filename: str):
    path = _resolve_report(filename)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["metadata"])
def model_info():
    model_files = {name: str(path) for name, path in _model_files().items()}
    return {
        "classes": CLASS_NAMES,
        "classification_input": f"{IMG_SIZE}x{IMG_SIZE}",
        "segmentation_input": f"{SEG_SIZE}x{SEG_SIZE}",
        "mri_params": MRI_PARAMS,
        "model_files": model_files,
    }


@app.get("/api/config", tags=["metadata"])
def runtime_config():
    return {
        "max_upload_mb": MAX_UPLOAD_MB,
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "models_dir": str(MODELS_DIR),
        "reports_dir": str(REPORT_DIR),
        "started_at": app.state.started_at.isoformat(),
    }
