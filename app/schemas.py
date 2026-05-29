from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApiError(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    models_available: bool
    missing_models: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    no_tumor: bool
    label: str
    confidence: float
    severity: str
    filename: str
    patient_name: str = ""
    patient_id: str = ""
    report: str
    pdf_path: str | None = None
    pdf_file: str | None = None
    pdf_url: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    fusion: dict[str, Any] = Field(default_factory=dict)
    model_scores: dict[str, Any] = Field(default_factory=dict)
    clinical: dict[str, Any] = Field(default_factory=dict)
    size_info: dict[str, Any] | None = None
    shape_info: dict[str, Any] | None = None
    mass_info: dict[str, Any] | None = None
    risk_info: dict[str, Any] | None = None
    cal_info: dict[str, Any] | None = None
    rano: dict[str, Any] | None = None
    overlap: dict[str, Any] | None = None
    gate_info: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None


class HistoryResponse(BaseModel):
    count: int
    items: list[dict[str, Any]]


class ReportFile(BaseModel):
    filename: str
    url: str
    size_bytes: int
    modified_at: str


class ReportsResponse(BaseModel):
    count: int
    items: list[ReportFile]


class ModelInfoResponse(BaseModel):
    classes: list[str]
    classification_input: str
    segmentation_input: str
    mri_params: dict[str, str]
    model_files: dict[str, str]
