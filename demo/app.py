#!/usr/bin/env python3
"""FastAPI demo UI for the final uveitis late-fusion model."""

from __future__ import annotations

import io
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_final_uveitis import lab_input_summary, source_lab_summary_from_sources
from late_fusion_inference import ImageEmbeddingExtractor, input_quality_summary, score_fusion
from uveitis_case import CLINICAL_COLUMNS
from uveitis_record_inputs import labs_with_source_summary_from_values


DEMO_DIR = REPO_ROOT / "demo"
FINAL_MODEL = REPO_ROOT / "artifacts/late_fusion_model.joblib"
SHOWCASE_METRICS = REPO_ROOT / "artifacts/model_metadata.json"
POSITIVE_LABEL = "感染性葡萄膜炎"
NEGATIVE_LABEL = "非感染性葡萄膜炎"
DISCLAIMER = "本系统用于项目演示和辅助分类，不作为独立临床诊断依据。"


app = FastAPI(title="感染性葡萄膜炎二分类展示", version="1.0.0")
app.mount("/static", StaticFiles(directory=DEMO_DIR / "static"), name="static")
templates = Environment(
    loader=FileSystemLoader(DEMO_DIR / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    if not FINAL_MODEL.exists():
        raise FileNotFoundError(f"missing model: {FINAL_MODEL}")
    return joblib.load(FINAL_MODEL)


@lru_cache(maxsize=4)
def load_extractor(model_name: str, device_name: str) -> ImageEmbeddingExtractor:
    return ImageEmbeddingExtractor(model_name, torch.device(device_name))


def read_showcase_metrics() -> dict[str, Any]:
    if not SHOWCASE_METRICS.exists():
        raise FileNotFoundError(f"missing model metadata: {SHOWCASE_METRICS}")
    return json.loads(SHOWCASE_METRICS.read_text(encoding="utf-8"))


async def uploaded_images(files: list[UploadFile]) -> list[Image.Image]:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传 1 张眼底图像。")
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="演示界面最多支持一次上传 4 张图像。")
    images: list[Image.Image] = []
    for file in files:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail=f"图像文件为空：{file.filename}")
        try:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"无法读取图像：{file.filename}") from exc
    return images


def labs_from_text(lab_text: str | None) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    cleaned = (lab_text or "").strip()
    if not cleaned:
        return {}, None, False
    result = labs_with_source_summary_from_values(
        structured_raw=None,
        structured_source=None,
        raw_text=cleaned,
        text_source="前端血常规文本",
    )
    summary = result["summary"]
    recognized_count = int(summary.get("merged_recognized_lab_count") or 0)
    if recognized_count <= 0:
        return {}, summary, True
    return result["labs"], summary, False


def compact_warning_messages(warnings: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for warning in warnings:
        message = warning.get("message")
        if message:
            messages.append(str(message))
    return messages


def predict_with_images(
    *,
    images: list[Image.Image],
    lab_text: str | None,
    threshold_preset: str,
    device_name: str = "cpu",
) -> dict[str, Any]:
    bundle = load_bundle()
    embedding_model_name = bundle.get("embedding_model") or "mobilenet_v3_large"
    extractor = load_extractor(str(embedding_model_name), device_name)
    image_embedding = extractor(images)
    labs, lab_source_summary, unusable_lab_text = labs_from_text(lab_text)
    prompt_mode = "full" if labs else "image_only"
    input_summary = lab_input_summary(labs)
    source_summary = source_lab_summary_from_sources(lab_source_summary)
    (
        raw_probability,
        calibrated_probability,
        threshold,
        positive,
        branch,
        resolved_preset,
        preset_metrics,
    ) = score_fusion(
        fusion_bundle=bundle,
        prompt_mode=prompt_mode,
        labs=labs,
        score_losses=None,
        score_delta=None,
        image_score_losses=None,
        image_score_delta=None,
        image_embedding=image_embedding,
        threshold_preset=threshold_preset,
    )
    quality = input_quality_summary(
        branch=branch,
        num_images=len(images),
        has_labs=bool(input_summary["has_labs"]),
        recognized_lab_count=int(input_summary["recognized_lab_count"]),
        missing_lab_count=int(input_summary["missing_lab_count"]),
        fusion_probability=raw_probability,
        fusion_threshold=threshold,
        source_has_labs=bool(source_summary["source_has_labs"]),
        lab_source_summary=lab_source_summary,
        expected_image_count=4,
        min_full_lab_values=5,
    )
    warning_codes = list(quality["warning_codes"])
    warnings = list(quality["warnings"])
    if unusable_lab_text:
        warning_codes.append("provided_lab_text_without_supported_cbc_values")
        warnings.append(
            {
                "code": "provided_lab_text_without_supported_cbc_values",
                "severity": "warning",
                "message": "已输入文本，但未识别到支持的血常规项目，本次自动退回仅图像分支。",
            }
        )
    prediction = POSITIVE_LABEL if positive else NEGATIVE_LABEL
    return {
        "prediction": prediction,
        "is_infectious": bool(positive),
        "infectious_probability": raw_probability,
        "calibrated_infectious_probability": calibrated_probability,
        "threshold": threshold,
        "threshold_preset": resolved_preset,
        "threshold_preset_metrics": preset_metrics,
        "branch": branch,
        "branch_display": "图像 + 血常规" if branch == "full" else "仅图像",
        "uses_labs": branch == "full",
        "recognized_lab_count": input_summary["recognized_lab_count"],
        "recognized_lab_columns": input_summary["recognized_lab_columns"],
        "missing_lab_count": input_summary["missing_lab_count"],
        "source_lab_summary": source_summary,
        "num_images": len(images),
        "image_embedding_model": embedding_model_name,
        "decision_margin": quality["decision_margin"],
        "warning_codes": warning_codes,
        "warning_messages": compact_warning_messages(warnings),
        "disclaimer": DISCLAIMER,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    template = templates.get_template("index.html")
    return HTMLResponse(template.render(request=request))


@app.get("/api/showcase")
async def api_showcase() -> dict[str, Any]:
    try:
        return read_showcase_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def api_predict(
    images: list[UploadFile] = File(...),
    lab_text: str | None = Form(default=None),
    threshold_preset: str = Form(default="selected"),
) -> dict[str, Any]:
    if threshold_preset not in {"selected", "balanced", "high_sensitivity", "high_specificity"}:
        raise HTTPException(status_code=400, detail="阈值模式不支持。")
    pil_images = await uploaded_images(images)
    try:
        return predict_with_images(
            images=pil_images,
            lab_text=lab_text,
            threshold_preset=threshold_preset,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"推理失败：{exc}") from exc
