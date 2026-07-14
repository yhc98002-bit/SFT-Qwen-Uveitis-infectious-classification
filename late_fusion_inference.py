#!/usr/bin/env python3
"""Lightweight inference helpers for the deployable uveitis late-fusion model."""

from __future__ import annotations

import math
from typing import Any

import torch
from PIL import Image

from uveitis_case import lab_feature_vector


def load_torchvision_embedding_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, Any]:
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        EfficientNet_B0_Weights,
        MobileNet_V3_Large_Weights,
        ResNet18_Weights,
        ResNet50_Weights,
        convnext_tiny,
        efficientnet_b0,
        mobilenet_v3_large,
        resnet18,
        resnet50,
    )

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[-1] = torch.nn.Identity()
    elif model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = torch.nn.Identity()
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        model.classifier[-1] = torch.nn.Identity()
    else:
        raise ValueError(f"unsupported embedding model: {model_name}")
    model.to(device)
    model.eval()
    return model, weights.transforms()


def parse_embedding_model_names(model_name: str) -> list[str]:
    model_names = [
        name.strip()
        for part in model_name.split("+")
        for name in part.split(",")
        if name.strip()
    ]
    if not model_names:
        raise ValueError("embedding model name is empty")
    return model_names


class ImageEmbeddingExtractor:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.models: list[tuple[torch.nn.Module, Any]] = [
            load_torchvision_embedding_model(single_model_name, device)
            for single_model_name in parse_embedding_model_names(model_name)
        ]

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> list[float]:
        values: list[float] = []
        for model, transform in self.models:
            batch = torch.stack([transform(image) for image in images]).to(self.device)
            features = model(batch).detach().cpu().float()
            values.extend(features.mean(dim=0).tolist())
        return values


@torch.no_grad()
def image_embedding_values(
    images: list[Image.Image],
    model_name: str,
    device: torch.device,
) -> list[float]:
    return ImageEmbeddingExtractor(model_name, device)(images)


def qwen_score_feature_values(score_losses: dict[str, float], score_delta: float) -> list[float]:
    positive_loss = float(score_losses["感染性葡萄膜炎"])
    negative_loss = float(score_losses["非感染性葡萄膜炎"])
    return [
        float(score_delta),
        positive_loss,
        negative_loss,
        negative_loss - positive_loss,
    ]


def input_quality_summary(
    *,
    branch: str,
    num_images: int,
    has_labs: bool,
    recognized_lab_count: int,
    missing_lab_count: int,
    fusion_probability: float,
    fusion_threshold: float,
    source_has_labs: bool | None = None,
    lab_source_summary: dict[str, Any] | None = None,
    expected_image_count: int = 4,
    min_full_lab_values: int = 5,
    borderline_margin: float = 0.03,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    if num_images != expected_image_count:
        warnings.append(
            {
                "code": "image_count_differs_from_training_default",
                "severity": "warning",
                "message": (
                    f"Expected {expected_image_count} images based on the training setup, "
                    f"but received {num_images}."
                ),
                "expected_image_count": expected_image_count,
                "actual_image_count": num_images,
            }
        )

    if branch == "full":
        if not has_labs or recognized_lab_count <= 0:
            warnings.append(
                {
                    "code": "full_branch_without_recognized_labs",
                    "severity": "warning",
                    "message": "The full branch was selected but no supported blood routine values were recognized.",
                    "recognized_lab_count": recognized_lab_count,
                }
            )
        elif recognized_lab_count < min_full_lab_values:
            warnings.append(
                {
                    "code": "limited_lab_values",
                    "severity": "warning",
                    "message": (
                        f"Only {recognized_lab_count} supported blood routine values were recognized; "
                        f"at least {min_full_lab_values} are recommended for a more stable full-branch score."
                    ),
                    "recognized_lab_count": recognized_lab_count,
                    "recommended_min_lab_values": min_full_lab_values,
                }
            )
        if missing_lab_count > 0:
            warnings.append(
                {
                    "code": "missing_lab_values_imputed",
                    "severity": "info",
                    "message": (
                        f"{missing_lab_count} training blood routine fields were missing and will be handled "
                        "by the model pipeline imputer."
                    ),
                    "missing_lab_count": missing_lab_count,
                }
            )
        if lab_source_summary:
            conflict_count = int(lab_source_summary.get("conflict_count") or 0)
            if conflict_count:
                warnings.append(
                    {
                        "code": "conflicting_lab_values_across_sources",
                        "severity": "warning",
                        "message": (
                            f"{conflict_count} blood routine values conflict across structured and text sources; "
                            "structured values are used for duplicate columns."
                        ),
                        "conflict_count": conflict_count,
                        "conflicts": lab_source_summary.get("conflicts") or [],
                    }
                )
            merged_count = int(lab_source_summary.get("merged_recognized_lab_count") or 0)
            structured_count = int(lab_source_summary.get("structured_recognized_lab_count") or 0)
            text_count = int(lab_source_summary.get("text_recognized_lab_count") or 0)
            if (
                lab_source_summary.get("merged_from_multiple_sources") is True
                and merged_count > max(structured_count, text_count)
            ):
                warnings.append(
                    {
                        "code": "merged_lab_values_from_multiple_sources",
                        "severity": "info",
                        "message": "Structured and text blood routine sources were merged for full-branch inference.",
                        "structured_recognized_lab_count": structured_count,
                        "text_recognized_lab_count": text_count,
                        "merged_recognized_lab_count": merged_count,
                    }
                )
    elif source_has_labs is True:
        warnings.append(
            {
                "code": "image_only_branch_ignores_source_labs",
                "severity": "info",
                "message": "The source case contains blood routine values, but this image-only branch does not use them.",
            }
        )

    if lab_source_summary:
        source_fields_present = bool(
            lab_source_summary.get("structured_lab_source") or lab_source_summary.get("text_lab_source")
        )
        merged_count = int(lab_source_summary.get("merged_recognized_lab_count") or 0)
        if branch == "image_only" and source_fields_present and merged_count <= 0:
            warnings.append(
                {
                    "code": "image_only_fallback_no_supported_labs",
                    "severity": "warning",
                    "message": (
                        "The source included text or structured lab fields, but no supported blood routine "
                        "values were recognized. The image-only branch was used."
                    ),
                    "structured_lab_source": lab_source_summary.get("structured_lab_source"),
                    "text_lab_source": lab_source_summary.get("text_lab_source"),
                }
            )
        unsupported_structured_keys = lab_source_summary.get("unsupported_structured_lab_keys") or []
        if unsupported_structured_keys:
            warnings.append(
                {
                    "code": "unsupported_structured_lab_keys_ignored",
                    "severity": "info",
                    "message": (
                        "Some structured lab keys are not supported blood routine fields and were not used "
                        "as model features."
                    ),
                    "unsupported_structured_lab_keys": unsupported_structured_keys,
                }
            )
        ignored_text_terms = lab_source_summary.get("ignored_text_signal_terms") or []
        if ignored_text_terms:
            warnings.append(
                {
                    "code": "unsupported_text_infection_or_inflammation_markers_ignored",
                    "severity": "info",
                    "message": (
                        "The text contains non-CBC infection or inflammation markers. This model uses only "
                        "supported blood routine fields plus image embeddings, so these markers were not used "
                        "as model features."
                    ),
                    "ignored_text_signal_terms": ignored_text_terms,
                }
            )

    decision_margin = abs(float(fusion_probability) - float(fusion_threshold))
    if decision_margin <= borderline_margin:
        warnings.append(
            {
                "code": "raw_probability_near_threshold",
                "severity": "warning",
                "message": (
                    f"The raw decision score is close to the selected threshold "
                    f"(margin={decision_margin:.4f})."
                ),
                "decision_margin": decision_margin,
                "borderline_margin": borderline_margin,
            }
        )

    return {
        "expected_image_count": expected_image_count,
        "min_full_lab_values": min_full_lab_values,
        "borderline_margin": borderline_margin,
        "decision_margin": decision_margin,
        "warning_count": len(warnings),
        "warning_codes": [warning["code"] for warning in warnings],
        "warnings": warnings,
    }


def probability_field_metadata(
    *,
    raw_probability_field: str,
    calibrated_probability_field: str,
    threshold_field: str,
    positive_label: str = "感染性葡萄膜炎",
    negative_label: str = "非感染性葡萄膜炎",
) -> dict[str, Any]:
    return {
        "classification_probability_field": raw_probability_field,
        "classification_threshold_field": threshold_field,
        "classification_rule": f"{positive_label} if {raw_probability_field} >= {threshold_field}, else {negative_label}",
        "calibrated_probability_field": calibrated_probability_field,
        "calibrated_probability_role": "risk_display_only",
        "note": "The calibrated probability is for risk display and calibration diagnostics; classification uses the raw probability and branch threshold.",
    }


def fusion_feature_values(
    branch: dict[str, Any],
    labs: dict[str, Any],
    score_losses: dict[str, float] | None,
    score_delta: float | None,
    image_score_losses: dict[str, float] | None,
    image_score_delta: float | None,
    image_embedding: list[float] | None,
) -> list[float | None]:
    values: list[float | None] = []
    for group in branch["feature_groups"]:
        if group == "labs":
            values.extend(lab_feature_vector(labs))
        elif group == "full_scores":
            if score_losses is None or score_delta is None:
                raise ValueError("fusion branch requires full Qwen score features")
            values.extend(qwen_score_feature_values(score_losses, score_delta))
        elif group == "image_scores":
            if image_score_losses is None or image_score_delta is None:
                raise ValueError("fusion branch requires image-only Qwen score features")
            values.extend(qwen_score_feature_values(image_score_losses, image_score_delta))
        elif group == "image_embedding":
            if image_embedding is None:
                raise ValueError("fusion branch requires image embedding features")
            values.extend(image_embedding)
        else:
            raise ValueError(f"unsupported fusion feature group: {group}")
    if len(values) != len(branch["feature_names"]):
        raise ValueError(
            f"fusion feature length mismatch: got {len(values)}, expected {len(branch['feature_names'])}"
        )
    return values


def calibrated_fusion_probability(branch: dict[str, Any], raw_probability: float) -> float | None:
    calibrator = branch.get("probability_calibrator")
    if calibrator is None:
        return None
    epsilon = float(calibrator.get("clip_epsilon", 1e-6))
    clipped = min(max(float(raw_probability), epsilon), 1.0 - epsilon)
    logit = math.log(clipped / (1.0 - clipped))
    return float(calibrator["model"].predict_proba([[logit]])[0][1])


def score_fusion(
    fusion_bundle: dict[str, Any],
    prompt_mode: str,
    labs: dict[str, Any],
    score_losses: dict[str, float] | None,
    score_delta: float | None,
    image_score_losses: dict[str, float] | None,
    image_score_delta: float | None,
    image_embedding: list[float] | None,
    threshold_preset: str,
) -> tuple[float, float | None, float, bool, str, str, dict[str, Any] | None]:
    branch_name = "full" if prompt_mode == "full" and labs else "image_only"
    branch = fusion_bundle[branch_name]
    features = fusion_feature_values(
        branch=branch,
        labs=labs,
        score_losses=score_losses,
        score_delta=score_delta,
        image_score_losses=image_score_losses,
        image_score_delta=image_score_delta,
        image_embedding=image_embedding,
    )
    probability = float(branch["model"].predict_proba([features])[0][1])
    calibrated_probability = calibrated_fusion_probability(branch, probability)
    preset_metrics = None
    resolved_preset = threshold_preset
    if threshold_preset == "selected":
        threshold = float(branch["threshold"])
        preset_metrics = (branch.get("threshold_presets") or {}).get("selected", {}).get("metrics")
    else:
        presets = branch.get("threshold_presets") or {}
        if threshold_preset not in presets:
            available = ["selected", *sorted(presets)]
            raise ValueError(f"unknown fusion threshold preset: {threshold_preset}; available={available}")
        threshold = float(presets[threshold_preset]["threshold"])
        preset_metrics = presets[threshold_preset].get("metrics")
    return probability, calibrated_probability, threshold, probability >= threshold, branch_name, resolved_preset, preset_metrics
