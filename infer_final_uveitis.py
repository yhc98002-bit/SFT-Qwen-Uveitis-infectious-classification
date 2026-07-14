#!/usr/bin/env python3
"""Run lightweight inference with the final uveitis late-fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import torch
from PIL import Image

from late_fusion_inference import (
    ImageEmbeddingExtractor,
    input_quality_summary,
    parse_embedding_model_names,
    probability_field_metadata,
    score_fusion,
)
from uveitis_case import (
    CLINICAL_COLUMNS,
    build_image_only_prompt,
    build_prompt,
    lab_feature_vector,
    labs_from_prompt,
    normalize_labs_mapping,
    prompt_has_lab_values,
)
from uveitis_image_inputs import image_paths_from_dir, image_paths_from_record
from uveitis_record_inputs import labs_with_source_summary_from_values, record_labs_with_source_summary, record_text


REPO_ROOT = Path(__file__).resolve().parent
POSITIVE_LABEL = "感染性葡萄膜炎"
NEGATIVE_LABEL = "非感染性葡萄膜炎"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_labs_json(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("labs JSON must be an object")
    return data


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing image: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def lab_source_fields_present(lab_source_summary: dict[str, Any] | None) -> bool:
    if not lab_source_summary:
        return False
    return bool(lab_source_summary.get("structured_lab_source") or lab_source_summary.get("text_lab_source"))


def source_lab_summary_from_sources(lab_source_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not lab_source_summary:
        return {
            "source_has_labs": False,
            "source_lab_fields_present": False,
            "source_recognized_lab_count": 0,
            "source_recognized_lab_columns": [],
            "unusable_lab_source": False,
        }
    source_recognized_lab_count = int(lab_source_summary.get("merged_recognized_lab_count") or 0)
    source_fields_present = lab_source_fields_present(lab_source_summary)
    return {
        "source_has_labs": source_recognized_lab_count > 0,
        "source_lab_fields_present": source_fields_present,
        "source_recognized_lab_count": source_recognized_lab_count,
        "source_recognized_lab_columns": lab_source_summary.get("merged_recognized_lab_columns") or [],
        "unusable_lab_source": source_fields_present and source_recognized_lab_count <= 0,
    }


def record_input(
    record: dict[str, Any],
    *,
    image_base_dir: Path,
    record_base_dir: Path,
    record_label: str,
    allow_image_only_fallback: bool = False,
) -> tuple[str | None, str, list[Path], str | None, str, dict[str, Any], dict[str, Any] | None]:
    image_paths = image_paths_from_record(record, image_base_dir, record_label=record_label)
    lab_source_result = record_labs_with_source_summary(record, record_base_dir, record_label=record_label)
    labs = lab_source_result["labs"]
    lab_source_summary = lab_source_result["summary"]
    raw_prompt = record_text(record, record_base_dir, record_label=record_label)

    if labs:
        prompt = build_prompt(
            sex=record.get("sex", record.get("性别")),
            age=record.get("age", record.get("年龄")),
            chief_complaint=record.get("chief_complaint", record.get("主诉")),
            labs=labs,
        )
        if prompt_has_lab_values(prompt):
            prompt_mode = "full"
        elif raw_prompt:
            prompt = raw_prompt
            labs = labs_from_prompt(prompt)
            if not labs:
                raise ValueError(f"{record_label} labs/prompt must include at least one supported blood routine value")
            prompt_mode = "full"
        else:
            raise ValueError(f"{record_label} labs must include at least one supported blood routine value")
    elif raw_prompt or lab_source_summary.get("structured_lab_source") or lab_source_summary.get("text_lab_source"):
        if allow_image_only_fallback:
            prompt = build_image_only_prompt()
            labs = {}
            prompt_mode = "image_only"
        else:
            raise ValueError(
                f"{record_label} text/labs inputs must include at least one supported blood routine value. "
                "Omit the text/labs to run pure image-only inference, or pass --allow-image-only-fallback "
                "to explicitly fall back to image-only when provided text/labs are unusable."
            )
    else:
        prompt = build_image_only_prompt()
        prompt_mode = "image_only"

    target = record.get("answer") or record.get("label") or record.get("target")
    record_id = record.get("id")
    return record_id, prompt, image_paths, target, prompt_mode, labs, lab_source_summary


def read_case_json(
    path: Path,
    *,
    allow_image_only_fallback: bool = False,
) -> tuple[str | None, str, list[Path], str | None, str, dict[str, Any], dict[str, Any] | None]:
    case = json.loads(path.read_text(encoding="utf-8"))
    return record_input(
        case,
        image_base_dir=path.parent,
        record_base_dir=path.parent,
        record_label=f"case JSON {path}",
        allow_image_only_fallback=allow_image_only_fallback,
    )


def command_line_input(args: argparse.Namespace) -> tuple[str | None, str, list[Path], str | None, str, dict[str, Any], dict[str, Any] | None]:
    image_paths = list(args.image)
    if args.image_dir is not None:
        image_paths.extend(image_paths_from_dir(args.image_dir, label="--image-dir"))
    if not image_paths:
        raise ValueError("provide images for command-line inference via --image or --image-dir")
    if args.labs_json is not None and args.labs_json_file is not None:
        raise ValueError("provide only one of --labs-json or --labs-json-file")
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("provide only one of --prompt or --prompt-file")

    structured_labs: dict[str, Any] | None = None
    structured_source: str | None = None
    if args.labs_json_file is not None:
        structured_labs = parse_labs_json(args.labs_json_file.read_text(encoding="utf-8"))
        structured_source = "--labs-json-file"
    elif args.labs_json is not None:
        structured_labs = parse_labs_json(args.labs_json)
        structured_source = "--labs-json"

    raw_prompt: str | None = None
    text_source: str | None = None
    if args.prompt_file is not None:
        raw_prompt = args.prompt_file.read_text(encoding="utf-8")
        text_source = "--prompt-file"
    elif args.prompt is not None:
        raw_prompt = args.prompt
        text_source = "--prompt"

    lab_source_result = labs_with_source_summary_from_values(
        structured_raw=structured_labs,
        structured_source=structured_source,
        raw_text=raw_prompt,
        text_source=text_source,
    )
    labs = lab_source_result["labs"]
    lab_source_summary = lab_source_result["summary"] if structured_source is not None or text_source is not None else None
    recognized_lab_count = 0
    if lab_source_summary:
        recognized_lab_count = int(lab_source_summary.get("merged_recognized_lab_count") or 0)

    if recognized_lab_count > 0:
        prompt = build_prompt(
            sex=args.sex,
            age=args.age,
            chief_complaint=args.chief_complaint,
            labs=labs,
        )
        if not prompt_has_lab_values(prompt):
            raise ValueError("provided blood routine inputs did not produce a valid full-branch prompt")
        prompt_mode = "full"
    elif structured_source is None and text_source is None:
        prompt = build_image_only_prompt()
        prompt_mode = "image_only"
    elif args.allow_image_only_fallback:
        prompt = build_image_only_prompt()
        labs = {}
        prompt_mode = "image_only"
    else:
        sources = " and ".join(source for source in [structured_source, text_source] if source is not None)
        raise ValueError(
            f"{sources} must include at least one supported blood routine value. "
            "Omit the text/labs to run pure image-only inference, or pass --allow-image-only-fallback "
            "to explicitly fall back to image-only when provided text/labs are unusable."
        )

    return None, prompt, image_paths, None, prompt_mode, labs, lab_source_summary


def lab_input_summary(labs: dict[str, Any]) -> dict[str, Any]:
    normalized_labs = normalize_labs_mapping(labs)
    lab_values = lab_feature_vector(normalized_labs)
    recognized_columns = [
        column
        for column, value in zip(CLINICAL_COLUMNS, lab_values)
        if value is not None
    ]
    missing_columns = [
        column
        for column in CLINICAL_COLUMNS
        if column not in recognized_columns
    ]
    return {
        "has_labs": bool(recognized_columns),
        "recognized_lab_count": len(recognized_columns),
        "recognized_lab_columns": recognized_columns,
        "missing_lab_count": len(missing_columns),
        "missing_lab_columns": missing_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data/prepared_4img384",
        help=(
            "Root used to resolve image paths from JSONL records. Defaults to data/prepared_4img384 "
            "for prepared repo JSONL files. JSONL prompt/labs sidecar files are resolved relative to the JSONL file."
        ),
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None, help="Path to a text file containing blood routine values.")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing case images; supported files are read in filename order.",
    )
    parser.add_argument("--labs-json", type=str, default=None, help="Blood routine values as a JSON object.")
    parser.add_argument("--labs-json-file", type=Path, default=None, help="Path to a JSON object with blood routine values.")
    parser.add_argument("--sex", type=str, default=None)
    parser.add_argument("--age", type=str, default=None)
    parser.add_argument("--chief-complaint", type=str, default=None)
    parser.add_argument(
        "--fusion-model",
        type=Path,
        default=REPO_ROOT / "artifacts/late_fusion_model.joblib",
    )
    parser.add_argument("--fusion-threshold-preset", default="selected")
    parser.add_argument(
        "--expected-image-count",
        type=int,
        default=4,
        help="Expected number of images for input-quality warnings. Defaults to the training setup of 4.",
    )
    parser.add_argument(
        "--min-full-lab-values",
        type=int,
        default=5,
        help="Minimum recognized blood routine values recommended before trusting full-branch input quality.",
    )
    parser.add_argument(
        "--strict-full-labs",
        action="store_true",
        help="Fail instead of only warning when full-branch input has fewer than --min-full-lab-values values.",
    )
    parser.add_argument(
        "--allow-image-only-fallback",
        action="store_true",
        help=(
            "If text or structured lab inputs are provided but no supported blood routine values are recognized, "
            "use the image-only branch and emit an input-quality warning instead of failing."
        ),
    )
    parser.add_argument(
        "--no-image-only-control",
        action="store_true",
        help="Do not include the image-only control score when labs drive the full branch.",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mode_count = sum(
        [
            args.jsonl is not None,
            args.case_json is not None,
            args.prompt is not None
            or args.prompt_file is not None
            or bool(args.image)
            or args.image_dir is not None
            or args.labs_json is not None
            or args.labs_json_file is not None,
        ]
    )
    if mode_count != 1:
        raise ValueError(
            "provide exactly one input mode: --jsonl/--index, --case-json, or --image/--image-dir with optional prompt/labs"
        )

    if args.jsonl is not None:
        records = read_jsonl(args.jsonl)
        record = records[args.index]
        record_id, prompt, image_paths, target, prompt_mode, labs, lab_source_summary = record_input(
            record,
            image_base_dir=args.data_root,
            record_base_dir=args.jsonl.parent,
            record_label=f"JSONL {args.jsonl} index {args.index}",
            allow_image_only_fallback=args.allow_image_only_fallback,
        )
    elif args.case_json is not None:
        record_id, prompt, image_paths, target, prompt_mode, labs, lab_source_summary = read_case_json(
            args.case_json,
            allow_image_only_fallback=args.allow_image_only_fallback,
        )
    else:
        record_id, prompt, image_paths, target, prompt_mode, labs, lab_source_summary = command_line_input(args)

    input_summary = lab_input_summary(labs)
    source_summary = source_lab_summary_from_sources(lab_source_summary)
    if (
        args.strict_full_labs
        and prompt_mode == "full"
        and int(input_summary["recognized_lab_count"]) < args.min_full_lab_values
    ):
        raise ValueError(
            "full-branch inference recognized too few supported blood routine values: "
            f"{input_summary['recognized_lab_count']} < {args.min_full_lab_values}. "
            "Provide more values, lower --min-full-lab-values, or omit text/labs to use image-only inference."
        )

    fusion_bundle = joblib.load(args.fusion_model)
    embedding_model_name = args.embedding_model or fusion_bundle.get("embedding_model") or "mobilenet_v3_large"
    device = torch.device(args.device)
    images = load_images(image_paths)
    image_embedding = ImageEmbeddingExtractor(embedding_model_name, device)(images)
    (
        fusion_raw_probability,
        fusion_calibrated_probability,
        fusion_threshold,
        fusion_positive,
        fusion_branch,
        fusion_threshold_preset,
        fusion_threshold_preset_metrics,
    ) = score_fusion(
        fusion_bundle=fusion_bundle,
        prompt_mode=prompt_mode,
        labs=labs,
        score_losses=None,
        score_delta=None,
        image_score_losses=None,
        image_score_delta=None,
        image_embedding=image_embedding,
        threshold_preset=args.fusion_threshold_preset,
    )
    branch_feature_groups = list(fusion_bundle[fusion_branch]["feature_groups"])
    quality_summary = input_quality_summary(
        branch=fusion_branch,
        num_images=len(image_paths),
        has_labs=bool(input_summary["has_labs"]),
        recognized_lab_count=int(input_summary["recognized_lab_count"]),
        missing_lab_count=int(input_summary["missing_lab_count"]),
        fusion_probability=fusion_raw_probability,
        fusion_threshold=fusion_threshold,
        lab_source_summary=lab_source_summary,
        expected_image_count=args.expected_image_count,
        min_full_lab_values=args.min_full_lab_values,
    )
    prediction = POSITIVE_LABEL if fusion_positive else NEGATIVE_LABEL
    result = {
        "id": record_id,
        "prediction": prediction,
        "normalized_prediction": prediction,
        "decision_method": "fusion",
        "target": target,
        "prompt_mode": prompt_mode,
        "num_images": len(image_paths),
        **input_summary,
        **source_summary,
        "image_embedding_model": embedding_model_name,
        "image_embedding_models": parse_embedding_model_names(embedding_model_name),
        "image_embedding_dim": len(image_embedding),
        "fusion_feature_groups": branch_feature_groups,
        "fusion_uses_labs": "labs" in branch_feature_groups,
        "fusion_uses_image_embedding": "image_embedding" in branch_feature_groups,
        "fusion_probability": fusion_raw_probability,
        "fusion_raw_probability": fusion_raw_probability,
        "fusion_calibrated_probability": fusion_calibrated_probability,
        "fusion_threshold": fusion_threshold,
        "fusion_threshold_preset": fusion_threshold_preset,
        "fusion_threshold_preset_metrics": fusion_threshold_preset_metrics,
        "fusion_positive": fusion_positive,
        "fusion_branch": fusion_branch,
        "fusion_source": str(args.fusion_model),
        "lab_source_summary": lab_source_summary,
        "probability_fields": probability_field_metadata(
            raw_probability_field="fusion_raw_probability",
            calibrated_probability_field="fusion_calibrated_probability",
            threshold_field="fusion_threshold",
        ),
        "decision_margin": quality_summary["decision_margin"],
        "input_quality": quality_summary,
        "input_warnings": quality_summary["warnings"],
        "input_warning_codes": quality_summary["warning_codes"],
    }
    if not args.no_image_only_control and prompt_mode == "full" and bool(input_summary["has_labs"]) and "image_only" in fusion_bundle:
        (
            image_raw_probability,
            image_calibrated_probability,
            image_threshold,
            image_positive,
            image_branch,
            image_threshold_preset,
            image_threshold_preset_metrics,
        ) = score_fusion(
            fusion_bundle=fusion_bundle,
            prompt_mode="image_only",
            labs={},
            score_losses=None,
            score_delta=None,
            image_score_losses=None,
            image_score_delta=None,
            image_embedding=image_embedding,
            threshold_preset=args.fusion_threshold_preset,
        )
        image_branch_feature_groups = list(fusion_bundle[image_branch]["feature_groups"])
        image_quality_summary = input_quality_summary(
            branch=image_branch,
            num_images=len(image_paths),
            has_labs=False,
            recognized_lab_count=0,
            missing_lab_count=len(CLINICAL_COLUMNS),
            fusion_probability=image_raw_probability,
            fusion_threshold=image_threshold,
            source_has_labs=bool(input_summary["has_labs"]),
            lab_source_summary=lab_source_summary,
            expected_image_count=args.expected_image_count,
            min_full_lab_values=args.min_full_lab_values,
        )
        image_prediction = POSITIVE_LABEL if image_positive else NEGATIVE_LABEL
        result["image_only_control"] = {
            "prediction": image_prediction,
            "normalized_prediction": image_prediction,
            "fusion_probability": image_raw_probability,
            "fusion_raw_probability": image_raw_probability,
            "fusion_calibrated_probability": image_calibrated_probability,
            "fusion_threshold": image_threshold,
            "fusion_threshold_preset": image_threshold_preset,
            "fusion_threshold_preset_metrics": image_threshold_preset_metrics,
            "fusion_positive": image_positive,
            "fusion_branch": image_branch,
            "fusion_feature_groups": image_branch_feature_groups,
            "fusion_uses_labs": "labs" in image_branch_feature_groups,
            "fusion_uses_image_embedding": "image_embedding" in image_branch_feature_groups,
            "probability_fields": probability_field_metadata(
                raw_probability_field="fusion_raw_probability",
                calibrated_probability_field="fusion_calibrated_probability",
                threshold_field="fusion_threshold",
            ),
            "decision_margin": image_quality_summary["decision_margin"],
            "input_quality": image_quality_summary,
            "input_warnings": image_quality_summary["warnings"],
            "input_warning_codes": image_quality_summary["warning_codes"],
        }
        result["fusion_vs_image_only_delta"] = {
            "raw_probability_delta": fusion_raw_probability - image_raw_probability,
            "calibrated_probability_delta": None
            if fusion_calibrated_probability is None or image_calibrated_probability is None
            else fusion_calibrated_probability - image_calibrated_probability,
            "prediction_changed": prediction != image_prediction,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
