#!/usr/bin/env python3
"""Evaluate the final late-fusion model on an external case JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import torch
from PIL import Image

from late_fusion_inference import ImageEmbeddingExtractor, input_quality_summary, probability_field_metadata, score_fusion
from uveitis_case import CLINICAL_COLUMNS, lab_feature_vector, normalize_labs_mapping
from uveitis_image_inputs import image_paths_from_record
from uveitis_metrics import LABELS, binary_metrics, calibration_bins, probability_summary, usable_rows, write_jsonl
from uveitis_record_inputs import record_labs_with_source_summary


REPO_ROOT = Path(__file__).resolve().parent
POSITIVE_LABEL = "感染性葡萄膜炎"
NEGATIVE_LABEL = "非感染性葡萄膜炎"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing image: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


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


def lab_source_fields_present(lab_source_summary: dict[str, Any] | None) -> bool:
    if not lab_source_summary:
        return False
    return bool(lab_source_summary.get("structured_lab_source") or lab_source_summary.get("text_lab_source"))


def has_unusable_lab_source(lab_source_summary: dict[str, Any] | None) -> bool:
    if not lab_source_fields_present(lab_source_summary):
        return False
    return int(lab_source_summary.get("merged_recognized_lab_count") or 0) <= 0


def labs_from_record(record: dict[str, Any], record_base_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record_label = f"record {record.get('id')}"
    lab_source_result = record_labs_with_source_summary(record, record_base_dir, record_label=record_label)
    return lab_source_result["labs"], lab_source_result["summary"]


def target_from_record(record: dict[str, Any]) -> str | None:
    target = record.get("label") or record.get("answer") or record.get("target")
    if target is None:
        return None
    if target not in LABELS:
        raise ValueError(f"record {record.get('id')} has unsupported target label: {target!r}")
    return str(target)


def modes_for_record(record: dict[str, Any], mode: str, labs: dict[str, Any]) -> list[str]:
    if mode == "auto":
        return ["full" if labs else "image_only"]
    if mode == "both":
        if not labs:
            raise ValueError(f"record {record.get('id')} cannot run mode=both without labs/text values")
        return ["full", "image_only"]
    if mode == "full" and not labs:
        raise ValueError(f"record {record.get('id')} cannot run mode=full without labs/text values")
    return [mode]


def evaluate_record(
    record: dict[str, Any],
    image_embedding: list[float],
    labs: dict[str, Any],
    fusion_bundle: dict[str, Any],
    threshold_preset: str,
    mode: str,
    num_images: int,
    expected_image_count: int,
    min_full_lab_values: int,
    strict_full_labs: bool,
    lab_source_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    target = target_from_record(record)
    rows: list[dict[str, Any]] = []
    source_lab_summary = lab_input_summary(labs)
    source_lab_fields_present = lab_source_fields_present(lab_source_summary)
    unusable_lab_source = has_unusable_lab_source(lab_source_summary)
    for branch_mode in modes_for_record(record, mode, labs):
        prompt_mode = "full" if branch_mode == "full" else "image_only"
        branch_labs = labs if branch_mode == "full" else {}
        branch_lab_summary = lab_input_summary(branch_labs)
        if (
            strict_full_labs
            and branch_mode == "full"
            and int(branch_lab_summary["recognized_lab_count"]) < min_full_lab_values
        ):
            raise ValueError(
                f"record {record.get('id')} full branch recognized too few supported blood routine values: "
                f"{branch_lab_summary['recognized_lab_count']} < {min_full_lab_values}"
            )
        (
            probability,
            calibrated_probability,
            threshold,
            positive,
            branch,
            resolved_preset,
            preset_metrics,
        ) = score_fusion(
            fusion_bundle=fusion_bundle,
            prompt_mode=prompt_mode,
            labs=branch_labs,
            score_losses=None,
            score_delta=None,
            image_score_losses=None,
            image_score_delta=None,
            image_embedding=image_embedding,
            threshold_preset=threshold_preset,
        )
        prediction = POSITIVE_LABEL if positive else NEGATIVE_LABEL
        feature_groups = list(fusion_bundle[branch]["feature_groups"])
        quality_summary = input_quality_summary(
            branch=branch,
            num_images=num_images,
            has_labs=bool(branch_lab_summary["has_labs"]),
            recognized_lab_count=int(branch_lab_summary["recognized_lab_count"]),
            missing_lab_count=int(branch_lab_summary["missing_lab_count"]),
            fusion_probability=probability,
            fusion_threshold=threshold,
            source_has_labs=bool(source_lab_summary["has_labs"]),
            lab_source_summary=lab_source_summary,
            expected_image_count=expected_image_count,
            min_full_lab_values=min_full_lab_values,
        )
        row = {
            "id": record.get("id"),
            "branch": branch,
            "mode": branch_mode,
            "prediction": prediction,
            "target": target,
            "fusion_feature_groups": feature_groups,
            "fusion_uses_labs": "labs" in feature_groups,
            "fusion_uses_image_embedding": "image_embedding" in feature_groups,
            "infectious_probability": probability,
            "raw_infectious_probability": probability,
            "calibrated_infectious_probability": calibrated_probability,
            "fusion_threshold": threshold,
            "fusion_threshold_preset": resolved_preset,
            "fusion_threshold_preset_metrics": preset_metrics,
            "probability_fields": probability_field_metadata(
                raw_probability_field="raw_infectious_probability",
                calibrated_probability_field="calibrated_infectious_probability",
                threshold_field="fusion_threshold",
            ),
            "num_images": num_images,
            **branch_lab_summary,
            "source_has_labs": source_lab_summary["has_labs"],
            "source_lab_fields_present": source_lab_fields_present,
            "source_recognized_lab_count": source_lab_summary["recognized_lab_count"],
            "source_recognized_lab_columns": source_lab_summary["recognized_lab_columns"],
            "lab_source_summary": lab_source_summary,
            "unusable_lab_source": unusable_lab_source,
            "decision_margin": quality_summary["decision_margin"],
            "input_quality": quality_summary,
            "input_warnings": quality_summary["warnings"],
            "input_warning_codes": quality_summary["warning_codes"],
        }
        rows.append(row)
    return rows


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_branch: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_branch.setdefault(str(row["branch"]), []).append(row)
    branches: dict[str, Any] = {}
    for branch, branch_rows in sorted(by_branch.items()):
        labeled_rows = [row for row in branch_rows if row.get("target") in LABELS]
        calibrated_rows = usable_rows(
            labeled_rows,
            probability_field="calibrated_infectious_probability",
        )
        warning_counts = Counter(
            code
            for row in branch_rows
            for code in (row.get("input_warning_codes") or [])
        )
        branches[branch] = {
            "rows": len(branch_rows),
            "labeled_rows": len(labeled_rows),
            "unlabeled_rows": len(branch_rows) - len(labeled_rows),
            "rows_with_lab_source_fields": sum(row.get("source_lab_fields_present") is True for row in branch_rows),
            "rows_with_source_labs": sum(row.get("source_has_labs") is True for row in branch_rows),
            "rows_with_unusable_lab_sources": sum(row.get("unusable_lab_source") is True for row in branch_rows),
            "rows_with_branch_labs": sum(row.get("has_labs") is True for row in branch_rows),
            "rows_using_labs": sum(row.get("fusion_uses_labs") is True for row in branch_rows),
            "warning_counts": dict(sorted(warning_counts.items())),
            "metrics": binary_metrics(labeled_rows) if labeled_rows else None,
            "calibrated_probability_diagnostics": None
            if not calibrated_rows
            else {
                "metrics": binary_metrics(calibrated_rows),
                "probability_summary": probability_summary(calibrated_rows),
                "calibration": calibration_bins(calibrated_rows, n_bins=10),
            },
        }
    return {
        "rows": len(rows),
        "unique_cases": len({row.get("id") for row in rows}),
        "rows_with_unusable_lab_sources": sum(row.get("unusable_lab_source") is True for row in rows),
        "unique_cases_with_unusable_lab_sources": len(
            {row.get("id") for row in rows if row.get("unusable_lab_source") is True}
        ),
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True, help="External case JSONL. Rows use the same schema as case JSON.")
    parser.add_argument("--data-root", type=Path, default=None, help="Base directory for relative image paths. Defaults to JSONL parent.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fusion-model",
        type=Path,
        default=REPO_ROOT / "artifacts/late_fusion_model.joblib",
    )
    parser.add_argument("--mode", choices=["auto", "full", "image_only", "both"], default="auto")
    parser.add_argument("--fusion-threshold-preset", default="selected")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model recorded in the fusion bundle.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--expected-image-count", type=int, default=4)
    parser.add_argument(
        "--min-full-lab-values",
        "--min-recognized-labs-for-full",
        dest="min_full_lab_values",
        type=int,
        default=5,
        help="Minimum recognized blood routine values recommended before trusting full-branch input quality.",
    )
    parser.add_argument(
        "--strict-full-labs",
        action="store_true",
        help="Fail instead of only warning when full-branch rows have fewer than --min-full-lab-values values.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.jsonl)
    if args.max_cases is not None:
        records = records[: args.max_cases]
    base_dir = args.data_root if args.data_root is not None else args.jsonl.parent
    record_base_dir = args.jsonl.parent
    fusion_bundle = joblib.load(args.fusion_model)
    embedding_model = args.embedding_model or fusion_bundle.get("embedding_model") or "mobilenet_v3_large"
    device = torch.device(args.device)
    extractor = ImageEmbeddingExtractor(embedding_model, device)

    predictions: list[dict[str, Any]] = []
    for record in records:
        image_paths = image_paths_from_record(record, base_dir)
        images = load_images(image_paths)
        image_embedding = extractor(images)
        labs, lab_source_summary = labs_from_record(record, record_base_dir)
        predictions.extend(
            evaluate_record(
                record=record,
                image_embedding=image_embedding,
                labs=labs,
                fusion_bundle=fusion_bundle,
                threshold_preset=args.fusion_threshold_preset,
                mode=args.mode,
                num_images=len(image_paths),
                expected_image_count=args.expected_image_count,
                min_full_lab_values=args.min_full_lab_values,
                strict_full_labs=args.strict_full_labs,
                lab_source_summary=lab_source_summary,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    write_jsonl(predictions_path, predictions)
    summary = {
        "jsonl": str(args.jsonl),
        "data_root": str(base_dir),
        "fusion_model": str(args.fusion_model),
        "embedding_model": embedding_model,
        "mode": args.mode,
        "fusion_threshold_preset": args.fusion_threshold_preset,
        "probability_fields": probability_field_metadata(
            raw_probability_field="raw_infectious_probability",
            calibrated_probability_field="calibrated_infectious_probability",
            threshold_field="fusion_threshold",
        ),
        "expected_image_count": args.expected_image_count,
        "min_full_lab_values": args.min_full_lab_values,
        "min_recognized_labs_for_full": args.min_full_lab_values,
        "strict_full_labs": args.strict_full_labs,
        "predictions": str(predictions_path),
        **summarize_predictions(predictions),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
