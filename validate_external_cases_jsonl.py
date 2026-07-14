#!/usr/bin/env python3
"""Preflight-check external case JSONL files before final-model evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from uveitis_case import CLINICAL_COLUMNS, lab_feature_vector, labs_from_prompt, normalize_labs_mapping
from uveitis_image_inputs import image_paths_from_record, resolve_relative_path
from uveitis_metrics import LABELS, NEGATIVE_LABEL, POSITIVE_LABEL
from uveitis_record_inputs import TEXT_FIELDS, TEXT_FILE_FIELDS, record_labs_with_source_summary


REPO_ROOT = Path(__file__).resolve().parent
LEAKAGE_KEYWORDS = [
    "入院诊断",
    "出院诊断",
    "诊断：",
    "诊断:",
    "确诊",
    "最终诊断",
    "病因诊断",
]
LABEL_LEAKAGE_TERMS = [
    *LABELS,
    "感染类葡萄膜炎",
    "非感染类葡萄膜炎",
    "infectious uveitis",
    "noninfectious uveitis",
    "non-infectious uveitis",
]
EXTRA_TEXT_FIELDS_FOR_LEAKAGE = ("chief_complaint", "主诉")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, f"line {lineno} is not valid JSON: {exc}")
                continue
            if not isinstance(row, dict):
                fail(errors, f"line {lineno} must be a JSON object")
                continue
            row["_line_number"] = lineno
            rows.append(row)
    return rows


def check_image_open(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # noqa: BLE001 - report corrupt or unsupported image details.
        return False, str(exc)


def lab_input_summary(labs: dict[str, Any]) -> dict[str, Any]:
    normalized_labs = normalize_labs_mapping(labs)
    lab_values = lab_feature_vector(normalized_labs)
    recognized_columns = [
        column
        for column, value in zip(CLINICAL_COLUMNS, lab_values)
        if value is not None
    ]
    return {
        "has_labs": bool(recognized_columns),
        "recognized_lab_count": len(recognized_columns),
        "recognized_lab_columns": recognized_columns,
        "missing_lab_count": len(CLINICAL_COLUMNS) - len(recognized_columns),
    }


def text_fragments_for_leakage(
    record: dict[str, Any],
    record_base_dir: Path,
    case_id: str,
    errors: list[str],
) -> list[dict[str, str]]:
    fragments: list[dict[str, str]] = []
    for field in (*TEXT_FIELDS, *EXTRA_TEXT_FIELDS_FOR_LEAKAGE):
        if field not in record or record[field] is None:
            continue
        value = record[field]
        if not isinstance(value, str):
            fail(errors, f"{case_id} field {field!r} must be a string")
            continue
        fragments.append({"source": field, "text": value})
    for field in TEXT_FILE_FIELDS:
        if field not in record or record[field] is None:
            continue
        value = record[field]
        if not isinstance(value, str):
            fail(errors, f"{case_id} field {field!r} must be a file path string")
            continue
        try:
            text_path = resolve_relative_path(value, record_base_dir)
            fragments.append({"source": field, "text": text_path.read_text(encoding="utf-8")})
        except OSError as exc:
            fail(errors, f"{case_id} cannot read text sidecar {field!r}: {exc}")
    return fragments


def text_leakage_findings(
    record: dict[str, Any],
    record_base_dir: Path,
    case_id: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for fragment in text_fragments_for_leakage(record, record_base_dir, case_id, errors):
        text = fragment["text"]
        label_terms = [term for term in LABEL_LEAKAGE_TERMS if term.lower() in text.lower()]
        keywords = [keyword for keyword in LEAKAGE_KEYWORDS if keyword in text]
        if label_terms or keywords:
            findings.append(
                {
                    "source": fragment["source"],
                    "label_terms": label_terms,
                    "diagnosis_keywords": keywords,
                }
            )
    return findings


def labs_from_record(record: dict[str, Any], record_base_dir: Path, errors: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    case_id = record_id(record)
    try:
        lab_source_result = record_labs_with_source_summary(record, record_base_dir, record_label=case_id)
        return lab_source_result["labs"], lab_source_result["summary"]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        fail(errors, str(exc))
        return {}, None


def record_id(record: dict[str, Any]) -> str:
    raw_id = record.get("id")
    if raw_id is None or str(raw_id).strip() == "":
        return f"line_{record.get('_line_number')}"
    return str(raw_id)


def target_from_record(record: dict[str, Any]) -> str | None:
    target = record.get("label") or record.get("answer") or record.get("target")
    if target is None:
        return None
    return str(target)


def planned_branch(mode: str, has_labs: bool) -> str:
    if mode == "auto":
        return "full" if has_labs else "image_only"
    return mode


def lab_source_fields_present(lab_source_summary: dict[str, Any] | None) -> bool:
    if not lab_source_summary:
        return False
    return bool(lab_source_summary.get("structured_lab_source") or lab_source_summary.get("text_lab_source"))


def has_unusable_lab_source(lab_source_summary: dict[str, Any] | None) -> bool:
    if not lab_source_fields_present(lab_source_summary):
        return False
    return int(lab_source_summary.get("merged_recognized_lab_count") or 0) <= 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="Base directory for relative image paths. Defaults to JSONL parent.")
    parser.add_argument("--mode", choices=["auto", "full", "image_only", "both"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--min-labeled-cases", type=int, default=50)
    parser.add_argument("--min-positive-cases", type=int, default=10)
    parser.add_argument("--min-negative-cases", type=int, default=30)
    parser.add_argument("--expected-image-count", type=int, default=4)
    parser.add_argument("--max-image-count-warning-fraction", type=float, default=0.50)
    parser.add_argument(
        "--min-recognized-labs-for-full",
        type=int,
        default=5,
        help="Minimum recognized blood routine values required before a case can run the full branch.",
    )
    parser.add_argument("--max-full-missing-lab-fraction", type=float, default=0.50)
    parser.add_argument("--skip-image-file-check", action="store_true")
    parser.add_argument("--skip-image-open-check", action="store_true")
    parser.add_argument(
        "--allow-text-leakage",
        action="store_true",
        help="Warn instead of failing when prompt/text/chief_complaint contains label or diagnosis leakage terms.",
    )
    parser.add_argument(
        "--allow-unusable-lab-sources",
        action="store_true",
        help=(
            "Warn instead of failing when text/lab fields are present but contain no supported blood routine values. "
            "Use only for debugging or deliberate image-only fallback cohorts."
        ),
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    jsonl_path = resolve_path(args.jsonl)
    if not jsonl_path.exists():
        fail(errors, f"missing JSONL file: {jsonl_path}")
        report = {"ok": False, "jsonl": str(jsonl_path), "errors": errors, "warnings": warnings}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    data_root = args.data_root if args.data_root is not None else jsonl_path.parent
    data_root = resolve_path(data_root)
    record_base_dir = jsonl_path.parent
    rows = read_jsonl(jsonl_path, errors)
    original_rows = len(rows)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    ids = [record_id(row) for row in rows]
    duplicate_ids = sorted([case_id for case_id, count in Counter(ids).items() if count > 1])
    if duplicate_ids:
        fail(errors, f"duplicate case ids: {duplicate_ids[:20]}")

    label_counts: Counter[str] = Counter()
    image_count_distribution: Counter[int] = Counter()
    lab_count_distribution: Counter[int] = Counter()
    branch_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    missing_images: list[dict[str, Any]] = []
    unreadable_images: list[dict[str, Any]] = []
    text_leakage_rows: list[dict[str, Any]] = []
    unusable_lab_source_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []

    for row in rows:
        case_id = record_id(row)
        target = target_from_record(row)
        if target is None:
            if args.require_labels:
                fail(errors, f"{case_id} is missing label/answer/target")
        elif target not in LABELS:
            fail(errors, f"{case_id} has unsupported label: {target!r}")
        else:
            label_counts[target] += 1

        leakage_findings = text_leakage_findings(row, record_base_dir, case_id, errors)
        if leakage_findings:
            warning_counts["text_label_or_diagnosis_leakage"] += 1
            leakage_row = {"id": case_id, "findings": leakage_findings}
            text_leakage_rows.append(leakage_row)
            if not args.allow_text_leakage:
                fail(errors, f"{case_id} text fields contain label/diagnosis leakage terms: {leakage_findings}")

        try:
            image_paths = image_paths_from_record(row, data_root, record_label=case_id)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            fail(errors, str(exc))
            image_paths = []
        image_count_distribution[len(image_paths)] += 1
        if not image_paths:
            fail(errors, f"{case_id} has no usable images/image or image_dir field")
        if len(image_paths) != args.expected_image_count:
            warning_counts["image_count_differs_from_training_default"] += 1
        for image_path in image_paths:
            if args.skip_image_file_check:
                continue
            if not image_path.exists():
                missing_images.append({"id": case_id, "image": str(image_path)})
                continue
            if not args.skip_image_open_check:
                ok, detail = check_image_open(image_path)
                if not ok:
                    unreadable_images.append({"id": case_id, "image": str(image_path), "error": detail})

        labs, lab_source_summary = labs_from_record(row, record_base_dir, errors)
        lab_summary = lab_input_summary(labs)
        recognized_lab_count = int(lab_summary["recognized_lab_count"])
        lab_count_distribution[recognized_lab_count] += 1
        if lab_source_summary:
            if int(lab_source_summary.get("conflict_count") or 0) > 0:
                warning_counts["conflicting_lab_values_across_sources"] += 1
            merged_count = int(lab_source_summary.get("merged_recognized_lab_count") or 0)
            structured_count = int(lab_source_summary.get("structured_recognized_lab_count") or 0)
            text_count = int(lab_source_summary.get("text_recognized_lab_count") or 0)
            if lab_source_summary.get("merged_from_multiple_sources") is True and merged_count > max(structured_count, text_count):
                warning_counts["merged_lab_values_from_multiple_sources"] += 1
        unusable_lab_source = has_unusable_lab_source(lab_source_summary)
        if unusable_lab_source:
            warning_counts["unusable_lab_source_no_supported_blood_routine"] += 1
            unusable_row = {
                "id": case_id,
                "line_number": row.get("_line_number"),
                "structured_lab_source": (lab_source_summary or {}).get("structured_lab_source"),
                "text_lab_source": (lab_source_summary or {}).get("text_lab_source"),
                "ignored_text_signal_terms": (lab_source_summary or {}).get("ignored_text_signal_terms") or [],
                "unsupported_structured_lab_keys": (lab_source_summary or {}).get("unsupported_structured_lab_keys") or [],
            }
            unusable_lab_source_rows.append(unusable_row)
            if not args.allow_unusable_lab_sources:
                fail(
                    errors,
                    (
                        f"{case_id} has text/lab source fields but no supported blood routine values were recognized; "
                        "remove those fields for a true image-only case, add supported CBC/blood-routine values, "
                        "or pass --allow-unusable-lab-sources for debugging or deliberate image-only fallback."
                    ),
                )
        branch = planned_branch(args.mode, bool(lab_summary["has_labs"]))
        if args.mode == "both":
            branch_counts["full"] += 1
            branch_counts["image_only"] += 1
        else:
            branch_counts[branch] += 1
        if args.mode in {"full", "both"} and recognized_lab_count < args.min_recognized_labs_for_full:
            fail(
                errors,
                (
                    f"{case_id} cannot run mode={args.mode}: recognized_lab_count="
                    f"{recognized_lab_count} below required {args.min_recognized_labs_for_full}"
                ),
            )
        if branch == "full" and recognized_lab_count < args.min_recognized_labs_for_full:
            fail(
                errors,
                (
                    f"{case_id} planned full branch but recognized_lab_count="
                    f"{recognized_lab_count} below required {args.min_recognized_labs_for_full}"
                ),
            )
        full_branch_will_run = args.mode == "both" or branch == "full"
        if full_branch_will_run and lab_summary["missing_lab_count"]:
            warning_counts["missing_lab_values_will_be_imputed"] += 1
        row_summaries.append(
            {
                "id": case_id,
                "line_number": row.get("_line_number"),
                "image_count": len(image_paths),
                "label": target,
                "recognized_lab_count": recognized_lab_count,
                "recognized_lab_columns": lab_summary["recognized_lab_columns"],
                "lab_source_summary": lab_source_summary,
                "unusable_lab_source": unusable_lab_source,
                "text_leakage_findings": leakage_findings,
                "planned_branch": branch,
            }
        )

    if missing_images:
        fail(errors, f"missing image files: {missing_images[:20]}")
    if unreadable_images:
        fail(errors, f"unreadable image files: {unreadable_images[:20]}")

    labeled_count = sum(label_counts.values())
    if args.require_labels:
        if labeled_count < args.min_labeled_cases:
            fail(errors, f"labeled_cases={labeled_count} below required {args.min_labeled_cases}")
        if label_counts[POSITIVE_LABEL] < args.min_positive_cases:
            fail(errors, f"positive_cases={label_counts[POSITIVE_LABEL]} below required {args.min_positive_cases}")
        if label_counts[NEGATIVE_LABEL] < args.min_negative_cases:
            fail(errors, f"negative_cases={label_counts[NEGATIVE_LABEL]} below required {args.min_negative_cases}")
    if rows:
        image_count_warning_fraction = warning_counts["image_count_differs_from_training_default"] / len(rows)
        full_rows = branch_counts["full"]
        full_missing_lab_fraction = (
            warning_counts["missing_lab_values_will_be_imputed"] / full_rows
            if full_rows
            else None
        )
        if image_count_warning_fraction > args.max_image_count_warning_fraction:
            warn(
                warnings,
                (
                    f"image_count_warning_fraction={image_count_warning_fraction:.3f} "
                    f"above suggested {args.max_image_count_warning_fraction:.3f}"
                ),
            )
        if full_missing_lab_fraction is not None and full_missing_lab_fraction > args.max_full_missing_lab_fraction:
            warn(
                warnings,
                (
                    f"full_missing_lab_fraction={full_missing_lab_fraction:.3f} "
                    f"above suggested {args.max_full_missing_lab_fraction:.3f}"
                ),
            )
        if unusable_lab_source_rows and args.allow_unusable_lab_sources:
            warn(
                warnings,
                (
                    f"unusable_lab_source_count={len(unusable_lab_source_rows)}; rows contain text/lab fields but "
                    "no supported blood routine values, so they cannot be interpreted as multimodal lab input."
                ),
            )
    else:
        image_count_warning_fraction = None
        full_missing_lab_fraction = None

    report = {
        "ok": not errors,
        "jsonl": str(jsonl_path),
        "data_root": str(data_root),
        "mode": args.mode,
        "criteria": {
            "max_cases": args.max_cases,
            "require_labels": args.require_labels,
            "min_labeled_cases": args.min_labeled_cases,
            "min_positive_cases": args.min_positive_cases,
            "min_negative_cases": args.min_negative_cases,
            "expected_image_count": args.expected_image_count,
            "max_image_count_warning_fraction": args.max_image_count_warning_fraction,
            "min_recognized_labs_for_full": args.min_recognized_labs_for_full,
            "max_full_missing_lab_fraction": args.max_full_missing_lab_fraction,
            "image_file_check": not args.skip_image_file_check,
            "image_open_check": not args.skip_image_open_check,
            "allow_text_leakage": args.allow_text_leakage,
            "allow_unusable_lab_sources": args.allow_unusable_lab_sources,
        },
        "rows": len(rows),
        "original_rows": original_rows,
        "unique_ids": len(set(ids)),
        "label_counts": dict(sorted(label_counts.items())),
        "unlabeled_rows": len(rows) - labeled_count,
        "image_count_distribution": dict(sorted(image_count_distribution.items())),
        "lab_count_distribution": dict(sorted(lab_count_distribution.items())),
        "branch_counts": dict(sorted(branch_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "image_count_warning_fraction": image_count_warning_fraction,
        "full_missing_lab_fraction": full_missing_lab_fraction,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "unreadable_image_count": len(unreadable_images),
        "unreadable_image_examples": unreadable_images[:20],
        "text_leakage_count": len(text_leakage_rows),
        "text_leakage_examples": text_leakage_rows[:20],
        "unusable_lab_source_count": len(unusable_lab_source_rows),
        "unusable_lab_source_examples": unusable_lab_source_rows[:20],
        "row_examples": row_summaries[:20],
        "warnings": warnings,
        "errors": errors,
    }
    if args.output is not None:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
