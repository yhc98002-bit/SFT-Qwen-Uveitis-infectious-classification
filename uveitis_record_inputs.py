"""Shared non-image record input helpers for uveitis cases."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from uveitis_case import labs_from_prompt, normalize_labs_mapping, parse_numeric_value
from uveitis_image_inputs import resolve_relative_path


LAB_OBJECT_FIELDS = ("labs", "blood_routine", "血常规指标")
LAB_FILE_FIELDS = (
    "labs_file",
    "labs_json_file",
    "blood_routine_file",
    "blood_routine_json_file",
    "血常规文件",
    "血常规JSON文件",
)
TEXT_FIELDS = ("prompt", "text", "clinical_text", "临床文本", "病历文本")
TEXT_FILE_FIELDS = (
    "prompt_file",
    "text_file",
    "clinical_text_file",
    "case_text_file",
    "临床文本文件",
    "病历文本文件",
)
IGNORED_TEXT_SIGNAL_PATTERNS = (
    ("降钙素原PCT", r"降钙素原|前降钙素|procalcitonin|pro-calcitonin"),
    ("C反应蛋白CRP", r"超敏C反应蛋白|C反应蛋白|hs[-\s]?CRP|\bCRP\b"),
    ("血沉ESR", r"红细胞沉降率|血沉|\bESR\b"),
    ("白介素6IL-6", r"白介素[-\s]?6|\bIL[-\s]?6\b"),
    ("结核IGRA/T-SPOT", r"T[-\s]?SPOT|TB[-\s]?IGRA|\bIGRA\b|结核感染T细胞"),
    ("梅毒血清学", r"梅毒|TPPA|TRUST|\bRPR\b"),
    ("HIV", r"\bHIV\b|人类免疫缺陷病毒"),
    ("CMV", r"\bCMV\b|巨细胞病毒"),
    ("HSV", r"\bHSV\b|单纯疱疹病毒"),
    ("VZV", r"\bVZV\b|水痘[-\s]?带状疱疹病毒|带状疱疹病毒"),
    ("弓形虫", r"弓形虫|toxoplasma"),
)


def first_record_value(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, Any] | tuple[None, None]:
    for field in fields:
        if field in record and record[field] is not None:
            return field, record[field]
    return None, None


def record_labs_mapping(record: dict[str, Any], base_dir: Path, *, record_label: str | None = None) -> dict[str, Any]:
    labs, _ = record_labs_mapping_with_source(record, base_dir, record_label=record_label)
    return labs


def record_labs_mapping_with_source(
    record: dict[str, Any],
    base_dir: Path,
    *,
    record_label: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    label = record_label or f"record {record.get('id')}"
    field, raw_labs = first_record_value(record, LAB_OBJECT_FIELDS)
    if field is not None:
        if not isinstance(raw_labs, dict):
            raise ValueError(f"{label} field {field!r} must be an object")
        return raw_labs, field

    file_field, raw_file = first_record_value(record, LAB_FILE_FIELDS)
    if file_field is None:
        return {}, None
    if not isinstance(raw_file, str):
        raise ValueError(f"{label} field {file_field!r} must be a file path string")
    labs_path = resolve_relative_path(raw_file, base_dir)
    data = json.loads(labs_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} field {file_field!r} must point to a JSON object")
    return data, file_field


def record_text(record: dict[str, Any], base_dir: Path, *, record_label: str | None = None) -> str | None:
    text, _ = record_text_with_source(record, base_dir, record_label=record_label)
    return text


def record_text_with_source(
    record: dict[str, Any],
    base_dir: Path,
    *,
    record_label: str | None = None,
) -> tuple[str | None, str | None]:
    label = record_label or f"record {record.get('id')}"
    field, raw_text = first_record_value(record, TEXT_FIELDS)
    if field is not None:
        if not isinstance(raw_text, str):
            raise ValueError(f"{label} field {field!r} must be a string")
        return raw_text, field

    file_field, raw_file = first_record_value(record, TEXT_FILE_FIELDS)
    if file_field is None:
        return None, None
    if not isinstance(raw_file, str):
        raise ValueError(f"{label} field {file_field!r} must be a file path string")
    text_path = resolve_relative_path(raw_file, base_dir)
    return text_path.read_text(encoding="utf-8"), file_field


def sorted_recognized_columns(labs: dict[str, Any]) -> list[str]:
    normalized = normalize_labs_mapping(labs)
    return sorted(column for column, value in normalized.items() if parse_numeric_value(value) is not None)


def unsupported_structured_lab_keys(values: dict[str, Any]) -> list[str]:
    unsupported: list[str] = []
    for key, value in values.items():
        if parse_numeric_value(value) is None:
            continue
        if normalize_labs_mapping({key: value}):
            continue
        unsupported.append(str(key))
    return sorted(unsupported)


def recognized_labs_mapping(values: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_labs_mapping(values)
    return {
        column: value
        for column, value in normalized.items()
        if parse_numeric_value(value) is not None
    }


def lab_value_conflicts(
    structured_labs: dict[str, Any],
    text_labs: dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for column in sorted(set(structured_labs) & set(text_labs)):
        structured_value = parse_numeric_value(structured_labs.get(column))
        text_value = parse_numeric_value(text_labs.get(column))
        if structured_value is None or text_value is None:
            continue
        if abs(structured_value - text_value) > tolerance:
            conflicts.append(
                {
                    "column": column,
                    "structured_value": structured_value,
                    "text_value": text_value,
                    "chosen_source": "structured",
                    "chosen_value": structured_value,
                }
            )
    return conflicts


def ignored_text_signal_terms(raw_text: str | None) -> list[str]:
    if not raw_text:
        return []
    terms: list[str] = []
    for label, pattern in IGNORED_TEXT_SIGNAL_PATTERNS:
        if re.search(pattern, raw_text, flags=re.IGNORECASE):
            terms.append(label)
    return terms


def labs_with_source_summary_from_values(
    *,
    structured_raw: dict[str, Any] | None = None,
    structured_source: str | None = None,
    raw_text: str | None = None,
    text_source: str | None = None,
) -> dict[str, Any]:
    structured_labs = recognized_labs_mapping(structured_raw or {})
    text_labs = labs_from_prompt(raw_text) if raw_text else {}
    merged_labs = {**text_labs, **structured_labs}
    structured_columns = sorted_recognized_columns(structured_labs)
    text_columns = sorted_recognized_columns(text_labs)
    merged_columns = sorted_recognized_columns(merged_labs)
    overridden_columns = sorted(set(structured_columns) & set(text_columns))
    conflicts = lab_value_conflicts(structured_labs, text_labs)
    unsupported_keys = unsupported_structured_lab_keys(structured_raw or {})
    ignored_terms = ignored_text_signal_terms(raw_text)
    return {
        "labs": merged_labs,
        "summary": {
            "structured_lab_source": structured_source,
            "text_lab_source": text_source,
            "structured_recognized_lab_count": len(structured_columns),
            "structured_recognized_lab_columns": structured_columns,
            "text_recognized_lab_count": len(text_columns),
            "text_recognized_lab_columns": text_columns,
            "merged_recognized_lab_count": len(merged_columns),
            "merged_recognized_lab_columns": merged_columns,
            "merged_from_multiple_sources": bool(structured_columns and text_columns),
            "structured_overrode_text_columns": overridden_columns,
            "conflict_count": len(conflicts),
            "conflicts": conflicts,
            "unsupported_structured_lab_keys": unsupported_keys,
            "unsupported_structured_lab_key_count": len(unsupported_keys),
            "ignored_text_signal_terms": ignored_terms,
            "ignored_text_signal_count": len(ignored_terms),
            "ignored_signal_note": (
                "Only supported blood-routine fields are used by this model; non-CBC infection or inflammation "
                "markers are reported here for transparency and are not used as model features."
            ),
        },
    }


def record_labs_with_source_summary(
    record: dict[str, Any],
    base_dir: Path,
    *,
    record_label: str | None = None,
) -> dict[str, Any]:
    structured_raw, structured_source = record_labs_mapping_with_source(record, base_dir, record_label=record_label)
    raw_text, text_source = record_text_with_source(record, base_dir, record_label=record_label)
    return labs_with_source_summary_from_values(
        structured_raw=structured_raw,
        structured_source=structured_source,
        raw_text=raw_text,
        text_source=text_source,
    )


def record_labs_from_all_sources(
    record: dict[str, Any],
    base_dir: Path,
    *,
    record_label: str | None = None,
) -> dict[str, Any]:
    """Return supported blood-routine values from structured labs plus text sidecars.

    Structured values take precedence for the same canonical lab column, while
    free-text values fill any additional supported columns.
    """

    return record_labs_with_source_summary(record, base_dir, record_label=record_label)["labs"]
