#!/usr/bin/env python3
"""Verify delivery hashes and reject common patient-data artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/delivery_manifest.json"
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".js", ".css", ".html", ".gitignore"}
FORBIDDEN_NAMES = {
    "all.jsonl",
    "train.jsonl",
    "val.jsonl",
    "oof_predictions_full.jsonl",
    "oof_predictions_image_only.jsonl",
}
FORBIDDEN_PATH_TERMS = (
    "image_embeddings",
    "qwen25vl",
    "fitted_predictions",
    "data_quality_report",
)
FORBIDDEN_TEXT_PATTERNS = {
    "patient_identifier": re.compile(r"(?<![0-9A-Za-z])[1-9]\d{17,18}(?![0-9A-Za-z])"),
    "visit_identifier": re.compile(r"\bZY\d{6,}\b", flags=re.IGNORECASE),
    "patient_field": re.compile(r"patient_id|visit_id|患者ID|就诊号", flags=re.IGNORECASE),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def privacy_findings(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        lower_relative = relative.lower()
        if path.name.lower() in FORBIDDEN_NAMES or any(term in lower_relative for term in FORBIDDEN_PATH_TERMS):
            findings.append({"path": relative, "reason": "forbidden_delivery_artifact"})
            continue
        if relative == "tools/verify_delivery.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in FORBIDDEN_TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append({"path": relative, "reason": name})
    model_path = root / "artifacts/late_fusion_model.joblib"
    if model_path.exists():
        raw = model_path.read_bytes()
        for token in (b"patient_id", b"visit_id", b"ZY202"):
            if token in raw:
                findings.append({"path": model_path.relative_to(root).as_posix(), "reason": f"model_contains_{token.decode()}"})
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for entry in manifest.get("files", []):
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != int(entry["size_bytes"]):
            errors.append(f"size mismatch: {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")

    expected_paths = {entry["path"] for entry in manifest.get("files", [])}
    actual_paths = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative == manifest_path.relative_to(root).as_posix():
            continue
        if any(part in {".venv", ".pytest_cache", "test-output", "external-output"} for part in path.relative_to(root).parts):
            continue
        actual_paths.add(relative)
    for extra in sorted(actual_paths - expected_paths):
        errors.append(f"untracked by manifest: {extra}")

    privacy = privacy_findings(root)
    if privacy:
        errors.extend(f"privacy: {item['path']} ({item['reason']})" for item in privacy)
    result = {
        "ok": not errors,
        "manifest": str(manifest_path),
        "checked_files": len(expected_paths),
        "privacy_findings": privacy,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
