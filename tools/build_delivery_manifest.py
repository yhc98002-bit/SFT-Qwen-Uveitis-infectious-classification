#!/usr/bin/env python3
"""Build a SHA-256 manifest for the public engineering delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/delivery_manifest.json"
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "test-output", "external-output"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path, output: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    entries = []
    for path in included_files(root, output):
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    report = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "file_count": len(entries), "total_bytes": report["total_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

