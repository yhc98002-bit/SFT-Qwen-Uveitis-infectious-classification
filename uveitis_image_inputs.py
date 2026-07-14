"""Shared image input helpers for uveitis inference and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_LIST_FIELDS = ("images", "image", "image_paths")
IMAGE_DIR_FIELDS = ("image_dir", "image_directory", "image_folder", "图像目录", "图片目录")


def resolve_relative_path(raw_path: Any, base_dir: Path) -> Path:
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = base_dir / path
    return path


def image_paths_from_dir(path: Path, *, label: str = "image_dir") -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"missing image directory for {label}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {path}")
    image_paths = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda item: item.name.casefold(),
    )
    if not image_paths:
        raise ValueError(f"{label} contains no supported image files: {path}")
    return image_paths


def image_paths_from_record(
    record: dict[str, Any],
    base_dir: Path,
    *,
    record_label: str | None = None,
) -> list[Path]:
    label = record_label or f"record {record.get('id')}"
    image_paths: list[Path] = []

    for field in IMAGE_LIST_FIELDS:
        raw_images = record.get(field)
        if raw_images is None:
            continue
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        elif not isinstance(raw_images, list):
            raise ValueError(f"{label} field {field!r} must be a string or list")
        image_paths.extend(resolve_relative_path(raw_path, base_dir) for raw_path in raw_images)

    for field in IMAGE_DIR_FIELDS:
        raw_dirs = record.get(field)
        if raw_dirs is None:
            continue
        if isinstance(raw_dirs, str):
            raw_dirs = [raw_dirs]
        elif not isinstance(raw_dirs, list):
            raise ValueError(f"{label} field {field!r} must be a string or list")
        for raw_dir in raw_dirs:
            image_dir = resolve_relative_path(raw_dir, base_dir)
            image_paths.extend(image_paths_from_dir(image_dir, label=f"{label}.{field}"))

    if not image_paths:
        raise ValueError(f"{label} must contain images/image or image_dir")
    return image_paths
