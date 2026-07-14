#!/usr/bin/env python3
"""End-to-end smoke test using generated images and synthetic case records."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib
import torch
warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.app import app
from late_fusion_inference import ImageEmbeddingExtractor, score_fusion
from uveitis_case import normalize_labs_mapping


MODEL_PATH = REPO_ROOT / "artifacts/late_fusion_model.joblib"
METADATA_PATH = REPO_ROOT / "artifacts/model_metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_images(directory: Path) -> list[Path]:
    paths: list[Path] = []
    colors = [(35, 70, 45), (72, 40, 38), (42, 58, 90), (84, 72, 32)]
    for index, color in enumerate(colors, start=1):
        image = Image.new("RGB", (384, 384), color)
        draw = ImageDraw.Draw(image)
        draw.ellipse((72, 72, 312, 312), outline=(180, 155, 110), width=10)
        draw.line((40, 192, 344, 192), fill=(120 + index * 8, 80, 75), width=5)
        path = directory / f"synthetic_{index}.jpg"
        image.save(path, quality=92)
        paths.append(path)
    return paths


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {"command": command, "returncode": completed.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model_sha256 = sha256_file(MODEL_PATH)
    if model_sha256 != metadata["artifact"]["sha256"]:
        raise RuntimeError("model SHA-256 does not match model_metadata.json")

    bundle = joblib.load(MODEL_PATH)
    if set(bundle) < {"positive_label", "negative_label", "embedding_model", "full", "image_only"}:
        raise RuntimeError("model bundle is missing required keys")
    if bundle["full"]["feature_groups"] != ["labs", "image_embedding"]:
        raise RuntimeError("full branch feature groups are invalid")
    if bundle["image_only"]["feature_groups"] != ["image_embedding"]:
        raise RuntimeError("image-only branch feature groups are invalid")

    device = torch.device(args.device)
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="uveitis_delivery_smoke_") as raw_temp_dir:
        temp_dir = Path(raw_temp_dir)
        image_paths = make_images(temp_dir)
        images = [Image.open(path).convert("RGB") for path in image_paths]
        extractor = ImageEmbeddingExtractor(bundle.get("embedding_model") or "mobilenet_v3_large", device)
        embedding = extractor(images)
        if len(embedding) != 1280:
            raise RuntimeError(f"unexpected embedding dimension: {len(embedding)}")

        labs = normalize_labs_mapping({"WBC": 5.34, "NEUT%": 73.3, "LYMPH#": 1.2, "HGB": 135, "PLT": 190})
        full = score_fusion(
            fusion_bundle=bundle,
            prompt_mode="full",
            labs=labs,
            score_losses=None,
            score_delta=None,
            image_score_losses=None,
            image_score_delta=None,
            image_embedding=embedding,
            threshold_preset="selected",
        )
        image_only = score_fusion(
            fusion_bundle=bundle,
            prompt_mode="image_only",
            labs={},
            score_losses=None,
            score_delta=None,
            image_score_losses=None,
            image_score_delta=None,
            image_embedding=embedding,
            threshold_preset="selected",
        )
        checks["core_inference"] = {
            "embedding_dim": len(embedding),
            "full_branch": full[4],
            "image_only_branch": image_only[4],
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            client = TestClient(app)
            root_response = client.get("/")
            showcase_response = client.get("/api/showcase")
            files = [("images", (path.name, path.read_bytes(), "image/jpeg")) for path in image_paths]
            predict_response = client.post(
                "/api/predict",
                files=files,
                data={
                    "lab_text": "WBC 5.34; NEUT% 73.3%; LYMPH# 1.20; HGB 135; PLT 190",
                    "threshold_preset": "selected",
                },
            )
        if root_response.status_code != 200 or showcase_response.status_code != 200 or predict_response.status_code != 200:
            raise RuntimeError(
                f"web smoke failed: root={root_response.status_code}, "
                f"showcase={showcase_response.status_code}, predict={predict_response.status_code}"
            )
        predict_payload = predict_response.json()
        if predict_payload.get("branch") != "full" or predict_payload.get("recognized_lab_count") != 5:
            raise RuntimeError(f"unexpected web prediction payload: {predict_payload}")
        checks["web_api"] = {
            "root_status": root_response.status_code,
            "showcase_status": showcase_response.status_code,
            "predict_status": predict_response.status_code,
            "branch": predict_payload["branch"],
            "recognized_lab_count": predict_payload["recognized_lab_count"],
            "num_images": predict_payload["num_images"],
        }

        cases_path = temp_dir / "cases.jsonl"
        rows = [
            {
                "id": "synthetic_full_case",
                "images": [path.name for path in image_paths],
                "labs": {"WBC": 5.34, "NEUT%": 73.3, "LYMPH#": 1.2, "HGB": 135, "PLT": 190},
                "label": "非感染性葡萄膜炎",
            },
            {
                "id": "synthetic_image_only_case",
                "images": [path.name for path in image_paths],
                "label": "感染性葡萄膜炎",
            },
        ]
        cases_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        preflight_path = temp_dir / "preflight.json"
        output_dir = temp_dir / "batch_output"
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "validate_external_cases_jsonl.py"),
                "--jsonl",
                str(cases_path),
                "--data-root",
                str(temp_dir),
                "--mode",
                "auto",
                "--output",
                str(preflight_path),
            ],
            cwd=REPO_ROOT,
        )
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("ok") is not True:
            raise RuntimeError(f"batch preflight failed: {preflight}")
        run_command(
            [
                sys.executable,
                str(REPO_ROOT / "evaluate_final_external_jsonl.py"),
                "--jsonl",
                str(cases_path),
                "--data-root",
                str(temp_dir),
                "--output-dir",
                str(output_dir),
                "--mode",
                "auto",
                "--device",
                args.device,
            ],
            cwd=REPO_ROOT,
        )
        prediction_rows = [
            json.loads(line)
            for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        branches = {row.get("branch") for row in prediction_rows}
        if branches != {"full", "image_only"}:
            raise RuntimeError(f"unexpected batch branches: {branches}")
        checks["batch_jsonl"] = {
            "preflight_ok": True,
            "prediction_rows": len(prediction_rows),
            "branches": sorted(branches),
        }

    result = {
        "ok": True,
        "model_version": metadata["version"],
        "model_sha256": model_sha256,
        "device": args.device,
        "checks": checks,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
