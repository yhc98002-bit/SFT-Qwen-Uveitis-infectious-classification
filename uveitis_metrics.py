"""Runtime metrics and JSONL helpers used by batch evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


POSITIVE_LABEL = "感染性葡萄膜炎"
NEGATIVE_LABEL = "非感染性葡萄膜炎"
LABELS = (POSITIVE_LABEL, NEGATIVE_LABEL)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_to_int(label: str) -> int:
    return 1 if label == POSITIVE_LABEL else 0


def safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def probability_metrics(
    predictions: list[dict[str, Any]],
    *,
    probability_field: str = "infectious_probability",
    prefix: str = "",
) -> dict[str, Any]:
    labels: list[int] = []
    probabilities: list[float] = []
    invalid_probability_count = 0
    for item in predictions:
        target = item.get("target")
        if target not in LABELS:
            continue
        raw_probability = item.get(probability_field)
        if raw_probability is None:
            continue
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError):
            invalid_probability_count += 1
            continue
        if not np.isfinite(probability):
            invalid_probability_count += 1
            continue
        labels.append(label_to_int(str(target)))
        probabilities.append(probability)

    result: dict[str, Any] = {
        f"{prefix}probability_count": len(probabilities),
        f"{prefix}invalid_probability_count": invalid_probability_count,
        f"{prefix}positive_prevalence": (sum(labels) / len(labels)) if labels else None,
        f"{prefix}roc_auc": None,
        f"{prefix}average_precision": None,
        f"{prefix}brier_score": None,
    }
    if not probabilities:
        return result

    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(probabilities, dtype=float)
    result[f"{prefix}brier_score"] = float(brier_score_loss(y_true, y_score))
    if len(set(labels)) >= 2:
        result[f"{prefix}roc_auc"] = float(roc_auc_score(y_true, y_score))
        result[f"{prefix}average_precision"] = float(average_precision_score(y_true, y_score))
    return result


def binary_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = invalid = 0
    for item in predictions:
        target = item["target"]
        prediction = item.get("prediction")
        invalid += int(prediction is None)
        if target == POSITIVE_LABEL:
            if prediction == POSITIVE_LABEL:
                tp += 1
            else:
                fn += 1
        elif prediction == NEGATIVE_LABEL:
            tn += 1
        else:
            fp += 1

    total = len(predictions)
    correct = sum(1 for item in predictions if item.get("prediction") == item.get("target"))
    infectious_precision = safe_div(tp, tp + fp)
    infectious_recall = safe_div(tp, tp + fn)
    noninfectious_precision = safe_div(tn, tn + fn)
    noninfectious_recall = safe_div(tn, tn + fp)
    infectious_f1 = f1_score(infectious_precision, infectious_recall)
    noninfectious_f1 = f1_score(noninfectious_precision, noninfectious_recall)
    recalls = [value for value in (infectious_recall, noninfectious_recall) if value is not None]
    f1_values = [value for value in (infectious_f1, noninfectious_f1) if value is not None]
    metrics = {
        "checked": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "invalid_predictions": invalid,
        "confusion_matrix": {
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "infectious_precision": infectious_precision,
        "infectious_recall": infectious_recall,
        "noninfectious_precision": noninfectious_precision,
        "noninfectious_recall": noninfectious_recall,
        "specificity": noninfectious_recall,
        "infectious_f1": infectious_f1,
        "noninfectious_f1": noninfectious_f1,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        **probability_metrics(predictions),
    }
    if any("calibrated_infectious_probability" in item for item in predictions):
        metrics.update(
            probability_metrics(
                predictions,
                probability_field="calibrated_infectious_probability",
                prefix="calibrated_",
            )
        )
    return metrics


def usable_rows(
    rows: list[dict[str, Any]],
    probability_field: str = "infectious_probability",
) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for row in rows:
        target = row.get("target")
        if target not in LABELS:
            continue
        try:
            probability = float(row.get(probability_field))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(probability):
            continue
        usable.append(
            {
                "id": row.get("id"),
                "target": str(target),
                "prediction": row.get("prediction"),
                "infectious_probability": probability,
            }
        )
    return usable


def probability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = np.asarray([float(row["infectious_probability"]) for row in rows], dtype=float)
    labels = np.asarray([label_to_int(str(row["target"])) for row in rows], dtype=int)
    if probabilities.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean_positive_probability": None,
            "mean_negative_probability": None,
        }
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    quantiles = np.quantile(probabilities, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(probabilities.size),
        "min": float(probabilities.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(probabilities.mean()),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(probabilities.max()),
        "mean_positive_probability": float(positives.mean()) if positives.size else None,
        "mean_negative_probability": float(negatives.mean()) if negatives.size else None,
    }


def calibration_bins(rows: list[dict[str, Any]], n_bins: int) -> dict[str, Any]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    probabilities = [float(row["infectious_probability"]) for row in rows]
    labels = [label_to_int(str(row["target"])) for row in rows]
    total = len(probabilities)
    expected_error = 0.0
    maximum_error = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        members = [
            row_index
            for row_index, probability in enumerate(probabilities)
            if low <= probability < high or (index == n_bins - 1 and low <= probability <= high)
        ]
        if members:
            mean_probability = float(np.mean([probabilities[item] for item in members]))
            observed_rate = float(np.mean([labels[item] for item in members]))
            absolute_error = abs(mean_probability - observed_rate)
            expected_error += (len(members) / total) * absolute_error
            maximum_error = max(maximum_error, absolute_error)
        else:
            mean_probability = observed_rate = absolute_error = None
        bins.append(
            {
                "low": low,
                "high": high,
                "count": len(members),
                "positives": int(sum(labels[item] for item in members)),
                "mean_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_error": absolute_error,
            }
        )
    return {
        "n_bins": n_bins,
        "expected_calibration_error": float(expected_error) if total else None,
        "maximum_calibration_error": float(maximum_error) if total else None,
        "bins": bins,
    }
