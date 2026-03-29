"""Shared paths, data loading, and metric helpers for additional experiments."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ADDITIONAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = ADDITIONAL_DIR / "cache"
OUTPUT_DIR = ADDITIONAL_DIR / "outputs"

LABEL_NAMES = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    valid_df = pd.read_csv(DATA_DIR / "valid.csv")
    test_df = pd.read_csv(DATA_DIR / "test_no_label.csv")
    return train_df, valid_df, test_df


def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_p": p,
        "macro_r": r,
        "macro_f1": macro_f1,
        "per_class_report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
        ),
    }


def save_predictions(valid_ids, valid_preds, test_ids, test_preds, prefix: str) -> None:
    pd.DataFrame({"id": valid_ids, "label": valid_preds}).to_csv(
        OUTPUT_DIR / f"{prefix}_valid_pred.csv", index=False
    )
    pd.DataFrame({"id": test_ids, "label": test_preds}).to_csv(
        OUTPUT_DIR / f"{prefix}_test_pred.csv", index=False
    )


def save_metrics_txt(path: Path, experiment_name: str, metrics: dict) -> None:
    lines = [
        f"Experiment: {experiment_name}",
        f"  Accuracy : {metrics['accuracy']:.4f}",
        f"  Macro-P  : {metrics['macro_p']:.4f}",
        f"  Macro-R  : {metrics['macro_r']:.4f}",
        f"  Macro-F1 : {metrics['macro_f1']:.4f}",
        "",
        "Per-class F1:",
        metrics["per_class_report"],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
