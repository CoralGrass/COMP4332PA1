from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


LABEL_NAMES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MODEL_FILES = {
    "MLP": "mlp_valid_pred.csv",
    "RNN": "rnn_valid_pred.csv",
    "GRU": "gru_valid_pred.csv",
    "LSTM": "lstm_valid_pred.csv",
}


def load_eval_split(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "valid.csv", usecols=["id", "label"])


def load_predictions(root: Path, pred_name: str) -> pd.DataFrame:
    return pd.read_csv(root / pred_name, usecols=["id", "label"])


def aligned_labels(root: Path, pred_name: str) -> tuple[np.ndarray, np.ndarray]:
    gold = load_eval_split(root)
    pred = load_predictions(root, pred_name)
    merged = gold.merge(pred, on="id", how="left", suffixes=("_true", "_pred"))
    merged["label_pred"] = merged["label_pred"].fillna(-1).astype(int)
    return merged["label_true"].to_numpy(), merged["label_pred"].to_numpy()


def collect_f1_scores(root: Path) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for model_name, pred_name in MODEL_FILES.items():
        y_true, y_pred = aligned_labels(root, pred_name)
        per_class = f1_score(
            y_true,
            y_pred,
            labels=list(range(len(LABEL_NAMES))),
            average=None,
            zero_division=0,
        )
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[model_name] = np.concatenate([per_class, [macro]])
    return results


def plot_grouped_f1(root: Path, out_dir: Path) -> None:
    scores = collect_f1_scores(root)
    group_labels = LABEL_NAMES + ["Macro-F1"]
    x = np.arange(len(group_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5.8))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    for model_name, values, offset, color in zip(scores.keys(), scores.values(), offsets, colors):
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            label=model_name,
            color=color,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.3f}" for v in values],
            padding=2,
            fontsize=7,
            rotation=90,
        )

    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_title("Per-class F1 and Macro-F1 on the validation set")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()

    fig.savefig(out_dir / "baseline_f1_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "baseline_f1_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gru_confusion(root: Path, out_dir: Path) -> None:
    y_true, y_pred = aligned_labels(root, MODEL_FILES["GRU"])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized ratio")

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("GRU baseline confusion matrix on the validation set")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "gru_confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "gru_confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    root = out_dir.parent
    plot_grouped_f1(root, out_dir)
    plot_gru_confusion(root, out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
