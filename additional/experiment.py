"""Formal training pipeline for frozen CLIP/BERT token embeddings + GRU/LSTM."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    OUTPUT_DIR,
    compute_metrics,
    ensure_dirs,
    load_splits,
    save_metrics_txt,
    save_predictions,
)
from data import (
    MemmapSequenceDataset,
    MemmapSequenceTestDataset,
    collate_infer,
    collate_train,
)
from models import FrozenEmbeddingBiGRU, FrozenEmbeddingBiLSTM
from trainer import evaluate_macro_f1, predict_ids, train_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 7
EPOCHS = 20
LR = 1e-3
DROPOUT = 0.3
TRAIN_BATCH = 64
EVAL_BATCH = 256
SEED = 42


def _open_memmaps(spec: dict, n_tr: int, n_va: int, n_te: int):
    seq_len = spec["seq_len"]
    feat_dim = spec["feature_dim"]
    mm_tr_f = np.memmap(
        spec["train_feat_path"], dtype=np.float16, mode="r", shape=(n_tr, seq_len, feat_dim)
    )
    mm_tr_l = np.memmap(spec["train_len_path"], dtype=np.int16, mode="r", shape=(n_tr,))
    mm_va_f = np.memmap(
        spec["valid_feat_path"], dtype=np.float16, mode="r", shape=(n_va, seq_len, feat_dim)
    )
    mm_va_l = np.memmap(spec["valid_len_path"], dtype=np.int16, mode="r", shape=(n_va,))
    mm_te_f = np.memmap(
        spec["test_feat_path"], dtype=np.float16, mode="r", shape=(n_te, seq_len, feat_dim)
    )
    mm_te_l = np.memmap(spec["test_len_path"], dtype=np.int16, mode="r", shape=(n_te,))
    return mm_tr_f, mm_tr_l, mm_va_f, mm_va_l, mm_te_f, mm_te_l


def _build_model(model_name: str, input_dim: int):
    if model_name == "gru":
        return FrozenEmbeddingBiGRU(
            input_dim, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT
        ).to(DEVICE)
    if model_name == "lstm":
        return FrozenEmbeddingBiLSTM(
            input_dim, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT
        ).to(DEVICE)
    raise ValueError(f"Unsupported model_name: {model_name}")


def run_sequence_experiment(embedding_name: str, model_name: str, prepare_features_fn):
    ensure_dirs()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_df, valid_df, test_df = load_splits()
    train_texts = train_df["text"].astype(str).tolist()
    valid_texts = valid_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    spec = prepare_features_fn(train_texts, valid_texts, test_texts, DEVICE)
    n_tr, n_va, n_te = len(train_df), len(valid_df), len(test_df)
    mmaps = _open_memmaps(spec, n_tr, n_va, n_te)
    mm_tr_f, mm_tr_l, mm_va_f, mm_va_l, mm_te_f, mm_te_l = mmaps

    train_ds = MemmapSequenceDataset(
        mm_tr_f, mm_tr_l, train_df["label"].values, spec["seq_len"]
    )
    valid_ds = MemmapSequenceDataset(
        mm_va_f, mm_va_l, valid_df["label"].values, spec["seq_len"]
    )
    test_ds = MemmapSequenceTestDataset(mm_te_f, mm_te_l, spec["seq_len"])

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_train,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_train,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_infer,
    )

    model = _build_model(model_name, spec["feature_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    experiment_prefix = f"{embedding_name}_{model_name}"
    experiment_name = f"{spec['label']} + Bi{model_name.upper()}"

    print(f"Device: {DEVICE} | experiment={experiment_prefix}")
    print(
        f"Feature dim: {spec['feature_dim']} | seq_len: {spec['seq_len']} | "
        f"hidden_dim: {HIDDEN_DIM}"
    )

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_f1 = evaluate_macro_f1(model, valid_loader, DEVICE)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={loss:.4f} | val Macro-F1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest Validation Macro-F1: {best_f1:.4f}")
    model.load_state_dict(best_state)

    valid_preds = predict_ids(model, valid_loader, DEVICE)
    test_preds = predict_ids(model, test_loader, DEVICE)

    save_predictions(
        valid_df["id"],
        valid_preds,
        test_df["id"],
        test_preds,
        experiment_prefix,
    )

    metrics = compute_metrics(valid_df["label"].astype(int).values, valid_preds)
    save_metrics_txt(OUTPUT_DIR / f"{experiment_prefix}_metrics.txt", experiment_name, metrics)
    print("Saved", OUTPUT_DIR / f"{experiment_prefix}_valid_pred.csv")
    print("Saved", OUTPUT_DIR / f"{experiment_prefix}_test_pred.csv")
    print("Saved", OUTPUT_DIR / f"{experiment_prefix}_metrics.txt")
