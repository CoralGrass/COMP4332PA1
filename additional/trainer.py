"""Training and inference helpers for sequence classifiers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_macro_f1(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths = x.to(device), lengths.to(device)
            preds = model(x, lengths).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return float(f1_score(all_labels, all_preds, average="macro"))


@torch.no_grad()
def predict_ids(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        if len(batch) == 3:
            x, lengths, _ = batch
        else:
            x, lengths = batch
        x, lengths = x.to(device), lengths.to(device)
        preds = model(x, lengths).argmax(dim=1).cpu().numpy()
        out.extend(preds.tolist())
    return np.array(out, dtype=np.int64)
