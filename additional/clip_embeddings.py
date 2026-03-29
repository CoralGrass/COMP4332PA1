"""Frozen CLIP token embedding loader and precompute module."""
from __future__ import annotations

import os

import numpy as np
import torch

from common import CACHE_DIR

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **_):
        return it


CLIP_MODEL = "openai/clip-vit-base-patch32"
SEQ_LEN = 64
CLIP_BATCH = 128
MAX_TOKEN_LEN = 77


@torch.no_grad()
def precompute_clip_tokens(
    texts: list[str],
    path_feats: str,
    path_lens: str,
    tokenizer,
    text_model,
    hidden_size: int,
    device,
):
    n = len(texts)
    feats = np.memmap(
        path_feats, dtype=np.float16, mode="w+", shape=(n, SEQ_LEN, hidden_size)
    )
    lens = np.memmap(path_lens, dtype=np.int16, mode="w+", shape=(n,))

    text_model.eval()
    for start in tqdm(range(0, n, CLIP_BATCH), desc="CLIP encode (tokens)"):
        end = min(n, start + CLIP_BATCH)
        batch = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = text_model(**batch)
        h = out.last_hidden_state[:, :SEQ_LEN, :].detach().cpu().numpy().astype(np.float16)
        _, t, _ = h.shape
        if t < SEQ_LEN:
            h = np.pad(h, ((0, 0), (0, SEQ_LEN - t), (0, 0)), mode="constant")
        mask = batch["attention_mask"][:, :SEQ_LEN].detach().cpu().numpy()
        actual = np.clip(mask.sum(axis=1).astype(np.int16), 1, SEQ_LEN)
        feats[start:end] = h
        lens[start:end] = actual
        feats.flush()
        lens.flush()


def prepare_clip_memmaps(train_texts, valid_texts, test_texts, device):
    from transformers import CLIPTextModel, CLIPTokenizerFast

    path_tr_f = str(CACHE_DIR / "clip_token_feats_train.mmap")
    path_tr_l = str(CACHE_DIR / "clip_token_lens_train.mmap")
    path_va_f = str(CACHE_DIR / "clip_token_feats_valid.mmap")
    path_va_l = str(CACHE_DIR / "clip_token_lens_valid.mmap")
    path_te_f = str(CACHE_DIR / "clip_token_feats_test.mmap")
    path_te_l = str(CACHE_DIR / "clip_token_lens_test.mmap")

    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL)
    text_model = CLIPTextModel.from_pretrained(CLIP_MODEL).to(device)
    for p in text_model.parameters():
        p.requires_grad = False
    hidden_size = text_model.config.hidden_size

    if not os.path.exists(path_tr_f):
        precompute_clip_tokens(
            train_texts, path_tr_f, path_tr_l, tokenizer, text_model, hidden_size, device
        )
    if not os.path.exists(path_va_f):
        precompute_clip_tokens(
            valid_texts, path_va_f, path_va_l, tokenizer, text_model, hidden_size, device
        )
    if not os.path.exists(path_te_f):
        precompute_clip_tokens(
            test_texts, path_te_f, path_te_l, tokenizer, text_model, hidden_size, device
        )

    return {
        "encoder_name": "clip",
        "label": "CLIP tokens (frozen)",
        "prefix": "clip",
        "seq_len": SEQ_LEN,
        "feature_dim": hidden_size,
        "train_feat_path": path_tr_f,
        "train_len_path": path_tr_l,
        "valid_feat_path": path_va_f,
        "valid_len_path": path_va_l,
        "test_feat_path": path_te_f,
        "test_len_path": path_te_l,
    }
