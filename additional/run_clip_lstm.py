"""Entry script: frozen CLIP token embeddings + BiLSTM."""
from __future__ import annotations

from clip_embeddings import prepare_clip_memmaps
from experiment import run_sequence_experiment


if __name__ == "__main__":
    run_sequence_experiment("clip", "lstm", prepare_clip_memmaps)
