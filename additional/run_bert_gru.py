"""Entry script: frozen BERT token embeddings + BiGRU."""
from __future__ import annotations

from bert_embeddings import prepare_bert_memmaps
from experiment import run_sequence_experiment


if __name__ == "__main__":
    run_sequence_experiment("bert", "gru", prepare_bert_memmaps)
