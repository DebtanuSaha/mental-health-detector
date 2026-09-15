"""
token_length_analysis.py

Tokenizes the real corpus with the actual model tokenizer and reports
real BPE token-length percentiles — replacing the word-count-based
estimate used earlier to pick a starting `max_seq_length`.

RoBERTa-family models cap out at 512 tokens architecturally, so this
tells you what fraction of real posts actually exceed that ceiling
(i.e., get truncated no matter what `max_seq_length` is set to), rather
than guessing from a word-count proxy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def compute_token_length_stats(
    texts: pd.Series,
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int | None = 20000,
    random_seed: int = 42,
) -> dict:
    """
    Tokenizes (without truncation) a sample of `texts` and returns
    token-count percentiles, plus the fraction that would be truncated
    at common max_seq_length choices.

    sample_size: tokenizing the full 232K-row Komati corpus without
    truncation is slow on CPU; None tokenizes everything, an int
    subsamples first (reproducibly, via random_seed) for a fast but
    representative estimate. Use None for the final reported numbers,
    a sample for quick iteration.
    """
    if sample_size is not None and len(texts) > sample_size:
        texts = texts.sample(n=sample_size, random_state=random_seed)
        logger.info("Tokenizing a %d-row sample (of %d total) for length analysis", sample_size, len(texts))
    else:
        logger.info("Tokenizing all %d rows for length analysis", len(texts))

    lengths = []
    batch_size = 256
    text_list = texts.astype(str).tolist()
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        encoded = tokenizer(batch, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    lengths = np.array(lengths)
    stats = {
        "n_tokenized": int(len(lengths)),
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(np.median(lengths)), 2),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "std": round(float(lengths.std()), 2),
        "p90": round(float(np.percentile(lengths, 90)), 2),
        "p95": round(float(np.percentile(lengths, 95)), 2),
        "p99": round(float(np.percentile(lengths, 99)), 2),
    }
    for cap in (128, 256, 384, 512):
        pct_truncated = round(100 * float((lengths > cap).mean()), 2)
        stats[f"pct_truncated_at_{cap}"] = pct_truncated

    return stats


def print_token_length_report(stats: dict, dataset_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Real token-length distribution: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Tokenized:  {stats['n_tokenized']} posts")
    print(f"Mean:       {stats['mean']}")
    print(f"Median:     {stats['median']}")
    print(f"Min / Max:  {stats['min']} / {stats['max']}")
    print(f"p90/p95/p99: {stats['p90']} / {stats['p95']} / {stats['p99']}")
    print("\n% of posts truncated at each max_seq_length:")
    for cap in (128, 256, 384, 512):
        print(f"  {cap:>4} tokens: {stats[f'pct_truncated_at_{cap}']}% truncated")
