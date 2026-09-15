"""
dataset_loader.py

Loads the two Phase-1 raw datasets (Komati suicide-watch, Dreaddit) and
normalizes each into a common minimal schema:

    text   : str   -- the raw post text
    label  : int   -- 0/1 binary label, meaning defined per-dataset (see below)
    source : str   -- which dataset the row came from ("komati" / "dreaddit")

Design notes:
- Komati's binary label is (non-suicide=0, suicide=1).
- Dreaddit's binary label is (no_stress=0, stress=1).
  These are DIFFERENT phenomena (crisis-risk vs. general stress) — this
  loader does NOT merge them onto one shared label. Phase 4 (risk scoring)
  will decide how each dataset's signal feeds the multi-label target
  (suicide_indicator vs. emotional_distress) rather than doing that here.
- Only `text` + `label` (+ light metadata) are kept in memory by default.
  Dreaddit's 100+ LIWC feature columns are dropped unless explicitly
  requested via `keep_liwc_features=True`, since Phase 1-3 don't need them
  and they bloat memory for no benefit yet.
- No raw text is ever written to logs from this module. Logging here is
  restricted to counts/shapes/paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DatasetLoadError(Exception):
    """Raised when a raw dataset file is missing or doesn't match the expected schema."""


@dataclass
class LoadedDataset:
    """Container for a normalized dataset plus a bit of provenance."""

    name: str
    df: pd.DataFrame  # columns: text, label, source
    positive_label_name: str
    negative_label_name: str


def load_config(config_path: str | Path = "configs/dataset_config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise DatasetLoadError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DatasetLoadError(
            f"{dataset_name}: expected column(s) {missing} not found. "
            f"Found columns: {list(df.columns)}. "
            f"This usually means the raw file doesn't match the version "
            f"documented in Phase 0 — re-check the download."
        )


def load_komati(config: dict, base_dir: str | Path = ".") -> LoadedDataset:
    """Load the Komati 'Suicide and Depression Detection' CSV."""
    cfg = config["komati"]
    path = Path(base_dir) / cfg["raw_path"]
    if not path.exists():
        raise DatasetLoadError(
            f"Komati raw file not found at {path}. "
            f"Download it first — see datasets/README.md."
        )

    raw = pd.read_csv(path)
    text_col, label_col = cfg["text_column"], cfg["label_column"]
    _require_columns(raw, [text_col, label_col], "komati")

    label_map = cfg["label_map"]
    unmapped = set(raw[label_col].unique()) - set(label_map.keys())
    if unmapped:
        raise DatasetLoadError(
            f"komati: found label value(s) {unmapped} not present in "
            f"configs/dataset_config.yaml label_map ({label_map}). "
            f"Update the config rather than guessing a mapping in code."
        )

    df = pd.DataFrame(
        {
            "text": raw[text_col].astype(str),
            "label": raw[label_col].map(label_map).astype(int),
            "source": "komati",
        }
    )
    logger.info("Loaded komati: %d rows from %s", len(df), path)
    return LoadedDataset(
        name="komati",
        df=df,
        positive_label_name=cfg["positive_label_name"],
        negative_label_name=cfg["negative_label_name"],
    )


def load_dreaddit(
    config: dict,
    base_dir: str | Path = ".",
    split: str = "both",
    keep_liwc_features: bool = False,
) -> LoadedDataset:
    """
    Load Dreaddit train/test/both.

    split: "train", "test", or "both" (default — concatenates train+test,
           since Phase 1 does its own splitting later rather than reusing
           the original authors' split boundaries).
    """
    cfg = config["dreaddit"]
    text_col, label_col = cfg["text_column"], cfg["label_column"]

    frames = []
    paths_to_load = []
    if split in ("train", "both"):
        paths_to_load.append(Path(base_dir) / cfg["raw_path_train"])
    if split in ("test", "both"):
        paths_to_load.append(Path(base_dir) / cfg["raw_path_test"])
    if not paths_to_load:
        raise ValueError(f"Invalid split '{split}', expected 'train', 'test', or 'both'.")

    for path in paths_to_load:
        if not path.exists():
            raise DatasetLoadError(
                f"Dreaddit raw file not found at {path}. "
                f"Download it first — see datasets/README.md."
            )
        raw = pd.read_csv(path)
        _require_columns(raw, [text_col, label_col], "dreaddit")

        keep_cols = {text_col: "text", label_col: "label"}
        subset = raw[list(keep_cols.keys())].rename(columns=keep_cols)
        subset["label"] = subset["label"].astype(int)

        if keep_liwc_features:
            liwc_cols = [c for c in raw.columns if c.startswith("lex_liwc_")]
            subset = pd.concat([subset, raw[liwc_cols]], axis=1)

        subset["source"] = "dreaddit"
        frames.append(subset)
        logger.info("Loaded dreaddit split from %s: %d rows", path, len(subset))

    df = pd.concat(frames, ignore_index=True)
    return LoadedDataset(
        name="dreaddit",
        df=df,
        positive_label_name=cfg["positive_label_name"],
        negative_label_name=cfg["negative_label_name"],
    )


def load_all(
    config_path: str | Path = "configs/dataset_config.yaml",
    base_dir: str | Path = ".",
) -> dict[str, LoadedDataset]:
    """Convenience: load both datasets in one call, config path/base_dir driven."""
    config = load_config(config_path)
    return {
        "komati": load_komati(config, base_dir=base_dir),
        "dreaddit": load_dreaddit(config, base_dir=base_dir),
    }
