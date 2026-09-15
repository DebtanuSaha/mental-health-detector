"""
train_baseline.py

Orchestrates Phase 2: load a dataset (via the Phase 1 loader), preprocess,
split train/val/test, report class distribution before training, fit the
TF-IDF + Logistic Regression baseline, evaluate on held-out test data, and
save the fitted pipeline + a metrics JSON.

This is deliberately dataset-agnostic (works for Komati or Dreaddit, or
any future dataset loaded into the same text/label schema) — which
dataset to use is a config choice (`baseline.dataset` in training.yaml),
not a code branch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.data.dataset_loader import DatasetLoadError, load_config as load_dataset_config, load_dreaddit, load_komati
from ml.models.baseline import build_baseline_pipeline, save_pipeline
from ml.preprocessing.text_cleaner import PreprocessConfig, clean_series
from ml.splitting.dataset_split import group_split, stratified_split
from ml.training.evaluate import evaluate_binary, print_evaluation

logger = logging.getLogger(__name__)


def load_training_config(path: str | Path = "configs/training.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dataset(dataset_name: str, dataset_config: dict, base_dir: Path):
    if dataset_name == "komati":
        return load_komati(dataset_config, base_dir=base_dir)
    elif dataset_name == "dreaddit":
        return load_dreaddit(dataset_config, base_dir=base_dir, split="both")
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}' — expected 'komati' or 'dreaddit'")


def report_class_distribution(df: pd.DataFrame, label_col: str, split_name: str) -> None:
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    logger.info("Class distribution (%s, n=%d): %s", split_name, total,
                {str(k): f"{v} ({100 * v / total:.1f}%)" for k, v in counts.items()})


def run_training(
    training_config_path: str | Path = "configs/training.yaml",
    dataset_config_path: str | Path = "configs/dataset_config.yaml",
    base_dir: str | Path = ".",
) -> dict:
    base_dir = Path(base_dir)
    train_cfg = load_training_config(base_dir / training_config_path)
    dataset_cfg = load_dataset_config(base_dir / dataset_config_path)

    baseline_cfg = train_cfg["baseline"]
    dataset_name = baseline_cfg["dataset"]

    loaded = _load_dataset(dataset_name, dataset_cfg, base_dir)
    df = loaded.df.copy()
    logger.info("Loaded '%s': %d rows", dataset_name, len(df))

    # --- Drop duplicate text rows BEFORE splitting ---
    # Phase 1's real-data run found 0 duplicates in Komati but 21 in
    # Dreaddit. With a random (non-group) split, a duplicate left in
    # place can land the same text in both train and test, inflating
    # test metrics on a leaked example rather than a genuinely unseen
    # one. Dropping here, not just reporting, per brief §12's leakage
    # requirement.
    if train_cfg.get("preprocessing", {}).get("drop_duplicate_text", True):
        before = len(df)
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
        n_dropped = before - len(df)
        if n_dropped:
            logger.info("Dropped %d duplicate-text rows before splitting", n_dropped)

    # --- Preprocessing ---
    pp_cfg = PreprocessConfig.from_dict(train_cfg["preprocessing"])
    df["text"] = clean_series(df["text"], pp_cfg)
    before = len(df)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d rows that became empty after preprocessing", before - len(df))

    # --- Report class distribution BEFORE training (brief §13) ---
    report_class_distribution(df, "label", "full dataset")

    # --- Split ---
    split_cfg = train_cfg["split"]
    if split_cfg["group_based"]:
        result = group_split(
            df,
            group_col=split_cfg["group_column"],
            label_col="label",
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            random_seed=split_cfg["random_seed"],
        )
    else:
        result = stratified_split(
            df,
            label_col="label",
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            random_seed=split_cfg["random_seed"],
        )

    for name, split_df in (("train", result.train), ("val", result.val), ("test", result.test)):
        report_class_distribution(split_df, "label", name)

    # --- Fit baseline pipeline on train only (avoids TF-IDF vocabulary leakage) ---
    pipeline = build_baseline_pipeline(
        baseline_cfg["tfidf"], baseline_cfg["logistic_regression"]
    )
    logger.info("Fitting TF-IDF + LogisticRegression on %d train rows...", len(result.train))
    pipeline.fit(result.train["text"], result.train["label"])

    # --- Evaluate on held-out test set ---
    y_true = result.test["label"].to_numpy()
    y_pred = pipeline.predict(result.test["text"])
    y_proba = pipeline.predict_proba(result.test["text"])[:, 1]  # prob of class 1

    eval_result = evaluate_binary(y_true, y_pred, y_proba, positive_label=1)
    print_evaluation(eval_result, positive_label_name=loaded.positive_label_name)

    # --- Val-set metrics too (useful once we start comparing to Phase 3) ---
    y_val_true = result.val["label"].to_numpy()
    y_val_pred = pipeline.predict(result.val["text"])
    y_val_proba = pipeline.predict_proba(result.val["text"])[:, 1]
    val_eval_result = evaluate_binary(y_val_true, y_val_pred, y_val_proba, positive_label=1)

    # --- Save model + metrics ---
    output_dir = base_dir / baseline_cfg["output_dir"]
    model_path = save_pipeline(pipeline, output_dir, filename=f"{dataset_name}_baseline_pipeline.joblib")
    logger.info("Saved baseline pipeline to %s", model_path)

    metrics = {
        "dataset": dataset_name,
        "n_train": len(result.train),
        "n_val": len(result.val),
        "n_test": len(result.test),
        "test_metrics": eval_result.to_dict(),
        "val_metrics": val_eval_result.to_dict(),
        "config": {
            "tfidf": baseline_cfg["tfidf"],
            "logistic_regression": baseline_cfg["logistic_regression"],
            "split": split_cfg,
            "preprocessing": train_cfg["preprocessing"],
        },
    }
    metrics_path = output_dir / f"{dataset_name}_baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    return metrics
