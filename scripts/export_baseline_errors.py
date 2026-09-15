"""
export_baseline_errors.py

Reproduces the exact deterministic train/val/test split used during
training (same config, same random_seed), loads the already-saved
baseline pipeline, and exports the test-set false negatives and false
positives to CSV for manual qualitative review.

This exists because a single FNR number ("6.58%") doesn't tell you
*why* those posts were missed — reading a sample of the actual missed
posts is how you catch things like subreddit-artifact overfitting
(see README Phase 2 section) before assuming a model is doing genuine
crisis-language understanding.

Usage:
    python scripts/export_baseline_errors.py
    python scripts/export_baseline_errors.py --dataset dreaddit
    python scripts/export_baseline_errors.py --use-fixtures

Privacy: like Phase 1's sample_inspection, this writes real (truncated)
post text to datasets/processed/ for local review only — delete it when
done, never commit it, never attach it to bug reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.data.dataset_loader import DatasetLoadError, load_config as load_dataset_config  # noqa: E402
from ml.models.baseline import load_pipeline  # noqa: E402
from ml.preprocessing.text_cleaner import PreprocessConfig, clean_series  # noqa: E402
from ml.splitting.dataset_split import stratified_split, group_split  # noqa: E402
from ml.training.train_baseline import _load_dataset, load_training_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_CHARS_PREVIEW = 500  # generous vs. Phase 1's 240, since this is for real error inspection


def main() -> int:
    parser = argparse.ArgumentParser(description="Export baseline false negatives/positives for review")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    parser.add_argument("--dataset", choices=["komati", "dreaddit"], default=None)
    parser.add_argument("--use-fixtures", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    dataset_config_path = args.dataset_config
    if args.use_fixtures:
        import yaml
        with open(repo_root / args.dataset_config, "r", encoding="utf-8") as f:
            ds_config = yaml.safe_load(f)
        ds_config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
        ds_config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
        ds_config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
        tmp = repo_root / "configs" / "_dataset_config_fixtures.yaml"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(ds_config, f)
        dataset_config_path = tmp.relative_to(repo_root)
        logger.info("Using synthetic fixtures (NOT real data).")

    train_cfg = load_training_config(repo_root / args.training_config)
    dataset_cfg = load_dataset_config(repo_root / dataset_config_path)
    dataset_name = args.dataset or train_cfg["baseline"]["dataset"]

    try:
        loaded = _load_dataset(dataset_name, dataset_cfg, repo_root)
    except DatasetLoadError as e:
        logger.error(str(e))
        return 1
    df = loaded.df.copy()

    if train_cfg.get("preprocessing", {}).get("drop_duplicate_text", True):
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    pp_cfg = PreprocessConfig.from_dict(train_cfg["preprocessing"])
    raw_text = df["text"].copy()  # keep untouched original for the export
    df["text"] = clean_series(df["text"], pp_cfg)
    df["raw_text"] = raw_text
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    split_cfg = train_cfg["split"]
    if split_cfg["group_based"]:
        result = group_split(df, group_col=split_cfg["group_column"], label_col="label",
                              train_ratio=split_cfg["train_ratio"], val_ratio=split_cfg["val_ratio"],
                              test_ratio=split_cfg["test_ratio"], random_seed=split_cfg["random_seed"])
    else:
        result = stratified_split(df, label_col="label",
                                   train_ratio=split_cfg["train_ratio"], val_ratio=split_cfg["val_ratio"],
                                   test_ratio=split_cfg["test_ratio"], random_seed=split_cfg["random_seed"])

    model_path = repo_root / train_cfg["baseline"]["output_dir"] / f"{dataset_name}_baseline_pipeline.joblib"
    try:
        pipeline = load_pipeline(model_path)
    except FileNotFoundError:
        logger.error("No saved model at %s — run scripts/run_phase2_baseline.py first.", model_path)
        return 1

    test_df = result.test.copy()
    test_df["pred"] = pipeline.predict(test_df["text"])
    test_df["pred_proba_class1"] = pipeline.predict_proba(test_df["text"])[:, 1]

    false_negatives = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)].copy()
    false_positives = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)].copy()

    for name, subset in (("false_negatives", false_negatives), ("false_positives", false_positives)):
        subset = subset.copy()
        subset["text_preview"] = subset["raw_text"].str.slice(0, MAX_CHARS_PREVIEW)
        out_cols = ["text_preview", "pred_proba_class1"]
        out_path = repo_root / dataset_cfg["processed_dir"] / f"{dataset_name}_{name}.csv"
        subset[out_cols].to_csv(out_path, index=False)
        logger.info("Wrote %d %s to %s", len(subset), name, out_path)

    logger.info(
        "FN rate: %.2f%% (%d/%d)  |  FP rate: %.2f%% (%d/%d)",
        100 * len(false_negatives) / max((test_df["label"] == 1).sum(), 1),
        len(false_negatives), (test_df["label"] == 1).sum(),
        100 * len(false_positives) / max((test_df["label"] == 0).sum(), 1),
        len(false_positives), (test_df["label"] == 0).sum(),
    )

    if args.use_fixtures:
        (repo_root / "configs" / "_dataset_config_fixtures.yaml").unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
