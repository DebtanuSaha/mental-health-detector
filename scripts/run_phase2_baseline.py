"""
run_phase2_baseline.py

Phase 2 CLI: trains and evaluates the TF-IDF + Logistic Regression
baseline on either Komati or Dreaddit (per configs/training.yaml, or
overridden with --dataset), and saves the fitted pipeline + metrics
JSON under models/baseline/.

Usage:
    # Against the real downloaded datasets:
    python scripts/run_phase2_baseline.py
    python scripts/run_phase2_baseline.py --dataset dreaddit

    # Against the small synthetic fixtures (pipeline sanity check):
    python scripts/run_phase2_baseline.py --use-fixtures
    python scripts/run_phase2_baseline.py --use-fixtures --dataset dreaddit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.data.dataset_loader import DatasetLoadError  # noqa: E402
from ml.training.train_baseline import load_training_config, run_training  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: TF-IDF + Logistic Regression baseline")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    parser.add_argument(
        "--dataset",
        choices=["komati", "dreaddit"],
        default=None,
        help="Override baseline.dataset from training.yaml",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Load small synthetic fixtures under tests/fixtures/ instead of real raw data.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.use_fixtures:
        # Write a temp dataset config pointing at fixtures, same pattern as Phase 1's runner.
        import yaml

        with open(repo_root / args.dataset_config, "r", encoding="utf-8") as f:
            ds_config = yaml.safe_load(f)
        ds_config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
        ds_config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
        ds_config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
        tmp_dataset_config = repo_root / "configs" / "_dataset_config_fixtures.yaml"
        with open(tmp_dataset_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(ds_config, f)
        dataset_config_path = tmp_dataset_config.relative_to(repo_root)
        logger.info("Using synthetic fixtures under tests/fixtures/ (NOT real data).")
    else:
        dataset_config_path = args.dataset_config

    training_config_path = args.training_config
    if args.dataset:
        import yaml

        with open(repo_root / training_config_path, "r", encoding="utf-8") as f:
            t_config = yaml.safe_load(f)
        t_config["baseline"]["dataset"] = args.dataset
        tmp_training_config = repo_root / "configs" / "_training_config_override.yaml"
        with open(tmp_training_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(t_config, f)
        training_config_path = tmp_training_config.relative_to(repo_root)

    try:
        run_training(
            training_config_path=training_config_path,
            dataset_config_path=dataset_config_path,
            base_dir=repo_root,
        )
    except DatasetLoadError as e:
        logger.error(str(e))
        logger.error(
            "If you haven't downloaded the raw datasets yet, see "
            "datasets/README.md, or re-run with --use-fixtures first."
        )
        return 1
    finally:
        # Clean up any temp config files we wrote.
        for tmp in ("configs/_dataset_config_fixtures.yaml", "configs/_training_config_override.yaml"):
            p = repo_root / tmp
            if p.exists():
                p.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
