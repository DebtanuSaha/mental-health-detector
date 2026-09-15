"""
run_phase3_finetune.py

Phase 3 CLI: fine-tunes a RoBERTa-family model on Komati or Dreaddit,
evaluates with the same metrics as the Phase 2 baseline, and saves the
model + metrics under models/roberta/<dataset>/.

Usage:
    # Real fine-tune (downloads mental/mental-roberta-base on first run):
    python scripts/run_phase3_finetune.py
    python scripts/run_phase3_finetune.py --dataset dreaddit
    python scripts/run_phase3_finetune.py --model roberta-base   # ablation

    # Fast initial pass on a class-balanced subsample of the real data —
    # to validate the pipeline / estimate real training time before
    # committing to a full run (results from this are REDUCED-SCOPE,
    # not the final Phase 3 numbers — the run itself logs a reminder):
    python scripts/run_phase3_finetune.py --max-train-samples 20000
    python scripts/run_phase3_finetune.py --max-train-samples 20000 --max-val-samples 3000

    # Pipeline smoke test — tiny local random-init checkpoint, no download:
    python scripts/run_phase3_finetune.py --use-fixtures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.data.dataset_loader import DatasetLoadError  # noqa: E402
from ml.training.train_roberta import run_finetuning  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: RoBERTa fine-tuning")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    parser.add_argument("--dataset", choices=["komati", "dreaddit"], default=None)
    parser.add_argument("--model", default=None, help="Override roberta.model_name_or_path")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Override roberta.max_train_samples — class-balanced subsample "
                              "of the training set for a fast initial pass.")
    parser.add_argument("--max-val-samples", type=int, default=None,
                         help="Override roberta.max_val_samples.")
    parser.add_argument("--max-test-samples", type=int, default=None,
                         help="Override roberta.max_test_samples.")
    parser.add_argument("--use-fixtures", action="store_true",
                         help="Use the tiny local random-init checkpoint + synthetic CSVs "
                              "for a pipeline smoke test (no network access needed).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    dataset_config_path = args.dataset_config
    training_config_path = args.training_config
    if args.use_fixtures:
        import yaml
        with open(repo_root / args.dataset_config, "r", encoding="utf-8") as f:
            ds_config = yaml.safe_load(f)
        ds_config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
        ds_config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
        ds_config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
        tmp_ds = repo_root / "configs" / "_dataset_config_fixtures.yaml"
        with open(tmp_ds, "w", encoding="utf-8") as f:
            yaml.safe_dump(ds_config, f)
        dataset_config_path = tmp_ds.relative_to(repo_root)

        with open(repo_root / args.training_config, "r", encoding="utf-8") as f:
            t_config = yaml.safe_load(f)
        # Tiny checkpoint has max_position_embeddings=130 and the fixture
        # dataset has ~8-10 train rows — scale hyperparameters accordingly.
        t_config["roberta"]["max_seq_length"] = 64
        t_config["roberta"]["batch_size"] = 2
        t_config["roberta"]["eval_batch_size"] = 2
        t_config["roberta"]["num_epochs"] = 1
        t_config["roberta"]["eval_steps"] = 2
        t_config["roberta"]["logging_steps"] = 1
        t_config["roberta"]["early_stopping_patience"] = 5
        tmp_train = repo_root / "configs" / "_training_config_fixtures.yaml"
        with open(tmp_train, "w", encoding="utf-8") as f:
            yaml.safe_dump(t_config, f)
        training_config_path = tmp_train.relative_to(repo_root)
        logger.info("Using synthetic fixtures + tiny local checkpoint (NOT real data/model).")

    model_override = args.model or ("tests/fixtures/tiny_roberta" if args.use_fixtures else None)

    # Apply --max-*-samples CLI overrides onto the training config (write
    # a temp config if any were given, same pattern as --use-fixtures).
    if any(v is not None for v in (args.max_train_samples, args.max_val_samples, args.max_test_samples)):
        import yaml
        with open(repo_root / training_config_path, "r", encoding="utf-8") as f:
            t_config = yaml.safe_load(f)
        if args.max_train_samples is not None:
            t_config["roberta"]["max_train_samples"] = args.max_train_samples
        if args.max_val_samples is not None:
            t_config["roberta"]["max_val_samples"] = args.max_val_samples
        if args.max_test_samples is not None:
            t_config["roberta"]["max_test_samples"] = args.max_test_samples
        tmp_subsample = repo_root / "configs" / "_training_config_subsample.yaml"
        with open(tmp_subsample, "w", encoding="utf-8") as f:
            yaml.safe_dump(t_config, f)
        training_config_path = tmp_subsample.relative_to(repo_root)

    try:
        run_finetuning(
            training_config_path=training_config_path,
            dataset_config_path=dataset_config_path,
            base_dir=repo_root,
            model_name_or_path_override=model_override,
            dataset_override=args.dataset,
        )
    except DatasetLoadError as e:
        logger.error(str(e))
        return 1
    finally:
        for tmp_name in (
            "configs/_dataset_config_fixtures.yaml",
            "configs/_training_config_fixtures.yaml",
            "configs/_training_config_subsample.yaml",
        ):
            tmp_path = repo_root / tmp_name
            if tmp_path.exists():
                tmp_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
