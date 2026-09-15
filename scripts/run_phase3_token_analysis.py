"""
run_phase3_token_analysis.py

Tokenizes the real corpus with the actual `mental-roberta-base` (or
whichever model you point it at) tokenizer and reports real BPE
token-length percentiles, to finalize `roberta.max_seq_length` in
configs/training.yaml before fine-tuning.

Usage:
    python scripts/run_phase3_token_analysis.py
    python scripts/run_phase3_token_analysis.py --dataset dreaddit
    python scripts/run_phase3_token_analysis.py --sample-size 5000   # faster, approximate
    python scripts/run_phase3_token_analysis.py --sample-size 0      # tokenize everything (slow)
    python scripts/run_phase3_token_analysis.py --use-fixtures       # tiny local checkpoint, no download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from transformers import AutoTokenizer  # noqa: E402

from ml.data.dataset_loader import DatasetLoadError, load_config as load_dataset_config  # noqa: E402
from ml.data.token_length_analysis import compute_token_length_stats, print_token_length_report  # noqa: E402
from ml.training.train_baseline import _load_dataset, load_training_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: real token-length distribution analysis")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    parser.add_argument("--dataset", choices=["komati", "dreaddit"], default="komati")
    parser.add_argument("--model", default=None, help="Override roberta.model_name_or_path")
    parser.add_argument("--sample-size", type=int, default=20000,
                         help="Rows to sample for tokenization (0 = tokenize all rows)")
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

    train_cfg = load_training_config(repo_root / args.training_config)
    dataset_cfg = load_dataset_config(repo_root / dataset_config_path)

    model_name = args.model or (
        "tests/fixtures/tiny_roberta" if args.use_fixtures else train_cfg["roberta"]["model_name_or_path"]
    )
    logger.info("Loading tokenizer: %s", model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            logger.error(
                "'%s' is a gated model on the Hugging Face Hub. To fix: "
                "1) accept its terms at https://huggingface.co/%s while logged in, "
                "2) create a Read token under Settings -> Access Tokens, "
                "3) run `huggingface-cli login` and paste it, then re-run this script.",
                model_name, model_name,
            )
            return 1
        raise

    try:
        loaded = _load_dataset(args.dataset, dataset_cfg, repo_root)
    except DatasetLoadError as e:
        logger.error(str(e))
        return 1

    sample_size = None if args.sample_size == 0 else args.sample_size
    stats = compute_token_length_stats(loaded.df["text"], tokenizer, sample_size=sample_size)
    print_token_length_report(stats, dataset_name=args.dataset)

    if args.use_fixtures:
        (repo_root / "configs" / "_dataset_config_fixtures.yaml").unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
