"""
run_phase1_exploration.py

Phase 1 CLI: loads Komati + Dreaddit, prints dataset statistics
(counts, class distribution, missing/duplicates, text-length distribution),
and writes a small stratified sample-inspection CSV to datasets/processed/
for manual review.

Usage:
    # Against the real downloaded datasets (see datasets/README.md first):
    python scripts/run_phase1_exploration.py

    # Against the small synthetic fixtures (to sanity-check the pipeline
    # logic without needing the real ~230K-row download):
    python scripts/run_phase1_exploration.py --use-fixtures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/run_phase1_exploration.py` from repo root.
sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.data.dataset_loader import (  # noqa: E402
    DatasetLoadError,
    load_config,
    load_dreaddit,
    load_komati,
)
from ml.data.dataset_statistics import (  # noqa: E402
    compute_statistics,
    print_report,
    sample_inspection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: dataset exploration")
    parser.add_argument(
        "--config",
        default="configs/dataset_config.yaml",
        help="Path to dataset_config.yaml",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        help="Load the small synthetic CSVs under tests/fixtures/ instead of "
        "the real raw datasets (useful for a quick pipeline sanity check).",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory raw_path entries in the config are relative to.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.use_fixtures:
        config = load_config(repo_root / args.config)
        # Point the config at the fixtures instead of the real raw/ paths.
        config["komati"]["raw_path"] = "tests/fixtures/sample_komati.csv"
        config["dreaddit"]["raw_path_train"] = "tests/fixtures/sample_dreaddit.csv"
        config["dreaddit"]["raw_path_test"] = "tests/fixtures/sample_dreaddit.csv"
        base_dir = repo_root
        logger.info("Using synthetic fixtures under tests/fixtures/ (NOT real data).")
    else:
        config = load_config(repo_root / args.config)
        base_dir = Path(args.base_dir).resolve()

    try:
        komati = load_komati(config, base_dir=base_dir)
        dreaddit = load_dreaddit(config, base_dir=base_dir, split="both" if not args.use_fixtures else "train")
    except DatasetLoadError as e:
        logger.error(str(e))
        logger.error(
            "If you haven't downloaded the raw datasets yet, see "
            "datasets/README.md, or re-run with --use-fixtures to test the "
            "pipeline against small synthetic data first."
        )
        return 1

    processed_dir = base_dir / config["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    for loaded in (komati, dreaddit):
        stats = compute_statistics(loaded.df, dataset_name=loaded.name)
        print_report(stats)

        insp_cfg = config["sample_inspection"]
        preview = sample_inspection(
            loaded.df,
            n_per_class=insp_cfg["n_samples_per_class"],
            max_chars_preview=insp_cfg["max_chars_preview"],
            random_seed=insp_cfg["random_seed"],
        )
        out_path = processed_dir / f"{loaded.name}_sample_inspection.csv"
        preview.to_csv(out_path, index=False)
        logger.info("Wrote sample inspection preview: %s (%d rows)", out_path, len(preview))

    logger.info("Phase 1 exploration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
