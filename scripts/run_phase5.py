"""
run_phase5_explain.py

Phase 5 CLI: explains a single crisis or distress prediction —
Integrated Gradients token attributions for a fine-tuned RoBERTa model,
or linear-coefficient feature attributions if only the Phase 2 baseline
is available.

Usage:
    python scripts/run_phase5_explain.py --text "..." --signal crisis
    python scripts/run_phase5_explain.py --text "..." --signal distress --prefer baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.explainability.explainer import explain  # noqa: E402
from ml.inference.model_predictor import load_predictor  # noqa: E402
from ml.risk.risk_scoring import load_risk_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: explain a single crisis/distress prediction")
    parser.add_argument("--text", required=True)
    parser.add_argument("--signal", choices=["crisis", "distress"], default="crisis")
    parser.add_argument("--config", default="configs/risk_config.yaml")
    parser.add_argument("--prefer", choices=["roberta", "baseline"], default="roberta",
                         help="Prefer the fine-tuned RoBERTa model if available (default), "
                              "or force the Phase 2 baseline (and its linear-coefficient explainer).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config = load_risk_config(repo_root / args.config)
    signal_cfg = config["model_paths"][args.signal]

    try:
        predictor = load_predictor(
            repo_root / signal_cfg["roberta_dir"], repo_root / signal_cfg["baseline_path"], prefer=args.prefer
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    result = explain(args.text, predictor)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
