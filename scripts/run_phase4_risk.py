"""
run_phase4_risk_assessment.py

Phase 4 CLI: runs the full risk-assessment pipeline (crisis model +
distress model + sentiment -> combined risk score/level) on a single
piece of text, and prints the structured JSON result.

Usage:
    python scripts/run_phase4_risk_assessment.py --text "some text here"
    python scripts/run_phase4_risk_assessment.py --text "..." --prefer baseline
    python scripts/run_phase4_risk_assessment.py --text "..." --country IN
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.inference.model_predictor import load_predictor  # noqa: E402
from ml.inference.sentiment import SentimentAnalyzer  # noqa: E402
from ml.risk.risk_scoring import assess_risk, load_risk_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: risk assessment on a single text")
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", default="configs/risk_config.yaml")
    parser.add_argument("--prefer", choices=["roberta", "baseline"], default="roberta",
                         help="Prefer the fine-tuned RoBERTa model if available (default), "
                              "or force the Phase 2 baseline regardless.")
    parser.add_argument("--country", default=None,
                         help="Explicit country code (e.g. IN, US, GB) for country-specific "
                              "resources alongside the international directory. Never inferred "
                              "from the text itself — omit this flag rather than guessing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config = load_risk_config(repo_root / args.config)

    crisis_cfg = config["model_paths"]["crisis"]
    distress_cfg = config["model_paths"]["distress"]

    try:
        crisis_predictor = load_predictor(
            repo_root / crisis_cfg["roberta_dir"], repo_root / crisis_cfg["baseline_path"], prefer=args.prefer
        )
        distress_predictor = load_predictor(
            repo_root / distress_cfg["roberta_dir"], repo_root / distress_cfg["baseline_path"], prefer=args.prefer
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    sentiment_analyzer = SentimentAnalyzer(config["sentiment"]["model_name"])
    resources_path = repo_root / config["resources_path"]

    result = assess_risk(
        args.text, crisis_predictor, distress_predictor, sentiment_analyzer, config,
        country_code=args.country, resources_path=resources_path,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
