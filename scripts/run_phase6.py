"""
run_phase6_resources.py

Phase 6 CLI: browse the verified resource database directly (mirrors
the brief's eventual GET /resources endpoint), or preview what a given
risk level + optional country would return.

Usage:
    python scripts/run_phase6_resources.py --all
    python scripts/run_phase6_resources.py --risk-level CRITICAL
    python scripts/run_phase6_resources.py --risk-level CRITICAL --country IN
    python scripts/run_phase6_resources.py --risk-level LOW   # always []
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.resources.resource_loader import ResourceLoadError  # noqa: E402
from ml.resources.resource_service import get_all_resources, get_resources_for_risk_level  # noqa: E402
from ml.risk.risk_scoring import RISK_LEVELS, load_risk_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: browse the verified resource database")
    parser.add_argument("--config", default="configs/risk_config.yaml")
    parser.add_argument("--all", action="store_true", help="Print the full verified resource database.")
    parser.add_argument("--risk-level", choices=RISK_LEVELS, default=None,
                         help="Preview what resources this risk level (+ optional --country) would return.")
    parser.add_argument("--country", default=None,
                         help="Explicit country code (e.g. IN, US, GB). Never inferred — only used if you pass it.")
    args = parser.parse_args()

    if not args.all and not args.risk_level:
        parser.error("Pass --all or --risk-level")

    repo_root = Path(__file__).resolve().parent.parent
    config = load_risk_config(repo_root / args.config)
    resources_path = repo_root / config["resources_path"]

    try:
        if args.all:
            result = get_all_resources(resources_path)
        else:
            result = get_resources_for_risk_level(args.risk_level, country_code=args.country, resources_path=resources_path)
    except ResourceLoadError as e:
        logger.error(str(e))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
