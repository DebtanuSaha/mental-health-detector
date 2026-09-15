"""
resource_loader.py

Loads resources/verified_resources.json — the static, hand-verified
crisis-resource database sourced in Phase 0 from official government/
organization pages (Tele-MANAS, iCALL, 988, Samaritans, Findahelpline).

Per the brief's own rule (§17): the LLM must never invent helpline
numbers or resources. This loader is the ONLY path resource data enters
the application — nothing downstream (risk_scoring.py, any future API
layer) constructs or edits resource entries itself, it only filters
what this file returns.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = {"name", "country", "country_code", "type", "phone", "website", "availability", "source"}


class ResourceLoadError(Exception):
    """Raised when the resource database file is missing or malformed."""


def load_resources(path: str | Path = "res/verified_resources.json") -> list:
    path = Path(path)
    if not path.exists():
        raise ResourceLoadError(f"Resource database not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            resources = json.load(f)
        except json.JSONDecodeError as e:
            raise ResourceLoadError(f"Resource database at {path} is not valid JSON: {e}") from e

    if not isinstance(resources, list):
        raise ResourceLoadError(f"Resource database at {path} must be a JSON list of resource objects")

    for i, entry in enumerate(resources):
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            raise ResourceLoadError(
                f"Resource entry #{i} ({entry.get('name', '?')}) in {path} is missing field(s): {missing}"
            )

    return resources
