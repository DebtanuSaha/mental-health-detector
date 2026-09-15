"""
resource_service.py

Retrieval logic on top of the verified resource database
(resource_loader.py). Implements the policy documented back in Phase 0:

    "Since a text post has no dependable location signal, the safest
    default response for CRITICAL/HIGH risk should return both a
    couple of geography-agnostic international directories
    (Findahelpline) and whatever country-specific resources the user
    explicitly filters for via a query param — never guess a country
    from writing style or spelling."

Concretely:
  - LOW risk -> no specific resources (brief §18: general wellbeing
    info only at this level, not crisis-line contact details).
  - MODERATE / HIGH / CRITICAL -> the international directory always;
    plus country-specific resources ONLY if the caller explicitly
    passes a country_code. No inference from the text itself, ever —
    country_code must come from an explicit user choice upstream
    (e.g. a UI dropdown in a later phase), never guessed here.
"""

from __future__ import annotations

from pathlib import Path

from ml.resources.resource_loader import load_resources

NO_RESOURCE_RISK_LEVELS = {"LOW"}


def get_all_resources(resources_path: str | Path = "resources/verified_resources.json") -> list:
    """Full verified resource list — for a future GET /resources-style endpoint."""
    return load_resources(resources_path)


def get_resources_for_risk_level(
    risk_level: str,
    country_code: str | None = None,
    resources_path: str | Path = "resources/verified_resources.json",
) -> list:
    """
    risk_level: one of LOW / MODERATE / HIGH / CRITICAL (see
    ml/risk/risk_scoring.py RISK_LEVELS).
    country_code: an explicit, caller-provided code (e.g. "IN", "US") —
    NEVER inferred from the text itself. None means "no filter was
    given", which correctly returns only the international directory,
    not a guess at every country's resources.
    """
    if risk_level in NO_RESOURCE_RISK_LEVELS:
        return []

    resources = load_resources(resources_path)
    international = [r for r in resources if r["country_code"] == "INTL"]

    if country_code:
        country_specific = [r for r in resources if r["country_code"] == country_code.upper()]
        return international + country_specific

    return international
