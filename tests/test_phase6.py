"""
Tests for Phase 6: resource_loader (schema validation) and
resource_service (the LOW=empty, international-always,
country-only-if-explicit filtering policy).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))  # append, not insert(0) — repo root has a datasets/ folder that would otherwise shadow the pip "datasets" package

from ml.resources.resource_loader import ResourceLoadError, load_resources  # noqa: E402
from ml.resources.resource_service import get_all_resources, get_resources_for_risk_level  # noqa: E402

REAL_RESOURCES_PATH = REPO_ROOT / "resources" / "verified_resources.json"


# --- resource_loader ---

def test_load_real_resources_file_is_valid():
    resources = load_resources(REAL_RESOURCES_PATH)
    assert len(resources) >= 5
    names = {r["name"] for r in resources}
    assert "Findahelpline" in names
    assert "Tele-MANAS" in names


def test_load_resources_missing_file_raises(tmp_path):
    with pytest.raises(ResourceLoadError):
        load_resources(tmp_path / "does_not_exist.json")


def test_load_resources_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ResourceLoadError):
        load_resources(bad)


def test_load_resources_not_a_list_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "not a list"}))
    with pytest.raises(ResourceLoadError):
        load_resources(bad)


def test_load_resources_missing_field_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"name": "Incomplete Entry"}]))  # missing everything else
    with pytest.raises(ResourceLoadError):
        load_resources(bad)


# --- resource_service ---

def test_get_all_resources_matches_loader():
    assert get_all_resources(REAL_RESOURCES_PATH) == load_resources(REAL_RESOURCES_PATH)


def test_low_risk_always_returns_empty_regardless_of_country():
    assert get_resources_for_risk_level("LOW", resources_path=REAL_RESOURCES_PATH) == []
    assert get_resources_for_risk_level("LOW", country_code="IN", resources_path=REAL_RESOURCES_PATH) == []


@pytest.mark.parametrize("risk_level", ["MODERATE", "HIGH", "CRITICAL"])
def test_non_low_risk_without_country_returns_only_international(risk_level):
    result = get_resources_for_risk_level(risk_level, country_code=None, resources_path=REAL_RESOURCES_PATH)
    assert len(result) >= 1
    assert all(r["country_code"] == "INTL" for r in result)


def test_non_low_risk_with_country_includes_international_and_country_specific():
    result = get_resources_for_risk_level("CRITICAL", country_code="IN", resources_path=REAL_RESOURCES_PATH)
    country_codes = {r["country_code"] for r in result}
    assert "INTL" in country_codes
    assert "IN" in country_codes
    # Should include both Tele-MANAS and iCALL (both IN) plus Findahelpline (INTL)
    names = {r["name"] for r in result}
    assert "Tele-MANAS" in names
    assert "iCALL Psychosocial Helpline" in names
    assert "Findahelpline" in names


def test_country_code_matching_is_case_insensitive():
    lower = get_resources_for_risk_level("HIGH", country_code="in", resources_path=REAL_RESOURCES_PATH)
    upper = get_resources_for_risk_level("HIGH", country_code="IN", resources_path=REAL_RESOURCES_PATH)
    assert lower == upper


def test_unknown_country_code_returns_only_international():
    result = get_resources_for_risk_level("HIGH", country_code="ZZ", resources_path=REAL_RESOURCES_PATH)
    assert all(r["country_code"] == "INTL" for r in result)
    assert len(result) >= 1  # international directory still present


def test_never_fabricates_phone_numbers_beyond_the_verified_file():
    """Sanity check: every phone number returned must exist verbatim in
    the source file — resource_service must never construct/alter entries."""
    raw_entries = load_resources(REAL_RESOURCES_PATH)
    valid_phones = {r["phone"] for r in raw_entries}
    result = get_resources_for_risk_level("CRITICAL", country_code="US", resources_path=REAL_RESOURCES_PATH)
    for r in result:
        assert r["phone"] in valid_phones
