"""Tests for .env loading and player slug helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from league_stats.core.champions import parse_riot_id, players_group_slug
from league_stats.core.config import AppConfig, load_config


def test_parse_riot_id_splits_name_and_tag() -> None:
    """Riot IDs are parsed from Name#Tag form."""
    assert parse_riot_id("Hide on Bush#KR1") == ("Hide on Bush", "KR1")


def test_parse_riot_id_rejects_missing_hash() -> None:
    """Invalid Riot ID strings raise a clear error."""
    with pytest.raises(ValueError, match="Name#Tag"):
        parse_riot_id("NoTagHere")


def test_players_group_slug_joins_multiple_players() -> None:
    """Multi-player groups get a stable sorted slug."""
    slug = players_group_slug([("Bob", "NA1"), ("Alice", "EUW")])
    assert slug == "alice_euw__bob_na1"


def test_load_config_reads_api_key_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RIOT_API_KEY is loaded from .env when not already in the environment."""
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("RIOT_API_KEY=RGAPI-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(riot_id="Test", tagline="EUW")
    assert config.api_key == "RGAPI-from-dotenv"


def test_load_config_prefers_existing_env_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment variable wins over .env contents."""
    monkeypatch.setenv("RIOT_API_KEY", "RGAPI-from-env")
    dotenv = tmp_path / ".env"
    dotenv.write_text("RIOT_API_KEY=RGAPI-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(riot_id="Test", tagline="EUW")
    assert config.api_key == "RGAPI-from-env"


def test_load_config_cli_api_key_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI --api-key still takes precedence over .env."""
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("RIOT_API_KEY=RGAPI-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(riot_id="Test", tagline="EUW", api_key="RGAPI-from-cli")
    assert config.api_key == "RGAPI-from-cli"


def test_gemini_api_key_defaults_to_none() -> None:
    """gemini_api_key is optional; AppConfig doesn't require it."""
    config = AppConfig(riot_id="Test", tagline="EUW", api_key="RGAPI-test")
    assert config.gemini_api_key is None


def test_load_config_reads_gemini_api_key_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GEMINI_API_KEY is loaded from .env when not already in the environment."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=AIza-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = load_config(riot_id="Test", tagline="EUW", api_key="RGAPI-test")
    assert config.gemini_api_key == "AIza-from-dotenv"
