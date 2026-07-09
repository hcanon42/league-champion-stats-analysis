"""Application configuration.

Configuration is resolved in order of precedence:

1. CLI options,
2. environment variables (``RIOT_API_KEY``, ``ANALYZER_*`` / legacy ``VIKTOR_*``),
3. an optional ``config.toml`` file,
4. built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Final

from champions import build_label, champion_slug, normalize_role, player_slug, role_display
from pydantic import BaseModel, Field, field_validator

# Platform routing values -> regional routing hosts used by account-v1/match-v5.
PLATFORM_TO_REGION: Final[dict[str, str]] = {
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "oc1": "sea",
    "kr": "asia",
    "jp1": "asia",
    "vn2": "sea",
    "tw2": "sea",
    "sg2": "sea",
    "ph2": "sea",
    "th2": "sea",
}
VALID_REGIONS: Final[frozenset[str]] = frozenset({"europe", "americas", "asia", "sea"})
VALID_PLATFORMS: Final[frozenset[str]] = frozenset(PLATFORM_TO_REGION.keys())
REGION_DEFAULT_PLATFORM: Final[dict[str, str]] = {
    "europe": "euw1",
    "americas": "na1",
    "asia": "kr",
    "sea": "oc1",
}

RANKED_SOLO_QUEUE_ID: Final[int] = 420
REMAKE_MAX_DURATION_S: Final[int] = 300


class AppConfig(BaseModel):
    """Validated runtime configuration for a full analysis run."""

    riot_id: str
    tagline: str
    region: str = "europe"
    platform: str | None = None
    api_key: str
    match_count: int = Field(default=500, ge=1, le=2000)
    min_games: int = Field(default=20, ge=1)
    champion: str = "Viktor"
    role: str = "MIDDLE"
    queue_id: int = RANKED_SOLO_QUEUE_ID
    output_dir: Path = Path("output")
    graphs_dir: Path = Path("graphs")
    cache_dir: Path = Path(".cache")
    template_dir: Path = Path("templates")
    requests_per_second: int = Field(default=18, ge=1)
    requests_per_two_minutes: int = Field(default=95, ge=1)
    max_retries: int = Field(default=5, ge=0)
    request_timeout_s: float = Field(default=15.0, gt=0)
    verbose: bool = False

    @field_validator("role", mode="before")
    @classmethod
    def _normalise_role(cls, value: str) -> str:
        """Accept lane aliases (``mid``, ``support``, ...) and Riot values."""
        return normalize_role(str(value))

    @property
    def role_display(self) -> str:
        """Short lane label for reports (``mid``, ``top``, ...)."""
        return role_display(self.role)

    @property
    def build_label(self) -> str:
        """Champion + lane label (e.g. ``Viktor mid``)."""
        return build_label(self.champion, self.role)

    @property
    def player_reports_dir(self) -> Path:
        """Directory holding every build report for this player."""
        return self.output_dir / "reports" / player_slug(self.riot_id, self.tagline)

    @property
    def report_dir(self) -> Path:
        """Per-player/champion/lane output directory (overwritten on re-run)."""
        return (
            self.output_dir
            / "reports"
            / player_slug(self.riot_id, self.tagline)
            / champion_slug(self.champion, self.role)
        )

    @property
    def run_graphs_dir(self) -> Path:
        """Graph assets for the current report run."""
        return self.report_dir / "graphs"

    @field_validator("region", mode="before")
    @classmethod
    def _normalise_region(cls, value: str) -> str:
        """Accept both regional ("europe") and platform ("euw1") routing values."""
        region = str(value).strip().lower()
        region = PLATFORM_TO_REGION.get(region, region)
        if region not in VALID_REGIONS:
            raise ValueError(
                f"Unknown region {value!r}; use one of {sorted(VALID_REGIONS)} "
                f"or a platform code like 'euw1'."
            )
        return region

    @field_validator("platform", mode="before")
    @classmethod
    def _normalise_platform(cls, value: str | None) -> str | None:
        """Normalise an optional platform routing value."""
        if value is None:
            return None
        platform = str(value).strip().lower()
        if platform not in VALID_PLATFORMS:
            raise ValueError(
                f"Unknown platform {value!r}; use one of {sorted(VALID_PLATFORMS)}."
            )
        return platform

    @classmethod
    def platform_from_region_input(cls, region_input: str) -> str | None:
        """If ``region_input`` is a platform code, return it."""
        key = str(region_input).strip().lower()
        return key if key in VALID_PLATFORMS else None

    @property
    def routing_platform(self) -> str:
        """Platform host for league-v4 / summoner-v4 (e.g. ``euw1``)."""
        if self.platform:
            return self.platform
        return REGION_DEFAULT_PLATFORM.get(self.region, "euw1")

    @field_validator("api_key")
    @classmethod
    def _check_api_key(cls, value: str) -> str:
        """Reject an obviously missing API key early with a clear message."""
        if not value or value == "RGAPI-xxxxxxxx":
            raise ValueError(
                "Missing Riot API key. Set RIOT_API_KEY or pass --api-key "
                "(get one at https://developer.riotgames.com)."
            )
        return value

    @property
    def db_path(self) -> Path:
        """Path of the SQLite match store."""
        return self.cache_dir / "matches.sqlite"

    @property
    def http_cache_dir(self) -> Path:
        """Directory of the diskcache HTTP cache."""
        return self.cache_dir / "http"

    def ensure_directories(self) -> None:
        """Create output, player report and cache directories if missing."""
        for path in (self.output_dir, self.player_reports_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML config file, returning an empty dict when absent.

    Args:
        path: Path of the TOML file.

    Returns:
        Parsed key/value pairs (top-level table only).
    """
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(config_file: Path | None = None, **overrides: Any) -> AppConfig:
    """Build an :class:`AppConfig` from file, environment and overrides.

    Args:
        config_file: Optional path to a ``config.toml``; defaults to
            ``./config.toml`` when present.
        **overrides: CLI-level overrides; ``None`` values are ignored.

    Returns:
        A fully validated configuration object.

    Raises:
        pydantic.ValidationError: If required values are missing or invalid.
    """
    data: dict[str, Any] = _read_toml(config_file or Path("config.toml"))
    env_map = {
        "api_key": os.environ.get("RIOT_API_KEY"),
        "riot_id": os.environ.get("ANALYZER_RIOT_ID") or os.environ.get("VIKTOR_RIOT_ID"),
        "tagline": os.environ.get("ANALYZER_TAGLINE") or os.environ.get("VIKTOR_TAGLINE"),
        "region": os.environ.get("ANALYZER_REGION") or os.environ.get("VIKTOR_REGION"),
        "platform": os.environ.get("ANALYZER_PLATFORM") or os.environ.get("VIKTOR_PLATFORM"),
    }
    data.update({k: v for k, v in env_map.items() if v})
    region_override = overrides.get("region")
    if region_override and not data.get("platform") and not overrides.get("platform"):
        inferred = AppConfig.platform_from_region_input(str(region_override))
        if inferred:
            data["platform"] = inferred
    data.update({k: v for k, v in overrides.items() if v is not None})
    return AppConfig(**data)
