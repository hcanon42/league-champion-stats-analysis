"""Application configuration.

Configuration is resolved in order of precedence:

1. CLI options,
2. environment variables (``RIOT_API_KEY``, ``ANALYZER_*`` / legacy ``VIKTOR_*``),
3. a ``.env`` file in the project root (when a variable is not already set),
4. an optional ``config.toml`` file,
5. built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Final

from champions import (
    build_label,
    champion_slug,
    normalize_role,
    players_group_slug,
    role_display,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

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
RANKED_FLEX_QUEUE_ID: Final[int] = 440
RANKED_QUEUE_IDS: Final[tuple[int, ...]] = (RANKED_SOLO_QUEUE_ID, RANKED_FLEX_QUEUE_ID)
REMAKE_MAX_DURATION_S: Final[int] = 300
# Ranked surrender vote opens at 15:00; shorter surrender endings are not real games.
SURRENDER_VOTE_OPENS_S: Final[int] = 900
GAME_WINDOW_OPTIONS: Final[tuple[int, ...]] = (50, 100)
DEFAULT_GAME_WINDOW: Final[int] = 100
QUEUE_FILTER_OPTIONS: Final[tuple[str, ...]] = ("solo", "flex", "all")
DEFAULT_QUEUE_FILTER: Final[str] = "solo"
QUEUE_LABELS: Final[dict[str, str]] = {
    "solo": "Solo/Duo",
    "flex": "Flex",
    "all": "All ranked",
}
QUEUE_SUBTITLE_LABELS: Final[dict[str, str]] = {
    "solo": "ranked solo queue",
    "flex": "ranked flex queue",
    "all": "ranked",
}


class PlayerIdentity(BaseModel):
    """One tracked Riot account."""

    riot_id: str
    tagline: str

    @property
    def label(self) -> str:
        """Display label (``Name#TAG``)."""
        return f"{self.riot_id}#{self.tagline}"


class AppConfig(BaseModel):
    """Validated runtime configuration for a full analysis run."""

    riot_id: str
    tagline: str
    players: list[PlayerIdentity] = Field(default_factory=list)
    region: str = "europe"
    platform: str | None = None
    api_key: str
    # TODO(security): gemini_api_key later gets embedded directly into generated
    # static HTML (see main.py's chatbot context wiring + templates/report.html) so
    # browser JS can call the Gemini API without a backend. That's a deliberate,
    # temporary tradeoff — move this behind a real backend proxy before reports are
    # ever shared publicly.
    gemini_api_key: str | None = None
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

    @model_validator(mode="after")
    def _default_players(self) -> "AppConfig":
        """Ensure at least the primary player is tracked."""
        if not self.players:
            self.players = [PlayerIdentity(riot_id=self.riot_id, tagline=self.tagline)]
        return self

    @property
    def players_label(self) -> str:
        """Comma-separated display label for all tracked players."""
        return ", ".join(player.label for player in self.players)

    @property
    def reports_group_slug(self) -> str:
        """Filesystem slug for this player or multi-player group."""
        return players_group_slug([(player.riot_id, player.tagline) for player in self.players])

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
        """Directory holding every build report for this player or group."""
        return self.output_dir / "reports" / self.reports_group_slug

    @property
    def report_dir(self) -> Path:
        """Per-player/champion/lane output directory (overwritten on re-run)."""
        return (
            self.output_dir
            / "reports"
            / self.reports_group_slug
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
                "Missing Riot API key. Set RIOT_API_KEY in the environment or a "
                ".env file, or pass --api-key "
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

    @property
    def assets_dir(self) -> Path:
        """Shared champion/rune icons for generated HTML reports."""
        return self.output_dir / "assets"

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


def _load_env_file() -> None:
    """Load ``.env`` from the working directory or project root."""
    paths = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        load_dotenv(path, override=False, encoding="utf-8-sig")


def _missing_api_key_hint() -> str:
    """Build a helpful message when no API key was resolved."""
    env_paths = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    for path in env_paths:
        if path.is_file() and path.stat().st_size == 0:
            return (
                f"Missing Riot API key. {path} exists but is empty — save the file "
                "in your editor (Ctrl+S), set RIOT_API_KEY in the environment, or pass --api-key."
            )
    return (
        "Missing Riot API key. Set RIOT_API_KEY in the environment or a .env file "
        "(get one at https://developer.riotgames.com), or pass --api-key."
    )


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
    _load_env_file()
    data: dict[str, Any] = _read_toml(config_file or Path("config.toml"))
    env_map = {
        "api_key": os.environ.get("RIOT_API_KEY"),
        "riot_id": os.environ.get("ANALYZER_RIOT_ID") or os.environ.get("VIKTOR_RIOT_ID"),
        "tagline": os.environ.get("ANALYZER_TAGLINE") or os.environ.get("VIKTOR_TAGLINE"),
        "region": os.environ.get("ANALYZER_REGION") or os.environ.get("VIKTOR_REGION"),
        "platform": os.environ.get("ANALYZER_PLATFORM") or os.environ.get("VIKTOR_PLATFORM"),
        "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
    }
    data.update({k: v for k, v in env_map.items() if v})
    players_override = overrides.pop("players", None)
    if players_override:
        data["players"] = players_override
        primary = players_override[0]
        data["riot_id"] = primary.riot_id
        data["tagline"] = primary.tagline
    region_override = overrides.get("region")
    if region_override and not data.get("platform") and not overrides.get("platform"):
        inferred = AppConfig.platform_from_region_input(str(region_override))
        if inferred:
            data["platform"] = inferred
    data.update({k: v for k, v in overrides.items() if v is not None})
    if not data.get("api_key"):
        raise ValueError(_missing_api_key_hint())
    return AppConfig(**data)
