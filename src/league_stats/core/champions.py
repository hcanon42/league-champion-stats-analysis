"""Champion and lane normalization plus Data Dragon name resolution."""

from __future__ import annotations

from typing import Final

VALID_ROLES: Final[frozenset[str]] = frozenset(
    {"MIDDLE", "TOP", "JUNGLE", "BOTTOM", "UTILITY"}
)

ROLE_ALIASES: Final[dict[str, str]] = {
    "mid": "MIDDLE",
    "middle": "MIDDLE",
    "top": "TOP",
    "jungle": "JUNGLE",
    "jg": "JUNGLE",
    "jgl": "JUNGLE",
    "bot": "BOTTOM",
    "bottom": "BOTTOM",
    "adc": "BOTTOM",
    "support": "UTILITY",
    "sup": "UTILITY",
    "utility": "UTILITY",
    "util": "UTILITY",
}

ROLE_DISPLAY: Final[dict[str, str]] = {
    "MIDDLE": "mid",
    "TOP": "top",
    "JUNGLE": "jungle",
    "BOTTOM": "bot",
    "UTILITY": "support",
}

# Riot match-v5 / Data Dragon ids that differ from the in-game display name.
CHAMPION_DISPLAY_NAMES: Final[dict[str, str]] = {
    "MonkeyKing": "Wukong",
    "DrMundo": "Dr. Mundo",
}


def normalize_role(value: str) -> str:
    """Map a user-facing lane name to Riot's ``teamPosition`` value.

    Args:
        value: Lane string (``mid``, ``MIDDLE``, ``support``, ...).

    Returns:
        One of ``MIDDLE``, ``TOP``, ``JUNGLE``, ``BOTTOM``, ``UTILITY``.

    Raises:
        ValueError: When the lane cannot be recognised.
    """
    raw = str(value).strip()
    upper = raw.upper()
    if upper in VALID_ROLES:
        return upper
    key = raw.lower().replace(" ", "_").replace("-", "_")
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    raise ValueError(
        f"Unknown lane {value!r}. Use mid, top, jungle, bot, or support "
        f"(Riot values: {sorted(VALID_ROLES)})."
    )


def role_display(role: str) -> str:
    """Short display label for a Riot team position.

    Args:
        role: Normalised role (``MIDDLE``, ...).

    Returns:
        Human-readable lane name (``mid``, ``top``, ...).
    """
    return ROLE_DISPLAY.get(role.upper(), role.lower())


def champion_display_name(riot_id: str) -> str:
    """Map a Riot champion id to the player-facing name.

    Args:
        riot_id: Official champion id from match-v5 payloads.

    Returns:
        The display name (e.g. ``Wukong`` instead of ``MonkeyKing``).
    """
    return CHAMPION_DISPLAY_NAMES.get(str(riot_id), str(riot_id))


def build_label(champion: str, role: str) -> str:
    """Display label for a champion + lane pair (e.g. ``Viktor mid``).

    Args:
        champion: Riot champion id (``Viktor``, ``Ahri``, ...).
        role: Normalised team position.

    Returns:
        A short label for reports and logs.
    """
    return f"{champion_display_name(champion)} {role_display(role)}"


def parse_riot_id(value: str) -> tuple[str, str]:
    """Parse a Riot ID in ``Name#Tag`` form.

    Args:
        value: Combined Riot ID (e.g. ``Faker#KR1``).

    Returns:
        ``(game_name, tagline)`` without the ``#`` separator.

    Raises:
        ValueError: When the value is not a valid ``Name#Tag`` string.
    """
    if "#" not in value:
        raise ValueError(f"Expected Riot ID as 'Name#Tag', got {value!r}")
    name, tag = value.rsplit("#", 1)
    name, tag = name.strip(), tag.strip()
    if not name or not tag:
        raise ValueError(f"Expected Riot ID as 'Name#Tag', got {value!r}")
    return name, tag


def player_slug(riot_id: str, tagline: str) -> str:
    """Filesystem slug for a Riot ID (``hugros_euw``).

    Args:
        riot_id: Game name portion of the Riot ID.
        tagline: Tagline without ``#``.

    Returns:
        Lowercase ``{riot_id}_{tagline}`` with unsafe characters replaced.
    """

    def _part(value: str) -> str:
        cleaned = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in str(value).strip().lower()
        )
        return cleaned.strip("_") or "player"

    return f"{_part(riot_id)}_{_part(tagline)}"


def players_group_slug(players: list[tuple[str, str]]) -> str:
    """Filesystem slug for one or more tracked players.

    Args:
        players: ``(riot_id, tagline)`` pairs.

    Returns:
        A single-player slug, or sorted multi-player slugs joined with ``__``.
    """
    if not players:
        return "player"
    if len(players) == 1:
        riot_id, tagline = players[0]
        return player_slug(riot_id, tagline)
    return "__".join(sorted(player_slug(riot_id, tagline) for riot_id, tagline in players))


def champion_slug(champion: str, role: str) -> str:
    """Filesystem slug for benchmark lookup (``ahri_middle``).

    Args:
        champion: Riot champion id.
        role: Normalised team position.

    Returns:
        Lowercase ``{champion}_{role}`` slug.
    """
    return f"{champion.lower()}_{role.lower()}"


def resolve_champion_name(user_input: str, catalog: dict[str, str]) -> str:
    """Resolve user input to the official Riot champion id.

    Matching is case- and space-insensitive against Data Dragon ids and
    display names.

    Args:
        user_input: Raw CLI value (``ahri``, ``Miss Fortune``, ``LeeSin``).
        catalog: Mapping of normalised lookup keys to official ids.

    Returns:
        The official champion id used in match-v5 payloads.

    Raises:
        ValueError: When no champion matches.
    """
    key = _champion_key(user_input)
    if key in catalog:
        return catalog[key]
    raise ValueError(
        f"Unknown champion {user_input!r}. Use the Riot id "
        f"(e.g. Ahri, LeeSin, MissFortune). "
        f"Run with --verbose after a successful fetch to see the catalog size."
    )


def _champion_key(value: str) -> str:
    """Normalise a champion string for catalog lookup."""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("'", "")
        .replace(".", "")
        .replace("-", "")
    )


def build_champion_catalog(ddragon_champions: dict[str, dict]) -> dict[str, str]:
    """Build a lookup table from Data Dragon ``champion.json`` data.

    Args:
        ddragon_champions: Raw ``data`` dict from Data Dragon.

    Returns:
        Mapping of normalised keys to official champion ids.
    """
    catalog: dict[str, str] = {}
    for champion_id, data in ddragon_champions.items():
        official = str(data.get("id", champion_id))
        display_name = str(data.get("name", official))
        catalog[_champion_key(official)] = official
        catalog[_champion_key(display_name)] = official
        if " " in official:
            catalog[_champion_key(official.replace(" ", ""))] = official
    for official, display_name in CHAMPION_DISPLAY_NAMES.items():
        catalog[_champion_key(display_name)] = official
    return catalog
