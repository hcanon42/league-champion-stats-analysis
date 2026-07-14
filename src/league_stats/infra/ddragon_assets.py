"""Download and serve local Data Dragon champion and rune icons for HTML reports."""

from __future__ import annotations

import base64
import json
import os
import struct
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from league_stats.core.champions import VALID_ROLES, champion_display_name
from league_stats.core.config import AppConfig
from league_stats.ingest.parser import PERK_NAMES
from league_stats.infra.riot_api import DDRAGON_BASE
from league_stats.utils import get_logger

# Keystones live in the 8000+ perk id range (stat shards are 5000+).
KEYSTONE_ID_MIN: int = 8000
KEYSTONE_ID_MAX: int = 10_000

COMMUNITY_DRAGON_BASE = "https://raw.communitydragon.org/latest"
ROLE_ICON_FILES: dict[str, str] = {
    "TOP": "icon-position-top.png",
    "JUNGLE": "icon-position-jungle.png",
    "MIDDLE": "icon-position-middle.png",
    "BOTTOM": "icon-position-bottom.png",
    "UTILITY": "icon-position-utility.png",
}
ROLE_ICON_URL = (
    "{base}/plugins/rcp-fe-lol-clash/global/default/assets/images/"
    "position-selector/positions/{filename}"
)
SCOREBOARD_BASE = f"{COMMUNITY_DRAGON_BASE}/game/assets/ux/scoreboard"
MATCH_HISTORY_BASE = (
    f"{COMMUNITY_DRAGON_BASE}/plugins/rcp-fe-lol-match-history/global/default"
)
MINIMAP_ICONS_BASE = f"{COMMUNITY_DRAGON_BASE}/game/assets/ux/minimap/icons"
UI_ICON_URLS: dict[str, str] = {
    "minions.png": (
        "{base}/plugins/rcp-fe-lol-match-history/global/default/icon_minions.png"
    ),
    "tower.png": f"{MATCH_HISTORY_BASE}/tower_building_blue.png",
}
OBJECTIVE_ICON_SOURCES: dict[str, str] = {
    "dragon.png": f"{SCOREBOARD_BASE}/_dragon.png",
    "elder.png": f"{SCOREBOARD_BASE}/_elderdrake.png",
    "baron.png": f"{SCOREBOARD_BASE}/_baronnashor.png",
    "herald.png": f"{SCOREBOARD_BASE}/_riftherald.png",
    "grubs.png": f"{MINIMAP_ICONS_BASE}/grub.png",
    "tower.png": f"{MATCH_HISTORY_BASE}/tower_building_blue.png",
    "inhibitor.png": f"{MATCH_HISTORY_BASE}/inhibitor_building_blue.png",
    "nexus.png": f"{MATCH_HISTORY_BASE}/nexus_building_blue.png",
    "chemtech_drake.png": f"{SCOREBOARD_BASE}/_chemtechdrake.png",
    "cloud_drake.png": f"{SCOREBOARD_BASE}/_clouddrake.png",
    "hextech_drake.png": f"{SCOREBOARD_BASE}/_hextechdrake.png",
    "infernal_drake.png": f"{SCOREBOARD_BASE}/_infernaldrake.png",
    "mountain_drake.png": f"{SCOREBOARD_BASE}/_mountaindrake.png",
    "ocean_drake.png": f"{SCOREBOARD_BASE}/_oceandrake.png",
}
OBJECTIVE_KIND_FILES: dict[str, str] = {
    "dragon": "dragon.png",
    "elder": "elder.png",
    "baron": "baron.png",
    "herald": "herald.png",
    "grubs": "grubs.png",
}


class DDragonAssets:
    """Local cache of champion and keystone icons under ``output/assets/``."""

    def __init__(
        self,
        config: AppConfig,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._assets_root = config.output_dir / "assets"
        self._champions_dir = self._assets_root / "champions"
        self._runes_dir = self._assets_root / "runes"
        self._items_dir = self._assets_root / "items"
        self._roles_dir = self._assets_root / "roles"
        self._ui_dir = self._assets_root / "ui"
        self._objectives_dir = self._assets_root / "objectives"
        self._manifest_path = config.cache_dir / "static" / "manifest.json"
        self._session = session or requests.Session()
        self._log = get_logger("ddragon_assets")
        self._version: str | None = None
        self._perk_name_to_id: dict[str, int] = {name: perk_id for perk_id, name in PERK_NAMES.items()}
        self._item_name_to_id: dict[str, int] = {}

    @property
    def version(self) -> str | None:
        """Data Dragon patch version used for the cached icons."""
        return self._version

    @property
    def assets_root(self) -> Path:
        """Root directory containing ``champions/``, ``runes/``, ``items/`` and ``roles/``."""
        return self._assets_root

    def _roles_cached(self) -> bool:
        return self._roles_dir.is_dir() and all(
            (self._roles_dir / f"{role}.png").is_file() for role in ROLE_ICON_FILES
        )

    def ensure_downloaded(self, *, force: bool = False) -> str:
        """Download champion and keystone icons when missing or ``force`` is set.

        Returns:
            The Data Dragon patch version used for the download.
        """
        self._assets_root.mkdir(parents=True, exist_ok=True)
        self._champions_dir.mkdir(parents=True, exist_ok=True)
        self._runes_dir.mkdir(parents=True, exist_ok=True)
        self._items_dir.mkdir(parents=True, exist_ok=True)
        self._roles_dir.mkdir(parents=True, exist_ok=True)
        self._ui_dir.mkdir(parents=True, exist_ok=True)
        self._objectives_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_role_icons(force=force)
        self._ensure_ui_icons(force=force)
        self._ensure_objective_icons(force=force)

        manifest = self._read_manifest()
        if (
            not force
            and manifest.get("version")
            and self._champions_dir.is_dir()
            and any(self._champions_dir.glob("*.png"))
            and self._runes_dir.is_dir()
            and any(self._runes_dir.glob("*.png"))
            and self._items_dir.is_dir()
            and any(self._items_dir.glob("*.png"))
        ):
            self._version = str(manifest["version"])
            self._item_name_to_id = {
                str(name): int(item_id)
                for name, item_id in manifest.get("item_names", {}).items()
            }
            self._log.info("Using cached Data Dragon assets for patch %s", self._version)
            return self._version

        try:
            version = self._fetch_latest_version()
        except requests.RequestException as exc:
            self._log.warning("Could not reach Data Dragon: %s", exc)
            cached_version = str(manifest.get("version", ""))
            if cached_version:
                self._version = cached_version
                return cached_version
            return ""

        if (
            not force
            and manifest.get("version") == version
            and self._champions_dir.is_dir()
            and any(self._champions_dir.glob("*.png"))
            and self._runes_dir.is_dir()
            and any(self._runes_dir.glob("*.png"))
            and self._items_dir.is_dir()
            and any(self._items_dir.glob("*.png"))
        ):
            self._version = version
            self._item_name_to_id = {
                str(name): int(item_id)
                for name, item_id in manifest.get("item_names", {}).items()
            }
            self._log.info("Using cached Data Dragon assets for patch %s", version)
            return version

        try:
            champions = self._fetch_champions(version)
            rune_icons = self._fetch_keystone_icons(version)
            items = self._fetch_items(version)
        except requests.RequestException as exc:
            self._log.warning("Could not download Data Dragon assets: %s", exc)
            cached_version = str(manifest.get("version", ""))
            if cached_version and self.champion_icon_path("Ahri"):
                self._version = cached_version
                return cached_version
            return version

        self._download_champion_icons(version, champions, force=force)
        self._download_rune_icons(version, rune_icons, force=force)
        self._download_item_icons(version, items, force=force)

        self._item_name_to_id = {
            str(data.get("name", "")): int(item_id)
            for item_id, data in items.items()
            if data.get("name")
        }
        self._version = version
        manifest = {
            "version": version,
            "champions": len(champions),
            "keystones": len(rune_icons),
            "items": len(items),
            "roles": len(ROLE_ICON_FILES),
            "item_names": self._item_name_to_id,
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._log.info(
            "Downloaded %d champion icons, %d keystone icons, %d item icons and %d role icons (patch %s)",
            len(champions),
            len(rune_icons),
            len(items),
            len(ROLE_ICON_FILES),
            version,
        )
        return version

    def champion_icon_path(self, champion: str) -> Path | None:
        """Return the on-disk champion icon path when it exists."""
        path = self._champions_dir / f"{champion}.png"
        return path if path.is_file() else None

    def keystone_icon_path(self, keystone_name: str) -> Path | None:
        """Return the on-disk keystone icon path when it exists."""
        perk_id = self._perk_name_to_id.get(keystone_name)
        if perk_id is None:
            return None
        path = self._runes_dir / f"{perk_id}.png"
        return path if path.is_file() else None

    def item_icon_path(self, item_id: int) -> Path | None:
        """Return the on-disk item icon path when it exists."""
        path = self._items_dir / f"{item_id}.png"
        return path if path.is_file() else None

    def role_icon_path(self, role: str) -> Path | None:
        """Return the on-disk lane icon path when it exists."""
        normalized = role.strip().upper()
        if normalized not in VALID_ROLES:
            return None
        path = self._roles_dir / f"{normalized}.png"
        return path if path.is_file() else None

    def role_href(self, role: str, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to a lane icon."""
        path = self.role_icon_path(role)
        if path is None:
            return None
        return _relative_href(from_dir, path)

    def role_chart_source(self, role: str) -> str | None:
        """Base64 data URI for embedding a lane icon in Plotly charts."""
        return path_to_data_uri(self.role_icon_path(role))

    def ui_icon_href(self, filename: str, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to a cached UI icon."""
        path = self._ui_dir / filename
        if not path.is_file():
            return None
        return _relative_href(from_dir, path)

    def objective_icon_path(self, kind: str) -> Path | None:
        """Return the on-disk scoreboard icon for an objective kind."""
        filename = OBJECTIVE_KIND_FILES.get(kind.strip().lower())
        if filename is None:
            return None
        path = self._objectives_dir / filename
        return path if path.is_file() else None

    def objective_href(self, kind: str, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to an objective scoreboard icon."""
        path = self.objective_icon_path(kind)
        if path is None:
            return None
        return _relative_href(from_dir, path)

    def enrich_objective_rows(self, rows: list[dict[str, Any]], *, from_dir: Path) -> list[dict[str, Any]]:
        """Attach ``objective_icon`` hrefs to objective table rows."""
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["objective_icon"] = self.objective_href(str(row.get("kind", "")), from_dir=from_dir)
            enriched.append(item)
        return enriched

    def item_href(self, item_id: int, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to an item icon."""
        path = self.item_icon_path(item_id)
        if path is None:
            return None
        return _relative_href(from_dir, path)

    def item_href_by_name(self, item_name: str, *, from_dir: Path) -> str | None:
        """Resolve an item display name to a cached icon href."""
        item_id = self._item_name_to_id.get(item_name)
        if item_id is None:
            return None
        return self.item_href(item_id, from_dir=from_dir)

    def champion_chart_source(self, champion: str) -> str | None:
        """Base64 data URI for embedding a champion icon in Plotly charts."""
        return path_to_data_uri(self.champion_icon_path(champion))

    def item_chart_source(self, item_name: str) -> str | None:
        """Base64 data URI for embedding an item icon in Plotly charts."""
        item_id = self._item_name_to_id.get(item_name)
        if item_id is None:
            return None
        return path_to_data_uri(self.item_icon_path(item_id))

    def keystone_chart_source(self, keystone_name: str) -> str | None:
        """Base64 data URI for embedding a keystone icon in Plotly charts."""
        return path_to_data_uri(self.keystone_icon_path(keystone_name))

    def champion_href(self, champion: str, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to a champion icon."""
        path = self.champion_icon_path(champion)
        if path is None:
            return None
        return _relative_href(from_dir, path)

    def keystone_href(self, keystone_name: str, *, from_dir: Path) -> str | None:
        """Relative URL from an HTML directory to a keystone icon."""
        path = self.keystone_icon_path(keystone_name)
        if path is None:
            return None
        return _relative_href(from_dir, path)

    def enrich_rune_rows(self, rows: list[dict[str, Any]], *, from_dir: Path) -> list[dict[str, Any]]:
        """Attach ``keystone_icon`` hrefs to rune table rows."""
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["keystone_icon"] = self.keystone_href(str(row.get("keystone", "")), from_dir=from_dir)
            enriched.append(item)
        return enriched

    def enrich_matchup_rows(self, rows: list[dict[str, Any]], *, from_dir: Path) -> list[dict[str, Any]]:
        """Attach ``opponent_icon`` hrefs to matchup table rows."""
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            riot_id = str(row.get("opponent", ""))
            item["opponent_icon"] = self.champion_href(riot_id, from_dir=from_dir)
            item["opponent"] = champion_display_name(riot_id)
            enriched.append(item)
        return enriched

    def enrich_build_path_rows(self, rows: list[dict[str, Any]], *, from_dir: Path) -> list[dict[str, Any]]:
        """Attach item icon hrefs to two-item core table rows."""
        enriched: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["first_item_icon"] = self.item_href_by_name(str(row.get("first_item", "")), from_dir=from_dir)
            item["second_item_icon"] = self.item_href_by_name(str(row.get("second_item", "")), from_dir=from_dir)
            enriched.append(item)
        return enriched

    def _read_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.is_file():
            return {}
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _fetch_latest_version(self) -> str:
        response = self._session.get(f"{DDRAGON_BASE}/api/versions.json", timeout=15)
        response.raise_for_status()
        return str(response.json()[0])

    def _fetch_champions(self, version: str) -> dict[str, dict[str, Any]]:
        response = self._session.get(
            f"{DDRAGON_BASE}/cdn/{version}/data/en_US/champion.json",
            timeout=15,
        )
        response.raise_for_status()
        return dict(response.json()["data"])

    def _fetch_keystone_icons(self, version: str) -> dict[int, str]:
        response = self._session.get(
            f"{DDRAGON_BASE}/cdn/{version}/data/en_US/runesReforged.json",
            timeout=15,
        )
        response.raise_for_status()
        icons: dict[int, str] = {}
        for style in response.json():
            slots = style.get("slots") or []
            if not slots:
                continue
            for rune in slots[0].get("runes") or []:
                rune_id = int(rune["id"])
                if KEYSTONE_ID_MIN <= rune_id < KEYSTONE_ID_MAX:
                    icons[rune_id] = str(rune["icon"])
        return icons

    def _fetch_items(self, version: str) -> dict[int, dict[str, Any]]:
        response = self._session.get(
            f"{DDRAGON_BASE}/cdn/{version}/data/en_US/item.json",
            timeout=15,
        )
        response.raise_for_status()
        return {int(item_id): data for item_id, data in response.json()["data"].items()}

    def _download_champion_icons(
        self,
        version: str,
        champions: dict[str, dict[str, Any]],
        *,
        force: bool,
    ) -> None:
        for data in tqdm(champions.values(), desc="Champion icons", unit="icon"):
            champion_id = str(data.get("id", ""))
            if not champion_id:
                continue
            destination = self._champions_dir / f"{champion_id}.png"
            if destination.is_file() and not force:
                continue
            url = f"{DDRAGON_BASE}/cdn/{version}/img/champion/{champion_id}.png"
            self._download_binary(url, destination)

    def _download_rune_icons(
        self,
        version: str,
        rune_icons: dict[int, str],
        *,
        force: bool,
    ) -> None:
        for perk_id, icon_path in tqdm(rune_icons.items(), desc="Keystone icons", unit="icon"):
            destination = self._runes_dir / f"{perk_id}.png"
            if destination.is_file() and not force:
                continue
            url = f"{DDRAGON_BASE}/cdn/img/{icon_path}"
            self._download_binary(url, destination)

    def _download_item_icons(
        self,
        version: str,
        items: dict[int, dict[str, Any]],
        *,
        force: bool,
    ) -> None:
        for item_id, data in tqdm(items.items(), desc="Item icons", unit="icon"):
            image = data.get("image") or {}
            filename = str(image.get("full", ""))
            if not filename:
                continue
            destination = self._items_dir / f"{item_id}.png"
            if destination.is_file() and not force:
                continue
            url = f"{DDRAGON_BASE}/cdn/{version}/img/item/{filename}"
            self._download_binary(url, destination)

    def _ensure_role_icons(self, *, force: bool = False) -> None:
        if self._roles_cached() and not force:
            return
        self._download_role_icons(force=force)

    def _download_role_icons(self, *, force: bool) -> None:
        for role, filename in ROLE_ICON_FILES.items():
            destination = self._roles_dir / f"{role}.png"
            if destination.is_file() and not force:
                continue
            url = ROLE_ICON_URL.format(base=COMMUNITY_DRAGON_BASE, filename=filename)
            self._download_binary(url, destination)

    def _ensure_ui_icons(self, *, force: bool = False) -> None:
        destination = self._ui_dir / "minions.png"
        source = self._ui_dir / "_minions_source.png"
        if force or not source.is_file():
            url = UI_ICON_URLS["minions.png"].format(base=COMMUNITY_DRAGON_BASE)
            self._download_binary(url, source)
        if source.is_file() and (
            force or not destination.is_file() or _needs_minion_crop(source, destination)
        ):
            _crop_top_half_png(source, destination)
        tower_destination = self._ui_dir / "tower.png"
        if force or not tower_destination.is_file():
            self._download_binary(UI_ICON_URLS["tower.png"], tower_destination)

    def _objectives_cached(self) -> bool:
        return all(
            (self._objectives_dir / filename).is_file()
            for filename in OBJECTIVE_KIND_FILES.values()
        )

    def _ensure_objective_icons(self, *, force: bool = False) -> None:
        if self._objectives_cached() and not force and not _needs_grub_refresh(self._objectives_dir):
            return
        for filename, url in OBJECTIVE_ICON_SOURCES.items():
            destination = self._objectives_dir / filename
            if destination.is_file() and not force and not (
                filename == "grubs.png" and _needs_grub_refresh(self._objectives_dir)
            ):
                continue
            self._download_binary(url, destination)

    def _download_binary(self, url: str, destination: Path) -> None:
        try:
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
            destination.write_bytes(response.content)
        except requests.RequestException as exc:
            self._log.warning("Failed to download %s: %s", url, exc)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        width, height = struct.unpack(">II", path.read_bytes()[16:24])
        return width, height
    except (OSError, struct.error):
        return None


def _needs_grub_refresh(objectives_dir: Path) -> bool:
    """Detect the legacy match-history grub sprite cached as the objective icon."""
    path = objectives_dir / "grubs.png"
    if not path.is_file():
        return False
    return path.stat().st_size > 10_000


def _needs_minion_crop(source: Path, destination: Path) -> bool:
    """Detect the uncropped stacked CS sprite (two silhouettes)."""
    source_size = _png_dimensions(source)
    dest_size = _png_dimensions(destination)
    if source_size is None or dest_size is None:
        return True
    _, source_height = source_size
    _, dest_height = dest_size
    return dest_height > (source_height // 2) + 1


def _crop_top_half_png(source: Path, destination: Path) -> None:
    """Keep a single minion silhouette from the stacked match-history CS sprite."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    image = mpimg.imread(source)
    height = image.shape[0]
    cropped = image[: height // 2, ...]
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(destination, cropped)


def path_to_data_uri(path: Path | None) -> str | None:
    """Encode a local PNG as a data URI for Plotly ``layout.images``."""
    if path is None or not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _relative_href(from_dir: Path, asset_path: Path) -> str:
    """Build a relative href from an HTML directory to an asset file."""
    return Path(os.path.relpath(asset_path.resolve(), from_dir.resolve())).as_posix()


def icon_cell(name: str, icon_href: str | None) -> str:
    """Render a table cell with an optional icon beside a label."""
    if icon_href:
        return (
            f'<span class="icon-cell">'
            f'<img src="{icon_href}" alt="" class="game-icon" loading="lazy">'
            f"<span>{name}</span></span>"
        )
    return name
