# AGENTS.md — AI navigation guide

Champion Stats Analyzer: downloads ranked LoL matches via Riot Match-V5, parses timelines,
runs coaching analytics, and renders interactive HTML dashboards plus CSV/JSON exports.

## Layer map (where to put code)

| Layer | Path | Rule |
|-------|------|------|
| CLI | `src/league_stats/cli/app.py` | Typer commands only — no business logic |
| Core | `src/league_stats/core/` | Config, Pydantic models, champion/role helpers — no I/O |
| Infra | `src/league_stats/infra/` | HTTP, SQLite cache, DDragon assets |
| Ingest | `src/league_stats/ingest/` | Raw JSON → `MatchRecord` |
| Pipeline | `src/league_stats/pipeline/` | Orchestration, frames, bundles, view models |
| Analysis | `src/league_stats/analysis/` | Pure stats on records/DataFrames |
| Presentation | `src/league_stats/presentation/` | HTML, charts, exports, icons |

## Naming glossary

- **role** = Riot `teamPosition` = UI "lane" (`TOP`, `MIDDLE`, `JUNGLE`, …)
- **build** = champion + role pair (e.g. Viktor mid → `viktor_middle`)
- **reports_group_slug** = filesystem slug for one player or a pooled multi-player group

## Add a new metric (recipe)

1. Timeline field? Add `extract_*` in `analysis/<domain>.py` and wire in `ingest/parser.py`.
2. Column on matches/deaths table? Add to `pipeline/frames.py` → `build_analysis_frames()`.
3. Dashboard card? Add label to `presentation/ui_icons.py` `METRIC_ICONS` and a card row in `pipeline/bundles.py`.
4. Coaching tip? Add a rule in `analysis/coach/engine.py`.
5. Chatbot should know? Extend `pipeline/summaries.py` → `build_export_summary()`.

## Icons

- Iconify keys in `presentation/ui_icons.py` → `ICONIFY_ICONS`
- Local PNG assets via `DDragonAssets.ui_icon_href` → `ICON_ASSET_FILES`

## Security

Generated reports may embed `GEMINI_API_KEY` in static HTML for client-side chat.
Do not share reports publicly while a real key is baked in.

## Commands

```bash
uv sync
uv run python main.py analyze --riot-id "Name" --tagline "EUW"
uv run python -m league_stats.cli.app report --riot-id "Name" --tagline "EUW"
uv run pytest
```

## Tests

Synthetic fixtures in `tests/fixtures.py`. One module per area: `tests/test_<module>.py`.
