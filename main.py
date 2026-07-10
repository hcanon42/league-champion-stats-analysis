"""Backward-compatible CLI shim. Prefer ``league-champion-stats`` or ``python -m league_stats.cli.app``."""

from league_stats.cli.app import app

if __name__ == "__main__":
    app()
