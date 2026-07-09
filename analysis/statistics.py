"""Advanced statistics: correlations, win-condition analysis, clustering
and a RandomForest early-game win predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from utils import get_logger

FEATURE_COLUMNS: tuple[str, ...] = (
    "deaths_pre20",
    "deaths_pre14",
    "control_wards",
    "first_item_min",
    "cs10",
    "gd10",
    "gd15",
    "xpd10",
    "dpm",
    "vspm",
    "avg_unspent_gold",
    "roams_pre15",
    "lane_priority",
    "solo_deaths",
    "kill_participation",
    "damage_share",
)
EARLY_FEATURES: tuple[str, ...] = (
    "gd10",
    "xpd10",
    "csd10",
    "cs10",
    "deaths_pre14",
    "first_item_min",
    "roams_pre15",
    "lane_priority",
    "gd15",
)
MIN_GAMES_FOR_ML: int = 20
MIN_GAMES_FOR_CLUSTERS: int = 12


@dataclass
class WinCorrelation:
    """Point-biserial correlation of a feature with winning."""

    feature: str
    correlation: float
    p_value: float
    n: int


@dataclass
class ModelResult:
    """Outcome of training the early-game win predictor."""

    trained: bool
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_auc_mean: float | None = None
    cv_auc_std: float | None = None
    n_games: int = 0
    reason: str = ""


class StatisticsEngine:
    """Statistical analysis over the master per-game table."""

    def __init__(self, matches_df: pd.DataFrame, output_dir: Path) -> None:
        """Create the engine.

        Args:
            matches_df: One row per game (from ``MatchRecord.to_row``).
            output_dir: Directory for model artefacts.
        """
        self._df = matches_df
        self._output_dir = output_dir
        self._log = get_logger("statistics")

    # ---------------------------------------------------------- Correlation

    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation matrix over the analysis features + win.

        Returns:
            The correlation matrix (empty when there is no data).
        """
        columns = [c for c in (*FEATURE_COLUMNS, "win") if c in self._df.columns]
        numeric = self._df[columns].apply(pd.to_numeric, errors="coerce")
        usable = numeric.dropna(axis=1, how="all")
        if usable.empty:
            return pd.DataFrame()
        return usable.corr(numeric_only=True).round(3)

    def win_correlations(self) -> list[WinCorrelation]:
        """Point-biserial correlation of every feature with winning.

        Returns:
            Correlations sorted by absolute strength, strongest first.
        """
        results: list[WinCorrelation] = []
        wins = pd.to_numeric(self._df.get("win"), errors="coerce")
        for feature in FEATURE_COLUMNS:
            if feature not in self._df.columns:
                continue
            values = pd.to_numeric(self._df[feature], errors="coerce")
            mask = values.notna() & wins.notna()
            if mask.sum() < 8 or values[mask].nunique() < 2 or wins[mask].nunique() < 2:
                continue
            corr, p_value = scipy_stats.pointbiserialr(wins[mask], values[mask])
            if np.isnan(corr):
                continue
            results.append(
                WinCorrelation(
                    feature=feature,
                    correlation=round(float(corr), 3),
                    p_value=round(float(p_value), 5),
                    n=int(mask.sum()),
                )
            )
        return sorted(results, key=lambda r: abs(r.correlation), reverse=True)

    # ------------------------------------------------------------------- ML

    def train_win_predictor(self) -> ModelResult:
        """Train a RandomForest predicting the win from early-game metrics.

        The model is persisted with :mod:`joblib` next to the exports and its
        quality is estimated with cross-validated ROC-AUC when the sample
        allows it.

        Returns:
            A :class:`ModelResult` with feature importances (or the reason
            training was skipped).
        """
        features = [f for f in EARLY_FEATURES if f in self._df.columns]
        frame = self._df[[*features, "win"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["win"])
        frame[features] = frame[features].fillna(frame[features].median())
        frame = frame.dropna()
        if len(frame) < MIN_GAMES_FOR_ML or frame["win"].nunique() < 2:
            reason = f"need >= {MIN_GAMES_FOR_ML} games with both results, have {len(frame)}"
            self._log.info("Skipping ML training: %s", reason)
            return ModelResult(trained=False, n_games=len(frame), reason=reason)

        X = frame[features].to_numpy()
        y = frame["win"].astype(int).to_numpy()
        model = RandomForestClassifier(
            n_estimators=400, max_depth=5, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        cv_mean: float | None = None
        cv_std: float | None = None
        if len(frame) >= 30:
            folds = min(5, int(np.min(np.bincount(y))))
            if folds >= 2:
                scores = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
        model.fit(X, y)
        joblib.dump(model, self._output_dir / "win_predictor.joblib")
        importance = (
            pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .round({"importance": 4})
            .reset_index(drop=True)
        )
        return ModelResult(
            trained=True,
            feature_importance=importance,
            cv_auc_mean=round(cv_mean, 3) if cv_mean is not None else None,
            cv_auc_std=round(cv_std, 3) if cv_std is not None else None,
            n_games=len(frame),
        )

    # ------------------------------------------------------------ Clustering

    def label_games(self) -> pd.Series:
        """Rule-based archetype label for every game.

        Labels: "Lane stomp win", "Comeback win", "Scaling win", "Clean win",
        "Throw", "One-sided loss", "Close loss".

        Returns:
            A string series aligned with the match table index.
        """

        def label(row: pd.Series) -> str:
            """Classify one game from its result, gold lead and length."""
            gd15 = row.get("gd15")
            gd15 = float(gd15) if pd.notna(gd15) else 0.0
            if row["win"] == 1:
                if gd15 >= 1000:
                    return "Lane stomp win"
                if gd15 <= -750:
                    return "Comeback win"
                if row["duration_min"] >= 32:
                    return "Scaling win"
                return "Clean win"
            if gd15 >= 750:
                return "Throw"
            if gd15 <= -1000:
                return "One-sided loss"
            return "Close loss"

        if self._df.empty:
            return pd.Series(dtype=str)
        return self._df.apply(label, axis=1)

    def cluster_games(self) -> pd.DataFrame:
        """Unsupervised KMeans clustering of games plus a 2-D PCA projection.

        Returns:
            A dataframe with ``match_id``, ``cluster``, ``pca_x``, ``pca_y``,
            ``win`` and the rule-based ``label``; empty when the sample is
            too small.
        """
        features = [
            f for f in ("gd10", "gd15", "deaths", "dpm", "damage_share", "vspm", "duration_min")
            if f in self._df.columns
        ]
        frame = self._df[features].apply(pd.to_numeric, errors="coerce")
        frame = frame.fillna(frame.median())
        if len(frame) < MIN_GAMES_FOR_CLUSTERS or frame.dropna().empty:
            return pd.DataFrame()
        scaled = StandardScaler().fit_transform(frame.to_numpy())
        n_clusters = max(2, min(6, len(frame) // 8))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)
        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
        return pd.DataFrame(
            {
                "match_id": self._df["match_id"].to_numpy(),
                "cluster": clusters,
                "pca_x": coords[:, 0].round(3),
                "pca_y": coords[:, 1].round(3),
                "win": self._df["win"].to_numpy(),
                "label": self.label_games().to_numpy(),
            }
        )

    # -------------------------------------------------------------- Testing

    def winrate_split_test(self, column: str, threshold: float) -> dict[str, Any] | None:
        """Fisher exact test of win rates above/below a feature threshold.

        Args:
            column: Feature column to split on.
            threshold: Split point (``>=`` goes to the "high" group).

        Returns:
            Win rates per group, odds ratio and p-value, or ``None`` when a
            group is empty or the column is missing.
        """
        if column not in self._df.columns:
            return None
        values = pd.to_numeric(self._df[column], errors="coerce")
        wins = pd.to_numeric(self._df["win"], errors="coerce")
        mask = values.notna() & wins.notna()
        high = wins[mask & (values >= threshold)]
        low = wins[mask & (values < threshold)]
        if high.empty or low.empty:
            return None
        table = [
            [int(high.sum()), int(len(high) - high.sum())],
            [int(low.sum()), int(len(low) - low.sum())],
        ]
        odds_ratio, p_value = scipy_stats.fisher_exact(table)
        return {
            "column": column,
            "threshold": threshold,
            "winrate_high": round(float(high.mean()), 3),
            "winrate_low": round(float(low.mean()), 3),
            "n_high": int(len(high)),
            "n_low": int(len(low)),
            "odds_ratio": round(float(odds_ratio), 3),
            "p_value": round(float(p_value), 5),
        }
