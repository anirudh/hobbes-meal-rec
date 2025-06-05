"""
core/meal_recommender.py
────────────────────────────────────────────────────────────────────────
Pure-nutrition recommender.

Responsibilities
----------------
1.   `filter_meals()` – drop meals that conflict with
     dietary restrictions, meal_type slot, or cuisine list.
2.   `calculate_meal_scores()` – compute a simple distance score
     between meal macros and *per-day* macro targets.
3.   `recommend_meals()` – convenience wrapper that returns
     the top-k lowest-distance meals.

The class **does NOT** consider user history or nudging – that’s the
job of `core.recommendation.NudgeMealRecommender`.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


class MealRecommender:
    def __init__(self, calc, meals_df: pd.DataFrame) -> None:
        # calc is kept for future extensions (micros, etc.)
        self._meals = meals_df.copy()

    # ─────────────────────────────── filter ───────────────────────── #
    def filter_meals(
        self,
        dietary: List[str],
        meal_type: str,
        cuisines: List[str],
    ) -> pd.DataFrame:
        df = self._meals

        if dietary:
            for d in dietary:
                df = df[~df["dietary_flags"].str.contains(d, case=False, na=False)]

        if meal_type.lower() != "any":
            df = df[df["meal_slot"].str.lower() == meal_type.lower()]

        # only apply cuisine filter if user requested cuisines AND we actually have that column
        if cuisines and "cuisine" in df.columns:
            df = df[df["cuisine"].str.lower().isin([c.lower() for c in cuisines])]

        return df.reset_index(drop=True)

    # ──────────────────────────── scoring ─────────────────────────── #
    def calculate_meal_scores(
        self, df: pd.DataFrame, targets: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Score = Euclidean distance on the four macro keys
                (kcal, protein_g, carbs_g, fat_g).

        Lower score = better nutritional match.
        """
        keys = ["kcal", "protein_g", "carbs_g", "fat_g"]
        if any(k not in df.columns for k in keys):
            missing = [k for k in keys if k not in df.columns]
            raise KeyError(f"Meal DataFrame missing columns: {missing}")

        tgt_vec = np.array([targets[k] for k in keys], dtype=float)

        meal_mat = df[keys].values.astype(float)
        distances = np.linalg.norm(meal_mat - tgt_vec, axis=1)

        return df.assign(score=distances)

    # ──────────────────────────── wrapper ─────────────────────────── #
    def recommend_meals(
        self,
        user_profile: Dict[str, float],
        targets: Dict[str, float],
        dietary: List[str],
        meal_type: str,
        cuisines: List[str],
        k: int = 6,
    ) -> pd.DataFrame:
        filtered = self.filter_meals(dietary, meal_type, cuisines)
        if filtered.empty:
            _LOG.warning("No meals left after filtering – returning empty DataFrame")
            return filtered

        scored = self.calculate_meal_scores(filtered, targets)
        # Return lowest-distance k
        cols_to_drop = ["score"]
        return scored.nsmallest(k, "score").drop(columns=cols_to_drop)
