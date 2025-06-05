"""
core/recommendation.py
────────────────────────────────────────────────────────────────────────
Tiered meal recommendations that combine:

  • MealRecommender           → nutrition-only score
  • NutritionNudgeAgent       → next-week macro targets
  • Similarity (Gemini embed) → user meal history
  • Gemini LLM (optional)     → extra candidate meals

All public I/O happens through `NudgeMealRecommender.recommend(...)`
which returns   familiar / transition / target   tiers + a per-day split.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Tuple
import numbers

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from core.nutrition_calc import NutritionalCalculator
from core.nutrition_nudge import NutritionNudgeAgent
from services import gemini
from scripts.helpers import extract_clean_json
from .meal_recommender import MealRecommender

_LOG = logging.getLogger(__name__)

FAMILIAR_HISTORY_WEIGHT = 1.25  # boost for user-history meals


class NudgeMealRecommender:
    def __init__(
        self,
        calc: NutritionalCalculator,
        meals_df: pd.DataFrame,
        user_meals: List[Dict[str, Any]] | None = None,
    ) -> None:
        self._calc = calc
        self._base_meals = meals_df
        self._user_meals = user_meals or []
        self._base = MealRecommender(calc, meals_df)

    def set_user_meals(self, meals: List[Dict[str, Any]]) -> None:
        self._user_meals = meals

    async def recommend(
        self,
        user_profile: Dict[str, Any],
        current_targets: Dict[str, float],
        dietary: List[str],
        meal_type: str,
        cuisines: List[str],
        k: int = 6,
        num_days: int = 1,
        **_: Any,
    ) -> Dict[str, Any]:
        if not self._user_meals:
            flat = self._base.recommend_meals(
                user_profile, current_targets, dietary, meal_type, cuisines, k
            )
            _LOG.debug("no history → nutrition-only path (rows=%d)", len(flat))
            return _package(flat, num_days)

        nudge = NutritionNudgeAgent(user_profile, self._user_meals)
        result = await self._pipeline(
            user_profile, current_targets, nudge,
            dietary, meal_type, cuisines, k, num_days
        )
        return result

    async def _pipeline(
        self,
        user_profile: Dict[str, Any],
        cur_targets: Dict[str, float],
        nudge: NutritionNudgeAgent,
        dietary: List[str],
        meal_type: str,
        cuisines: List[str],
        k: int,
        num_days: int,
    ) -> Dict[str, Any]:
        history = _select_history(self._user_meals, meal_type)
        candidates = self._base.filter_meals(dietary, meal_type, cuisines)
        if candidates.empty:
            _LOG.warning("no meals after filtering")
            return _package(pd.DataFrame(), num_days)

        targets = await nudge.next_week_target()
        adj_targets = _merge_targets(cur_targets, targets)

        base_scored = self._base.calculate_meal_scores(candidates, adj_targets)
        gemini_meals = await self._augment_with_gemini(cur_targets, targets, meal_type, dietary, cuisines)

        history_rows = []
        for meal in history:
            history_rows.append({
                "meal_id": random.randint(1_000_000, 9_000_000),
                "meal_name": meal.get("meal_name", ""),
                "meal_slot": meal.get("meal_slot", meal_type),
                "dietary_flags": ", ".join(meal.get("dietary_flags", [])) if isinstance(meal.get("dietary_flags"), list) else str(meal.get("dietary_flags", "")),
                "cuisine": meal.get("cuisine", ""),
                "kcal": float(meal.get("kcal", 0)),
                "protein_g": float(meal.get("protein_g", 0)),
                "carbs_g": float(meal.get("carbs_g", 0)),
                "fat_g": float(meal.get("fat_g", 0)),
                "score": 999.0,
            })
        history_df = pd.DataFrame(history_rows)

        all_candidates = pd.concat([base_scored, gemini_meals, history_df], ignore_index=True)
        all_candidates = all_candidates.drop_duplicates(subset="meal_name")

        all_candidates = await _add_similarity(all_candidates, history)
        all_candidates = _add_tier_scores(all_candidates)

        history_names = {m.get("meal_name", "").lower() for m in history}
        match = all_candidates["meal_name"].str.lower().isin(history_names)
        all_candidates.loc[match, "familiar_score"] *= FAMILIAR_HISTORY_WEIGHT

        fam, tran, targ = _split_tiers(all_candidates, k, num_days)
        days = _split_by_day(fam, tran, targ, k, num_days)
        flat = pd.concat([fam, tran, targ], ignore_index=True)

        return {
            "flat_df": flat,
            "familiar": fam.to_dict("records"),
            "transition": tran.to_dict("records"),
            "target": targ.to_dict("records"),
            "days": days,
            "num_days": num_days,
        }

    async def _augment_with_gemini(
        self,
        baseline: Dict[str, float],
        targets: Dict[str, float],
        meal_type: str,
        dietary: List[str],
        cuisines: List[str],
    ) -> pd.DataFrame:
        prompt = _gen_meal_prompt(baseline, targets, meal_type, dietary, cuisines)
        try:
            raw = gemini.generate(prompt)
            print("[DEBUG] raw Gemini >>>\n", raw)
            generated = extract_clean_json(raw)
            print("[DEBUG] parsed Gemini JSON >>>\n", generated)

            new_rows = []
            for tier in ("familiar", "transition", "target"):
                for meal in generated.get(tier, {}).values():
                    new_rows.append({
                        "meal_id": random.randint(9_000_000, 9_999_999),
                        "meal_slot": meal.get("meal_slot", meal_type),
                        "meal_name": meal.get("meal_name", ""),
                        "dietary_flags": ", ".join(meal.get("dietary_flags", [])) if isinstance(meal.get("dietary_flags", []), list) else str(meal.get("dietary_flags", "")),
                        "cuisine": meal.get("cuisine", ""),
                        "kcal": float(meal.get("kcal", 0)),
                        "protein_g": float(meal.get("protein_g", 0)),
                        "carbs_g": float(meal.get("carbs_g", 0)),
                        "fat_g": float(meal.get("fat_g", 0)),
                        "score": 999.0,
                        "tier_hint": tier
                    })

            return pd.DataFrame(new_rows)

        except Exception as e:
            _LOG.error("Gemini augmentation failed: %s", e)
            return pd.DataFrame()


# ──────────────────────────────── Helpers ────────────────────────────────

def _boost_history_meals(df: pd.DataFrame, history: List[Dict[str, Any]]) -> pd.DataFrame:
    if not history or df.empty:
        return df
    history_names = {m.get("meal_name", "").lower() for m in history}
    match = df["meal_name"].str.lower().isin(history_names)
    df.loc[match, "familiar_score"] = df.loc[match, "familiar_score"] * FAMILIAR_HISTORY_WEIGHT
    return df


def _select_history(hist: List[Dict[str, Any]], meal_type: str) -> List[Dict[str, Any]]:
    slot = None if meal_type.lower() == "any" else meal_type.lower()
    subset = [
        m for m in hist
        if slot is None
        or m.get("meal_slot", "").lower() == slot
        or m.get("meal_type", "").lower() == slot
    ]
    return subset if len(subset) <= 5 else random.sample(subset, 5)


def _merge_targets(cur: Dict[str, float], nxt: Dict[str, float]) -> Dict[str, float]:
    out = cur.copy()
    for k in ("kcal", "protein_g", "carbs_g", "fat_g"):
        if k in nxt:
            out[k] = nxt[k]
    return out


def _gen_meal_prompt(baseline, target, meal_type, dietary, cuisines) -> str:
    return (
        "You're a dietitian AI.\n\n"
        f"Baseline daily macros:\n{json.dumps(_numeric_only(baseline), indent=2)}\n"
        f"Target daily macros:\n{json.dumps(_numeric_only(target), indent=2)}\n\n"
        f"Dietary preferences: {', '.join(dietary)}\n"
        f"Cuisine preferences: {', '.join(cuisines)}\n"
        f"Meal type requested: {meal_type}\n\n"
        "Create 3 disjoint sets of meals (familiar, transition, target) "
        "each with breakfast, lunch, dinner. Structure the output as:\n"
        "{\n"
        '  "familiar": {"breakfast": {...}, "lunch": {...}, "dinner": {...}},\n'
        '  "transition": {...},\n'
        '  "target": {...}\n'
        "}\n\n"
        "Each meal must include:\n"
        "- meal_name\n"
        "- meal_slot (breakfast/lunch/dinner)\n"
        "- kcal\n"
        "- protein_g\n"
        "- carbs_g\n"
        "- fat_g\n"
        "- cuisine (optional)\n"
        "- dietary_flags (optional)"
    )


async def _embed_async(text: str) -> np.ndarray:
    try:
        return gemini.embed(text)
    except Exception:
        return np.zeros((768,), dtype=np.float32)


async def _add_similarity(df: pd.DataFrame,
                          history: List[Dict[str, Any]]) -> pd.DataFrame:
    if df.empty or not history:
        return df.assign(similarity_score=0.0)

    hist_vecs = await asyncio.gather(
        *[_embed_async(m.get("meal_name", "")) for m in history]
    )
    cand_vecs = await asyncio.gather(
        *[_embed_async(row["meal_name"]) for _, row in df.iterrows()]
    )

    hist_mat = np.vstack(hist_vecs).astype(np.float32, copy=False)
    cand_mat = np.vstack(cand_vecs).astype(np.float32, copy=False)

    sims = cosine_similarity(cand_mat, hist_mat).max(axis=1)
    return df.assign(similarity_score=sims)


def _safe_score_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "score" not in df.columns:
        df = df.assign(score=np.nan_to_num(df.get("similarity_score", 0), nan=1.0))
    return df


def _add_tier_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = _safe_score_df(df)
    if df.empty:
        return df
    rng = df["score"].max() - df["score"].min()
    norm = 1 - (df["score"] - df["score"].min()) / rng if rng else 1.0
    return df.assign(
        norm_score=norm,
        familiar_score=0.7 * df.get("similarity_score", 0) + 0.3 * norm,
        transition_score=0.3 * df.get("similarity_score", 0) + 0.7 * norm,
        target_score=0.1 * df.get("similarity_score", 0) + 0.9 * norm,
    )


def _split_tiers(df: pd.DataFrame, k: int, days: int) -> Tuple[pd.DataFrame, ...]:
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    per_day = max(3, k // 3)
    fam_n = max(1, per_day // 3) * days
    tran_n = max(1, per_day // 3) * days
    targ_n = per_day * days - fam_n - tran_n

    fam = df.nlargest(fam_n, "familiar_score")
    tran = df[~df["meal_id"].isin(fam["meal_id"])].nlargest(tran_n, "transition_score")
    targ = df[~df["meal_id"].isin(fam["meal_id"].tolist() + tran["meal_id"].tolist())
              ].nlargest(targ_n, "target_score")

    drops = [c for c in (
        "similarity_score", "norm_score", "familiar_score",
        "transition_score", "target_score"
    ) if c in fam.columns]
    return fam.drop(columns=drops), tran.drop(columns=drops), targ.drop(columns=drops)


def _split_by_day(fam: pd.DataFrame, tran: pd.DataFrame, targ: pd.DataFrame,
                  k: int, days: int) -> Dict[str, Any]:
    per_day = max(3, k // 3)
    fam_d, tran_d = max(1, per_day // 3), max(1, per_day // 3)
    targ_d = per_day - fam_d - tran_d

    out = {}
    for d in range(1, days + 1):
        i = d - 1
        out[f"day_{d}"] = {
            "breakfast": fam.iloc[i * fam_d: (i + 1) * fam_d].to_dict("records"),
            "lunch": tran.iloc[i * tran_d: (i + 1) * tran_d].to_dict("records"),
            "dinner": targ.iloc[i * targ_d: (i + 1) * targ_d].to_dict("records"),
        }
    return out


def _package(flat: pd.DataFrame, days: int) -> Dict[str, Any]:
    return {
        "flat_df": flat,
        "familiar": flat.to_dict("records"),
        "transition": [],
        "target": [],
        "days": {},
        "num_days": days,
    }


def _numeric_only(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, numbers.Number):
            out[k] = float(v)
    return out
