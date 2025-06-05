"""
core/recommendation.py
────────────────────────────────────────────────────────────────────────
Tiered meal recommendations that combine:

  • base MealRecommender  → nutrition-only score
  • NutritionNudgeAgent   → “next-week” macro targets
  • Similarity feature    → user meal history
  • Gemini LLM-based fallback with error logging and embeddings

Outputs three tiers (familiar / transition / target) plus an even
per-day split that UI clients can consume directly.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import json
import re


from core.nutrition_calc import NutritionalCalculator
from core.nutrition_nudge import NutritionNudgeAgent
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from .meal_recommender import MealRecommender
from services.gemini import generate,embed

_LOG = logging.getLogger(__name__)


class NudgeMealRecommender:
    def __init__(
        self,
        calc: NutritionalCalculator,
        meals_df: pd.DataFrame,
        user_meals: List[dict[str, Any]] | None = None,
    ) -> None:
        self._calc = calc
        self._base_meals = meals_df
        self._user_meals = user_meals or []
        self._base = MealRecommender(calc, meals_df)

    def set_user_meals(self, meals: List[dict[str, Any]]) -> None:
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
        db: AsyncSession = None,
    ) -> Dict[str, Any]:
        if not self._user_meals:
            flat = self._base.recommend_meals(
                user_profile, current_targets, dietary, meal_type, cuisines, k
            )
            return {
                "flat_df": flat,
                "familiar": flat.to_dict("records"),
                "transition": [],
                "target": [],
                "days": {},
                "num_days": 0,
            }

        nudge = NutritionNudgeAgent(user_profile, self._user_meals)
        return await self._nudge_pipeline(
            user_profile,
            current_targets,
            nudge,
            dietary,
            meal_type,
            cuisines,
            k,
            num_days,
            db,
        )

    async def _nudge_pipeline(
            self,
            user_profile: Dict[str, Any],
            current_targets: Dict[str, float],
            nudge: NutritionNudgeAgent,
            dietary: List[str],
            meal_type: str,
            cuisines: List[str],
            k: int,
            num_days: int,
            db: AsyncSession
        ) -> Dict[str, Any]:
            history = _select_history(self._user_meals, meal_type)
            candidates = self._base.filter_meals(dietary, meal_type, cuisines)
            if candidates.empty:
                _LOG.warning("No meals after filtering step")
                return {"flat_df": pd.DataFrame(), "familiar": [], "transition": [], "target": []}

            print("[DEBUG] Calling next_week_target() via NutritionNudgeAgent")
            next_week = await nudge.next_week_target()
            print("[DEBUG] Received next_week targets:", next_week)

            adj_targets = await _merge_next_week_targets(current_targets, next_week)
            scored = self._base.calculate_meal_scores(candidates, adj_targets)

            prompt = ""  # Safe default initialization
            try:
                # Remove any unserializable items (e.g., SQLAlchemy InstanceState)
                serializable_targets = {
                    k: float(v) for k, v in current_targets.items() if isinstance(v, (int, float))
                }
                serializable_next_week = {
                    k: float(v) for k, v in adj_targets.items() if isinstance(v, (int, float))
                }

                prompt = f"""
                You are a helpful nutrition assistant.

                The user is currently eating meals with this baseline nutrition:
                {json.dumps(serializable_targets, indent=2)}

                Their next-week target is:
                {json.dumps(serializable_next_week, indent=2)}

                Select appropriate meals from the list below. Choose:
                1. Familiar meals — close to the baseline.
                2. Transition meals — between baseline and target.
                3. Target meals — close to the target nutrition.

                You MUST return a JSON object like:
                {{
                    "familiar": {{
                        "breakfast": <meal dict>,
                        "lunch": <meal dict>,
                        "dinner": <meal dict>
                    }},
                    "transition": {{
                        "breakfast": <meal dict>,
                        "lunch": <meal dict>,
                        "dinner": <meal dict>
                    }},
                    "target": {{
                        "breakfast": <meal dict>,
                        "lunch": <meal dict>,
                        "dinner": <meal dict>
                    }}
                }}

                Choose only from these candidate meals:
                {json.dumps(candidates.to_dict(orient='records'), indent=2)}
                """

                try:
                    generated_text = generate(prompt)
                    print("[DEBUG] Raw Gemini response before cleaning:\n", generated_text)

                    generated_meals = extract_clean_json(generated_text)
                    print("[DEBUG] Cleaned Gemini JSON object:\n", generated_meals)

                except Exception as e:
                    _LOG.error("Gemini generation failed: %s", e)
                    generated_meals = {"familiar": {}, "transition": {}, "target": {}}

                for tier in ["familiar", "transition", "target"]:
                    for m in generated_meals.get(tier, {}).values():
                        try:
                            if m:
                                scored.loc[len(scored)] = {
                                    "meal_id": m.get("meal_id", 900000 + random.randint(1, 99999)),
                                    "meal_slot": m.get("meal_slot", "dinner"),
                                    "meal_name": m.get("meal_name", ""),
                                    "dietary_flags": m.get("dietary_flags", ""),
                                    "cuisine": m.get("cuisine", ""),
                                    **{key: m.get(key, 0.0) for key in adj_targets.keys()},
                                    "score": 999.0
                                }
                        except Exception as est:
                            _LOG.warning(f"Failed to append generated meal in tier '{tier}': {est}")

            except Exception as e:
                user_id = user_profile.get("user_id")
                if user_id is not None:
                    await gemini.log_failure_to_db(
                        db=db,
                        user_id=user_id,
                        stage="gemini_generate",
                        error=str(e),
                        raw_input=prompt,
                        raw_output=""
                    )
                else:
                    _LOG.error("Skipping Gemini failure logging due to missing user_id.")

            scored = await _add_similarity(scored, history)
            scored = _add_tier_scores(scored)
            fam, tran, targ = _split_tiers(scored, k, num_days)
            days = _split_by_day(fam, tran, targ, k, num_days)

            flat_df = pd.concat([fam, tran, targ], ignore_index=True)

            return {
                "flat_df": flat_df,
                "familiar": fam.to_dict("records"),
                "transition": tran.to_dict("records"),
                "target": targ.to_dict("records"),
                "days": days,
                "num_days": num_days,
            }





# ──────────────────────────────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────────────────────────────

def _select_history(hist: List[dict[str, Any]], meal_type: str) -> List[dict[str, Any]]:
    if not hist:
        return []
    slot = None if meal_type.lower() == "any" else meal_type.lower()
    subset = [
        m for m in hist
        if slot is None
        or m.get("meal_slot", "").lower() == slot
        or m.get("meal_type", "").lower() == slot
    ]
    return subset if len(subset) <= 5 else random.sample(subset, 5)


async def _embed_vector(text: str) -> np.ndarray:
    try:
        return gemini.embed(text)
    except Exception:
        return np.zeros((768,), dtype=np.float32)


async def _add_similarity(df: pd.DataFrame, history: List[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return df.assign(similarity_score=0.0)

    print(f"[DEBUG] History received: {history}")

    hist_vecs = await asyncio.gather(
        *[_embed_vector(m["meal_name"]) for m in history if "meal_name" in m]
    )
    if not hist_vecs:
        return df.assign(similarity_score=0.0)

    cand_vecs = await asyncio.gather(
        *[_embed_vector(r["meal_name"]) for _, r in df.iterrows()]
    )

    hist_mat = np.vstack(hist_vecs)
    cand_mat = np.vstack(cand_vecs)

    sims = cosine_similarity(cand_mat, hist_mat).max(axis=1)
    return df.assign(similarity_score=sims)


async def _merge_next_week_targets(cur: Dict[str, float], nxt: Dict[str, float]) -> Dict[str, float]:
    merged = cur.copy()
    for key in ("kcal", "protein_g", "carbs_g", "fat_g"):
        if key in nxt:
            merged[key] = nxt[key]
    return merged


def _add_tier_scores(df: pd.DataFrame) -> pd.DataFrame:
    rng = df["score"].max() - df["score"].min()
    norm = 1 - (df["score"] - df["score"].min()) / rng if rng else 1.0
    return df.assign(
        norm_score=norm,
        familiar_score=0.7 * df["similarity_score"] + 0.3 * norm,
        transition_score=0.3 * df["similarity_score"] + 0.7 * norm,
        target_score=0.1 * df["similarity_score"] + 0.9 * norm,
    )


def _split_tiers(df: pd.DataFrame, k: int, days: int) -> Tuple[pd.DataFrame, ...]:
    per_day = max(3, k // 3)
    fam_n = max(1, per_day // 3) * days
    tran_n = max(1, per_day // 3) * days
    targ_n = per_day * days - fam_n - tran_n

    fam = df.nlargest(fam_n, "familiar_score")
    tran = df[~df["meal_id"].isin(fam["meal_id"])].nlargest(tran_n, "transition_score")
    targ = df[
        ~df["meal_id"].isin(fam["meal_id"].tolist() + tran["meal_id"].tolist())
    ].nlargest(targ_n, "target_score")

    drops = [
        c for c in (
            "similarity_score",
            "norm_score",
            "familiar_score",
            "transition_score",
            "target_score",
        ) if c in fam.columns
    ]
    return fam.drop(columns=drops), tran.drop(columns=drops), targ.drop(columns=drops)


def _split_by_day(
    fam: pd.DataFrame, tran: pd.DataFrame, targ: pd.DataFrame, k: int, days: int
) -> Dict[str, Any]:
    per_day = max(3, k // 3)
    fam_d = max(1, per_day // 3)
    tran_d = max(1, per_day // 3)
    targ_d = per_day - fam_d - tran_d

    out: Dict[str, Any] = {}
    for d in range(1, days + 1):
        i = d - 1
        out[f"day_{d}"] = {
            "familiar": fam.iloc[i * fam_d: (i + 1) * fam_d].to_dict("records"),
            "transition": tran.iloc[i * tran_d: (i + 1) * tran_d].to_dict("records"),
            "target": targ.iloc[i * targ_d: (i + 1) * targ_d].to_dict("records"),
        }
    return out

def extract_clean_json(response) -> dict:
    if isinstance(response, dict):
        return response  # Already a dict

    if not isinstance(response, str):
        raise ValueError("Expected string or dict-like response.")

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from messy text
    match = re.search(r"\{[\s\S]*?\}", response)
    if match:
        json_text = match.group(0)
        if not json_text.strip().endswith("}"):
            json_text += "}"
        try:
            return json.loads(json_text)
        except Exception as e:
            print("[ERROR] Still failed after patching:", e)
            raise e
    raise ValueError("No valid JSON object found in response")