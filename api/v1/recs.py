# api/v1/recs.py
from __future__ import annotations
import json
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from core.nutrition_calc import NutritionalCalculator
from core.recommendation import NudgeMealRecommender
from services.db import GeneratedMeal, User, UserNutritionTargets, UserPreferences, get_session
from api.v1.schemas import RecRequest, RecResponse

router = APIRouter()
_calc = NutritionalCalculator()


def _pick_one(
    candidates: list[dict[str, Any]],
    slot: str,
    fallback: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """
    From `candidates`, return the highest‑score meal whose
    "meal_slot" == slot.  If none, return fallback[slot].
    """
    slot = slot.lower()
    filtered = [m for m in candidates if m.get("meal_slot", "").lower() == slot]
    if filtered:
        return max(filtered, key=lambda m: m["score"])
    return fallback.get(slot)


@router.post("", response_model=RecResponse, status_code=status.HTTP_200_OK)
async def recommend(
    body: RecRequest,
    db: AsyncSession = Depends(get_session),
) -> RecResponse:
    # 1) Load user
    user = await db.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2) Load nutrition targets
    tgt_row = (
        await db.execute(
            select(UserNutritionTargets)
            .where(UserNutritionTargets.user_id == body.user_id)
        )
    ).scalar_one_or_none()
    if not tgt_row:
        raise HTTPException(400, "Nutrition targets missing – calculate first")
    targets = tgt_row.__dict__

    # 3) Load preferences
    prefs = (
        await db.execute(
            select(UserPreferences)
            .where(UserPreferences.user_id == body.user_id)
        )
    ).scalar_one_or_none()
    dietary = (
        json.loads(prefs.dietary_restrictions)
        if prefs and prefs.dietary_restrictions else []
    )
    cuisines = (
        json.loads(prefs.preferred_cuisines)
        if prefs and prefs.preferred_cuisines else []
    )

    # 4) Fetch this user’s generated meals
    gm_q = await db.execute(
        select(GeneratedMeal)
        .where(GeneratedMeal.user_id == body.user_id)
    )
    meals = gm_q.scalars().all()
    if not meals:
        raise HTTPException(404, "No meals found for this user")

    # 5) Build a DataFrame + list of nutrition dicts
    df_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for m in meals:
        nut = m.nutrition or {}
        history.append(nut)
        df_rows.append({
            "meal_id":       m.id,
            "meal_slot":     m.meal_type,
            "meal_name":     m.meal_name,
            "dietary_flags": "",   # placeholder
            "cuisine":       "",   # placeholder
            **nut,               # merges in macros + micros
        })
    meals_df = pd.DataFrame(df_rows)

    # 6) Build profile dict
    profile = {
        "user_id":           user.id,
        "age":               user.age or 30,
        "gender":            (user.sex or "Male").capitalize(),
        "weight_kg":         user.weight_kg or 70,
        "height_cm":         user.height_cm or 170,
        "daily_activity":    int(user.activity_level or 3),
        "workouts_per_week": user.exercise_frequency_per_week or 0,
        "goal":              (prefs.goal_type if prefs else "maintain").lower(),
        "health_conditions": (
            json.loads(prefs.health_conditions)
            if prefs and prefs.health_conditions else []
        ),
    }

    # 7) Run the nudge recommender
    recommender = NudgeMealRecommender(_calc, meals_df, history)
    result = await recommender.recommend(
        profile,
        targets,
        dietary,
        body.meal_type,
        cuisines,
        k=body.k,
        num_days=body.days,
        db=db,  # <── new argument
    )

    # 8) Unpack -> flat candidates + raw tiers dict
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        flat_scores, raw_tiers = result[0], result[-1]
    else:
        # if no flat list returned, build one by merging all three tiers
        raw_tiers = result
        flat_scores = (
            raw_tiers.get("familiar", []) +
            raw_tiers.get("transition", []) +
            raw_tiers.get("target", [])
        )

    # 9) Build a global “fallback” (best breakfast, lunch, dinner)
    fallback: dict[str, dict[str, Any] | None] = {}
    for slot in ("breakfast", "lunch", "dinner"):
        by_slot = [m for m in flat_scores if m.get("meal_slot","").lower() == slot]
        fallback[slot] = max(by_slot, key=lambda m: m["score"]) if by_slot else None

    # 10) For each tier, pick exactly one per slot (fallback if missing)
    tiered: dict[str, dict[str, dict[str, Any] | None]] = {}
    for tier_name in ("familiar", "transition", "target"):
        scored_list = raw_tiers.get(tier_name, [])
        tiered[tier_name] = {
            slot: _pick_one(scored_list, slot, fallback)
            for slot in ("breakfast", "lunch", "dinner")
        }

    # 11) Build per‑day menus (here: always use the target menu)
    days_out: dict[str, dict[str, dict[str, Any] | None]] = {}
    for i in range(1, body.days + 1):
        days_out[f"day_{i}"] = tiered["target"]

    # 12) Return in your Pydantic shape
    return RecResponse(
        recommendations={
            "familiar":   tiered["familiar"],
            "transition": tiered["transition"],
            "target":     tiered["target"],
            "days":       days_out,
            "num_days":   raw_tiers.get("num_days", body.days),
        }
    )
