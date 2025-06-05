"""
scripts/init_targets.py
────────────────────────────────────────────────────────────────────────
Populate – or refresh – `user_nutrition_targets`:

Bootstrap every user once:

    python -m scripts.init_targets           # all users

Run weekly for one user (cron / Cloud Scheduler):

    python -m scripts.init_targets --user 123
"""
from __future__ import annotations

import asyncio
import json
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv() 

import httpx
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.nutrition_calc import NutritionalCalculator, UserAnthro
from core.nutrition_nudge import NutritionNudgeAgent, TRACKED
from services.db import (
    MealHistory,           # NEW canonical source of “eaten” meals
    User,
    UserPreferences,
    UserNutritionTargets,
    get_session,
)

# ───────────────────────────────
# External API (Edamam fallback)
# ───────────────────────────────
EDAMAM_ID  = os.getenv("EDAMAM_APP_ID", "")
EDAMAM_KEY = os.getenv("EDAMAM_APP_KEY", "")
EDAMAM_URL = "https://api.edamam.com/api/nutrition-data"

calc = NutritionalCalculator()


async def _edamam_fetch(name: str) -> Dict[str, Any]:
    """
    Minimal Edamam lookup – only the four macros we store in history.
    
    """
    if not (EDAMAM_ID and EDAMAM_KEY):
        raise RuntimeError("EDAMAM_* env vars not set")

    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.get(
            EDAMAM_URL,
            params={"app_id": EDAMAM_ID, "app_key": EDAMAM_KEY, "ingr": name},
        )
    r.raise_for_status()
    data = r.json()
    tot = data.get("totalNutrients", {})
    return {
        "kcal":        data["calories"],
        "protein_g":   tot.get("PROCNT", {}).get("quantity", 0),
        "carbs_g":     tot.get("CHOCDF", {}).get("quantity", 0),
        "fat_g":       tot.get("FAT",    {}).get("quantity", 0),
    }


# ───────────────────────────────
# Target updater per–user
# ───────────────────────────────
async def _refresh_user(db: AsyncSession, user_id: int) -> None:
    # ── 1. basics ───────────────────────────────────────────
    user: User | None = await db.get(User, user_id)
    if not user:
        print(f"· skip {user_id} – user not found")
        return

    prefs: UserPreferences | None = (
        await db.execute(select(UserPreferences).where(UserPreferences.user_id == user_id))
    ).scalar_one_or_none()

    # ── 2. gather 30-day meal history ──────────────────────
    since = datetime.utcnow() - timedelta(days=30)
    q = await db.execute(
        select(MealHistory).where(
            MealHistory.user_id == user_id,
            MealHistory.eaten_at >= since,
        )
    )
    meals: List[MealHistory] = [row[0] for row in q.all()]

    if not meals:
        print(f"· skip {user_id} – no meals in last 30 days")
        return
    
    # ── 2a. enrich every meal in history with full macros+micros ──

    # We only need the Edamam helper, so we can instantiate with empty history
    dummy_agent = NutritionNudgeAgent({}, [])
    updated = False

    for m in meals:
        nut: Dict[str, Any] = dict(m.nutrition or {})
        # if any of our 14 tracked nutrients is missing
        if any(k not in nut for k in TRACKED):
            try:
                full = dummy_agent.get_edamam_nutrition(m.meal_name)
                # merge in any new keys
                nut.update(full)
                m.nutrition = nut
                db.add(m)
                updated = True
            except Exception as e:
                print(f"  ! failed to fetch nutrition for «{m.meal_name}»: {e}")

    if updated:
        # persist any in‑place enrichment
        await db.commit()


    # ensure each meal dict has macros; fill via Edamam if blank
    enriched: List[Dict[str, Any]] = []
    for m in meals:
        nut = dict(m.nutrition or {})
        if {"kcal", "protein_g", "carbs_g", "fat_g"} - nut.keys():
            try:
                nut = await _edamam_fetch(m.meal_name)
            except Exception as e:  # noqa: BLE001
                print(f"  ! Edamam failed for «{m.meal_name}»: {e}")
                continue
        enriched.append(nut)

    if not enriched:
        print(f"· skip {user_id} – all meals lacked nutrition data")
        return

    # ── 3. anthropometrics & calculator ────────────────────
    anth = UserAnthro(
        age=user.age,
        gender=user.sex.capitalize(),
        weight_kg=user.weight_kg,
        height_cm=user.height_cm,
        daily_activity=int(user.activity_level or 3),
        workouts_per_week=user.exercise_frequency_per_week or 0,
        goal=(prefs.goal_type if prefs else "maintain").lower(),
        wants_muscle="muscle" in ((prefs.goal_type or "").lower()) if prefs else False,
        health_conditions=(
            json.loads(prefs.health_conditions)
            if prefs and prefs.health_conditions
            else []
        ),
    )

    # ── 4. compute targets (macros + micros) ───────────────
    base_targets = calc.targets(anth)

    # rename kcal→optimal_calories to match DB column
    base_targets["optimal_calories"] = base_targets.pop("kcal")

    # ── 5. nudge baseline averages (for future features) ───
    agent = NutritionNudgeAgent(anth.__dict__, enriched)
    baseline = agent._weekly_average(enriched)          # protected method OK in script
    # Store baseline snapshot if you need it (optional)

    # ── 6. upsert row ──────────────────────────────────────
    row = UserNutritionTargets(
        user_id=user_id,
        **base_targets,
        goals=prefs.goal_type if prefs else "maintain",
        health_conditions=prefs.health_conditions if prefs else "[]",
        motivation=prefs.motivation if prefs else None,
    )
    await db.merge(row)
    await db.commit()
    print(f"✓ targets updated for user {user_id}")


# ───────────────────────────────
# CLI entrypoint
# ───────────────────────────────
async def _async_main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--user", type=int, help="update only this user-id")
    args = ap.parse_args()

    async with get_session() as db:
        if args.user:
            await _refresh_user(db, args.user)
        else:
            ids = (await db.execute(select(User.id))).scalars().all()
            for uid in ids:
                await _refresh_user(db, uid)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_async_main())
