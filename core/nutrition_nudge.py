# modules/nutrition_nudge.py
"""
Compute weekly “nudge” targets that move a user gradually from their
current baseline toward the optimal targets produced by
`core.nutrition_calc.NutritionalCalculator`.

Tracks exactly the 14 nutrients/macros used elsewhere:

    kcal · protein_g · carbs_g · fat_g ·
    sodium_mg · potassium_mg · magnesium_mg · calcium_mg · iron_mg ·
    folate_mcg · vitamin_b12_mcg · omega3_mg · fiber_g · added_sugar_g
"""

from __future__ import annotations
import logging
import os
import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from core.nutrition_calc import NutritionalCalculator, UserAnthro
from scripts.helpers import extract_clean_json
from services.gemini import generate  # synchronous call

_LOG = logging.getLogger(__name__)

# ──────────────── constants ──────────────────
MACROS = ["kcal", "protein_g", "carbs_g", "fat_g"]
MICROS = [
    "sodium_mg", "potassium_mg", "magnesium_mg", "calcium_mg",
    "iron_mg", "folate_mcg", "vitamin_b12_mcg", "omega3_mg",
    "fiber_g", "added_sugar_g",
]
TRACKED = MACROS + MICROS

_EDAMAM_APP_ID  = os.getenv("EDAMAM_APP_ID", "")
_EDAMAM_APP_KEY = os.getenv("EDAMAM_APP_KEY", "")


def _defaults() -> Dict[str, float]:
    return {
        "kcal": 2000,
        "protein_g": 60,
        "carbs_g": 250,
        "fat_g": 70,
        "sodium_mg": 2300,
        "potassium_mg": 3400,
        "magnesium_mg": 420,
        "calcium_mg": 1000,
        "iron_mg": 8,
        "folate_mcg": 400,
        "vitamin_b12_mcg": 2.4,
        "omega3_mg": 500,
        "fiber_g": 38,
        "added_sugar_g": 0.10 * 2000 / 4,
    }


class NutritionNudgeAgent:
    def __init__(
        self,
        user_profile: Dict[str, Any],
        meal_history: List[Dict[str, Any]],
        calc: NutritionalCalculator | None = None,
    ) -> None:
        self._calc = calc or NutritionalCalculator()
        self.profile = user_profile
        self.meal_history = meal_history
        self.baseline = self._weekly_average(meal_history)
        anthro = _make_anthro(user_profile)
        self.target = self._calc.targets(anthro)
        self.health_weights = _health_weights(
            user_profile.get("health_conditions", [])
        )

    def _weekly_average(self, meals: List[Dict[str, Any]]) -> Dict[str, float]:
        if not meals:
            return _defaults()

        df = pd.DataFrame(meals)
        for col in TRACKED:
            if col not in df.columns:
                df[col] = pd.NA

        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily = df.groupby("date")[TRACKED].sum()
            avg = daily.mean()
        else:
            avg = df[TRACKED].mean()

        base = _defaults()
        for k in TRACKED:
            if k in avg and not pd.isna(avg[k]):
                base[k] = float(avg[k])
        return base

    def compute_nutrient_gaps(self) -> Dict[str, Dict[str, float]]:
        abs_gap, rel_gap = {}, {}
        for k in TRACKED:
            b, t = self.baseline.get(k, 0), self.target.get(k, 0)
            gap = t - b
            abs_gap[k] = gap
            rel_gap[k] = 0 if b == 0 else (gap / b) * 100
        return {"absolute": abs_gap, "relative": rel_gap}

    def compute_nudge_vector(self) -> Dict[str, float]:
        gaps = self.compute_nutrient_gaps()["relative"]
        raw = {k: abs(gaps[k]) / 100 * self.health_weights.get(k, 1.0) for k in TRACKED}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    async def _call_gemini_agent(self, prompt: str) -> Dict[str, float]:
        """Invoke the sync generate() and parse its JSON output."""
        _LOG.debug("Calling Gemini for next-week targets…")
        try:
            text = generate(prompt)
            # strip any code fences
            text = re.sub(r"^```(?:json)?\s*|```$", "", text).strip()
            # try a clean helper first
            data = extract_clean_json(text) or json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            _LOG.warning("Gemini failed: %s; falling back", e)
            return {}

    async def next_week_target(self, pct: float = 10) -> Dict[str, float]:
        # build a concise prompt
        prompt = (
            "You are a nutritionist AI.  Given this user profile:\n"
            f"{json.dumps(self.profile)}\n\n"
            "and this past 7-day meal history:\n"
            f"{json.dumps(self.meal_history[-7:])}\n\n"
            "Return ONLY a flat JSON object with these keys:\n"
            f"{TRACKED}"
        )

        # try LLM-driven
        gemini_out = await self._call_gemini_agent(prompt)
        if gemini_out:
            _LOG.debug("✅ Gemini returned targets: %s", gemini_out)
            return gemini_out

        # numeric fallback
        _LOG.debug("Using numeric fallback for next_week_target")
        vec = self.compute_nudge_vector()
        out: Dict[str, float] = {}
        for k in TRACKED:
            delta = self.target[k] - self.baseline[k]
            step  = min(pct * vec.get(k, 0.5), 15)
            out[k] = self.baseline[k] + delta * (step / 100)
        return out

    def get_edamam_nutrition(self, meal_name: str) -> Dict[str, Any]:
        if not (_EDAMAM_APP_ID and _EDAMAM_APP_KEY):
            _LOG.warning("Edamam creds missing—using estimate()")
            return _estimate_nutrition(meal_name)
        try:
            resp = requests.get(
                "https://api.edamam.com/api/nutrition-data",
                params=dict(
                    app_id=_EDAMAM_APP_ID,
                    app_key=_EDAMAM_APP_KEY,
                    ingr=meal_name,
                ),
                timeout=10,
            )
            resp.raise_for_status()
            nu = resp.json().get("totalNutrients", {})
            return {
                "kcal":   resp.json().get("calories", 0),
                "protein_g": nu.get("PROCNT", {}).get("quantity", 0),
                "carbs_g":   nu.get("CHOCDF", {}).get("quantity", 0),
                "fat_g":     nu.get("FAT", {}).get("quantity", 0),
                "fiber_g":   nu.get("FIBTG", {}).get("quantity", 0),
                "sodium_mg": nu.get("NA", {}).get("quantity", 0),
                "potassium_mg": nu.get("K", {}).get("quantity", 0),
                "calcium_mg":  nu.get("CA", {}).get("quantity", 0),
                "iron_mg":     nu.get("FE", {}).get("quantity", 0),
                "vitamin_b12_mcg": nu.get("VITB12", {}).get("quantity", 0),
                "folate_mcg":     nu.get("FOLDFE", {}).get("quantity", 0),
                "omega3_mg":      0,  # not provided
                "magnesium_mg":   nu.get("MG", {}).get("quantity", 0),
                "added_sugar_g":  0,
            }
        except Exception as exc:
            _LOG.error("Edamam lookup failed: %s", exc)
            return _estimate_nutrition(meal_name)


# ─────────────────── Helpers ───────────────────

def _estimate_nutrition(name: str) -> Dict[str, float]:
    """Fallback keyword-based nutrition estimates."""
    n = name.lower()
    base = {
        "kcal": 400, "protein_g": 15, "carbs_g": 40, "fat_g": 15,
        "fiber_g": 4,   "sodium_mg": 500, "potassium_mg": 400,
        "calcium_mg": 100, "iron_mg": 2, "folate_mcg": 50,
        "vitamin_b12_mcg": 0.2, "magnesium_mg": 40,
        "omega3_mg": 0, "added_sugar_g": 10,
    }
    # simple adjustments...
    if any(w in n for w in ("pizza", "burger", "fried")):
        base["kcal"]   += 300
        base["fat_g"]   += 15
        base["sodium_mg"] += 500
    elif any(w in n for w in ("salad", "vegetable", "veggie")):
        base["kcal"]   -= 150
        base["fiber_g"] += 5
    for k, v in base.items():
        base[k] = max(v, 0)
    return base


def _make_anthro(p: Dict[str, Any]) -> UserAnthro:
    return UserAnthro(
        age=p.get("age", 30),
        gender=p.get("gender", "Male"),
        weight_kg=p.get("weight_kg", 70),
        height_cm=p.get("height_cm", 170),
        daily_activity=p.get("daily_activity", 3),
        workouts_per_week=p.get("workouts_per_week", 3),
        goal=p.get("goal", "maintain"),
        wants_muscle=p.get("wants_muscle", False),
        health_conditions=p.get("health_conditions", []),
    )


def _health_weights(conds: List[str]) -> Dict[str, float]:
    w = {k: 1.0 for k in TRACKED}
    low = [c.lower() for c in conds or []]
    if any("diabet" in c for c in low):
        w.update({"carbs_g": 2.5, "added_sugar_g": 3.0, "fiber_g": 2.0})
    if any("hypertens" in c or "blood pressure" in c for c in low):
        w.update({"sodium_mg": 3.0, "potassium_mg": 2.5, "magnesium_mg": 2.0, "calcium_mg": 2.0})
    if any("lipidemia" in c or "cholesterol" in c for c in low):
        w.update({"omega3_mg": 3.0, "added_sugar_g": 2.0})
    return w
