# tests/test_nutrition_calc.py
from __future__ import annotations

import math
import pytest

from core.nutrition_calc import NutritionalCalculator, UserAnthro

calc = NutritionalCalculator()

MALE_70KG = UserAnthro(
    age=30,
    gender="Male",
    weight_kg=70,
    height_cm=175,
    daily_activity=3,
    workouts_per_week=3,
)

# ── BMR / TDEE ───────────────────────────────────────────────────────
def test_bmr_mifflin_male():
    expected = 10 * 70 + 6.25 * 175 - 5 * 30 + 5   # 1648.75
    assert math.isclose(calc.bmr(MALE_70KG), expected, rel_tol=1e-4)


def test_tdee_pal_multiplier():
    pal = 1.55 + 0.02 * 3       # 1.61
    expected = calc.bmr(MALE_70KG) * pal
    assert math.isclose(calc.tdee(MALE_70KG), expected, rel_tol=1e-4)


# ── targets() must contain all keys ─────────────────────────────────
def test_targets_contain_macros_and_micros():
    t = calc.targets(MALE_70KG)

    for k in ("kcal", "protein_g", "carbs_g", "fat_g", "ibw_kg"):
        assert k in t

    for k in (
        "sodium_mg",
        "potassium_mg",
        "magnesium_mg",
        "calcium_mg",
        "iron_mg",
        "folate_mcg",
        "vitamin_b12_mcg",
        "omega3_mg",
        "fiber_g",
        "added_sugar_g",
    ):
        assert k in t
    assert t["protein_g"] > 100
    assert t["sodium_mg"] == 2300
