"""
Smoke-tests for core/nutrition_nudge.py
"""

import pytest
from core.nutrition_nudge import NutritionNudgeAgent

# --- minimal dummy data -------------------------------------------------
USER = dict(
    user_id=1,
    age=30,
    gender="male",
    weight_kg=70,
    height_cm=175,
    activity_level="moderate",
    goal="maintain",
    health_conditions=["hypertension"],
)

MEALS = [  # pretend each is a row from generated_meals.nutrition
    dict(calories=650, protein_g=35, carbs_g=50, fat_g=25, sodium_mg=900),
    dict(calories=550, protein_g=30, carbs_g=60, fat_g=15, sodium_mg=800),
]

agent = NutritionNudgeAgent(USER, MEALS)


def test_baseline_keys():
    # should include the core macro keys
    for k in ("calories", "protein_g", "carbs_g", "fat_g"):
        assert k in agent.baseline


def test_targets_move_toward_guidelines():
    # calorie target should usually fall between baseline±500
    diff = abs(agent.target["calories"] - agent.baseline["calories"])
    assert diff < 500


def test_next_week_delta_small():
    nxt = agent.next_week_target(percent_change=10.0)
    # only 10 % shift, so numbers should be close to baseline
    assert abs(nxt["sodium_mg"] - agent.baseline["sodium_mg"]) < 0.2 * agent.baseline[
        "sodium_mg"
    ]
