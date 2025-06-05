"""
End-to-end (no DB) – build a tiny meals catalogue and verify tier splits.
"""
import pandas as pd
from core.nutrition_calc import NutritionalCalculator
from core.recommendation import NudgeMealRecommender

calc = NutritionalCalculator()

# --- catalogue of 6 dummy meals ----------------------------------------
CATALOGUE = pd.DataFrame(
    [
        {"meal_id": 1, "meal_name": "Oatmeal", "calories": 350, "protein_g": 15, "carbs_g": 55, "fat_g": 8},
        {"meal_id": 2, "meal_name": "Chicken Salad", "calories": 450, "protein_g": 40, "carbs_g": 20, "fat_g": 18},
        {"meal_id": 3, "meal_name": "Veggie Stir-Fry", "calories": 500, "protein_g": 22, "carbs_g": 60, "fat_g": 14},
        {"meal_id": 4, "meal_name": "Beef Bowl", "calories": 650, "protein_g": 38, "carbs_g": 50, "fat_g": 28},
        {"meal_id": 5, "meal_name": "Pasta", "calories": 700, "protein_g": 25, "carbs_g": 90, "fat_g": 20},
        {"meal_id": 6, "meal_name": "Smoothie", "calories": 300, "protein_g": 18, "carbs_g": 40, "fat_g": 6},
    ]
)

# user history (simple overlap with catalogue for similarity calc)
HISTORY = [
    {"meal_id": 7, "calories": 650, "protein_g": 35, "carbs_g": 50, "fat_g": 25},
    {"meal_id": 8, "calories": 450, "protein_g": 40, "carbs_g": 20, "fat_g": 18},
]

PROFILE = {
    "age": 30,
    "gender": "male",
    "weight_kg": 70,
    "height_cm": 175,
    "activity_level": "Moderately Active",
    "goal": "maintain",
}

REQ = calc.calculate_requirements(PROFILE, goal="maintain")

def test_recommender_tiers():
    r = NudgeMealRecommender(calc, CATALOGUE, HISTORY)
    flat, tiers = r.recommend(
        PROFILE,
        REQ,
        dietary_restrictions=[],
        meal_type="any",
        cuisine_preference=[],
        k=6,
        num_days=1,
    )
    assert len(flat) == 6
    # at least one meal per tier
    assert all(tiers[key] for key in ("familiar", "transition", "target"))
