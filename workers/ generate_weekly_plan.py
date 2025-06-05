"""
`python -m workers.generate_weekly_plan --user-id=123`
Run as Cloud Run Jobs or Cloud Tasks later.
"""
import argparse

from services.db import MealDAO, UserDAO
from core.recommendation import NudgeMealRecommender
from core.nutrition_calc import NutritionalCalculator, UserAnthro

def _run(uid: str) -> None:
    user = UserDAO.get(uid)
    if not user:
        raise SystemExit(f"user {uid} not found")

    calc = NutritionalCalculator()
    anth = UserAnthro(
        age=user.age,
        gender=user.gender,
        weight_kg=user.weight_kg,
        height_cm=user.height_cm,
        activity_level=user.activity_level,
    )
    reqs = calc.macros(calc.tdee(calc.bmr(anth), anth.activity_level), user.goal)
    recommender = NudgeMealRecommender(calc, MealDAO.as_dataframe(), MealDAO.history(uid))
    plan = recommender.recommend_meals(
        user.dict(), reqs, [], "any", [], num_recommendations=21
    )
    print(plan)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    _run(ap.parse_args().user_id)
