from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class GeneratedMeal(BaseModel):
    id: int
    meal_type: str
    meal_name: str
    created_at: datetime
    nutrition: dict

    model_config = ConfigDict(from_attributes=True)

class MealSelected(BaseModel):
    meal_type: str
    meal_name: str
    nutrition: dict
    tier: str                   # familiar / transition / target

class MealSelectionIn(BaseModel):
    user_id: int
    meals: list[MealSelected]
