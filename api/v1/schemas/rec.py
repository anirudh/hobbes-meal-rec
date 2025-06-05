# api/v1/schemas/rec.py
from __future__ import annotations
from typing import Literal, Dict
from pydantic import BaseModel

# only these three slots are valid
Slot = Literal["breakfast", "lunch", "dinner"]


class MealRec(BaseModel):
    meal_id:        int
    meal_slot:      Slot
    meal_name:      str
    dietary_flags:  str
    cuisine:        str
    kcal:           float
    protein_g:      float
    carbs_g:        float
    fat_g:          float
    score:          float


class TieredSlotSet(BaseModel):
    breakfast: MealRec | None
    lunch:     MealRec | None
    dinner:    MealRec | None


class TieredMeals(BaseModel):
    familiar:   TieredSlotSet
    transition: TieredSlotSet
    target:     TieredSlotSet

    # for each day: same structure as above
    days: Dict[str, TieredSlotSet]

    num_days: int


class RecRequest(BaseModel):
    user_id: int
    meal_type: str = "any"
    k: int = 6
    days: int = 1


class RecResponse(BaseModel):
    recommendations: TieredMeals
