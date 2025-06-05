from pydantic import BaseModel


class Meal(BaseModel):
    id: str
    name: str
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    cuisine: str | None = None
    meal_slot: str | None = None   # breakfast / lunch / dinner / snack
