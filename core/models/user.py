from pydantic import BaseModel


class User(BaseModel):
    id: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_level: str
    goal: str = "maintain"
    health_conditions: list[str] = []
