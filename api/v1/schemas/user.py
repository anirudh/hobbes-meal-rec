from __future__ import annotations
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class Sex(str, Enum):
    male = "male"
    female = "female"


class UserCreate(BaseModel):
    id: int
    name: str | None = None
    email: str | None = None
    age: int
    sex: Sex = Field(..., description="male or female, case-insensitive")
    weight_kg: float
    height_cm: float
    activity_level: str
    exercise_frequency_per_week: int

    model_config = ConfigDict(from_attributes=True)


class UserOut(UserCreate):
    """Same fields as input, different semantic meaning."""
    pass
