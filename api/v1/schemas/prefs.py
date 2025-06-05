from __future__ import annotations
from typing import List

from pydantic import BaseModel, Field


class UserPrefsIn(BaseModel):
    goal_type: str = Field(..., examples=["lose", "gain", "maintain"])
    motivation: str | None = None
    health_conditions: List[str] = []
    dietary_restrictions: List[str] = []
    preferred_cuisines: List[str] = []


class UserPrefsOut(UserPrefsIn):
    user_id: int
