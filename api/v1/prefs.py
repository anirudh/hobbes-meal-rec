from __future__ import annotations

import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from services.db import UserPreferences, get_session
from api.v1.schemas.prefs import UserPrefsIn, UserPrefsOut

router = APIRouter()


# ───────────────────────── helpers ──────────────────────────
def _serialize(row: UserPreferences) -> UserPrefsOut:
    """Convert SQLAlchemy row ➜ Pydantic schema with proper JSON decoding."""
    return UserPrefsOut(
        user_id=row.user_id,
        goal_type=row.goal_type or "maintain",
        motivation=row.motivation,
        health_conditions=json.loads(row.health_conditions or "[]"),
        dietary_restrictions=json.loads(row.dietary_restrictions or "[]"),
        preferred_cuisines=json.loads(row.preferred_cuisines or "[]"),
    )


# ───────────────────────── read ─────────────────────────────
@router.get(
    "/{user_id}/preferences",
    response_model=UserPrefsOut,
    status_code=status.HTTP_200_OK,
)
async def get_preferences(
    user_id: int,
    db: AsyncSession = Depends(get_session),
) -> UserPrefsOut:
    prefs = (
        await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
    ).scalar_one_or_none()

    if prefs is None:
        raise HTTPException(404, "preferences not set")

    return _serialize(prefs)


# ───────────────────────── upsert ───────────────────────────
@router.put(
    "/{user_id}/preferences",
    response_model=UserPrefsOut,
    status_code=status.HTTP_200_OK,
)
async def upsert_preferences(
    user_id: int,
    body: UserPrefsIn,
    db: AsyncSession = Depends(get_session),
) -> UserPrefsOut:
    payload = {
        "goal_type": body.goal_type,
        "motivation": body.motivation,
        "health_conditions": json.dumps(body.health_conditions),
        "dietary_restrictions": json.dumps(body.dietary_restrictions),
        "preferred_cuisines": json.dumps(body.preferred_cuisines),
    }

    # Try update → if row doesn’t exist we’ll insert.
    res = await db.execute(
        update(UserPreferences)
        .where(UserPreferences.user_id == user_id)
        .values(**payload)
        .returning(UserPreferences)
    )
    prefs = res.scalar_one_or_none()

    if prefs is None:                          # Insert
        prefs = UserPreferences(user_id=user_id, **payload)  # type: ignore[arg-type]
        db.add(prefs)

    await db.commit()
    await db.refresh(prefs)
    return _serialize(prefs)
