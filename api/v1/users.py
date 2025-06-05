from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from services.db import User, get_session
from api.v1.schemas import UserCreate, UserOut

router = APIRouter()


# ───────────────────────── create ──────────────────────────
@router.post(
    "",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    if await db.get(User, body.id):
        raise HTTPException(status_code=409, detail="User already exists")

    user = User(**body.model_dump())
    db.add(user)
    await db.commit()
    return UserOut.model_validate(user, from_attributes=True)


# ───────────────────────── fetch one ────────────────────────
@router.get("/{user_id}", response_model=UserOut)
async def fetch_user(
    user_id: int,
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    usr = await db.get(User, user_id)
    if usr is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserOut.model_validate(usr, from_attributes=True)
