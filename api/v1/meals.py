# api/v1/meals.py
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.db import GeneratedMeal, MealHistory, get_session
from api.v1.schemas.meal import GeneratedMeal as MealOut

router = APIRouter()


@router.get(
    "/{user_id}",
    response_model=list[MealOut],
    status_code=status.HTTP_200_OK,
    summary="List all generated meals for a user",
)
async def list_user_meals(
    user_id: int,
    db: AsyncSession = Depends(get_session),
) -> list[MealOut]:
    """
    Fetch all meals generated for `user_id`.
    """
    result = await db.execute(
        select(GeneratedMeal).where(GeneratedMeal.user_id == user_id)
    )
    meals = result.scalars().all()
    return [
        MealOut.model_validate(m, from_attributes=True)
        for m in meals
    ]


@router.delete(
    "/{meal_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a generated meal by its ID",
)
async def delete_generated(
    meal_id: int,
    db: AsyncSession = Depends(get_session),
) -> Response:
    meal = await db.get(GeneratedMeal, meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    await db.delete(meal)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{meal_id}/select",
    status_code=status.HTTP_201_CREATED,
    summary="Mark a generated meal as chosen and record it in meal_history",
)
async def select_meal(
    meal_id: int,
    db: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    gm = await db.get(GeneratedMeal, meal_id)
    if not gm:
        raise HTTPException(status_code=404, detail="Generated meal not found")

    if gm.verified:
        return {"status": "already-recorded"}

    gm.verified = True
    hist = MealHistory(
        src_generated=gm.id,
        user_id=gm.user_id,
        meal_type=gm.meal_type,
        meal_name=gm.meal_name,
        nutrition=gm.nutrition,
    )
    db.add(hist)
    await db.commit()
    return {"status": "recorded"}
