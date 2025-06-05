#!/usr/bin/env python3
"""
Back‑fill `meal_history` from every existing `generated_meals` row for a given user.
Usage:
    python -m scripts.backfill_meal_history <user_id>
"""
import sys
import asyncio
from datetime import datetime

from sqlalchemy import select
from services.db import get_session, GeneratedMeal, MealHistory

async def backfill_meal_history(user_id: int) -> None:
    async with get_session() as db:
        # 1) Load all generated meals for this user
        res = await db.execute(
            select(GeneratedMeal).where(GeneratedMeal.user_id == user_id)
        )
        generated = res.scalars().all()

        if not generated:
            print(f"No generated_meals found for user {user_id}")
            return

        # 2) For each, insert into meal_history if not already present
        new_count = 0
        for gm in generated:
            exists = await db.execute(
                select(MealHistory).where(MealHistory.src_generated == gm.id)
            )
            if exists.first():
                continue

            db.add(
                MealHistory(
                    src_generated=gm.id,
                    user_id=gm.user_id,
                    meal_type=gm.meal_type,
                    meal_name=gm.meal_name,
                    nutrition=gm.nutrition,
                    eaten_at=datetime.utcnow(),
                )
            )
            new_count += 1

        # 3) Commit once at the end
        await db.commit()
        print(f"✓ back‑filled {new_count} new rows into meal_history (out of {len(generated)}) for user {user_id}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python backfill_meal_history.py <user_id>")
        sys.exit(1)

    try:
        user_id = int(sys.argv[1])
    except ValueError:
        print("Error: user_id must be an integer")
        sys.exit(1)

    asyncio.run(backfill_meal_history(user_id))

if __name__ == "__main__":
    main()
