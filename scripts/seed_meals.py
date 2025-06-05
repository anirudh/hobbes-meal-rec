"""
Seed a few demo meals into the `generated_meals` table.

Usage
-----

    # default hard-coded trio of Indian meals
    python -m scripts.seed_meals <USER_ID>

    # custom list (same schema) in a JSON file
    python -m scripts.seed_meals <USER_ID> --file path/to/meals.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List

from services.db import GeneratedMeal, get_session

# ────────────────────────────────────────────────────────────────────
_DEFAULT_MEALS: List[dict[str, Any]] = [
    {
        "meal_type": "breakfast",
        "meal_name": "Masala Oats with Veggies",
        "nutrition": {"kcal": 380, "protein_g": 14, "carbs_g": 58, "fat_g": 9},
    },
    {
        "meal_type": "lunch",
        "meal_name": "Grilled Tandoori Chicken & Quinoa Khichdi",
        "nutrition": {"kcal": 510, "protein_g": 42, "carbs_g": 48, "fat_g": 17},
    },
    {
        "meal_type": "dinner",
        "meal_name": "Palak Paneer with Brown-Rice Phulka",
        "nutrition": {"kcal": 560, "protein_g": 32, "carbs_g": 55, "fat_g": 22},
    },
]


async def _seed(user_id: int, meals: list[dict[str, Any]]) -> None:
    async with get_session() as db:
        for m in meals:
            db.add(
                GeneratedMeal(
                    user_id=user_id,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    verified=False,
                    tier="seed",
                    source_model="seed",
                    **m,
                )
            )
        await db.commit()
    print(f"✓ inserted {len(meals)} meals for user {user_id}")


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of meal dictionaries")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id", type=int, help="target user id")
    parser.add_argument(
        "--file",
        type=Path,
        help="optional JSON file with meals to seed (overrides defaults)",
    )
    args = parser.parse_args()

    meals = _load_json(args.file) if args.file else _DEFAULT_MEALS
    asyncio.run(_seed(args.user_id, meals))


if __name__ == "__main__":
    main()
