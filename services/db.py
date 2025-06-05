"""
services/db.py
────────────────────────────────────────────────────────────────────────
* Async SQLAlchemy v2 setup
* Models that map to the five existing tables
* Small DAO helpers used by routers / workers
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, List
from datetime import datetime



from sqlalchemy import JSON, BigInteger, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncAttrs,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from config import settings

# ───────── connection helper ────────────────────────────────────────
_ENGINE: AsyncEngine | None = None


async def _create_engine() -> AsyncEngine:
    # 1) plain TCP URL
    if settings.database_url:
        return create_async_engine(settings.database_url, pool_pre_ping=True)

    # 2) Cloud SQL connector (only if URL not supplied)
    if not settings.cloud_sql_instance:
        raise RuntimeError(
            "Set either DATABASE_URL or CLOUD_SQL_CONNECTION_NAME env var"
        )

    # lazy import here
    try:
        from google.cloud.sql.connector import ConnectorAsync, IPTypes  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "cloud-sql-python-connector missing. Run:\n"
            "pip install 'cloud-sql-python-connector[asyncpg]>=1.4.0'"
        ) from exc

    connector = ConnectorAsync()

    async def _getconn():  # type: ignore[name-defined]
        return await connector.connect_async(
            settings.cloud_sql_instance,
            "asyncpg",
            user=settings.db_user,
            password=settings.db_pass,
            db=settings.db_name,
            ip_type=IPTypes.PRIVATE,
        )

    return create_async_engine(
        "postgresql+asyncpg://",
        creator=_getconn,
        pool_pre_ping=True,
    )



async def engine() -> AsyncEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = await _create_engine()
    return _ENGINE


# ───────── declarative base ──────────────────────────────────────────
Base = declarative_base(cls=AsyncAttrs)

# ───────── models reflect existing table layout ─────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    email: Mapped[str | None] = mapped_column(String)
    age: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    height_cm: Mapped[float | None] = mapped_column(Float)
    activity_level: Mapped[str | None] = mapped_column(String)
    exercise_frequency_per_week: Mapped[int | None] = mapped_column(Integer)


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[int] = mapped_column(primary_key=True)
    health_conditions: Mapped[str | None] = mapped_column(Text)  # serialized list
    goal_type: Mapped[str | None] = mapped_column(String)
    motivation: Mapped[str | None] = mapped_column(String)
    dietary_restrictions: Mapped[str | None] = mapped_column(Text)  # serialized list
    preferred_cuisines: Mapped[str | None] = mapped_column(Text)  # serialized list


class UserNutritionTargets(Base):
    __tablename__ = "user_nutrition_targets"

    user_id: Mapped[int] = mapped_column(primary_key=True)
    optimal_calories: Mapped[float]
    protein_g: Mapped[float]
    carbs_g: Mapped[float]
    fat_g: Mapped[float]
    ibw_kg: Mapped[float]
    # ➜ new micro columns added via SQL below
    sodium_mg: Mapped[float | None] = mapped_column(Float)
    potassium_mg: Mapped[float | None] = mapped_column(Float)
    magnesium_mg: Mapped[float | None] = mapped_column(Float)
    calcium_mg: Mapped[float | None] = mapped_column(Float)
    iron_mg: Mapped[float | None] = mapped_column(Float)
    folate_mcg: Mapped[float | None] = mapped_column(Float)
    vitamin_b12_mcg: Mapped[float | None] = mapped_column(Float)
    omega3_mg: Mapped[float | None] = mapped_column(Float)
    fiber_g: Mapped[float | None] = mapped_column(Float)
    added_sugar_g: Mapped[float | None] = mapped_column(Float)
    health_conditions: Mapped[str | None] = mapped_column(Text)
    goals: Mapped[str | None] = mapped_column(Text)
    motivation: Mapped[str | None] = mapped_column(Text)


class GeneratedMeal(Base):
    __tablename__ = "generated_meals"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int]
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    meal_type: Mapped[str]
    meal_name: Mapped[str]
    meal_description: Mapped[str | None] = mapped_column(Text)
    source_model: Mapped[str | None] = mapped_column(String)
    nutrition: Mapped[dict] = mapped_column(JSON)
    rationale: Mapped[str | None] = mapped_column(Text)
    tier:      Mapped[str | None]  = mapped_column(String)   # new
    verified:  Mapped[bool]        = mapped_column(default=False)
    failed_reason: Mapped[str | None] = mapped_column(Text)



class GenerationFailure(Base):
    __tablename__ = "generation_failures"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int]
    meal_type: Mapped[str]
    stage: Mapped[str]
    error_message: Mapped[str]
    raw_input: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())


class MealHistory(Base):
    __tablename__ = "meal_history"

    id:             Mapped[int]   = mapped_column(primary_key=True)
    src_generated:  Mapped[int]   = mapped_column(BigInteger, index=True)
    user_id:        Mapped[int]
    meal_type:      Mapped[str]
    meal_name:      Mapped[str]
    eaten_at:       Mapped["datetime"] = mapped_column(DateTime, server_default=func.now())
    nutrition:      Mapped[dict]  = mapped_column(JSON)



# ───────── session helper ────────────────────────────────────────────

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    eng = await engine()
    async_session = async_sessionmaker(eng, expire_on_commit=False)
    async with async_session() as session:
        yield session