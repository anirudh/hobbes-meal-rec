"""
core/nutrition_calc.py
────────────────────────────────────────────────────────────────────────
Implements Section A of the Meal-planning algorithm word-for-word:

1. BMR  (Mifflin–St Jeor)
2. TDEE (PAL multiplier)
3. IBW  (Devine)
4. Optimal calories + macros for five goal branches
5. Ten micronutrients (unchanged from previous version)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

Logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  User dataclass
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UserAnthro:
    # core
    age: int
    gender: str            # "Male" | "Female"
    weight_kg: float
    height_cm: float
    daily_activity: int    # 1–5 (how active in daily life)
    workouts_per_week: int # 0–7
    # goals
    goal: str = "maintain"         # "lose" | "gain" | "maintain"
    wants_muscle: bool = False
    health_conditions: list[str] | None = None

    # -------------------------------- convenience flags -------------
    @property
    def diabetes(self) -> bool:
        return _flag(self.health_conditions, ("diabetes", "pre-diabetes"))

    @property
    def hypertension(self) -> bool:
        return _flag(self.health_conditions, ("hypertension", "blood pressure"))

    @property
    def hyperlipidemia(self) -> bool:
        return _flag(self.health_conditions, ("hyperlipidemia", "cholesterol"))


def _flag(conds: list[str] | None, keys: tuple[str, ...]) -> bool:
    if not conds:
        return False
    clow = [c.lower() for c in conds]
    return any(any(k in c for k in keys) for c in clow)


# ──────────────────────────────────────────────────────────────────────
#  Calculator
# ──────────────────────────────────────────────────────────────────────
class NutritionalCalculator:
    """Source-of-truth for kcal+macros (plus micros from Spec 5)."""

    _PAL = [1.2, 1.375, 1.55, 1.725, 1.9]  # index = daily_activity-1

    # --------------- public entrypoint --------------------------------
    def targets(self, u: UserAnthro) -> dict[str, float]:
        kcal = self._optimal_calories(u)
        ibw = self._ibw(u)                   # might be needed by caller
        macros = self._macro_targets(u, kcal, ibw)
        micros = self._micro_targets(u, kcal)   # identical logic from earlier version
        macros["ibw_kg"] = round(ibw, 1)        # expose for transparency
        return {**macros, **micros}

    # --------------- BMR / TDEE / IBW -------------------------------
    def bmr(self, u: UserAnthro) -> float:
        base = 10 * u.weight_kg + 6.25 * u.height_cm - 5 * u.age
        return base + (5 if u.gender == "Male" else -161)

    def tdee(self, u: UserAnthro) -> float:
        idx = min(max(u.daily_activity, 1), 5) - 1
        pal = self._PAL[idx] + 0.02 * u.workouts_per_week
        return self.bmr(u) * pal

    def _ibw(self, u: UserAnthro) -> float:
        """Devine formula (kg)."""
        height_in = u.height_cm / 2.54
        base = 50 if u.gender == "Male" else 45.5
        return base + 2.3 * (height_in - 60)

    # --------------- Calories ---------------------------------------
    def _optimal_calories(self, u: UserAnthro) -> float:
        tdee_val = self.tdee(u)
        if u.goal == "lose":
            return tdee_val - 500
        if u.goal == "gain":
            return tdee_val + 300
        return tdee_val  # maintain

    # --------------- Macros (3 + kcal) ------------------------------
    def _macro_targets(self, u: UserAnthro, kcal: float, ibw: float) -> dict[str, float]:
        """
        4a-e from the specification:
        a. weight-loss / b. gain / c. maintain
        d. muscle gain (overrides protein, fat fixed 25 %, carbs remainder)
        e. diabetes (carbs 40 %, protein uses previous value)
        """
        # (1) start with default ratio (weight-maintain path)
        carbs_pc, prot_pc, fat_pc = 0.45, 0.30, 0.25

        # Adjust kcal already done; now special ratios
        if u.wants_muscle:                       # 4d
            prot_g = 2.4 * ibw
            fat_g = fat_pc * kcal / 9
            carbs_g = (kcal - (prot_g * 4 + fat_g * 9)) / 4
        else:
            # protein/fat/carbs via ratios first
            prot_g = prot_pc * kcal / 4
            fat_g = fat_pc * kcal / 9
            carbs_g = carbs_pc * kcal / 4

        # Diabetes adjustment overrides *only carbohydrate ratio*
        if u.diabetes:                           # 4e
            carbs_g = 0.40 * kcal / 4
            # protein remains whatever was set above
            fat_g = (kcal - (prot_g * 4 + carbs_g * 4)) / 9

        return {
            "kcal": round(kcal, 0),
            "protein_g": round(prot_g, 1),
            "carbs_g": round(carbs_g, 1),
            "fat_g": round(fat_g, 1),
        }

    # --------------- Micros (unchanged from previous answer) --------
    def _micro_targets(self, u: UserAnthro, kcal: float) -> dict[str, float]:
        male = u.gender == "Male"
        over50 = u.age >= 50

        micros = {
            "sodium_mg": 2300,
            "potassium_mg": 3400 if male else 2600,
            "magnesium_mg": 420 if male else 320,
            "calcium_mg": 1000,
            "iron_mg": 8 if male or over50 else 18,
            "folate_mcg": 400,
            "vitamin_b12_mcg": 2.4,
            "omega3_mg": 500,
            "fiber_g": 38 if male else 28,
            "added_sugar_g": 0.10 * kcal / 4,
        }

        if u.diabetes:
            micros.update(
                potassium_mg=4700,
                magnesium_mg=500,
                fiber_g=round(14 * (kcal / 1000), 1),
                added_sugar_g=0.06 * kcal / 4,
            )

        if u.hypertension:
            micros.update(
                sodium_mg=1500,
                potassium_mg=4700,
                magnesium_mg=500,
                calcium_mg=1250,
                added_sugar_g=0.06 * kcal / 4,
                omega3_mg=3000,
            )

        if u.hyperlipidemia:
            micros.update(
                added_sugar_g=0.06 * kcal / 4,
                omega3_mg=3000,
            )

        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in micros.items()}
