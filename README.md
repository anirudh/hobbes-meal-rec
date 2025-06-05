# Run 
 # In a virtual env terminal( after installing the requirments) run the app
 uvicorn main:app --reload --port 8000

 # In a different terminal run db
 cloud-sql-proxy hobbes-meal-rec:us-central1:hobbes-users-db \
  --address 127.0.0.1 \
  --port 5432

# Add user( Demo user 10000)
curl -X POST localhost:8000/api/v1/users -H "Content-Type: application/json" \
     -d '{"id":1,"name":"Bob","email":"b@x","age":30,"sex":"Male","weight_kg":70,"height_cm":175,
          "activity_level":"3","exercise_frequency_per_week":4}'

# Add User Preferences 
curl -X PUT http://127.0.0.1:8000/api/v1/users/10000/preferences \
     -H "Content-Type: application/json" \
     -d '{
           "goal_type": "maintain",
           "motivation": "feel healthier",
           "health_conditions": ["pre-diabetes"],
           "dietary_restrictions": [],
           "preferred_cuisines": ["mediterranean","thai"]
         }'



# Seed meals( meals can be editted in /scripts/seed_ meals.py) for user.
python -m scripts.seed_meals 10000



# List generated meals for user
curl localhost:8000/api/v1/meals/10000

# Initialize USer Targets 
python -m scripts.init_targets --user 10000

# Get recommendations (will warn if no targets yet)
curl -X POST localhost:8000/api/v1/recommendations \
     -H "Content-Type: application/json" \
     -d '{"user_id":10000,"meal_type":"any","k":6,"days":1}'




## Architecture & Data Flow

Below is a high‑level overview of how our app turns raw user data into personalized, tiered meal recommendations.

---

### 1. Nutrition Calculator  
**Location:** `core/nutrition_calc.py`  
- **BMR** (Mifflin–St Jeor):  
  \[
    \text{BMR} = 10 \times \text{weight}_{kg} + 6.25 \times \text{height}_{cm} - 5 \times \text{age} + (\pm)\!
  \]
  + `+5` for Male, `–161` for Female  
- **TDEE** (Total Daily Energy Expenditure):  
  \[
    \text{TDEE} = \text{BMR} \times \bigl(\text{PAL[daily_activity]} + 0.02 \times \text{workouts_per_week}\bigr)
  \]
- **IBW** (Ideal Body Weight, Devine):  
  \[
    \text{IBW}_{kg} = 
      \begin{cases}
        50 + 2.3 \times (\text{height}_{in} - 60), & \text{Male}\\
        45.5 + 2.3 \times (\text{height}_{in} - 60), & \text{Female}
      \end{cases}
  \]
- **Macro Targets**  
  - **Maintain**: 45 % carbs / 30 % protein / 25 % fat  
  - **Lose** (–500 kcal) / **Gain** (+300 kcal) adjust energy first  
  - Overrides for **muscle gain** (protein = 2.4 g × IBW) or **diabetes** (carbs = 40 % kcal→g)  
- **Micro Targets** (10 nutrients)  
  Base RDAs (sodium, potassium, magnesium, etc.), tweaked for diabetes, hypertension, hyperlipidemia.

---

### 2. Nutrition Nudge Agent  
**Location:** `core/nutrition_nudge.py`  
- **Baseline**  
  - Average _daily_ intake over last 7 days of **`meal_history`**, per nutrient  
  - Fallback to healthy‑adult defaults if no history  
- **Gap Computation**  
  - **Absolute**:  `target – baseline`  
  - **Relative**:  `(target – baseline) / baseline × 100 %`  
- **Health Weights**  
  - Boost weighting for nutrients tied to conditions (e.g. ↑fiber & ↓sugar for diabetes)  
- **Nudge Vector**  
  \[
    v[n] = \frac{\lvert \text{rel_gap}[n]\rvert \times \text{health_weight}[n]}%
                 {\sum_m \lvert \text{rel_gap}[m]\rvert \times \text{health_weight}[m]}
  \]
- **Weekly Step**  
  - Move **pct%** of the way baseline→target each week, capped at 15 % per nutrient.

---

### 3. External Nutrition Verification  
**Where:**  
- In **`scripts/init_targets.py`** (or per‑meal fallback in `core/nutrition_nudge`)  
- **Edamam Nutrition API** — called when a meal’s stored `nutrition` JSON is missing core macros  
- Maps Edamam’s `totalNutrients` keys → our fields (`kcal`, `protein_g`, … `fat_g`, plus micros )

---

### 4. Meal Generation  
**Where:**  
- Within our **meal‑generation service** (e.g. `core/meal_recommender.py` or `core/recommendation.py`)  
- **Gemini (or OpenAI) API** → prompts:
  - User profile (anthropometrics, targets, preferences)  
  - “Generate 5 healthy breakfast options high in X, low in Y…”  
- Response parsed into JSON objects, stored in `generated_meals` with:
  ```jsonc
  {
    "meal_type": "lunch",
    "meal_name": "Quinoa Tabbouleh Bowl",
    "nutrition": { … },
    "tier": "transition",
    "rationale": "…LLM explanation…"
  }
