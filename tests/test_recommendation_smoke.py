import httpx, os, json

BASE = "http://127.0.0.1:8000"

def test_full_recommendation_roundtrip():
    uid = 555
    with httpx.Client() as c:
        # 1. upsert user
        c.post(f"{BASE}/api/v1/users",
               json={"id": uid, "age": 30, "sex": "male",
                     "weight_kg": 70, "height_cm": 175,
                     "activity_level": "3", "exercise_frequency_per_week": 3})
        # 2. ensure targets exist
        os.system(f"python -m scripts.init_targets --user {uid}")
        # 3. call recommender
        r = c.post(f"{BASE}/api/v1/recommendations",
                   json={"user_id": uid, "meal_type": "any"})
        assert r.status_code == 200
        data = r.json()["recommendations"]
        # 4. We expect ≥1 meal in each tier
        assert all(len(data[tier]) > 0 for tier in ("familiar", "transition", "target"))
