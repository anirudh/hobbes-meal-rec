# api/v1/router.py
from fastapi import APIRouter

from . import users, meals, recs, prefs

api_router = APIRouter()

api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(meals.router, prefix="/meals", tags=["Meals"])
api_router.include_router(recs.router,  prefix="/recommendations", tags=["Recommendations"])

# preferences live *under* the user resource
api_router.include_router(
    prefs.router,
    prefix="/users",          # results in /users/{user_id}/preferences
    tags=["Preferences"],
)
