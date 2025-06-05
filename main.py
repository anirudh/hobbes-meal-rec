from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from config import settings
from api.v1.router import api_router



app = FastAPI(title="Meal-Rec API", version="1.0.0")

# CORS (public demo only – lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.env_name}
