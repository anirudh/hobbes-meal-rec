# services/gemini.py
import os
import time
import random
import functools
import json
import numpy as np

from google import genai
from google.genai import types, errors as gerrors
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from services.db import GenerationFailure

# ───────────── API Key & Client ─────────────
GEMINI_API_KEY = settings.gemini_api_key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")
_client = genai.Client(api_key=GEMINI_API_KEY)

# ───────────── Model Names ─────────────
CHAT_MODEL  = "models/gemini-2.0-flash"
EMBED_MODEL = "models/gemini-embedding-exp-03-07"

# ───────────── Embedding (sync + cached + retry) ─────────────
@functools.lru_cache(maxsize=4096)
def embed(text: str) -> np.ndarray:
    """Return a 768-dim embedding for `text`, retrying on rate limits."""
    for attempt in range(5):
        try:
            resp = _client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            # resp.embeddings is a list, take first
            vals = resp.embeddings[0].values
            return np.array(vals, dtype=np.float32)
        except gerrors.ClientError as e:
            if getattr(e, "status", None) == "RESOURCE_EXHAUSTED":
                backoff = (2 ** attempt) + random.random()
                print(f"⚠️ 429, retrying embed in {backoff:.1f}s…")
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError("Embedding retries exhausted")


# ───────────── Generation (sync) ─────────────
def generate(
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 2000
) -> str:
    """Run a chat completion and return the LLM’s text response."""
    try:
        resp = _client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        # take the first candidate’s text
        return resp.candidates[0].content.parts[0].text
    except Exception as e:
        # bubble up, or you can choose to return "" here
        print(f"[ERROR] Gemini generation failed: {e}")
        raise


# ───────────── Error Logging ─────────────
async def log_failure_to_db(
    db: AsyncSession,
    user_id: int,
    stage: str,
    error: str,
    raw_input: str = "",
    raw_output: str = ""
) -> None:
    """
    Persist a Gemini generation failure to the database.
    """
    failure = GenerationFailure(
        user_id=user_id,
        meal_type="any",
        stage=stage,
        error_message=error,
        raw_input=raw_input,
        raw_output=raw_output,
    )
    db.add(failure)
    await db.commit()
