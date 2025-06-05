import re
import json

def extract_clean_json(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', raw)
        if not match:
            raise ValueError("No JSON block found in Gemini response")

        json_str = match.group(1)
        data = json.loads(json_str)

        # Unwrap if "target_values" exists
        if "target_values" in data:
            data = data["target_values"]

        return data
    except Exception as e:
        print(f"[ERROR] Failed to extract JSON: {e}")
        return {}
