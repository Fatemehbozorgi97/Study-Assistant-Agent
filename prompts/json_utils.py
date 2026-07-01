import json
import re


def safe_parse_json(text: str) -> dict | None:
    """
    Parse JSON from an LLM response that may be wrapped in markdown fences.
    Returns the parsed dict, or None if parsing fails.
    """
    if not text:
        return None

    # strip ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: find the first {...} block in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None
