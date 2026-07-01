import json
from pathlib import Path

def load_course_profile():
    path = Path("data/course_profile.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)