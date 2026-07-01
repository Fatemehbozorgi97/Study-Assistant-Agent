import json

from models.llm import llm
from prompts.json_utils import safe_parse_json

LECTURE_MAP = {
    1: "ML basics, regression, logistic regression, bias-variance",
    2: "Neural network representation, forward propagation",
    3: "MLP, backpropagation, gradient descent",
    4: "CNN, convolution, pooling, feature maps",
    6: "RNN, LSTM, sequence modeling"
}
import re

def detect_lecture(question):

    q = question.lower()

    match = re.search(r"lecture\s*(\d+)", q)

    if match:
        return f"Lecture {match.group(1)}"

    return None
def lecture_router_node(state):
    question = state["question"]

    prompt = f"""
You are a lecture classifier.

Lectures:
{json.dumps(LECTURE_MAP, indent=2)}

Question:
{question}

Return ONLY JSON:
{{
  "lecture_ids": [1,2],
  "confidence": 0.0-1.0,
  "needs_general_course": true/false
}}
"""

    response = llm.invoke(prompt)

    result = safe_parse_json(response) or {
        "lecture_ids": [],
        "needs_general_course": True,
        "confidence": 0.0
    }

    return {
        **state,
        "lecture_route": result
    }