import json
from models.llm import llm
from prompts.history_utils import format_history
from prompts.json_utils import safe_parse_json


QUIZ_PROMPT = """
You are an expert AI Study Tutor that creates quizzes for Applied Machine Learning students.

Rules:
- Use ONLY the provided context when available.
- If context is weak, draw on general ML/DL knowledge.
- Adjust difficulty based on the plan.
- If the student has weak topics listed, prioritise questions on those topics.
- Output ONLY valid JSON — no explanations outside the JSON block.

Return format:
{
  "questions": [
    {
      "q": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A"
    }
  ]
}

Difficulty guidelines:
- beginner      → simple definitions and recall
- intermediate  → concept understanding + application
- advanced      → nuanced reasoning, edge cases, maths
"""


def quiz_node(state):
    question = state["question"]
    plan = state.get("plan", {})
    context = state.get("context", "")

    difficulty = plan.get("difficulty", "beginner")
    focus = plan.get("focus", [])

    # Phase 2 — adaptive: if the student has known weak topics, bump difficulty
    # and steer questions toward those areas
    weak_topics = state.get("weak_topics", [])
    student_memory = state.get("student_memory", {})
    total_sessions = student_memory.get("total_quiz_sessions", 0)

    if weak_topics and total_sessions > 0:
        # escalate difficulty after first session
        if difficulty == "beginner":
            difficulty = "intermediate"
        elif difficulty == "intermediate":
            difficulty = "advanced"

    adaptive_note = ""
    if weak_topics:
        adaptive_note = (
            f"\nStudent's previously identified weak topics: {', '.join(weak_topics)}. "
            "Prioritise questions that address these areas."
        )

    history_block = format_history(state.get("chat_history", []))

    prompt = f"""
{QUIZ_PROMPT}

Difficulty: {difficulty}
Focus topics: {focus}
{adaptive_note}

Context (lecture material):
{context}

{history_block}

Student request:
{question}
"""

    response = llm.invoke(prompt)

    quiz = safe_parse_json(response) or {
        "questions": [
            {
                "q": "What is the main idea of the requested topic?",
                "options": ["A. Option A", "B. Option B", "C. Option C", "D. Option D"],
                "answer": "A"
            }
        ]
    }

    return {
        **state,
        "answer": json.dumps(quiz, indent=2)
    }
