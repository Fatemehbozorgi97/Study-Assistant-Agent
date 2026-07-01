from models.llm import llm
from prompts.history_utils import format_history
from prompts.json_utils import safe_parse_json

ALLOWED_MODES = {"explain", "quiz", "summarize", "mixed", "greeting", "course_info"}
ALLOWED_DIFFICULTY = {"beginner", "intermediate", "advanced"}

PLANNER_PROMPT = """
You are a routing and planning agent for an AI Study Tutor.

Analyze the student's question — and the conversation history if provided — and decide
how the tutoring system should respond.

Return ONLY raw valid JSON — no explanations, no markdown fences.

Allowed modes:
- explain      → questions asking for explanations of ML/DL concepts from the lectures
- quiz         → questions asking for tests, MCQs, or practice
- summarize    → questions asking for summaries or brief notes of lecture content
- mixed        → multi-part or compound requests
- greeting     → greetings, casual small talk
- course_info  → questions about the course itself: its name, code, credits, teachers,
                 language, university, or "what topics/what is this course about"
                 (as opposed to questions about a specific lecture's content)

RAG rules:
- needs_rag = true  for: ML/DL concepts, lecture-specific content, exam questions
- needs_rag = false for: greetings, casual conversation, course_info (answered from
  the official course profile, not lecture retrieval)

IMPORTANT: if the question contains pronouns or references like "that", "it", "the same topic",
resolve them using the conversation history before deciding the mode and focus.

Return format:
{
  "mode": "explain|quiz|summarize|mixed|greeting",
  "needs_rag": true,
  "difficulty": "beginner|intermediate|advanced",
  "focus": ["topic1", "topic2"],
  "response_style": "short|step-by-step|detailed"
}
"""


def planner_node(state):
    question = state["question"]
    history_block = format_history(state.get("chat_history", []))

    prompt = f"{PLANNER_PROMPT}\n\n{history_block}\n\nQuestion: {question}"
    response = llm.invoke(prompt)

    plan = safe_parse_json(response) or {
        "mode": "explain",
        "needs_rag": True,
        "difficulty": "beginner",
        "focus": [],
        "response_style": "step-by-step"
    }

    if plan.get("mode") not in ALLOWED_MODES:
        plan["mode"] = "explain"

    if plan.get("difficulty") not in ALLOWED_DIFFICULTY:
        plan["difficulty"] = "beginner"

    return {
        **state,
        "plan": plan
    }
