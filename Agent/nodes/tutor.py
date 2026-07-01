from models.llm import llm
from prompts.history_utils import format_history

TUTOR_SYSTEM_PROMPT = """
You are a strict AI tutor for Applied Machine Learning at the University of Bologna.

RULES:
- For questions about the course itself (name, teachers, credits, language, topics covered), trust the COURSE INFO below as ground truth.
- For questions about lecture content, use ONLY the provided lecture context as your source of truth.
- If lecture-specific information is missing from the context say: "I could not find this in the lecture materials."
- Do NOT hallucinate or invent facts.
- If there is conversation history, maintain continuity — refer back to earlier points when relevant.
"""


def tutor_node(state):
    question = state["question"]
    context = state.get("context", "")
    lecture_ids = state.get("lecture_route", {}).get("lecture_ids", [])
    history_block = format_history(state.get("chat_history", []))
    course_profile = state.get("course_profile", {})

    prompt = f"""{TUTOR_SYSTEM_PROMPT}

COURSE INFO:
{course_profile}

LECTURE IDS IN SCOPE: {lecture_ids}

LECTURE CONTEXT:
{context}

{history_block}

STUDENT QUESTION:
{question}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response
    }