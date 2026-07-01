from models.llm import llm
from prompts.tutor_prompt import TUTOR_SYSTEM_PROMPT
from prompts.history_utils import format_history


def summarize_node(state):
    question = state["question"]
    context = state.get("context", "")
    history_block = format_history(state.get("chat_history", []))

    prompt = f"""
{TUTOR_SYSTEM_PROMPT}

{history_block}

Summarize the following lecture content clearly and concisely in response to the student's request.

STUDENT REQUEST:
{question}

LECTURE CONTENT:
{context}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": str(response)
    }