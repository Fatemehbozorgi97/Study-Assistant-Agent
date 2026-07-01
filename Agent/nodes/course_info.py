import json

from models.llm import llm

COURSE_INFO_PROMPT = """
You are an AI study assistant answering a question about the course itself
(not about lecture content) — e.g. its name, code, credits, teachers,
language, or the list of topics it covers.

Answer using ONLY the OFFICIAL COURSE INFORMATION JSON below. Do not invent
details that are not present in it. If the student asks for something not
present in this JSON (e.g. exam dates, office hours), say so explicitly
rather than guessing.

OFFICIAL COURSE INFORMATION:
{course_info}

STUDENT QUESTION:
{question}
"""


def course_info_node(state):
    """
    Answers questions about the course itself (name, credits, teachers,
    topics covered) directly from course_profile, bypassing lecture
    retrieval entirely. `context` is set to the course profile JSON so the
    Critic Agent still has something concrete to verify the answer against.
    """
    course_profile = state.get("course_profile", {})
    course_info = json.dumps(course_profile, indent=2)

    prompt = COURSE_INFO_PROMPT.format(course_info=course_info, question=state["question"])
    response = llm.invoke(prompt)

    return {
        **state,
        "context": course_info,
        "answer": response,
    }
