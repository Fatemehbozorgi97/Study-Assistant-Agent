from models.llm import llm
from prompts.json_utils import safe_parse_json

CRITIC_PROMPT = """
You are a hallucination-detection critic for an AI tutoring system.

Your job:
1. Compare the answer against the provided context.
2. Decide whether the answer is GROUNDED (supported by context) or HALLUCINATED (contains unsupported claims).
3. If hallucinated, produce a corrected answer that stays strictly within the context.

Rules:
- If no context is available, mark as grounded (nothing to contradict).
- Be conservative: flag only clear contradictions or invented facts, not minor phrasing differences.

Return ONLY valid JSON:
{
  "grounded": true,
  "issues": [],
  "revised_answer": "..."
}

Fields:
- grounded: true if the answer is supported by context, false otherwise
- issues: list of specific unsupported claims (empty list if grounded)
- revised_answer: corrected answer if not grounded; otherwise identical to the original answer
"""

REFLECT_PROMPT = """
You are a self-reflective AI tutor.

Your previous answer was flagged as potentially containing unsupported claims.
Issues identified:
{issues}

Original answer:
{answer}

Context from lecture materials:
{context}

Write a corrected, honest answer that:
- Stays strictly within the provided context
- Clearly states "I could not find this in the lecture materials" for any missing information
- Does not introduce any new claims not in the context
"""

MAX_REFLECTIONS = 1  # prevent infinite loops


def critic_node(state):
    """
    Phase 3: Checks the answer for hallucinations against retrieved context.
    If issues are found and we have not yet reflected, triggers a self-reflection
    pass to produce a corrected answer. Sets critic_passed in state.
    """
    answer = state.get("answer", "")
    context = state.get("context", "")
    question = state.get("question", "")
    reflection_count = state.get("reflection_count", 0)

    # nothing to check against — skip
    if not context.strip():
        return {**state, "critic_passed": True}

    # quiz JSON is structured output — skip hallucination check
    plan = state.get("plan", {})
    if plan.get("mode") == "quiz":
        return {**state, "critic_passed": True}

    prompt = f"""
{CRITIC_PROMPT}

Context:
{context}

Question:
{question}

Answer to verify:
{answer}
"""

    response = llm.invoke(prompt)

    result = safe_parse_json(response)
    if result is None:
        # unparseable critic output — trust the answer rather than discard it
        return {**state, "critic_passed": True}

    grounded = result.get("grounded", True)
    issues = result.get("issues", [])

    if grounded or not issues:
        return {**state, "critic_passed": True}

    # --- Self-reflection (Phase 3) ---
    # Only attempt one revision to avoid loops
    if reflection_count >= MAX_REFLECTIONS:
        # append a disclaimer rather than looping again
        disclaimer = (
            "\n\n---\n*Note: Some parts of this answer could not be fully verified "
            "against the lecture materials. Please cross-check with your slides.*"
        )
        return {
            **state,
            "answer": answer + disclaimer,
            "critic_passed": False
        }

    reflect_prompt = REFLECT_PROMPT.format(
        issues="\n".join(f"- {i}" for i in issues),
        answer=answer,
        context=context
    )

    revised = llm.invoke(reflect_prompt)

    return {
        **state,
        "answer": revised,
        "critic_passed": False,
        "reflection_count": reflection_count + 1
    }
