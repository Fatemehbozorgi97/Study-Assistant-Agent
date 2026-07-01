from rag.retriever import retrieve


def rag_node(state):
    """
    Retrieves lecture chunks relevant to the question and stores them in state.
    Respects the lecture_ids detected by lecture_router when available.
    """
    question = state["question"]
    plan = state.get("plan", {})

    # skip retrieval for non-academic interactions
    if not plan.get("needs_rag", True):
        return {**state, "retrieved_chunks": [], "context": ""}

    lecture_ids = state.get("lecture_route", {}).get("lecture_ids", [])
    lecture_filter = lecture_ids if lecture_ids else None

    docs = retrieve(question, lecture_filter=lecture_filter, k=5)
    chunks = [d.page_content for d in docs]
    context = "\n\n".join(chunks)

    return {
        **state,
        "retrieved_chunks": chunks,
        "context": context
    }
