def memory_node(state):
    """
    Appends the current turn to chat_history.
    For quiz sessions, records the planner's focus topics as weak areas so
    future sessions can adapt difficulty — no extra LLM call needed.
    """
    history = state.get("chat_history", [])
    student_memory = state.get("student_memory", {})
    weak_topics = state.get("weak_topics", [])
    plan = state.get("plan", {})

    # append current exchange
    updated_history = history + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["answer"]}
    ]

    # Phase 2: track weak topics from quiz sessions using planner's focus list
    if plan.get("mode") == "quiz":
        new_topics = [t.lower() for t in plan.get("focus", []) if t]

        # deduplicate while preserving order
        for topic in new_topics:
            if topic and topic not in weak_topics:
                weak_topics = weak_topics + [topic]

        seen = student_memory.get("quiz_topics_seen", [])
        for topic in new_topics:
            if topic not in seen:
                seen = seen + [topic]

        student_memory = {
            **student_memory,
            "quiz_topics_seen": seen,
            "weak_topics": weak_topics,
            "total_quiz_sessions": student_memory.get("total_quiz_sessions", 0) + 1
        }

    return {
        **state,
        "chat_history": updated_history,
        "weak_topics": weak_topics,
        "student_memory": student_memory
    }
