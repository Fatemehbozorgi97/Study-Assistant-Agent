def format_history(chat_history: list, max_turns: int = 4) -> str:
    """
    Format the last `max_turns` exchanges from chat_history into a
    readable block for inclusion in agent prompts.
    Returns an empty string when there is no prior history.
    """
    if not chat_history:
        return ""

    # take only the last max_turns * 2 messages (each turn = user + assistant)
    recent = chat_history[-(max_turns * 2):]

    lines = ["CONVERSATION HISTORY (most recent exchanges):"]
    for msg in recent:
        role = "Student" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '').strip()}")

    return "\n".join(lines)
