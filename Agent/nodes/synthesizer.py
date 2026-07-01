import json


def synthesizer_node(state):

    answer = state.get("answer", "")
    plan = state.get("plan", {})

    mode = plan.get("mode", "explain")

    # -----------------------------------
    # try parse structured output
    # -----------------------------------
    try:
        data = json.loads(answer)
    except:
        return {
            **state,
            "answer": answer
        }

    # -----------------------------------
    # format based on mode
    # -----------------------------------

    if mode == "quiz":

        text = "🧪 Quiz Time:\n\n"

        for i, q in enumerate(data.get("questions", []), 1):
            text += f"{i}. {q['q']}\n"
            for opt in q["options"]:
                text += f"   - {opt}\n"
            text += "\n"

        return {**state, "answer": text}


    if mode == "explain":

        text = "📘 Explanation:\n\n"
        text += data.get("explanation", "") + "\n\n"

        if data.get("key_points"):
            text += "Key Points:\n"
            for p in data["key_points"]:
                text += f"- {p}\n"

        return {**state, "answer": text}


    if mode == "summarize":

        text = "📝 Summary:\n\n"
        text += data.get("summary", "") + "\n\n"

        if data.get("key_topics"):
            text += "Key Topics:\n"
            for t in data["key_topics"]:
                text += f"- {t}\n"

        return {**state, "answer": text}


    return {**state, "answer": str(data)}