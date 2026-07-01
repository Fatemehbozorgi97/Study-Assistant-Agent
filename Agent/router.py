def router_node(state):

    question = state["question"].lower()

    # -------------------------
    # quiz requests
    # -------------------------
    if any(word in question for word in [
        "quiz",
        "mcq",
        "multiple choice",
        "question",
        "test",
        "test me"
    ]):
        route="quiz"
        print("ROUTE:", route)
        return {
            **state,
            "route": "quiz"
        }

    # -------------------------
    # summarize requests
    # -------------------------
    if any(word in question for word in [
        "summarize",
        "summary",
        "short explanation"
    ]):
        route="summarize"
        print("ROUTE:", route)
        return {
            **state,
            "route": "summarize"
        }

    # -------------------------
    # default tutor mode
    # -------------------------
    route="tutor"
    print("ROUTE:", route)
    return {
        **state,
        "route": "tutor"
    }