from rag.ask import ask_question


while True:

    question = input("\nAsk a question (or type exit):\n> ")

    if question.lower() == "exit":
        break

    answer = ask_question(question)

    print("\nAI Tutor:\n")

    print(answer)