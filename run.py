from Agent.graph import app

while True:
    question = input("\nYou: ")

    result = app.invoke({
        "question": question,
        "route": "",
        "context": "",
        "answer": ""
    })

    print("\nAssistant:")
    print(result["answer"])