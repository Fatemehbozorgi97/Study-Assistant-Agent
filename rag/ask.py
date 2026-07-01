from rag.retriever import retrieve
from models.llm import llm


def ask_question(question):

    docs = retrieve(question)

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = f"""
You are an AI study assistant.
only use the following context to answer the question.
Do not use any other information.
Do not make up any information that is not in the context.
Answer the question ONLY using the provided context.
If they say something that is not in the context, say you don't know.
If they say Hello,Hi,hello,hi , say Hi, and exxplain that you are an AI study assistant and can only answer questions based on the provided context.
If the answer is not in the context, say:
"I could not find the answer in the study materials."

---------------------

Context:

{context}

---------------------

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return str(response)