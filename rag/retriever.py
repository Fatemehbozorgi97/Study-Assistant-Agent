from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11435"
)

vectorstore = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)


def retrieve(question, lecture_filter=None, k=5):

    if lecture_filter:

        print(f"📚 Lecture filter: {lecture_filter}")

        docs = vectorstore.similarity_search(
            question,
            k=k,
            filter={
                "lecture": {
                    "$in": lecture_filter
                }
            }
        )

    else:

        print("🌍 Global search")

        docs = vectorstore.similarity_search(
            question,
            k=k
        )

    return docs