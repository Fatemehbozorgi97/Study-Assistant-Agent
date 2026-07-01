from rag.ingest import ingest_folder

if __name__ == "__main__":
    ingest_folder("data/raw")

    question = input("\nAsk a question:\n> ")
    print("Done ingestion. Now run your RAG app.")