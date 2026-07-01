from rag.retriever import retrieve

docs = retrieve("what is cnn")

print("\n\nRETRIEVED DOCS:\n")

for i, doc in enumerate(docs):

    print("=" * 50)

    print(f"DOC {i+1}")

    print(doc.page_content[:1000])