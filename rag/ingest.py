import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


PERSIST_DIR = "data/chroma_db"


def get_lecture_id(filename: str):
    """
    Extract lecture number from filename
    Examples:
        AML2425Adv_Lecture4.pdf -> Lecture 4
        Lecture2_tmp.pdf -> Lecture 2
    """
    name = filename.lower()

    for i in range(1, 10):
        if f"lecture{i}" in name:
            return i

    return "General"


def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return []

    return loader.load()


def ingest_file(file_path, filename, vectorstore):
    print(f"\n📥 Ingesting: {filename}")

    docs = load_file(file_path)
    print(f"Loaded pages/chunks: {len(docs)}")

    # metadata enrichment
    lecture_id = get_lecture_id(filename)

    for d in docs:

        d.metadata["source"] = filename
        d.metadata["lecture"] = lecture_id

        if lecture_id == 0:
            d.metadata["type"] = "summary"
        else:
            d.metadata["type"] = "lecture"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    vectorstore.add_documents(chunks)

    return len(chunks)


def build_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11435")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def ingest_folder(raw_folder="data/raw"):
    vectorstore = build_vectorstore()

    total_chunks = 0

    for filename in os.listdir(raw_folder):
        file_path = os.path.join(raw_folder, filename)

        if filename.endswith(".pdf") or filename.endswith(".txt"):
            total_chunks += ingest_file(file_path, filename, vectorstore)

    vectorstore.persist()

    print("\n✅ INGESTION COMPLETE")
    print(f"Total chunks: {total_chunks}")