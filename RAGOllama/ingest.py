from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

DATA_DIR = './data'
CHROMA_DIR = './chroma_db'
EMBED_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Add PDF files to ./data and run again.")
        exit(1)

    loader = DirectoryLoader(
        data_dir,
        glob='**/*.pdf',
        loader_cls=PyPDFLoader
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Auto-persist (no .persist() needed)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} pages")

    if not docs:
        print("No PDFs found.")
        exit(1)

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    create_vector_store(chunks)

    print("✅ Ingestion complete")