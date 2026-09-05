from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os

load_dotenv()

PDF_PATH = "./OOPS_Step_by_Step_for_CSE_Students.pdf"
DB_PATH = "vectorstore/"

def ingest():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("Storing in FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(DB_PATH)

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest()