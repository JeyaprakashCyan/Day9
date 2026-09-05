from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "vectorstore/"

def load_chain():
    embeddings = OpenAIEmbeddings()

    print("Loading FAISS DB...")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
# 🔥 LangSmith Trace Wrapper
@traceable(name="RAG-QA")
def ask_question(chain, query):
    return chain.invoke(
        {"query": query},
        config={
            "metadata": {"app": "pdf-rag"},
            "tags": ["rag", "faiss", "pdf"]
        }
    )


def ask_question(chain):
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        result = chain.invoke({"query": query})

        print("\n📌 Answer:")
        print(result["result"])

        print("\n📚 Sources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "unknown"))


if __name__ == "__main__":
    chain = load_chain()
    ask_question(chain)