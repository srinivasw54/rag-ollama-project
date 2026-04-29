from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

CHROMA_DIR = './chroma_db'
EMBED_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'llama3.2'
TOP_K = 3

RAG_PROMPT = """You are a helpful assistant.
Answer clearly using the context.

Context:
{context}

Question: {question}
Answer:"""


def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever):
    llm = OllamaLLM(model=LLM_MODEL)

    prompt = PromptTemplate.from_template(RAG_PROMPT)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    retriever = load_retriever()
    chain = build_chain(retriever)

    while True:
        query = input("\nAsk (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = chain.invoke(query)
        print("\nAnswer:\n", answer)