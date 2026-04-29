import streamlit as st
import os
import time

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ───────────────────────────
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

# ── Prompt ───────────────────────────
RAG_PROMPT = """You are a helpful assistant.

Use the conversation history and context to answer clearly.

Chat History:
{history}

Context:
{context}

Question: {question}
Answer:"""


# ── PDF Processing (with metadata) ───
def process_pdfs(uploaded_files):
    docs = []
    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join("data", file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path)
        file_docs = loader.load()

        # ✅ Add file metadata
        for doc in file_docs:
            doc.metadata["source"] = file.name

        docs.extend(file_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )


# ── History Formatter ────────────────
def format_history(messages):
    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history


# ── Load System ──────────────────────
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model=LLM_MODEL)
    prompt = PromptTemplate.from_template(RAG_PROMPT)

    return retriever, llm, prompt


# ── UI Config ────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Custom Styling ───────────────────
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────
with st.sidebar:
    st.title("⚙️ Control Panel")

    # PDF Upload
    uploaded_files = st.file_uploader(
        "📂 Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("📥 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                process_pdfs(uploaded_files)
                st.cache_resource.clear()
            st.success("✅ Documents processed!")
        else:
            st.warning("Upload files first")

    # Image Upload
    image_files = st.file_uploader(
        "🖼️ Upload Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if image_files:
        st.markdown("### 🖼️ Preview")
        for img in image_files:
            st.image(img, use_column_width=True)

    # Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")
    st.markdown("💡 Tips:")
    st.markdown("- Ask specific questions")
    st.markdown("- Use follow-ups")
    st.markdown("- Upload multiple PDFs")

# ── Main UI ──────────────────────────
st.title("🤖 AI Document Assistant")
st.caption("Chat with your PDFs + Images")

# Load system
retriever, llm, prompt_template = load_system()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────
if user_input := st.chat_input("Ask your question..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            # Retrieve docs
            docs = retriever.invoke(user_input)

            context = "\n\n---\n\n".join(
                doc.page_content for doc in docs
            )

            history = format_history(st.session_state.messages)

            final_prompt = prompt_template.format(
                context=context,
                question=user_input,
                history=history
            )

            full_response = llm.invoke(final_prompt)

        # ── Streaming Effect ───────────
        response_placeholder = st.empty()
        streamed_text = ""

        for char in full_response:
            streamed_text += char
            response_placeholder.markdown(streamed_text)
            time.sleep(0.005)

        # ── Source Display ────────────
        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown File")

                st.markdown(f"**📁 File:** {source}")
                st.write(doc.page_content[:400] + "...")

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })