# 🤖 RAG AI Document Assistant (Ollama + LangChain + Streamlit)

A powerful **Retrieval-Augmented Generation (RAG)** application that lets you chat with your PDF documents using a local LLM powered by Ollama.

This project supports **multi-document intelligence**, **chat memory**, **source tracking**, and a **modern ChatGPT-style UI**.

---

## 🚀 Features

* 📂 Upload and process multiple PDFs
* 🧠 Semantic search using embeddings
* 💬 Chat with your documents (context-aware)
* 🧾 Source tracking (see which file answered)
* 🧠 Conversation memory (like ChatGPT)
* 🖼️ Image upload + preview
* ⚡ Streaming responses (typing effect)
* 🎨 Clean Streamlit UI
* 🔒 Fully local (no API required)

---

## 🛠️ Tech Stack

* **LLM:** Ollama (Llama 3.2)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (MiniLM)
* **UI:** Streamlit
* **PDF Parsing:** PyPDF

---

## 📁 Project Structure

```
RagOllamaWithUI/
│
├── app.py              # Main Streamlit application
├── ingest.py           # (Optional) PDF ingestion script
├── rag_pipeline.py     # (Optional) CLI RAG pipeline
├── data/               # Uploaded PDFs
├── chroma_db/          # Vector database
└── venv/               # Virtual environment
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/srinivasw54/rag-ollama-project.git
cd rag-ollama-project/RAGOllama
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
```

---

### 3️⃣ Activate environment

**Windows (PowerShell):**

```
.\venv\Scripts\Activate.ps1
```

---

### 4️⃣ Install dependencies

```
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-ollama langchain-chroma chromadb sentence-transformers pypdf streamlit
```

---

## 🤖 Setup Ollama

Install Ollama from: https://ollama.com

Then run:

```
ollama pull llama3.2
ollama run llama3.2
```

---

## ▶️ Run the App

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📖 How to Use

1. Upload PDFs from sidebar
2. Click **"Process Documents"**
3. Ask questions in chat
4. View answers + sources

---

## 🧠 How It Works

1. PDFs are loaded and split into chunks
2. Embeddings are generated using HuggingFace
3. Stored in Chroma vector database
4. User query retrieves relevant chunks
5. LLM (Ollama) generates answer using:

   * Context
   * Chat history

---

## 🔥 Example Queries

* "What is the career objective?"
* "Summarize this document"
* "Who is mentioned in the PDF?"
* "Explain the concept in simple terms"

---

## ⚠️ Limitations

* Images inside PDFs are not processed (text only)
* Requires Ollama running locally
* Large PDFs may take time to process

---

## 🚀 Future Improvements

* 🔍 OCR for images inside PDFs
* 🧠 Multi-modal support (LLaVA)
* 🌍 Deployment (Render / HuggingFace Spaces)
* 🔐 User authentication
* 📊 Document analytics

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

**Srinivasulu N**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
