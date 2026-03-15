#  RAG ChatPDF — Local

Assistant Q&A sur tes PDF, 100% local.  
Stack : **LangChain · Ollama · ChromaDB · Streamlit**

---

## Architecture

```
PDF
 └── vector.py
      ├── Chargement (PyMuPDF)
      ├── Chunking (500 tokens / 50 overlap)
      ├── Embedding (nomic-embed-text via Ollama)
      └── ChromaDB (persistance locale)
           └── app.py (Streamlit)
                ├── Upload & indexation
                ├── Retrieval (top-4 chunks)
                └── Génération (Mistral via Ollama)
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. virtuel Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

### 3.Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama
```bash
# Install Ollama : https://ollama.com/download
ollama pull mistral              # LLM for the generation
ollama pull nomic-embed-text     # Embedding model
```

### 5. Run the application
```bash
streamlit run app.py
```

---

##  Project Structure

```
├── app.py            # Interface Streamlit + chaîne RAG
├── vector.py         # Chunking, embedding, ChromaDB
├── requirements.txt  # Dépendances Python
├── .gitignore        # Exclut chroma_db/ and venv/
└── README.md
```

> **Note :** the chroma_db/ folder is excluded from the repository (.gitignore).
It is automatically created when a PDF is indexed for the first time.

---

## Usage

1. Open `http://localhost:8501` On your Browser
2.Drag your PDF into the left panel → click **Index**
3. Ask your questions in the chat.

**Test Questions with `rag_test_document_AI.pdf` :**
- *"What is RAG ?"*
- *"What is the difference between Racall and precision ?"*
- *"Which open-source tools can be used for a local RAG?"*
- *"How do Transformers work?"*
- *"What is the AI act?"*

---

##  Configuration

In `app.py` and `vector.py`, you can modify :

| Parameter | Default value| Description |
|-----------|------------------|-------------|
| `LLM_MODEL` | `mistral` | Ollama Model (llama3, qwen2.5...) |
| `LLM_TEMP` | `0.2` | Generation Temperature |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding Model |
| `CHUNK_SIZE` | `500` | Chunks size |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_CHUNKS` | `4` | Number of retrieved chunks per query |

---


