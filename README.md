# 📚 RAG ChatPDF — Local

Assistant Q&A sur tes PDF, 100% local.  
Stack : **LangChain · Ollama · ChromaDB · Streamlit**

---

## 🏗️ Architecture

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

## 🚀 Installation

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/TON_REPO.git
cd TON_REPO
```

### 2. Environnement virtuel
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Installer et lancer Ollama
```bash
# Installer Ollama : https://ollama.com/download
ollama pull mistral              # LLM pour la génération
ollama pull nomic-embed-text     # Modèle d'embedding
```

### 5. Lancer l'application
```bash
streamlit run app.py
```

---

## 📁 Structure du projet

```
├── app.py            # Interface Streamlit + chaîne RAG
├── vector.py         # Chunking, embedding, ChromaDB
├── requirements.txt  # Dépendances Python
├── .gitignore        # Exclut chroma_db/ et venv/
└── README.md
```

> **Note :** Le dossier `chroma_db/` est exclu du dépôt (`.gitignore`).  
> Il se crée automatiquement à la première indexation d'un PDF.

---

## 💬 Utilisation

1. Ouvre `http://localhost:8501` dans ton navigateur
2. Glisse ton PDF dans le panneau gauche → clique **Indexer**
3. Pose tes questions dans le chat !

**Questions de test avec `rag_test_document_AI.pdf` :**
- *"Qu'est-ce que le RAG ?"*
- *"Quelle est la différence entre Précision et Recall ?"*
- *"Quels outils open-source utiliser pour un RAG local ?"*
- *"Comment fonctionnent les Transformers ?"*
- *"Qu'est-ce que l'AI Act ?"*

---

## ⚙️ Configuration

Dans `app.py` et `vector.py`, tu peux modifier :

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `LLM_MODEL` | `mistral` | Modèle Ollama (llama3, qwen2.5...) |
| `LLM_TEMP` | `0.2` | Température de génération |
| `EMBED_MODEL` | `nomic-embed-text` | Modèle d'embedding |
| `CHUNK_SIZE` | `500` | Taille des chunks |
| `CHUNK_OVERLAP` | `50` | Overlap entre chunks |
| `TOP_K_CHUNKS` | `4` | Chunks récupérés par requête |

---

## 👩‍💻 Auteur

Khadija — ENSA Agadir · Data Science & IA
