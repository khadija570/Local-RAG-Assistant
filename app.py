"""
app.py — Interface Streamlit pour le RAG ChatPDF local
Auteur : Khadija
Description : Upload PDF, indexation, chat Q&A avec Ollama + LangChain + ChromaDB
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from vector import (
    index_pdf,
    load_vector_store,
    get_retriever,
    vector_store_exists,
    get_collection_info,
    EMBED_MODEL,
)

# ─── Configuration Streamlit ─────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG ChatPDF — Local",
    page_icon="📚",
    layout="wide",
)

# ─── Constantes ──────────────────────────────────────────────────────────────

LLM_MODEL     = "mistral"   # Modèle Ollama pour la génération (ou "llama3", "qwen2.5")
LLM_TEMP      = 0.2         # Température : bas = plus déterministe
TOP_K_CHUNKS  = 4           # Nombre de chunks récupérés par requête

# ─── Prompt RAG ──────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """Tu es un assistant expert qui répond aux questions en te basant UNIQUEMENT sur le contexte fourni.
Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne sais pas — ne génère pas de réponse inventée.

Contexte extrait du document :
{context}

Question de l'utilisateur :
{question}

Réponse (en français, claire et structurée) :"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# ─── Initialisation de l'état de session ─────────────────────────────────────

if "messages"      not in st.session_state: st.session_state.messages      = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = None
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = None
if "qa_chain"      not in st.session_state: st.session_state.qa_chain      = None


# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_llm():
    """Initialise et met en cache l'instance du LLM Ollama."""
    return ChatOllama(model=LLM_MODEL, temperature=LLM_TEMP)


def build_qa_chain(vector_store):
    """Construit la chaîne RetrievalQA LangChain."""
    llm = get_llm()
    retriever = get_retriever(vector_store, k=TOP_K_CHUNKS)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )


def answer_question(question: str) -> dict:
    """Interroge la chaîne RAG et retourne réponse + sources."""
    if st.session_state.qa_chain is None:
        return {"result": "Veuillez d'abord indexer un PDF.", "source_documents": []}
    result = st.session_state.qa_chain.invoke({"query": question})
    return result


# ─── UI ──────────────────────────────────────────────────────────────────────

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG ChatPDF")
    st.caption("Local · LangChain · Ollama · ChromaDB")
    st.divider()

    # Infos modèles
    st.markdown("**Modèles actifs**")
    st.markdown(f"- 🤖 LLM : `{LLM_MODEL}`")
    st.markdown(f"- 🧠 Embedding : `{EMBED_MODEL}`")
    st.divider()

    # Upload PDF
    st.markdown("**📄 Charger un PDF**")
    uploaded_file = st.file_uploader(
        label="Glisse ton PDF ici",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            index_btn = st.button("🔄 Indexer", use_container_width=True, type="primary")
        with col2:
            reset_btn = st.button("🗑️ Reset", use_container_width=True)

        if index_btn:
            # Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner(f"Indexation de **{uploaded_file.name}**..."):
                try:
                    vs = index_pdf(tmp_path, reset=True)
                    st.session_state.vector_store = vs
                    st.session_state.pdf_name     = uploaded_file.name
                    st.session_state.qa_chain     = build_qa_chain(vs)
                    st.session_state.messages     = []
                    st.success(f"✅ {uploaded_file.name} indexé !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    os.unlink(tmp_path)

        if reset_btn:
            st.session_state.vector_store = None
            st.session_state.qa_chain     = None
            st.session_state.pdf_name     = None
            st.session_state.messages     = []
            st.rerun()

    st.divider()

    # Charger base existante
    if vector_store_exists() and st.session_state.vector_store is None:
        st.markdown("**💾 Base existante détectée**")
        if st.button("⚡ Charger la base", use_container_width=True):
            with st.spinner("Chargement..."):
                vs = load_vector_store()
                if vs:
                    st.session_state.vector_store = vs
                    st.session_state.qa_chain     = build_qa_chain(vs)
                    st.success("Base chargée !")
                    st.rerun()

    # Infos collection
    info = get_collection_info()
    if info["exists"]:
        st.markdown("**📊 Collection ChromaDB**")
        st.metric("Vecteurs indexés", info["count"])
        st.caption(f"Chunk : {info['chunk_size']}t / overlap : {info['chunk_overlap']}t")

    st.divider()

    # Bouton effacer historique
    if st.button("🧹 Effacer le chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Zone principale ───────────────────────────────────────────────────────────
st.title("💬 RAG ChatPDF — Assistant local")

if st.session_state.pdf_name:
    st.info(f"📄 Document actif : **{st.session_state.pdf_name}** · {TOP_K_CHUNKS} chunks récupérés par requête")
else:
    st.warning("⚠️ Aucun document indexé. Charge un PDF dans le panneau de gauche.")

st.divider()

# Affichage de l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Afficher les sources si disponibles
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources utilisées"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "?")
                    chunk_idx = src.metadata.get("chunk_index", "?")
                    st.markdown(f"**Chunk {i}** — page {page + 1} (chunk #{chunk_idx})")
                    st.caption(src.page_content[:300] + "...")

# Zone de saisie
question = st.chat_input(
    placeholder="Pose ta question sur le document... (ex: Qu'est-ce qu'un RAG ?)",
    disabled=(st.session_state.qa_chain is None),
)

if question:
    # Ajout question utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Génération réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le document..."):
            start = time.time()
            result = answer_question(question)
            elapsed = time.time() - start

        answer   = result.get("result", "Aucune réponse générée.")
        sources  = result.get("source_documents", [])

        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.1f}s · {len(sources)} chunk(s) utilisé(s)")

        if sources:
            with st.expander("📎 Sources utilisées"):
                for i, src in enumerate(sources, 1):
                    page = src.metadata.get("page", "?")
                    chunk_idx = src.metadata.get("chunk_index", "?")
                    st.markdown(f"**Chunk {i}** — page {page + 1} (chunk #{chunk_idx})")
                    st.caption(src.page_content[:300] + "...")

    # Sauvegarde dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
