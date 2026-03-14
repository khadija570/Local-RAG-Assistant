"""
vector.py — Gestion des embeddings et de la base vectorielle ChromaDB
Auteur : Khadija
Description : Chargement PDF, chunking, embedding via Ollama, stockage ChromaDB
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ─── Configuration ────────────────────────────────────────────────────────────

CHROMA_DB_DIR   = "./chroma_db"          # Répertoire de persistance ChromaDB
COLLECTION_NAME = "rag_pdf_collection"   # Nom de la collection
EMBED_MODEL     = "nomic-embed-text"     # Modèle d'embedding Ollama
CHUNK_SIZE      = 500                    # Taille des chunks (tokens approximatifs)
CHUNK_OVERLAP   = 50                     # Overlap entre chunks

# ─── Initialisation des embeddings ───────────────────────────────────────────

def get_embeddings() -> OllamaEmbeddings:
    """Retourne le modèle d'embedding Ollama configuré."""
    return OllamaEmbeddings(model=EMBED_MODEL)


# ─── Chargement et découpage du PDF ──────────────────────────────────────────

def load_and_split_pdf(pdf_path: str) -> List:
    """
    Charge un PDF et le découpe en chunks.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        Liste de documents (chunks) avec métadonnées
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    print(f"[INFO] Chargement du PDF : {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[INFO] {len(documents)} page(s) chargée(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"[INFO] {len(chunks)} chunks créés (taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Enrichissement des métadonnées
    source_name = Path(pdf_path).name
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["source_file"] = source_name
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


# ─── Gestion de la base vectorielle ──────────────────────────────────────────

def build_vector_store(chunks: List, reset: bool = False) -> Chroma:
    """
    Crée ou recharge la base ChromaDB à partir des chunks.

    Args:
        chunks   : Liste de documents (chunks) à indexer
        reset    : Si True, supprime et recrée la collection

    Returns:
        Instance Chroma prête à l'emploi
    """
    embeddings = get_embeddings()

    if reset and Path(CHROMA_DB_DIR).exists():
        print(f"[INFO] Suppression de l'ancienne base : {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)

    print(f"[INFO] Création de la base vectorielle dans : {CHROMA_DB_DIR}")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"[INFO] Base vectorielle créée avec {len(chunks)} vecteurs")
    return vector_store


def load_vector_store() -> Optional[Chroma]:
    """
    Charge une base ChromaDB existante depuis le disque.

    Returns:
        Instance Chroma ou None si la base n'existe pas
    """
    if not Path(CHROMA_DB_DIR).exists():
        print("[WARN] Aucune base vectorielle trouvée — veuillez d'abord indexer un PDF")
        return None

    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    count = vector_store._collection.count()
    print(f"[INFO] Base vectorielle chargée : {count} vecteurs")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Retourne un retriever basé sur la similarité cosinus.

    Args:
        vector_store : Instance Chroma
        k            : Nombre de chunks à récupérer par requête

    Returns:
        Retriever LangChain
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def vector_store_exists() -> bool:
    """Vérifie si une base vectorielle existe déjà sur le disque."""
    return Path(CHROMA_DB_DIR).exists() and any(Path(CHROMA_DB_DIR).iterdir())


def get_collection_info() -> dict:
    """Retourne des infos sur la collection actuelle."""
    if not vector_store_exists():
        return {"exists": False, "count": 0}
    vs = load_vector_store()
    if vs is None:
        return {"exists": False, "count": 0}
    return {
        "exists": True,
        "count": vs._collection.count(),
        "dir": CHROMA_DB_DIR,
        "model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def index_pdf(pdf_path: str, reset: bool = True) -> Chroma:
    """
    Pipeline complet : charger PDF → chunker → embedder → stocker.

    Args:
        pdf_path : Chemin vers le PDF
        reset    : Réinitialiser la base avant indexation

    Returns:
        Instance Chroma prête à l'emploi
    """
    chunks = load_and_split_pdf(pdf_path)
    vector_store = build_vector_store(chunks, reset=reset)
    return vector_store
