"""FAISS vector store construction."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from app.config import Settings


def create_retriever(
    documents: list[Document],
    embeddings: Embeddings,
    settings: Settings,
) -> BaseRetriever:
    """Build a FAISS index and return a configured retriever."""
    if not documents:
        raise ValueError("No document chunks were created.")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store.as_retriever(
        search_kwargs={"k": settings.top_k_results}
    )
