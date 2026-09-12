"""Gemini embedding model construction."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings


def create_embeddings(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    """Create Gemini embeddings using private application settings."""
    settings.validate()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
