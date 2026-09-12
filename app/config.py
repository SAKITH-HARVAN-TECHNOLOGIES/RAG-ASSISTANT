"""Private application configuration loaded from environment variables."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _read_setting(name: str, default: str = "") -> str:
    """Read a deployment secret first, then fall back to the local environment."""
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError

        secret_value = st.secrets.get(name)
        if secret_value is not None:
            return str(secret_value).strip()
    except (ImportError, StreamlitSecretNotFoundError, KeyError):
        return os.getenv(name, default).strip()
    return os.getenv(name, default).strip()


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


@dataclass(frozen=True)
class Settings:
    google_api_key: str = _read_setting("GOOGLE_API_KEY")
    chunk_size: int = _read_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _read_int("CHUNK_OVERLAP", 200)
    top_k_results: int = _read_int("TOP_K_RESULTS", 4)
    embedding_model: str = _read_setting(
        "EMBEDDING_MODEL", "models/gemini-embedding-001"
    )
    gemini_model: str = _read_setting("GEMINI_MODEL", "gemini-3.6-flash")
    temperature: float = _read_float("TEMPERATURE", 0.0)

    def validate(self) -> None:
        """Validate settings needed before making an external Gemini call."""
        if not self.google_api_key:
            raise ValueError(
                "Google Gemini API key is missing. Add GOOGLE_API_KEY to .env."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


settings = Settings()
