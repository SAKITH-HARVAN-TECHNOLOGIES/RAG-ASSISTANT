"""Load supported uploaded files into LangChain documents."""

import logging
from pathlib import Path
import tempfile

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
logger = logging.getLogger(__name__)


def load_uploaded_document(uploaded_file: object) -> list[Document]:
    """Save an uploaded file temporarily, load it, and remove the temp file."""
    file_name = getattr(uploaded_file, "name", "")
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PDF, TXT, or DOCX file.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded document is empty.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        if suffix == ".pdf":
            documents = PyPDFLoader(temp_path).load()
        elif suffix == ".docx":
            documents = Docx2txtLoader(temp_path).load()
        else:
            documents = TextLoader(temp_path, encoding="utf-8").load()
    except Exception as exc:
        logger.exception("Document loading failed for extension %s", suffix)
        raise RuntimeError("The document could not be loaded.") from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

    if not any(document.page_content.strip() for document in documents):
        raise ValueError("The uploaded document does not contain readable text.")
    logger.info("Loaded %d document pages from %s", len(documents), suffix)
    return documents
