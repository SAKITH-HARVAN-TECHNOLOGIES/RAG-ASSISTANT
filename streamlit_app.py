"""Streamlit frontend for the simple Gemini and FAISS RAG assistant."""

import logging

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.data_loader import load_uploaded_document
from app.embeddings import create_embeddings
from app.rag_pipeline import create_answer_chain, create_chat_model, format_context
from app.vector_store import create_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hide_file_uploader_help() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_embeddings():
    return create_embeddings(settings)


@st.cache_resource
def get_chat_model():
    return create_chat_model(settings)


def reset_document_state() -> None:
    st.session_state.pop("retriever", None)
    st.session_state.pop("answer_chain", None)
    st.session_state.pop("processed_file", None)
    st.session_state.messages = []


def process_document(uploaded_file: object) -> None:
    try:
        documents = load_uploaded_document(uploaded_file)
    except Exception:
        logger.exception("Document loading stage failed")
        raise

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
    except Exception as exc:
        logger.exception("Text splitting stage failed")
        raise RuntimeError("Text splitting failed.") from exc
    if not chunks:
        raise ValueError("The document did not produce any readable text chunks.")

    try:
        settings.validate()
        retriever = create_retriever(chunks, get_embeddings(), settings)
    except Exception as exc:
        logger.exception("Embedding or vector-store stage failed")
        raise RuntimeError("Embeddings or vector-store creation failed.") from exc

    st.session_state.retriever = retriever
    st.session_state.answer_chain = create_answer_chain(get_chat_model())
    st.session_state.processed_file = getattr(uploaded_file, "name", "document")
    st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title="Simple RAG Assistant", page_icon="📚")
    hide_file_uploader_help()
    st.title("📚 Simple RAG Assistant")
    st.write("Upload a document and ask questions based on its content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader(
            "Upload PDF, TXT, or DOCX",
            type=["pdf", "txt", "docx"],
        )
        if st.button("Process document", type="primary", disabled=uploaded_file is None):
            reset_document_state()
            try:
                with st.spinner("Processing document..."):
                    process_document(uploaded_file)
                st.success("Document processed successfully.")
            except ValueError as exc:
                logger.warning("Document processing validation failed: %s", exc)
                st.error(str(exc))
            except RuntimeError as exc:
                logger.exception("Document processing stage failed")
                st.error(str(exc))
            except Exception:
                logger.exception("RAG document processing failed")
                st.error("Unable to process the document. Check the application configuration.")

        processed_file = st.session_state.get("processed_file")
        if processed_file:
            st.caption(f"Ready: {processed_file}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your document")
    if not question:
        return

    if "retriever" not in st.session_state:
        st.warning("Upload and process a document before asking a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                documents = st.session_state.retriever.invoke(question)
        except Exception:
            logger.exception("RAG retrieval stage failed")
            st.error("Unable to search the processed document. Check the application logs for details.")
            return

        try:
            with st.spinner("Generating an answer..."):
                answer = st.session_state.answer_chain.invoke(
                    {"context": format_context(documents), "question": question}
                )
        except ValueError as exc:
            logger.warning("Question validation failed: %s", exc)
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("RAG answer generation failed")
            error_name = type(exc).__name__.lower()
            if "quota" in error_name or "rate" in error_name or "resourceexhausted" in error_name:
                message = "Gemini rate limit reached. Please try again later."
            elif (
                "authentication" in error_name
                or "permission" in error_name
                or "unauthenticated" in error_name
            ):
                message = "Gemini authentication failed. Check the configured API key."
            elif "modelnotfound" in error_name or "notfound" in error_name:
                message = "The configured Gemini model is unavailable. Update GEMINI_MODEL."
            else:
                message = "Unable to generate an answer. Check the application logs for details."
            st.error(message)
            return

        st.markdown(answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


if __name__ == "__main__":
    main()
