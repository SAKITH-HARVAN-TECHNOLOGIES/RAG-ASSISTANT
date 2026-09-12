"""Prompt and answer-chain construction for document-grounded questions."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings

RAG_SYSTEM_MESSAGE = """You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say exactly:
I couldn't find that information in the uploaded document.
Do not use outside knowledge or make up facts.
Keep the answer clear and concise.

Context:
{context}"""


def create_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model without exposing credentials to the UI."""
    settings.validate()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.temperature,
        google_api_key=settings.google_api_key,
    )


def create_answer_chain(model: ChatGoogleGenerativeAI) -> Runnable:
    """Build an LCEL chain that answers from supplied retrieved context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_MESSAGE),
            ("human", "Question:\n{question}"),
        ]
    )
    return prompt | model | StrOutputParser()


def format_context(documents: list[object]) -> str:
    """Combine retrieved document text for the grounded prompt."""
    return "\n\n---\n\n".join(
        getattr(document, "page_content", "") for document in documents
    )
