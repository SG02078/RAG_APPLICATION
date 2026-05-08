import os
import smtplib
import tempfile
import time
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Iterable, List

import streamlit as st
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from pypdf.errors import DependencyError as PdfDependencyError
from pypdf.errors import PdfReadError
from dotenv import load_dotenv


load_dotenv()


DEFAULT_DEPARTMENTS = ["HR", "Marketing", "Finance", "Sales Enablement"]
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".pptx"}
EMBEDDING_MODEL = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1024"))
DEFAULT_RETRIEVAL_K = 12


st.set_page_config(page_title="Sales Knowledge RAG", page_icon="chat", layout="wide")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        st.error(f"Missing environment variable: {name}")
        st.stop()
    placeholder_tokens = ("your-", "sk-your-", "sk-or-your-", "enter-")
    if any(token in value.lower() for token in placeholder_tokens):
        st.error(f"Replace the placeholder value for {name} in your .env file.")
        st.stop()
    return value


@st.cache_resource(show_spinner=False)
def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=require_env("PINECONE_API_KEY"))


@st.cache_resource(show_spinner=False)
def get_chat_llm() -> ChatOpenAI:
    headers = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_APP_NAME"):
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=require_env("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        default_headers=headers or None,
        temperature=0.2,
    )


def ensure_index(pc: Pinecone, index_name: str) -> None:
    if index_name in pc.list_indexes().names():
        description = pc.describe_index(index_name)
        existing_dimension = getattr(description, "dimension", None)
        if existing_dimension and int(existing_dimension) != EMBEDDING_DIMENSION:
            st.error(
                f"Pinecone index `{index_name}` has dimension {existing_dimension}, "
                f"but `{EMBEDDING_MODEL}` is configured for {EMBEDDING_DIMENSION}. "
                "Use a new PINECONE_INDEX_NAME or recreate the index with the correct dimension."
            )
            st.stop()
        return

    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": os.getenv("PINECONE_CLOUD", "aws"),
                "region": os.getenv("PINECONE_REGION", "us-east-1"),
            }
        },
    )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(2)


def embed_texts(texts: List[str], input_type: str) -> List[List[float]]:
    pc = get_pinecone_client()
    batch_size = int(os.getenv("PINECONE_EMBED_BATCH_SIZE", "96"))
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=batch,
            parameters={
                "input_type": input_type,
                "truncate": "END",
            },
        )
        embeddings.extend(extract_embedding_values(response))

    return embeddings


def extract_embedding_values(response: Any) -> List[List[float]]:
    if hasattr(response, "data"):
        items = response.data
    elif hasattr(response, "embeddings_list"):
        items = response.embeddings_list.get("data", [])
    else:
        items = list(response)

    values = []
    for item in items:
        if isinstance(item, dict):
            values.append(item["values"])
        else:
            values.append(item.values)
    return values


def normalize_namespace(department: str) -> str:
    return department.strip().lower().replace(" ", "-")


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getvalue())
    temp.flush()
    temp.close()
    return Path(temp.name)


def load_documents(path: Path, original_name: str, department: str) -> List[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".csv":
        loader = CSVLoader(str(path), encoding="utf-8")
    elif suffix == ".pptx":
        loader = UnstructuredPowerPointLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.update(
            {
                "department": department,
                "source": original_name,
                "namespace": normalize_namespace(department),
            }
        )
    return docs


def split_documents(documents: Iterable[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def get_pinecone_index():
    index_name = require_env("PINECONE_INDEX_NAME")
    pc = get_pinecone_client()
    ensure_index(pc, index_name)
    return pc.Index(index_name)


def add_documents_to_namespace(uploaded_files, department: str) -> int:
    namespace = normalize_namespace(department)
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        temp_path = save_uploaded_file(uploaded_file)
        try:
            all_docs.extend(load_documents(temp_path, uploaded_file.name, department))
        finally:
            temp_path.unlink(missing_ok=True)

    chunks = split_documents(all_docs)
    if not chunks:
        return 0

    embeddings = embed_texts([chunk.page_content for chunk in chunks], input_type="passage")
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = dict(chunk.metadata)
        metadata["text"] = chunk.page_content
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": metadata,
            }
        )

    index = get_pinecone_index()
    batch_size = int(os.getenv("PINECONE_UPSERT_BATCH_SIZE", "100"))
    for start in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[start : start + batch_size], namespace=namespace)

    return len(chunks)


def expand_retrieval_queries(question: str) -> List[str]:
    normalized_question = " ".join(question.replace(",", " ").split())
    queries = [question]
    if normalized_question != question:
        queries.append(normalized_question)

    economic_terms = {
        "economy",
        "economic",
        "growth",
        "gdp",
        "manufacture",
        "manufacturing",
        "industry",
        "industrial",
    }
    if any(term in normalized_question.lower() for term in economic_terms):
        queries.append(
            f"{normalized_question} US GDP growth manufacturing industrial production economy outlook"
        )

    return list(dict.fromkeys(queries))


def retrieve_documents(question: str, departments: List[str]) -> List[Document]:
    namespaces = [normalize_namespace(dept) for dept in departments]
    retrieval_queries = expand_retrieval_queries(question)
    query_embeddings = embed_texts(retrieval_queries, input_type="query")
    index = get_pinecone_index()
    top_k = int(os.getenv("RETRIEVAL_K", str(DEFAULT_RETRIEVAL_K)))
    docs = []
    seen = set()

    for namespace in namespaces:
        for query_embedding in query_embeddings:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
            for match in results.get("matches", []):
                metadata = dict(match.get("metadata") or {})
                text = metadata.pop("text", "").strip()
                if not text:
                    continue
                key = (
                    metadata.get("source"),
                    metadata.get("page"),
                    text[:120],
                )
                if key not in seen:
                    seen.add(key)
                    docs.append(Document(page_content=text, metadata=metadata))

    return docs


def format_context(source_docs: List[Document]) -> str:
    context_blocks = []
    for index, doc in enumerate(source_docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        department = doc.metadata.get("department", "Unknown department")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        context_blocks.append(
            f"[Source {index}: {department} - {source}{page_label}]\n{doc.page_content}"
        )
    return "\n\n".join(context_blocks)


def format_chat_history(chat_history: List[Any]) -> str:
    if not chat_history:
        return "No previous conversation."

    lines = []
    for user_message, assistant_message in chat_history[-5:]:
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_message}")
    return "\n".join(lines)


def answer_question(question: str, departments: List[str]):
    source_docs = retrieve_documents(question, departments)
    context = format_context(source_docs)
    chat_history = format_chat_history(st.session_state.get("chat_history", []))

    system_prompt = """You are a helpful internal sales knowledge assistant.
Answer using only the provided departmental context. If the context contains relevant facts,
synthesize them directly even when the user's question is broad or informal. If the context is
truly not enough, say what is missing and suggest which department documents may contain the answer.
Be concise, practical, and sales-friendly."""
    user_prompt = f"""Conversation history:
{chat_history}

Departmental context:
{context or "No relevant context was found."}

Question:
{question}
"""

    response = get_chat_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return {"answer": response.content, "source_documents": source_docs}


def format_sources(source_docs: List[Document]) -> str:
    lines = []
    for doc in source_docs[:8]:
        source = doc.metadata.get("source", "Unknown source")
        department = doc.metadata.get("department", "Unknown department")
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        lines.append(f"- {department}: {source}{page_label}")
    return "\n".join(lines)


def send_answer_email(recipient: str, subject: str, question: str, answer: str, source_docs: List[Document]) -> None:
    sender = require_env("GMAIL_ID")
    password = require_env("GMAIL_PASSWORD")
    body = f"""Question:
{question}

Answer:
{answer}

Sources:
{format_sources(source_docs) or "No sources returned."}
"""

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)


def render_email_form() -> None:
    latest = st.session_state.get("last_answer_email")
    if not latest:
        return

    with st.expander("Email latest answer"):
        with st.form("email_answer_form", clear_on_submit=False):
            recipient = st.text_input("Recipient Gmail address")
            subject = st.text_input("Subject", value="Sales knowledge answer")
            submitted = st.form_submit_button("Send email")
            if submitted:
                if not recipient:
                    st.warning("Enter a recipient Gmail address.")
                else:
                    with st.spinner("Sending email..."):
                        send_answer_email(
                            recipient,
                            subject,
                            latest["question"],
                            latest["answer"],
                            latest["source_documents"],
                        )
                    st.success(f"Email sent to {recipient}.")


def render_sidebar():
    with st.sidebar:
        st.header("Knowledge Base")
        department_options = DEFAULT_DEPARTMENTS.copy()
        custom_department = st.text_input("Add department namespace")
        if custom_department:
            department_options.append(custom_department)

        selected_department = st.selectbox("Upload to department", department_options)
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[ext.removeprefix(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )

        if st.button("Index documents", type="primary", disabled=not uploaded_files):
            with st.spinner(f"Indexing into `{normalize_namespace(selected_department)}`..."):
                try:
                    chunk_count = add_documents_to_namespace(uploaded_files, selected_department)
                except PdfDependencyError:
                    st.error(
                        "This PDF uses AES encryption and needs the `cryptography` package. "
                        "Redeploy after installing the updated requirements, then upload it again."
                    )
                except PdfReadError as exc:
                    st.error(f"Could not read one of the uploaded PDFs: {exc}")
                else:
                    st.success(f"Indexed {chunk_count} chunks in `{normalize_namespace(selected_department)}`.")

        st.divider()
        st.caption("Set these in Railway variables: OPENROUTER_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, GMAIL_ID, GMAIL_PASSWORD.")

    return department_options


def render_chat(department_options):
    st.title("Sales Knowledge Chatbot")
    st.write("Ask questions across HR policies, marketing strategy, finance reports, and other departmental documents.")

    selected_departments = st.multiselect(
        "Search departments",
        department_options,
        default=department_options[:3],
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a sales question...")
    if not prompt:
        render_email_form()
        return

    if not selected_departments:
        st.warning("Choose at least one department to search.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching departmental knowledge..."):
            response = answer_question(prompt, selected_departments)
        answer = response["answer"]
        st.markdown(answer)

        source_docs = response.get("source_documents", [])
        if source_docs:
            with st.expander("Sources"):
                for line in format_sources(source_docs).splitlines():
                    st.markdown(line.replace("- ", "- **", 1).replace(":", "**:", 1))

        st.session_state.last_answer_email = {
            "question": prompt,
            "answer": answer,
            "source_documents": source_docs,
        }

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((prompt, answer))
    render_email_form()


def main():
    department_options = render_sidebar()
    render_chat(department_options)


if __name__ == "__main__":
    main()
