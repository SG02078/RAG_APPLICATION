# Sales Knowledge RAG Chatbot

A Streamlit RAG application for sales teams to query department-specific documents. Uploaded files are chunked with LangChain, embedded with Pinecone Inference, stored in Pinecone under departmental namespaces such as `hr`, `marketing`, and `finance`, and answered with an OpenRouter-hosted LLM.

## Features

- Upload documents into a selected department namespace.
- Query one or more departments from the chat UI.
- Cite source documents and PDF page numbers when available.
- Send the latest answer and sources to Gmail recipients.
- Supports `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, and `.pptx`.
- Ready for Railway deployment.

## Local Setup

Use Python 3.14 locally if you prefer. This app uses LangChain for document loading, text splitting, and OpenRouter chat calls, while Pinecone embedding, vector upsert, and vector query are handled through the official Pinecone SDK.

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Fill in `.env` with your keys.

```powershell
notepad .env
```

4. Run the app.

```powershell
streamlit run app.py
```

## Pinecone Setup

The app can create the Pinecone serverless index automatically when it has a valid `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`.

Default index settings:

- Embedding model: `llama-text-embed-v2`
- Dimension: `1024`
- Metric: `cosine`
- Cloud: `aws`
- Region: `us-east-1`

These match Pinecone Inference `llama-text-embed-v2`. If you previously created an index with OpenAI embeddings at dimension `1536`, use a new `PINECONE_INDEX_NAME` or recreate the index at dimension `1024`.

## OpenRouter Setup

The chatbot LLM uses OpenRouter through LangChain's OpenAI-compatible client.

Required variables:

```text
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENROUTER_BASE_URL
```

OpenRouter is used for answer generation. Embeddings are generated with Pinecone Inference through `PINECONE_API_KEY`, so no OpenAI API key is required.

## Gmail Setup

Set these in `.env`:

```text
GMAIL_ID
GMAIL_PASSWORD
```

For Gmail, use an app password rather than your normal account password. After the chatbot answers, open **Email latest answer**, enter the recipient Gmail address, and send the answer plus sources.

## Railway Deployment

1. Push this repository to GitHub.
2. Create a new Railway project from the GitHub repository.
3. Add these Railway variables:

```text
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENROUTER_BASE_URL
OPENROUTER_SITE_URL
OPENROUTER_APP_NAME
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_CLOUD
PINECONE_REGION
PINECONE_DIMENSION
PINECONE_EMBEDDING_MODEL
PINECONE_EMBED_BATCH_SIZE
RETRIEVAL_K
GMAIL_ID
GMAIL_PASSWORD
```

4. Deploy. Railway will use `railway.json` or the `Procfile` start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Upload 403 Troubleshooting

If document upload shows `AxiosError: Request failed with status code 403`, the
browser-side Streamlit uploader is being rejected before the Python app receives
the file. This commonly happens behind corporate proxies, SSO layers, or iframe
embeds when Streamlit's upload request fails an origin/XSRF check.

This repo sets the required Streamlit server options in `.streamlit/config.toml`
and repeats them in the Railway start command:

```toml
[server]
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 50
```

After changing these settings, redeploy or restart the Streamlit server. If the
same company laptop still gets 403 while another network works, ask IT to allow
POST requests to the app's Streamlit upload endpoint, especially
`/_stcore/upload_file`.

As a workaround when company security blocks file uploads, use the sidebar's
**Paste text** tab. Copy text from the document, give it a title, and click
**Index pasted text**. This avoids Streamlit's browser upload endpoint and still
stores the content in the selected Pinecone department namespace.

## Suggested Department Flow

1. Choose a department in the sidebar.
2. Upload department documents.
3. Click **Index documents**.
4. Select departments to search in the main chat.
5. Ask sales-facing questions such as:

```text
What discounting rules should I follow for enterprise prospects?
How should I position the new Q3 campaign to healthcare leads?
What reimbursement policy applies to client travel?
```
### application flow

Sure. In 
app.py
, the RAG pipeline is mainly these sections:

1. Loading Documents

Document loading happens in:


load_documents (line 157)

def load_documents(path: Path, original_name: str, department: str) -> List[Document]:
This checks the uploaded file extension and chooses the correct LangChain loader:

.pdf -> PyPDFLoader
.docx -> Docx2txtLoader
.txt / .md -> TextLoader
.csv -> CSVLoader
.pptx -> UnstructuredPowerPointLoader
Then it attaches metadata like:

department
source
namespace
So every loaded document knows where it came from.

2. Chunking

Chunking happens in:


split_documents (line 181)

def split_documents(documents: Iterable[Document]) -> List[Document]:
It uses:

RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=180,
)
That means the app breaks long documents into chunks of about 1200 characters, with 180 characters repeated between neighboring chunks. The overlap helps avoid losing context between chunks.

3. Embedding

Embedding happens in:


embed_texts (line 97)

def embed_texts(texts: List[str], input_type: str) -> List[List[float]]:
This sends text to Pinecone Inference:

pc.inference.embed(
    model=EMBEDDING_MODEL,
    inputs=batch,
    parameters={
        "input_type": input_type,
        "truncate": "END",
    },
)
For document chunks, the app calls it with:

input_type="passage"
For user questions, it calls it with:

input_type="query"
4. Upload + Store In Pinecone

The full indexing flow is here:


add_documents_to_namespace (line 199)

def add_documents_to_namespace(uploaded_files, department: str) -> int:
This function does the full ingestion pipeline:

load documents
-> split into chunks
-> embed chunks
-> create Pinecone vectors
-> upsert into namespace
The vectors stored in Pinecone contain:

{
    "id": str(uuid.uuid4()),
    "values": embedding,
    "metadata": metadata,
}
The actual chunk text is stored inside metadata:

metadata["text"] = chunk.page_content
5. Retrieval

Retrieval starts here:


retrieve_documents (line 259)

def retrieve_documents(question: str, departments: List[str]) -> List[Document]:
It does this:

question -> embed as query -> search Pinecone namespaces -> return matching chunks
The app searches only the departments selected in the UI:

namespaces = [normalize_namespace(dept) for dept in departments]
Then it queries Pinecone:

index.query(
    vector=query_embedding,
    top_k=top_k,
    namespace=namespace,
    include_metadata=True,
)
It pulls the stored chunk text back from metadata:

text = metadata.pop("text", "").strip()
6. Passing Retrieved Context To The LLM

This happens in:


answer_question (line 317)

def answer_question(question: str, departments: List[str]):
It calls:

source_docs = retrieve_documents(question, departments)
context = format_context(source_docs)
Then sends the retrieved chunks to the LLM inside the prompt:

Departmental context:
{context or "No relevant context was found."}

Question:
{question}
So the high-level flow is:

Upload file
  -> load_documents()
  -> split_documents()
  -> embed_texts(..., input_type="passage")
  -> Pinecone upsert

Ask question
  -> embed_texts(..., input_type="query")
  -> retrieve_documents()
  -> format_context()
  -> answer_question()
  -> LLM response

## Simple LLM Demo

Run this minimal app when you only want the `hi` -> `hello` behavior:

```powershell
streamlit run simple_llm_app.py
```
