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
