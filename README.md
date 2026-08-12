# Intelligent PDF Research Assistant (RAG + LLM)

This project is a simple Retrieval-Augmented Generation (RAG) application. You upload PDFs, the app splits their text into small overlapping chunks, turns each chunk into an embedding, and saves it in ChromaDB with its filename and page number. When you ask a question, it finds the most similar chunks, gives only those chunks to OpenAI or Groq, and returns the answer together with source citations. This makes the answer easier to check against the original PDFs.

## Architecture in plain English

PDF upload → `pypdf` reads text page by page → LangChain splits each page into overlapping chunks → Sentence Transformers changes chunks into embeddings → ChromaDB stores the embeddings, chunk text, filename, page number, and document ID. For a question, the app embeds the question with the same model, retrieves the closest chunks from ChromaDB, sends that evidence plus recent conversation history to the selected LLM, and returns the answer with citations.

## Run locally

1. Create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the packages.

   ```bash
   pip install -r requirements.txt
   ```

3. Choose an LLM provider and set its API key. Use one of these commands in the same terminal where you start the server.

   ```bash
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY="your_openai_key"
   ```

   ```bash
   export LLM_PROVIDER=groq
   export GROQ_API_KEY="your_groq_key"
   ```

4. Start the API server.

   ```bash
   uvicorn app:app --reload
   ```

5. In a second terminal, activate the same virtual environment and start the Streamlit interface.

   ```bash
   streamlit run streamlit_app.py
   ```

6. Open the Streamlit address shown in that terminal, normally `http://localhost:8501`. Use its **Choose PDF files** button to upload documents, then ask questions and manage documents from the same page.

FastAPI's `http://127.0.0.1:8000/docs` page is still available if you want to inspect or test the raw API endpoints.

## API endpoints

| Endpoint | What it does |
| --- | --- |
| `POST /upload` | Upload one or more PDFs and index them. Optional query settings: `chunk_size` and `chunk_overlap`. |
| `POST /ask` | Ask a question. Send `question`, optional `session_id`, and optional `top_k` in JSON. Returns an answer and citations. |
| `GET /documents` | List indexed PDFs with their IDs and chunk counts. |
| `DELETE /documents/{doc_id}` | Delete one PDF and all of its ChromaDB chunks. Use the `document_id` returned by `/documents`. |

## Example question request

After uploading a PDF, send this JSON in the `/ask` section of `/docs`:

```json
{
  "question": "What are the main conclusions of this document?",
  "session_id": "student-demo",
  "top_k": 4
}
```

`session_id` is just a label. Questions with the same label remember the last three question-and-answer turns while the server is running. Restarting the server clears this memory, but ChromaDB keeps indexed documents in `chroma_data/`.

## Configurable settings

Edit `config.py` to change the default chunk size, chunk overlap, retrieval `top_k`, history length, or default models. You can also provide `chunk_size`, `chunk_overlap`, and `top_k` per request through FastAPI's `/docs` page.

The first run downloads the embedding model. ChromaDB saves its local index in `chroma_data/`. A scanned PDF without selectable text needs OCR before this project can read it.
