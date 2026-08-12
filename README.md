# PDF Knowledge Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application for querying information contained in PDF documents. The project ingests PDF files, converts their content into searchable embeddings, and uses a local Chroma vector database to retrieve relevant context for user questions.

## Features

- Upload and ingest PDF documents
- Extract and process document text for semantic search
- Store embeddings locally with ChromaDB
- Ask natural-language questions through a simple Streamlit interface
- Retrieve relevant document context to support responses

## Technology Stack

- **Python**
- **Streamlit** for the web interface
- **ChromaDB** for local vector storage
- PDF processing and embedding libraries defined in `requirements.txt`

## Project Structure

```text
.
├── app.py               # Application logic
├── streamlit_app.py     # Streamlit user interface
├── ingestion.py         # PDF ingestion and indexing workflow
├── config.py            # Project configuration
├── pdf/                 # Source PDF documents
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- An environment configured with the required API credentials, if applicable

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit in your browser. Add PDFs to the `pdf/` directory and use the ingestion workflow to index them before querying.

## Local Data and Secrets

The vector database, virtual environments, Streamlit secrets, and other generated local files are excluded from version control via `.gitignore`. Do not commit API keys or other credentials.

## License

This project is available for personal and educational use. Add a license file before distributing or using it commercially.
