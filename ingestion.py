"""Stage 1: read PDF pages, split them into chunks, embed them, and save them."""

# argparse lets this file accept PDF file paths when run from the terminal.
import argparse
# Path makes file names and file locations easier to handle safely.
from pathlib import Path
# uuid creates a unique id for every uploaded document.
import uuid

# chromadb provides the persistent local vector database.
import chromadb
# PdfReader extracts text from each page of a PDF.
from pypdf import PdfReader
# SentenceTransformer creates vector embeddings using PyTorch under the hood.
from sentence_transformers import SentenceTransformer
# TokenTextSplitter splits text by tokens instead of roughly counting characters.
from langchain_text_splitters import TokenTextSplitter

# These settings live in one simple file instead of being hidden in the code below.
from config import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)


# This function opens (or creates) the local ChromaDB collection used by the project.
def get_collection():
    # PersistentClient saves ChromaDB files in our chosen local folder.
    client = chromadb.PersistentClient(path=CHROMA_DIRECTORY)
    # get_or_create_collection means the first run creates it and later runs reuse it.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    # Return the collection so other functions can add or search chunks later.
    return collection


# This function loads the embedding model once whenever the script is started.
def get_embedding_model():
    # The model downloads automatically on first use and is cached for later use.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # Return the ready-to-use model to the caller.
    return model


# This function reads each PDF page separately to preserve correct citation page numbers.
def read_pdf_pages(pdf_path):
    # PdfReader opens the PDF and gives us access to its pages.
    reader = PdfReader(str(pdf_path))
    # This list will hold one dictionary for every page that contains readable text.
    pages = []

    # enumerate starts page numbers at 1 because that is how people cite PDF pages.
    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text returns the page's text, or None for pages without readable text.
        page_text = page.extract_text()
        # Skip blank or image-only pages because there is no text to embed.
        if not page_text or not page_text.strip():
            continue
        # Save the page number and text together so every later chunk keeps its source.
        pages.append({"page_number": page_number, "text": page_text})

    # Return the readable PDF pages to the ingestion function.
    return pages


# This function uses LangChain to create overlapping, token-sized chunks from one page.
def split_page_into_chunks(page_text, chunk_size, chunk_overlap):
    # Chunking gives retrieval small focused passages; overlap protects meaning split at an edge.
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # split_text returns a normal Python list where every item is one text chunk.
    chunks = splitter.split_text(page_text)
    # Return the list so the caller can attach metadata and embed every chunk.
    return chunks


# This function performs the full Stage 1 ingestion process for one PDF file.
def ingest_pdf(pdf_file_path, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    # Path gives us a clean way to validate the supplied PDF location.
    pdf_path = Path(pdf_file_path)
    # Stop early with a clear message if the path does not point to a real file.
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file was not found: {pdf_path}")
    # Stop early if a user accidentally supplies a file that is not a PDF.
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a file ending in .pdf")
    # Reject invalid settings now so LangChain does not fail later with a confusing message.
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller than chunk_size")

    # Create one id shared by every chunk from this uploaded document.
    document_id = str(uuid.uuid4())
    # Read page text before doing model work, so a bad PDF fails quickly.
    pages = read_pdf_pages(pdf_path)
    # Stop with a useful message when a PDF has no selectable text to index.
    if not pages:
        raise ValueError("This PDF has no readable text. Scanned PDFs need OCR before indexing.")

    # This list will contain the chunk text that ChromaDB stores and searches.
    chunk_texts = []
    # This list will contain filename, page number, and document id for each chunk.
    chunk_metadatas = []
    # This list will contain a unique ChromaDB id for each chunk.
    chunk_ids = []
    # Start at zero so every chunk in this document gets a simple sequential number.
    chunk_number = 0

    # Process every readable page while preserving its original page number.
    for page in pages:
        # Split only this page so a chunk never receives the wrong page citation.
        page_chunks = split_page_into_chunks(page["text"], chunk_size, chunk_overlap)
        # Add each chunk and its matching metadata to parallel lists.
        for chunk_text in page_chunks:
            # Ignore any unusual empty chunk because it cannot help answer a question.
            if not chunk_text.strip():
                continue
            # Save the actual text which ChromaDB will return during retrieval.
            chunk_texts.append(chunk_text)
            # Save simple citation metadata beside this exact text chunk.
            chunk_metadatas.append({
                "document_id": document_id,
                "source_filename": pdf_path.name,
                "page_number": page["page_number"],
            })
            # Make an id that is unique within the ChromaDB collection.
            chunk_ids.append(f"{document_id}_chunk_{chunk_number}")
            # Move to the next chunk number for the following chunk.
            chunk_number += 1

    # Stop before embedding if extraction somehow produced no useful chunks.
    if not chunk_texts:
        raise ValueError("No text chunks were created from this PDF")

    # Load the sentence-transformer model used for both documents and future questions.
    model = get_embedding_model()
    # encode converts every text chunk into a list of semantic numbers (an embedding).
    embeddings = model.encode(chunk_texts, convert_to_numpy=True).tolist()
    # Open the persistent collection where chunks and their embeddings will be saved.
    collection = get_collection()
    # Add matching ids, text, embeddings, and citation metadata in one ChromaDB operation.
    collection.add(ids=chunk_ids, documents=chunk_texts, embeddings=embeddings, metadatas=chunk_metadatas)

    # Return a small summary that later FastAPI endpoints can return as JSON.
    return {
        "document_id": document_id,
        "source_filename": pdf_path.name,
        "page_count": len(pages),
        "chunk_count": len(chunk_texts),
    }


# This function makes Stage 1 usable directly from the terminal before FastAPI is added.
def main():
    # Create the command-line parser and show a short description in --help output.
    parser = argparse.ArgumentParser(description="Index one PDF in the local ChromaDB database")
    # Require the path to one PDF file as the first terminal argument.
    parser.add_argument("pdf_path", help="Path to the PDF file to index")
    # Allow learners to change the chunk size without editing source code.
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Tokens in each chunk")
    # Allow learners to change the chunk overlap without editing source code.
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Repeated tokens between chunks")
    # Read the command-line arguments supplied by the user.
    arguments = parser.parse_args()
    # Run the ingestion process with the requested PDF and settings.
    result = ingest_pdf(arguments.pdf_path, arguments.chunk_size, arguments.chunk_overlap)
    # Print the result so the user can confirm what was indexed.
    print(result)


# This condition runs main only when this file is executed directly, not when imported later.
if __name__ == "__main__":
    # Start the simple command-line program.
    main()
