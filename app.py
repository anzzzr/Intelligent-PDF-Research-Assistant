"""FastAPI endpoints for uploading PDFs, asking questions, and managing documents."""

# shutil copies uploaded file data into a temporary file on this computer.
import shutil
# tempfile creates a temporary folder that is removed after each upload request.
import tempfile
# Path makes it easy to create a safe file path for every uploaded PDF.
from pathlib import Path

# FastAPI creates the web API, validates inputs, and returns helpful HTTP errors.
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
# BaseModel describes the JSON body accepted by the /ask endpoint.
from pydantic import BaseModel
# OpenAI's Python client can call OpenAI directly or Groq through its compatible API.
from openai import OpenAI

# These settings are kept in one beginner-friendly file instead of hidden in this API code.
from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_TOP_K,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL_NAME,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
)
# These existing functions provide ChromaDB access, embeddings, and PDF ingestion.
from ingestion import get_collection, get_embedding_model, ingest_pdf


# Create the FastAPI application that Uvicorn runs locally.
app = FastAPI(title="Intelligent PDF Research Assistant")

# Keep question-and-answer turns in memory while the server is running.
conversation_history = {}


# This class describes the JSON a user sends to POST /ask.
class AskRequest(BaseModel):
    # The question is searched against the indexed PDF chunks.
    question: str
    # This id keeps each user's conversation history separate in memory.
    session_id: str = "default"
    # This value controls how many relevant chunks ChromaDB returns for this question.
    top_k: int = DEFAULT_TOP_K


# This function returns the last few turns for one session, or an empty list for a new session.
def get_recent_history(session_id):
    # Get the session's saved turns without creating a session when it does not exist yet.
    session_turns = conversation_history.get(session_id, [])
    # Keep only recent turns so an old conversation does not make the LLM prompt too long.
    recent_turns = session_turns[-DEFAULT_HISTORY_TURNS:]
    # Return the simple Python list of recent question-and-answer dictionaries.
    return recent_turns


# This function changes conversation turns into readable text for the LLM prompt.
def format_history(recent_turns):
    # This list will hold one small text section for each previous conversation turn.
    history_sections = []

    # Add the earlier question and answer from every saved turn.
    for turn in recent_turns:
        # Keep the labels explicit so the LLM can tell a question from an answer.
        history_sections.append(f"Previous question: {turn['question']}\nPrevious answer: {turn['answer']}")

    # Join the sections with blank lines to make each old turn easy for the LLM to read.
    return "\n\n".join(history_sections)


# This function embeds a question and asks ChromaDB for its closest indexed chunks.
def retrieve_relevant_chunks(question, top_k):
    # Open the persistent collection that contains PDF chunks and their embeddings.
    collection = get_collection()
    # Count chunks first because ChromaDB cannot return more results than it contains.
    indexed_chunk_count = collection.count()

    # Stop with a helpful API error when no PDF has been uploaded yet.
    if indexed_chunk_count == 0:
        raise HTTPException(status_code=400, detail="Upload at least one PDF before asking a question.")

    # Reject an invalid retrieval count before sending a query to ChromaDB.
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than zero.")

    # Use the same embedding model as the PDF chunks so their vectors are comparable.
    model = get_embedding_model()
    # Put the one question in a list because encode expects a group of texts.
    question_embedding = model.encode([question], convert_to_numpy=True)[0].tolist()
    # Use the smaller number when the user requests more chunks than the database contains.
    number_of_results = min(top_k, indexed_chunk_count)
    # Ask ChromaDB to return texts, citation metadata, and distances for the closest chunks.
    search_result = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"],
    )

    # This list will hold easy-to-read dictionaries instead of ChromaDB's nested result lists.
    relevant_chunks = []
    # ChromaDB puts the matches for our one question in position zero of each result list.
    documents = search_result["documents"][0]
    # Read the matching filename, page number, and document id for every returned chunk.
    metadatas = search_result["metadatas"][0]
    # Read the distance scores, where lower normally means a closer semantic match.
    distances = search_result["distances"][0]

    # Combine every text chunk with the metadata and distance at the same list position.
    for index in range(len(documents)):
        # Save only the fields our prompt and API response need.
        relevant_chunks.append({
            "text": documents[index],
            "document_id": metadatas[index]["document_id"],
            "source_filename": metadatas[index]["source_filename"],
            "page_number": metadatas[index]["page_number"],
            "distance": distances[index],
        })

    # Return the flattened chunks to the question-answering endpoint.
    return relevant_chunks


# This function builds a strict source-grounded prompt from chunks, history, and the new question.
def build_prompt(question, relevant_chunks, recent_turns):
    # This list will contain one labelled evidence section for each retrieved PDF chunk.
    context_sections = []

    # Add every chunk with its filename and page number so the source remains visible to the LLM.
    for chunk in relevant_chunks:
        # Keep each evidence section clear and easy to connect to a later citation.
        context_sections.append(
            f"Source: {chunk['source_filename']}, page {chunk['page_number']}\n"
            f"Text: {chunk['text']}"
        )

    # Turn all retrieved chunks into one context block for the LLM.
    context_text = "\n\n".join(context_sections)
    # Turn the last few conversation turns into text for follow-up questions.
    history_text = format_history(recent_turns)

    # Give the LLM direct rules that prevent it from inventing information outside the PDFs.
    prompt = (
        "You are a PDF research assistant. Answer only from the PDF context below. "
        "If the context does not contain the answer, say that you do not know based on the uploaded documents. "
        "Do not use outside knowledge.\n\n"
        f"Conversation history:\n{history_text or 'No earlier conversation.'}\n\n"
        f"PDF context:\n{context_text}\n\n"
        f"Current question: {question}"
    )

    # Return the completed prompt to the LLM-calling function.
    return prompt


# This function calls either OpenAI or Groq according to the setting in config.py.
def ask_llm(prompt):
    # Convert the configured provider to lowercase so OPENAI and openai both work.
    provider = LLM_PROVIDER.lower()

    # Create the normal OpenAI client when the project is configured for OpenAI.
    if provider == "openai":
        # Stop with a clear message instead of making a request with a missing secret key.
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="Set the OPENAI_API_KEY environment variable.")
        # Give the client the user's OpenAI API key.
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Choose the configured OpenAI model name.
        model_name = OPENAI_MODEL_NAME
    # Create an OpenAI-compatible client pointed at Groq when Groq is selected.
    elif provider == "groq":
        # Stop with a clear message instead of making a request with a missing secret key.
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="Set the GROQ_API_KEY environment variable.")
        # Give the client Groq's key and compatible API address.
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        # Choose the configured Groq model name.
        model_name = GROQ_MODEL_NAME
    # Reject any spelling mistake in the provider setting before calling an external service.
    else:
        raise HTTPException(status_code=500, detail="LLM_PROVIDER must be 'openai' or 'groq'.")

    try:
        # Send the complete source-grounded prompt as a single user chat message.
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception as error:
        # Hide provider-specific details but explain that the external LLM request failed.
        raise HTTPException(status_code=502, detail=f"The {provider} request failed: {error}") from error

    # Read the text from the first answer choice returned by the chat model.
    answer = completion.choices[0].message.content

    # Return a fallback sentence if a provider unexpectedly returns an empty answer.
    return answer or "The language model returned an empty answer."


# This function creates a short unique source list for the answer response.
def create_citations(relevant_chunks):
    # This list will hold citation dictionaries in the same order as retrieval results.
    citations = []
    # This set remembers sources already added so repeated chunks do not repeat a citation.
    seen_sources = set()

    # Check every retrieved chunk for a source that has not been included yet.
    for chunk in relevant_chunks:
        # Build a small unique key from the document id and page number.
        source_key = (chunk["document_id"], chunk["page_number"])
        # Skip this chunk when its PDF page has already been cited.
        if source_key in seen_sources:
            continue
        # Remember this source before adding it to the response list.
        seen_sources.add(source_key)
        # Return the document id as well as its human-readable filename and page number.
        citations.append({
            "document_id": chunk["document_id"],
            "source_filename": chunk["source_filename"],
            "page_number": chunk["page_number"],
        })

    # Return the simple citation list used by POST /ask.
    return citations


# This endpoint accepts one or more PDF files and adds their chunks to ChromaDB.
@app.post("/upload")
def upload_pdfs(
    files: list[UploadFile] = File(...),
    chunk_size: int = Query(DEFAULT_CHUNK_SIZE, ge=1),
    chunk_overlap: int = Query(DEFAULT_CHUNK_OVERLAP, ge=0),
):
    # Reject an overlap that is too large before trying to create the text splitter.
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size.")

    # This list will hold one success or error result for every uploaded file.
    upload_results = []

    # A temporary folder prevents uploaded source PDF files from being kept permanently.
    with tempfile.TemporaryDirectory() as temporary_folder:
        # Process every file separately so one bad file does not stop the rest of the upload.
        for uploaded_file in files:
            # Use an empty name when FastAPI does not receive a filename from the client.
            filename = uploaded_file.filename or ""
            # Reject non-PDF uploads before saving them to the temporary folder.
            if not filename.lower().endswith(".pdf"):
                upload_results.append({"filename": filename or "unknown file", "status": "error", "message": "Only PDF files can be uploaded."})
                # Close this rejected upload because it will not reach the normal finally block below.
                uploaded_file.file.close()
                continue

            # Path(name).name removes any folder parts supplied in the upload filename.
            safe_filename = Path(filename).name
            # Join the temporary folder with the safe filename to choose where to save the PDF.
            temporary_pdf_path = Path(temporary_folder) / safe_filename

            try:
                # Open a new local file in binary write mode for the uploaded PDF bytes.
                with temporary_pdf_path.open("wb") as temporary_pdf_file:
                    # Copy the uploaded PDF contents into the temporary local file.
                    shutil.copyfileobj(uploaded_file.file, temporary_pdf_file)
                # Reuse Stage 1 with the chosen settings so uploads follow the same indexing process.
                indexing_result = ingest_pdf(temporary_pdf_path, chunk_size, chunk_overlap)
                # Return the document id and counts produced by successful indexing.
                upload_results.append({"filename": safe_filename, "status": "indexed", "document": indexing_result})
            except Exception as error:
                # Return a clear per-file error message while continuing with other PDF uploads.
                upload_results.append({"filename": safe_filename, "status": "error", "message": str(error)})
            finally:
                # Close FastAPI's uploaded file handle after this file has been processed.
                uploaded_file.file.close()

    # Return every result together because the request may have contained multiple PDFs.
    return {"results": upload_results}


# This endpoint retrieves PDF evidence, asks the selected LLM, and returns traceable citations.
@app.post("/ask")
def ask_question(request: AskRequest):
    # Remove surrounding whitespace so an empty-looking question is rejected clearly.
    question = request.question.strip()
    # Stop before retrieval when the user did not write a meaningful question.
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty.")
    # Retrieve the semantic matches that will be the only evidence available to the LLM.
    relevant_chunks = retrieve_relevant_chunks(question, request.top_k)
    # Read recent turns from this session so follow-up questions retain context.
    recent_turns = get_recent_history(request.session_id)
    # Build a prompt containing only the retrieved evidence, history, and new question.
    prompt = build_prompt(question, relevant_chunks, recent_turns)
    # Call the configured OpenAI or Groq model with the strict source-grounded prompt.
    answer = ask_llm(prompt)
    # Create a simple source list that a user can check in the original PDFs.
    citations = create_citations(relevant_chunks)

    # Create this session's list on its first question without needing a separate database.
    if request.session_id not in conversation_history:
        conversation_history[request.session_id] = []
    # Save this turn so the next question in the same session can refer back to it.
    conversation_history[request.session_id].append({"question": question, "answer": answer})

    # Return the answer, citations, and the actual configured retrieval count to the API client.
    return {"answer": answer, "citations": citations, "retrieved_chunk_count": len(relevant_chunks)}


# This endpoint lists each indexed PDF once, along with its id and number of stored chunks.
@app.get("/documents")
def list_documents():
    # Open the persistent ChromaDB collection containing all PDF chunk metadata.
    collection = get_collection()
    # Read metadata because it holds the document id and original filename for every chunk.
    stored_data = collection.get(include=["metadatas"])
    # This dictionary groups many chunks back into one document summary.
    documents_by_id = {}

    # Examine the metadata attached to every stored chunk.
    for metadata in stored_data["metadatas"]:
        # Read the common id shared by all chunks from one uploaded PDF.
        document_id = metadata["document_id"]
        # Create a document summary when this is the first chunk with that id.
        if document_id not in documents_by_id:
            documents_by_id[document_id] = {
                "document_id": document_id,
                "source_filename": metadata["source_filename"],
                "chunk_count": 0,
            }
        # Count this chunk as part of its document summary.
        documents_by_id[document_id]["chunk_count"] += 1

    # This list converts the simple grouping dictionary into JSON-friendly document summaries.
    documents = []
    # Add every document summary to the response list.
    for document in documents_by_id.values():
        # Keep the append explicit so it is easy to explain in an interview.
        documents.append(document)

    # Return an empty list naturally when no document has been indexed yet.
    return {"documents": documents}


# This endpoint removes every ChromaDB chunk belonging to one uploaded document.
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    # Open the persistent collection that holds the document's chunks.
    collection = get_collection()
    # Find chunk ids first so the API can report a missing document instead of silently succeeding.
    matching_chunks = collection.get(where={"document_id": document_id})

    # Return a standard 404 response when the requested id does not exist.
    if not matching_chunks["ids"]:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete every chunk with this document id, which also removes its embeddings and metadata.
    collection.delete(where={"document_id": document_id})

    # Confirm exactly which document was removed and how many chunks it contained.
    return {"message": "Document deleted.", "document_id": document_id, "deleted_chunk_count": len(matching_chunks["ids"])}
