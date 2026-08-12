"""A simple Streamlit screen for the PDF Research Assistant FastAPI backend."""

# uuid creates a stable unique session id for one browser session.
import uuid

# requests sends upload, question, and document-management requests to FastAPI.
import requests
# streamlit creates the browser interface with simple Python commands.
import streamlit as st


# This is the address of the FastAPI server started with Uvicorn.
API_URL = "http://127.0.0.1:8000"

# Configure the browser tab title and use a wide layout for comfortable reading.
st.set_page_config(page_title="PDF Research Assistant", layout="wide")

# Create one session id the first time this browser opens the Streamlit app.
if "session_id" not in st.session_state:
    # Save the id in Streamlit memory so follow-up questions keep their FastAPI history.
    st.session_state.session_id = str(uuid.uuid4())

# Show the project name at the top of the page.
st.title("Intelligent PDF Research Assistant")
# Explain the short workflow before the user starts uploading documents.
st.write("Upload PDFs, then ask questions answered only from their retrieved content.")

# Create a sidebar for settings that affect upload and retrieval.
with st.sidebar:
    # Give the settings area a clear heading.
    st.header("Settings")
    # Let the user change chunk size without editing Python code.
    chunk_size = st.number_input("Chunk size", min_value=1, value=500, step=50)
    # Let the user choose how many tokens overlap neighbouring chunks.
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=50, step=10)
    # Let the user choose how many ChromaDB chunks are retrieved for each question.
    top_k = st.number_input("Top matching chunks", min_value=1, value=4, step=1)

# Show a heading for the first part of the workflow.
st.header("1. Upload PDFs")
# Display a normal browser file picker that accepts multiple PDF files.
uploaded_pdfs = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

# Start the indexing request only after the user clicks this button.
if st.button("Upload and index PDFs"):
    # Stop with a clear message if the button was clicked before selecting a PDF.
    if not uploaded_pdfs:
        st.warning("Choose at least one PDF first.")
    # Reject settings that would make LangChain's text splitter fail.
    elif chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        # This list holds file data in the format expected by FastAPI's upload endpoint.
        files_to_send = []

        # Convert every selected Streamlit file into a named multipart upload field.
        for uploaded_pdf in uploaded_pdfs:
            # Read the PDF bytes and include its filename and content type for FastAPI.
            file_data = ("files", (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf"))
            # Add this PDF to the group sent in one upload request.
            files_to_send.append(file_data)

        try:
            # Send all selected PDFs and the chosen chunk settings to the FastAPI server.
            response = requests.post(
                f"{API_URL}/upload",
                files=files_to_send,
                params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                timeout=120,
            )
            # Turn FastAPI's JSON response into a normal Python dictionary.
            response_data = response.json()
            # Show a server error message instead of pretending the upload worked.
            if not response.ok:
                st.error(response_data.get("detail", "The upload failed."))
            else:
                # Confirm that the upload request completed successfully.
                st.success("Upload request finished.")
                # Show each PDF's document id, page count, and chunk count.
                st.json(response_data)
        except requests.RequestException as error:
            # Explain that Uvicorn must be running when Streamlit cannot reach FastAPI.
            st.error(f"Could not reach FastAPI at {API_URL}. Start Uvicorn first. Error: {error}")

# Show a heading for the question-answering part of the workflow.
st.header("2. Ask a question")
# Let the user write a natural-language question about the uploaded PDFs.
question = st.text_area("Question", placeholder="What are the main conclusions of the document?")

# Send the question only when the user clicks the button.
if st.button("Ask PDFs"):
    # Stop with a clear message when the user has not written a question.
    if not question.strip():
        st.warning("Write a question first.")
    else:
        try:
            # Send the question, stable session id, and chosen retrieval count to FastAPI.
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "session_id": st.session_state.session_id, "top_k": top_k},
                timeout=120,
            )
            # Turn FastAPI's JSON response into a normal Python dictionary.
            response_data = response.json()
            # Show an API error instead of attempting to read a missing answer.
            if not response.ok:
                st.error(response_data.get("detail", "The question could not be answered."))
            else:
                # Give the LLM answer its own clear section.
                st.subheader("Answer")
                # Display the source-grounded answer text.
                st.write(response_data["answer"])
                # Give the source list its own clear section.
                st.subheader("Citations")

                # Show every filename and page number used as retrieved evidence.
                for citation in response_data["citations"]:
                    # Display the citation in a readable form for the user.
                    st.write(f"- {citation['source_filename']}, page {citation['page_number']}")
        except requests.RequestException as error:
            # Explain that Uvicorn must be running when Streamlit cannot reach FastAPI.
            st.error(f"Could not reach FastAPI at {API_URL}. Start Uvicorn first. Error: {error}")

# Show a heading for the indexed-document controls.
st.header("3. Manage indexed documents")

# Load the document list only when the user asks, keeping the page simple and predictable.
if st.button("Refresh documents"):
    try:
        # Ask FastAPI for one summary per indexed PDF.
        response = requests.get(f"{API_URL}/documents", timeout=30)
        # Turn FastAPI's JSON response into a normal Python dictionary.
        response_data = response.json()
        # Show an API error instead of attempting to display missing documents.
        if not response.ok:
            st.error(response_data.get("detail", "Could not load documents."))
        # Explain the empty state when no PDFs have been indexed yet.
        elif not response_data["documents"]:
            st.info("No documents are indexed yet.")
        else:
            # Show every indexed document in its own expandable area.
            for document in response_data["documents"]:
                # Use the filename as a friendly label and include the document id for deletion.
                with st.expander(f"{document['source_filename']} — {document['chunk_count']} chunks"):
                    # Show the id needed by the backend to identify this exact document.
                    st.code(document["document_id"])
                    # Give each document a separate delete button by using its id as the button key.
                    if st.button("Delete this document", key=document["document_id"]):
                        # Ask FastAPI to delete all ChromaDB chunks belonging to this document.
                        delete_response = requests.delete(
                            f"{API_URL}/documents/{document['document_id']}",
                            timeout=30,
                        )
                        # Turn the delete response into a normal Python dictionary.
                        delete_data = delete_response.json()
                        # Show the deletion result or a clear error message.
                        if delete_response.ok:
                            st.success(delete_data["message"])
                        else:
                            st.error(delete_data.get("detail", "Could not delete the document."))
    except requests.RequestException as error:
        # Explain that Uvicorn must be running when Streamlit cannot reach FastAPI.
        st.error(f"Could not reach FastAPI at {API_URL}. Start Uvicorn first. Error: {error}")
