"""Small, easy-to-change settings for the PDF research assistant."""

# os reads secret API keys and optional settings from the terminal environment.
import os

# This folder holds ChromaDB's saved vectors so they survive program restarts.
CHROMA_DIRECTORY = "chroma_data"

# This name identifies our one ChromaDB collection inside the saved database.
COLLECTION_NAME = "pdf_chunks"

# This small model is fast and creates a 384-number embedding for each text chunk.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# This is the target number of tokens in each chunk; it is easy to adjust here.
DEFAULT_CHUNK_SIZE = 500

# This repeats 50 tokens between neighbouring chunks so ideas at a boundary are not lost.
DEFAULT_CHUNK_OVERLAP = 50

# This is the default number of semantically similar chunks used to answer one question.
DEFAULT_TOP_K = 4

# This is the number of recent question-and-answer turns kept in the LLM prompt.
DEFAULT_HISTORY_TURNS = 3

# Set this to "openai" or "groq", or set LLM_PROVIDER in the terminal before starting Uvicorn.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Read the OpenAI key from the environment so no secret is written into the project files.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This is the OpenAI chat model used when LLM_PROVIDER is "openai".
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Read the Groq key from the environment so no secret is written into the project files.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq offers an API compatible with the OpenAI Python client at this address.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# This is the Groq chat model used when LLM_PROVIDER is "groq".
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
