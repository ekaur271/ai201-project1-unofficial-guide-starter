import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (grounded generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Embeddings ---
# all-MiniLM-L6-v2 runs locally via sentence-transformers: no API key, no
# rate limits. 384-dim embeddings, ~256 token effective context — a good fit
# for short review text where each chunk is a single review.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector store ---
CHROMA_COLLECTION = "unofficial_guide"
CHROMA_PATH = "./chroma_db"

# --- Retrieval ---
# Reviews are short and opinions are scattered across many of them, so we
# retrieve more chunks than a long-form corpus would need. Tune in eval.
N_RESULTS = 5

# --- Documents ---
DOCS_PATH = "./documents"
