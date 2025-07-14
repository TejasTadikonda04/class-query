from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
INDEX_PATH = BASE_DIR / "data" / "vectorstore" / "classquery.faiss"
META_PATH = INDEX_PATH.with_suffix(".meta.json")

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 200   # words
CHUNK_OVERLAP = 40 # words

# Embedding parameters
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True
