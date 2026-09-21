"""
Centralized configuration for the Comment Intelligence Platform.
Loads API keys from .env and defines model names, thresholds, and paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Keys ────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CommentAnalyzer/1.0")

# ─── Model Configuration ────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini model used for analysis, summary, and Q&A.
# Change this to any Gemini model you have access to:
#   "gemini-2.0-flash"       — fast, great quality, FREE: 15 RPM / 1M TPM
#   "gemini-2.0-flash-lite"  — fastest, cheapest, FREE: 30 RPM / 1M TPM
#   "gemini-1.5-flash"       — older but solid,   FREE: 15 RPM / 1M TPM
#   "gemini-1.5-pro"         — highest quality,    FREE: 2 RPM / 32K TPM (slow!)
#   "gemini-2.5-flash"       — newest flash,       FREE: 10 RPM (if available)
#
# If you're hitting daily limits, switch to "gemini-2.0-flash" or "gemini-2.0-flash-lite"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Gemini embedding model used for RAG vector search.
# Available on your API key:
#   "gemini-embedding-001"        — stable, recommended
#   "gemini-embedding-2"          — newer
#   "gemini-embedding-2-preview"  — preview of v2
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"

# ─── Processing Settings ────────────────────────────────────────
MAX_COMMENTS_DEFAULT = 500
MAX_ANALYSIS_COMMENTS = 150
RAG_TOP_K = 10
SUMMARIZATION_CHUNK_SIZE = 200
BATCH_SIZE = 16
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MIN_TOPIC_SIZE = 5
NR_TOPICS = "auto"

# ─── Server Settings ────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
