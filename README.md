# Comment Intelligence Platform

A web-based AI platform that fetches social media comments (YouTube, Reddit), runs ML-powered analysis, and provides an interactive chat interface for exploring insights.

## Features

- **Comment Fetching** — Paste a YouTube or Reddit URL to fetch up to 1000 comments via official APIs.
- **Sentiment Analysis** — Classifies every comment as positive, negative, or neutral using a RoBERTa transformer model.
- **Topic Modeling** — Automatically discovers discussion themes using BERTopic.
- **Executive Summary** — Generates a structured summary with key themes, notable opinions, and controversies via Gemini.
- **RAG Chat** — Ask natural-language questions about the comments. Uses ChromaDB vector search + Gemini to provide grounded answers with source citations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, Plotly.js |
| LLM | Google Gemini 3.6 Flash (via `google-genai` SDK) |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace) |
| Topics | BERTopic + SentenceTransformers |
| Vector DB | ChromaDB (persistent, local) |
| Data Sources | YouTube Data API v3, Reddit (PRAW) |

## Project Structure

```
├── server.py                  # FastAPI application entry point
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── static/
│   ├── index.html             # Single-page application
│   ├── style.css              # Dark-mode UI theme
│   └── app.js                 # Frontend logic
├── fetchers/
│   ├── base.py                # Comment dataclass + base fetcher ABC
│   ├── platform_detector.py   # URL detection + fetcher factory
│   ├── youtube_fetcher.py     # YouTube Data API v3 integration
│   └── reddit_fetcher.py      # PRAW-based Reddit fetcher
├── analyzers/
│   ├── sentiment.py           # RoBERTa sentiment classifier
│   ├── topics.py              # BERTopic topic modeler
│   └── summarizer.py          # Gemini-powered summarizer
├── rag/
│   ├── vector_store.py        # ChromaDB wrapper
│   └── qa_chain.py            # Gemini RAG Q&A chain
└── visualizations/
    ├── charts.py              # Plotly chart generators
    └── topic_network.py       # Topic visualization
```

## Setup

### 1. Clone and install dependencies

```bash
cd c:/VIT/Semester5/ai/prjt
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional — only needed for Reddit
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

### 3. Run the server

```bash
python -m uvicorn server:app --reload
```

Open **http://localhost:8000** in your browser.

## Usage

1. **Ingest** — Paste a YouTube or Reddit URL, set max comments, click "Fetch Comments".
2. **Analyze** — Select analysis modules (Sentiment, Topics, Executive Summary), click "Run Analysis".
3. **Chat** — Switch to the Chat tab. The app automatically builds a search index and lets you ask questions about the comments with AI-generated answers and source citations.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Detect platform from URL |
| POST | `/api/fetch` | Fetch comments and store in session |
| POST | `/api/analyze` | Run selected analysis modules |
| POST | `/api/rag/index` | Build vector search index |
| GET | `/api/rag/suggestions` | Get suggested questions |
| POST | `/api/rag/ask` | Ask a question (RAG) |
| GET | `/api/export/csv` | Export comments as CSV |
