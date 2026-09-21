import logging
import json
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from fetchers.base import Comment
from fetchers.platform_detector import detect_platform, get_fetcher

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Comment Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session Management ─────────────────────────────────────────
sessions: Dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if session_id not in sessions:
        sessions[session_id] = {
            'comments': [],
            'metadata': {},
            'platform': '',
            'analysis_results': {},
            'vector_store': None,
            'qa_chain': None,
            'chat_history': [],
            'rag_status': 'idle',  # idle | building | ready | error
            'rag_error': None,
        }
    return sessions[session_id]

# ─── Pydantic Models ────────────────────────────────────────────
class FetchRequest(BaseModel):
    url: str
    max_comments: int = 500
    session_id: str

class AskRequest(BaseModel):
    session_id: str
    question: str

class SessionRequest(BaseModel):
    session_id: str

# ─── Static Files ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ─── Gemini-powered analysis ────────────────────────────────────
def _run_analysis_gemini(comment_texts: List[str], metadata: dict) -> dict:
    """
    Analyze comments using Gemini. Supports batched analysis for large sets.
    Sends up to MAX_ANALYSIS_COMMENTS in a single prompt (Gemini context
    window handles it well). For very large sets this keeps analysis
    thorough while staying within token limits.
    """
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    import config
    import time

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    safety = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

    # Use up to MAX_ANALYSIS_COMMENTS (default 500) instead of the old 150
    max_to_analyze = min(len(comment_texts), config.MAX_ANALYSIS_COMMENTS)
    comments_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(comment_texts[:max_to_analyze]))

    total_comments_note = ""
    if len(comment_texts) > max_to_analyze:
        total_comments_note = f"\n\nNote: {len(comment_texts)} total comments were collected. The {max_to_analyze} most relevant are shown above."

    meta_info = ""
    if metadata:
        title = metadata.get('title', '')
        if title:
            meta_info = f"\nSource: {title}"

    prompt = f"""Analyze the following {max_to_analyze} social media comments.{meta_info}{total_comments_note}

COMMENTS:
{comments_block}

Return a JSON object with this EXACT structure:
{{
  "sentiment": {{
    "positive": <count of positive comments>,
    "negative": <count of negative comments>,
    "neutral": <count of neutral comments>
  }},
  "topics": [
    {{"name": "<topic name>", "count": <number of comments about this>, "description": "<1 sentence>"}},
    ... (top 5-10 topics)
  ],
  "summary": "<A detailed 3-5 paragraph executive summary of what people are saying, highlighting major opinions, recurring themes, and overall audience reception>",
  "key_themes": ["<theme 1>", "<theme 2>", ...],
  "notable_opinions": ["<opinion 1>", "<opinion 2>", ...],
  "controversies": ["<controversy 1>", ...]
}}

Be thorough and accurate. Count sentiments carefully across ALL {max_to_analyze} comments provided. The sentiment counts MUST add up to {max_to_analyze}."""

    schema = {
        "type": "OBJECT",
        "properties": {
            "sentiment": {
                "type": "OBJECT",
                "properties": {
                    "positive": {"type": "INTEGER"},
                    "negative": {"type": "INTEGER"},
                    "neutral": {"type": "INTEGER"}
                },
                "required": ["positive", "negative", "neutral"]
            },
            "topics": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "count": {"type": "INTEGER"},
                        "description": {"type": "STRING"}
                    },
                    "required": ["name", "count", "description"]
                }
            },
            "summary": {"type": "STRING"},
            "key_themes": {"type": "ARRAY", "items": {"type": "STRING"}},
            "notable_opinions": {"type": "ARRAY", "items": {"type": "STRING"}},
            "controversies": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["sentiment", "topics", "summary", "key_themes", "notable_opinions", "controversies"]
    }

    gen_config = types.GenerateContentConfig(
        safety_settings=safety,
        response_mime_type="application/json",
        response_schema=schema
    )

    # Retry logic — 4 attempts with capped backoff
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config
            )
            if not response.candidates:
                raise ValueError("Blocked or empty response")
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("No content in response")
            return json.loads(candidate.content.parts[0].text)
        except APIError as e:
            if e.code in (429, 503) and attempt < 3:
                delay = min(5 * (attempt + 1), 15)  # 5s, 10s, 15s
                logger.warning(f"Gemini {e.code}, retry {attempt+1}/4 in {delay}s")
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
            else:
                raise

    return {"sentiment": {"positive": 0, "negative": 0, "neutral": 0}, "topics": [], "summary": "", "key_themes": [], "notable_opinions": [], "controversies": []}


def _build_index(comments: list, session_id: str, session: dict):
    """Build ChromaDB vector index for RAG. Updates session status."""
    from rag.vector_store import VectorStore
    from rag.qa_chain import QAChain

    try:
        session['rag_status'] = 'building'

        collection_name = f"comments_{session_id.replace('-', '')}"
        vs = VectorStore(collection_name=collection_name)
        vs.clear()

        docs = [
            {"id": f"comment_{i}", "text": c.text, "metadata": {"author": c.author or "Unknown", "platform": c.platform or "", "likes": c.likes or 0, "timestamp": c.timestamp or ""}}
            for i, c in enumerate(comments)
        ]
        for i in range(0, len(docs), 50):
            vs.add_comments(docs[i:i + 50])

        qa = QAChain(vector_store=vs)
        session['vector_store'] = vs
        session['qa_chain'] = qa
        session['rag_status'] = 'ready'
        session['rag_error'] = None
    except Exception as e:
        logger.error(f"Background index error: {e}")
        session['rag_status'] = 'error'
        session['rag_error'] = str(e)


# ─── Main Endpoint: Fetch + Analyze + Index ─────────────────────

@app.post("/api/fetch")
async def api_fetch(req: FetchRequest):
    try:
        # 1. Fetch comments
        fetcher = get_fetcher(req.url)
        metadata = fetcher.get_metadata(req.url)
        comments = fetcher.fetch_comments(req.url, max_comments=req.max_comments)
        platform = detect_platform(req.url)

        session = get_session(req.session_id)
        session['comments'] = comments
        session['metadata'] = metadata
        session['platform'] = platform
        session['analysis_results'] = {}
        session['vector_store'] = None
        session['qa_chain'] = None
        session['chat_history'] = []
        session['rag_status'] = 'idle'
        session['rag_error'] = None

        sample = [vars(c) for c in comments[:5]]

        # 2. Sort by engagement (likes + text length) for best quality analysis
        sorted_comments = sorted(comments, key=lambda c: (c.likes or 0, len(c.text)), reverse=True)
        comment_texts = [c.text for c in sorted_comments]

        loop = asyncio.get_running_loop()

        # Build index in the background — pass session so it can update rag_status
        def _background_index():
            _build_index(sorted_comments, req.session_id, session)

        loop.run_in_executor(executor, _background_index)

        # Analyze with Gemini (up to MAX_ANALYSIS_COMMENTS)
        analysis_results = await loop.run_in_executor(executor, _run_analysis_gemini, comment_texts, metadata)

        session['analysis_results'] = analysis_results

        return {
            "count": len(comments),
            "metadata": metadata,
            "platform": platform,
            "sample": sample,
            "analysis": analysis_results,
            "comments_analyzed": min(len(comment_texts), 500),
            "rag_ready": session.get('rag_status') == 'ready',
            "suggested_questions": ["What is the general consensus?", "Are there any complaints?", "What do people praise the most?"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fetch pipeline: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please check the URL and try again.")


@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    if session_id not in sessions:
        return {"count": 0, "metadata": None, "platform": None, "has_analysis": False, "analysis_results": {}, "rag_ready": False, "rag_status": "idle"}
    s = sessions[session_id]
    return {
        "count": len(s.get('comments', [])),
        "metadata": s.get('metadata'),
        "platform": s.get('platform'),
        "has_analysis": bool(s.get('analysis_results')),
        "analysis_results": s.get('analysis_results', {}),
        "rag_ready": s.get('rag_status') == 'ready',
        "rag_status": s.get('rag_status', 'idle'),
    }


@app.get("/api/rag/status/{session_id}")
async def api_rag_status(session_id: str):
    """Check whether RAG indexing is complete for a session."""
    session = get_session(session_id)
    return {
        "status": session.get('rag_status', 'idle'),
        "error": session.get('rag_error'),
        "ready": session.get('rag_status') == 'ready',
    }


@app.post("/api/rag/ask")
async def api_rag_ask(req: AskRequest):
    try:
        session = get_session(req.session_id)
        qa = session.get('qa_chain')
        if not qa:
            rag_status = session.get('rag_status', 'idle')
            if rag_status == 'building':
                raise HTTPException(status_code=202, detail="Search index is still building. Please wait a moment and try again.")
            raise HTTPException(status_code=400, detail="No data loaded. Fetch comments first.")

        history = session.get('chat_history', [])
        result = qa.ask(question=req.question, history=history)

        answer = result.get('answer', '')
        source_comments = result.get('source_comments', [])
        relevance_scores = result.get('relevance_scores', [])
        source_metadatas = result.get('source_metadatas', [])

        sources = []
        for i, text in enumerate(source_comments):
            score = relevance_scores[i] if i < len(relevance_scores) else 0.0
            meta = source_metadatas[i] if i < len(source_metadatas) else {}
            sources.append({
                "text": text if isinstance(text, str) else str(text),
                "author": meta.get("author", "Unknown") if isinstance(meta, dict) else "Unknown",
                "likes": meta.get("likes", 0) if isinstance(meta, dict) else 0,
                "relevance_score": round(score, 3)
            })

        history.append((req.question, answer))
        # Bound history to last 10 exchanges
        session['chat_history'] = history[-10:]

        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during Q&A: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")


@app.post("/api/rag/clear")
async def api_rag_clear(req: SessionRequest):
    session = get_session(req.session_id)
    session['chat_history'] = []
    return {"status": "cleared"}


@app.get("/api/export/csv")
async def api_export_csv(session_id: str):
    try:
        session = get_session(session_id)
        comments = session.get('comments', [])
        if not comments:
            raise HTTPException(status_code=400, detail="No data to export")

        df = pd.DataFrame([vars(c) for c in comments])
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        return response
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import SERVER_HOST, SERVER_PORT
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
