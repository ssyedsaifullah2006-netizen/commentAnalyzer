"""
QA Chain using Gemini for answering questions about comments via RAG.
Uses bounded chat history to prevent prompt bloat.
"""
import logging
import json
import time
from typing import List, Tuple, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
import config

logger = logging.getLogger(__name__)

# Reusable Gemini client
_client = None
def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


class QAChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

    def _call_with_retry(self, prompt: str, config_kwargs: dict) -> str:
        """Call Gemini with retry on transient errors."""
        gen_config = types.GenerateContentConfig(
            safety_settings=self.safety_settings,
            **config_kwargs
        )

        for attempt in range(3):
            try:
                response = _get_client().models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=gen_config
                )
                if not response.candidates:
                    raise ValueError("Blocked or empty response")
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    raise ValueError("No content in response")
                return candidate.content.parts[0].text

            except APIError as e:
                if e.code in (429, 503) and attempt < 2:
                    delay = 3 * (2 ** attempt)  # 3s, 6s
                    logger.warning(f"QA retry {attempt+1}/3 in {delay}s (code {e.code})")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                if attempt < 2 and any(k in str(e).lower() for k in ['timeout', 'ssl', 'connection']):
                    time.sleep(2)
                else:
                    raise

    def ask(self, question: str, n_results: int = None, history: Optional[List[Tuple[str, str]]] = None) -> dict:
        """Answer a question using RAG. Uses config.RAG_TOP_K for retrieval."""
        if n_results is None:
            n_results = config.RAG_TOP_K

        results = self.vector_store.search(question, n_results=n_results)

        source_comments = []
        relevance_scores = []
        source_metadatas = []

        if results and 'documents' in results and results['documents']:
            docs = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
            source_comments = docs

        if results and 'metadatas' in results and results['metadatas']:
            metas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
            source_metadatas = metas

        if results and 'distances' in results and results['distances']:
            dists = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
            relevance_scores = [max(0.0, 1.0 - d / 2.0) for d in dists]

        # Bound history to last 5 exchanges to prevent prompt bloat
        history_str = ""
        if history:
            recent = history[-5:]
            history_str = "Recent Conversation:\n"
            for q, a in recent:
                history_str += f"User: {q}\nAI: {a}\n\n"

        context = "\n".join(source_comments)
        prompt = f"""You are an expert community manager and data analyst.
Answer the user's question using the relevant comments below.

- Be thorough and use specific examples from the comments.
- Use Markdown formatting for structure.
- If the context doesn't fully answer the question, say so.

{history_str}

### Context (Social Media Comments):
{context}

### User's Question:
{question}
"""

        try:
            answer = self._call_with_retry(prompt, {})
        except Exception as e:
            logger.error(f"QA error: {e}")
            answer = "Sorry, I couldn't generate an answer right now. Please try again."

        return {
            "answer": answer,
            "source_comments": source_comments,
            "relevance_scores": relevance_scores,
            "source_metadatas": source_metadatas
        }

    def get_suggested_questions(self, comments_sample: List[str]) -> List[str]:
        if not comments_sample:
            return []

        sample = "\n".join(comments_sample[:10])
        prompt = f"Based on these comments, suggest 3 insightful questions a user might ask. Return as a JSON array of strings.\n\n{sample}"

        try:
            res_text = self._call_with_retry(prompt, {
                "response_mime_type": "application/json",
                "response_schema": {"type": "ARRAY", "items": {"type": "STRING"}}
            })
            return json.loads(res_text)
        except Exception as e:
            logger.error(f"Suggested questions error: {e}")
            return []
