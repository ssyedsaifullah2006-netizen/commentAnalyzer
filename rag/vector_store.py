"""
VectorStore module using ChromaDB for persistent storage and retrieval of comments.
Uses Gemini text-embedding-004 for embeddings via a reusable client.
"""
import logging
from typing import List, Dict, Any, Optional

import config

logger = logging.getLogger(__name__)

# ─── Reusable Gemini Embedding Client (created once) ────────────
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


from chromadb.api.types import EmbeddingFunction

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function using Gemini with batching and retry."""
    
    def name(self) -> str:
        return "GeminiEmbeddingFunction"

    def __call__(self, input: list[str]) -> list[list[float]]:
        import time
        client = _get_gemini_client()
        all_embeddings = []

        # Batch in chunks of 100 (API maximum)
        batch_size = 100
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            embs = self._embed_with_retry(client, batch)
            all_embeddings.extend(embs)
            if i + batch_size < len(input):
                time.sleep(1.0)

        return all_embeddings

    def _embed_with_retry(self, client, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        import time
        for attempt in range(max_retries):
            try:
                res = client.models.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    contents=texts
                )
                return [e.values for e in res.embeddings]
            except Exception as e:
                error_msg = str(e).lower()
                is_retriable = any(k in error_msg for k in ['429', 'rate', 'quota', '503', 'timeout', 'connection'])
                if is_retriable and attempt < max_retries - 1:
                    delay = 2 * (2 ** attempt)  # 2s, 4s
                    logger.warning(f"Embedding retry {attempt+1}/{max_retries} in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Embedding failed: {e}") from e
        return []


class VectorStore:
    """Manages a ChromaDB collection for comments."""

    def __init__(self, collection_name: str = 'comments_gemini'):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.emb_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_comments(self, comments: List[Dict[str, Any]]):
        """Add comments to the vector store."""
        if not comments:
            return

        ids = []
        documents = []
        metadatas = []

        for i, c in enumerate(comments):
            text = c.get('text', '').strip()
            if not text:
                continue
                
            doc_id = str(c.get('id', f"doc_{i}"))
            ids.append(doc_id)
            documents.append(text)

            meta = {}
            if 'metadata' in c and isinstance(c['metadata'], dict):
                for key, val in c['metadata'].items():
                    if val is not None:
                        meta[key] = val
            for key in ['author', 'platform', 'likes', 'timestamp']:
                if key in c and c[key] is not None:
                    meta[key] = c[key]

            # ChromaDB requires non-empty metadata
            if not meta:
                meta = {"source": "comment"}
            metadatas.append(meta)

        if not ids:
            return

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, n_results: int = None, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search the vector store. Uses config.RAG_TOP_K by default."""
        if n_results is None:
            n_results = config.RAG_TOP_K

        query_kwargs = {"query_texts": [query], "n_results": min(n_results, self.count() or 1)}
        if where:
            query_kwargs["where"] = where
        return self.collection.query(**query_kwargs)

    def clear(self):
        """Delete the collection entirely and recreate it (efficient)."""
        name = self.collection.name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.emb_fn
        )

    def count(self) -> int:
        return self.collection.count()
