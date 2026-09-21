"""
VectorStore module using ChromaDB for persistent storage and retrieval of comments.
"""
import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import time
from chromadb.api.types import EmbeddingFunction

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function using Google's text-embedding-004 with retry and sub-batching."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        all_embeddings = []

        # Sub-batch to avoid token limits (20 texts per API call)
        sub_batch_size = 20
        for i in range(0, len(input), sub_batch_size):
            sub_batch = input[i:i + sub_batch_size]
            embeddings = self._embed_with_retry(client, sub_batch)
            all_embeddings.extend(embeddings)

            # Small delay between sub-batches to respect rate limits
            if i + sub_batch_size < len(input):
                time.sleep(0.5)

        return all_embeddings

    def _embed_with_retry(self, client, texts: list[str], max_retries: int = 4) -> list[list[float]]:
        """Call embed_content with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                res = client.models.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    contents=texts
                )
                return [e.values for e in res.embeddings]
            except Exception as e:
                error_msg = str(e).lower()
                is_retriable = any(k in error_msg for k in ['429', 'rate', 'quota', 'resource', '503', 'timeout', 'ssl', 'connection'])
                if is_retriable and attempt < max_retries - 1:
                    delay = 3 * (2 ** attempt)  # 3s, 6s, 12s, 24s
                    print(f"Embedding rate limit/error (attempt {attempt+1}/{max_retries}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    print(f"Embedding error (final): {e}")
                    raise RuntimeError(f"Failed to generate embeddings: {e}") from e
        return []


class VectorStore:
    """
    VectorStore manages a ChromaDB collection for comments.
    """
    
    def __init__(self, collection_name: str = 'comments_gemini'):
        """
        Initialize the ChromaDB client and collection.
        
        Args:
            collection_name (str): Name of the ChromaDB collection.
        """
        import chromadb
        from chromadb.utils import embedding_functions
        
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        
        self.emb_fn = GeminiEmbeddingFunction()
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_comments(self, comments: List[Dict[str, Any]], embeddings: Optional[np.ndarray] = None):
        """
        Add comments to the vector store.
        
        Args:
            comments (List[Dict[str, Any]]): List of comment dictionaries. 
                Must contain at least 'id' and 'text'. Optionally: author, platform, likes, timestamp, sentiment.
            embeddings (Optional[np.ndarray]): Pre-computed embeddings. If None, ChromaDB computes them.
        """
        if not comments:
            return

        ids = []
        documents = []
        metadatas = []
        
        for i, c in enumerate(comments):
            # Ensure unique ID if not provided
            doc_id = str(c.get('id', f"doc_{i}"))
            ids.append(doc_id)
            
            # The actual text to embed
            documents.append(c.get('text', ''))
            
            # Metadata filtering
            meta = {}
            if 'metadata' in c and isinstance(c['metadata'], dict):
                for key, val in c['metadata'].items():
                    if val is not None:
                        meta[key] = val
            for key in ['author', 'platform', 'likes', 'timestamp', 'sentiment']:
                if key in c and c[key] is not None:
                    meta[key] = c[key]
            
            # Store subset of keys as metadata
            metadatas.append(meta)

        add_kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas
        }
        
        if embeddings is not None:
            # ChromaDB expects a list of lists for embeddings
            add_kwargs["embeddings"] = embeddings.tolist()
            
        self.collection.upsert(**add_kwargs)

    def search(self, query: str, n_results: int = 10, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the vector store for comments matching the query.
        
        Args:
            query (str): The search query.
            n_results (int): Number of results to return.
            where (Optional[Dict]): Metadata filter.
            
        Returns:
            Dict: Dictionary containing search results.
        """
        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results
        }
        
        if where:
            query_kwargs["where"] = where
            
        results = self.collection.query(**query_kwargs)
        return results

    def clear(self):
        """
        Delete all documents in the collection.
        """
        docs = self.collection.get()
        if docs and docs.get("ids"):
            self.collection.delete(ids=docs["ids"])

    def count(self) -> int:
        """
        Return the number of stored documents.
        """
        return self.collection.count()

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return all stored documents.
        
        Returns:
            List[Dict]: List of document dictionaries with 'id', 'text', and metadata.
        """
        results = self.collection.get()
        all_docs = []
        
        if not results or not results.get("ids"):
            return all_docs
            
        for i in range(len(results["ids"])):
            doc = {
                "id": results["ids"][i],
                "text": results["documents"][i] if results.get("documents") else "",
            }
            if results.get("metadatas") and results["metadatas"][i]:
                doc.update(results["metadatas"][i])
            all_docs.append(doc)
            
        return all_docs
