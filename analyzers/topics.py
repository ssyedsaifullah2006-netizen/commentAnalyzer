import sys
import os
import pandas as pd
from typing import List, Dict, Any

# Import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import MIN_TOPIC_SIZE, NR_TOPICS, EMBEDDING_MODEL
except ImportError:
    MIN_TOPIC_SIZE = 5
    NR_TOPICS = "auto"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class TopicModeler:
    """
    Discovers topics in comments using BERTopic.
    """
    
    def __init__(self):
        self.model = None
        self.topic_info = None

    def _load_model(self):
        """Lazy load the BERTopic model."""
        if self.model is None:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Use sentence transformers for embeddings
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=MIN_TOPIC_SIZE,
                nr_topics=NR_TOPICS
            )

    def fit(self, comments: List[str], progress_callback=None) -> Dict[str, Any]:
        """
        Run topic modeling on a list of comments.
        """
        # BERTopic's underlying UMAP requires at least 15 documents by default. 
        # For stability, we enforce a minimum of 20 documents.
        if len(comments) < max(20, MIN_TOPIC_SIZE):
            return {"topics": [{"id": -1, "name": "Not enough data for topic clustering", "size": len(comments), "representative_docs": comments}]}
            
        self._load_model()
        
        if progress_callback:
            progress_callback(0.1)
            
        try:
            topics, probs = self.model.fit_transform(comments)
            
            if progress_callback:
                progress_callback(0.8)
                
            self.topic_info = self.model.get_topic_info()
            
            result_topics = []
            for _, row in self.topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id == -1:
                    continue  # Skip outliers
                    
                rep_docs = self.model.get_representative_docs(topic_id)
                result_topics.append({
                    "id": topic_id,
                    "name": row['Name'],
                    "size": row['Count'],
                    "representative_docs": rep_docs if rep_docs else []
                })
                
            if progress_callback:
                progress_callback(1.0)
                
            return {"topics": result_topics}
            
        except Exception as e:
            print(f"Error during topic modeling: {e}")
            return {"topics": [{"id": -1, "name": "Error during modeling", "size": len(comments), "representative_docs": []}]}

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get the raw BERTopic topic info dataframe.
        
        Returns:
            pd.DataFrame: Topic info
        """
        if self.topic_info is not None:
            return self.topic_info
        return pd.DataFrame()
