import sys
import os
import pandas as pd
from typing import List, Dict

# Import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import BATCH_SIZE, SENTIMENT_MODEL
except ImportError:
    BATCH_SIZE = 16
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentAnalyzer:
    """
    Analyzes sentiment of comments using a pre-trained RoBERTa model.
    """
    
    def __init__(self):
        self.model_name = SENTIMENT_MODEL
        self.pipeline = None
        self._distribution = {"positive": 0, "negative": 0, "neutral": 0}

    def _load_model(self):
        """Lazy load the sentiment model pipeline."""
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline("sentiment-analysis", model=self.model_name, tokenizer=self.model_name, max_length=512, truncation=True, top_k=None)

    def analyze(self, comments: List[str], progress_callback=None) -> pd.DataFrame:
        """
        Analyze a list of comments for sentiment.
        
        Args:
            comments (List[str]): List of comment texts.
            progress_callback (callable, optional): Callback for progress updates.
            
        Returns:
            pd.DataFrame: DataFrame containing text, sentiment_label, sentiment_score, and individual scores.
        """
        self._load_model()
        results = []
        
        # Reset distribution
        self._distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        # Batch processing
        total_batches = (len(comments) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(comments), BATCH_SIZE):
            batch = comments[i:i + BATCH_SIZE]
            
            try:
                # Get scores for all classes
                batch_results = self.pipeline(batch)
                
                for j, item_scores in enumerate(batch_results):
                    scores = {score['label']: score['score'] for score in item_scores}
                    
                    # Normalize labels
                    pos_score = scores.get('positive', 0)
                    neg_score = scores.get('negative', 0)
                    neu_score = scores.get('neutral', 0)
                    
                    # Determine dominant label
                    max_label = max(scores, key=scores.get)
                    
                    self._distribution[max_label] += 1
                    
                    results.append({
                        'text': batch[j],
                        'sentiment_label': max_label,
                        'sentiment_score': scores[max_label],
                        'positive': pos_score,
                        'negative': neg_score,
                        'neutral': neu_score
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE}: {e}")
                # Append None or defaults for failed batch
                for text in batch:
                    results.append({
                        'text': text,
                        'sentiment_label': 'unknown',
                        'sentiment_score': 0.0,
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 0.0
                    })
                    
            if progress_callback:
                progress = min(1.0, (i + BATCH_SIZE) / len(comments))
                progress_callback(progress)
                
        return pd.DataFrame(results)

    def get_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of sentiments from the last analysis.
        
        Returns:
            dict: Counts of positive, negative, and neutral sentiments.
        """
        return self._distribution
