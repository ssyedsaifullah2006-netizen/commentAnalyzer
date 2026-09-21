import praw
from praw.models import MoreComments
from typing import List, Dict, Any, Optional, Callable
from .base import Comment, BaseFetcher

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

class RedditFetcher(BaseFetcher):
    def __init__(self):
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
        else:
            self.reddit = None

    def fetch_comments(self, url: str, max_comments: int, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Comment]:
        if not self.reddit:
            raise ValueError("Reddit API credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) not configured in .env")
            
        try:
            submission = self.reddit.submission(url=url)
            submission.comments.replace_more(limit=0)
            
            comments = []
            all_comments = submission.comments.list()
            
            for comment in all_comments:
                if len(comments) >= max_comments:
                    break
                    
                author = comment.author.name if comment.author else "[deleted]"
                
                comments.append(Comment(
                    text=comment.body,
                    author=author,
                    timestamp=str(comment.created_utc),
                    likes=comment.score,
                    replies_count=len(comment.replies) if hasattr(comment, 'replies') else 0,
                    platform="reddit",
                    comment_id=comment.id
                ))
                
                if progress_callback:
                    progress_callback(len(comments), max_comments)
                    
            return comments
        except Exception as e:
            print(f"Error fetching Reddit comments via API: {e}")
            raise RuntimeError(f"Reddit API fetch error: {e}")

    def get_metadata(self, url: str) -> Dict[str, Any]:
        if not self.reddit:
            raise ValueError("Reddit API credentials not configured")
            
        try:
            submission = self.reddit.submission(url=url)
            return {
                "title": submission.title,
                "subreddit": submission.subreddit.display_name,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "author": submission.author.name if submission.author else "[deleted]"
            }
        except Exception as e:
            print(f"Error fetching Reddit metadata: {e}")
            return {}
