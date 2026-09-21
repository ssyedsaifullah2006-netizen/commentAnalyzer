from .base import Comment, BaseFetcher
from .youtube_fetcher import YouTubeFetcher
from .reddit_fetcher import RedditFetcher
from .platform_detector import detect_platform, get_fetcher

__all__ = [
    'Comment',
    'BaseFetcher',
    'YouTubeFetcher',
    'RedditFetcher',
    'detect_platform',
    'get_fetcher'
]
