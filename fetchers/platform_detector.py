import re
from typing import Type
from .base import BaseFetcher
from .youtube_fetcher import YouTubeFetcher
from .reddit_fetcher import RedditFetcher

def detect_platform(url: str) -> str:
    youtube_pattern = r"(?:https?:\/\/)?(?:(?:www|m)\.)?(?:youtube\.com|youtu\.be)"
    reddit_pattern = r"(?:https?:\/\/)?(?:(?:www|old|new|np)\.)?reddit\.com"
    
    if re.search(youtube_pattern, url, re.IGNORECASE):
        return 'youtube'
    elif re.search(reddit_pattern, url, re.IGNORECASE):
        return 'reddit'
    return 'unknown'

def get_fetcher(url: str) -> BaseFetcher:
    platform = detect_platform(url)
    if platform == 'youtube':
        return YouTubeFetcher()
    elif platform == 'reddit':
        return RedditFetcher()
    else:
        raise ValueError(f"No fetcher available for the given URL. Platform detected: {platform}")
