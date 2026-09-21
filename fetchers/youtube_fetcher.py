import re
from typing import List, Dict, Any, Optional, Callable
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .base import Comment, BaseFetcher

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import YOUTUBE_API_KEY

class YouTubeFetcher(BaseFetcher):
    def __init__(self):
        self.api_key = YOUTUBE_API_KEY
        if self.api_key:
            self.youtube = build("youtube", "v3", developerKey=self.api_key)
        else:
            self.youtube = None

    def _extract_video_id(self, url: str) -> str:
        pattern = r"(?:https?:\/\/)?(?:(?:www|m)\.)?(?:youtube\.com\/(?:shorts\/|live\/|[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def fetch_comments(self, url: str, max_comments: int, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Comment]:
        if not self.youtube:
            raise ValueError("YOUTUBE_API_KEY is not configured in .env file.")
            
        video_id = self._extract_video_id(url)
        comments = []
        next_page_token = None
        
        try:
            while len(comments) < max_comments:
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
                
                for item in response.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    replies_count = item["snippet"]["totalReplyCount"]
                    
                    comments.append(Comment(
                        text=top_comment["textDisplay"],
                        author=top_comment["authorDisplayName"],
                        timestamp=top_comment["publishedAt"],
                        likes=top_comment["likeCount"],
                        replies_count=replies_count,
                        platform="youtube",
                        comment_id=item["id"]
                    ))
                    
                    if len(comments) >= max_comments:
                        break
                        
                    if "replies" in item and len(comments) < max_comments:
                        for reply_item in item["replies"]["comments"]:
                            reply = reply_item["snippet"]
                            comments.append(Comment(
                                text=reply["textDisplay"],
                                author=reply["authorDisplayName"],
                                timestamp=reply["publishedAt"],
                                likes=reply["likeCount"],
                                replies_count=0,
                                platform="youtube",
                                comment_id=reply_item["id"]
                            ))
                            if len(comments) >= max_comments:
                                break
                                
                if progress_callback:
                    progress_callback(len(comments), max_comments)
                    
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            raise RuntimeError(f"YouTube Data API error: {e}")
            
        return comments[:max_comments]

    def get_metadata(self, url: str) -> Dict[str, Any]:
        if not self.youtube:
            raise ValueError("YOUTUBE_API_KEY is not configured in .env file.")
            
        video_id = self._extract_video_id(url)
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            )
            response = request.execute()
            if response.get("items"):
                item = response["items"][0]
                snippet = item["snippet"]
                stats = item["statistics"]
                return {
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0))
                }
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            
        return {}
