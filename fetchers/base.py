from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

@dataclass
class Comment:
    text: str
    author: str
    timestamp: str
    likes: int
    replies_count: int
    platform: str
    comment_id: str

class BaseFetcher(ABC):
    @abstractmethod
    def fetch_comments(self, url: str, max_comments: int, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Comment]:
        pass
        
    @abstractmethod
    def get_metadata(self, url: str) -> Dict[str, Any]:
        pass
