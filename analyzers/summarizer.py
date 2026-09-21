import json
import logging
import time
from typing import List, Dict, Any, Callable
from google import genai
from google.genai import types
from google.genai.errors import APIError
from config import GEMINI_API_KEY, GEMINI_MODEL, SUMMARIZATION_CHUNK_SIZE

logger = logging.getLogger(__name__)

class CommentSummarizer:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

    def _call_with_retry(self, prompt: str, config: types.GenerateContentConfig) -> str:
        """Helper method to handle retries for network and transient API errors."""
        max_retries = 4
        base_delay = 15
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=config
                )
                
                if not response.candidates:
                    raise ValueError("Blocked or empty response from Gemini API")
                    
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    raise ValueError("Response candidate has no content parts")
                    
                return candidate.content.parts[0].text
                
            except APIError as e:
                # Retry on 429 Too Many Requests, 503 Service Unavailable
                if e.code in (429, 503):
                    if attempt < max_retries - 1:
                        # 15s, 30s, 60s
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"APIError {e.code}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for APIError {e.code}")
                        raise
                else:
                    # Fail fast for other errors (e.g. auth errors like 401, 403)
                    logger.error(f"Non-retriable APIError {e.code}: {e}")
                    raise
            except Exception as e:
                # Catch generic network errors like SSL handshake timeouts
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "ssl" in error_msg.lower() or "connection" in error_msg.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Network error: {error_msg}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                logger.error(f"Unexpected error during generation: {e}")
                raise

    def summarize(self, comments: List[str], progress_callback: Callable[[int, int], None] = None) -> str:
        if not comments:
            return ""
            
        chunks = [comments[i:i + SUMMARIZATION_CHUNK_SIZE] for i in range(0, len(comments), SUMMARIZATION_CHUNK_SIZE)]
        
        config = types.GenerateContentConfig(
            safety_settings=self.safety_settings
        )
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, len(chunks))
            prompt = "Summarize the following comments:\n\n" + "\n".join(chunk)
            summary = self._call_with_retry(prompt, config)
            chunk_summaries.append(summary)
            
        if len(chunk_summaries) > 1:
            if progress_callback:
                progress_callback(len(chunks), len(chunks))
            final_prompt = "Combine and summarize these summaries:\n\n" + "\n".join(chunk_summaries)
            return self._call_with_retry(final_prompt, config)
            
        if progress_callback:
            progress_callback(1, 1)
        return chunk_summaries[0]

    def generate_key_themes(self, comments: List[str]) -> List[str]:
        if not comments:
            return []
        
        prompt = "Identify 3 to 5 key themes from the following comments. Return them as a JSON list of strings.\n\n" + "\n".join(comments[:SUMMARIZATION_CHUNK_SIZE * 2])
        
        config = types.GenerateContentConfig(
            safety_settings=self.safety_settings,
            response_mime_type="application/json",
            response_schema={"type": "ARRAY", "items": {"type": "STRING"}}
        )
        
        try:
            res = self._call_with_retry(prompt, config)
            return json.loads(res)
        except Exception as e:
            logger.error(f"Error generating key themes: {e}")
            return []

    def generate_executive_summary(self, comments: List[str], metadata: dict = None) -> dict:
        if not comments:
            return {"summary": "", "key_themes": [], "notable_opinions": [], "controversies": []}
            
        chunks = [comments[i:i + SUMMARIZATION_CHUNK_SIZE] for i in range(0, len(comments), SUMMARIZATION_CHUNK_SIZE)]
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "summary": {"type": "STRING"},
                "key_themes": {"type": "ARRAY", "items": {"type": "STRING"}},
                "notable_opinions": {"type": "ARRAY", "items": {"type": "STRING"}},
                "controversies": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["summary", "key_themes", "notable_opinions", "controversies"]
        }
        
        config = types.GenerateContentConfig(
            safety_settings=self.safety_settings,
            response_mime_type="application/json",
            response_schema=schema
        )
        
        chunk_results = []
        for chunk in chunks:
            prompt = "Generate an executive summary for these comments:\n\n" + "\n".join(chunk)
            try:
                res = self._call_with_retry(prompt, config)
                chunk_results.append(json.loads(res))
            except Exception as e:
                logger.error(f"Error processing executive summary chunk: {e}")
                
        if len(chunk_results) > 1:
            combined_prompt = "Combine the following executive summaries into one final executive summary with the exact same structure:\n\n" + json.dumps(chunk_results)
            try:
                final_res = self._call_with_retry(combined_prompt, config)
                return json.loads(final_res)
            except Exception as e:
                logger.error(f"Error parsing final executive summary: {e}")
                return {"summary": "", "key_themes": [], "notable_opinions": [], "controversies": []}
        elif len(chunk_results) == 1:
            return chunk_results[0]
            
        return {"summary": "", "key_themes": [], "notable_opinions": [], "controversies": []}
