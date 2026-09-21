import logging
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
import config

logger = logging.getLogger(__name__)

class QAChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

    def _call_with_retry(self, prompt: str, config_kwargs: dict) -> str:
        """Helper method to handle retries for network and transient API errors."""
        max_retries = 3
        base_delay = 2
        
        generation_config = types.GenerateContentConfig(
            safety_settings=self.safety_settings,
            **config_kwargs
        )
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=prompt,
                    config=generation_config
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
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"APIError {e.code}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for APIError {e.code}")
                        raise
                else:
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
                logger.error(f"Error during generation: {e}")
                raise

    def ask(self, question: str, n_results: int = 5, history: Optional[List[Tuple[str, str]]] = None) -> dict:
        results = self.vector_store.search(question, n_results=n_results)
        
        source_comments = []
        relevance_scores = []
        
        # ChromaDB returns a dictionary, not an object
        if results and 'documents' in results and results['documents']:
            docs = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
            source_comments = docs
            
        source_metadatas = []
        if results and 'metadatas' in results and results['metadatas']:
            metas = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
            source_metadatas = metas
            
        if results and 'distances' in results and results['distances']:
            dists = results['distances'][0] if isinstance(results['distances'][0], list) else results['distances']
            relevance_scores = [max(0.0, 1.0 - d / 2.0) for d in dists]
            
        history_str = ""
        if history:
            history_str = "Conversation History:\n"
            for q, a in history:
                history_str += f"User: {q}\nAI: {a}\n\n"
                
        context = "\n".join(source_comments)
        prompt = f"""You are an expert community manager and data analyst.
Below is a user's question, followed by relevant comments extracted from social media (the Context), and the conversation history.

Please answer the user's question with a rich, detailed, and conversational response.
- Analyze and synthesize the themes in the comments deeply.
- Use Markdown formatting (bullet points, bold text) to make your answer structured and easy to read.
- Write in the highly capable, helpful, and natural tone typical of Gemini.
- If the context doesn't contain enough information to fully answer, provide what you can and politely note the limitations.

{history_str}

### Context (Social Media Comments):
{context}

### User's Question:
{question}
"""
        
        try:
            answer = self._call_with_retry(prompt, {})
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Error generating answer: {str(e)}"
            
        return {
            "answer": answer,
            "source_comments": source_comments,
            "relevance_scores": relevance_scores,
            "source_metadatas": source_metadatas
        }

    def get_suggested_questions(self, comments_sample: List[str]) -> List[str]:
        if not comments_sample:
            return []
            
        prompt = "Based on the following comments, suggest 3 questions a user might ask about this data. Return the questions as a JSON array of strings.\n\n" + "\n".join(comments_sample)
        
        try:
            res_text = self._call_with_retry(prompt, {
                "response_mime_type": "application/json",
                "response_schema": {"type": "ARRAY", "items": {"type": "STRING"}}
            })
            return json.loads(res_text)
        except Exception as e:
            logger.error(f"Error generating suggested questions: {e}")
            return []
