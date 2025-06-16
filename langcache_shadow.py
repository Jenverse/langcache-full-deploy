"""
LangCache Shadow Mode Wrapper for Python

This wrapper enables shadow mode for LangCache, allowing you to run semantic caching
alongside your existing LLM applications without affecting production traffic.

Usage:
    from langcache_shadow import shadow_llm_call
    
    response = shadow_llm_call(
        llm_function=openai.chat.completions.create,
        query="What is AI?",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is AI?"}]
    )
"""

import os
import json
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Any, Dict, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShadowModeConfig:
    """Configuration for shadow mode"""
    
    def __init__(self):
        self.enabled = os.getenv('LANGCACHE_SHADOW_MODE', 'false').lower() == 'true'
        self.api_key = os.getenv('LANGCACHE_API_KEY')
        self.cache_id = os.getenv('LANGCACHE_CACHE_ID')
        self.base_url = os.getenv('LANGCACHE_BASE_URL', 'https://api.langcache.com')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.timeout = int(os.getenv('LANGCACHE_TIMEOUT', '10'))
        
        # Validate required configuration
        if self.enabled and not all([self.api_key, self.cache_id]):
            logger.warning("Shadow mode enabled but missing required configuration")
            self.enabled = False

# Global configuration instance
config = ShadowModeConfig()

class LangCacheClient:
    """Client for interacting with LangCache API"""
    
    def __init__(self, config: ShadowModeConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def search_cache(self, query: str) -> Dict:
        """Search LangCache for semantic matches"""
        try:
            url = f"{self.config.base_url}/v1/caches/{self.config.cache_id}/search"
            data = {"prompt": query}

            logger.info(f"🔍 LANGCACHE SEARCH: {url}")
            logger.info(f"🔍 SEARCH QUERY: {query}")

            response = self.session.post(url, json=data, timeout=self.config.timeout)

            logger.info(f"🔍 LANGCACHE RESPONSE: {response.status_code}")
            logger.info(f"🔍 RESPONSE BODY: {response.text[:200]}...")

            if response.status_code == 200:
                results = response.json()
                logger.info(f"🔍 SEARCH RESULTS: {len(results)} matches found")
                if results:
                    logger.info(f"🔍 FIRST RESULT: {results[0]}")
                return {
                    "hit": len(results) > 0,
                    "results": results,
                    "latency_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                logger.warning(f"Cache search failed: {response.status_code} - {response.text}")
                return {"hit": False, "results": [], "latency_ms": 0}

        except Exception as e:
            logger.error(f"Cache search error: {e}")
            return {"hit": False, "results": [], "latency_ms": 0}
    
    def add_to_cache(self, query: str, response: str) -> bool:
        """Add entry to LangCache"""
        try:
            url = f"{self.config.base_url}/v1/caches/{self.config.cache_id}/entries"
            data = {"prompt": query, "response": response}
            
            result = self.session.post(url, json=data, timeout=self.config.timeout)
            return result.status_code == 201
            
        except Exception as e:
            logger.error(f"Cache add error: {e}")
            return False

class ShadowLogger:
    """Handles logging of shadow mode data"""
    
    def __init__(self, config: ShadowModeConfig):
        self.config = config
        self.redis_client = None
        
        # Initialize Redis client if available
        try:
            import redis
            self.redis_client = redis.from_url(config.redis_url)
        except ImportError:
            logger.warning("Redis not available - shadow data will be logged to file")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    def log_shadow_data(self, shadow_data: Dict):
        """Log shadow mode data asynchronously"""
        def _log():
            try:
                # Shadow data already contains request_id and ts_request
                if self.redis_client:
                    # Store in Redis
                    key = f"shadow:{shadow_data['request_id']}"
                    self.redis_client.set(key, json.dumps(shadow_data))
                else:
                    # Fallback to file logging (JSONL format)
                    with open("shadow_mode.log", "a") as f:
                        f.write(json.dumps(shadow_data) + "\n")

            except Exception as e:
                logger.error(f"Shadow logging error: {e}")

        # Log asynchronously to avoid blocking
        threading.Thread(target=_log, daemon=True).start()

# Global instances
langcache_client = LangCacheClient(config) if config.enabled else None
shadow_logger = ShadowLogger(config) if config.enabled else None

def shadow_llm_call(llm_function: Callable, query: str, *args, **kwargs) -> Any:
    """
    Wrapper function that calls LLM and performs shadow mode operations

    Args:
        llm_function: The LLM function to call (e.g., openai.chat.completions.create)
        query: The user query string
        *args, **kwargs: Arguments to pass to the LLM function

    Returns:
        The LLM response (unchanged)
    """
    start_time = time.time()

    # Always call the LLM function first
    llm_response = llm_function(*args, **kwargs)
    llm_latency = (time.time() - start_time) * 1000

    # Extract response text and metadata
    response_text = _extract_response_text(llm_response)
    model_name = _extract_model_name(llm_response, kwargs)
    token_count = _extract_token_count(llm_response)

    # Perform shadow mode operations if enabled
    if config.enabled and langcache_client and shadow_logger:
        _perform_shadow_operations(query, response_text, llm_latency, model_name, token_count)

    return llm_response

def _extract_response_text(llm_response: Any) -> str:
    """Extract text response from LLM response object"""
    try:
        # Handle OpenAI response format
        if hasattr(llm_response, 'choices') and llm_response.choices:
            return llm_response.choices[0].message.content

        # Handle string responses
        if isinstance(llm_response, str):
            return llm_response

        # Handle dict responses
        if isinstance(llm_response, dict):
            return str(llm_response)

        # Fallback
        return str(llm_response)

    except Exception as e:
        logger.error(f"Error extracting response text: {e}")
        return str(llm_response)

def _extract_model_name(llm_response: Any, kwargs: Dict) -> str:
    """Extract model name from LLM response or kwargs"""
    try:
        # Try to get from kwargs first
        if 'model' in kwargs:
            return kwargs['model']

        # Try to get from response object
        if hasattr(llm_response, 'model'):
            return llm_response.model

        # Check for OpenAI response format
        if hasattr(llm_response, '_request_id'):
            return getattr(llm_response, 'model', 'openai/unknown')

        return "unknown"

    except Exception as e:
        logger.error(f"Error extracting model name: {e}")
        return "unknown"

def _extract_token_count(llm_response: Any) -> int:
    """Extract token count from LLM response if available"""
    try:
        # OpenAI response format
        if hasattr(llm_response, 'usage') and llm_response.usage:
            return getattr(llm_response.usage, 'total_tokens', None)

        return None

    except Exception as e:
        logger.error(f"Error extracting token count: {e}")
        return None

def _estimate_tokens(text: str) -> int:
    """Rough estimation of token count (approximately 4 characters per token)"""
    if not text:
        return 0
    return max(1, len(text) // 4)

def _perform_shadow_operations(query: str, llm_response: str, llm_latency: float, model_name: str = None, tokens_llm: int = None):
    """Perform shadow mode cache operations"""
    def _shadow_ops():
        try:
            # Search cache
            cache_start_time = time.time()
            cache_result = langcache_client.search_cache(query)
            cache_latency = (time.time() - cache_start_time) * 1000

            # Extract cache data if hit
            cache_query = None
            cache_response = None
            vector_distance = None
            cached_id = None

            if cache_result["hit"] and cache_result["results"]:
                first_result = cache_result["results"][0]
                cache_query = first_result.get("prompt", "")
                cache_response = first_result.get("response", "")
                vector_distance = first_result.get("distance", 1.0)
                cached_id = first_result.get("id", "")

            # Add to cache if miss
            if not cache_result["hit"]:
                langcache_client.add_to_cache(query, llm_response)

            # Prepare shadow data in required format
            shadow_data = {
                "request_id": str(uuid.uuid4()),
                "ts_request": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "query": query,
                "rag_response": llm_response,
                "cache_hit": cache_result["hit"],
                "cache_query": cache_query,
                "cache_response": cache_response,
                "vector_distance": vector_distance,
                "cached_id": cached_id,
                "latency_cache_ms": round(cache_latency, 1),
                "latency_llm_ms": round(llm_latency, 1),
                "tokens_llm": tokens_llm or _estimate_tokens(llm_response),
                "model_name": model_name or "unknown",
                "langcache_version": "v0.9.1"
            }

            # Log shadow data
            shadow_logger.log_shadow_data(shadow_data)

        except Exception as e:
            logger.error(f"Shadow operations error: {e}")

    # Run shadow operations in background thread
    threading.Thread(target=_shadow_ops, daemon=True).start()

# Convenience function for simple usage
def track(query: str, llm_response: str):
    """
    Simple tracking function for manual shadow mode integration

    Args:
        query: The user query
        llm_response: The LLM response text
    """
    if config.enabled and langcache_client and shadow_logger:
        _perform_shadow_operations(query, llm_response, 0)

def _perform_shadow_operations(query: str, llm_response: str, llm_latency: float, model_name: str = None, tokens_llm: int = None):
    """Perform shadow mode cache operations"""
    def _shadow_ops():
        try:
            # Search cache
            cache_start_time = time.time()
            cache_result = langcache_client.search_cache(query)
            cache_latency = (time.time() - cache_start_time) * 1000

            # Extract cache data if hit
            cache_query = None
            cache_response = None
            if cache_result["hit"]:
                cache_query = cache_result["results"][0]["prompt"]
                cache_response = cache_result["results"][0]["response"]

            # Extract response data
            response_data = {
                "request_id": str(uuid.uuid4()),
                "ts_request": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "response": llm_response,
                "model": model_name,
                "token_count_llm": tokens_llm,
                "cache_hit": cache_result["hit"],
                "cache_query": cache_query,
                "cache_response": cache_response,
                "latency_ms_llm": llm_latency,
                "latency_ms_cache": cache_latency
            }

            # Log shadow data
            shadow_logger.log_shadow_data(response_data)

        except Exception as e:
            logger.error(f"Shadow operations error: {e}")

    # Log asynchronously to avoid blocking
    threading.Thread(target=_shadow_ops, daemon=True).start()

# ... rest of the file ... 