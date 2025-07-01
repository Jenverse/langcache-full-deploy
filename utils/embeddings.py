"""
Unified Embedding Services
Integrates all embedding models directly into the main Flask app
"""

import os
import json
import uuid
import time
import redis
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

# Global variables for models
_openai_client = None

def get_redis_client(redis_url=None):
    """Get Redis client"""
    if not redis_url:
        raise ValueError("Redis URL is required - no localhost fallback in serverless environment")
    return redis.from_url(redis_url, decode_responses=True)

def get_openai_client(api_key=None):
    """Get OpenAI client"""
    global _openai_client
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError("OpenAI API key required")
    
    if not _openai_client or api_key:  # Create new client if API key provided
        _openai_client = OpenAI(api_key=key)
    return _openai_client

# Removed Redis LangCache model - using OpenAI only for Vercel compatibility

def get_embedding(text: str, model_type: str = "openai", api_key: str = None) -> List[float]:
    """
    Get embedding for text using OpenAI (simplified for Vercel deployment)

    Args:
        text: Input text
        model_type: Any model type (all will use OpenAI for Vercel compatibility)
        api_key: API key for OpenAI

    Returns:
        List of embedding values
    """
    # For Vercel deployment, always use OpenAI embeddings regardless of model_type
    client = get_openai_client(api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_cache_and_store_id(redis_url: str) -> str:
    """
    Create cache when user saves configuration and store cache_id in Redis

    Returns:
        cache_id: Unique identifier for the cache
    """
    try:
        redis_client = get_redis_client(redis_url)

        # Check if cache_id already exists
        existing_cache_id = redis_client.get("langcache_cache_id")
        if existing_cache_id:
            print(f"✓ Cache already exists: {existing_cache_id}")
            return existing_cache_id

        # Create new cache_id and store it in Redis
        cache_id = str(uuid.uuid4())
        redis_client.set("langcache_cache_id", cache_id)

        # Store cache metadata
        cache_config = {
            "cacheId": cache_id,
            "created_at": str(time.time()),
            "embedding_model": "openai-text-embedding-small"
        }
        redis_client.hset(f"cache_config:{cache_id}", mapping=cache_config)

        print(f"✓ Created cache and stored ID in Redis: {cache_id}")
        return cache_id

    except Exception as e:
        print(f"✗ Error creating cache: {e}")
        return None

def get_cache_id_from_redis(redis_url: str) -> str:
    """
    Read cache_id from Redis on every request

    Returns:
        cache_id: Cached identifier or None if not found
    """
    try:
        redis_client = get_redis_client(redis_url)
        cache_id = redis_client.get("langcache_cache_id")
        if cache_id:
            return cache_id
        return None
    except Exception as e:
        print(f"✗ Error reading cache_id from Redis: {e}")
        return None

def add_to_cache(cache_id: str, prompt: str, response: str, redis_url: str, 
                embedding_model: str = "openai-text-embedding-small", api_key: str = None) -> bool:
    """
    Add an entry to the cache
    
    Returns:
        bool: Success status
    """
    try:
        redis_client = get_redis_client(redis_url)
        
        # Get embedding for the prompt
        embedding = get_embedding(prompt, embedding_model, api_key)
        
        # Create entry ID
        entry_id = str(uuid.uuid4())
        
        # Store in Redis
        cache_data = {
            "entryId": entry_id,
            "prompt": prompt,
            "response": response,
            "embedding": json.dumps(embedding),
            "embedding_model": embedding_model,
            "created_at": str(time.time())
        }
        
        redis_client.hset(f"cache_entry:{cache_id}:{entry_id}", mapping=cache_data)
        redis_client.sadd(f"cache_entries:{cache_id}", entry_id)
        
        return True
        
    except Exception as e:
        print(f"✗ Error adding to cache: {e}")
        return False

def search_cache(cache_id: str, prompt: str, redis_url: str, 
                similarity_threshold: float = 0.85, embedding_model: str = "openai-text-embedding-small",
                api_key: str = None) -> Optional[Dict]:
    """
    Search for similar entries in the cache
    
    Returns:
        Dict with match info or None if no match
    """
    try:
        redis_client = get_redis_client(redis_url)
        
        # Get embedding for the search prompt
        query_embedding = get_embedding(prompt, embedding_model, api_key)

        # Get all entries for this cache
        entry_ids = redis_client.smembers(f"cache_entries:{cache_id}")

        best_match = None
        best_similarity = 0

        for entry_id in entry_ids:
            entry_data = redis_client.hgetall(f"cache_entry:{cache_id}:{entry_id}")
            if not entry_data:
                continue

            stored_embedding = json.loads(entry_data['embedding'])

            # Calculate cosine similarity using pure Python
            dot_product = sum(a * b for a, b in zip(query_embedding, stored_embedding))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in stored_embedding) ** 0.5
            similarity = dot_product / (norm_a * norm_b)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = {
                    "entryId": entry_data['entryId'],
                    "prompt": entry_data['prompt'],
                    "response": entry_data['response'],
                    "similarity": float(similarity),
                    "embedding_model": entry_data.get('embedding_model', 'unknown')
                }
        
        return best_match
        
    except Exception as e:
        print(f"✗ Error searching cache: {e}")
        return None
