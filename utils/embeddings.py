"""
Unified Embedding Services
Integrates all embedding models directly into the main Flask app
"""

import os
import json
import uuid
import time
import numpy as np
import redis
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

# Global variables for models
_redis_langcache_model = None
_openai_client = None

def get_redis_client(redis_url=None):
    """Get Redis client"""
    url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379')
    return redis.from_url(url, decode_responses=True)

def get_openai_client(api_key=None):
    """Get OpenAI client"""
    global _openai_client
    key = api_key or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError("OpenAI API key required")
    
    if not _openai_client or api_key:  # Create new client if API key provided
        _openai_client = OpenAI(api_key=key)
    return _openai_client

def get_redis_langcache_model():
    """Get Redis LangCache model (sentence-transformers)"""
    global _redis_langcache_model
    if _redis_langcache_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            hf_token = os.environ.get("HF_TOKEN")

            if not hf_token:
                print("⚠ Warning: HF_TOKEN not provided. Redis LangCache model requires Hugging Face authentication.")
                print("   Get your token from: https://huggingface.co/settings/tokens")
                _redis_langcache_model = None
                return None

            model_name = "redis/langcache-embed-v1"
            _redis_langcache_model = SentenceTransformer(model_name, use_auth_token=hf_token)
            print(f"✓ Loaded Redis LangCache model: {model_name}")
        except Exception as e:
            print(f"⚠ Warning: Could not load Redis LangCache model: {e}")
            print("   This might be due to:")
            print("   1. Invalid or missing HF_TOKEN")
            print("   2. No access to redis/langcache-embed-v1 model")
            print("   3. Network connectivity issues")
            print("   Falling back to OpenAI embeddings for redis-langcache requests.")
            _redis_langcache_model = None
    return _redis_langcache_model

def get_embedding(text: str, model_type: str = "openai", api_key: str = None) -> List[float]:
    """
    Get embedding for text using specified model
    
    Args:
        text: Input text
        model_type: 'openai', 'redis-langcache', or 'ollama-bge'
        api_key: API key for OpenAI (if using OpenAI model)
    
    Returns:
        List of embedding values
    """
    if model_type == "openai-text-embedding-small":
        client = get_openai_client(api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    elif model_type == "redis-langcache":
        model = get_redis_langcache_model()
        if model is None:
            # Fallback to OpenAI if Redis model not available
            return get_embedding(text, "openai-text-embedding-small", api_key)
        
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding.astype(np.float32).tolist()
    
    elif model_type == "ollama-bge":
        # For now, fallback to redis-langcache (same BGE model)
        return get_embedding(text, "redis-langcache", api_key)
    
    else:
        # Default to OpenAI
        return get_embedding(text, "openai-text-embedding-small", api_key)

def create_cache(cache_name: str, redis_url: str) -> str:
    """
    Create a new cache
    
    Returns:
        cache_id: Unique identifier for the cache
    """
    try:
        cache_id = str(uuid.uuid4())
        redis_client = get_redis_client(redis_url)
        
        cache_config = {
            "cacheId": cache_id,
            "indexName": cache_name,
            "redisUrl": redis_url,
            "defaultSimilarityThreshold": "0.85",
            "created_at": str(time.time())
        }
        
        # Store cache configuration
        redis_client.hset(f"cache_config:{cache_id}", mapping=cache_config)
        
        print(f"✓ Created cache: {cache_id}")
        return cache_id
        
    except Exception as e:
        print(f"✗ Error creating cache: {e}")
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
        query_embedding = np.array(get_embedding(prompt, embedding_model, api_key))
        
        # Get all entries for this cache
        entry_ids = redis_client.smembers(f"cache_entries:{cache_id}")
        
        best_match = None
        best_similarity = 0
        
        for entry_id in entry_ids:
            entry_data = redis_client.hgetall(f"cache_entry:{cache_id}:{entry_id}")
            if not entry_data:
                continue
            
            stored_embedding = np.array(json.loads(entry_data['embedding']))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
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
