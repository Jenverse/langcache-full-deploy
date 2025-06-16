import os
import time
import json
import uuid
import redis
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Redis connection
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

# Initialize the Redis LangCache embedding model from Hugging Face
print("Loading redis/langcache-embed-v1 model from Hugging Face...")
hf_token = os.getenv("HF_TOKEN")
model = SentenceTransformer("redis/langcache-embed-v1", use_auth_token=hf_token)
print(f"Model loaded successfully. Embedding dimensions: {model.get_sentence_embedding_dimension()}")

app = FastAPI(
    title="Redis LangCache Service",
    description="API for managing a Redis LangCache with redis/langcache-embed-v1 model",
    version="1.0"
)

# Models matching the Redis LangCache API
class CacheCreateRequest(BaseModel):
    indexName: str
    redisUrls: List[str]
    overwriteIfExists: bool = False
    allowExistingData: bool = False
    modelName: str
    defaultSimilarityThreshold: float = 0.9
    defaultTtlMillis: Optional[int] = None
    attributes: Optional[List[str]] = None

class CacheCreateResponse(BaseModel):
    cacheId: str
    timestamp: str

class CacheSearchRequest(BaseModel):
    prompt: str
    similarityThreshold: Optional[float] = None
    attributes: Optional[Dict[str, str]] = None

class CacheEntry(BaseModel):
    id: str
    prompt: str
    response: str
    attributes: Dict[str, str] = {}
    similarity: float
    embedding_time: Optional[float] = None
    search_time: Optional[float] = None
    total_time: Optional[float] = None

class CreateEntryRequest(BaseModel):
    prompt: str
    response: str
    attributes: Optional[Dict[str, str]] = None
    ttlMillis: Optional[int] = None

class CreateEntryResponse(BaseModel):
    entryId: str
    timestamp: str

class ApiErrorResponse(BaseModel):
    code: str
    message: str
    status: str = "error"
    timestamp: str

# Global cache configurations
cache_configs = {}

@app.get("/")
async def root():
    return {
        "message": "Redis LangCache Service is running",
        "model": "redis/langcache-embed-v1",
        "dimensions": model.get_sentence_embedding_dimension()
    }

@app.post("/v1/admin/caches", response_model=CacheCreateResponse)
async def create_cache(request: CacheCreateRequest):
    """Create a new cache"""
    try:
        # Validate model name
        if request.modelName != "redis/langcache-embed-v1":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "MODEL_NOT_FOUND",
                    "message": f"Model '{request.modelName}' not found. Only 'redis/langcache-embed-v1' is supported.",
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        
        cache_id = str(uuid.uuid4())
        
        cache_config = {
            "cacheId": cache_id,
            "indexName": request.indexName,
            "redisUrls": request.redisUrls,
            "modelName": request.modelName,
            "defaultSimilarityThreshold": request.defaultSimilarityThreshold,
            "defaultTtlMillis": request.defaultTtlMillis,
            "attributes": request.attributes or [],
            "created_at": time.time()
        }
        
        # Store cache configuration
        cache_configs[cache_id] = cache_config
        
        # Store in Redis for persistence
        redis_client.hset(f"langcache_config:{cache_id}", mapping={
            k: json.dumps(v) if isinstance(v, (list, dict)) else str(v) 
            for k, v in cache_config.items()
        })
        
        return CacheCreateResponse(
            cacheId=cache_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"Error creating cache: {str(e)}",
                "status": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

@app.post("/v1/caches/{cache_id}/entries", response_model=CreateEntryResponse)
async def create_entry(cache_id: str, request: CreateEntryRequest):
    """Create a new cache entry"""
    try:
        # Check if cache exists
        cache_config = cache_configs.get(cache_id)
        if not cache_config:
            # Try to load from Redis
            config_data = redis_client.hgetall(f"langcache_config:{cache_id}")
            if not config_data:
                raise HTTPException(status_code=400, detail="Cache not found")
            
            cache_config = {}
            for k, v in config_data.items():
                key = k.decode()
                value = v.decode()
                try:
                    cache_config[key] = json.loads(value)
                except:
                    cache_config[key] = value
            cache_configs[cache_id] = cache_config
        
        # Generate embedding for the prompt
        embedding = model.encode([request.prompt], normalize_embeddings=True)[0]
        
        # Create entry ID
        entry_id = str(uuid.uuid4())
        
        # Store in Redis
        entry_data = {
            "id": entry_id,
            "prompt": request.prompt,
            "response": request.response,
            "attributes": json.dumps(request.attributes or {}),
            "embedding": json.dumps(embedding.astype(np.float32).tolist()),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        redis_client.hset(f"langcache_entry:{cache_id}:{entry_id}", mapping=entry_data)
        redis_client.sadd(f"langcache_entries:{cache_id}", entry_id)
        
        return CreateEntryResponse(
            entryId=entry_id,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INTERNAL_ERROR", 
                "message": f"Error creating entry: {str(e)}",
                "status": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

@app.post("/v1/caches/{cache_id}/search", response_model=List[CacheEntry])
async def search_cache(cache_id: str, request: CacheSearchRequest):
    """Search for similar entries in the cache"""
    try:
        # Check if cache exists
        cache_config = cache_configs.get(cache_id)
        if not cache_config:
            # Try to load from Redis
            config_data = redis_client.hgetall(f"langcache_config:{cache_id}")
            if not config_data:
                raise HTTPException(status_code=400, detail="Cache not found")
            
            cache_config = {}
            for k, v in config_data.items():
                key = k.decode()
                value = v.decode()
                try:
                    cache_config[key] = json.loads(value)
                except:
                    cache_config[key] = value
            cache_configs[cache_id] = cache_config
        
        # Generate embedding for the search prompt
        embedding_start_time = time.time()
        query_embedding = model.encode([request.prompt], normalize_embeddings=True)[0]
        embedding_time = time.time() - embedding_start_time

        similarity_threshold = request.similarityThreshold or float(cache_config.get('defaultSimilarityThreshold', 0.9))

        # Get all entries for this cache and search
        search_start_time = time.time()
        entry_ids = redis_client.smembers(f"langcache_entries:{cache_id}")

        results = []

        for entry_id in entry_ids:
            entry_data = redis_client.hgetall(f"langcache_entry:{cache_id}:{entry_id.decode()}")
            if not entry_data:
                continue
                
            # Decode entry data
            entry = {k.decode(): v.decode() for k, v in entry_data.items()}
            stored_embedding = np.array(json.loads(entry['embedding']))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= similarity_threshold:
                results.append(CacheEntry(
                    id=entry['id'],
                    prompt=entry['prompt'],
                    response=entry['response'],
                    attributes=json.loads(entry.get('attributes', '{}')),
                    similarity=float(similarity)
                ))
        
        search_time = time.time() - search_start_time

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.similarity, reverse=True)

        # Always add timing information, even for cache misses
        if results:
            # Add timing data to the first result for cache hits
            first_result = results[0]
            first_result.embedding_time = embedding_time
            first_result.search_time = search_time
            first_result.total_time = embedding_time + search_time
        else:
            # For cache misses, create a dummy result with just timing data
            dummy_result = CacheEntry(
                id="cache_miss_timing",
                prompt="",
                response="",
                attributes={},
                similarity=0.0,
                embedding_time=embedding_time,
                search_time=search_time,
                total_time=embedding_time + search_time
            )
            results.append(dummy_result)

        return results
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "INTERNAL_ERROR",
                "message": f"Error searching cache: {str(e)}",
                "status": "error", 
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

if __name__ == "__main__":
    print(f"Starting Redis LangCache service with model: redis/langcache-embed-v1")
    uvicorn.run(app, host="0.0.0.0", port=8080)
