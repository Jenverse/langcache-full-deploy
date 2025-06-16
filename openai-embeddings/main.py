import os
import time
import json
import uuid
import redis
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Redis connection
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

app = FastAPI()

# Models
class EmbeddingRequest(BaseModel):
    input: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    usage: dict

class CacheEntry(BaseModel):
    prompt: str
    response: str

class SearchRequest(BaseModel):
    prompt: str
    similarity_threshold: Optional[float] = 0.85

class CreateCacheRequest(BaseModel):
    indexName: str
    redisUrls: List[str]
    overwriteIfExists: bool = True
    allowExistingData: bool = True
    defaultSimilarityThreshold: float = 0.85

# Global variables
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-small')
cache_configs = {}

@app.get("/")
async def root():
    return {"message": "OpenAI Embeddings API is running", "model": EMBEDDING_MODEL}

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """Create embeddings using OpenAI API"""
    try:
        start_time = time.time()
        
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=request.input
        )
        
        embedding_time = time.time() - start_time
        
        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=EMBEDDING_MODEL,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
                "embedding_time": embedding_time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")

@app.post("/v1/admin/caches")
async def create_cache(request: CreateCacheRequest):
    """Create a new cache configuration"""
    try:
        cache_id = str(uuid.uuid4())
        
        cache_config = {
            "cacheId": cache_id,
            "indexName": request.indexName,
            "redisUrls": request.redisUrls,
            "defaultSimilarityThreshold": request.defaultSimilarityThreshold,
            "created_at": time.time()
        }
        
        # Store cache configuration
        cache_configs[cache_id] = cache_config
        
        # Store in Redis for persistence
        redis_client.hset(f"cache_config:{cache_id}", mapping={
            k: str(v) for k, v in cache_config.items()
        })
        
        return {"cacheId": cache_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating cache: {str(e)}")

@app.post("/v1/caches/{cache_id}/entries")
async def add_cache_entry(cache_id: str, entry: CacheEntry):
    """Add an entry to the cache"""
    try:
        # Get embedding for the prompt
        embedding_response = await create_embedding(EmbeddingRequest(input=entry.prompt))
        embedding = embedding_response.embedding
        
        # Create entry ID
        entry_id = str(uuid.uuid4())
        
        # Store in Redis
        cache_data = {
            "entryId": entry_id,
            "prompt": entry.prompt,
            "response": entry.response,
            "embedding": json.dumps(embedding),
            "created_at": time.time()
        }
        
        redis_client.hset(f"cache_entry:{cache_id}:{entry_id}", mapping={
            k: str(v) for k, v in cache_data.items()
        })
        redis_client.sadd(f"cache_entries:{cache_id}", entry_id)
        
        return {"entryId": entry_id, "status": "added"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding cache entry: {str(e)}")

@app.post("/v1/caches/{cache_id}/search")
async def search_cache(cache_id: str, request: SearchRequest):
    """Search for similar entries in the cache"""
    try:
        # Get embedding for the search prompt
        embedding_response = await create_embedding(EmbeddingRequest(input=request.prompt))
        query_embedding = np.array(embedding_response.embedding)
        
        # Get cache configuration
        cache_config = cache_configs.get(cache_id)
        if not cache_config:
            # Try to load from Redis
            config_data = redis_client.hgetall(f"cache_config:{cache_id}")
            if not config_data:
                raise HTTPException(status_code=404, detail="Cache not found")
            cache_config = {k.decode(): v.decode() for k, v in config_data.items()}
            cache_configs[cache_id] = cache_config
        
        similarity_threshold = request.similarity_threshold or float(cache_config.get('defaultSimilarityThreshold', 0.85))
        
        # Get all entries for this cache
        entry_ids = redis_client.smembers(f"cache_entries:{cache_id}")
        
        best_match = None
        best_similarity = 0
        
        for entry_id in entry_ids:
            entry_data = redis_client.hgetall(f"cache_entry:{cache_id}:{entry_id.decode()}")
            if not entry_data:
                continue
                
            # Decode entry data
            entry = {k.decode(): v.decode() for k, v in entry_data.items()}
            stored_embedding = np.array(json.loads(entry['embedding']))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = {
                    "entryId": entry['entryId'],
                    "prompt": entry['prompt'],
                    "response": entry['response'],
                    "similarity": float(similarity)
                }
        
        if best_match:
            return [best_match]
        else:
            return []
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching cache: {str(e)}")

if __name__ == "__main__":
    print(f"Starting OpenAI Embeddings service with model: {EMBEDDING_MODEL}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
