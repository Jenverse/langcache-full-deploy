from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import numpy
import uvicorn
import os
import uuid
import time
import json
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Redis connection
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

app = FastAPI(
    title="Embeddings API",
    description="OpenAI-compatible API for generating embeddings using sentence-transformers",
    version="0.0.1",
)

# Global cache configurations
cache_configs = {}

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN environment variable not set. Some models may not be accessible.")

model_name = "redis/langcache-embed-v1"
# Pass token to SentenceTransformer if available
model = SentenceTransformer(model_name, use_auth_token=hf_token)
embedding_dimensions = model.get_sentence_embedding_dimension()

# Print the embedding dimensions for configuration
print(f"Model: {model_name}, Embedding Dimensions: {embedding_dimensions}")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    usage: Dict[str, int]


class CreateCacheRequest(BaseModel):
    indexName: str
    redisUrls: List[str]
    overwriteIfExists: bool = True
    allowExistingData: bool = True
    defaultSimilarityThreshold: float = 0.85


class CacheEntry(BaseModel):
    prompt: str
    response: str


class SearchRequest(BaseModel):
    prompt: str
    similarity_threshold: Optional[float] = 0.85


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingRequest):
    try:

        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        embeddings = model.encode(texts, normalize_embeddings=True)

        data = []
        for i, embedding in enumerate(embeddings):
            embedding_list = embedding.astype(numpy.float32).tolist()
            data.append(EmbeddingData(embedding=embedding_list, index=i))

        total_tokens = sum(len(text.split()) *
                           1.3 for text in texts)

        return EmbeddingsResponse(
            data=data,
            usage={
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Embeddings API is running",
        "model": model_name,
        "dimensions": embedding_dimensions
    }


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
        embedding_request = EmbeddingRequest(input=entry.prompt)
        embedding_response = await create_embeddings(embedding_request)
        embedding = embedding_response.data[0].embedding

        # Create entry ID
        entry_id = str(uuid.uuid4())

        # Store in Redis
        cache_data = {
            "entryId": entry_id,
            "prompt": entry.prompt,
            "response": entry.response,
            "embedding": json.dumps(embedding),
            "created_at": str(time.time())
        }

        redis_client.hset(f"cache_entry:{cache_id}:{entry_id}", mapping=cache_data)
        redis_client.sadd(f"cache_entries:{cache_id}", entry_id)

        return {"entryId": entry_id, "status": "added"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding cache entry: {str(e)}")


@app.post("/v1/caches/{cache_id}/search")
async def search_cache(cache_id: str, request: SearchRequest):
    """Search for similar entries in the cache"""
    try:
        # Get embedding for the search prompt
        embedding_request = EmbeddingRequest(input=request.prompt)
        embedding_response = await create_embeddings(embedding_request)
        query_embedding = numpy.array(embedding_response.data[0].embedding)

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
            stored_embedding = numpy.array(json.loads(entry['embedding']))

            # Calculate cosine similarity
            similarity = numpy.dot(query_embedding, stored_embedding) / (
                numpy.linalg.norm(query_embedding) * numpy.linalg.norm(stored_embedding)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
