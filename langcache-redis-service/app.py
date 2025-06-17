#!/usr/bin/env python3
"""
Redis LangCache Service
A simple LangCache-compatible service for Redis semantic caching
"""

from flask import Flask, request, jsonify
import redis
import json
import uuid
import time
import os
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize embedding model with HF token
print("Loading Redis LangCache embedding model...")
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    embedding_model = SentenceTransformer('redis/langcache-embed-v1', token=hf_token)
else:
    embedding_model = SentenceTransformer('redis/langcache-embed-v1')
print("✓ Redis LangCache embedding model loaded")

def get_redis_client(redis_url=None):
    """Get Redis client from parameter or environment"""
    if not redis_url:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    return redis.from_url(redis_url, decode_responses=True)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'redis-langcache'})

@app.route('/v1/admin/caches', methods=['POST'])
def create_cache():
    """Create a new cache"""
    try:
        data = request.get_json()
        
        # Extract parameters
        index_name = data.get('indexName', 'default_cache')
        redis_urls = data.get('redisUrls', [])
        model_name = data.get('modelName', 'redis/langcache-embed-v1')

        # Get Redis URL from request
        redis_url = redis_urls[0] if redis_urls else None

        # Generate unique cache ID
        cache_id = str(uuid.uuid4())

        # Store cache metadata
        redis_client = get_redis_client(redis_url)
        cache_key = f"langcache:cache:{cache_id}"
        cache_metadata = {
            'cache_id': cache_id,
            'index_name': index_name,
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'redis_urls': json.dumps(redis_urls)
        }
        
        redis_client.hset(cache_key, mapping=cache_metadata)
        
        print(f"Created Redis LangCache: {cache_id}")
        
        return jsonify({
            'cacheId': cache_id,
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        print(f"Error creating cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/caches/<cache_id>/search', methods=['POST'])
def search_cache(cache_id):
    """Search for similar queries in cache"""
    try:
        data = request.get_json()
        query = data.get('prompt', '')
        similarity_threshold = data.get('similarityThreshold', 0.85)
        
        if not query:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Generate embedding for query
        query_embedding = embedding_model.encode(query)
        
        # Get Redis URL from headers or use default
        redis_url = request.headers.get('X-Redis-URL')

        # Search for similar entries in Redis
        redis_client = get_redis_client(redis_url)
        
        # Get all cached entries for this cache
        pattern = f"langcache:entry:{cache_id}:*"
        entry_keys = redis_client.keys(pattern)
        
        best_match = None
        best_similarity = 0
        
        for entry_key in entry_keys:
            entry_data = redis_client.hgetall(entry_key)
            if not entry_data:
                continue
                
            # Get stored embedding
            stored_embedding_str = entry_data.get('embedding')
            if not stored_embedding_str:
                continue
                
            stored_embedding = np.array(json.loads(stored_embedding_str))
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = {
                    'id': entry_data.get('entry_id'),
                    'prompt': entry_data.get('prompt'),
                    'response': entry_data.get('response'),
                    'similarity': float(similarity),
                    'entryId': entry_data.get('entry_id')
                }
        
        if best_match:
            print(f"Cache hit! Similarity: {best_similarity:.4f}")
            return jsonify([best_match])
        else:
            print("Cache miss - no similar queries found")
            return jsonify([])
            
    except Exception as e:
        print(f"Error searching cache: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/v1/caches/<cache_id>/entries', methods=['POST'])
def add_to_cache(cache_id):
    """Add a new entry to cache"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        response = data.get('response', '')
        
        if not prompt or not response:
            return jsonify({'error': 'Both prompt and response required'}), 400
        
        # Generate embedding for prompt
        embedding = embedding_model.encode(prompt)
        
        # Generate entry ID
        entry_id = str(uuid.uuid4())
        
        # Get Redis URL from headers or use default
        redis_url = request.headers.get('X-Redis-URL')

        # Store in Redis
        redis_client = get_redis_client(redis_url)
        entry_key = f"langcache:entry:{cache_id}:{entry_id}"
        
        entry_data = {
            'entry_id': entry_id,
            'cache_id': cache_id,
            'prompt': prompt,
            'response': response,
            'embedding': json.dumps(embedding.tolist()),
            'created_at': datetime.now().isoformat()
        }
        
        redis_client.hset(entry_key, mapping=entry_data)
        
        print(f"Added to cache: {entry_id}")
        
        return jsonify({
            'entryId': entry_id,
            'timestamp': datetime.now().isoformat()
        }), 201
        
    except Exception as e:
        print(f"Error adding to cache: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)
