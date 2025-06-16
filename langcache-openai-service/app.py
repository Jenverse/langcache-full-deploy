#!/usr/bin/env python3
"""
OpenAI Embeddings LangCache Service
A LangCache-compatible service using OpenAI text-embedding-small
"""

from flask import Flask, request, jsonify
import redis
import json
import uuid
import time
import os
from datetime import datetime
import numpy as np
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client
openai_client = None

def get_openai_client():
    """Get OpenAI client with API key from request headers"""
    api_key = request.headers.get('X-OpenAI-API-Key')
    if not api_key:
        # Try environment variable as fallback
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    return OpenAI(api_key=api_key)

def get_redis_client():
    """Get Redis client from environment or default"""
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    return redis.from_url(redis_url, decode_responses=True)

def get_embedding(text, client):
    """Get embedding using OpenAI text-embedding-small"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'openai-embeddings'})

@app.route('/v1/admin/caches', methods=['POST'])
def create_cache():
    """Create a new cache"""
    try:
        data = request.get_json()
        
        # Extract parameters
        index_name = data.get('indexName', 'default_cache')
        redis_urls = data.get('redisUrls', [])
        embedding_model = data.get('embeddingModel', 'openai-text-embedding-small')
        
        # Generate unique cache ID
        cache_id = str(uuid.uuid4())
        
        # Store cache metadata
        redis_client = get_redis_client()
        cache_key = f"langcache:cache:{cache_id}"
        cache_metadata = {
            'cache_id': cache_id,
            'index_name': index_name,
            'embedding_model': embedding_model,
            'created_at': datetime.now().isoformat(),
            'redis_urls': json.dumps(redis_urls)
        }
        
        redis_client.hset(cache_key, mapping=cache_metadata)
        
        print(f"Created OpenAI embeddings cache: {cache_id}")
        
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
        similarity_threshold = data.get('similarity_threshold', 0.85)
        
        if not query:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Generate embedding for query
        query_embedding = get_embedding(query, client)
        
        # Search for similar entries in Redis
        redis_client = get_redis_client()
        
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
                
            stored_embedding = json.loads(stored_embedding_str)
            
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
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Generate embedding for prompt
        embedding = get_embedding(prompt, client)
        
        # Generate entry ID
        entry_id = str(uuid.uuid4())
        
        # Store in Redis
        redis_client = get_redis_client()
        entry_key = f"langcache:entry:{cache_id}:{entry_id}"
        
        entry_data = {
            'entry_id': entry_id,
            'cache_id': cache_id,
            'prompt': prompt,
            'response': response,
            'embedding': json.dumps(embedding),
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
    port = int(os.environ.get('PORT', 8082))
    app.run(host='0.0.0.0', port=port, debug=False)
