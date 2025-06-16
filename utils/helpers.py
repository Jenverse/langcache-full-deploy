import datetime
import time
import requests
import statistics
from collections import defaultdict
import os
import json
import redis
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize API clients
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')

genai_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

LANGCACHE_INDEX_NAME = 'llm_cache'
LANGCACHE_URLS = {
    'ollama-bge': 'https://langcache-ollama.onrender.com',
    'redis-langcache': 'https://langcache-redis.onrender.com',
    'openai-text-embedding-small': 'https://langcache-openai.onrender.com'
}

cache_ids = {}
DEFAULT_EMBEDDING_MODEL = 'redis-langcache'

latency_data = {
    'ollama-bge': {
        'llm': defaultdict(list),
        'cache': defaultdict(list),
        'embedding': defaultdict(list),
        'redis': defaultdict(list),
        'cache_hits': 0,
        'cache_misses': 0
    },
    'redis-langcache': {
        'llm': defaultdict(list),
        'cache': defaultdict(list),
        'embedding': defaultdict(list),
        'redis': defaultdict(list),
        'cache_hits': 0,
        'cache_misses': 0
    },
    'openai-text-embedding-small': {
        'llm': defaultdict(list),
        'cache': defaultdict(list),
        'embedding': defaultdict(list),
        'redis': defaultdict(list),
        'cache_hits': 0,
        'cache_misses': 0
    },
    'direct-llm': {
        'llm': defaultdict(list),
        'models': defaultdict(int)
    }
}

operations_log = {
    'query': '',
    'embedding_model': '',
    'cache_id': '',
    'timestamp': '',
    'steps': [],
    'result': {}
}

query_matches = []

# Redis connection for persistent shadow mode data
redis_client = None
try:
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    redis_client = redis.from_url(redis_url, decode_responses=True)
    # Test connection
    redis_client.ping()
    print(f"Connected to Redis for shadow mode data: {redis_url}")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    redis_client = None

# Shadow mode data key in Redis
SHADOW_MODE_KEY = "langcache:shadow_mode_data"

# Global variable to store the last used Redis URL for shadow mode
last_user_redis_url = None

# In-memory fallback for shadow mode data (if Redis is not available)
shadow_mode_data = {
    'is_active': True,  # Always active
    'start_time': datetime.datetime.now(),  # Start immediately
    'queries': [],
    'stats': {
        'total_queries': 0,
        'cache_hits': 0,
        'total_llm_time': 0,
        'total_cache_time': 0,
        'total_savings': 0
    }
}

def add_shadow_mode_query(query_data, user_redis_url=None):
    """Add a shadow mode query to Redis (or memory fallback)"""
    global last_user_redis_url
    user_redis_client = None

    # Try to use user-provided Redis URL first
    if user_redis_url:
        try:
            import redis
            user_redis_client = redis.from_url(user_redis_url, decode_responses=True)
            user_redis_client.ping()  # Test connection
            print(f"Using user Redis for shadow mode: {user_redis_url[:20]}...")
            # Store the Redis URL globally for later retrieval
            last_user_redis_url = user_redis_url
        except Exception as e:
            print(f"Failed to connect to user Redis: {e}")
            user_redis_client = None

    # Fallback to global redis_client or memory
    active_redis_client = user_redis_client or redis_client

    if active_redis_client:
        try:
            # Get existing data from Redis
            existing_data = active_redis_client.get(SHADOW_MODE_KEY)
            if existing_data:
                shadow_data = json.loads(existing_data)
            else:
                shadow_data = {
                    'is_active': True,
                    'start_time': datetime.datetime.now().isoformat(),
                    'queries': [],
                    'stats': {
                        'total_queries': 0,
                        'cache_hits': 0,
                        'total_llm_time': 0,
                        'total_cache_time': 0,
                        'total_savings': 0
                    }
                }

            # Add new query
            shadow_data['queries'].append(query_data)

            # Update stats
            shadow_data['stats']['total_queries'] += 1
            if query_data.get('cache_hit'):
                shadow_data['stats']['cache_hits'] += 1
            shadow_data['stats']['total_llm_time'] += query_data.get('llm_time', 0)
            shadow_data['stats']['total_cache_time'] += query_data.get('cache_time', 0)
            shadow_data['stats']['total_savings'] += query_data.get('potential_savings', 0)

            # Store back to Redis
            active_redis_client.set(SHADOW_MODE_KEY, json.dumps(shadow_data))
            print(f"Shadow mode query added to Redis: {query_data.get('query', '')[:50]}...")

        except Exception as e:
            print(f"Error storing shadow mode data to Redis: {e}")
            # Fallback to memory
            shadow_mode_data['queries'].append(query_data)
            print(f"Shadow mode query added to memory fallback: {query_data.get('query', '')[:50]}...")
    else:
        # Use memory fallback
        shadow_mode_data['queries'].append(query_data)
        print(f"Shadow mode query added to memory fallback: {query_data.get('query', '')[:50]}...")

def get_shadow_mode_data(user_redis_url=None):
    """Get shadow mode data from Redis (or memory fallback)"""
    global last_user_redis_url
    user_redis_client = None

    # Try to use user-provided Redis URL first, or fall back to the last used one
    redis_url_to_use = user_redis_url or last_user_redis_url

    if redis_url_to_use:
        try:
            import redis
            user_redis_client = redis.from_url(redis_url_to_use, decode_responses=True)
            user_redis_client.ping()  # Test connection
            print(f"Using user Redis for shadow mode retrieval: {redis_url_to_use[:20]}...")
        except Exception as e:
            print(f"Failed to connect to user Redis for retrieval: {e}")
            user_redis_client = None

    # Fallback to global redis_client
    active_redis_client = user_redis_client or redis_client

    if active_redis_client:
        try:
            existing_data = active_redis_client.get(SHADOW_MODE_KEY)
            if existing_data:
                print(f"Retrieved shadow mode data from Redis: {len(json.loads(existing_data).get('queries', []))} queries")
                return json.loads(existing_data)
        except Exception as e:
            print(f"Error retrieving shadow mode data from Redis: {e}")

    # Return memory fallback
    print(f"Using memory fallback for shadow mode data: {len(shadow_mode_data.get('queries', []))} queries")
    return shadow_mode_data

def estimate_tokens(text):
    """Estimate token count for text (1 token ≈ 4 characters)"""
    if not text:
        return 0
    # Simple and accurate rule: 1 token per 4 characters
    return len(text) // 4

def get_current_timestamp():
    now = datetime.datetime.now()
    minute = 15 * (now.minute // 15)
    rounded_time = now.replace(minute=minute, second=0, microsecond=0)
    return rounded_time.strftime('%Y-%m-%d %H:%M')

def create_cache(user_redis_url=None):
    global cache_ids

    # Use user Redis URL or fallback to environment
    redis_url = user_redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379')

    # Handle custom embedding services (ollama-bge and openai-embeddings)
    for model_name in ['ollama-bge', 'openai-text-embedding-small']:
        base_url = LANGCACHE_URLS[model_name]
        url = f"{base_url}/v1/admin/caches"
        payload = {
            "indexName": f"{LANGCACHE_INDEX_NAME}_{model_name}",
            "redisUrls": [redis_url],
            "embeddingModel": model_name,
            "overwriteIfExists": True,
            "allowExistingData": True,
            "defaultSimilarityThreshold": 0.85
        }
        try:
            # Add OpenAI API key header for OpenAI service
            headers = {'Content-Type': 'application/json'}
            if model_name == 'openai-text-embedding-small':
                # Get OpenAI API key from user settings (passed via user_redis_url parameter)
                # For now, we'll need to pass this differently
                pass

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200 or response.status_code == 201:
                data = response.json()
                cache_id = data.get('cacheId')
                cache_ids[model_name] = cache_id
                print(f"Created cache for {model_name}: {cache_id}")
        except Exception as e:
            print(f"Error creating cache for {model_name}: {e}")

    # Handle Redis LangCache service (different API)
    model_name = 'redis-langcache'
    try:
        base_url = LANGCACHE_URLS[model_name]

        payload = {
            "indexName": f"{LANGCACHE_INDEX_NAME}_{model_name}",
            "redisUrls": [redis_url],
            "modelName": "redis/langcache-embed-v1",
            "overwriteIfExists": True,
            "allowExistingData": True,
            "defaultSimilarityThreshold": 0.85
        }

        url = f"{base_url}/v1/admin/caches"
        response = requests.post(url, json=payload)

        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            cache_id = data.get('cacheId')
            cache_ids[model_name] = cache_id
            print(f"Created Redis LangCache: {cache_id}")
        else:
            print(f"Redis LangCache creation failed: {response.status_code} - {response.text}")
            # For now, use a default cache ID for Redis LangCache
            cache_ids[model_name] = f"{LANGCACHE_INDEX_NAME}_{model_name}"
            print(f"Using default cache ID for {model_name}: {cache_ids[model_name]}")

    except Exception as e:
        print(f"Error creating Redis LangCache: {e}")
        # Use a default cache ID as fallback
        cache_ids[model_name] = f"{LANGCACHE_INDEX_NAME}_{model_name}"
        print(f"Using fallback cache ID for {model_name}: {cache_ids[model_name]}")

    return len(cache_ids) > 0

def search_cache(query, embedding_model="ollama-bge", similarity_threshold=None, user_redis_url=None):
    """Search for a similar query in the cache using the specified embedding model"""
    global operations_log

    # Reset operations log for new query
    operations_log = {
        'query': query,
        'embedding_model': embedding_model,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steps': [],
        'result': {}
    }

    # Log the query processing step
    operations_log['steps'].append({
        'step': 'QUERY PROCESSING',
        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
        'details': {
            'user_query': query,
            'embedding_model': embedding_model
        }
    })

    # Get the cache ID for the selected embedding model
    cache_id = cache_ids.get(embedding_model)
    if not cache_id:
        print(f"No cache_id available for {embedding_model}, skipping cache search")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'error': f'No cache_id available for {embedding_model}'
            }
        })
        return None

    # Use default similarity threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = 0.85

    # Get the base URL for the selected embedding model
    base_url = LANGCACHE_URLS.get(embedding_model)
    if not base_url:
        print(f"No base URL available for {embedding_model}")
        return None

    # Use provided threshold or default to 0.85
    threshold = similarity_threshold if similarity_threshold is not None else 0.85

    # Handle different API formats
    if embedding_model == 'redis-langcache':
        # Redis LangCache API format
        url = f"{base_url}/v1/caches/{cache_id}/search"
        payload = {
            "prompt": query,
            "similarityThreshold": threshold
        }
    else:
        # Custom embedding services API format
        url = f"{base_url}/v1/caches/{cache_id}/search"
        payload = {
            "prompt": query,
            "similarity_threshold": threshold
        }

    try:
        print(f"Searching cache with {embedding_model} at {url}")

        # Track embedding generation time
        embedding_start_time = time.time()

        response = requests.post(url, json=payload)

        total_request_time = time.time() - embedding_start_time

        if response.status_code == 200:
            data = response.json()
            print(f"Cache search response: {data}")

            # Log cache search step
            operations_log['steps'].append({
                'step': 'CACHE SEARCH',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'cache_id': cache_id,
                    'search_time': f"{total_request_time:.3f}s",
                    'results_found': len(data) if isinstance(data, list) else 0
                }
            })

            if data and len(data) > 0:
                # Found a match
                match = data[0]
                similarity = match.get('similarity', 0)
                cached_response = match.get('response', '')
                entry_id = match.get('entryId', '')
                matched_query = match.get('prompt', '')

                print(f"Cache hit! Similarity: {similarity}, Entry ID: {entry_id}")

                # Log cache hit
                operations_log['steps'].append({
                    'step': 'CACHE HIT',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'similarity': similarity,
                        'entry_id': entry_id,
                        'matched_query': matched_query[:100] + '...' if len(matched_query) > 100 else matched_query
                    }
                })

                # Update result summary
                operations_log['result'] = {
                    'source': 'cache',
                    'similarity': similarity,
                    'total_time': total_request_time
                }

                # Extract actual timing data if available
                embedding_time = match.get('embedding_time', total_request_time * 0.8)
                search_time = match.get('search_time', total_request_time * 0.2)

                return {
                    'response': cached_response,
                    'similarity': similarity,
                    'entryId': entry_id,
                    'matched_query': matched_query,
                    'embedding_time': embedding_time,
                    'redis_search_time': search_time,
                    'total_cache_time': total_request_time
                }
            else:
                print("No cache match found")
                # Log cache miss
                operations_log['steps'].append({
                    'step': 'CACHE MISS',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'message': 'No similar queries found in cache'
                    }
                })
                return None
        else:
            print(f"Cache search failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error searching cache: {e}")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'message': f"Cache search error: {str(e)}"
            }
        })
        return None

def add_to_cache(query, response, embedding_model="ollama-bge", user_redis_url=None):
    """Add a query-response pair to the cache using the specified embedding model"""
    # Get the cache ID for the selected embedding model
    cache_id = cache_ids.get(embedding_model)
    if not cache_id:
        print(f"No cache_id available for {embedding_model}, skipping cache addition")
        return False

    # Get the base URL for the selected embedding model
    base_url = LANGCACHE_URLS.get(embedding_model)
    if not base_url:
        print(f"No base URL available for {embedding_model}")
        return False

    # Handle different API formats
    if embedding_model == 'redis-langcache':
        # Redis LangCache API format
        url = f"{base_url}/v1/caches/{cache_id}/entries"
        payload = {
            "prompt": query,
            "response": response
        }
    else:
        # Custom embedding services API format
        url = f"{base_url}/v1/caches/{cache_id}/entries"
        payload = {
            "prompt": query,
            "response": response
        }

    try:
        print(f"Adding to cache with {embedding_model} at {url}")
        response_obj = requests.post(url, json=payload)

        if response_obj.status_code == 200 or response_obj.status_code == 201:
            data = response_obj.json()
            entry_id = data.get('entryId', 'Unknown')
            print(f"Successfully added to cache. Entry ID: {entry_id}")

            # Log cache addition step
            if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
                operations_log['steps'].append({
                    'step': 'CACHE ADDITION',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'entry_id': entry_id,
                        'cache_id': cache_id
                    }
                })

            return True
        else:
            print(f"Failed to add to cache: {response_obj.status_code} - {response_obj.text}")
            return False
    except Exception as e:
        print(f"Error adding to cache: {e}")
        return False

def generate_llm_response(query, model_name="gpt-4o-mini"):
    """Generate a response using the specified LLM (OpenAI or Gemini)"""
    try:
        if model_name.startswith('gpt-') or model_name.startswith('o1-'):
            # Use OpenAI
            print(f"Calling OpenAI API with query: {query}, model: {model_name}")
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                max_tokens=150
            )
            print(f"OpenAI API response received from model: {model_name}")
            return response.choices[0].message.content
        else:
            # Use Google Gemini (fallback for gemini models)
            print(f"Calling Gemini API with query: {query}, model: {model_name}")
            response = genai_client.models.generate_content(
                model=model_name,
                contents=[query]
            )
            print(f"Gemini API response received from model: {model_name}")
            return response.text
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        # Fallback to a generic response if there's an error
        return f"I encountered an issue processing your query about '{query}'. Please try again later."

def generate_llm_response_with_user_key(query, model_name="gpt-4o-mini", user_api_key=None):
    """Generate a response using the specified LLM with user-provided API key"""
    if not user_api_key:
        raise ValueError("User API key is required. Please configure it in Settings.")

    try:
        if model_name.startswith('gpt-') or model_name.startswith('o1-'):
            # Use OpenAI with user's API key
            from openai import OpenAI
            user_openai_client = OpenAI(api_key=user_api_key)

            print(f"Calling OpenAI API with user key: {user_api_key[:10]}... model: {model_name}")
            response = user_openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                max_tokens=150
            )
            print(f"OpenAI API response received from model: {model_name}")
            return response.choices[0].message.content
        else:
            # For non-OpenAI models, fall back to env-based client for now
            # (You could extend this to support user-provided Gemini keys too)
            print(f"Calling Gemini API with query: {query}, model: {model_name}")
            response = genai_client.models.generate_content(
                model=model_name,
                contents=[query]
            )
            print(f"Gemini API response received from model: {model_name}")
            return response.text
    except Exception as e:
        print(f"Error calling LLM API with user key: {e}")
        raise ValueError(f"Failed to generate response: {str(e)}")

# Keep the old function name for backward compatibility
def generate_gemini_response(query, model_name="gpt-4o-mini"):
    """Legacy function name - now routes to generate_llm_response"""
    return generate_llm_response(query, model_name)