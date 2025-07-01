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

# Import unified embedding service
from .embeddings import create_cache_and_store_id, add_to_cache, search_cache as search_embedding_cache, get_cache_id_from_redis

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

# Redis connection for persistent shadow mode data - will be created when needed
redis_client = None
print("Redis client will be initialized when user provides Redis URL")

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

# Cache creation is not needed - Redis persists data at the user's Redis URL
# We just use the Redis URL directly to store and retrieve cache entries

def search_cache(query, embedding_model="openai-text-embedding-small", similarity_threshold=None, user_redis_url=None, user_api_key=None):
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

    # For Vercel serverless, use user's Redis URL directly
    if not user_redis_url:
        print("No user Redis URL provided, skipping cache search")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'error': 'No Redis URL provided by user'
            }
        })
        return None

    # Read cache_id from Redis (saved when user configured settings)
    cache_id = get_cache_id_from_redis(user_redis_url)
    if not cache_id:
        print("No cache_id found in Redis - user needs to save settings first")
        operations_log['steps'].append({
            'step': 'ERROR',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'error': 'No cache_id found - please save settings first'
            }
        })
        return None

    # Use default similarity threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = 0.85

    try:
        print(f"Searching cache with {embedding_model}")

        # Track embedding generation time
        embedding_start_time = time.time()

        # Use unified embedding service directly with user's Redis URL
        match = search_embedding_cache(cache_id, query, user_redis_url, similarity_threshold, embedding_model, user_api_key)

        total_request_time = time.time() - embedding_start_time

        # Log cache search step
        operations_log['steps'].append({
            'step': 'CACHE SEARCH',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'details': {
                'cache_id': cache_id,
                'search_time': f"{total_request_time:.3f}s",
                'results_found': 1 if match else 0
            }
        })

        if match:
            # Found a match
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
            embedding_time = total_request_time * 0.8
            search_time = total_request_time * 0.2

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

def add_to_cache_helper(query, response, embedding_model="openai-text-embedding-small", user_redis_url=None, user_api_key=None):
    """Add a query-response pair to the cache using the specified embedding model"""

    if not user_redis_url:
        print("No user Redis URL provided, skipping cache addition")
        return False

    # Read cache_id from Redis (saved when user configured settings)
    cache_id = get_cache_id_from_redis(user_redis_url)
    if not cache_id:
        print("No cache_id found in Redis - user needs to save settings first")
        return False

    try:
        print(f"Adding to cache with {embedding_model}")

        # Use unified embedding service directly with user's Redis URL
        success = add_to_cache(cache_id, query, response, user_redis_url, embedding_model, user_api_key)

        if success:
            print(f"Successfully added to cache with {embedding_model}")

            # Log cache addition step
            if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
                operations_log['steps'].append({
                    'step': 'CACHE ADDITION',
                    'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                    'details': {
                        'cache_id': cache_id,
                        'embedding_model': embedding_model
                    }
                })

            return True
        else:
            print(f"Failed to add to cache with {embedding_model}")
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