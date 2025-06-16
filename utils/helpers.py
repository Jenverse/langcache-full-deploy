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
    'ollama-bge': 'http://localhost:8080',
    'redis-langcache': 'http://localhost:8081',
    'openai-text-embedding-small': 'http://localhost:8082'
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
    import uuid

    # Use user Redis URL or fallback to environment
    redis_url = user_redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379')

    if not redis_url or redis_url == 'redis://localhost:6379':
        print("No valid Redis URL provided, using fallback cache IDs")
        # Create fallback cache IDs
        for model_name in ['ollama-bge', 'openai-text-embedding-small', 'redis-langcache']:
            cache_ids[model_name] = f"fallback_{model_name}_{int(time.time())}"
        return True

    try:
        # Test Redis connection
        import redis
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        print(f"Connected to Redis: {redis_url[:30]}...")

        # Create 3 cache IDs directly in Redis
        for model_name in ['ollama-bge', 'openai-text-embedding-small', 'redis-langcache']:
            # Generate unique cache ID
            cache_id = str(uuid.uuid4())
            cache_ids[model_name] = cache_id

            # Store cache metadata in Redis
            cache_key = f"langcache:cache:{cache_id}"
            cache_metadata = {
                'cache_id': cache_id,
                'embedding_model': model_name,
                'created_at': datetime.datetime.now().isoformat(),
                'index_name': f"{LANGCACHE_INDEX_NAME}_{model_name}",
                'redis_url': redis_url[:30] + "..." if len(redis_url) > 30 else redis_url
            }
            redis_client.hset(cache_key, mapping=cache_metadata)
            print(f"Created cache for {model_name}: {cache_id}")

        return True

    except Exception as e:
        print(f"Error creating caches with Redis URL: {e}")
        # Use fallback cache IDs
        for model_name in ['ollama-bge', 'openai-text-embedding-small', 'redis-langcache']:
            cache_ids[model_name] = f"fallback_{model_name}_{int(time.time())}"
            print(f"Using fallback cache ID for {model_name}: {cache_ids[model_name]}")
        return False

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
                'message': f"No cache_id available for {embedding_model}"
            }
        })
        return None

    # Store cache ID in operations log
    operations_log['cache_id'] = cache_id

    # Skip LangCache service - simulate cache search for now
    print(f"Simulating cache search for {embedding_model} with cache_id: {cache_id}")

    # Use provided threshold or default to 0.85
    threshold = similarity_threshold if similarity_threshold is not None else 0.85

    # Simulate cache search timing
    embedding_start_time = time.time()

    # For now, always return cache miss since we don't have LangCache services
    # In a real implementation, you'd search your Redis directly here
    time.sleep(0.1)  # Simulate some processing time

    total_request_time = time.time() - embedding_start_time

    print(f"Cache search completed - simulated cache miss for query: {query}")

    # Log cache miss
    operations_log['steps'].append({
        'step': 'CACHE MISS',
        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
        'details': {
            'cache_id': cache_id,
            'similarity_threshold': threshold,
            'message': 'Cache search bypassed - no LangCache services available',
            'search_time': f"{total_request_time:.3f}s"
        }
    })

    # Return cache miss timing data
    return {
        'cache_miss': True,
        'embedding_time': total_request_time * 0.8,
        'redis_search_time': total_request_time * 0.2,
        'total_cache_time': total_request_time
    }

def add_to_cache(query, response, embedding_model="ollama-bge", user_redis_url=None):
    """Add a query-response pair to the cache using Redis directly"""
    # Get the cache ID for the selected embedding model
    cache_id = cache_ids.get(embedding_model)
    if not cache_id:
        print(f"No cache_id available for {embedding_model}, skipping cache addition")
        return False

    try:
        # For now, simulate cache addition since we don't have LangCache services
        print(f"Simulating cache addition for {embedding_model} with cache_id: {cache_id}")

        # In a real implementation, you'd store the query-response pair in Redis here
        # using the cache_id and embedding_model

        # Generate a fake entry ID for logging
        import uuid
        entry_id = str(uuid.uuid4())[:8]

        print(f"Simulated cache addition. Entry ID: {entry_id}")

        # Log cache addition step
        if operations_log.get('query') == query and operations_log.get('embedding_model') == embedding_model:
            operations_log['steps'].append({
                'step': 'CACHE ADDITION',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
                'details': {
                    'entry_id': entry_id,
                    'cache_id': cache_id,
                    'status': 'simulated'
                }
            })

        return True
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