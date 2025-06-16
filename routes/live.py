from flask import Blueprint, render_template, request, jsonify
import datetime
import time
import statistics
from collections import defaultdict
from utils.helpers import (
    get_current_timestamp, create_cache, search_cache, add_to_cache, generate_gemini_response,
    generate_llm_response_with_user_key, latency_data, operations_log, query_matches, cache_ids,
    DEFAULT_EMBEDDING_MODEL, shadow_mode_data
)

live_bp = Blueprint('live', __name__)

@live_bp.route('/')
def index():
    return render_template('index.html')

@live_bp.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query', '')
    use_cache = data.get('use_cache', False)
    llm_model = data.get('llm_model', 'gemini-1.5-flash')
    embedding_model = data.get('embedding_model', 'ollama-bge')
    similarity_threshold = data.get('similarity_threshold', 0.85)
    shadow_mode = data.get('shadow_mode', False)

    # Get user-provided settings (no fallback to env)
    user_openai_key = data.get('user_openai_key')
    user_redis_url = data.get('user_redis_url')

    # Validate that user has provided required settings
    if not user_openai_key:
        return jsonify({
            'error': 'OpenAI API key required. Please configure it in Settings tab.',
            'requires_settings': True
        }), 400

    if not user_redis_url:
        return jsonify({
            'error': 'Redis URL required. Please configure it in Settings tab.',
            'requires_settings': True
        }), 400

    print(f"Processing query: '{query}', use_cache: {use_cache}, llm_model: {llm_model}, embedding_model: {embedding_model}")

    # Track shadow mode if active
    if shadow_mode_data['is_active']:
        shadow_mode_data['stats']['total_queries'] += 1
        
        # Always check cache in shadow mode
        cache_start_time = time.time()
        cached_result = search_cache(query, embedding_model, similarity_threshold, user_redis_url)
        cache_time = time.time() - cache_start_time
        
        # Always call LLM in shadow mode
        llm_start_time = time.time()
        response = generate_llm_response_with_user_key(query, llm_model, user_openai_key)
        llm_time = time.time() - llm_start_time
        
        # Note: Shadow mode statistics are now handled by add_shadow_mode_query() function

    if use_cache:
        # This is the semantic cache panel
        if shadow_mode:
            # SHADOW MODE: Always call LLM but measure cache performance in background

            # First, check cache in background for analysis
            cache_start_time = time.time()
            cached_result = search_cache(query, embedding_model, similarity_threshold, user_redis_url)
            cache_time = time.time() - cache_start_time

            # Get actual timing breakdown if available
            embedding_time = 0
            redis_search_time = 0
            if cached_result and 'embedding_time' in cached_result:
                embedding_time = cached_result['embedding_time']
            if cached_result and 'redis_search_time' in cached_result:
                redis_search_time = cached_result['redis_search_time']

            # For debugging - show actual measured times
            print(f"SHADOW MODE - Cache operation breakdown:")
            print(f"  Total cache time: {cache_time:.4f}s")
            print(f"  Embedding generation: {embedding_time:.4f}s")
            print(f"  Redis search: {redis_search_time:.6f}s")
            print(f"  Network/overhead: {(cache_time - embedding_time - redis_search_time):.4f}s")

            # Track total cache operation latency for the specific embedding model
            timestamp = get_current_timestamp()
            latency_data[embedding_model]['cache'][timestamp].append(cache_time)

            # Log cache hit/miss for shadow mode analysis
            if cached_result and 'response' in cached_result and not cached_result.get('cache_miss'):
                print(f"SHADOW MODE - Cache HIT recorded (similarity: {cached_result.get('similarity', 'N/A')})")
                # Track cache hit for the specific embedding model
                latency_data[embedding_model]['cache_hits'] += 1

                # Track query match for analysis
                if 'entryId' in cached_result:
                    matched_query = cached_result.get('matched_query', 'Unknown')
                    query_matches.append({
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'query': query,
                        'matched_query': matched_query,
                        'model': embedding_model,
                        'similarity': cached_result.get('similarity', 0),
                        'embedding_time': cached_result.get('embedding_time', 0),
                        'cache_id': cached_result.get('entryId', 'Unknown')
                    })
            else:
                print("SHADOW MODE - Cache MISS recorded")
                # Track cache miss for the specific embedding model
                latency_data[embedding_model]['cache_misses'] += 1

            # SHADOW MODE: Always call LLM regardless of cache hit/miss
            print("SHADOW MODE - Always calling LLM for user response")
            llm_start_time = time.time()
            response = generate_llm_response_with_user_key(query, llm_model, user_openai_key)
            llm_time = time.time() - llm_start_time

            # Track LLM latency for the specific embedding model
            timestamp = get_current_timestamp()
            latency_data[embedding_model]['llm'][timestamp].append(llm_time)

            # Store this response in the cache for future shadow mode analysis
            try:
                add_to_cache(query, response, embedding_model, user_redis_url)
                print("SHADOW MODE - Response added to cache for future analysis")
            except Exception as e:
                print(f"SHADOW MODE - Error adding response to cache: {e}")

            # Add shadow mode query data to Redis
            from utils.helpers import add_shadow_mode_query, estimate_tokens

            # Debug: Check what cached_result contains
            print(f"SHADOW MODE DEBUG - cached_result: {cached_result}")

            # Calculate output tokens saved (more accurate estimation)
            # For cache hits: we save all the output tokens since we don't generate them
            # For cache misses: we save 0 tokens since we still generate the response
            output_tokens_saved = 0
            is_cache_hit = cached_result and 'response' in cached_result and not cached_result.get('cache_miss')

            if is_cache_hit:
                # Cache hit - we saved generating the entire response
                cached_response = cached_result.get('response', '')
                output_tokens_saved = estimate_tokens(cached_response)  # More accurate token count
                print(f"SHADOW MODE DEBUG - Cache HIT detected, tokens saved: {output_tokens_saved}")
            else:
                # Cache miss - we still generated the response, so no tokens saved
                output_tokens_saved = 0
                print(f"SHADOW MODE DEBUG - Cache MISS detected, tokens saved: 0")

            # Get the current active cache ID for this embedding model (same as Shadow Mode uses)
            from utils.helpers import cache_ids
            current_cache_id = cache_ids.get(embedding_model, 'Unknown')

            shadow_query_data = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'query': query,
                'matched_query': cached_result.get('matched_query', '') if cached_result else '',
                'similarity': cached_result.get('similarity', 0) if cached_result else 0,
                'llm_time': llm_time,
                'cache_time': cache_time,
                'potential_savings': max(0, llm_time - cache_time),
                'tokens_saved': output_tokens_saved,  # Output tokens we saved by not generating
                'input_tokens': estimate_tokens(query),   # Input tokens (more accurate)
                'output_tokens': estimate_tokens(response), # Actual output tokens generated (more accurate)
                'cache_hit': is_cache_hit,
                'model': embedding_model,
                'cache_id': current_cache_id  # Always use current active cache ID for every record
            }

            print(f"SHADOW MODE DEBUG - Recording query data: cache_hit={is_cache_hit}, model={embedding_model}")
            add_shadow_mode_query(shadow_query_data, user_redis_url)

            # Return LLM response with cache timing for shadow mode display
            return jsonify({
                'response': response,
                'source': 'shadow_llm',  # Special source to indicate shadow mode
                'time_taken': cache_time,  # Show cache timing for analysis
                'llm_time': llm_time,  # Also include LLM time
                'cache_hit': cached_result and 'response' in cached_result and not cached_result.get('cache_miss'),
                'similarity': cached_result.get('similarity') if cached_result and 'response' in cached_result else None
            })
        else:
            # NORMAL LIVE MODE: Use cache if available, otherwise call LLM

            # First, check if we have a similar query in the Redis semantic cache
            cache_start_time = time.time()
            cached_result = search_cache(query, embedding_model, similarity_threshold, user_redis_url)
            cache_time = time.time() - cache_start_time

            # Get actual timing breakdown if available
            embedding_time = 0
            redis_search_time = 0
            if cached_result and 'embedding_time' in cached_result:
                embedding_time = cached_result['embedding_time']
            if cached_result and 'redis_search_time' in cached_result:
                redis_search_time = cached_result['redis_search_time']

            # For debugging - show actual measured times
            print(f"Cache operation breakdown:")
            print(f"  Total cache time: {cache_time:.4f}s")
            print(f"  Embedding generation: {embedding_time:.4f}s")
            print(f"  Redis search: {redis_search_time:.6f}s")
            print(f"  Network/overhead: {(cache_time - embedding_time - redis_search_time):.4f}s")

            # Track total cache operation latency for the specific embedding model
            timestamp = get_current_timestamp()
            latency_data[embedding_model]['cache'][timestamp].append(cache_time)

            # Check if this is a cache miss with timing data
            if cached_result and cached_result.get('cache_miss'):
                print("Cache miss with timing data")
                # Track cache miss for the specific embedding model
                latency_data[embedding_model]['cache_misses'] += 1

                # Continue to LLM call below
                cached_result = None  # Treat as cache miss for the rest of the logic

            if cached_result and 'response' in cached_result:
                # We found a similar query in the cache
                similarity = cached_result.get('similarity', 'N/A')
                print(f"Cache hit! Returning cached response with similarity {similarity}")
                # Track cache hit for the specific embedding model
                latency_data[embedding_model]['cache_hits'] += 1

                # Track query match for analysis
                if 'entryId' in cached_result:
                    # Get the matched query from the cache entry ID
                    matched_query = cached_result.get('matched_query', 'Unknown')
                    # Add to query matches list
                    query_matches.append({
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'query': query,
                        'matched_query': matched_query,
                        'model': embedding_model,
                        'similarity': similarity,
                        'embedding_time': cached_result.get('embedding_time', 0),
                        'cache_id': cached_result.get('entryId', 'Unknown')
                    })

                return jsonify({
                    'response': cached_result['response'],
                    'source': 'cache',
                    'time_taken': cache_time,
                    'similarity': similarity
                })

            # No cache hit, need to call the LLM and store the result
            print("Cache miss. Calling LLM and storing result...")
            # Track cache miss for the specific embedding model
            latency_data[embedding_model]['cache_misses'] += 1

            llm_start_time = time.time()
            response = generate_llm_response_with_user_key(query, llm_model, user_openai_key)
            llm_time = time.time() - llm_start_time

            # Calculate total time (cache search + LLM)
            total_time = cache_time + llm_time

            # Track LLM latency for the specific embedding model
            timestamp = get_current_timestamp()
            latency_data[embedding_model]['llm'][timestamp].append(llm_time)

            # Store this response in the cache for future use
            try:
                add_to_cache(query, response, embedding_model, user_redis_url)
            except Exception as e:
                print(f"Error adding response to cache: {e}")

            return jsonify({
                'response': response,
                'source': 'llm',
                'time_taken': total_time  # Return the total time
            })
    else:
        # This is the direct LLM panel - always call the LLM directly
        print("Direct LLM query (no cache). Calling LLM...")
        llm_start_time = time.time()
        response = generate_llm_response_with_user_key(query, llm_model, user_openai_key)
        llm_time = time.time() - llm_start_time

        # Track LLM latency for direct queries in the separate 'direct-llm' category
        timestamp = get_current_timestamp()
        latency_data['direct-llm']['llm'][timestamp].append(llm_time)

        # Also track which LLM model was used
        latency_data['direct-llm']['models'][llm_model] += 1

        return jsonify({
            'response': response,
            'source': 'llm',
            'time_taken': llm_time
        })

@live_bp.route('/latency-data')
def get_latency_data():
    """Return latency data for all embedding models"""
    # Get data for the current interval
    current_timestamp = get_current_timestamp()

    # Prepare result data structure
    result = {}

    # Process each embedding model
    for model_name in ['ollama-bge', 'redis-langcache', 'openai-text-embedding-small']:
        model_data = latency_data[model_name]

        # Calculate current interval averages
        current_llm_latency = None
        current_cache_latency = None
        current_embedding_latency = None
        current_redis_latency = None
        cache_hit_rate = 0

        # LLM latency
        if current_timestamp in model_data['llm'] and model_data['llm'][current_timestamp]:
            current_llm_latency = statistics.mean(model_data['llm'][current_timestamp])

        # Cache operation latency
        if current_timestamp in model_data['cache'] and model_data['cache'][current_timestamp]:
            current_cache_latency = statistics.mean(model_data['cache'][current_timestamp])

        # Embedding generation latency
        if current_timestamp in model_data['embedding'] and model_data['embedding'][current_timestamp]:
            current_embedding_latency = statistics.mean(model_data['embedding'][current_timestamp])

        # Redis search latency
        if current_timestamp in model_data['redis'] and model_data['redis'][current_timestamp]:
            current_redis_latency = statistics.mean(model_data['redis'][current_timestamp])

        # Cache hit rate
        total_cache_requests = model_data['cache_hits'] + model_data['cache_misses']
        if total_cache_requests > 0:
            cache_hit_rate = model_data['cache_hits'] / total_cache_requests

        # Store metrics for this model
        result[model_name] = {
            'current_llm_latency': round(current_llm_latency, 3) if current_llm_latency is not None else None,
            'current_cache_latency': round(current_cache_latency, 3) if current_cache_latency is not None else None,
            'current_embedding_latency': round(current_embedding_latency, 3) if current_embedding_latency is not None else None,
            'current_redis_latency': round(current_redis_latency, 6) if current_redis_latency is not None else None,
            'cache_hit_rate': round(cache_hit_rate, 2) if cache_hit_rate > 0 else 0
        }

    # Process direct LLM queries (not tied to any embedding model)
    direct_llm_data = latency_data['direct-llm']
    current_direct_llm_latency = None

    # Direct LLM latency
    if current_timestamp in direct_llm_data['llm'] and direct_llm_data['llm'][current_timestamp]:
        current_direct_llm_latency = statistics.mean(direct_llm_data['llm'][current_timestamp])

    # Get most used LLM model
    most_used_model = None
    if direct_llm_data['models']:
        most_used_model = max(direct_llm_data['models'].items(), key=lambda x: x[1])[0]

    # Store metrics for direct LLM queries
    result['direct-llm'] = {
        'current_llm_latency': round(current_direct_llm_latency, 3) if current_direct_llm_latency is not None else None,
        'most_used_model': most_used_model,
        'query_count': sum(direct_llm_data['models'].values())
    }

    return jsonify(result)

@live_bp.route('/query-analysis')
def get_query_analysis():
    """Return query match data for analysis"""
    # Get the embedding model filter from query parameters
    model_filter = request.args.get('model', None)

    # Filter query matches by model if specified
    if model_filter and model_filter in ['ollama-bge', 'redis-langcache', 'openai-text-embedding-small']:
        filtered_matches = [match for match in query_matches if match['model'] == model_filter]
    else:
        filtered_matches = query_matches

    # Sort by timestamp (newest first)
    sorted_matches = sorted(filtered_matches, key=lambda x: x['timestamp'], reverse=True)

    # Return the data
    return jsonify({
        'matches': sorted_matches,
        'total': len(sorted_matches),
        'models': ['ollama-bge', 'redis-langcache', 'openai-text-embedding-small']
    })

@live_bp.route('/operations-log')
def get_operations_log():
    """Return the operations log for the most recent query"""
    return jsonify(operations_log) 