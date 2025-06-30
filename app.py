from flask import Flask, send_from_directory, jsonify, render_template, request
from routes.live import live_bp
from routes.shadow import shadow_bp
from utils.helpers import create_cache
import json
import os

app = Flask(__name__)

# Register blueprints
app.register_blueprint(live_bp)
app.register_blueprint(shadow_bp)

# Ensure static files are served properly on Vercel
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/shadow-chatbot')
def shadow_chatbot():
    """Serve the simple chatbot from shadow_mode_public"""
    return send_from_directory('shadow_mode_public/examples/simple-chatbot/templates', 'index.html')

@app.route('/shadow-analysis')
def shadow_analysis():
    """Serve the shadow analysis page"""
    return render_template('shadow_analysis.html')

@app.route('/api/init-cache', methods=['POST'])
def init_cache():
    """Initialize cache with user's Redis URL"""
    try:
        data = request.get_json()
        user_redis_url = data.get('redis_url')

        if not user_redis_url:
            return jsonify({'error': 'Redis URL required'}), 400

        # Create cache using user's Redis URL
        from utils.helpers import create_cache
        success = create_cache(user_redis_url)

        if success:
            return jsonify({'success': True, 'message': 'Cache initialized successfully'})
        else:
            return jsonify({'success': False, 'message': 'Cache initialization failed'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shadow-data')
def get_shadow_data():
    """API endpoint for shadow mode data analysis"""
    from utils.helpers import get_shadow_mode_data

    try:
        # Get shadow data from Redis (or memory fallback)
        shadow_data = get_shadow_mode_data()
        shadow_queries = shadow_data.get('queries', [])

        # Get current active cache IDs (same as Shadow Mode uses)
        from utils.helpers import cache_ids

        # Filter queries to only show data from currently active cache IDs
        filtered_queries = []
        active_cache_info = {}

        for query in shadow_queries:
            model = query.get('model', 'unknown')
            query_cache_id = query.get('cache_id', 'unknown')

            # Get the current active cache ID for this model (same as Shadow Mode uses)
            current_active_cache_id = cache_ids.get(model, 'unknown')

            # STRICT MATCHING: Only include queries that EXACTLY match the current active cache ID
            # This ensures we only see data from the current session, not old cache sessions
            if query_cache_id == current_active_cache_id:
                filtered_queries.append(query)
                active_cache_info[model] = current_active_cache_id
            else:
                # Debug: Show what we're filtering out
                print(f"  Filtering out query with cache_id {query_cache_id} (current active: {current_active_cache_id})")

        print(f"Shadow data filtering: {len(shadow_queries)} total queries -> {len(filtered_queries)} filtered queries")
        for model, cache_id in active_cache_info.items():
            print(f"  {model}: using ONLY current active cache_id {cache_id}")

        if len(filtered_queries) == 0 and len(shadow_queries) > 0:
            print(f"  WARNING: Found {len(shadow_queries)} total queries but none match current cache IDs")
            print(f"  Current cache IDs: {cache_ids}")
            for query in shadow_queries[:3]:  # Show first 3 for debugging
                print(f"    Sample query cache_id: {query.get('cache_id')} for model {query.get('model')}")

        # Convert filtered shadow mode data format to the expected format for the analysis page
        converted_data = []
        for query in filtered_queries:
            converted_data.append({
                'ts_request': query.get('timestamp', ''),
                'query': query.get('query', ''),
                'cache_hit': query.get('cache_hit', False),
                'cache_query': query.get('matched_query', ''),
                'similarity': query.get('similarity', 0),
                'latency_llm_ms': int(query.get('llm_time', 0) * 1000),  # Convert to ms
                'latency_cache_ms': int(query.get('cache_time', 0) * 1000),  # Convert to ms
                'tokens_llm': query.get('tokens_saved', 0),  # Tokens saved by not generating output
                'cache_response': 'Available' if query.get('cache_hit') else 'Not available',
                'embedding_model': query.get('model', 'unknown'),
                'cache_id': query.get('cache_id', 'unknown')
            })

        # Sort by timestamp (newest first)
        converted_data.sort(key=lambda x: x['ts_request'], reverse=True)

        # Calculate metrics
        metrics = calculate_shadow_metrics(converted_data)

        return jsonify({
            "status": "success",
            "data": converted_data,
            "metrics": metrics,
            "total_entries": len(converted_data),
            "message": "No shadow data collected yet" if len(converted_data) == 0 else f"Found {len(converted_data)} entries from latest cache sessions"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "data": [],
            "metrics": get_empty_metrics(),
            "total_entries": 0,
            "message": f"Error reading shadow data: {str(e)}"
        })

def calculate_shadow_metrics(data):
    """Calculate metrics from shadow data"""
    if not data:
        return get_empty_metrics()

    total_queries = len(data)
    cache_hits = sum(1 for item in data if item.get('cache_hit', False))
    cache_misses = total_queries - cache_hits
    hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0

    # Calculate average latencies
    llm_latencies = [item.get('latency_llm_ms', 0) for item in data if item.get('latency_llm_ms')]
    cache_latencies = [item.get('latency_cache_ms', 0) for item in data if item.get('latency_cache_ms')]

    avg_llm_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0
    avg_cache_latency = sum(cache_latencies) / len(cache_latencies) if cache_latencies else 0

    # Calculate latency improvement
    latency_improvement = 0
    if avg_llm_latency > 0 and avg_cache_latency > 0:
        latency_improvement = ((avg_llm_latency - avg_cache_latency) / avg_llm_latency * 100)

    # Calculate cost savings
    total_tokens_saved = sum(item.get('tokens_llm', 0) for item in data if item.get('cache_hit', False))
    total_tokens_used = sum(item.get('tokens_llm', 0) for item in data)

    # Estimate cost savings (assuming $0.002 per 1K tokens for GPT-4)
    cost_per_1k_tokens = 0.002
    cost_saved = (total_tokens_saved / 1000) * cost_per_1k_tokens
    total_cost_without_cache = (total_tokens_used / 1000) * cost_per_1k_tokens
    cost_savings_percentage = (cost_saved / total_cost_without_cache * 100) if total_cost_without_cache > 0 else 0

    return {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate": round(hit_rate, 1),
        "avg_llm_latency": round(avg_llm_latency, 1),
        "avg_cache_latency": round(avg_cache_latency, 1),
        "latency_improvement": round(latency_improvement, 1),
        "total_tokens_saved": total_tokens_saved,
        "cost_saved": round(cost_saved, 4),
        "cost_savings_percentage": round(cost_savings_percentage, 1)
    }

def get_empty_metrics():
    """Return empty metrics structure"""
    return {
        "total_queries": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "hit_rate": 0,
        "avg_llm_latency": 0,
        "avg_cache_latency": 0,
        "latency_improvement": 0,
        "total_tokens_saved": 0,
        "cost_saved": 0,
        "cost_savings_percentage": 0
    }

# For Vercel deployment, we don't use app.run()
# The app instance is automatically used by Vercel

# Caches will be created when users provide Redis URL in Settings
print("🚀 LangCache Demo ready - caches will be created when users provide Redis URL")

# For local development
if __name__ == '__main__':
    print("Starting Flask application locally...")
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
