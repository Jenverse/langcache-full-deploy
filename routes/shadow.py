from flask import Blueprint, jsonify, request
import datetime
import statistics
from utils.helpers import shadow_mode_data, get_current_timestamp

shadow_bp = Blueprint('shadow', __name__)

# Shadow mode is always active - no start/stop endpoints needed

@shadow_bp.route('/shadow-mode/status', methods=['GET'])
def get_shadow_mode_status():
    """Get current shadow mode status and basic stats - Always active"""
    global shadow_mode_data

    # Calculate stats
    total_queries = shadow_mode_data['stats']['total_queries']
    cache_hits = shadow_mode_data['stats']['cache_hits']
    cache_hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0

    avg_llm_time = (shadow_mode_data['stats']['total_llm_time'] / total_queries * 1000) if total_queries > 0 else 0  # Convert to ms
    avg_cache_time = (shadow_mode_data['stats']['total_cache_time'] / cache_hits * 1000) if cache_hits > 0 else 0  # Convert to ms

    return jsonify({
        'is_active': True,  # Always active
        'start_time': shadow_mode_data['start_time'].isoformat() if shadow_mode_data['start_time'] else None,
        'duration_minutes': (datetime.datetime.now() - shadow_mode_data['start_time']).total_seconds() / 60 if shadow_mode_data['start_time'] else 0,
        'stats': {
            'total_queries': total_queries,
            'cache_hits': cache_hits,
            'cache_hit_rate': round(cache_hit_rate, 1),
            'avg_llm_time': round(avg_llm_time, 2),
            'avg_cache_time': round(avg_cache_time, 2),
            'total_savings': round(shadow_mode_data['stats']['total_savings'], 2)
        }
    })

@shadow_bp.route('/shadow-mode/analysis', methods=['GET'])
def get_shadow_mode_analysis():
    """Get detailed shadow mode analysis"""
    global shadow_mode_data
    
    if not shadow_mode_data['queries']:
        return jsonify({
            'queries': [],
            'patterns': {
                'peak_hours': [],
                'common_queries': [],
                'cache_performance': {
                    'hit_rate_by_hour': [],
                    'avg_savings_per_hit': 0
                }
            },
            'cost_analysis': {
                'estimated_llm_cost': 0,
                'estimated_cache_cost': 0,
                'total_savings': 0
            }
        })
    
    # Analyze query patterns
    queries_by_hour = {}
    cache_hits_by_hour = {}
    
    for query in shadow_mode_data['queries']:
        hour = query['timestamp'][:13]  # YYYY-MM-DD HH
        queries_by_hour[hour] = queries_by_hour.get(hour, 0) + 1
        if query.get('cache_hit'):
            cache_hits_by_hour[hour] = cache_hits_by_hour.get(hour, 0) + 1
    
    # Find peak hours
    peak_hours = sorted(queries_by_hour.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate hit rate by hour
    hit_rate_by_hour = []
    for hour, total in queries_by_hour.items():
        hits = cache_hits_by_hour.get(hour, 0)
        hit_rate = (hits / total * 100) if total > 0 else 0
        hit_rate_by_hour.append({
            'hour': hour,
            'hit_rate': round(hit_rate, 1),
            'total_queries': total
        })
    
    # Cost analysis (simplified estimates)
    total_queries = len(shadow_mode_data['queries'])
    cache_hits = sum(1 for q in shadow_mode_data['queries'] if q.get('cache_hit'))
    
    # Rough cost estimates (per 1K tokens)
    llm_cost_per_1k = 0.0015  # Example cost for Gemini
    cache_cost_per_1k = 0.0001  # Much lower cost for cache
    
    estimated_llm_cost = total_queries * llm_cost_per_1k
    estimated_cache_cost = cache_hits * cache_cost_per_1k
    total_savings = estimated_llm_cost - estimated_cache_cost
    
    return jsonify({
        'queries': shadow_mode_data['queries'][-50:],  # Last 50 queries
        'patterns': {
            'peak_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
            'cache_performance': {
                'hit_rate_by_hour': hit_rate_by_hour,
                'avg_savings_per_hit': round(shadow_mode_data['stats']['total_savings'] / cache_hits, 2) if cache_hits > 0 else 0
            }
        },
        'cost_analysis': {
            'estimated_llm_cost': round(estimated_llm_cost, 4),
            'estimated_cache_cost': round(estimated_cache_cost, 4),
            'total_savings': round(total_savings, 4)
        }
    }) 