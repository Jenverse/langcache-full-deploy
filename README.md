# LangCache Full Demo

A comprehensive demonstration of LangCache semantic caching capabilities with live comparison, shadow mode analysis, and performance metrics.

## 🚀 Features

- **Live Mode**: Real-time comparison between semantic cache and direct LLM responses
- **Shadow Mode**: Background cache operation recording without affecting user experience
- **Shadow Analysis**: Detailed performance analysis with cache hit/miss statistics
- **Settings Management**: Secure in-browser API key and Redis URL configuration
- **Multiple Embedding Models**: Support for Redis LangCache and OpenAI embeddings
- **Real-time Metrics**: Latency tracking, token estimation, and cost analysis
- **No Server Secrets**: Users provide their own API keys via secure Settings UI

## Architecture

- **Flask Web Application**: Modern UI with pills navigation
- **Three Redis Langcache Instances**: Different embedding models
- **Custom Embeddings Service**: Ollama BGE-3 model
- **Redis Database**: Vector storage and search
- **Google Gemini API**: LLM integration

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Google Gemini API key

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd langcache-demo
```

2. Create a `.env` file:
```bash
# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Redis Configuration (optional, defaults provided)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Running with Docker Compose

1. Start all services:
```bash
docker-compose up -d
```

This will start:
- Redis database (port 6379)
- Redis Langcache-Embed service (port 8081)
- Ollama BGE-3 service (port 8080)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python3 app.py
```

4. Open your browser to `http://localhost:5001`

### Manual Service Setup

If you prefer to run services individually:

```bash
# Start Redis
docker run -d --name redis-cache -p 6379:6379 redis:latest

# Start Redis Langcache service
docker run -d --name langcache-redis -p 8081:8080 artifactory.dev.redislabs.com:443/cloud-docker-dev-local/ai-services/langcache:0.0.7

# Start Ollama BGE service
cd custom-embeddings
docker build -t ollama-bge .
docker run -d --name ollama-bge -p 8080:8080 ollama-bge
```

## Usage

### Live Mode
- **Demo Tab**: Interactive comparison between cached and direct LLM queries
- **Latency Tab**: Real-time performance metrics
- **Query Analysis**: Cache hit analysis and query matching
- **Operations Log**: Detailed operation tracking
- **Settings**: Configure similarity thresholds

### Shadow Mode
- **Zero-Risk Testing**: Run cache checks alongside production queries
- **Real-Time Metrics**: Track performance without affecting responses
- **Cost Analysis**: Estimate potential savings

### Shadow Mode Analysis
- **Query Patterns**: Analyze usage patterns and peak times
- **Performance Insights**: Detailed charts and statistics
- **Cost Breakdown**: ROI analysis for caching implementation

## API Endpoints

### Live Mode
- `GET /` - Main application interface
- `POST /query` - Process queries (with/without cache)
- `GET /latency-data` - Get performance metrics
- `GET /query-analysis` - Get query match analysis
- `GET /operations-log` - Get detailed operation logs

### Shadow Mode
- `POST /shadow-mode/start` - Start shadow mode tracking
- `POST /shadow-mode/stop` - Stop shadow mode tracking
- `GET /shadow-mode/status` - Get current status and stats
- `GET /shadow-mode/analysis` - Get detailed analysis data

## Configuration

### Similarity Threshold
Adjust the semantic similarity threshold (0.1-1.0) to control cache hit sensitivity:
- Higher values: More strict matching, fewer cache hits
- Lower values: More lenient matching, more cache hits

### Embedding Models
Switch between embedding models to compare performance:
- **Redis Langcache-Embed**: Optimized for speed
- **Ollama BGE-3**: High-quality embeddings

## Development

### Project Structure
```
├── app.py                 # Main Flask application
├── routes/
│   ├── live.py           # Live mode routes
│   └── shadow.py         # Shadow mode routes
├── utils/
│   └── helpers.py        # Shared utilities
├── templates/
│   └── index.html        # Main UI template
├── static/
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript files
├── custom-embeddings/    # Ollama BGE-3 service
└── docker-compose.yaml   # Service orchestration
```

### Adding New Features
1. Add routes to appropriate blueprint (`routes/live.py` or `routes/shadow.py`)
2. Add shared utilities to `utils/helpers.py`
3. Update frontend in `templates/index.html` and `static/js/`

## Troubleshooting

### Common Issues

1. **Port 5000 in use**: The app now runs on port 5001 by default
2. **Redis connection failed**: Ensure Redis is running on port 6379
3. **Langcache services not responding**: Check Docker containers are running
4. **API key errors**: Verify your Gemini API key in `.env` file

### Logs
Check application logs for detailed error information:
```bash
docker-compose logs -f
```

## Performance Tips

1. **Optimal Similarity Threshold**: Start with 0.85 and adjust based on your use case
2. **Cache Warm-up**: Run common queries to populate the cache
3. **Monitor Hit Rates**: Use the analytics to optimize your caching strategy

## License

This project is for demonstration purposes. Please check individual component licenses for production use.
