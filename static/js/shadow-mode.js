// Shadow mode functionality - Auto-active mode
// No start/stop buttons needed - shadow mode is always active

// Chart.js instances
let peakTimesChart = null;
let queryPatternsChart = null;
let costAnalysisChart = null;

// Initialize charts
function initializeCharts() {
    // Peak Times Chart
    const peakTimesCtx = document.getElementById('peak-times-chart').getContext('2d');
    peakTimesChart = new Chart(peakTimesCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => `${i}:00`),
            datasets: [{
                label: 'Queries per Hour',
                data: Array(24).fill(0),
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Query Patterns Chart
    const queryPatternsCtx = document.getElementById('query-patterns-chart').getContext('2d');
    queryPatternsChart = new Chart(queryPatternsCtx, {
        type: 'pie',
        data: {
            labels: ['Short', 'Medium', 'Long'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(255, 99, 132, 0.5)'
                ]
            }]
        },
        options: {
            responsive: true
        }
    });

    // Cost Analysis Chart
    const costAnalysisCtx = document.getElementById('cost-analysis-chart').getContext('2d');
    costAnalysisChart = new Chart(costAnalysisCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Cost Savings',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Update shadow mode statistics
function updateShadowModeStats(data) {
    document.getElementById('total-queries').textContent = data.stats.total_queries;
    document.getElementById('cache-hits').textContent = data.stats.cache_hits;
    document.getElementById('hit-rate').textContent = `${data.stats.cache_hit_rate}%`;
    document.getElementById('avg-llm-time').textContent = `${data.stats.avg_llm_time}ms`;
    document.getElementById('avg-cache-time').textContent = `${data.stats.avg_cache_time}ms`;
    document.getElementById('total-savings').textContent = `${data.stats.total_savings}s`;
}

// Update shadow mode analysis
function updateShadowModeAnalysis(data) {
    // Update peak times chart
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    const queryCounts = hours.map(hour => data.peak_times[parseInt(hour)] || 0);
    peakTimesChart.data.datasets[0].data = queryCounts;
    peakTimesChart.update();

    // Update query patterns chart
    const patterns = data.query_patterns;
    queryPatternsChart.data.datasets[0].data = [
        patterns.short || 0,
        patterns.medium || 0,
        patterns.long || 0
    ];
    queryPatternsChart.update();

    // Update cost analysis chart
    const costData = data.cost_analysis;
    costAnalysisChart.data.labels.push(new Date().toLocaleTimeString());
    costAnalysisChart.data.datasets[0].data.push(costData.potential_savings);
    if (costAnalysisChart.data.labels.length > 10) {
        costAnalysisChart.data.labels.shift();
        costAnalysisChart.data.datasets[0].data.shift();
    }
    costAnalysisChart.update();
}

// Update shadow mode table
function updateShadowModeTable(queries) {
    const tbody = document.getElementById('shadow-matches-body');
    tbody.innerHTML = '';

    queries.forEach(query => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${query.timestamp}</td>
            <td>${query.query}</td>
            <td>${query.matched_query}</td>
            <td>${(query.similarity * 100).toFixed(2)}%</td>
            <td>${query.llm_time.toFixed(3)}s</td>
            <td>${query.cache_time.toFixed(3)}s</td>
            <td>${query.potential_savings.toFixed(3)}s</td>
        `;
        tbody.appendChild(row);
    });
}

// Shadow mode is always active - no manual start/stop needed

// Update shadow mode data periodically
let updateInterval = null;

function startShadowModeUpdates() {
    updateInterval = setInterval(async () => {
        try {
            // Get status
            const statusResponse = await fetch('/shadow-mode/status');
            const statusData = await statusResponse.json();
            
            if (statusData.is_active) {
                updateShadowModeStats(statusData);
                updateShadowModeTable(statusData.recent_queries);
            }

            // Get analysis
            const analysisResponse = await fetch('/shadow-mode/analysis');
            const analysisData = await analysisResponse.json();
            
            if (analysisData.is_active) {
                updateShadowModeAnalysis(analysisData);
            }
        } catch (error) {
            console.error('Error updating shadow mode data:', error);
        }
    }, 5000);  // Update every 5 seconds
}

function stopShadowModeUpdates() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

// Initialize on page load - Shadow mode is always active
document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts and start updates immediately since shadow mode is always active
    initializeCharts();
    startShadowModeUpdates();
});