document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const modelFilter = document.getElementById('model-filter');
    const queryMatchesBody = document.getElementById('query-matches-body');
    const noMatchesMessage = document.getElementById('no-matches-message');

    // Add event listener for model filter
    modelFilter.addEventListener('change', function() {
        fetchQueryMatches();
    });

    // Function to fetch query matches
    function fetchQueryMatches() {
        const selectedModel = modelFilter.value;
        const url = selectedModel === 'all' ? '/query-analysis' : `/query-analysis?model=${selectedModel}`;

        fetch(url)
            .then(response => response.json())
            .then(data => {
                updateQueryMatchesTable(data);
            })
            .catch(error => {
                console.error('Error fetching query matches:', error);
            });
    }

    // Function to update the query matches table
    function updateQueryMatchesTable(data) {
        // Clear the table
        queryMatchesBody.innerHTML = '';

        // Check if there are any matches
        if (data.matches.length === 0) {
            noMatchesMessage.style.display = 'block';
            return;
        }

        // Hide the no matches message
        noMatchesMessage.style.display = 'none';

        // Add rows for each match
        data.matches.forEach(match => {
            const row = document.createElement('tr');

            // Format the similarity as a percentage
            const similarityFormatted = typeof match.similarity === 'number'
                ? (match.similarity * 100).toFixed(2) + '%'
                : match.similarity;

            // Format the embedding time
            const embeddingTimeFormatted = typeof match.embedding_time === 'number'
                ? match.embedding_time.toFixed(3) + 's'
                : match.embedding_time;

            // Create the row content
            row.innerHTML = `
                <td>${match.timestamp}</td>
                <td>${match.query}</td>
                <td>${match.matched_query}</td>
                <td>${getModelLabel(match.model)}</td>
                <td>${similarityFormatted}</td>
                <td>${embeddingTimeFormatted}</td>
            `;

            queryMatchesBody.appendChild(row);
        });
    }

    // Helper function to truncate text
    function truncateText(text, maxLength) {
        if (!text) return 'N/A';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    }

    // Helper function to format model names
    function getModelLabel(model) {
        switch (model) {
            case 'redis-langcache':
                return 'Redis Langcache-Embed';
            case 'ollama-bge':
                return 'Ollama BGE-3';
            default:
                return model;
        }
    }

    // Fetch query matches when the tab is shown
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            if (this.getAttribute('data-tab') === 'query-analysis') {
                fetchQueryMatches();
            }
        });
    });

    // No auto-refresh - data will be updated when a new query is submitted
});
