document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');

            // Hide all tab contents
            tabContents.forEach(content => content.style.display = 'none');

            // Show the selected tab content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId + '-content').style.display = 'block';
        });
    });

    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-query');
    const cachedChat = document.getElementById('cached-chat');
    const standardChat = document.getElementById('standard-chat');
    const cachedTimeDisplay = document.getElementById('cached-time');
    const standardTimeDisplay = document.getElementById('standard-time');
    const llmModelSelect = document.getElementById('llm-model');
    // Fixed embedding model for simplified deployment
    const embeddingModel = 'openai-text-embedding-small';
    const similarityThresholdInput = document.getElementById('similarity-threshold');

    // Clear welcome messages when first query is submitted
    let isFirstQuery = true;

    submitButton.addEventListener('click', handleSubmit);
    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSubmit();
        }
    });

    function handleSubmit() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Always clear previous messages for each new query
        cachedChat.innerHTML = '';
        standardChat.innerHTML = '';
        if (isFirstQuery) {
            isFirstQuery = false;
        }

        // Add user message to both panels
        addMessage(cachedChat, query, 'user');
        addMessage(standardChat, query, 'user');

        // Add loading indicators
        const cachedLoadingMsg = addLoadingMessage(cachedChat);
        const standardLoadingMsg = addLoadingMessage(standardChat);

        // Reset time displays and styling
        cachedTimeDisplay.textContent = '';
        standardTimeDisplay.textContent = '';

        // Reset and hide performance indicators
        const cachedIndicator = cachedTimeDisplay.parentElement;
        const standardIndicator = standardTimeDisplay.parentElement;
        cachedIndicator.className = 'performance-indicator';
        standardIndicator.className = 'performance-indicator';

        // Hide indicators when empty (remove visible class)
        cachedIndicator.classList.remove('visible');
        standardIndicator.classList.remove('visible');

        // Start timers
        const cachedStartTime = performance.now();
        const standardStartTime = performance.now();

        // Get stored settings
        const settings = window.getStoredSettings ? window.getStoredSettings() : {};

        // Make requests to both endpoints

        // 1. Cached version
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                use_cache: true,
                llm_model: llmModelSelect.value,
                embedding_model: embeddingModel,
                similarity_threshold: parseFloat(similarityThresholdInput.value) || 0.85,
                user_openai_key: settings.openaiApiKey,
                user_redis_url: settings.redisUrl
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            cachedChat.removeChild(cachedLoadingMsg);

            // Use server-reported time for more accurate demonstration
            const timeTaken = data.time_taken.toFixed(2);

            // Display response time with enhanced styling
            const cachedIndicator = cachedTimeDisplay.parentElement;
            cachedIndicator.classList.add('visible'); // Show the indicator

            if (data.source === 'shadow_llm') {
                // Shadow mode - always show LLM response but indicate cache status
                cachedTimeDisplay.textContent = `🔍 ${timeTaken}s`;
                cachedIndicator.classList.add('shadow-mode');

                // Add response message with shadow mode indicator
                const shadowInfo = data.cache_hit ?
                    `Cache HIT recorded (Similarity: ${data.similarity ? parseFloat(data.similarity).toFixed(2) : 'N/A'})` :
                    'Cache MISS recorded';
                addResponseMessage(cachedChat, data.response, false, null, `Shadow Mode: ${shadowInfo}`);
            } else if (data.source === 'cache') {
                // Cache hit - show lightning bolt and keep green styling
                cachedTimeDisplay.textContent = `⚡ ${timeTaken}s`;
                cachedIndicator.classList.add('cache-hit');
                // Keep the green styling - no timeout to remove it
                addResponseMessage(cachedChat, data.response, data.source === 'cache', data.similarity);
            } else {
                // LLM response - simple gray display, no animation
                cachedTimeDisplay.textContent = `${timeTaken}s`;
                cachedIndicator.classList.add('llm-response');
                addResponseMessage(cachedChat, data.response, data.source === 'cache', data.similarity);
            }
        })
        .catch(error => {
            cachedChat.removeChild(cachedLoadingMsg);
            // Check if this is a settings error
            if (error.message && error.message.includes('required')) {
                addErrorMessage(cachedChat, '⚙️ Please configure your API settings in the Settings tab');
            } else {
                addErrorMessage(cachedChat, 'Error with semantic cache: ' + error.message);
            }
            console.error('Error with semantic cache:', error);
        });

        // 2. Standard LLM version
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                use_cache: false,
                llm_model: llmModelSelect.value,
                user_openai_key: settings.openaiApiKey,
                user_redis_url: settings.redisUrl
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            standardChat.removeChild(standardLoadingMsg);

            // Use server-reported time for more accurate demonstration
            const timeTaken = data.time_taken.toFixed(2);

            // Display response time with subtle gray LLM styling
            standardTimeDisplay.textContent = `${timeTaken}s`;

            // Show indicator and add subtle gray LLM styling (no animation)
            const standardIndicator = standardTimeDisplay.parentElement;
            standardIndicator.classList.add('visible');
            standardIndicator.classList.add('llm-response');

            // Add response message - always show as LLM-generated for the direct LLM panel
            // Even if the response is identical to the cached one, this panel always calls the LLM directly
            addResponseMessage(standardChat, data.response, false, null);
        })
        .catch(error => {
            standardChat.removeChild(standardLoadingMsg);
            // Check if this is a settings error
            if (error.message && error.message.includes('required')) {
                addErrorMessage(standardChat, '⚙️ Please configure your API settings in the Settings tab');
            } else {
                addErrorMessage(standardChat, 'Error with direct LLM: ' + error.message);
            }
            console.error('Error with direct LLM:', error);
        })
        .finally(() => {
            // Update latency and query analysis data after query completes
            updateDataAfterQuery();
        });

        // Clear input
        queryInput.value = '';
    }

    function addMessage(container, text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        messageDiv.textContent = text;
        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    function addResponseMessage(container, text, fromCache, similarity, shadowInfo) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'response');

        // Create a wrapper for the content
        const contentWrapper = document.createElement('div');

        // Add source indicator at the top
        const sourceIndicator = document.createElement('div');
        sourceIndicator.classList.add('source-indicator');

        if (shadowInfo) {
            // Shadow mode indicator
            sourceIndicator.classList.add('shadow');
            sourceIndicator.textContent = `🔍 ${shadowInfo}`;
        } else if (fromCache) {
            sourceIndicator.classList.add('cache');
            let cacheText = '✓ Redis LangCache';
            if (similarity) {
                // Format similarity as decimal (e.g., 0.99)
                const similarityDecimal = parseFloat(similarity).toFixed(2);
                cacheText += ` (Similarity: ${similarityDecimal})`;
            }
            sourceIndicator.textContent = cacheText;
        } else {
            sourceIndicator.textContent = 'Generated by LLM';
        }

        // Add the response text
        const responseText = document.createElement('div');
        responseText.textContent = text;

        // Add elements to the message div in the correct order
        messageDiv.appendChild(sourceIndicator);
        messageDiv.appendChild(responseText);

        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    function addLoadingMessage(container) {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'loading', 'response');

        // Add loading text
        const loadingText = document.createElement('span');
        loadingText.textContent = 'Generating response';
        loadingText.style.fontSize = '16px';
        loadingText.style.fontWeight = '500';
        loadingText.style.color = '#4A90E2';
        loadingText.style.marginRight = '12px';

        // Create prominent loading animation
        const loadingDots = document.createElement('div');
        loadingDots.classList.add('loading-dots');

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingDots.appendChild(dot);
        }

        loadingDiv.appendChild(loadingText);
        loadingDiv.appendChild(loadingDots);

        container.appendChild(loadingDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return loadingDiv;
    }

    function addErrorMessage(container, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', 'response', 'error');
        messageDiv.textContent = text;
        container.appendChild(messageDiv);
        // Scroll to the top like ChatGPT does
        container.scrollTop = 0;
        return messageDiv;
    }

    // Function to update latency and query analysis data after a query completes
    function updateDataAfterQuery() {
        // Update latency data if the latency tab exists and the function is available
        if (typeof fetchLatencyData === 'function' && document.getElementById('latency-content')) {
            fetchLatencyData();
        }

        // Update query analysis data if the query analysis tab exists and the function is available
        if (typeof fetchQueryMatches === 'function' && document.getElementById('query-analysis-content')) {
            fetchQueryMatches();
        }
    }

    // Function to clear previous messages while keeping welcome messages
    function clearPreviousMessages(container) {
        const messages = container.querySelectorAll('.message:not(.welcome-message)');
        messages.forEach(message => message.remove());
    }

    // Function to add shadow mode analysis message (not "generating response")
    function addShadowAnalysisMessage(container) {
        const analysisDiv = document.createElement('div');
        analysisDiv.classList.add('message', 'shadow-analysis', 'response');

        // Add analysis text
        const analysisText = document.createElement('span');
        analysisText.textContent = 'Analyzing cache performance...';
        analysisText.style.fontSize = '16px';
        analysisText.style.fontWeight = '500';
        analysisText.style.color = '#FF8C00';
        analysisText.style.marginRight = '12px';

        // Create loading animation
        const loadingDots = document.createElement('div');
        loadingDots.classList.add('loading-dots');

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingDots.appendChild(dot);
        }

        analysisDiv.appendChild(analysisText);
        analysisDiv.appendChild(loadingDots);

        container.appendChild(analysisDiv);
        container.scrollTop = 0;
        return analysisDiv;
    }

    // Shadow mode query handler
    async function handleShadowQuery() {
        console.log('handleShadowQuery called!');

        const queryInput = document.getElementById('shadow-query-input');
        const llmModelSelect = document.getElementById('shadow-llm-model');
        const similarityThresholdInput = document.getElementById('shadow-similarity-threshold');
        // Fixed embedding model for simplified deployment
        const embeddingModel = 'openai-text-embedding-small';

        console.log('Shadow mode form elements:', {
            queryInput: !!queryInput,
            llmModelSelect: !!llmModelSelect,
            similarityThresholdInput: !!similarityThresholdInput
        });

        if (!queryInput || !llmModelSelect || !similarityThresholdInput) {
            console.error('Shadow mode elements not found');
            return;
        }

        const query = queryInput.value.trim();
        console.log('Shadow mode query:', query);
        if (!query) {
            console.log('No query entered');
            return;
        }

        const llmModel = llmModelSelect.value;
        const similarityThreshold = parseFloat(similarityThresholdInput.value);

        const shadowCachedChat = document.getElementById('shadow-cached-chat');
        const shadowStandardChat = document.getElementById('shadow-standard-chat');
        const shadowCachedTimeDisplay = document.getElementById('shadow-cached-time');
        const shadowStandardTimeDisplay = document.getElementById('shadow-standard-time');

        // Reset and hide performance indicators
        const shadowCachedIndicator = shadowCachedTimeDisplay.parentElement;
        const shadowStandardIndicator = shadowStandardTimeDisplay.parentElement;
        shadowCachedIndicator.className = 'performance-indicator';
        shadowStandardIndicator.className = 'performance-indicator';

        // Hide indicators when empty (remove visible class)
        shadowCachedIndicator.classList.remove('visible');
        shadowStandardIndicator.classList.remove('visible');

        // Clear previous messages (keep welcome messages)
        clearPreviousMessages(shadowCachedChat);
        clearPreviousMessages(shadowStandardChat);

        // Add loading message only for the standard LLM panel
        // Shadow cache panel shows "Analyzing cache..." instead
        const shadowCachedLoading = addShadowAnalysisMessage(shadowCachedChat);
        const shadowStandardLoading = addLoadingMessage(shadowStandardChat);

        try {
            // Get stored settings for shadow mode
            const settings = window.getStoredSettings ? window.getStoredSettings() : {};

            // Make both requests simultaneously
            const [shadowCachedResponse, shadowStandardResponse] = await Promise.all([
                // Shadow mode request (always returns LLM but measures cache)
                fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        llm_model: llmModel,
                        embedding_model: embeddingModel,
                        similarity_threshold: similarityThreshold,
                        use_cache: true,
                        shadow_mode: true,  // Flag to indicate shadow mode
                        user_openai_key: settings.openaiApiKey,
                        user_redis_url: settings.redisUrl
                    })
                }),
                // Standard LLM request
                fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        llm_model: llmModel,
                        embedding_model: embeddingModel,
                        similarity_threshold: similarityThreshold,
                        use_cache: false,
                        user_openai_key: settings.openaiApiKey,
                        user_redis_url: settings.redisUrl
                    })
                })
            ]);

            // Remove loading messages
            shadowCachedLoading.remove();
            shadowStandardLoading.remove();

            // Handle shadow cached response
            if (shadowCachedResponse.ok) {
                const data = await shadowCachedResponse.json();

                // DO NOT show latency timing for shadow mode cache panel
                // Keep the indicator hidden since no response is shown to users
                const shadowCachedIndicator = shadowCachedTimeDisplay.parentElement;
                shadowCachedIndicator.classList.remove('visible'); // Keep it hidden

                // Show actual cache operation result
                let cacheResult = '';
                let cacheExplanation = '';

                if (data.cache_hit) {
                    const similarity = data.similarity ? parseFloat(data.similarity).toFixed(2) : 'N/A';
                    cacheResult = `Cache HIT`;
                    cacheExplanation = `Cache HIT recorded with ${similarity} similarity. Check your data in Shadow Analysis tab.`;
                } else {
                    cacheResult = `Cache MISS`;
                    cacheExplanation = `Cache MISS recorded. Check your data in Shadow Analysis tab.`;
                }

                // Create a status-only message showing actual cache operation
                const statusDiv = document.createElement('div');
                statusDiv.classList.add('message', 'shadow-status');
                statusDiv.innerHTML = `
                    <div class="source-indicator shadow">🔍 ${cacheResult}</div>
                    <div class="shadow-explanation">${cacheExplanation}</div>
                `;
                shadowCachedChat.appendChild(statusDiv);
                shadowCachedChat.scrollTop = 0;
            }

            // Handle standard response
            if (shadowStandardResponse.ok) {
                const data = await shadowStandardResponse.json();
                const timeTaken = data.time_taken.toFixed(2);

                // Show indicator and add subtle gray LLM styling (no animation)
                const shadowStandardIndicator = shadowStandardTimeDisplay.parentElement;
                shadowStandardIndicator.classList.add('visible');
                shadowStandardIndicator.classList.add('llm-response');
                shadowStandardTimeDisplay.textContent = `${timeTaken}s`;

                addResponseMessage(shadowStandardChat, data.response, false);
            }

        } catch (error) {
            console.error('Error:', error);
            shadowCachedLoading.remove();
            shadowStandardLoading.remove();
            addErrorMessage(shadowCachedChat, 'Error: Failed to get response');
            addErrorMessage(shadowStandardChat, 'Error: Failed to get response');
        }

        // Update data after query completes
        updateDataAfterQuery();
    }

    // Make handleShadowQuery globally available for pills-navigation.js
    window.handleShadowQuery = handleShadowQuery;
});
