document.addEventListener('DOMContentLoaded', function() {
    // Settings management
    const STORAGE_KEYS = {
        OPENAI_API_KEY: 'langcache_openai_key',
        REDIS_URL: 'langcache_redis_url'
    };

    // Load saved settings on page load
    loadSavedSettings();

    // Event listeners
    const saveBtn = document.getElementById('save-settings');
    const testBtn = document.getElementById('test-settings');
    const clearBtn = document.getElementById('clear-settings');

    if (saveBtn) saveBtn.addEventListener('click', saveSettings);
    if (testBtn) testBtn.addEventListener('click', testSettings);
    if (clearBtn) clearBtn.addEventListener('click', clearSettings);

    // Toggle visibility buttons
    document.querySelectorAll('.toggle-visibility').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const input = document.getElementById(targetId);

            if (input && input.type === 'password') {
                input.type = 'text';
                this.textContent = '🙈';
            } else if (input) {
                input.type = 'password';
                this.textContent = '👁️';
            }
        });
    });

    function loadSavedSettings() {
        try {
            const openaiKey = localStorage.getItem(STORAGE_KEYS.OPENAI_API_KEY);
            const redisUrl = localStorage.getItem(STORAGE_KEYS.REDIS_URL);

            const openaiInput = document.getElementById('openai-api-key');
            const redisInput = document.getElementById('redis-url');

            if (openaiKey && openaiInput) {
                openaiInput.value = openaiKey;
                showStatus('Settings loaded from browser storage', 'info');
            }

            if (redisUrl && redisInput) {
                redisInput.value = redisUrl;
            }
        } catch (error) {
            console.error('Error loading settings:', error);
            showStatus('Error loading saved settings', 'error');
        }
    }

    function saveSettings() {
        try {
            const openaiInput = document.getElementById('openai-api-key');
            const redisInput = document.getElementById('redis-url');

            if (!openaiInput || !redisInput) return;

            const openaiKey = openaiInput.value.trim();
            const redisUrl = redisInput.value.trim();

            // Validate inputs
            if (!openaiKey && !redisUrl) {
                showStatus('Please enter at least one setting to save', 'error');
                return;
            }

            if (openaiKey && !openaiKey.startsWith('sk-')) {
                showStatus('OpenAI API key should start with "sk-"', 'error');
                return;
            }

            if (redisUrl && !redisUrl.startsWith('redis://')) {
                showStatus('Redis URL should start with "redis://"', 'error');
                return;
            }

            // Save to localStorage
            if (openaiKey) {
                localStorage.setItem(STORAGE_KEYS.OPENAI_API_KEY, openaiKey);
            }

            if (redisUrl) {
                localStorage.setItem(STORAGE_KEYS.REDIS_URL, redisUrl);
            }

            showStatus('✅ Settings saved successfully! Initializing cache...', 'success');

            // Update global settings for immediate use
            window.userSettings = {
                openaiApiKey: openaiKey || null,
                redisUrl: redisUrl || null
            };

            // Initialize cache with Redis URL
            if (redisUrl) {
                initializeCache(redisUrl);
            }

        } catch (error) {
            console.error('Error saving settings:', error);
            showStatus('Error saving settings', 'error');
        }
    }

    async function testSettings() {
        const openaiInput = document.getElementById('openai-api-key');
        const redisInput = document.getElementById('redis-url');

        if (!openaiInput || !redisInput) return;

        const openaiKey = openaiInput.value.trim();
        const redisUrl = redisInput.value.trim();

        if (!openaiKey && !redisUrl) {
            showStatus('Please enter settings to test', 'error');
            return;
        }

        showStatus('🧪 Testing connections...', 'info');

        // For now, just validate format since we don't have test endpoints
        let testResults = [];

        if (openaiKey) {
            if (openaiKey.startsWith('sk-') && openaiKey.length > 20) {
                testResults.push('✅ OpenAI API key format is valid');
            } else {
                testResults.push('❌ OpenAI API key format is invalid');
            }
        }

        if (redisUrl) {
            if (redisUrl.startsWith('redis://') && redisUrl.includes('@')) {
                testResults.push('✅ Redis URL format is valid');
            } else {
                testResults.push('❌ Redis URL format is invalid');
            }
        }

        showStatus(testResults.join('<br>'), testResults.every(r => r.includes('✅')) ? 'success' : 'error');
    }

    function clearSettings() {
        if (confirm('Are you sure you want to clear all saved settings?')) {
            try {
                localStorage.removeItem(STORAGE_KEYS.OPENAI_API_KEY);
                localStorage.removeItem(STORAGE_KEYS.REDIS_URL);

                const openaiInput = document.getElementById('openai-api-key');
                const redisInput = document.getElementById('redis-url');

                if (openaiInput) openaiInput.value = '';
                if (redisInput) redisInput.value = '';

                // Clear global settings
                window.userSettings = {
                    openaiApiKey: null,
                    redisUrl: null
                };

                showStatus('🗑️ All settings cleared', 'info');
            } catch (error) {
                console.error('Error clearing settings:', error);
                showStatus('Error clearing settings', 'error');
            }
        }
    }

    function showStatus(message, type) {
        const statusDiv = document.getElementById('settings-status');
        if (!statusDiv) return;

        statusDiv.className = `settings-status ${type}`;
        statusDiv.innerHTML = message;
        statusDiv.style.display = 'block';

        // Auto-hide after 5 seconds for success/info messages
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }
    }

    function initializeCache(redisUrl) {
        fetch('/api/init-cache', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                redis_url: redisUrl
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showStatus('✅ Cache initialized successfully! Ready for queries.', 'success');
            } else {
                showStatus('⚠️ Cache initialization failed: ' + data.message, 'warning');
            }
        })
        .catch(error => {
            console.error('Cache initialization error:', error);
            showStatus('❌ Cache initialization error. Check console for details.', 'error');
        });
    }

    // Make settings available globally for other scripts
    window.getStoredSettings = function() {
        return {
            openaiApiKey: localStorage.getItem(STORAGE_KEYS.OPENAI_API_KEY),
            redisUrl: localStorage.getItem(STORAGE_KEYS.REDIS_URL)
        };
    };

    // Initialize global settings
    window.userSettings = window.getStoredSettings();
});
