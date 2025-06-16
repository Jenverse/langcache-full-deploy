// Pills Navigation JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Handle main pill navigation
    const pillButtons = document.querySelectorAll('.pill-button');
    const pillContents = document.querySelectorAll('.pill-content');

    pillButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetPill = this.getAttribute('data-pill');

            // Remove active class from all pills and contents
            pillButtons.forEach(btn => btn.classList.remove('active'));
            pillContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked pill and corresponding content
            this.classList.add('active');
            document.getElementById(targetPill + '-content').classList.add('active');

            // If shadow mode is activated, attach event listeners
            if (targetPill === 'shadow-mode') {
                attachShadowModeEventListeners();
            }

            // If shadow analysis is activated, refresh the iframe
            if (targetPill === 'shadow-analysis') {
                refreshShadowAnalysis();
            }
        });
    });

    // Handle sub-tab navigation within Live Mode
    const subTabButtons = document.querySelectorAll('.sub-tab-button');
    const subTabContents = document.querySelectorAll('.sub-tab-content');

    subTabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetSubTab = this.getAttribute('data-subtab');
            
            // Remove active class from all sub-tabs and contents
            subTabButtons.forEach(btn => btn.classList.remove('active'));
            subTabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked sub-tab and corresponding content
            this.classList.add('active');
            document.getElementById(targetSubTab + '-content').classList.add('active');
        });
    });

    // Initialize with Live Mode active
    document.querySelector('[data-pill="live-mode"]').classList.add('active');
    document.getElementById('live-mode-content').classList.add('active');
    
    // Initialize with Demo sub-tab active
    document.querySelector('[data-subtab="demo"]').classList.add('active');
    document.getElementById('demo-content').classList.add('active');

    // Function to attach shadow mode event listeners when shadow mode tab is activated
    function attachShadowModeEventListeners() {
        console.log('Attaching shadow mode event listeners...');

        const shadowAskBtn = document.getElementById('shadow-ask-btn');
        const shadowQueryInput = document.getElementById('shadow-query-input');

        console.log('Shadow mode elements found:', {
            shadowAskBtn: !!shadowAskBtn,
            shadowQueryInput: !!shadowQueryInput
        });

        if (shadowAskBtn && !shadowAskBtn.hasAttribute('data-listener-attached')) {
            console.log('Adding click event listener to shadow ask button');
            shadowAskBtn.addEventListener('click', function() {
                console.log('Shadow ask button clicked!');
                if (typeof handleShadowQuery === 'function') {
                    handleShadowQuery();
                } else {
                    console.error('handleShadowQuery function not found');
                }
            });
            shadowAskBtn.setAttribute('data-listener-attached', 'true');
        }

        if (shadowQueryInput && !shadowQueryInput.hasAttribute('data-listener-attached')) {
            shadowQueryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    console.log('Enter pressed in shadow query input');
                    if (typeof handleShadowQuery === 'function') {
                        handleShadowQuery();
                    } else {
                        console.error('handleShadowQuery function not found');
                    }
                }
            });
            shadowQueryInput.setAttribute('data-listener-attached', 'true');
        }
    }

    // Function to refresh shadow analysis iframe
    function refreshShadowAnalysis() {
        console.log('Refreshing shadow analysis...');
        const iframe = document.querySelector('#shadow-analysis-content iframe');
        if (iframe) {
            // Force refresh by updating the src
            const currentSrc = iframe.src;
            iframe.src = '';
            setTimeout(() => {
                iframe.src = currentSrc + '?t=' + Date.now(); // Add timestamp to force refresh
            }, 100);
        }
    }

    // Make the functions globally available
    window.attachShadowModeEventListeners = attachShadowModeEventListeners;
    window.refreshShadowAnalysis = refreshShadowAnalysis;
});