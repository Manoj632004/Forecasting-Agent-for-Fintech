const API_URL = 'http://localhost:5000/api/chat';
const UPLOAD_URL = 'http://localhost:5000/api/upload';
const JOB_URL = 'http://localhost:5000/api/job/';
const REGISTRY_URL = 'http://localhost:5000/api/registry';
const FORECAST_REPORT_URL = 'http://localhost:5000/api/forecast/';

const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const filePreviewName = document.getElementById('filePreviewName');
const removeFileBtn = document.getElementById('removeFileBtn');
const errorToast = document.getElementById('errorToast');
const chatPanel = document.getElementById('chatPanel');
const chatMenuBtn = document.getElementById('chatMenuBtn');
const menuBar = document.getElementById('menuBar');
const menuToggle = document.getElementById('menuToggle');
const forecastsBtn = document.getElementById('forecastsBtn');
const forecastList = document.getElementById('forecastList');
const reportFrame = document.getElementById('reportFrame');
const centerArea = document.getElementById('centerArea');
const dashboardBtn = document.getElementById('dashboardBtn');

let isLoading = false;
let isUploading = false;
let isChatOpen = false;
let isMenuCollapsed = false;
let isForecastListOpen = false;
let attachedFilePath = null;
let currentJobId = null;
let pollInterval = null;
let forecasts = [];

function openForecast(forecast) {
    const id = forecast.run_name;
    if (!id) {
        showError('This forecast has no ID.');
        return;
    }
    reportFrame.src = `${FORECAST_REPORT_URL}${encodeURIComponent(id)}/report`;
    centerArea.classList.add('has-report');
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + 'px';
}

async function loadForecasts() {
    try {
        const response = await fetch(REGISTRY_URL);
        if (!response.ok) {
            throw new Error(`Failed to load registry: ${response.status}`);
        }
        const data = await response.json();
        // Show each completed run once, newest first
        forecasts = [...new Map(data.map(f => [f.run_name, f])).values()].reverse();
        renderForecastList();
    } catch (error) {
        console.error('Error loading forecasts:', error);
        // Show empty list or error state
        forecasts = [];
        renderForecastList();
    }
}

function renderForecastList() {
    // Clear existing items
    forecastList.innerHTML = '';

    if (forecasts.length === 0) {
        const emptyItem = document.createElement('div');
        emptyItem.className = 'forecast-item';
        emptyItem.textContent = 'No forecasts available';
        emptyItem.style.color = '#9ca3af';
        emptyItem.style.cursor = 'default';
        forecastList.appendChild(emptyItem);
        return;
    }

    // Add each forecast as an item
    forecasts.forEach(forecast => {
        const forecastItem = document.createElement('div');
        forecastItem.className = 'forecast-item';
        forecastItem.dataset.forecastId = forecast.run_name || forecast.date; // Use run_name or date as ID
        forecastItem.dataset.targetCol = forecast.target_col;

        // Create the content
        const iconSpan = document.createElement('span');
        iconSpan.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3"></path></svg>';

        const textSpan = document.createElement('span');
        // textSpan.textContent = forecast.target_col || 'Unknown Target';
        textSpan.textContent = `${forecast.target_col || 'Unknown Target'} · ${forecast.run_name}`;

        // Add chevron to indicate this is a branch item
        const chevronSpan = document.createElement('span');
        chevronSpan.className = 'forecast-chevron';
        chevronSpan.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" style="width: 14px; height: 14px; margin-left: auto;">' +
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />' +
            '</svg>';

        forecastItem.appendChild(iconSpan);
        forecastItem.appendChild(textSpan);
        forecastItem.appendChild(chevronSpan);

        // Add click handler for future implementation
        forecastItem.addEventListener('click', () => {
            // For now, just highlight the selected item
            // document.querySelectorAll('.forecast-item').forEach(item => {
            //     item.classList.remove('active');
            // });
            document.querySelectorAll('.forecast-item').forEach(item => item.classList.remove('active'));
                forecastItem.classList.add('active');
                openForecast(forecast);
            forecastItem.classList.add('active');

            // TODO: Show forecast details in main panel when clicked
            console.log('Forecast selected:', forecast);
        });

        forecastList.appendChild(forecastItem);
    });
}

function showDashboard() {
    centerArea.classList.remove('has-report');   // hides the iframe, shows "Content area"
    reportFrame.src = 'about:blank';             // unloads the forecast report
    document.querySelectorAll('.forecast-item').forEach(item => item.classList.remove('active'));
}

function toggleForecastList() {
    isForecastListOpen = !isForecastListOpen;
    forecastList.classList.toggle('show', isForecastListOpen);

    // Also rotate the forecast button icon to indicate open/closed state
    const forecastIcon = forecastsBtn.querySelector('svg path');
    if (forecastIcon) {
        // We'll handle the rotation via CSS transform on the button itself
        forecastsBtn.classList.toggle('active', isForecastListOpen);
    }
}

messageInput.addEventListener('input', autoResizeTextarea);

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function showError(message) {
    errorToast.textContent = message;
    errorToast.classList.add('show');
    setTimeout(() => {
        errorToast.classList.remove('show');
    }, 5000);
}

function addMessage(content, type = 'assistant') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

function updateMessage(messageDiv, content) {
    messageDiv.textContent = content;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function toggleChat() {
    isChatOpen = !isChatOpen;
    chatPanel.classList.toggle('open', isChatOpen);
    chatMenuBtn.classList.toggle('active', isChatOpen);

    if (isChatOpen) {
        messageInput.focus();
    }
}

function toggleMenu() {
    isMenuCollapsed = !isMenuCollapsed;
    menuBar.classList.toggle('collapsed', isMenuCollapsed);
}

function showFilePreview(fileName) {
    filePreviewName.textContent = fileName;
    filePreview.style.display = 'flex';
}

function hideFilePreview() {
    filePreview.style.display = 'none';
    filePreviewName.textContent = '';
    attachedFilePath = null;
    fileInput.value = '';
}

async function handleFileAttach(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview immediately
    showFilePreview(file.name);
    isUploading = true;

    // Upload to backend to get path
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(UPLOAD_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }

        const data = await response.json();
        attachedFilePath = data.file_path;
        isUploading = false;
        console.log('File uploaded, path stored:', attachedFilePath);
    } catch (error) {
        console.error('File upload error:', error);
        showError('Failed to upload file');
        hideFilePreview();
        isUploading = false;
    }
}

async function sendMessage() {
    if (!attachedFilePath) {
        showError('Please attach a file first.');
        return;
    }
    if (isLoading) return;
    if (isUploading) {
        showError('Please wait for file upload to complete...');
        return;
    }

    const message = messageInput.value.trim();

    isLoading = true;
    sendBtn.disabled = true;
    attachBtn.disabled = true;
    messageInput.disabled = true;

    // Prepare message content
    let displayMessage = message;
    if (attachedFilePath) {
        displayMessage = message ? `${message} (📎 ${filePreviewName.textContent})` : `📎 ${filePreviewName.textContent}`;
    }

    addMessage(displayMessage, 'user');
    messageInput.value = '';
    autoResizeTextarea();

    // Capture file path BEFORE clearing the preview
    const currentFilePath = attachedFilePath;
    hideFilePreview();

    const loadingMessage = addMessage('Starting analysis...', 'loading assistant');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                file_path: currentFilePath
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.status === 'started' && data.job_id) {
            currentJobId = data.job_id;
            updateMessage(loadingMessage, 'Waiting for the forecasting pipeline...');
            pollJobStatus(data.job_id, loadingMessage);
        } else {
            updateMessage(loadingMessage, data.response || 'Analysis complete');
        }
    } catch (error) {
        console.error('Error:', error);
        updateMessage(loadingMessage, 'Sorry, I encountered an error. Please try again.');
        showError('Failed to connect to the server. Is the Python backend running?');
        resetLoadingState();
    }
}

function pollJobStatus(jobId, loadingMessage) {
    // Clear any existing poll
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${JOB_URL}${jobId}`);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                updateMessage(loadingMessage, data.response);
                resetLoadingState();
                currentJobId = null;
                // Refresh forecast list to show newly created forecast
                loadForecasts();
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                pollInterval = null;
                updateMessage(loadingMessage, `Analysis failed: ${data.error || 'Unknown error'}`);
                showError('Analysis failed. Please try again.');
                resetLoadingState();
                currentJobId = null;
            } else if (data.status === 'running') {
                // Update the loading message with progress
                updateMessage(loadingMessage, data.response || 'Still processing...');
            }
        } catch (error) {
            console.error('Polling error:', error);
            clearInterval(pollInterval);
            pollInterval = null;
            updateMessage(loadingMessage, 'Error checking status. Please try again.');
            showError('Failed to check analysis status.');
            resetLoadingState();
            currentJobId = null;
        }
    }, 1000);
}

function resetLoadingState() {
    isLoading = false;
    sendBtn.disabled = false;
    attachBtn.disabled = false;
    messageInput.disabled = false;
    attachedFilePath = null;
    messageInput.focus();
}

// Close chat when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 1024 && isChatOpen) {
        const chatPanelRect = chatPanel.getBoundingClientRect();
        const menuBtnRect = chatMenuBtn.getBoundingClientRect();
        const isClickInChatPanel = e.clientX >= chatPanelRect.left && e.clientX <= chatPanelRect.right && e.clientY >= chatPanelRect.top && e.clientY <= chatPanelRect.bottom;
        const isClickInMenuBtn = e.clientX >= menuBtnRect.left && e.clientX <= menuBtnRect.right && e.clientY >= menuBtnRect.top && e.clientY <= menuBtnRect.bottom;
        if (!isClickInChatPanel && !isClickInMenuBtn) {
            toggleChat();
        }
    }
});

// Event listeners
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileAttach);
removeFileBtn.addEventListener('click', hideFilePreview);
forecastsBtn.addEventListener('click', toggleForecastList);
dashboardBtn.addEventListener('click', showDashboard);

// Load forecasts on page load
loadForecasts();

// Poll registry every 10 seconds to auto-update forecast list
// setInterval(loadForecasts, 10000);

messageInput.focus();
