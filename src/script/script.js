const API_URL = 'http://localhost:5000/api/chat';
const UPLOAD_URL = 'http://localhost:5000/api/upload';
const JOB_URL = 'http://localhost:5000/api/job/';

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

let isLoading = false;
let isUploading = false;
let isChatOpen = false;
let isMenuCollapsed = false;
let attachedFilePath = null;
let currentJobId = null;
let pollInterval = null;

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 140) + 'px';
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
            updateMessage(loadingMessage, 'Analysis started. Please wait...');
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
    }, 2000); // Poll every 2 seconds
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

messageInput.focus();