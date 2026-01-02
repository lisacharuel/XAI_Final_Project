// Global state
let currentFileType = null;
let currentModel = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupUploadArea();
    console.log('🚀 Unified XAI Interface loaded');
});

// Setup upload area
function setupUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });
}

// Upload file
async function uploadFile(file) {
    showToast('Uploading file...', 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentFileType = data.file_type;
            displayFileInfo(data);
            showModelSelection(data.compatible_models);
            showXAIButtons(data.compatible_xai);
            
            // If image, display it
            if (data.file_type === 'image') {
                displayUploadedImage();
            }
            
            showToast('File uploaded successfully!', 'success');
            
            // Hide welcome section
            document.getElementById('welcome-section').style.display = 'none';
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed: ' + error.message, 'error');
    }
}

// Display file info
function displayFileInfo(data) {
    const fileInfo = document.getElementById('file-info');
    const info = data.info;
    
    let html = '<h3>✅ File Processed</h3>';
    html += `<p><strong>Type:</strong> ${data.file_type}</p>`;
    
    if (data.file_type === 'audio') {
        html += `<p><strong>Duration:</strong> ${info.duration}s</p>`;
        html += `<p><strong>Sample Rate:</strong> ${info.sample_rate} Hz</p>`;
    } else {
        html += `<p><strong>Original Size:</strong> ${info.width}×${info.height}</p>`;
    }
    
    html += `<p><strong>Processed Shape:</strong> ${info.shape}</p>`;
    
    fileInfo.innerHTML = html;
    fileInfo.style.display = 'block';
}

// Display uploaded image
function displayUploadedImage() {
    const container = document.getElementById('uploaded-image-container');
    const img = document.getElementById('uploaded-image');
    
    img.src = '/api/uploaded-image?' + new Date().getTime();
    container.style.display = 'block';
}

// Show model selection
function showModelSelection(models) {
    const section = document.getElementById('model-section');
    const buttonsContainer = document.getElementById('model-buttons');
    
    let html = '';
    models.forEach(model => {
        const icon = currentFileType === 'audio' ? '🎵' : '🏥';
        html += `
            <button class="btn btn-primary" onclick="selectModel('${model.key}')">
                ${icon} ${model.name}
            </button>
        `;
    });
    
    buttonsContainer.innerHTML = html;
    section.style.display = 'block';
}

// Show XAI buttons
function showXAIButtons(methods) {
    const buttonsContainer = document.getElementById('xai-buttons');
    
    let html = '';
    methods.forEach(method => {
        html += `
            <button class="btn btn-primary" onclick="generateExplanation('${method.key}')" style="display: none;" id="xai-${method.key}">
                🔍 Explain with ${method.name}
            </button>
        `;
    });
    
    buttonsContainer.innerHTML = html;
}

// Select model
async function selectModel(modelKey) {
    currentModel = modelKey;
    showToast('Loading model...', 'info');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelKey })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayPrediction(data);
            showToast('Prediction complete!', 'success');
            
            // Show XAI section
            document.getElementById('xai-section').style.display = 'block';
            
            // Enable XAI buttons
            document.querySelectorAll('[id^="xai-"]').forEach(btn => {
                btn.style.display = 'inline-flex';
            });
        } else {
            showToast(data.error || 'Prediction failed', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Prediction failed: ' + error.message, 'error');
    }
}

// Display prediction
function displayPrediction(data) {
    const section = document.getElementById('prediction-section');
    const content = document.getElementById('prediction-content');
    
    const confidencePercent = (data.confidence * 100).toFixed(1);
    
    let html = `
        <div class="prediction-box">
            <div class="prediction-label">Prediction</div>
            <div class="prediction-value">${data.prediction}</div>
            <span class="confidence-badge">${confidencePercent}% Confidence</span>
        </div>
        
        <div class="probabilities">
            <h3>All Probabilities</h3>
    `;
    
    for (const [className, prob] of Object.entries(data.probabilities)) {
        const percent = (prob * 100).toFixed(1);
        html += `
            <div class="prob-item">
                <div class="prob-label">
                    <span>${className}</span>
                    <span><strong>${percent}%</strong></span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width: ${percent}%">
                        ${percent}%
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    
    content.innerHTML = html;
    section.style.display = 'block';
    
    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Generate explanation
async function generateExplanation(method) {
    const loadingDiv = document.getElementById('explanation-loading');
    loadingDiv.style.display = 'block';
    
    showToast('Generating explanation... Please wait 30-60 seconds', 'info');
    
    try {
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ method: method })
        });
        
        const data = await response.json();
        
        loadingDiv.style.display = 'none';
        
        if (response.ok) {
            displayExplanation(data);
            showToast('Explanation generated!', 'success');
        } else {
            showToast(data.error || 'Explanation failed', 'error');
        }
    } catch (error) {
        loadingDiv.style.display = 'none';
        console.error('Explanation error:', error);
        showToast('Explanation failed: ' + error.message, 'error');
    }
}

// Display explanation
function displayExplanation(data) {
    const section = document.getElementById('explanation-section');
    const content = document.getElementById('explanation-content');
    
    let html = `
        <p><strong>Method:</strong> ${data.method.toUpperCase()}</p>
        <p><strong>Features Analyzed:</strong> ${data.num_features}</p>
        <p style="margin-top: 16px;">The visualization below shows which regions of your input were most important for the prediction.</p>
        <img src="${data.explanation_url}?${new Date().getTime()}" class="explanation-image" alt="LIME Explanation">
    `;
    
    content.innerHTML = html;
    section.style.display = 'block';
    
    // Scroll to explanation
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Reset session
async function resetSession() {
    if (!confirm('Are you sure you want to reset and start over?')) {
        return;
    }
    
    try {
        await fetch('/api/reset', { method: 'POST' });
        location.reload();
    } catch (error) {
        console.error('Reset error:', error);
        location.reload();
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}