// app.js - Общая логика для всех страниц

// State management
const AppState = {
  mode: 'doctor',
  uploadedFiles: [],
  currentCase: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  AppState.mode = localStorage.getItem('oncoai_mode') || 'doctor';
  console.log('Режим:', AppState.mode);
});

// File upload handling
function handleFileUpload(files) {
  Array.from(files).forEach(file => {
    if (!AppState.uploadedFiles.find(f => f.name === file.name)) {
      AppState.uploadedFiles.push(file);
    }
  });
  renderFileList();
}

function renderFileList() {
  const fileList = document.getElementById('fileList');
  const verifyBtn = document.getElementById('verifyBtn');
  
  if (!fileList) return;
  
  fileList.innerHTML = AppState.uploadedFiles.map((file, index) => `
    <div class="file-item">
      <div class="file-info">
        <div class="file-icon">📄</div>
        <div>
          <div class="file-name">${file.name}</div>
          <div class="file-size">${(file.size / 1024).toFixed(1)} КБ</div>
        </div>
      </div>
      <button class="file-remove" onclick="removeFile(${index})">✕</button>
    </div>
  `).join('');
  
  if (verifyBtn) {
    verifyBtn.disabled = AppState.uploadedFiles.length === 0;
  }
}

function removeFile(index) {
  AppState.uploadedFiles.splice(index, 1);
  renderFileList();
}

function startVerification() {
  if (AppState.uploadedFiles.length === 0) return;
  
  const verifyBtn = document.getElementById('verifyBtn');
  if (verifyBtn) {
    verifyBtn.disabled = true;
    verifyBtn.textContent = '⏳ Анализ...';
  }
  
  // Simulate API call
  setTimeout(() => {
    localStorage.setItem('oncoai_case_data', JSON.stringify({
      patientId: 'ONC-2026-0847',
      files: AppState.uploadedFiles.map(f => f.name),
      timestamp: new Date().toISOString()
    }));
    
    window.location.href = 'results.html';
  }, 2000);
}

// Setup drag and drop
function setupDragAndDrop() {
  const uploadSection = document.getElementById('uploadSection');
  const fileInput = document.getElementById('fileInput');
  
  if (!uploadSection || !fileInput) return;
  
  uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
  });
  
  uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
  });
  
  uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    handleFileUpload(e.dataTransfer.files);
  });
  
  fileInput.addEventListener('change', (e) => {
    handleFileUpload(e.target.files);
  });
}

// Initialize drag and drop on upload page
if (window.location.pathname.includes('upload.html')) {
  setupDragAndDrop();
}

// Export function
function exportReport() {
  console.log('Экспорт отчёта...');
  // В полной версии здесь будет генерация PDF
}