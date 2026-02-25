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

async function startVerification() {
  if (AppState.uploadedFiles.length === 0) return;

  const verifyBtn = document.getElementById('verifyBtn');
  const errorBox  = document.getElementById('uploadError');

  if (verifyBtn) {
    verifyBtn.disabled = true;
    verifyBtn.textContent = '⏳ Анализ...';
  }
  if (errorBox) errorBox.style.display = 'none';

  const mode = localStorage.getItem('oncoai_mode') || 'doctor';
  const formData = new FormData();
  formData.append('file', AppState.uploadedFiles[0]);
  formData.append('mode', mode);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Ошибка сервера');
    }

    const data = await res.json();
    localStorage.setItem('oncoai_session_id',      data.session_id);
    localStorage.setItem('oncoai_response',        data.response);
    localStorage.setItem('oncoai_patient_summary', data.patient_summary || '');
    localStorage.setItem('oncoai_structured',      data.structured ? JSON.stringify(data.structured) : '');
    localStorage.setItem('oncoai_case_data', JSON.stringify({
      files: AppState.uploadedFiles.map(f => f.name),
      timestamp: new Date().toISOString()
    }));

    window.location.href = 'results.html';

  } catch (err) {
    if (verifyBtn) {
      verifyBtn.disabled = false;
      verifyBtn.textContent = '🔍 Проверить соответствие рекомендациям';
    }
    if (errorBox) {
      errorBox.textContent = '⚠️ ' + err.message;
      errorBox.style.display = 'block';
    } else {
      alert('Ошибка: ' + err.message);
    }
  }
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