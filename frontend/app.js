const form = document.getElementById('inference-form');
const input = document.getElementById('image-input');
const preview = document.getElementById('preview-image');
const resultCard = document.getElementById('result-card');
const topLabel = document.getElementById('top-label');
const topConfidence = document.getElementById('top-confidence');
const topKList = document.getElementById('topk-list');
const errorMessage = document.getElementById('error-message');

const API_URL = 'http://localhost:8000/predict';

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = input.files[0];
  if (!file) {
    return showError('Please choose an image file first.');
  }

  resetUI();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || 'Prediction request failed.');
    }

    const data = await response.json();
    showResult(data);
    showPreview(file);
  } catch (error) {
    showError(error.message);
  }
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function showResult(data) {
  topLabel.textContent = `Disease: ${data.top_prediction.label}`;
  topConfidence.textContent = `Confidence: ${(data.top_prediction.confidence * 100).toFixed(2)}%`;
  topKList.innerHTML = data.predictions
    .map((item) => `<li>${item.label} — ${(item.confidence * 100).toFixed(2)}%</li>`)
    .join('');
  resultCard.classList.remove('hidden');
}

function showError(message) {
  errorMessage.textContent = message;
}

function resetUI() {
  errorMessage.textContent = '';
  resultCard.classList.add('hidden');
  topKList.innerHTML = '';
}
