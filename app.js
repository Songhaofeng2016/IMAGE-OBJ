const imageInput = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const imagePreview = document.querySelector('#image-preview');
const modelSelect = document.querySelector('#model-select');
const generateButton = document.querySelector('#generate-button');
const errorMessage = document.querySelector('#error-message');
const modeOptions = [...document.querySelectorAll('.mode-option')];
const canvas = document.querySelector('#mesh-canvas');
const context = canvas.getContext('2d');
let selectedImage = null;
let mesh = null;
let rotation = { x: -0.38, y: 0.62 };
let dragOrigin = null;
let downloadedUrl = null;
let reconstructionMode = 'scene';

function setError(message = '') {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function updateGenerateState() {
  generateButton.disabled = !selectedImage || !modelSelect.value;
}

async function loadModels() {
  const connectionLabel = document.querySelector('#connection-label');
  const connectionDot = document.querySelector('#connection-dot');
  const hint = document.querySelector('#model-hint');
  modelSelect.innerHTML = '<option value="">正在读取已安装模型…</option>';
  modelSelect.disabled = true;
  try {
    const response = await fetch('/api/models');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '读取本地模型失败。');
    connectionLabel.textContent = 'Ollama 已连接';
    connectionDot.className = 'connection-dot';
    modelSelect.innerHTML = '';
    if (!data.models.length) {
      modelSelect.innerHTML = '<option value="">未发现已安装模型</option>';
      hint.textContent = '请先在 Ollama 中安装支持图像输入的模型。';
    } else {
      const placeholder = document.createElement('option');
      placeholder.value = '';
      const visionModels = data.models.filter((model) => model.vision);
      placeholder.textContent = visionModels.length ? '选择一个本地视觉模型' : '没有可用的视觉模型';
      modelSelect.append(placeholder);
      data.models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = model.vision ? model.name : `${model.name}（不支持图片）`;
        option.disabled = !model.vision;
        modelSelect.append(option);
      });
      hint.textContent = visionModels.length
        ? '仅显示支持图像输入的模型。'
        : '已安装模型均不支持图像输入，请先在 Ollama 中准备视觉模型。';
    }
    modelSelect.disabled = false;
  } catch (error) {
    modelSelect.innerHTML = '<option value="">Ollama 未连接</option>';
    connectionLabel.textContent = 'Ollama 未连接';
    connectionDot.className = 'connection-dot offline';
    hint.textContent = error.message;
  }
  updateGenerateState();
}

function showImage(file) {
  if (!file || !file.type.startsWith('image/')) return setError('请选择一张图片文件。');
  if (file.size > 10 * 1024 * 1024) return setError('图片不能超过 10 MB。');
  const reader = new FileReader();
  reader.onload = () => {
    selectedImage = { dataUrl: reader.result, mime: file.type };
    imagePreview.src = reader.result;
    imagePreview.hidden = false;
    dropzone.classList.add('has-image');
    document.querySelector('#corner-image').src = reader.result;
    document.querySelector('#image-corner').hidden = false;
    setError();
    updateGenerateState();
  };
  reader.onerror = () => setError('无法读取这张图片。');
  reader.readAsDataURL(file);
}

imageInput.addEventListener('change', () => showImage(imageInput.files[0]));
dropzone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropzone.classList.add('dragging');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('dragging');
  showImage(event.dataTransfer.files[0]);
});
document.querySelector('#clear-image').addEventListener('click', (event) => {
  event.preventDefault();
  event.stopPropagation();
  selectedImage = null;
  imageInput.value = '';
  imagePreview.hidden = true;
  imagePreview.removeAttribute('src');
  dropzone.classList.remove('has-image');
  document.querySelector('#image-corner').hidden = true;
  updateGenerateState();
});
modelSelect.addEventListener('change', updateGenerateState);
document.querySelector('#refresh-models').addEventListener('click', loadModels);
modeOptions.forEach((option) => {
  option.addEventListener('click', () => {
    reconstructionMode = option.dataset.mode;
    modeOptions.forEach((item) => {
      item.setAttribute('aria-pressed', String(item === option));
    });
    document.querySelector('#mode-hint').textContent = reconstructionMode === 'scene'
      ? '补齐房间结构、家具，以及被遮挡或画外的部分。'
      : '聚焦一个主体，补齐被遮挡或裁切的部分，忽略背景。';
  });
});

function parseObj(text) {
  const vertices = [];
  const faces = [];
  for (const line of text.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v') vertices.push(parts.slice(1, 4).map(Number));
    if (parts[0] === 'f') faces.push(parts.slice(1).map((part) => Number(part.split('/')[0]) - 1));
  }
  return { vertices, faces };
}

function drawMesh() {
  if (!mesh || canvas.hidden) return;
  const bounds = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * dpr));
  const height = Math.max(1, Math.round(bounds.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.clearRect(0, 0, width, height);
  const sinX = Math.sin(rotation.x), cosX = Math.cos(rotation.x);
  const sinY = Math.sin(rotation.y), cosY = Math.cos(rotation.y);
  const scale = Math.min(width, height) * 0.34;
  const projected = mesh.vertices.map(([x, y, z]) => {
    const rx = x * cosY - y * sinY;
    const rotatedY = x * sinY + y * cosY;
    const rz = rotatedY * sinX + z * cosX;
    const depth = rotatedY * cosX - z * sinX;
    const perspective = 2.8 / (2.8 + depth * 0.26);
    return { x: width / 2 + rx * scale * perspective, y: height * 0.48 - rz * scale * perspective, depth };
  });
  const orderedFaces = mesh.faces.map((face) => ({ face, depth: face.reduce((sum, index) => sum + projected[index].depth, 0) / face.length })).sort((a, b) => a.depth - b.depth);
  for (const { face, depth } of orderedFaces) {
    context.beginPath();
    face.forEach((index, position) => {
      const point = projected[index];
      if (position === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.closePath();
    context.fillStyle = depth > 0 ? 'rgba(91, 119, 84, 0.10)' : 'rgba(237, 104, 72, 0.055)';
    context.fill();
    context.strokeStyle = depth > 0 ? 'rgba(51, 75, 55, 0.66)' : 'rgba(75, 92, 74, 0.48)';
    context.lineWidth = Math.max(0.7, dpr * 0.8);
    context.stroke();
  }
}

canvas.addEventListener('pointerdown', (event) => {
  dragOrigin = { x: event.clientX, y: event.clientY, rotation: { ...rotation } };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', (event) => {
  if (!dragOrigin) return;
  rotation = { x: dragOrigin.rotation.x + (event.clientY - dragOrigin.y) * 0.008, y: dragOrigin.rotation.y + (event.clientX - dragOrigin.x) * 0.008 };
  drawMesh();
});
canvas.addEventListener('pointerup', () => { dragOrigin = null; });
canvas.addEventListener('pointercancel', () => { dragOrigin = null; });
document.querySelector('#reset-view').addEventListener('click', () => {
  rotation = { x: -0.38, y: 0.62 };
  drawMesh();
});
window.addEventListener('resize', drawMesh);

function setBusy(busy) {
  generateButton.disabled = busy || !selectedImage || !modelSelect.value;
  modeOptions.forEach((option) => { option.disabled = busy; });
  generateButton.querySelector('.button-label').textContent = busy ? '正在生成网格…' : '生成 3D 网格';
  document.querySelector('#progress-wrap').hidden = !busy;
}

generateButton.addEventListener('click', async () => {
  if (!selectedImage || !modelSelect.value) return;
  setError();
  setBusy(true);
  document.querySelector('#preview-state').textContent = '正在构建形体';
  document.querySelector('#result-bar').hidden = true;
  document.querySelector('#progress-label').textContent = '正在准备本地模型…';
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelSelect.value, mode: reconstructionMode, mime: selectedImage.mime, image: selectedImage.dataUrl.split(',')[1] }),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || '生成失败，请重试。');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === 'progress') {
          document.querySelector('#progress-label').textContent = event.message || (
            event.phase === 'loading' ? '正在加载视觉模型…' :
            event.phase === 'generating' ? `正在生成网格，已接收 ${event.characters.toLocaleString()} 个字符…` :
            '正在校验并整理网格…'
          );
        } else if (event.type === 'error') {
          throw new Error(event.error || '生成失败，请重试。');
        } else if (event.type === 'complete') {
          result = event;
        }
      }
      if (done) break;
    }
    if (!result) throw new Error('生成流意外结束，请重试。');
    mesh = parseObj(result.obj);
    rotation = { x: -0.38, y: 0.62 };
    document.querySelector('#empty-state').hidden = true;
    canvas.hidden = false;
    document.querySelector('#mesh-badge').hidden = false;
    document.querySelector('#canvas-controls').hidden = false;
    document.querySelector('#mesh-count').textContent = `${result.vertices} VERT · ${result.faces} FACE`;
    document.querySelector('#preview-state').textContent = '网格已生成';
    document.querySelector('#result-description').textContent = result.description;
    document.querySelector('#result-bar').hidden = false;
    if (downloadedUrl) URL.revokeObjectURL(downloadedUrl);
    downloadedUrl = URL.createObjectURL(new Blob([result.obj], { type: 'model/obj' }));
    const download = document.querySelector('#download-button');
    download.href = downloadedUrl;
    download.download = `${modelSelect.value.replace(/[^\w.-]+/g, '-')}-model.obj`;
    drawMesh();
  } catch (error) {
    document.querySelector('#preview-state').textContent = '生成未完成';
    setError(error.message);
  } finally {
    setBusy(false);
  }
});

loadModels();
