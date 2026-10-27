import BreastTumorClassifier from './classifier.js';

const classifier = new BreastTumorClassifier();
const $ = id => document.getElementById(id);
let scan = null;
let previewUrl = null;
let selectionVersion = 0;
let runVersion = 0;
let modelInfo = null;

function updateMode() {
    const isModel = $('analysis-mode').value === 'model';
    $('analyze').replaceChildren(document.createTextNode(isModel ? 'Run model analysis ' : 'Run demo analysis '), Object.assign(document.createElement('span'), { textContent: '→' }));
    $('result-status').textContent = scan ? (isModel ? 'Ready for model' : 'Ready for demo') : 'Awaiting image';
}

async function findModel() {
    try {
        const response = await fetch('/api/model');
        if (!response.ok) throw new Error('Model check failed');
        const info = await response.json();
        if (!info.available || typeof info.name !== 'string' || !/^[a-f0-9]{64}$/.test(info.model_id)) throw new Error('Model unavailable');
        modelInfo = info;
        const option = $('analysis-mode').querySelector('option[value="model"]');
        option.disabled = false;
        option.textContent = 'Trained local model';
        $('model-status-title').textContent = 'Local model available';
        $('model-status').textContent = `${info.name} (${info.model_id.slice(0, 12)})`;
    } catch {
        $('model-status').textContent = 'No compatible trained model found';
    }
}
findModel();

function showError(message = '') {
    $('error').textContent = message;
    $('error').hidden = !message;
}

function resetResults() {
    runVersion++;
    classifier.clearScan();
    $('results').hidden = true;
    $('empty-state').hidden = false;
    $('result-status').textContent = scan ? 'Ready for demo' : 'Awaiting image';
}

function removeScan() {
    selectionVersion++;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    scan = null;
    $('scan-preview').removeAttribute('src');
    $('scan-file').value = '';
    $('preview-wrap').hidden = true;
    $('dropzone').hidden = false;
    $('analyze').disabled = true;
    resetResults();
    showError();
}

async function selectFile(file) {
    if (!file) return;
    removeScan();
    const version = selectionVersion;
    let candidateUrl;
    try {
        const candidate = classifier.processMRIScans(file);
        candidateUrl = URL.createObjectURL(file);
        const image = new Image();
        image.src = candidateUrl;
        await image.decode();
        if (version !== selectionVersion) return;
        scan = candidate;
        previewUrl = candidateUrl;
        $('scan-preview').src = previewUrl;
        $('file-name').textContent = file.name;
        $('file-size').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${image.naturalWidth} × ${image.naturalHeight} px`;
        $('preview-wrap').hidden = false;
        $('dropzone').hidden = true;
        $('analyze').disabled = false;
        updateMode();
        $('live-status').textContent = 'Image ready for analysis.';
    } catch (error) {
        if (version === selectionVersion) showError(error.name === 'EncodingError' ? 'This file could not be decoded as an image. Choose a valid PNG, JPEG, or WebP.' : error.message);
    } finally {
        if (candidateUrl && candidateUrl !== previewUrl) URL.revokeObjectURL(candidateUrl);
    }
}

$('scan-file').addEventListener('change', event => selectFile(event.target.files[0]));
$('remove-file').addEventListener('click', () => { removeScan(); $('scan-file').focus(); });
for (const type of ['dragenter', 'dragover']) {
    $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
}
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragging'));
$('dropzone').addEventListener('drop', event => {
    event.preventDefault();
    $('dropzone').classList.remove('dragging');
    if (event.dataTransfer.files.length !== 1) {
        showError('Choose one image at a time.');
        return;
    }
    selectFile(event.dataTransfer.files[0]);
});
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());

$('analysis-mode').addEventListener('change', () => {
    runVersion++;
    classifier.clearResults();
    $('results').hidden = true;
    $('empty-state').hidden = false;
    $('analyze').disabled = !scan;
    showError();
    updateMode();
});

$('analyze').addEventListener('click', async () => {
    const selectedScan = scan;
    const mode = $('analysis-mode').value;
    const version = ++runVersion;
    classifier.clearResults();
    $('results').hidden = true;
    $('empty-state').hidden = false;
    showError();
    $('analyze').disabled = true;
    $('result-status').textContent = 'Analyzing';
    try {
        let result;
        if (mode === 'demo') {
            result = classifier.runDemo(selectedScan);
        } else {
            if (!modelInfo) throw new Error('No compatible local model is available.');
            const info = modelInfo;
            await classifier.initializeModel({
                name: `${info.name} (SHA-256 ${info.model_id})`,
                async predict(input) {
                    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': input.type }, body: input.file });
                    const output = await response.json();
                    if (!response.ok) throw new Error(output.error || 'Model prediction failed.');
                    if (output.model_id !== info.model_id) throw new Error('The model changed. Reload the page to use the new model.');
                    return output;
                },
            });
            if (version !== runVersion || scan !== selectedScan) return;
            result = await classifier.classifyTumor(selectedScan);
        }
        if (version !== runVersion || scan !== selectedScan) return;
        showError();
        const demo = result.mode === 'demo';
        $('result-kind').textContent = demo ? 'SYNTHETIC SAMPLE' : 'RESEARCH MODEL OUTPUT';
        $('result-heading').textContent = demo ? 'Demo analysis complete' : 'Model analysis complete';
        $('result-description').textContent = demo ? 'These values are a fixed example. They do not describe the uploaded image.' : 'Image-based model scores are uncalibrated and are not for diagnosis.';
        document.querySelectorAll('.score-caption').forEach(element => { element.textContent = demo ? 'Example score' : 'Model score'; });
        $('result-source').textContent = result.model;
        $('result-type').textContent = demo ? 'Interface demonstration' : 'Local image inference';
        for (const label of ['benign', 'malignant']) {
            $(label + '-score').replaceChildren(document.createTextNode(String(Math.round(result[label] * 100))), Object.assign(document.createElement('span'), { textContent: '%' }));
            $(label + '-bar').value = result[label];
        }
        $('completed-at').textContent = new Date(result.createdAt).toLocaleString();
        $('empty-state').hidden = true;
        $('results').hidden = false;
        $('result-status').textContent = demo ? 'Demo complete' : 'Model complete';
        $('live-status').textContent = demo ? 'Synthetic demo complete. The example scores are not derived from your image.' : 'Research model analysis complete.';
    } catch (error) {
        if (version === runVersion) { showError(error.message); updateMode(); }
    } finally {
        if (version === runVersion) $('analyze').disabled = !scan;
    }
});

function getReport() {
    showError();
    if (!$('study-id').value.trim()) {
        $('study-id').focus();
        throw new Error('Enter an anonymous study reference before exporting or printing.');
    }
    return classifier.generateReport({ id: $('study-id').value }, scan);
}

$('export').addEventListener('click', () => {
    try {
        const report = getReport();
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a');
        link.href = url;
        link.download = `${report.classification.mode}-study-${report.studyReference.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        $('live-status').textContent = 'Demo study record exported.';
    } catch (error) { showError(error.message); }
});
$('print').addEventListener('click', () => {
    try { getReport(); window.print(); } catch (error) { showError(error.message); }
});
window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
