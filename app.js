import BreastTumorClassifier from './classifier.js';

const classifier = new BreastTumorClassifier();
const $ = id => document.getElementById(id);
let scan = null;
let previewUrl = null;
let selectionVersion = 0;

function showError(message = '') {
    $('error').textContent = message;
    $('error').hidden = !message;
}

function resetResults() {
    classifier.classificationResults = null;
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
        $('result-status').textContent = 'Ready for demo';
        $('live-status').textContent = 'Image ready. You can now run the synthetic demo.';
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

$('analyze').addEventListener('click', () => {
    try {
        const result = classifier.runDemo(scan);
        showError();
        for (const label of ['benign', 'malignant']) {
            $(label + '-score').replaceChildren(document.createTextNode(String(Math.round(result[label] * 100))), Object.assign(document.createElement('span'), { textContent: '%' }));
            $(label + '-bar').value = result[label];
        }
        $('completed-at').textContent = new Date(result.createdAt).toLocaleString();
        $('empty-state').hidden = true;
        $('results').hidden = false;
        $('result-status').textContent = 'Demo complete';
        $('live-status').textContent = 'Synthetic demo complete. The example scores are not derived from your image.';
    } catch (error) { showError(error.message); }
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
        link.download = `demo-study-${report.studyReference.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        $('live-status').textContent = 'Demo study record exported.';
    } catch (error) { showError(error.message); }
});
$('print').addEventListener('click', () => {
    try { getReport(); window.print(); } catch (error) { showError(error.message); }
});
window.addEventListener('pagehide', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
