import test from 'node:test';
import assert from 'node:assert/strict';
import BreastTumorClassifier from '../classifier.js';

const image = (name = 'scan.png') => new File(['image fixture'], name, { type: 'image/png' });

test('real inference requires an explicitly initialized model', async () => {
    const classifier = new BreastTumorClassifier();
    assert.equal(classifier.modelLoaded, false);
    await assert.rejects(classifier.initializeModel(), /adapter/);
    await assert.rejects(classifier.classifyTumor({ file: image() }), /No trained model/);
    assert.throws(() => classifier.generateReport({ id: 'S1' }), /Run an analysis/);
});

test('uploads reject unsupported, empty, oversized and missing files', () => {
    const classifier = new BreastTumorClassifier();
    for (const invalid of [null, { type: 'application/dicom', size: 20 }, { type: 'image/png', size: 0 },
        { type: 'image/png', size: 10 * 1024 * 1024 + 1 }, { type: 'image/png', size: NaN }]) {
        assert.throws(() => classifier.processMRIScans(invalid));
    }
    for (const type of ['image/png', 'image/jpeg', 'image/webp']) {
        assert.equal(classifier.processMRIScans({ name: 'scan', type, size: 10 * 1024 * 1024 }).type, type);
    }
});

test('demo exports retain the synthetic notice and exclude image pixels', () => {
    const classifier = new BreastTumorClassifier();
    const scan = classifier.processMRIScans(image());
    const result = classifier.runDemo(scan);
    assert.equal(result.mode, 'demo');
    assert.equal(classifier.modelLoaded, false);
    assert.match(result.notice, /Not derived from the uploaded image/);
    assert.throws(() => classifier.generateReport({ id: '  ' }, scan), /study reference/);
    const report = classifier.generateReport({ id: ' S-001 ' }, scan);
    assert.equal(report.studyReference, 'S-001');
    assert.equal(report.classification.mode, 'demo');
    assert.equal('file' in report.scan, false);
    assert.equal('confidence' in report.classification, false);
    result.benign = 0;
    report.classification.malignant = 1;
    assert.equal(classifier.classificationResults.benign, 0.64);
    assert.equal(classifier.classificationResults.malignant, 0.36);
});

test('selecting another file clears prior results, including on validation failure', () => {
    const classifier = new BreastTumorClassifier();
    classifier.runDemo(classifier.processMRIScans(image()));
    classifier.processMRIScans(image('another.png'));
    assert.equal(classifier.classificationResults, null);
    classifier.runDemo(classifier.processMRIScans(image()));
    assert.throws(() => classifier.processMRIScans(null));
    assert.equal(classifier.classificationResults, null);
});

test('adapter initialization and inference are awaited and tagged with model identity', async () => {
    const classifier = new BreastTumorClassifier();
    let loaded = false;
    const scan = classifier.processMRIScans(image());
    await classifier.initializeModel({
        name: 'test-model-v1',
        async load() { loaded = true; },
        async predict(input) {
            assert.equal(loaded, true);
            assert.equal(input, scan);
            return { benign: 0.7, malignant: 0.3 };
        },
    });
    const result = await classifier.classifyTumor(scan);
    assert.equal(result.mode, 'model');
    assert.equal(result.model, 'test-model-v1');
    assert.equal(result.benign, 0.7);
});

test('invalid model probabilities cannot produce a report', async () => {
    for (const output of [undefined, { benign: NaN, malignant: 0.5 }, { benign: -0.1, malignant: 1.1 },
        { benign: 0.7, malignant: 0.7 }, { benign: '0.4', malignant: 0.6 }]) {
        const classifier = new BreastTumorClassifier();
        const scan = classifier.processMRIScans(image());
        classifier.runDemo(scan);
        await classifier.initializeModel({ name: 'bad-output', predict: async () => output });
        await assert.rejects(classifier.classifyTumor(scan), /probabilities/);
        assert.equal(classifier.classificationResults, null);
    }
});

test('a failed replacement model leaves inference disabled and clears earlier results', async () => {
    const classifier = new BreastTumorClassifier();
    const scan = classifier.processMRIScans(image());
    await classifier.initializeModel({ name: 'old', predict: async () => ({ benign: 1, malignant: 0 }) });
    await classifier.classifyTumor(scan);
    await assert.rejects(classifier.initializeModel({
        name: 'unavailable', load: async () => { throw new Error('weights unavailable'); }, predict() {},
    }), /weights unavailable/);
    assert.equal(classifier.modelLoaded, false);
    assert.equal(classifier.classificationResults, null);
    await assert.rejects(classifier.classifyTumor(scan), /No trained model/);
});
