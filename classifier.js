// Synthetic demo results are separate from the trained model adapter.
export class BreastTumorClassifier {
    constructor() {
        this.model = null;
        this.modelLoaded = false;
        this.classificationResults = null;
        this.currentScan = null;
        this.resultScan = null;
        this.stateVersion = 0;
        this.modelVersion = 0;
    }

    clearScan() {
        this.clearResults();
        this.currentScan = null;
    }

    clearResults() {
        this.stateVersion++;
        this.resultScan = null;
        this.classificationResults = null;
    }

    async initializeModel(adapter) {
        if (!adapter || typeof adapter.predict !== 'function' ||
            typeof adapter.name !== 'string' || !adapter.name.trim()) {
            throw new Error('Provide a named model adapter with a predict(scan) function.');
        }
        this.model = null;
        this.modelLoaded = false;
        this.classificationResults = null;
        this.resultScan = null;
        this.stateVersion++;
        const modelVersion = ++this.modelVersion;
        await adapter.load?.();
        if (modelVersion !== this.modelVersion) throw new Error('Model initialization was superseded.');
        this.model = adapter;
        this.modelLoaded = true;
        return true;
    }

    processMRIScans(file) {
        this.clearScan();
        if (!file || !['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
            throw new Error('Choose a PNG, JPEG, or WebP image. DICOM is not supported yet.');
        }
        if (!Number.isFinite(file.size) || file.size <= 0 || file.size > 10 * 1024 * 1024) {
            throw new Error('Choose a nonempty image smaller than or equal to 10 MB.');
        }
        // Model-specific decoding and preprocessing belong in the adapter.
        this.currentScan = { file, name: file.name, type: file.type, size: file.size };
        return this.currentScan;
    }

    async classifyTumor(scan) {
        this.classificationResults = null;
        this.resultScan = null;
        const stateVersion = ++this.stateVersion;
        if (!this.modelLoaded) throw new Error('No trained model is connected. Use the explicit demo workflow.');
        if (!scan?.file || scan !== this.currentScan) throw new Error('Select and validate a scan first.');
        const model = this.model;
        const output = await model.predict(scan);
        if (stateVersion !== this.stateVersion || model !== this.model || scan !== this.currentScan) {
            throw new Error('Analysis was superseded by a newer action.');
        }
        const { benign, malignant } = output ?? {};
        if (![benign, malignant].every(value => Number.isFinite(value) && value >= 0 && value <= 1) ||
            Math.abs(benign + malignant - 1) > 0.001) {
            throw new Error('The model must return benign and malignant probabilities that sum to 1.');
        }
        this.classificationResults = {
            mode: 'model', model: model.name, benign, malignant,
            createdAt: new Date().toISOString(),
            notice: 'Research model output. Scores are uncalibrated and are not for diagnosis.',
        };
        this.resultScan = scan;
        return { ...this.classificationResults };
    }

    runDemo(scan) {
        this.classificationResults = null;
        this.resultScan = null;
        this.stateVersion++;
        if (!scan?.file || scan !== this.currentScan) throw new Error('Select and validate a scan first.');
        this.classificationResults = {
            mode: 'demo', model: 'Synthetic demo fixture', benign: 0.64, malignant: 0.36,
            createdAt: new Date().toISOString(),
            notice: 'Fixed synthetic values for interface testing. Not derived from the uploaded image. Not for diagnosis.',
        };
        this.resultScan = scan;
        return { ...this.classificationResults };
    }

    generateReport(study, scan) {
        if (!this.classificationResults) throw new Error('Run an analysis before generating a report.');
        if (typeof study?.id !== 'string' || !study.id.trim()) throw new Error('Enter a study reference.');
        if (!scan?.file || scan !== this.currentScan || scan !== this.resultScan) {
            throw new Error('The report must use the scan that was analyzed.');
        }
        return {
            schemaVersion: 1,
            studyReference: study.id.trim(),
            exportedAt: new Date().toISOString(),
            scan: { name: scan.name, type: scan.type, size: scan.size },
            classification: { ...this.classificationResults },
            intendedUse: 'Research prototype. Not a medical report or diagnostic device.',
        };
    }
}

export default BreastTumorClassifier;
