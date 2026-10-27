// Synthetic demo results are separate from the trained model adapter.
export class BreastTumorClassifier {
    constructor() {
        this.model = null;
        this.modelLoaded = false;
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
        await adapter.load?.();
        this.model = adapter;
        this.modelLoaded = true;
        return true;
    }

    processMRIScans(file) {
        this.classificationResults = null;
        if (!file || !['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
            throw new Error('Choose a PNG, JPEG, or WebP image. DICOM is not supported yet.');
        }
        if (!Number.isFinite(file.size) || file.size <= 0 || file.size > 10 * 1024 * 1024) {
            throw new Error('Choose a nonempty image smaller than or equal to 10 MB.');
        }
        // Model-specific decoding and preprocessing belong in the adapter.
        return { file, name: file.name, type: file.type, size: file.size };
    }

    async classifyTumor(scan) {
        this.classificationResults = null;
        if (!this.modelLoaded) throw new Error('No trained model is connected. Use the explicit demo workflow.');
        if (!scan?.file) throw new Error('Select and validate a scan first.');
        const output = await this.model.predict(scan);
        const { benign, malignant } = output ?? {};
        if (![benign, malignant].every(value => Number.isFinite(value) && value >= 0 && value <= 1) ||
            Math.abs(benign + malignant - 1) > 0.001) {
            throw new Error('The model must return benign and malignant probabilities that sum to 1.');
        }
        this.classificationResults = {
            mode: 'model', model: this.model.name, benign, malignant,
            createdAt: new Date().toISOString(),
        };
        return { ...this.classificationResults };
    }

    runDemo(scan) {
        this.classificationResults = null;
        if (!scan?.file) throw new Error('Select and validate a scan first.');
        this.classificationResults = {
            mode: 'demo', model: 'Synthetic demo fixture', benign: 0.64, malignant: 0.36,
            createdAt: new Date().toISOString(),
            notice: 'Fixed synthetic values for interface testing. Not derived from the uploaded image. Not for diagnosis.',
        };
        return { ...this.classificationResults };
    }

    generateReport(study, scan) {
        if (!this.classificationResults) throw new Error('Run an analysis before generating a report.');
        if (typeof study?.id !== 'string' || !study.id.trim()) throw new Error('Enter a study reference.');
        if (!scan?.file) throw new Error('Select and validate a scan first.');
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
