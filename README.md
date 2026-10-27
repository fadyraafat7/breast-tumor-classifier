# Breast Tumor Classifier

A local graduation-project prototype for uploading an exported scan image, previewing a **synthetic demo** classification, and exporting a JSON study record or printing a summary.

**No trained model is included.** Demo scores are fixed at 64% benign and 36% malignant, regardless of the image. They are interface fixtures, not predictions. This project is not a diagnostic device or a medical report generator.

## Run locally

Install Node.js 20 or newer, then run from this directory:

```sh
npm start
```

Open **http://127.0.0.1:3000**. No dependency installation or build step is needed. On Windows PowerShell, use `npm.cmd start` if execution policy blocks `npm.ps1`.

To change the port in PowerShell:

```powershell
$env:PORT = '3001'
npm.cmd start
```

## Use the prototype

1. Enter an anonymous study reference.
2. Choose or drop one PNG, JPEG, or WebP image, up to 10 MB.
3. Select **Run demo analysis** to view the labeled synthetic values.
4. Export a JSON record or print the summary. A study reference is required for either action.

Images are decoded locally in the browser and are not uploaded or stored. Refreshing clears the session. Exported JSON contains the reference, image filename/type/size, timestamps, and labeled demo scores, but no image pixels. Downloaded records and printed summaries persist wherever you save them; avoid including identifying information in references or filenames.

DICOM, 3D volumes, multi-slice processing, actual MRI preprocessing, and clinical interpretation are not implemented.

## Structure

- `index.html`, `styles.css`, `app.js`: responsive browser workspace.
- `classifier.js`: upload validation, explicit demo fixture, report data, and model adapter contract.
- `server.mjs`: dependency-free local static server, bound to loopback by default. It has no upload or inference endpoint.
- `test/`: classifier behavior and HTTP checks using Node's built-in test runner.

## Connect a trained model

`BreastTumorClassifier.initializeModel(adapter)` accepts an adapter with a name, optional asynchronous `load()`, and asynchronous `predict(scan)`. The adapter owns decoding and all preprocessing required by its specific model; `processMRIScans()` only validates image metadata. `scan.file` is the original browser `File`.

```js
import BreastTumorClassifier from './classifier.js';

const classifier = new BreastTumorClassifier();
await classifier.initializeModel({
  name: 'Your model and version',
  async load() { /* Load your model runtime and weights here. */ },
  async predict(scan) {
    // Decode scan.file, apply training-time preprocessing, and run inference.
    // Return { benign, malignant }: finite values in [0, 1] summing to 1.
    throw new Error('Implement model inference before using this adapter.');
  },
});
```

The browser UI intentionally calls `runDemo()`. To integrate real inference, update its analysis handler to await `classifyTumor(scan)` and update the fixed demo labels and source metadata to reflect the actual model. Do not represent scores as calibrated confidence without separate validation. A real implementation also needs documented dataset provenance, preprocessing, evaluation, and intended-use review.

## Checks

```sh
npm test
```

On PowerShell, `npm.cmd test` also works.
