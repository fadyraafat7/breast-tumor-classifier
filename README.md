# Breast Tumor Classifier

A local research prototype for exploring image classification. It includes an explicitly labeled synthetic demo and a trainable image-based baseline with local inference. **No dataset or trained model is included.** This project is not a diagnostic device and its scores are not medical advice.

The baseline is deliberately simple: logistic regression on a 16 × 16 grayscale image. It learns from labeled 2D exports; it does not process DICOM, 3D volumes, or complete MRI studies. Its suitability and performance depend on the dataset and must be established independently. Its scores are not calibrated clinical probabilities.

## Notebook-based IDC tissue classifier

The separate `ml/idc_train.py` workflow adapts [breast-cancer.ipynb](breast-cancer.ipynb) to train the notebook's ResNet-18 classifier on **histopathology image patches**, not MRI scans. Its classes are **0: no invasive ductal carcinoma (IDC)** and **1: IDC**. A negative IDC patch is not necessarily benign or cancer-free. The IDC model is not connected to the app's benign/malignant workflow, and cannot be used as an MRI model.

Create an isolated environment and install PyTorch and torchvision:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-idc.txt
```

The notebook's data and pretrained weights are not included. Obtain the dataset separately and arrange it as `dataset-root/<patient-id>/0/<image>.png` and `dataset-root/<patient-id>/1/<image>.png`, where `0` means no IDC and `1` means IDC. JPG and JPEG are also supported. Patient IDs must be anonymous and consistent across all patches from the same patient. Then create a patient-disjoint manifest:

```powershell
.\.venv\Scripts\python.exe ml/idc_data.py --dataset "C:\path\to\IDC_regular_ps50_idx5" --output data/idc-manifest.csv
```

The script assigns patients to approximately 70% train, 15% validation, and 15% test using a fixed seed (`--seed` changes it). It requires both labels in every split and rejects identical image bytes. If the split lacks a class, use a different seed or edit the CSV while keeping each patient in one split. The manifest stores image paths and patient identifiers; keep it private and outside version control. Byte comparisons cannot identify re-encoded duplicates, inconsistent patient IDs, or all related data. Review provenance and leakage before training.

Train the notebook-style model:

```powershell
.\.venv\Scripts\python.exe ml/idc_train.py train --manifest data/idc-manifest.csv --epochs 10
```

By default torchvision downloads official ImageNet ResNet-18 weights if they are not cached. For an already downloaded trusted ResNet-18 ImageNet state dictionary use `--pretrained-file "C:\path\to\resnet18-5c106cde.pth"`; for an offline smoke test use `--scratch` (this freezes a randomly initialized backbone, so it is not a meaningful substitute for pretrained training). Training uses 50 × 50 RGB patches, flips, class-weighted loss and a frozen backbone, and selects the best epoch by validation loss. It evaluates the test set once after model selection. It writes `artifacts/idc-model.pt` and `artifacts/idc-metrics.json`; files at those paths are overwritten. The metrics are per patch, not per patient. Avoid repeatedly tuning against the test set.

To run a single tissue patch through the trained checkpoint:

```powershell
.\.venv\Scripts\python.exe ml/idc_train.py predict --model artifacts/idc-model.pt --image "C:\path\to\patch.png"
```

The scores are uncalibrated research outputs, not a diagnosis. Only load model files from trusted sources. This workflow is an adaptation, not an exact reproduction of the notebook; its optimizer schedule and model selection differ. No medical performance is established until evaluated on a suitable independent dataset.

## Setup and run

Install Node.js 20+ and Python 3.10+, then install the Python dependency:

```sh
python -m pip install -r requirements.txt
npm start
```

Open **http://127.0.0.1:3000**. In PowerShell use `npm.cmd start` if the execution policy blocks `npm.ps1`. The synthetic demo is available immediately. To use a trained model, follow the training instructions below, then reload the page and choose **Trained local model** in the analysis mode selector.

Use the same Python environment for dependency installation, training, testing, and serving. By default the server calls `python` from your PATH; set `PYTHON` to the full path of the correct interpreter when necessary. For example, in PowerShell:

```powershell
$env:PYTHON = (Get-Command python).Source
npm.cmd start
```

The server listens on `127.0.0.1:3000` by default. Set `PORT` to change the port. Restart an already running server after updating its code or environment.

## Prepare a dataset

Supply an appropriately sourced, labeled and de-identified dataset of PNG, JPEG, or WebP images, up to 10 MB and 25 megapixels per image. Use consistent acquisition, image export, and labeling practices. Store it outside version control; the `data/` directory is ignored.

Create a CSV manifest with one image per row and these required columns:

```csv
path,label,patient_id,split
images/001.png,benign,P001,train
images/002.png,malignant,P002,train
images/003.png,benign,P003,validation
images/004.png,malignant,P004,validation
images/005.png,benign,P005,test
images/006.png,malignant,P006,test
```

Paths are relative to the manifest file (absolute paths also work). Labels must be `benign` or `malignant`; splits must be `train`, `validation`, or `test`. All images from the same patient must use the same nonempty anonymous `patient_id` and the same split. Each split must contain both labels. The example only shows the format: six images are not enough to establish performance.

The loader rejects patient IDs that cross splits and files with identical bytes anywhere in the manifest. It cannot recognize re-encoded or visually similar duplicates, inconsistent patient IDs, related patients, or other sources of leakage. Review dataset provenance, splits, and labels yourself. Keep the test set untouched during model development; repeatedly using it for decisions compromises its role as a held-out evaluation.

## Train and evaluate

```sh
python ml/pipeline.py train --manifest data/manifest.csv
```

This writes `artifacts/model.json` and `artifacts/metrics.json` (both ignored by Git). Training is deterministic for the same data, settings, and environment. It uses training images to fit the weights, selects the epoch with the lowest validation log loss, and evaluates that selected model on the test split. Metrics include image-level loss, accuracy, sensitivity, specificity, and a confusion matrix at a fixed 0.5 threshold. Images from the same patient are not statistically independent; image-level metrics are not patient-level performance estimates.

Optional arguments include `--epochs`, `--learning-rate`, `--seed`, `--model-out`, and `--metrics-out`. See `python ml/pipeline.py train --help`. Existing output files at those paths are overwritten. Record dataset version, labeling method, acquisition details, split strategy, software environment, and evaluation context alongside your experiment; these are not automatically captured by the model file.

For standalone inference:

```sh
python ml/pipeline.py inspect --model artifacts/model.json
python ml/pipeline.py predict --model artifacts/model.json --image data/images/example.png
```

The server uses `artifacts/model.json` by default. To select another model set `MODEL_PATH` before starting the server. The app checks the model when the page loads; reload after training or replacing it. A SHA-256 fingerprint in the report identifies the exact model file used. If a model changes after the page checks it, inference is rejected until reload.

## Use and privacy

1. Enter an anonymous study reference and choose a PNG, JPEG, or WebP image.
2. Choose **Synthetic demo** for fixed example values, or **Trained local model** for image-based inference when a compatible model is available.
3. Run the analysis, then export the JSON record or print the summary. A study reference is required for export or printing.

Demo analysis runs in the browser. Model analysis sends the image to the local Node server and its Python subprocess for inference; the app does not save the image. Refreshing clears the session. Exported records contain the reference, image filename/type/size, timestamps, and labeled output, but no image pixels. Downloaded records and printed summaries persist wherever you save them, so avoid identifying information in references and filenames. This is a local research tool without authentication; do not expose the server publicly or use it to handle identifiable patient data.

## Verify

```sh
npm test
```

On PowerShell, `npm.cmd test` also works. Tests cover classifier state, image validation, model output, dataset leakage checks, and a full training-to-local-HTTP-inference flow using generated images. Those fixtures do not establish medical performance.

`ml/pipeline.py` implements preprocessing, training, evaluation, and inference; `server.mjs` serves the app and the local model API; `classifier.js` manages analysis state and report generation; `app.js`, `index.html`, and `styles.css` implement the interface.
