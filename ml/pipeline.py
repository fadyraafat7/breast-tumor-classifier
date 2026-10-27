"""Train and run a small, transparent image-classification research baseline.

Inputs are exported 2D images, not DICOM or complete MRI studies.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

SIDE = 16
FEATURES = SIDE * SIDE
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MODEL_KIND = "grayscale-logistic-regression-v1"
SPLITS = ("train", "validation", "test")


def features_from_bytes(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be nonempty and at most 10 MB")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError("Only PNG, JPEG, and WebP images are supported")
        if image.width * image.height > 25_000_000:
            raise ValueError("Image dimensions exceed the 25 megapixel limit")
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        return [pixel / 255.0 for pixel in image.tobytes()]


def load_manifest(manifest):
    manifest = Path(manifest).resolve()
    records = {split: [] for split in SPLITS}
    patient_splits, image_splits, seen_files = {}, {}, set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label", "patient_id", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest requires path,label,patient_id,split columns")
        for line, row in enumerate(reader, 2):
            patient = (row.get("patient_id") or "").strip()
            split = (row.get("split") or "").strip().lower()
            label = (row.get("label") or "").strip().lower()
            filename = (row.get("path") or "").strip()
            if not patient or not filename or split not in SPLITS or label not in ("benign", "malignant"):
                raise ValueError(f"Invalid patient_id, path, label, or split on row {line}")
            previous = patient_splits.setdefault(patient, split)
            if previous != split:
                raise ValueError(f"Patient appears in multiple splits (row {line})")
            image_path = (manifest.parent / filename).resolve()
            if image_path in seen_files:
                raise ValueError(f"Image path is repeated (row {line})")
            seen_files.add(image_path)
            if image_path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError(f"Image exceeds 10 MB (row {line})")
            data = image_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in image_splits:
                raise ValueError(f"Identical image bytes appear more than once (row {line})")
            image_splits[digest] = split
            try:
                vector = features_from_bytes(data)
            except Exception as exc:
                raise ValueError(f"Invalid image on row {line}: {exc}") from exc
            records[split].append((vector, int(label == "malignant")))
    for split, examples in records.items():
        if {label for _, label in examples} != {0, 1}:
            raise ValueError(f"The {split} split must contain both benign and malignant examples")
    return records


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def predict_probability(weights, bias, vector):
    return sigmoid(bias + sum(weight * feature for weight, feature in zip(weights, vector)))


def evaluate(examples, weights, bias):
    tp = tn = fp = fn = 0
    loss = 0.0
    for vector, label in examples:
        probability = predict_probability(weights, bias, vector)
        clipped = max(1e-15, min(1 - 1e-15, probability))
        loss -= label * math.log(clipped) + (1 - label) * math.log(1 - clipped)
        prediction = int(probability >= 0.5)
        tp += int(prediction == 1 and label == 1)
        tn += int(prediction == 0 and label == 0)
        fp += int(prediction == 1 and label == 0)
        fn += int(prediction == 0 and label == 1)
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"images": len(examples), "threshold": 0.5, "log_loss": loss / len(examples),
            "accuracy": (tp + tn) / len(examples), "sensitivity": sensitivity, "specificity": specificity,
            "confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}}


def train(records, epochs, learning_rate, seed):
    rng = random.Random(seed)
    weights = [rng.uniform(-0.001, 0.001) for _ in range(FEATURES)]
    bias = 0.0
    best_loss, best_weights, best_bias, best_epoch = math.inf, None, 0.0, 0
    training = records["train"]
    for epoch in range(1, epochs + 1):
        gradient, bias_gradient = [0.0] * FEATURES, 0.0
        for vector, label in training:
            error = predict_probability(weights, bias, vector) - label
            bias_gradient += error
            for i, feature in enumerate(vector):
                gradient[i] += error * feature
        count = len(training)
        weights = [weight - learning_rate * (grad / count + 0.001 * weight)
                   for weight, grad in zip(weights, gradient)]
        bias -= learning_rate * bias_gradient / count
        validation_loss = evaluate(records["validation"], weights, bias)["log_loss"]
        if validation_loss < best_loss:
            best_loss, best_weights, best_bias, best_epoch = validation_loss, weights[:], bias, epoch
    model = {"format": MODEL_KIND, "name": "16x16 grayscale logistic regression baseline",
             "created_at": datetime.now(timezone.utc).isoformat(), "image_size": SIDE,
             "weights": best_weights, "bias": best_bias,
             "notice": "Research baseline only. Uncalibrated scores; not for clinical use."}
    metrics = {"model_format": MODEL_KIND, "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
               "learning_rate": learning_rate, "selection": "minimum validation log loss",
               "unit_of_analysis": "image", "warning": "Research evaluation only. Repeated images per patient are not independent.",
               "train": evaluate(records["train"], best_weights, best_bias),
               "validation": evaluate(records["validation"], best_weights, best_bias),
               "test": evaluate(records["test"], best_weights, best_bias)}
    return model, metrics


def read_model(path):
    data = Path(path).read_bytes()
    model = json.loads(data)
    weights = model.get("weights")
    bias = model.get("bias")
    if (model.get("format") != MODEL_KIND or model.get("image_size") != SIDE or
            not isinstance(model.get("name"), str) or not model["name"].strip() or
            not isinstance(weights, list) or len(weights) != FEATURES or
            not all(type(value) in (float, int) and math.isfinite(value) for value in weights) or
            type(bias) not in (float, int) or not math.isfinite(bias)):
        raise ValueError("Invalid or unsupported model file")
    return model, hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--model-out", default="artifacts/model.json")
    training.add_argument("--metrics-out", default="artifacts/metrics.json")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=0.1)
    training.add_argument("--seed", type=int, default=42)
    for name in ("inspect", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True)
        if name == "predict":
            command.add_argument("--image", help="Image path; if omitted, read raw image bytes from stdin")
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("Epochs and learning rate must be positive")
        if Path(args.model_out).resolve() == Path(args.metrics_out).resolve():
            raise ValueError("Model and metrics output paths must be different")
        records = load_manifest(args.manifest)
        model, metrics = train(records, args.epochs, args.learning_rate, args.seed)
        for name, value in ((args.model_out, model), (args.metrics_out, metrics)):
            destination = Path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"model": args.model_out, "metrics": args.metrics_out, "best_epoch": metrics["best_epoch"]}))
    else:
        model, model_id = read_model(args.model)
        if args.command == "inspect":
            print(json.dumps({"name": model["name"], "format": model["format"], "model_id": model_id}))
        else:
            if args.image:
                image_path = Path(args.image)
                if image_path.stat().st_size > MAX_IMAGE_BYTES:
                    raise ValueError("Image exceeds 10 MB")
                data = image_path.read_bytes()
            else:
                data = sys.stdin.buffer.read(MAX_IMAGE_BYTES + 1)
            probability = predict_probability(model["weights"], model["bias"], features_from_bytes(data))
            print(json.dumps({"benign": 1 - probability, "malignant": probability, "model_id": model_id}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
