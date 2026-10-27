"""Prepare patient-disjoint splits for the notebook's histopathology patches."""
import argparse
import csv
import hashlib
import random
from pathlib import Path

SPLITS = ('train', 'validation', 'test')
LABELS = {'0': 0, '1': 1}  # 0 = no IDC; 1 = IDC, not benign versus malignant.
EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def split_patients(patients, seed):
    patients = sorted(patients)
    if len(patients) < 3:
        raise ValueError('At least three patient folders are needed for patient-disjoint splits.')
    random.Random(seed).shuffle(patients)
    test_count = max(1, round(len(patients) * 0.15))
    validation_count = max(1, round(len(patients) * 0.15))
    if test_count + validation_count >= len(patients):
        raise ValueError('Not enough patients for the requested splits.')
    return {patient: ('test' if i < test_count else 'validation' if i < test_count + validation_count else 'train')
            for i, patient in enumerate(patients)}


def collect_dataset(root, seed=42):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Dataset root is not a directory.')
    patients = sorted(path for path in root.iterdir() if path.is_dir() and
                      any((path / category).is_dir() for category in LABELS))
    if not patients:
        raise ValueError('No patient folders with 0/ or 1/ subfolders were found at this dataset root.')
    assignment = split_patients([patient.name for patient in patients], seed)
    rows = []
    hashes = {}
    for patient in patients:
        for category, label in LABELS.items():
            folder = patient / category
            if not folder.is_dir():
                continue
            for image in sorted(folder.iterdir()):
                if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                    continue
                digest = hashlib.sha256()
                with image.open('rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                key = digest.hexdigest()
                previous = hashes.get(key)
                split = assignment[patient.name]
                if previous is not None:
                    raise ValueError(f'Identical image files found: {previous} and {image}. Remove duplicates before training.')
                hashes[key] = str(image)
                rows.append({'image_path': str(image.resolve()), 'patient_id': patient.name,
                             'label': label, 'split': split})
    check_rows(rows)
    return rows


def check_rows(rows, verify_hashes=False):
    by_patient = {}
    counts = {split: [0, 0] for split in SPLITS}
    seen_paths = set()
    hashes = set()
    for row in rows:
        patient, split = row['patient_id'], row['split']
        if not patient or split not in SPLITS or str(row['label']) not in ('0', '1'):
            raise ValueError('Each row needs a patient_id, label 0 or 1, and train/validation/test split.')
        previous = by_patient.setdefault(patient, split)
        if previous != split:
            raise ValueError(f'Patient {patient} appears in multiple splits.')
        image_path = Path(row['image_path']).resolve()
        if image_path in seen_paths:
            raise ValueError(f'Image path is repeated: {image_path}')
        if not image_path.is_file():
            raise ValueError(f'Image file does not exist: {image_path}')
        seen_paths.add(image_path)
        if verify_hashes:
            digest = hashlib.sha256()
            with image_path.open('rb') as image:
                for block in iter(lambda: image.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.digest() in hashes:
                raise ValueError(f'Identical image files found, including {image_path}. Remove duplicates before training.')
            hashes.add(digest.digest())
        counts[split][int(row['label'])] += 1
    for split, (negative, positive) in counts.items():
        if not negative or not positive:
            raise ValueError(f'The {split} split must have both 0 (no IDC) and 1 (IDC) patches. Try another seed or adjust the patient split.')
    return counts


def load_manifest(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        required = {'image_path', 'patient_id', 'label', 'split'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Manifest must contain image_path,patient_id,label,split columns.')
        rows = list(reader)
    for row in rows:
        row['patient_id'] = (row.get('patient_id') or '').strip()
        row['label'] = (row.get('label') or '').strip()
        row['split'] = (row.get('split') or '').strip().lower()
        image_path = (row.get('image_path') or '').strip()
        if not image_path:
            raise ValueError('Every row must include an image_path.')
        row['image_path'] = str((path.parent / image_path).resolve())
    check_rows(rows, verify_hashes=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Directory containing patient/0 and patient/1 subfolders')
    parser.add_argument('--output', default='data/idc-manifest.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = collect_dataset(args.dataset, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=['image_path', 'patient_id', 'label', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    counts = check_rows(rows)
    print(f'Wrote {len(rows)} image rows to {output}. Patient split seed: {args.seed}.')
    for split, values in counts.items():
        patients = len({row['patient_id'] for row in rows if row['split'] == split})
        print(f'{split}: {patients} patients; no IDC: {values[0]}; IDC: {values[1]}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
