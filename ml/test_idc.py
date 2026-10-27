import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from idc_data import collect_dataset, load_manifest, split_patients
from idc_train import metrics

ROOT = Path(__file__).resolve().parent.parent
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'images'
        for patient in range(6):
            for label in (0, 1):
                folder = self.dataset / f'p{patient}' / str(label)
                folder.mkdir(parents=True)
                Image.new('RGB', (50, 50), (patient * 10 + label, label * 100, 40)).save(folder / 'patch.png')

    def write_manifest(self, rows):
        manifest = self.root / 'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=['image_path', 'patient_id', 'label', 'split'])
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_patient_split_is_deterministic_and_disjoint(self):
        self.assertEqual(split_patients(['c', 'a', 'b', 'd'], 42), split_patients(['b', 'd', 'c', 'a'], 42))
        rows = collect_dataset(self.dataset, seed=42)
        self.assertEqual(len(rows), 12)
        groups = {split: {row['patient_id'] for row in rows if row['split'] == split}
                  for split in ('train', 'validation', 'test')}
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])
        self.assertEqual(len(load_manifest(self.write_manifest(rows))), 12)

    def test_rejects_patient_leakage_and_duplicate_images(self):
        rows = collect_dataset(self.dataset)
        first = rows[0]
        other = next(row for row in rows if row['split'] != first['split'])
        other['patient_id'] = first['patient_id']
        with self.assertRaisesRegex(ValueError, 'multiple splits'):
            load_manifest(self.write_manifest(rows))
        duplicate = self.dataset / 'p0' / '0' / 'duplicate.png'
        duplicate.write_bytes((self.dataset / 'p1' / '0' / 'patch.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Identical image files'):
            collect_dataset(self.dataset)

    def test_correct_precision_recall_and_f1(self):
        result = metrics([1, 1, 0, 0, 0], [1, 0, 1, 1, 0], 0.2)
        self.assertEqual(result['precision'], 0.5)
        self.assertAlmostEqual(result['recall'], 1 / 3)
        self.assertAlmostEqual(result['f1'], 0.4)

    @unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch and torchvision are not installed')
    def test_train_and_predict_with_generated_patches(self):
        manifest = self.write_manifest(collect_dataset(self.dataset))
        model = self.root / 'model.pt'
        report = self.root / 'metrics.json'
        weights = self.root / 'backbone.pt'
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        # This fixture is randomly initialized; it only tests loading the local weights path.
        subprocess.run([sys.executable, '-c',
                        'import sys, torch; from torchvision.models import resnet18; torch.save(resnet18(weights=None).state_dict(), sys.argv[1])',
                        str(weights)], check=True, capture_output=True, text=True, env=environment)
        subprocess.run([sys.executable, str(ROOT / 'ml/idc_train.py'), 'train', '--manifest', str(manifest),
                        '--model-out', str(model), '--metrics-out', str(report), '--epochs', '1',
                        '--batch-size', '2', '--pretrained-file', str(weights), '--cpu'], check=True, capture_output=True, text=True, env=environment)
        result = json.loads(report.read_text())
        self.assertEqual(result['test']['images'], 2)
        self.assertEqual(result['unit_of_analysis'], 'tissue patch')
        prediction = subprocess.run([sys.executable, str(ROOT / 'ml/idc_train.py'), 'predict', '--model', str(model),
                                     '--image', str(self.dataset / 'p0/1/patch.png')],
                                    check=True, capture_output=True, text=True, env=environment)
        scores = json.loads(prediction.stdout)
        self.assertAlmostEqual(scores['no_idc_score'] + scores['idc_score'], 1, places=5)


if __name__ == '__main__':
    unittest.main()
