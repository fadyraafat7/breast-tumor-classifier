import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from pipeline import load_manifest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'ml' / 'pipeline.py'


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = self.directory / 'manifest.csv'
        self.rows = []
        for index, split in enumerate(('train', 'validation', 'test')):
            for label, base in (('benign', 20), ('malignant', 230)):
                path = self.directory / f'{split}-{label}.png'
                image = Image.new('L', (20, 20), base + index)
                image.putpixel((index, index), 100 + index)
                image.save(path)
                self.rows.append({'path': path.name, 'label': label,
                                  'patient_id': f'{split}-{label}', 'split': split})
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open('w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=('path', 'label', 'patient_id', 'split'))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_rejects_patient_and_identical_image_leakage(self):
        self.rows[2]['patient_id'] = self.rows[0]['patient_id']
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Patient appears in multiple splits'):
            load_manifest(self.manifest)
        self.rows[2]['patient_id'] = 'validation-benign'
        (self.directory / self.rows[2]['path']).write_bytes((self.directory / self.rows[0]['path']).read_bytes())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Identical image bytes'):
            load_manifest(self.manifest)

    def test_training_inference_and_local_http(self):
        model = self.directory / 'model.json'
        metrics = self.directory / 'metrics.json'
        subprocess.run([sys.executable, str(SCRIPT), 'train', '--manifest', str(self.manifest),
                        '--model-out', str(model), '--metrics-out', str(metrics), '--epochs', '80'],
                       check=True, capture_output=True, text=True)
        report = json.loads(metrics.read_text())
        self.assertEqual(report['test']['images'], 2)
        self.assertEqual(report['unit_of_analysis'], 'image')
        def predict(filename):
            result = subprocess.run([sys.executable, str(SCRIPT), 'predict', '--model', str(model),
                                     '--image', str(self.directory / filename)], check=True, capture_output=True, text=True)
            return json.loads(result.stdout)
        low = predict('test-benign.png')
        high = predict('test-malignant.png')
        self.assertLess(low['malignant'], high['malignant'])
        self.assertAlmostEqual(high['benign'] + high['malignant'], 1.0)
        self.assertEqual(len(high['model_id']), 64)
        environment = os.environ.copy()
        environment.update({'PORT': '0', 'HOST': '127.0.0.1', 'MODEL_PATH': str(model), 'PYTHON': sys.executable})
        server = subprocess.Popen(['node', str(ROOT / 'server.mjs')], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=environment)
        def stop_server():
            server.terminate()
            server.communicate(timeout=10)
        self.addCleanup(stop_server)
        url = server.stdout.readline().strip().removeprefix('Local: ')
        self.assertTrue(url.startswith('http://127.0.0.1:'), url)
        with urllib.request.urlopen(url + '/api/model', timeout=10) as response:
            status = json.load(response)
        self.assertTrue(status['available'])
        self.assertEqual(status['model_id'], high['model_id'])
        body = (self.directory / 'test-malignant.png').read_bytes()
        request = urllib.request.Request(url + '/api/predict', data=body,
                                         headers={'Content-Type': 'image/png', 'Origin': url})
        with urllib.request.urlopen(request, timeout=10) as response:
            actual = json.load(response)
        self.assertEqual(actual, high)
        request = urllib.request.Request(url + '/api/predict', data=body, headers={'Content-Type': 'image/png', 'Origin': 'http://example.com'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 403)
        request = urllib.request.Request(url + '/api/predict', data=b'not an image',
                                         headers={'Content-Type': 'image/png', 'Origin': url})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        self.assertEqual(error.exception.code, 422)


if __name__ == '__main__':
    unittest.main()
