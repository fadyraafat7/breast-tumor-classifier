"""Train or run the notebook-inspired ResNet-18 tissue-patch classifier."""
import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from idc_data import check_rows, load_manifest

FORMAT = 'idc-resnet18-v1'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 50


def dependencies():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError('Install the IDC dependencies first: python -m pip install -r requirements-idc.txt') from exc
    return torch, nn, DataLoader, Dataset, models, transforms, Image, ImageOps


def make_model(nn, models, weights):
    model = models.resnet18(weights=weights)
    features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(features, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
                             nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
                             nn.Linear(256, 2))
    for layer in model.fc:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def make_transform(transforms, training):
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def metrics(predictions, targets, loss):
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, targets))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, targets))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, targets))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, targets))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {'images': len(targets), 'loss': loss, 'accuracy': (tp + tn) / len(targets),
            'precision': precision, 'recall': recall, 'specificity': specificity,
            'f1': 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0,
            'confusion_matrix': {'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn}}


def train(args):
    torch, nn, DataLoader, Dataset, models, transforms, Image, ImageOps = dependencies()
    if args.epochs < 1 or args.batch_size < 2 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Epochs must be positive, batch size at least 2, and learning rate finite and positive.')
    if Path(args.model_out).resolve() == Path(args.metrics_out).resolve():
        raise ValueError('Model and metrics output paths must be different.')
    rows = load_manifest(args.manifest)
    counts = check_rows(rows)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    class PatchDataset(Dataset):
        def __init__(self, entries, training):
            self.entries = entries
            self.transform = make_transform(transforms, training)

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, index):
            row = self.entries[index]
            with Image.open(row['image_path']) as image:
                tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
            return tensor, int(row['label'])

    datasets = {split: PatchDataset([row for row in rows if row['split'] == split], split == 'train')
                for split in ('train', 'validation', 'test')}
    train_size = len(datasets['train'])
    # BatchNorm in the notebook's classifier head cannot train on a batch of one.
    drop_last = train_size > args.batch_size and train_size % args.batch_size == 1
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train',
                                 drop_last=drop_last if split == 'train' else False, num_workers=0)
               for split, dataset in datasets.items()}

    weights = None if args.scratch or args.pretrained_file else models.ResNet18_Weights.DEFAULT
    model = make_model(nn, models, weights)
    if args.pretrained_file:
        backbone = models.resnet18(weights=None)
        backbone.load_state_dict(torch.load(args.pretrained_file, map_location='cpu', weights_only=True))
        model.load_state_dict({**{k: v for k, v in backbone.state_dict().items() if not k.startswith('fc.')},
                               **{k: v for k, v in model.state_dict().items() if k.startswith('fc.')}}, strict=True)
    model.to(device)
    total = sum(counts['train'])
    class_weights = torch.tensor([total / (2 * count) for count in counts['train']], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=0.9)

    def run_epoch(split, training=False):
        # Freeze the backbone's BatchNorm state as well as its parameters.
        model.eval()
        if training:
            model.fc.train()
        loss_total = weight_total = 0.0
        predictions, targets = [], []
        with torch.set_grad_enabled(training):
            for images, labels in loaders[split]:
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad()
                logits = model(images)
                loss_sum = criterion(logits, labels)
                denominator = class_weights[labels].sum()
                if training:
                    (loss_sum / denominator).backward()
                    optimizer.step()
                loss_total += loss_sum.item()
                weight_total += denominator.item()
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        return metrics(predictions, targets, loss_total / weight_total)

    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch('train', True)
        validation = run_epoch('validation')
        history.append({'epoch': epoch, 'train': train_metrics, 'validation': validation})
        print(f"Epoch {epoch}/{args.epochs}: train loss {train_metrics['loss']:.4f}, validation loss {validation['loss']:.4f}", flush=True)
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            best_state = copy.deepcopy({key: value.cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise ValueError('Training did not produce a finite validation loss.')
    model.load_state_dict(best_state)
    test_metrics = run_epoch('test')
    model_path, metrics_path = Path(args.model_out), Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'format': FORMAT, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'labels': ['no_idc', 'idc'], 'input': '50x50 RGB histopathology patch',
                  'state_dict': best_state}
    torch.save(checkpoint, model_path)
    report = {'format': FORMAT, 'unit_of_analysis': 'tissue patch', 'seed': args.seed,
              'best_epoch': best_epoch, 'model_selection': 'minimum validation weighted cross entropy',
              'pretraining': 'random initialization' if args.scratch else ('local ImageNet weights' if args.pretrained_file else 'torchvision ImageNet weights'),
              'counts': counts, 'history': history, 'test': test_metrics,
              'warning': 'Research only. Patch-level results are not patient-level or clinical performance.'}
    metrics_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Model: {model_path}\nMetrics: {metrics_path}\nSelected epoch: {best_epoch}')


def predict(args):
    torch, nn, _, _, models, transforms, Image, ImageOps = dependencies()
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT or checkpoint.get('labels') != ['no_idc', 'idc']:
        raise ValueError('The checkpoint is not a supported IDC model.')
    model = make_model(nn, models, None)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    with Image.open(args.image) as image:
        tensor = make_transform(transforms, False)(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0].tolist()
    print(json.dumps({'no_idc_score': scores[0], 'idc_score': scores[1],
                      'notice': 'Research only; scores are uncalibrated and not for diagnosis.'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--manifest', required=True)
    training.add_argument('--epochs', type=int, default=10)
    training.add_argument('--batch-size', type=int, default=32)
    training.add_argument('--learning-rate', type=float, default=0.001)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--model-out', default='artifacts/idc-model.pt')
    training.add_argument('--metrics-out', default='artifacts/idc-metrics.json')
    training.add_argument('--cpu', action='store_true')
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument('--scratch', action='store_true', help='Use random backbone initialization, without downloading weights')
    initialization.add_argument('--pretrained-file', help='Path to trusted official ResNet-18 ImageNet state_dict weights')
    inference = sub.add_parser('predict')
    inference.add_argument('--model', required=True)
    inference.add_argument('--image', required=True)
    args = parser.parse_args()
    train(args) if args.command == 'train' else predict(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
