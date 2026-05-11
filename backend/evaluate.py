import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from backend.model_utils import (
    load_config,
    load_model_for_inference,
    prepare_image,
    predict_from_tensor,
)

DEFAULT_CONFIG = PROJECT_ROOT / 'models' / 'model_config.json'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'models' / 'best_model.pth'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'path' not in df.columns or 'label' not in df.columns:
        raise ValueError('CSV file must contain path and label columns.')
    return df


def evaluate_split(split_name: str, split_csv: Path, model, cfg, device: str) -> dict:
    df = load_dataset(split_csv)
    y_true = []
    y_pred = []
    predictions = []

    for _, row in df.iterrows():
        image_path = Path(row['path'])
        tensor = prepare_image(image_path, cfg['img_size'], cfg['imagenet_mean'], cfg['imagenet_std'], device)
        result = predict_from_tensor(model, tensor, cfg['class_names'], topk=1)
        y_true.append(row['label'])
        y_pred.append(result['top_prediction']['label'])
        predictions.append(result['top_prediction'])

    label_names = cfg['class_names']
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=label_names, output_dict=True, zero_division=0)

    figure_path = RESULTS_DIR / f'confusion_matrix_{split_name}.png'
    _save_confusion_matrix(cm, label_names, split_name, figure_path)

    summary = {
        'split': split_name,
        'samples': len(df),
        'accuracy': float(acc),
        'confusion_matrix_image': str(figure_path),
        'classification_report': report,
    }
    summary_path = RESULTS_DIR / f'evaluation_summary_{split_name}.json'
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=4)

    return summary


def _save_confusion_matrix(matrix: np.ndarray, labels: list, title: str, output_path: Path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='OrRd', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {title}', fontsize=16)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = load_model_for_inference(DEFAULT_CONFIG, DEFAULT_CHECKPOINT, device)

    val_summary = evaluate_split('val', PROJECT_ROOT / 'data' / 'splits' / 'val.csv', model, cfg, device)
    test_summary = evaluate_split('test', PROJECT_ROOT / 'data' / 'splits' / 'test.csv', model, cfg, device)

    combined = {
        'validation': val_summary,
        'test': test_summary,
    }

    overall_path = RESULTS_DIR / 'evaluation_summary_all.json'
    with open(overall_path, 'w') as handle:
        json.dump(combined, handle, indent=4)
    print(f'Evaluation complete. Summary saved to: {overall_path}')


if __name__ == '__main__':
    main()
