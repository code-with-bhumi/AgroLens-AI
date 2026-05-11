import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from backend.model_utils import load_model_for_inference, prepare_image, predict_from_tensor

DEFAULT_CONFIG = PROJECT_ROOT / 'models' / 'model_config.json'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'models' / 'best_model.pth'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on a raw, unseen leaf image.')
    parser.add_argument('--image', '-i', required=True, help='Path to the input image file')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Model configuration JSON file')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Saved model weights file')
    parser.add_argument('--topk', type=int, default=3, help='Number of top class predictions to return')
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, cfg = load_model_for_inference(Path(args.config), Path(args.checkpoint), device)
    tensor = prepare_image(args.image, cfg['img_size'], cfg['imagenet_mean'], cfg['imagenet_std'], device)
    result = predict_from_tensor(model, tensor, cfg['class_names'], topk=args.topk)

    print('\nAgroLens-AI Inference Result')
    print('--------------------------------')
    print(f"Image: {args.image}")
    print(f"Prediction: {result['top_prediction']['label']}")
    print(f"Confidence: {result['top_prediction']['confidence']:.4f}\n")
    print('Top predictions:')
    for row in result['predictions']:
        print(f"  - {row['label']}: {row['confidence']:.4f}")


if __name__ == '__main__':
    main()
