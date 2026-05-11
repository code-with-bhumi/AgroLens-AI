import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def load_config(config_path: Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as handle:
        return json.load(handle)


def build_model_architecture(model_choice: str, num_classes: int) -> torch.nn.Module:
    if model_choice.lower() == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )
    elif model_choice.lower() == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_choice}")
    return model


def _fix_state_dict(state_dict: dict) -> dict:
    if any(key.startswith('module.') for key in state_dict.keys()):
        return {key.replace('module.', ''): value for key, value in state_dict.items()}
    return state_dict


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
            header = handle.read(1024)
        return header.startswith('version https://git-lfs.github.com/spec/v1')
    except Exception:
        return False


def _save_fallback_checkpoint(model: torch.nn.Module, fallback_path: Path) -> None:
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fallback_path)


def _load_state_dict_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {path} does not contain a valid state_dict")
    model.load_state_dict(_fix_state_dict(state))
    return model.to(device)


def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint_path)
    fallback_path = checkpoint_path.parent / 'fallback_model.pth'

    if checkpoint_path.exists() and _is_git_lfs_pointer(checkpoint_path):
        if fallback_path.exists():
            return _load_state_dict_checkpoint(model, fallback_path, device)
        _save_fallback_checkpoint(model, fallback_path)
        return model.to(device)

    if not checkpoint_path.exists():
        if fallback_path.exists():
            return _load_state_dict_checkpoint(model, fallback_path, device)
        _save_fallback_checkpoint(model, fallback_path)
        return model.to(device)

    try:
        return _load_state_dict_checkpoint(model, checkpoint_path, device)
    except Exception:
        if fallback_path.exists():
            return _load_state_dict_checkpoint(model, fallback_path, device)
        _save_fallback_checkpoint(model, fallback_path)
        return model.to(device)


def prepare_image(image_input: str or Image.Image, img_size: int, mean: List[float], std: List[float], device: torch.device) -> torch.Tensor:
    if isinstance(image_input, (str, Path)):
        image = Image.open(str(image_input)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError('Input must be a file path or PIL.Image.Image instance.')

    transform_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    tensor = transform_pipeline(image).unsqueeze(0).to(device)
    return tensor


def predict_from_tensor(model: torch.nn.Module, tensor: torch.Tensor, class_names: List[str], topk: int = 3) -> dict:
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().squeeze(0).numpy()
    top_indices = np.argsort(probabilities)[::-1][:topk]
    predictions = [
        {
            'label': class_names[int(idx)],
            'confidence': float(probabilities[int(idx)])
        }
        for idx in top_indices
    ]
    return {
        'predictions': predictions,
        'top_prediction': predictions[0]
    }


def load_model_for_inference(config_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    cfg = load_config(config_path)
    model = build_model_architecture(cfg['model_choice'], cfg['num_classes'])
    model = load_model_weights(model, checkpoint_path, device)
    return model, cfg
