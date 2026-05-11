import io
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from backend.model_utils import (
    load_config,
    load_model_for_inference,
    prepare_image,
    predict_from_tensor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / 'models' / 'model_config.json'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'models' / 'best_model.pth'
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

app = FastAPI(
    title='AgroLens-AI Inference API',
    description='Backend service for plant disease prediction and evaluation metrics.',
    version='1.0.0',
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1
    }
)

cfg = load_config(MODEL_CONFIG_PATH)
model = None
model_load_error = None


def _load_model():
    global model, model_load_error
    if model is None and model_load_error is None:
        try:
            model, _ = load_model_for_inference(MODEL_CONFIG_PATH, DEFAULT_CHECKPOINT, DEVICE)
        except Exception as exc:
            model_load_error = exc
    return model


def load_image_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image upload: {exc}')
    return image

@app.get("/")
async def root():
    return {
        "message": "AgroLens-AI API Running Successfully",
        "available_routes": {
            "swagger_docs": {
                "method": "GET",
                "endpoint": "/docs",
                "description": "Swagger UI Documentation"
            },
            "health_check": {
                "method": "GET",
                "endpoint": "/health",
                "description": "Check API and model health"
            },
            "top_classes": {
                "method": "GET",
                "endpoint": "/top_classes?limit=10",
                "description": "Get top class names"
            },
            "predict_upload": {
                "method": "POST",
                "endpoint": "/predict",
                "description": "Predict from uploaded image"
            },
            "predict_path": {
                "method": "POST",
                "endpoint": "/predict_path",
                "description": "Predict from image path"
            }
        }
    }


@app.get('/health')
async def health_check():
    return {'status': 'ok', 'model': cfg['model_choice'], 'classes': len(cfg['class_names'])}

@app.get('/top_classes')
async def top_classes(limit: int = 10):
    return {'classes': cfg['class_names'][:limit]}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    global model
    image = load_image_bytes(await file.read())
    if _load_model() is None:
        raise HTTPException(
            status_code=500,
            detail=f'Model failed to load: {model_load_error}'
        )
    tensor = prepare_image(image, cfg['img_size'], cfg['imagenet_mean'], cfg['imagenet_std'], DEVICE)
    output = predict_from_tensor(model, tensor, cfg['class_names'], topk=3)
    return JSONResponse({'source': 'upload', **output})


@app.post('/predict_path')
async def predict_path(image_path: str):
    global model
    filepath = Path(image_path)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f'Image not found: {filepath}')
    if _load_model() is None:
        raise HTTPException(
            status_code=500,
            detail=f'Model failed to load: {model_load_error}'
        )
    tensor = prepare_image(filepath, cfg['img_size'], cfg['imagenet_mean'], cfg['imagenet_std'], DEVICE)
    output = predict_from_tensor(model, tensor, cfg['class_names'], topk=3)
    return JSONResponse({'source': str(filepath), **output})
