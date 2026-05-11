# AgroLens-AI Backend

This backend provides:
- A FastAPI inference service for raw leaf images
- A reusable inference CLI for unseen image file paths
- Evaluation tooling to compute confusion matrices and classification reports
- A utility to save final model weights to `models/final_model.pth`

## Setup

```bash
cd backend
python -m pip install -r requirements.txt
```

## Run the API

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Inference CLI

```bash
python backend/inference.py --image /path/to/image.jpg
```

## Evaluation

```bash
python backend/evaluate.py
```

## Save Final Weights

```bash
python backend/save_final_weights.py
```

## Git LFS checkpoint note

If `backend/app.py` fails while loading `models/best_model.pth`, it may be a Git LFS pointer file rather than the actual checkpoint. In that case the backend automatically falls back to a randomly initialized model and creates `models/fallback_model.pth`.

For best results you should still restore the real weights using:

```bash
git lfs pull
```

or replace `models/best_model.pth` with the real PyTorch checkpoint file.
