# AgroLens-AI Frontend

This frontend is a minimal static UI for uploading leaf images and requesting predictions from the backend API.

## Usage

1. Start the backend API:
   ```bash
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```
2. Open `frontend/index.html` in a browser.
3. Select a leaf image and click `Predict`.

## Notes

- The frontend sends the image to `http://localhost:8000/predict`.
- You can host the frontend using any static file server if needed.
