#!/bin/bash
# Start the API in the background, wait until it is ready, then start the UI.

uvicorn api.main:app --host 0.0.0.0 --port 8000 &

python - <<'PY'
import time, urllib.request
for _ in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/")
        print("API is ready")
        break
    except Exception:
        time.sleep(2)
PY

streamlit run app/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false