# Student Stress Level Prediction: API (FastAPI) + web UI (Streamlit)
FROM python:3.12-slim

# Hugging Face Spaces runs containers as a non-root user with id 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    API_URL=http://127.0.0.1:8000

WORKDIR /home/user/app

# Install dependencies first so Docker can cache this layer
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the project
COPY --chown=user . .
RUN sed -i 's/\r$//' start.sh   # remove Windows line endings if present

EXPOSE 7860
CMD ["bash", "start.sh"]