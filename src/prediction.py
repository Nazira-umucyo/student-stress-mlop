"""Load the current model and make predictions."""
import pandas as pd
from tensorflow import keras

from src.preprocessing import MODELS_DIR, CLASS_NAMES, clean_data, scale_features, load_scaler

MODEL_PATH = MODELS_DIR / "stress_model.keras"

_model = None
_scaler = None


def load_artifacts(force_reload: bool = False):
    """Load the model and scaler once and keep them in memory.
    force_reload=True is used after retraining to pick up the new model."""
    global _model, _scaler
    if _model is None or force_reload:
        _model = keras.models.load_model(MODEL_PATH)
        _scaler = load_scaler()
    return _model, _scaler


def predict_one(features: dict) -> dict:
    """Predict the stress level for one student."""
    model, scaler = load_artifacts()
    df = clean_data(pd.DataFrame([features]))
    probs = model.predict(scale_features(df, scaler), verbose=0)[0]
    pred = int(probs.argmax())
    return {
        "stress_level": pred,
        "label": CLASS_NAMES[pred],
        "confidence": round(float(probs[pred]), 3),
        "probabilities": {CLASS_NAMES[i]: round(float(p), 3) for i, p in enumerate(probs)},
    }