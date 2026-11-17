"""Preprocessing shared by prediction, retraining, and the API."""
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TARGET = "stress_level"
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}

with open(MODELS_DIR / "features.json") as f:
    FEATURES = json.load(f)

# Allowed value range for each feature (from the original dataset)
FEATURE_RANGES = {f: (0, 5) for f in FEATURES}
FEATURE_RANGES.update({
    "anxiety_level": (0, 21),
    "self_esteem": (0, 30),
    "mental_health_history": (0, 1),
    "depression": (0, 27),
    "blood_pressure": (1, 3),
    "social_support": (0, 3),
})


def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.joblib")


def clean_data(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Validate and clean raw data: check columns, convert to numbers,
    drop incomplete rows and duplicates, and clip values to valid ranges."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().drop_duplicates()

    for col, (low, high) in FEATURE_RANGES.items():
        df[col] = df[col].clip(low, high)

    if require_target:
        df = df[df[TARGET].isin(CLASS_NAMES.keys())]
        df[TARGET] = df[TARGET].astype(int)

    if df.empty:
        raise ValueError("No valid rows left after cleaning.")
    return df.reset_index(drop=True)


def scale_features(df: pd.DataFrame, scaler=None):
    """Scale features with the saved scaler (same one used in training)."""
    scaler = scaler or load_scaler()
    return scaler.transform(df[FEATURES])