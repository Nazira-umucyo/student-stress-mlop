"""Retrain the saved model on newly uploaded data."""
import json
import shutil
from datetime import datetime

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src import database
from src.prediction import MODEL_PATH, load_artifacts
from src.preprocessing import BASE_DIR, MODELS_DIR, TARGET, clean_data, scale_features

SEED = 42
METADATA_PATH = MODELS_DIR / "metadata.json"
ARCHIVE_DIR = MODELS_DIR / "archive"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
TOLERANCE = 0.02 


def evaluate(model, scaler) -> dict:
    """Evaluate a model on the fixed test set."""
    test = pd.read_csv(TEST_PATH)
    X, y = scale_features(test, scaler), test[TARGET].values
    probs = model.predict(X, verbose=0)
    preds = probs.argmax(axis=1)
    loss = float(keras.losses.sparse_categorical_crossentropy(y, probs).numpy().mean())
    return {
        "accuracy": round(accuracy_score(y, preds), 3),
        "loss": round(loss, 3),
        "precision": round(precision_score(y, preds, average="macro"), 3),
        "recall": round(recall_score(y, preds, average="macro"), 3),
        "f1_score": round(f1_score(y, preds, average="macro"), 3),
    }


def retrain(epochs: int = 50, learning_rate: float = 1e-4) -> dict:
    """Fine-tune the current model on uploaded data that has not been used yet."""
   
    new_data = database.get_untrained_data()
    if new_data.empty:
        raise ValueError("No new data to retrain on. Upload a CSV first.")
    new_data = clean_data(new_data, require_target=True)

    original = pd.read_csv(TRAIN_PATH).sample(
        n=min(len(new_data) * 2, 693), random_state=SEED)
    combined = pd.concat([new_data, original], ignore_index=True)

    train_df, val_df = train_test_split(
        combined, test_size=0.15, stratify=combined[TARGET], random_state=SEED)

    model, scaler = load_artifacts(force_reload=True)
    old_metrics = evaluate(model, scaler)

    tf.random.set_seed(SEED)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        scale_features(train_df, scaler), train_df[TARGET].values,
        validation_data=(scale_features(val_df, scaler), val_df[TARGET].values),
        epochs=epochs, batch_size=32, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)])

    new_metrics = evaluate(model, scaler)
    accepted = new_metrics["accuracy"] >= old_metrics["accuracy"] - TOLERANCE

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    old_version = metadata.get("model_version", 1)

    if accepted:
        ARCHIVE_DIR.mkdir(exist_ok=True)
        shutil.copy(MODEL_PATH, ARCHIVE_DIR / f"stress_model_v{old_version}.keras")
        model.save(MODEL_PATH)
        metadata.update({
            "model_version": old_version + 1,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "test_metrics": new_metrics,
        })
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        database.mark_as_used()

    
    load_artifacts(force_reload=True)

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "accepted": bool(accepted),
        "model_version": metadata["model_version"],
        "new_rows": len(new_data),
        "replay_rows": len(original),
        "epochs_run": len(history.history["loss"]),
        **{f"old_{k}": v for k, v in old_metrics.items()},
        **{f"new_{k}": v for k, v in new_metrics.items()},
    }
    database.log_training_run(report)
    return report