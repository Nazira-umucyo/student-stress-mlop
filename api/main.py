"""FastAPI backend: prediction, data upload, and retraining."""
import io
import json
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import database
from src.model import METADATA_PATH, retrain
from src.prediction import load_artifacts, predict_one
from src.preprocessing import clean_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()  # load the model once when the server starts
    yield


app = FastAPI(
    title="Student Stress Level Prediction API",
    description="Predicts student stress level (Low / Medium / High) and supports "
                "uploading new data and retraining the model. "
                "An educational screening tool, not a diagnosis.",
    version="1.0",
    lifespan=lifespan,
)

retrain_lock = threading.Lock()


class StudentInput(BaseModel):
    anxiety_level: int = Field(..., ge=0, le=21, examples=[15])
    self_esteem: int = Field(..., ge=0, le=30, examples=[11])
    mental_health_history: int = Field(..., ge=0, le=1, examples=[1])
    depression: int = Field(..., ge=0, le=27, examples=[18])
    headache: int = Field(..., ge=0, le=5, examples=[3])
    blood_pressure: int = Field(..., ge=1, le=3, examples=[3])
    sleep_quality: int = Field(..., ge=0, le=5, examples=[1])
    breathing_problem: int = Field(..., ge=0, le=5, examples=[4])
    noise_level: int = Field(..., ge=0, le=5, examples=[4])
    living_conditions: int = Field(..., ge=0, le=5, examples=[1])
    safety: int = Field(..., ge=0, le=5, examples=[2])
    basic_needs: int = Field(..., ge=0, le=5, examples=[1])
    academic_performance: int = Field(..., ge=0, le=5, examples=[2])
    study_load: int = Field(..., ge=0, le=5, examples=[5])
    teacher_student_relationship: int = Field(..., ge=0, le=5, examples=[1])
    future_career_concerns: int = Field(..., ge=0, le=5, examples=[5])
    social_support: int = Field(..., ge=0, le=3, examples=[1])
    peer_pressure: int = Field(..., ge=0, le=5, examples=[4])
    extracurricular_activities: int = Field(..., ge=0, le=5, examples=[4])
    bullying: int = Field(..., ge=0, le=5, examples=[4])


@app.get("/")
def health():
    return {"status": "ok", "message": "Student Stress Prediction API is running"}


@app.get("/model-info")
def model_info():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return {**metadata, "data": database.upload_summary()}


@app.post("/predict")
def predict(student: StudentInput):
    try:
        return predict_one(student.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a CSV of labelled student data. It is cleaned and saved to the database."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
        clean = clean_data(raw, require_target=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the CSV file.")

    saved = database.save_upload(clean, file.filename)
    return {
        **saved,
        "rows_received": len(raw),
        "rows_removed_in_cleaning": len(raw) - len(clean),
        "class_counts": clean["stress_level"].map(
            {0: "Low", 1: "Medium", 2: "High"}).value_counts().to_dict(),
        "database": database.upload_summary(),
    }


@app.post("/retrain")
def retrain_model():
    """Fine-tune the current model on all uploaded data not yet used for training."""
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    try:
        return retrain()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        retrain_lock.release()


@app.get("/training-runs")
def training_runs():
    runs = database.get_training_runs()
    return runs.to_dict(orient="records")