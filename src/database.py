"""Database for uploaded training data and retraining history.

Uses the DATABASE_URL environment variable if it is set (e.g. a hosted
PostgreSQL database in deployment); otherwise a local SQLite file.
"""
import os
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from src.preprocessing import BASE_DIR

DEFAULT_DB = f"sqlite:///{(BASE_DIR / 'data' / 'stress.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB)
engine = create_engine(DATABASE_URL)

UPLOADS_TABLE = "uploaded_data"
RUNS_TABLE = "training_runs"


def save_upload(df: pd.DataFrame, filename: str) -> dict:
    """Save cleaned uploaded rows to the database, marked as not yet used for training."""
    upload_id = datetime.now().strftime("%Y%m%d%H%M%S")
    records = df.copy()
    records["upload_id"] = upload_id
    records["source_file"] = filename
    records["uploaded_at"] = datetime.now().isoformat(timespec="seconds")
    records["used_for_training"] = 0
    records.to_sql(UPLOADS_TABLE, engine, if_exists="append", index=False)
    return {"upload_id": upload_id, "rows_saved": len(records)}


def get_untrained_data() -> pd.DataFrame:
    """Return all uploaded rows not yet used for retraining."""
    try:
        return pd.read_sql(
            text(f"SELECT * FROM {UPLOADS_TABLE} WHERE used_for_training = 0"), engine)
    except Exception:
        return pd.DataFrame()


def mark_as_used():
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE {UPLOADS_TABLE} SET used_for_training = 1 "
                          "WHERE used_for_training = 0"))


def upload_summary() -> dict:
    """Counts of uploaded rows, total and pending retraining."""
    try:
        df = pd.read_sql(text(f"SELECT used_for_training FROM {UPLOADS_TABLE}"), engine)
        return {"total_rows": len(df), "pending_rows": int((df["used_for_training"] == 0).sum())}
    except Exception:
        return {"total_rows": 0, "pending_rows": 0}


def log_training_run(run: dict):
    pd.DataFrame([run]).to_sql(RUNS_TABLE, engine, if_exists="append", index=False)


def get_training_runs() -> pd.DataFrame:
    try:
        return pd.read_sql(text(f"SELECT * FROM {RUNS_TABLE} ORDER BY run_at DESC"), engine)
    except Exception:
        return pd.DataFrame()