import pandas as pd
from src import database
from src.preprocessing import clean_data
from src.model import retrain

raw = pd.read_csv("data/retrain_sample.csv")
clean = clean_data(raw, require_target=True)
print("Upload:", database.save_upload(clean, "retrain_sample.csv"))
print("Database:", database.upload_summary())

report = retrain()
for key, value in report.items():
    print(f"{key}: {value}")
print("Database after:", database.upload_summary())