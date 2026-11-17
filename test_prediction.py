import pandas as pd
from src.prediction import predict_one

test = pd.read_csv("data/test.csv")
for i in range(3):
    row = test.iloc[i]
    result = predict_one(row.drop("stress_level").to_dict())
    print("True:", int(row["stress_level"]), "| Predicted:", result)