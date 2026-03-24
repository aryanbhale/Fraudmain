import sys
import pandas as pd
from backend.cleaner import clean_data
from backend.features import engineer_features
from backend.model import train_and_predict

df = pd.read_csv("samplemain.csv")
print(f"Total rows: {len(df)}")
df, report = clean_data(df)
df = engineer_features(df)
df, metrics = train_and_predict(df)

fraud_count = df['fraud_label'].sum()
print(f"Fraud generated: {fraud_count} ({(fraud_count/len(df))*100:.2f}%)")
print(f"Predicted Fraud: {(df['fraud_prediction'] == 1).sum()}")
print(f"Metrics F1: {metrics.get('model_f1')}")
print(f"Precision: {metrics.get('precision')}")
print(f"Recall: {metrics.get('recall')}")
