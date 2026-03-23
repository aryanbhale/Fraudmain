import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def generate_fraud_label(row):
    score = 0
    if row.get('amount_zscore', 0) > 2.5: score += 2
    if row.get('is_cross_city', False): score += 1
    if row.get('is_unusual_hour', False): score += 1
    if row.get('is_new_device', False): score += 1
    if row.get('is_invalid_ip', False): score += 2
    if row.get('user_txn_velocity_7d', 0) > 10: score += 2
    if row.get('amount_to_balance_ratio', 0) > 0.8: score += 2
    if row.get('is_failed_status', False): score += 1
    return 1 if score >= 4 else 0

def train_and_predict(df):
    df['fraud_label'] = df.apply(generate_fraud_label, axis=1)

    features = [
        'hour_of_day', 'is_cross_city', 'amount_zscore', 'is_high_amount',
        'user_txn_velocity_7d', 'is_new_device', 'is_invalid_ip',
        'is_unusual_hour', 'amount_to_balance_ratio', 'is_failed_status'
    ]
    
    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].fillna(0).astype(float)
    y = df['fraud_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    df['fraud_probability'] = model.predict_proba(X)[:, 1].round(4)
    df['fraud_prediction'] = model.predict(X)

    def get_reasons(row):
        reasons = []
        if row.get('amount_zscore', 0) > 2.5: reasons.append("High amount velocity")
        if row.get('is_cross_city', False): reasons.append("Cross-City")
        if row.get('is_unusual_hour', False): reasons.append("Unusual hour")
        if row.get('is_new_device', False): reasons.append("New device")
        if row.get('is_invalid_ip', False): reasons.append("Invalid IP")
        if row.get('user_txn_velocity_7d', 0) > 10: reasons.append("High velocity")
        if row.get('amount_to_balance_ratio', 0) > 0.8: reasons.append("High balance drain")
        if row.get('is_failed_status', False): reasons.append("Previous failure")
        return reasons[:3] if reasons else ["Suspicious activity"]

    df['fraud_reasons'] = df.apply(get_reasons, axis=1)
    
    feature_importances = model.feature_importances_
    importance_dict = {f: float(imp) for f, imp in zip(features, feature_importances)}
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    
    metrics = {
        "model_f1": float(round(f1, 2)),
        "precision": float(round(precision, 2)),
        "recall": float(round(recall, 2)),
        "feature_importance": importance_dict
    }
    
    return df, metrics
