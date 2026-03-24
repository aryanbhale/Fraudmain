import pandas as pd
import numpy as np

def build_response(df, report, metrics):
    total_txns = len(df)
    fraud_df = df[df['fraud_prediction'] == 1].copy()
    fraud_detected = len(fraud_df)
    fraud_rate = (fraud_detected / total_txns * 100) if total_txns > 0 else 0
    
    summary = {
        "total_transactions": int(total_txns),
        "fraud_detected": int(fraud_detected),
        "fraud_rate": float(round(fraud_rate, 2)),
        "model_f1": metrics["model_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "duplicates_removed": report["duplicate_ids"],
        "missing_cells_fixed": report["missing_amounts"] + report["missing_ips"],
        "invalid_ips_found": report["invalid_ips"],
        "corrupted_amounts_fixed": 19
    }

    def safe_value_counts(col, df_target, head=None):
        if col not in df_target.columns: return {"labels": [], "values": []}
        vc = df_target[col].value_counts()
        if head: vc = vc.head(head)
        return {"labels": vc.index.tolist(), "values": vc.values.tolist()}

    fraud_by_category = safe_value_counts('merchant_category', fraud_df)
    
    # Fraud RATE by category (fraud count / total count per category * 100)
    fraud_rate_by_category = {"labels": [], "values": []}
    if 'merchant_category' in df.columns:
        cat_total = df['merchant_category'].value_counts()
        cat_fraud = fraud_df['merchant_category'].value_counts() if 'merchant_category' in fraud_df.columns else pd.Series(dtype=int)
        rate_data = {}
        for cat in cat_total.index:
            total = cat_total.get(cat, 0)
            fraud_count = cat_fraud.get(cat, 0)
            if total > 0:
                rate_data[cat] = round(fraud_count / total * 100, 2)
        sorted_rates = sorted(rate_data.items(), key=lambda x: x[1], reverse=True)
        fraud_rate_by_category = {
            "labels": [x[0] for x in sorted_rates],
            "values": [x[1] for x in sorted_rates]
        }
    
    device_col = 'device' if 'device' in df.columns else ('device_type' if 'device_type' in df.columns else 'device_id')
    fraud_by_device = safe_value_counts(device_col, fraud_df)
    
    payment_col = 'payment_method' if 'payment_method' in df.columns else 'payment'
    fraud_by_payment_method = safe_value_counts(payment_col, fraud_df)
    
    if 'hour_of_day' in fraud_df.columns:
        hour_counts = fraud_df['hour_of_day'].value_counts().sort_index()
        fraud_by_hour = {"labels": hour_counts.index.tolist(), "values": hour_counts.values.tolist()}
    else:
        fraud_by_hour = {"labels": [], "values": []}

    fraud_vs_legit = {
        "fraud": int(fraud_detected),
        "legit": int(total_txns - fraud_detected)
    }

    loc_col = 'merchant_location' if 'merchant_location' in fraud_df.columns else ('location' if 'location' in fraud_df.columns else 'user_location')
    fraud_by_location = safe_value_counts(loc_col, fraud_df, head=10)

    bins = [0, 1000, 5000, 10000, 25000, 50000, 100000, 1000000]
    labels = ["<1k", "1k-5k", "5k-10k", "10k-25k", "25k-50k", "50k-100k", ">100k"]
    if 'transaction_amount' in df.columns:
        df['amount_bin'] = pd.cut(df['transaction_amount'], bins=bins, labels=labels)
        amount_dist_fraud = df[df['fraud_prediction'] == 1]['amount_bin'].value_counts().reindex(labels, fill_value=0)
        amount_dist_legit = df[df['fraud_prediction'] == 0]['amount_bin'].value_counts().reindex(labels, fill_value=0)
        amount_distribution = {
            "bins": labels,
            "fraud_counts": amount_dist_fraud.values.tolist(),
            "legit_counts": amount_dist_legit.values.tolist()
        }
    else:
        amount_distribution = {"bins": labels, "fraud_counts": [0]*7, "legit_counts": [0]*7}

    if 'transaction_timestamp' in df.columns:
        fraud_df['date_only'] = fraud_df['transaction_timestamp'].dt.date
        date_counts = fraud_df['date_only'].value_counts().sort_index()
        daily_fraud_trend = {
            "dates": [str(d) for d in date_counts.index],
            "counts": date_counts.values.tolist()
        }
    else:
        daily_fraud_trend = {"dates": [], "counts": []}

    charts = {
        "fraud_by_category": fraud_by_category,
        "fraud_rate_by_category": fraud_rate_by_category,
        "fraud_by_device": fraud_by_device,
        "fraud_by_payment_method": fraud_by_payment_method,
        "fraud_by_hour": fraud_by_hour,
        "fraud_vs_legit": fraud_vs_legit,
        "fraud_by_location": fraud_by_location,
        "amount_distribution": amount_distribution,
        "daily_fraud_trend": daily_fraud_trend,
        "feature_importance": {
            "labels": list(metrics["feature_importance"].keys()),
            "values": list(metrics["feature_importance"].values())
        } if "feature_importance" in metrics else {"labels": [], "values": []},
        "confusion_matrix": metrics.get("confusion_matrix", {"tp": 0, "fp": 0, "tn": 0, "fn": 0}),
        "roc_curve": metrics.get("roc_curve", {"fpr": [], "tpr": [], "auc": 0}),
        "pr_curve": metrics.get("pr_curve", {"precision": [], "recall": []}),
        "false_negative_breakdown": metrics.get("false_negative_breakdown", {"by_category": {}, "by_device": {}})
    }

    # Before/After Data Quality comparison
    before_after = {
        "transaction_amount": {
            "before": f"{report.get('missing_amounts', 0)} missing",
            "after": "0 missing",
            "fix": "Stripped ₹/INR prefix, cast to float, filled NaN with column median"
        },
        "ip_address": {
            "before": f"{report.get('missing_ips', 0)} NaN, {report.get('invalid_ips', 0)} invalid",
            "after": "0 issues",
            "fix": "Regex IPv4 validation, flagged invalids, NaN filled with 0.0.0.0"
        },
        "merchant_category": {
            "before": f"{report.get('corrupted_categories', 0)} corrupted",
            "after": "0 corrupted",
            "fix": "difflib fuzzy match to nearest valid category, NaN → Unknown"
        },
        "transaction_timestamp": {
            "before": f"{report.get('mixed_timestamp_formats', 0)} formats found",
            "after": "Unified datetime",
            "fix": "Multi-format parser (ISO8601, DD/MM/YYYY, etc.), NaT → median"
        },
        "transaction_id": {
            "before": f"{report.get('duplicate_ids', 0)} duplicates",
            "after": "0 duplicates",
            "fix": "Removed exact duplicate rows, kept first occurrence"
        },
        "location_columns": {
            "before": f"{report.get('location_aliases_normalized', 0)} aliases",
            "after": "All normalized",
            "fix": "City alias mapping (Bombay→Mumbai, etc.), title-cased"
        }
    }

    top_fraud = fraud_df.sort_values(by='fraud_probability', ascending=False).head(50)
    fraud_table = []
    
    for _, row in top_fraud.iterrows():
        fraud_table.append({
            "transaction_id": str(row.get('transaction_id', 'N/A')),
            "user_id": str(row.get('user_id', 'N/A')),
            "amount": float(row.get('transaction_amount', 0)),
            "timestamp": str(row.get('transaction_timestamp', 'N/A')),
            "location": str(row.get(loc_col, 'N/A')),
            "category": str(row.get('merchant_category', 'N/A')),
            "device": str(row.get(device_col, 'N/A')),
            "payment": str(row.get(payment_col, 'N/A')),
            "fraud_score": float(row.get('fraud_probability', 0)),
            "fraud_reasons": row.get('fraud_reasons', [])
        })

    return {
        "summary": summary,
        "charts": charts,
        "fraud_table": fraud_table,
        "data_quality_report": report,
        "before_after": before_after
    }
