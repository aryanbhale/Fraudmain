import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # 1. hour_of_day
    if 'transaction_timestamp' in df.columns:
        df['hour_of_day'] = df['transaction_timestamp'].dt.hour
    else:
        df['hour_of_day'] = 12

    # 2. is_cross_city
    if 'user_location' in df.columns and 'merchant_location' in df.columns:
        df['is_cross_city'] = df['user_location'] != df['merchant_location']
    else:
        df['is_cross_city'] = False

    # 3. amount_zscore (per user)
    if 'user_id' in df.columns and 'transaction_amount' in df.columns:
        df['amount_zscore'] = df.groupby('user_id')['transaction_amount'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        ).fillna(0)
    else:
        df['amount_zscore'] = 0

    # 4. is_high_amount
    if 'transaction_amount' in df.columns:
        threshold = df['transaction_amount'].quantile(0.95)
        df['is_high_amount'] = df['transaction_amount'] > threshold
    else:
        df['is_high_amount'] = False

    # 5. user_txn_velocity_7d
    if 'user_id' in df.columns and 'transaction_timestamp' in df.columns:
        df['orig_idx'] = df.index
        df = df.sort_values(['user_id', 'transaction_timestamp'])
        df['user_txn_velocity_7d'] = df.set_index('transaction_timestamp').groupby('user_id')['user_id'].rolling('7D').count().values
        df = df.sort_values('orig_idx').drop(columns=['orig_idx'])
    else:
        df['user_txn_velocity_7d'] = 1

    # 6. is_new_device
    if 'device_id' in df.columns:
        device_counts = df['device_id'].value_counts()
        df['is_new_device'] = df['device_id'].map(device_counts) < 3
    else:
        df['is_new_device'] = False

    # 7. is_invalid_ip
    if 'ip_is_valid' not in df.columns:
        df['ip_is_valid'] = True
    df['is_invalid_ip'] = ~df['ip_is_valid']

    # 8. is_unusual_hour
    if 'hour_of_day' in df.columns:
        df['is_unusual_hour'] = (df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)
    else:
        df['is_unusual_hour'] = False

    # 9. amount_to_balance_ratio
    if 'transaction_amount' in df.columns and 'account_balance' in df.columns:
        df['amount_to_balance_ratio'] = df['transaction_amount'] / df['account_balance'].replace(0, np.nan)
        df['amount_to_balance_ratio'] = df['amount_to_balance_ratio'].fillna(0)
    else:
        df['amount_to_balance_ratio'] = 0

    # 10. is_failed_status
    if 'transaction_status' in df.columns:
        df['is_failed_status'] = df['transaction_status'].astype(str).str.lower() == 'failed'
    else:
        df['is_failed_status'] = False

    return df
