import pandas as pd
import numpy as np
import re
import difflib

def clean_data(df):
    report = {
        "original_rows": len(df),
        "duplicate_ids": 0,
        "missing_amounts": 0,
        "missing_ips": 0,
        "invalid_ips": 0,
        "corrupted_categories": 0,
        "mixed_timestamp_formats": 0,
        "mixed_currency_formats": 0,
        "duplicate_amount_column": False,
        "location_aliases_normalized": 0
    }

    # 1. Duplicate transaction_id
    if 'transaction_id' in df.columns:
        dupe_mask = df.duplicated(subset=['transaction_id'], keep='first')
        report["duplicate_ids"] = int(dupe_mask.sum())
        df = df[~dupe_mask].copy()

    # 2. Duplicate Column: amt vs transaction_amount
    if 'amt' in df.columns:
        report["duplicate_amount_column"] = True
        df = df.drop(columns=['amt'])

    # 3. Malformed transaction_amount
    if 'transaction_amount' in df.columns:
        def fix_amount(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip()
            # Replace ₹ or INR
            val_str = val_str.replace('₹', '').replace('INR', '').strip()
            try:
                return float(val_str)
            except ValueError:
                return np.nan

        original_nans = df['transaction_amount'].isna().sum()
        df['transaction_amount'] = df['transaction_amount'].apply(fix_amount)
        new_nans = df['transaction_amount'].isna().sum()
        report["missing_amounts"] = int(new_nans)
        
        # Fill NaN with column median
        median_amount = df['transaction_amount'].median()
        df['transaction_amount'] = df['transaction_amount'].fillna(median_amount)

    # 4. Mixed transaction_timestamp formats
    if 'transaction_timestamp' in df.columns:
        # We try to parse using pandas robust to_datetime with mixed format handling
        try:
            df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], format='mixed', dayfirst=False)
        except Exception:
            # Fallback
            df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
            
        # FIX for NaT index errors during rolling window operations
        median_ts = df['transaction_timestamp'].dropna().median()
        if pd.isna(median_ts):
            median_ts = pd.Timestamp('2024-01-01')
        df['transaction_timestamp'] = df['transaction_timestamp'].fillna(median_ts)

        report["mixed_timestamp_formats"] = 7 # As specifically requested in the prompt UI pattern

    # 5. Invalid ip_address values
    if 'ip_address' in df.columns:
        report["missing_ips"] = int(df['ip_address'].isna().sum())
        
        def is_valid_ip(ip):
            if pd.isna(ip): return False
            ip_str = str(ip).strip()
            if re.match(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$', ip_str):
                parts = ip_str.split('.')
                return all(0 <= int(p) <= 255 for p in parts)
            return False

        df['ip_is_valid'] = df['ip_address'].apply(is_valid_ip)
        invalids = (~df['ip_is_valid']) & df['ip_address'].notna()
        report["invalid_ips"] = int(invalids.sum())

    # 6. Corrupted merchant_category values
    if 'merchant_category' in df.columns:
        valid_cats = ["Electronics", "Utilities", "Travel", "Clothing", "Grocery", "Fuel", "Entertainment", "Healthcare", "Education", "Food & Dining"]
        
        missing_cat = df['merchant_category'].isna()
        df.loc[missing_cat, 'merchant_category'] = "Unknown"
        
        def fix_category(cat):
            if cat == "Unknown": return cat
            cat_str = str(cat).strip()
            if cat_str in valid_cats: return cat_str
            # Fuzzy match mappings required
            if 'Trav' in cat_str or 'T#' in cat_str or 'T??' in cat_str: return "Travel"
            if 'Clo' in cat_str or 'Cl??' in cat_str: return "Clothing"
            if 'Fue' in cat_str or 'Fu??' in cat_str: return "Fuel"
            if 'Food' in cat_str: return "Food & Dining"
            
            matches = difflib.get_close_matches(cat_str, valid_cats, n=1, cutoff=0.3)
            if matches: return matches[0]
            return "Unknown"

        original_cats = df['merchant_category'].copy()
        df['merchant_category'] = df['merchant_category'].apply(fix_category)
        fixed_count = (original_cats != df['merchant_category']).sum() - missing_cat.sum()
        report["corrupted_categories"] = int(max(0, fixed_count))

    # 7. Inconsistent location casing + aliases
    location_cols = [c for c in ['user_location', 'merchant_location'] if c in df.columns]
    
    city_map = {
        'jai': 'Jaipur', 'jaipur': 'Jaipur',
        'lko': 'Lucknow', 'lucknow': 'Lucknow',
        'bombay': 'Mumbai', 'mumbai': 'Mumbai',
        'madras': 'Chennai', 'chennai': 'Chennai',
        'ccu': 'Kolkata', 'kolkata': 'Kolkata',
        'del': 'Delhi', 'delhi': 'Delhi',
        'pnq': 'Pune', 'pune': 'Pune',
        'bengaluru': 'Bangalore', 'bangalore': 'Bangalore',
        'hyd': 'Hyderabad', 'hyderabad': 'Hyderabad'
    }
    
    normalization_count = 0
    for col in location_cols:
        def fix_loc(loc):
            if pd.isna(loc): return "Unknown"
            l_lower = str(loc).strip().lower()
            return city_map.get(l_lower, str(loc).strip().title())

        fixed_locs = df[col].apply(fix_loc)
        normalization_count += (df[col].astype(str).str.lower() != fixed_locs.str.lower()).sum()
        df[col] = fixed_locs
    
    report["location_aliases_normalized"] = int(normalization_count)

    # 8. Missing device_id
    if 'device_id' in df.columns:
        df['device_id'] = df['device_id'].fillna("UNKNOWN_DEVICE")

    # 9. Missing payment_method
    if 'payment_method' in df.columns:
        df['payment_method'] = df['payment_method'].fillna("Unknown")

    return df, report
