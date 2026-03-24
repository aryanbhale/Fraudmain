# SECURA: Development & Problem-Solving Log

This document serves as an archive of the critical architectural discussions, prompts, and problem-solving milestones during the development of the SECURA Fraud Detection System for the SIH 2026 Hackathon.

---

### 🟢 Milestone 1: Data Auditing & Pipeline Architecture
**Team Prompt:**
> "We have a large dataset of raw financial transactions with missing cells, corrupted categories, and invalid IP addresses. We need to build a robust Python pipeline that cleans this data before it reaches the ML engine, and we need to visually prove to the judges exactly what data was cleaned."

**Engineering Solution:**
> Building `cleaner.py`. We implemented a vectorized Pandas approach to handle the missing cells and invalid formats dynamically. To prove the data quality improvements, we exposed a "Before vs After" Data Quality dictionary in our API response. This allows the frontend dashboard to render a real-time table showing exactly how many duplicates were dropped, IPs sanitized, and null-values interpolated.

---

### 🟢 Milestone 2: Advanced Feature Engineering
**Team Prompt:**
> "Raw financial data isn't enough for the Random Forest to catch sophisticated fraud. How do we engineer new features to expose credit card cloning, rapid credential testing, and impossible travel scenarios?"

**Engineering Solution:**
> In `features.py`, we designed 10 custom heuristic features based on FinTech industry standards:
> 1. `amount_zscore`: Tracks standard deviations from a user's historical spending.
> 2. `user_txn_velocity_7d`: Uses a rolling 7-day window to catch rapid credential stuffing.
> 3. `is_cross_city`: Checks if the merchant coordinates drastically differ from billing addresses.
> 4. `is_invalid_ip`: Flags proxy servers and unroutable 0.0.x network origins.

---

### 🟢 Milestone 3: Tackling Extreme Class Imbalance (Low Recall)
**Team Prompt:**
> "Our model is struggling with False Negatives. Because fraud is so rare (less than 3% in real-world distributions), the Random Forest is heavily biased toward predicting 'Legitimate' to maximize raw accuracy. Our F1 Score and Recall are suffering. How do we fix this organically?"

**Engineering Solution:**
> This is a classic problem in anomalous detection. We modified `model.py` by natively injecting `class_weight='balanced'` into the `RandomForestClassifier` initialization. Furthermore, we abandoned the standard 0.50 `.predict()` threshold. Instead, we manually extracted the probability matrix using `.predict_proba()` and algorithmically smoothed the threshold down (e.g., to 0.40) to aggressively prioritize **Recall** over Precision—ensuring extreme-risk outliers aren't missed by the system.

---

### 🟢 Milestone 4: Explainable AI & Model Diagnostics
**Team Prompt:**
> "The deliverables require us to have a 'Visual Confusion Matrix' and clearly show 'Where do the false negatives mostly come from?' How do we make the Random Forest's decisions completely mathematically transparent to the judges?"

**Engineering Solution:**
> We aggressively upgraded the API output in `analyzer.py`. 
> * **Global Explainability**: We extracted the Gini impurity tracking (`model.feature_importances_`) to build a bar chart explaining exactly which features weigh heaviest on the model's brain.
> * **Model Robustness**: We utilized `roc_curve` and `auc` from Scikit-Learn to map the True Positive Rate against the False Positive Rate, plotting an ROC Curve on our frontend.
> * **FN Deep-Dive**: We ran a Pandas mask over the test results where `y_test == 1` and `y_pred == 0` to algorithmically map exactly which Device Subtypes and Merchant Categories our model was "blind" to.

---

### 🟢 Milestone 5: Full-Stack Cloud Deployment Bugs
**Team Prompt:**
> "We successfully built the frontend dashboard, but when hosting it on GitHub Pages, we are getting a 'Failed to Fetch' error because the JS is trying to hit `http://localhost:5050`."

**Engineering Solution:**
> GitHub Pages cannot host dynamic Python processes, and querying Localhost from a secure HTTPS browser triggers a Mixed-Content security block. 
> 
> To achieve a true Enterprise deployment, we decoupled the architecture:
> 1. We injected `Flask-CORS` into `app.py` to securely permit cross-origin requests.
> 2. We deployed the ML Backend to **Render.com** utilizing a `gunicorn` WSGI binding.
> 3. We updated the statically hosted GitHub Pages `index.html` to target the live Render cloud cluster.
> 
> The system now processes heavy local CSV files by piping them seamlessly over the cloud into our inference engine.

---
*Log generated to chronicle engineering decisions, obstacle resolution, and systemic problem-solving.*
