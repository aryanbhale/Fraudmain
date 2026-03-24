# FraudGuard — FinTech Fraud Detection Dashboard

## Live Demo
https://aryanbhale.github.io/Fraudmain/frontend/

## Tech Stack
Python · Flask · scikit-learn · pandas · Vanilla JS · Chart.js · TailwindCSS

## How to Run Locally

1. Setup the backend:
```bash
cd backend
pip install -r requirements.txt
python app.py
```

2. Open `frontend/index.html` in your browser. (Or visit the endpoint if deployed separately).

## Architecture
See `docs/architecture_diagram.png` for a flow visualization. 
The backend leverages Pandas mapping capabilities to parse unstandardized CSV logs into reliable ML datasets. Following data cleaning, rule-based pseudo-labeling evaluates transactions across 10 independent indicators providing ground-truth proxies to train a secondary real-time `RandomForestClassifier`.

## Team
- Prompt Engineer / Solo Developer
