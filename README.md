<div align="center">
  <img src="https://img.icons8.com/color/96/000000/shield.png" width="80" alt="Secura Logo">
  <h1>🛡️ SECURA</h1>
  <p><strong>Next-Gen FinTech Fraud Detection Dashboard</strong></p>
  <p><i>Developed for SIH Internal Hackathon 2026 · YTIET Bhivpuri</i></p>

  <a href="https://aryanbhale.github.io/Fraudmain/">
    <img src="https://img.shields.io/badge/Live_Demo-Frontend-blue?style=for-the-badge&logo=github">
  </a>
  <a href="https://fraudmain.onrender.com/api/health">
    <img src="https://img.shields.io/badge/API_Status-Online-brightgreen?style=for-the-badge&logo=render">
  </a>
</div>

<br>

## 📜 Overview
**SECURA** is an end-to-end Machine Learning pipeline and interactive analytical dashboard designed to identify anomalous financial transactions and credit card fraud in real-time. It transforms raw, dirty financial data into actionable intelligence with transparent ML explainability, data quality audits, and enterprise-grade metrics.

---

## ✨ Key Features
- **🚀 Automated Data Cleaning Audit:** Automatically detects missing cells, invalid IPs, corrupted categories, and location aliases, presenting a "Before vs After" Data Quality report interactively.
- **🧠 Random Forest ML Inference:** Employs a weak-supervision heuristic engine to bootstrap ground truth data, training a Random Forest Classifier to identify anomalous patterns across 10 engineered features.
- **📊 Enterprise KPI Matrix:** Real-time tracking of *Precision*, *Recall*, *F1 Score*, and *Fraud Rate*—statistically bounded to counteract dataset duplication leakage and provide organic, realistic testing parameters.
- **🔍 Deep Explainability (XAI):** Mathematically traces the "why" behind every flagged transaction. Includes Global Feature Importance charts, ROC-AUC Curves, and a detailed False-Negative Breakdown panel.
- **📈 CSV Export Native:** Single-click generation of fully structured Fraud Incident Reports directly to your local machine.

---

## 🏗️ Architecture & Tech Stack

### Frontend (Client-Side)
- **HTML5 & Vanilla JavaScript:** Lightning-fast dom manipulation without heavy framework bloat.
- **Tailwind CSS:** Fully responsive, modern, "Glassmorphism" inspired UI.
- **Chart.js:** Fluid SVG canvas rendering for complex data topologies (Donut, Line, Bar, and ROC plots).

### Backend (Server-Side Pipeline)
- **Python 3.11:** Core execution environment.
- **Flask & Gunicorn:** High-concurrency RESTful API serving the ML inference engine over HTTPS via Render.
- **Pandas & NumPy:** In-memory vectorized DataFrame operations for data engineering (`cleaner.py`, `features.py`).
- **Scikit-Learn:** Core Machine Learning library for algorithm training (`RandomForestClassifier`) and threshold tuning.

---

## ⚙️ How to Run Locally

If you want to run the Python API and Frontend locally on your own machine:

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/aryanbhale/Fraudmain.git
   cd Fraudmain
   ```

2. **Start the Backend:**
   ```bash
   pip install -r requirements.txt
   python backend/app.py
   ```
   *The Flask API will start running on `http://127.0.0.1:5050`.*

3. **Start the Frontend:**
   * Open `frontend/index.html` and change `const API_URL` on line 454 to:
     `const API_URL = 'http://localhost:5050/api/analyze';`
   * Open `frontend/index.html` in your web browser (or use Live Server).

4. **Upload Data:**
   Use the `samplemain.csv` file provided in the root directory to test the engine.

---

## ☁️ Deployment Architecture
This project is natively configured for seamless, free-tier cloud deployment:
1. **Frontend:** Hosted statically on **GitHub Pages** for maximum global edge-network speed.
2. **Backend Engine:** Hosted dynamically on **Render.com** (via the `gunicorn app:app` start command configured on the `main` branch).
3. **CORS:** Python `Flask-CORS` is enabled globally, securely allowing cross-origin resource sharing between GitHub Pages and Render.

---

<div align="center">
  <p>Built with ❤️ by Hackathon Team SECURA</p>
</div>
