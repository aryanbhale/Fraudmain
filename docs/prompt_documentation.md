# AI Prompt Documentation

This file serves as Deliverable 3 (D3)

**Engine:** Antigravity (Deepmind Agentic Coding Model)
**Initial Prompt Given:**
```text
MASTER PROMPT — FinTech Fraud Detection Dashboard
SIH Internal Hackathon 2026 · YTIET Bhivpuri
CONTEXT & OBJECTIVE
You are building a FinTech Fraud Detection Dashboard for a hackathon submission...
(Prompt defines 6 robust deliverables, strict tech stack, Python Pandas requirements...
and chart specifications).
```

**Implementation Strategy Followed:**
1. A Plan text artifact was generated aligning with the deliverables.
2. An isolated module structure `backend/cleaner.py`, `backend/features.py`, `backend/model.py`, and `backend/analyzer.py` was implemented iteratively over `pandas`.
3. The Flask application instance mapped REST POST endpoints `/api/analyze` handling `multipart/form-data` uploads with logic to read default datasets if empty.
4. An agnostic Chart.js visualization frontend built using HTML/CSS/JS alone was configured and placed in `frontend/index.html`.
