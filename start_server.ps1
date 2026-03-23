Write-Host "Installing required Python dependencies..." -ForegroundColor Cyan
pip install flask==3.0.3 flask-cors==4.0.1 pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.1 python-dateutil==2.9.0.post0

Write-Host "Starting Flask Backend Server on http://localhost:5050" -ForegroundColor Green
python backend\app.py
