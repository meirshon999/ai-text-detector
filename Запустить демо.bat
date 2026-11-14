@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Demo: Human or ChatGPT?  (Streamlit)
echo   The browser opens by itself in about 10-20 seconds:
echo       http://localhost:8501
echo   Do NOT close this window while you use the demo.
echo ============================================================
python -c "import streamlit" 2>nul || (echo Streamlit is not installed. Run:  pip install -r requirements.txt & pause & exit /b 1)
if not exist "models\tfidf_logreg.joblib" (echo Trained models not found. Run:  python scripts\run_experiments.py & pause & exit /b 1)
start "" cmd /c "timeout /t 8 >nul & start http://localhost:8501"
python -m streamlit run demo\streamlit_app.py --server.port 8501 --server.address localhost --server.headless true --browser.gatherUsageStats false
pause
