@echo off
title Deploying NightVision AI Studio
echo =========================================================
echo 🚀 NightVision AI — One-Click Production Deployment
echo =========================================================

cd /d "%~dp0"

echo [1/3] Verifying Python Dependencies...
python -m pip install -r backend\requirements.txt

echo [2/3] Verifying Frontend Build...
cd frontend
call npm install
call npm run build
cd ..

echo [3/3] Launching Production Server...
python start_production.py

pause
