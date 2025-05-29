@echo off
title Genre Predictor

:: Step 1: Install required dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Step 2: Start the Flask app
start cmd /k python app.py

:: Step 3: Wait for Flask to start
timeout /t 5 >nul

:: Step 4: Open the webpage in the default browser
start http://127.0.0.1:5000