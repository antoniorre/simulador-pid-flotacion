@echo off
setlocal
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
endlocal
