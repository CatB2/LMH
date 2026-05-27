"""Start the app server. Run via python.exe _start_server.py"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvicorn
from app.main import app
uvicorn.run(app, host="0.0.0.0", port=9901, log_level="info")
