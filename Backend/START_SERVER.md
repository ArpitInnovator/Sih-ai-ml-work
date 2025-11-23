# How to Start the Server

## Quick Start

1. **Open PowerShell in the Backend folder**

2. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   You should see `(venv)` at the start of your prompt.

3. **Start the server:**
   ```powershell
   .\start_server.ps1
   ```
   
   OR manually:
   ```powershell
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Verify it's running:**
   - Open: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Troubleshooting

### If you see "ModuleNotFoundError: No module named 'fastapi'"
- Make sure the virtual environment is activated (you should see `(venv)` in your prompt)
- Install dependencies: `pip install -r requirements.txt`

### If port 8000 is already in use
- Find the process: `netstat -ano | findstr :8000`
- Kill it: `taskkill /PID <process_id> /F`
- Or use a different port: Change `--port 8000` to `--port 8001`

### If the server starts but pages don't load
- Check if the server is actually running (look for "Uvicorn running on http://127.0.0.1:8000")
- Try accessing: http://127.0.0.1:8000 (instead of localhost)
- Check Windows Firewall settings

