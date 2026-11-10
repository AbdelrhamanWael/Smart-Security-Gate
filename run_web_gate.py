"""
run_web_gate.py - Launcher for Smart Security Gate Web System
Starts the FastAPI server on http://127.0.0.1:8000 and automatically opens the browser.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Set matplotlib cache to avoid permission warnings
os.environ["MPLCONFIGDIR"] = str(ROOT_DIR / "tmp")

if __name__ == "__main__":
    import uvicorn

    url = "http://127.0.0.1:8000"
    print("=" * 70)
    print("   SMART SECURITY GATE: DATA CENTER BIOMETRIC SYSTEM (WEB GUI)   ")
    print("=" * 70)
    print(f"\n[+] Starting FastAPI Web Server at: {url}")
    print("[+] Serving:")
    print("    - Biometric Verification Gate (Webcam, Microphone, Cascade Mode D)")
    print("    - Operational Dashboard & Interactive Google-Sheets Server Grid")
    print("    - Role-Based Access Control: Super Admin (Member 1) & Operator (Member 2)")
    print("\n[+] Press Ctrl+C in terminal to stop the server.\n")

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.2)
        print(f"[+] Opening {url} in your default browser...")
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
