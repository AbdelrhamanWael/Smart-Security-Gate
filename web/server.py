"""
server.py - FastAPI Web Server for Smart Security Gate
Provides:
  - Biometric Gate Verification API (Face, Voice, Fingerprint, Cascade Mode D).
  - Web UI serving (HTML / CSS / JS).
  - Role-Based Access Control (RBAC):
      * Member 1 -> Super Admin (Core Vault, Full Audit Logs, Sheet Edit)
      * Member 2 -> Operator (Gate Control, Telemetry Monitor, Sheet Read-Only)
  - Interactive Google-Sheets-style Server Inventory API.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import Biometric Engines
from src.face_models import face_score
from src.fingerprint import fingerprint_score
from src.fusion import fusion_score
from src.hierarchy import hierarchical_verify
from src.voice_verification import voice_score

app = FastAPI(title="Smart Security Gate API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_FILE = LOGS_DIR / "audit_log.csv"

# Global State
SESSION_TOKENS: Dict[str, Dict[str, Any]] = {}
GATE_STATE = {
    "perimeter_gate": "LOCKED",
    "vault_door": "LOCKED",
    "emergency_mode": False,
    "last_access_by": None,
    "last_access_time": None,
}

# In-Memory Server Inventory (Google Sheets style)
SERVER_INVENTORY: List[Dict[str, Any]] = [
    {
        "id": "SRV-101",
        "rack": "Rack-A01",
        "hostname": "dc-auth-primary.prod",
        "ip": "10.0.1.10",
        "role": "Biometric Authentication Server",
        "status": "OPTIMAL",
        "power_kw": 1.45,
        "temp_c": 22.4,
        "notes": "Cascade Mode D Inference Node",
    },
    {
        "id": "SRV-102",
        "rack": "Rack-A02",
        "hostname": "dc-vault-core.prod",
        "ip": "10.0.1.15",
        "role": "Security Vault HSM KeyStore",
        "status": "OPTIMAL",
        "power_kw": 2.10,
        "temp_c": 21.8,
        "notes": "Restricted Super-Admin Zone",
    },
    {
        "id": "SRV-103",
        "rack": "Rack-B01",
        "hostname": "dc-db-cluster-01.prod",
        "ip": "10.0.2.21",
        "role": "PostgreSQL Main Database",
        "status": "OPTIMAL",
        "power_kw": 3.80,
        "temp_c": 24.1,
        "notes": "Daily automated snapshot at 03:00",
    },
    {
        "id": "SRV-104",
        "rack": "Rack-B02",
        "hostname": "dc-db-replica-02.prod",
        "ip": "10.0.2.22",
        "role": "Database Streaming Replica",
        "status": "OPTIMAL",
        "power_kw": 3.65,
        "temp_c": 23.9,
        "notes": "Synchronous streaming active",
    },
    {
        "id": "SRV-105",
        "rack": "Rack-C01",
        "hostname": "dc-ai-infer-gpu.prod",
        "ip": "10.0.3.50",
        "role": "Vision & Audio Feature Node",
        "status": "OPTIMAL",
        "power_kw": 5.20,
        "temp_c": 26.5,
        "notes": "Liquid cooled rack block",
    },
    {
        "id": "SRV-106",
        "rack": "Rack-C02",
        "hostname": "dc-edge-gateway-01.prod",
        "ip": "10.0.3.55",
        "role": "CCTV & Sensor Ingestion",
        "status": "OPTIMAL",
        "power_kw": 1.90,
        "temp_c": 22.0,
        "notes": "Perimeter camera feeds",
    },
    {
        "id": "SRV-107",
        "rack": "Rack-D01",
        "hostname": "dc-hvac-controller.prod",
        "ip": "10.0.4.11",
        "role": "Cooling & Power Telemetry",
        "status": "MAINTENANCE",
        "power_kw": 0.85,
        "temp_c": 20.5,
        "notes": "Routine filter inspection scheduled",
    },
    {
        "id": "SRV-108",
        "rack": "Rack-D02",
        "hostname": "dc-backup-tape-01.prod",
        "ip": "10.0.4.88",
        "role": "Cold Storage Archival Unit",
        "status": "OPTIMAL",
        "power_kw": 1.20,
        "temp_c": 19.8,
        "notes": "Air-gapped offline tape backup",
    },
]

# Ensure Audit Log CSV exists with headers
if not AUDIT_LOG_FILE.exists():
    with open(AUDIT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "claimed_user", "verified", "level_reached",
            "face_score", "voice_score", "fusion_score", "fp_score",
            "decision", "role", "time_ms"
        ])


def log_audit_event(
    claimed_user: str,
    verified: bool,
    level_reached: int,
    scores: Dict[str, Optional[float]],
    decision: str,
    role: str,
    time_ms: float,
) -> None:
    """Appends an event row to logs/audit_log.csv."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(AUDIT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            claimed_user,
            verified,
            level_reached,
            f"{scores.get('face', 0.0):.4f}" if scores.get("face") is not None else "N/A",
            f"{scores.get('voice', 0.0):.4f}" if scores.get("voice") is not None else "N/A",
            f"{scores.get('fusion', 0.0):.4f}" if scores.get("fusion") is not None else "N/A",
            f"{scores.get('fingerprint', 0.0):.4f}" if scores.get("fingerprint") is not None else "N/A",
            decision,
            role,
            f"{time_ms:.1f}",
        ])


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decodes image bytes (JPEG, PNG, WebP) to OpenCV BGR numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_audio_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decodes audio bytes (WAV, OGG, FLAC, WebM) to 1D float32 numpy array at 16kHz."""
    try:
        audio_io = io.BytesIO(data)
        audio, sr = sf.read(audio_io, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            num_target_samples = int(round(len(audio) * float(16000) / float(sr)))
            audio = scipy.signal.resample(audio, num_target_samples).astype(np.float32)
        return audio
    except Exception as e:
        print(f"Error decoding audio bytes: {e}")
        return None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def get_presets():
    """Returns available testing presets for rapid one-click demoing."""
    return {
        "member1": {
            "name": "Member 1 (Super Admin)",
            "role": "Super Admin",
            "face_sample": "data/face/member1/session1/img_000.jpg",
            "voice_sample": "data/voice/member1/session1/passphrase_01.wav",
            "fp_sample": "data/fingerprint/member1/session1/img_001.jpg",
        },
        "member2": {
            "name": "Member 2 (Operator)",
            "role": "Operator",
            "face_sample": "data/face/member2/session1/img_000.jpg",
            "voice_sample": "data/voice/member2/session1/passphrase_01.wav",
            "fp_sample": "data/fingerprint/member2/session1/img_001.jpg",
        },
        "impostor": {
            "name": "Intruder / Impostor",
            "role": "Unverified Intruder",
            "face_sample": "data/face/impostors/Ariel_Sharon/session1/img_000.jpg",
            "voice_sample": "data/voice/impostor1/session1/passphrase_01.wav",
            "fp_sample": "data/fingerprint/impostors/finger_104/session1/img_001.jpg",
        },
    }


@app.get("/api/dataset/files")
def get_dataset_files():
    """Returns curated lists of files from data/ directory for dropdown selection."""
    import glob

    def scan_curated(pattern, max_items=10):
        found = sorted(glob.glob(pattern, recursive=True))
        return [f.replace("\\", "/") for f in found[:max_items]]

    return {
        "faces": [
            {"label": "Member 1 - Genuine (Session 1)", "path": "data/face/member1/session1/img_000.jpg", "category": "genuine_m1"},
            {"label": "Member 1 - Genuine (Session 2)", "path": "data/face/member1/session2/img_000.jpg", "category": "genuine_m1"},
            {"label": "Member 2 - Genuine (Session 1)", "path": "data/face/member2/session1/img_000.jpg", "category": "genuine_m2"},
            {"label": "Member 2 - Genuine (Session 2)", "path": "data/face/member2/session2/img_000.jpg", "category": "genuine_m2"},
            {"label": "Impostor Face 1 (Ariel Sharon)", "path": "data/face/impostors/Ariel_Sharon/session1/img_000.jpg", "category": "impostor"},
            {"label": "Impostor Face 2 (Colin Powell)", "path": "data/face/impostors/Colin_Powell/session1/img_000.jpg", "category": "impostor"},
        ],
        "voices": [
            {"label": "Member 1 - Genuine (Passphrase 01)", "path": "data/voice/member1/session1/passphrase_01.wav", "category": "genuine_m1"},
            {"label": "Member 1 - Genuine (Passphrase 02)", "path": "data/voice/member1/session2/passphrase_01.wav", "category": "genuine_m1"},
            {"label": "Member 2 - Genuine (Passphrase 01)", "path": "data/voice/member2/session1/passphrase_01.wav", "category": "genuine_m2"},
            {"label": "Member 2 - Genuine (Passphrase 02)", "path": "data/voice/member2/session2/passphrase_01.wav", "category": "genuine_m2"},
            {"label": "Impostor Voice 1 (Unseen Attacker)", "path": "data/voice/impostor1/session1/passphrase_01.wav", "category": "impostor"},
            {"label": "Impostor Voice 2 (Unseen Attacker)", "path": "data/voice/impostor2/session1/passphrase_01.wav", "category": "impostor"},
        ],
        "fingerprints": [
            {"label": "Member 1 - Genuine Ridge (Session 1)", "path": "data/fingerprint/member1/session1/img_001.jpg", "category": "genuine_m1"},
            {"label": "Member 1 - Genuine Ridge (Session 2)", "path": "data/fingerprint/member1/session2/img_001.jpg", "category": "genuine_m1"},
            {"label": "Member 2 - Genuine Ridge (Session 1)", "path": "data/fingerprint/member2/session1/img_001.jpg", "category": "genuine_m2"},
            {"label": "Member 2 - Genuine Ridge (Session 2)", "path": "data/fingerprint/member2/session2/img_001.jpg", "category": "genuine_m2"},
            {"label": "Impostor Fingerprint 1", "path": "data/fingerprint/impostors/finger_104/session1/img_001.jpg", "category": "impostor"},
            {"label": "Impostor Fingerprint 2", "path": "data/fingerprint/impostors/finger_105/session1/img_001.jpg", "category": "impostor"},
        ],
    }


@app.post("/api/verify/face")
async def api_verify_face(request: Request):
    """Verifies a single face attempt (Level 1 / Mode A)."""
    t0 = time.perf_counter()
    form = await request.form()
    user_id = str(form.get("user_id", "member1"))
    preset_path = form.get("preset_path")
    image_file = form.get("image")

    img = None
    if image_file and hasattr(image_file, "read"):
        contents = await image_file.read()
        if contents:
            img = decode_image_bytes(contents)
    if img is None and preset_path and (ROOT_DIR / str(preset_path)).exists():
        img = cv2.imread(str(ROOT_DIR / str(preset_path)), cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode face image")

    score = face_score(img, user_id=user_id, model_type="svm")
    threshold = 0.50
    verified = bool(score >= threshold)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "modality": "face",
        "user_id": user_id,
        "score": round(score, 4),
        "score_percent": round(score * 100, 1),
        "threshold": threshold,
        "verified": verified,
        "time_ms": round(dt_ms, 2),
    }


@app.post("/api/verify/voice")
async def api_verify_voice(request: Request):
    """Verifies a single voice attempt (Level 2 / Mode B)."""
    t0 = time.perf_counter()
    form = await request.form()
    user_id = str(form.get("user_id", "member1"))
    preset_path = form.get("preset_path")
    audio_file = form.get("audio")

    target_audio = None
    if audio_file and hasattr(audio_file, "read"):
        contents = await audio_file.read()
        if contents:
            target_audio = decode_audio_bytes(contents)
    if target_audio is None and preset_path and (ROOT_DIR / str(preset_path)).exists():
        target_audio = str(ROOT_DIR / str(preset_path))

    if target_audio is None:
        raise HTTPException(status_code=400, detail="No audio provided or file not found")

    score = voice_score(target_audio, user_id=user_id)
    threshold = 0.50
    verified = bool(score >= threshold)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "modality": "voice",
        "user_id": user_id,
        "score": round(score, 4),
        "score_percent": round(score * 100, 1),
        "threshold": threshold,
        "verified": verified,
        "time_ms": round(dt_ms, 2),
    }


@app.post("/api/verify/fingerprint")
async def api_verify_fingerprint(request: Request):
    """Verifies a single fingerprint attempt (Level 3 / Bonus 4)."""
    t0 = time.perf_counter()
    form = await request.form()
    user_id = str(form.get("user_id", "member1"))
    preset_path = form.get("preset_path")
    image_file = form.get("image")

    img = None
    if image_file and hasattr(image_file, "read"):
        contents = await image_file.read()
        if contents:
            img = decode_image_bytes(contents)
    if img is None and preset_path and (ROOT_DIR / str(preset_path)).exists():
        img = str(ROOT_DIR / str(preset_path))

    if img is None:
        raise HTTPException(status_code=400, detail="No fingerprint image provided or file not found")

    score = fingerprint_score(img, user_id=user_id, model_type="svm")
    threshold = 0.20
    verified = bool(score >= threshold)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "modality": "fingerprint",
        "user_id": user_id,
        "score": round(score, 4),
        "score_percent": round(score * 100, 1),
        "threshold": threshold,
        "verified": verified,
        "time_ms": round(dt_ms, 2),
    }


@app.post("/api/verify/cascade")
async def api_verify_cascade(request: Request):
    """
    Executes the full 3-level Hierarchical Cascade Pipeline (Mode D):
      Level 1: Face -> Level 2: Voice+Fusion -> Level 3: Fingerprint.
    Returns complete breakdown, timing per stage, decisions, and session token.
    """
    t_start = time.perf_counter()
    form = await request.form()

    user_id = str(form.get("user_id", "member1"))
    face_preset = form.get("face_preset")
    voice_preset = form.get("voice_preset")
    fp_preset = form.get("fp_preset")

    face_image = form.get("face_image")
    voice_audio = form.get("voice_audio")
    fp_image = form.get("fp_image")

    # 1. Resolve Face
    face_target = None
    if face_image and hasattr(face_image, "read"):
        contents = await face_image.read()
        if contents:
            face_target = decode_image_bytes(contents)
    if face_target is None and face_preset and (ROOT_DIR / str(face_preset)).exists():
        face_target = cv2.imread(str(ROOT_DIR / str(face_preset)), cv2.IMREAD_COLOR)

    if face_target is None:
        raise HTTPException(status_code=400, detail="Face input missing or unreadable")

    # 2. Resolve Voice
    voice_target = None
    if voice_audio and hasattr(voice_audio, "read"):
        contents = await voice_audio.read()
        if contents:
            voice_target = decode_audio_bytes(contents)
    if voice_target is None and voice_preset and (ROOT_DIR / str(voice_preset)).exists():
        voice_target = str(ROOT_DIR / str(voice_preset))

    # 3. Resolve Fingerprint
    fp_target = None
    if fp_image and hasattr(fp_image, "read"):
        contents = await fp_image.read()
        if contents:
            fp_target = decode_image_bytes(contents)
    if fp_target is None and fp_preset and (ROOT_DIR / str(fp_preset)).exists():
        fp_target = str(ROOT_DIR / str(fp_preset))

    # Run Hierarchical Cascade
    result = hierarchical_verify(
        user_id=user_id,
        face_img=face_target,
        audio=voice_target,
        fingerprint_img=fp_target,
        require_fingerprint=False,
    )

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    # Determine role & permissions based on verification
    is_verified = result["decision"]
    level_reached = result["level_reached"]

    if is_verified:
        if user_id == "member1":
            role = "Super Admin"
            role_key = "super_admin"
        elif user_id == "member2":
            role = "Data Center Operator"
            role_key = "operator"
        else:
            role = "Auditor"
            role_key = "auditor"
    else:
        role = "Unverified Intruder"
        role_key = "intruder"

    # Create session token if verified
    session_token = None
    if is_verified:
        session_token = f"gate_tok_{uuid.uuid4().hex[:16]}"
        SESSION_TOKENS[session_token] = {
            "user_id": user_id,
            "role": role,
            "role_key": role_key,
            "level_reached": level_reached,
            "created_at": time.time(),
        }
        GATE_STATE["perimeter_gate"] = "UNLOCKED"
        GATE_STATE["last_access_by"] = f"{user_id} ({role})"
        GATE_STATE["last_access_time"] = time.strftime("%H:%M:%S")

    # Log to audit trail
    log_audit_event(
        claimed_user=user_id,
        verified=is_verified,
        level_reached=level_reached,
        scores=result["scores"],
        decision="ACCESS GRANTED" if is_verified else "ACCESS DENIED",
        role=role,
        time_ms=total_time_ms,
    )

    timing_dict = result.get("timing_ms") or result.get("time_per_stage") or {}

    return {
        "decision": is_verified,
        "decision_str": "ACCESS GRANTED" if is_verified else "ACCESS DENIED",
        "level_reached": level_reached,
        "stopped_at": result["stopped_at"],
        "scores": result["scores"],
        "thresholds": result["thresholds"],
        "time_per_stage": timing_dict,
        "total_time_ms": round(total_time_ms, 2),
        "user_id": user_id,
        "role": role,
        "role_key": role_key,
        "session_token": session_token,
    }


@app.get("/api/dashboard/data")
def get_dashboard_data(token: Optional[str] = None):
    """Returns dashboard data, telemetry, server inventory sheet, and audit log."""
    session = SESSION_TOKENS.get(token)
    if not session:
        # Default guest/unauthenticated preview or fallback
        user_info = {
            "authenticated": False,
            "user_id": None,
            "role": "Guest / Intruder",
            "role_key": "guest",
            "can_edit_sheet": False,
            "can_access_vault": False,
        }
    else:
        can_edit = session["role_key"] == "super_admin"
        can_vault = session["role_key"] == "super_admin"
        user_info = {
            "authenticated": True,
            "user_id": session["user_id"],
            "role": session["role"],
            "role_key": session["role_key"],
            "level_reached": session["level_reached"],
            "can_edit_sheet": can_edit,
            "can_access_vault": can_vault,
        }

    # Read recent 10 audit logs
    audit_rows = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                audit_rows = list(reader)[-15:]
                audit_rows.reverse()
        except Exception as e:
            print(f"Error reading audit log: {e}")

    return {
        "user": user_info,
        "gate_state": GATE_STATE,
        "telemetry": {
            "temperature_c": 21.8,
            "humidity_pct": 42.5,
            "total_power_kw": 18.25,
            "active_nodes": len(SERVER_INVENTORY),
            "cooling_load_pct": 58.0,
            "ups_battery_pct": 99.4,
        },
        "inventory_sheet": SERVER_INVENTORY,
        "recent_audit_logs": audit_rows,
    }


@app.post("/api/dashboard/sheet/update")
def update_inventory_row(request_data: Dict[str, Any], token: Optional[str] = None):
    """Updates a row in the Google-Sheets-style Server Inventory table (Super Admin only)."""
    session = SESSION_TOKENS.get(token)
    if not session or session.get("role_key") != "super_admin":
        raise HTTPException(
            status_code=403,
            detail="Access Denied: Only Super Admin (Member 1) has permission to edit Server Inventory.",
        )

    row_id = request_data.get("id")
    for row in SERVER_INVENTORY:
        if row["id"] == row_id:
            row["rack"] = request_data.get("rack", row["rack"])
            row["hostname"] = request_data.get("hostname", row["hostname"])
            row["ip"] = request_data.get("ip", row["ip"])
            row["role"] = request_data.get("role", row["role"])
            row["status"] = request_data.get("status", row["status"])
            row["notes"] = request_data.get("notes", row["notes"])
            return {"status": "success", "message": f"Server {row_id} updated successfully.", "row": row}

    raise HTTPException(status_code=404, detail=f"Server {row_id} not found in inventory.")


@app.post("/api/dashboard/gate-action")
def execute_gate_action(action: str = Form(...), token: Optional[str] = None):
    """Executes physical gate or vault commands enforcing RBAC."""
    session = SESSION_TOKENS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized: Please verify identity first.")

    role_key = session.get("role_key")
    user_id = session.get("user_id")

    if action == "perimeter_unlock":
        GATE_STATE["perimeter_gate"] = "UNLOCKED"
        return {"status": "SUCCESS", "message": "Perimeter Security Gate Unlocked."}
    elif action == "perimeter_lock":
        GATE_STATE["perimeter_gate"] = "LOCKED"
        return {"status": "SUCCESS", "message": "Perimeter Security Gate Locked."}
    elif action == "vault_unlock":
        if role_key != "super_admin":
            raise HTTPException(status_code=403, detail="Forbidden: Core Server Vault requires Super Admin authorization.")
        GATE_STATE["vault_door"] = "UNLOCKED"
        return {"status": "SUCCESS", "message": "High-Security Vault Door Unlocked."}
    elif action == "vault_lock":
        GATE_STATE["vault_door"] = "LOCKED"
        return {"status": "SUCCESS", "message": "High-Security Vault Door Locked."}
    elif action == "emergency_lockdown":
        if role_key != "super_admin":
            raise HTTPException(status_code=403, detail="Forbidden: Emergency Lockdown can only be initiated by Super Admin.")
        GATE_STATE["perimeter_gate"] = "LOCKED"
        GATE_STATE["vault_door"] = "LOCKED"
        GATE_STATE["emergency_mode"] = True
        return {"status": "ALERT", "message": "EMERGENCY LOCKDOWN ACTIVATED! All doors secured."}
    elif action == "reset_emergency":
        GATE_STATE["emergency_mode"] = False
        return {"status": "SUCCESS", "message": "Emergency status cleared."}
    else:
        raise HTTPException(status_code=400, detail="Unknown gate action.")


# Mount Static Files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(ROOT_DIR / "data")), name="data")

@app.get("/")
def serve_index():
    """Serves the main Biometric Security Gate application."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)

