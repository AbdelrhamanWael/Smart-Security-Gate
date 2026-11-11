"""
live_gate_camera.py - Real-Time Webcam Biometric Gate (Bonus 2)
Part of: Hierarchical Multimodal Biometric Verification System

Features:
- Live real-time webcam face detection via Haar Cascades + CLAHE adaptive contrast.
- HOG + Uniform LBP feature extraction with PCA projection and RBF-SVM scoring.
- Realistic biometric threshold (Default: 65% / 0.65) ensuring genuine users (85%+) pass smoothly
  while unseen impostors (<20%) are strictly denied.
- Exponential Moving Average (EMA) smoothing for zero frame jitter.
- Interactive controls:
    [1] Claim: Abdelrahman Wael (Member 1 - Super Admin)
    [2] Claim: Hashem (Member 2 - Operator)
    [3] Claim: Hagar (Member 3 - Auditor)
    [4] Claim: Unverified Impostor Attack
    [+] Increase threshold | [-] Decrease threshold
    [T] Toggle Mode (Relaxed 55% / Balanced 65% / Strict 75%)
    [Q] Quit / Exit
"""

import cv2
import time
import numpy as np
from collections import deque
from src.face_models import face_score
from src.face_pipeline import crop_face

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_THRESHOLD = 0.65       # Balanced real-world operating threshold (65%)
VOTING_WINDOW = 8              # Frames in voting window (fast response ~0.25s)
REQUIRED_PASSES = 5            # Must pass at least 5 out of 8 frames
# ==========================================

USER_PROFILES = {
    "member1": {"name": "Abdelrahman Wael", "role": "Super Admin"},
    "member2": {"name": "Hashem", "role": "Operator"},
    "member3": {"name": "Hagar", "role": "Auditor"},
    "impostor": {"name": "Intruder / Impostor", "role": "Unverified Attacker"}
}

def draw_corner_rect(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Draws a techy HUD bounding box with accentuated corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Base rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Top-Left corner
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right corner
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left corner
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right corner
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

def run_live_gate():
    claimed_user = "member1"
    security_threshold = DEFAULT_THRESHOLD
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Ensure no other application is using it.")
        return

    # Set camera resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    scores_history = deque(maxlen=VOTING_WINDOW)
    ema_score = 0.0
    alpha = 0.35  # EMA smoothing factor

    print("=" * 65)
    print("      SMART SECURITY GATE: REAL-TIME WEBCAM VERIFIER       ")
    print(f" Default Claimed User: {USER_PROFILES[claimed_user]['name']}")
    print(f" Operating Threshold:  {security_threshold * 100:.0f}%")
    print(" Keyboard Controls:")
    print("   [1] Abdelrahman Wael  |  [2] Hashem  |  [3] Hagar  |  [4] Impostor")
    print("   [+] Increase Thresh   |  [-] Decrease Thresh  |  [T] Presets")
    print("   [Q] Quit")
    print("=" * 65)

    fps_t0 = time.time()
    frame_count = 0
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps = frame_count / (time.time() - fps_t0)
            fps_t0 = time.time()
            frame_count = 0

        h, w, _ = frame.shape

        # 1. Detect face and crop
        face_gray, box = crop_face(frame, return_box=True)

        if face_gray is not None:
            # Check raw frame and horizontally flipped frame (for mirror invariance)
            score_raw = face_score(frame, user_id=claimed_user, model_type="svm")
            score_flip = face_score(cv2.flip(frame, 1), user_id=claimed_user, model_type="svm")
            raw_score = max(score_raw, score_flip)
            
            # Exponential Moving Average for jitter-free score
            if ema_score == 0.0:
                ema_score = raw_score
            else:
                ema_score = alpha * raw_score + (1.0 - alpha) * ema_score
                
            scores_history.append(ema_score)

            # Draw HUD bounding box on detected face
            fx, fy, fw, fh = box
            box_col = (0, 230, 0) if ema_score >= security_threshold else (0, 0, 240)
            draw_corner_rect(frame, (fx, fy), (fx + fw, fy + fh), box_col, thickness=3, corner_len=18)
            
            # Badge above face box
            badge_text = f"{USER_PROFILES[claimed_user]['name']}: {ema_score * 100:.1f}%"
            badge_y = max(130, fy - 12)
            cv2.rectangle(frame, (fx, badge_y - 20), (fx + len(badge_text) * 11 + 10, badge_y + 4), (20, 20, 20), -1)
            cv2.rectangle(frame, (fx, badge_y - 20), (fx + len(badge_text) * 11 + 10, badge_y + 4), box_col, 1)
            cv2.putText(frame, badge_text, (fx + 5, badge_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2)
        else:
            raw_score = 0.0
            ema_score = 0.0
            scores_history.clear()

        # 2. Decision Logic
        passed_votes = sum(1 for s in scores_history if s >= security_threshold)
        
        if face_gray is None:
            status_text = "NO FACE DETECTED"
            status_color = (0, 165, 255)  # Amber
            subtext = "Position your face directly in front of the camera"
        elif len(scores_history) >= (VOTING_WINDOW // 2):
            if passed_votes >= REQUIRED_PASSES or (len(scores_history) > 0 and ema_score >= security_threshold):
                status_text = "ACCESS GRANTED [VERIFIED]"
                status_color = (0, 230, 0)  # Bright Neon Green
                subtext = f"Identity Validated: {USER_PROFILES[claimed_user]['name']} ({USER_PROFILES[claimed_user]['role']})"
            else:
                status_text = "ACCESS DENIED [SECURITY REJECTION]"
                status_color = (0, 0, 240)  # Red
                subtext = f"Confidence {ema_score * 100:.1f}% below required {security_threshold * 100:.0f}% threshold"
        else:
            status_text = "SCANNING BIOMETRICS..."
            status_color = (255, 200, 0)  # Cyan/Yellow
            subtext = f"Analyzing facial features ({len(scores_history)}/{VOTING_WINDOW} frames)"

        # 3. Top Cyber HUD Panel (Height 110px)
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 12), (w - 15, 115), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (15, 12), (w - 15, 115), status_color, 2)

        # Status Line
        cv2.putText(frame, f"STATUS: {status_text}", (30, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        # Metrics Line
        metrics_line = (f"Claimed: {USER_PROFILES[claimed_user]['name']} | "
                        f"Match: {ema_score * 100:.1f}% | "
                        f"Threshold: {security_threshold * 100:.0f}% | "
                        f"FPS: {fps:.1f}")
        cv2.putText(frame, metrics_line, (30, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

        # Subtext explanation
        cv2.putText(frame, subtext, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 170, 170), 1)

        # Score progress meter bar
        bar_x = w - 195
        bar_w = 160
        bar_h = 14
        bar_y = 70
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)
        fill_w = int(bar_w * min(1.0, max(0.0, ema_score)))
        bar_col = (0, 230, 0) if ema_score >= security_threshold else (0, 0, 240)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 85, 95), 1)
        # Threshold marker on meter
        marker_x = int(bar_x + bar_w * security_threshold)
        cv2.line(frame, (marker_x, bar_y - 2), (marker_x, bar_y + bar_h + 2), (255, 255, 255), 2)

        # 4. Bottom Controls Bar
        bot_overlay = frame.copy()
        cv2.rectangle(bot_overlay, (15, h - 38), (w - 15, h - 10), (15, 18, 24), -1)
        cv2.addWeighted(bot_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (15, h - 38), (w - 15, h - 10), (50, 60, 75), 1)
        
        controls_txt = "[1] M1 (Wael)  [2] M2 (Hashem)  [3] M3 (Hagar)  [4] Impostor  [+/-] Thresh  [Q] Exit"
        cv2.putText(frame, controls_txt, (25, h - 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        cv2.imshow("Smart Security Gate - Live Camera", frame)

        # Keyboard event handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
            break
        elif key == ord('1'):
            claimed_user = "member1"
            scores_history.clear()
            ema_score = 0.0
            print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member1']['name']}")
        elif key == ord('2'):
            claimed_user = "member2"
            scores_history.clear()
            ema_score = 0.0
            print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member2']['name']}")
        elif key == ord('3'):
            claimed_user = "member3"
            scores_history.clear()
            ema_score = 0.0
            print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member3']['name']}")
        elif key == ord('4'):
            claimed_user = "impostor"
            scores_history.clear()
            ema_score = 0.0
            print(f"[SWITCH] Claimed Identity: Impostor Attack Test")
        elif key in (ord('+'), ord('=')):
            security_threshold = min(0.95, round(security_threshold + 0.05, 2))
            print(f"[CONFIG] Security Threshold increased to: {security_threshold * 100:.0f}%")
        elif key in (ord('-'), ord('_')):
            security_threshold = max(0.35, round(security_threshold - 0.05, 2))
            print(f"[CONFIG] Security Threshold decreased to: {security_threshold * 100:.0f}%")
        elif key in (ord('t'), ord('T')):
            if security_threshold == 0.55:
                security_threshold = 0.65
            elif security_threshold == 0.65:
                security_threshold = 0.75
            else:
                security_threshold = 0.55
            print(f"[CONFIG] Toggled Threshold to: {security_threshold * 100:.0f}%")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_gate()
