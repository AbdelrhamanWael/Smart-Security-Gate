"""
live_gate_voice.py - Real-Time Microphone Biometric Gate (Bonus 2)
Part of: Hierarchical Multimodal Biometric Verification System

Features:
- Live real-time audio waveform visualizer using sounddevice & OpenCV.
- 16 kHz Mono resampling with automatic VAD (Voice Activity Detection).
- 56-D acoustic feature extraction (MFCC + Delta-MFCC + ZCR + RMS Energy).
- Identity verification against SVM (RBF) acoustic models.
- Interactive controls:
    [SPACE] Countdown & Record 2.5s passphrase ('Open security gate now')
    [1] Claim: Abdelrahman Wael (Member 1 - Super Admin)
    [2] Claim: Hashem (Member 2 - Operator)
    [3] Claim: Hagar (Member 3 - Auditor)
    [+] Increase threshold | [-] Decrease threshold
    [Q] Quit / Exit
"""

import os
import sys
import time
import numpy as np
import cv2
import sounddevice as sd

os.environ["MPLCONFIGDIR"] = os.path.abspath("tmp")

from src.voice_verification import voice_score

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000
DURATION = 2.5                 # Duration of voice recording in seconds
DEFAULT_THRESHOLD = 0.55       # Balanced security threshold (55%)
MIN_SPEECH_PEAK = 0.005        # Minimum peak required to detect speech
WIN_W, WIN_H = 800, 520
# ==========================================

USER_PROFILES = {
    "member1": {"name": "Abdelrahman Wael", "role": "Super Admin"},
    "member2": {"name": "Hashem", "role": "Operator"},
    "member3": {"name": "Hagar", "role": "Auditor"},
}

class LiveVoiceGate:
    def __init__(self):
        self.claimed_user = "member1"
        self.security_threshold = DEFAULT_THRESHOLD
        self.state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING", "VERIFIED", "DENIED"
        self.countdown_start = 0
        self.countdown_val = 3
        
        # Audio buffer
        self.audio_frames = []
        self.live_waveform = np.zeros(300, dtype=np.float32)
        
        # Verification results
        self.last_score = 0.0
        self.result_message = "Press 'SPACE' to record passphrase: 'Open security gate now'"
        
        # Stream for live audio
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """Live microphone callback to update waveform and record frames."""
        mono = indata[:, 0]
        # Keep sliding waveform for visualization
        chunk_step = len(mono) // 10
        if chunk_step > 0:
            resampled = mono[::chunk_step][:30]
            self.live_waveform = np.roll(self.live_waveform, -len(resampled))
            self.live_waveform[-len(resampled):] = resampled
            
        if self.state == "RECORDING":
            self.audio_frames.append(mono.copy())

    def start_recording(self):
        self.state = "RECORDING"
        self.audio_frames = []
        self.record_start = time.time()
        self.result_message = "RECORDING: Speak clearly: 'Open security gate now'..."

    def finish_recording(self):
        self.state = "ANALYZING"
        if len(self.audio_frames) > 0:
            audio = np.concatenate(self.audio_frames, axis=0)
            
            # Check Voice Activity / Energy: did the user actually speak?
            peak_val = float(np.max(np.abs(audio)))
            if peak_val < MIN_SPEECH_PEAK:
                self.state = "DENIED"
                self.last_score = 0.0
                self.result_message = "ACCESS DENIED: No speech detected (Microphone Silent)"
                return
            
            # Compute verification score
            self.last_score = voice_score(audio, user_id=self.claimed_user)
            
            if self.last_score >= self.security_threshold:
                self.state = "VERIFIED"
                self.result_message = f"ACCESS GRANTED! Identity Verified: {USER_PROFILES[self.claimed_user]['name']}"
            else:
                self.state = "DENIED"
                self.result_message = f"ACCESS DENIED! Score ({self.last_score*100:.1f}%) below {self.security_threshold*100:.0f}% threshold"
        else:
            self.state = "IDLE"
            self.result_message = "No audio captured. Press SPACE to try again."

    def draw_ui(self):
        img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        img[:] = (20, 22, 28)  # Modern dark background
        
        # 1. Header Box
        header_color = (0, 230, 0) if self.state == "VERIFIED" else ((0, 0, 240) if self.state == "DENIED" else (0, 200, 255))
        cv2.rectangle(img, (20, 15), (WIN_W - 20, 85), (30, 34, 42), -1)
        cv2.rectangle(img, (20, 15), (WIN_W - 20, 85), header_color, 2)
        
        cv2.putText(img, "SMART SECURITY GATE: LIVE VOICE VERIFICATION", (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2)
        cv2.putText(img, f"Claimed Identity: {USER_PROFILES[self.claimed_user]['name']} ({USER_PROFILES[self.claimed_user]['role']}) | Threshold: {self.security_threshold*100:.0f}%", 
                    (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # 2. Main Status Banner
        status_bg = (15, 60, 15) if self.state == "VERIFIED" else ((15, 15, 70) if self.state == "DENIED" else (35, 30, 15))
        cv2.rectangle(img, (20, 95), (WIN_W - 20, 160), status_bg, -1)
        cv2.rectangle(img, (20, 95), (WIN_W - 20, 160), header_color, 1)
        
        cv2.putText(img, f"STATUS: {self.state}", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, header_color, 2)
        cv2.putText(img, self.result_message, (35, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        # 3. Live Waveform Visualizer Area
        cv2.rectangle(img, (20, 175), (WIN_W - 20, 310), (16, 18, 24), -1)
        cv2.rectangle(img, (20, 175), (WIN_W - 20, 310), (50, 55, 68), 1)
        cv2.putText(img, "LIVE MICROPHONE INPUT (16 kHz MONO PCM)", (35, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 130, 150), 1)

        # Draw waveform line
        center_y = 242
        pts = []
        wave_pts = self.live_waveform
        num_pts = len(wave_pts)
        for i in range(num_pts):
            px = int(40 + (i / num_pts) * (WIN_W - 80))
            py = int(center_y + np.clip(wave_pts[i] * 120, -55, 55))
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], isClosed=False, color=(0, 230, 255), thickness=2)

        # Recording progress bar if recording
        if self.state == "RECORDING":
            elapsed = time.time() - self.record_start
            progress = min(1.0, elapsed / DURATION)
            bar_w = int((WIN_W - 80) * progress)
            cv2.rectangle(img, (40, 298), (40 + bar_w, 304), (0, 120, 255), -1)

        # 4. Score & Metrics Panel
        cv2.rectangle(img, (20, 325), (380, 445), (28, 32, 40), -1)
        cv2.rectangle(img, (20, 325), (380, 445), (55, 60, 72), 1)
        
        cv2.putText(img, "VOICE MATCH CONFIDENCE", (35, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
        score_pct = self.last_score * 100
        score_col = (0, 230, 0) if self.last_score >= self.security_threshold else (0, 0, 240)
        cv2.putText(img, f"{score_pct:.1f}%", (35, 395), cv2.FONT_HERSHEY_SIMPLEX, 1.3, score_col, 3)
        cv2.putText(img, f"Raw Score: {self.last_score:.4f} / 1.0", (35, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Score bar
        bar_fill = int(120 * min(1.0, max(0.0, self.last_score)))
        cv2.rectangle(img, (210, 365), (210 + 120, 385), (45, 48, 58), -1)
        cv2.rectangle(img, (210, 365), (210 + bar_fill, 385), score_col, -1)
        cv2.rectangle(img, (210, 365), (210 + 120, 385), (90, 95, 105), 1)
        thresh_x = int(210 + 120 * self.security_threshold)
        cv2.line(img, (thresh_x, 363), (thresh_x, 387), (255, 255, 255), 2)
        cv2.putText(img, f"Thresh ({self.security_threshold*100:.0f}%)", (215, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        # 5. Gate Access & Biometric Decision Panel
        cv2.rectangle(img, (400, 325), (WIN_W - 20, 445), (28, 32, 40), -1)
        cv2.rectangle(img, (400, 325), (WIN_W - 20, 445), (55, 60, 72), 1)
        
        cv2.putText(img, "GATE ACCESS DECISION", (415, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
        gate_state_str = "UNLOCKED (GRANTED)" if self.state == "VERIFIED" else "LOCKED (DENIED)"
        gate_color = (0, 230, 0) if self.state == "VERIFIED" else (0, 0, 240)
        cv2.putText(img, f"Gate: {gate_state_str}", (415, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gate_color, 2)
        cv2.putText(img, f"User: {USER_PROFILES[self.claimed_user]['name']}", (415, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
        cv2.putText(img, "Passphrase: 'Open security gate now'", (415, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

        # 6. Footer Controls
        cv2.rectangle(img, (20, 465), (WIN_W - 20, 505), (15, 18, 24), -1)
        cv2.rectangle(img, (20, 465), (WIN_W - 20, 505), (50, 55, 68), 1)
        cv2.putText(img, "[SPACE] Record Passphrase  |  [1/2/3] Switch User  |  [+/-] Threshold  |  [Q] Exit",
                    (35, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        return img

    def run(self):
        print("=" * 65)
        print("        SMART SECURITY GATE: REAL-TIME VOICE VERIFIER       ")
        print(" Controls:")
        print("   [SPACE] Start 3s countdown & record 2.5s passphrase")
        print("   [1] Abdelrahman Wael  |  [2] Hashem  |  [3] Hagar")
        print("   [+] Increase Thresh   |  [-] Decrease Thresh")
        print("   [Q] Exit")
        print("=" * 65)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024
        )
        
        with self.stream:
            while True:
                if self.state == "COUNTDOWN":
                    elapsed = time.time() - self.countdown_start
                    self.countdown_val = int(3.0 - elapsed) + 1
                    self.result_message = f"GET READY: Speak passphrase in {self.countdown_val}s..."
                    if elapsed >= 3.0:
                        self.start_recording()

                elif self.state == "RECORDING":
                    if time.time() - self.record_start >= DURATION:
                        self.finish_recording()

                frame = self.draw_ui()
                cv2.imshow("Smart Security Gate - Live Voice Verification", frame)

                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord(' '):
                    if self.state not in ("COUNTDOWN", "RECORDING"):
                        self.state = "COUNTDOWN"
                        self.countdown_start = time.time()
                elif key == ord('1'):
                    self.claimed_user = "member1"
                    self.state = "IDLE"
                    self.result_message = f"Switched identity to: {USER_PROFILES['member1']['name']}"
                    print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member1']['name']}")
                elif key == ord('2'):
                    self.claimed_user = "member2"
                    self.state = "IDLE"
                    self.result_message = f"Switched identity to: {USER_PROFILES['member2']['name']}"
                    print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member2']['name']}")
                elif key == ord('3'):
                    self.claimed_user = "member3"
                    self.state = "IDLE"
                    self.result_message = f"Switched identity to: {USER_PROFILES['member3']['name']}"
                    print(f"[SWITCH] Claimed Identity: {USER_PROFILES['member3']['name']}")
                elif key in (ord('+'), ord('=')):
                    self.security_threshold = min(0.95, round(self.security_threshold + 0.05, 2))
                    print(f"[CONFIG] Voice Threshold increased to: {self.security_threshold * 100:.0f}%")
                elif key in (ord('-'), ord('_')):
                    self.security_threshold = max(0.30, round(self.security_threshold - 0.05, 2))
                    print(f"[CONFIG] Voice Threshold decreased to: {self.security_threshold * 100:.0f}%")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveVoiceGate()
    app.run()
