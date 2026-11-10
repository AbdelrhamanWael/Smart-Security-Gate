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
DURATION = 2.5                 # Duration of voice sample in seconds
SECURITY_THRESHOLD = 0.60      # Realistic Security Threshold (60%)
MIN_SPEECH_PEAK = 0.005        # Minimum peak required to consider that a human spoke
CLAIMED_USER = "member1"       # Default user (press 1, 2, 3 to switch)

# Window dimensions
WIN_W, WIN_H = 800, 520
# ==========================================

class LiveVoiceGate:
    def __init__(self):
        self.claimed_user = CLAIMED_USER
        self.state = "IDLE"  # "IDLE", "COUNTDOWN", "RECORDING", "VERIFIED", "DENIED"
        self.countdown_start = 0
        self.countdown_val = 3
        
        # Audio buffer
        self.audio_frames = []
        self.live_waveform = np.zeros(300, dtype=np.float32)
        
        # Verification results
        self.last_score = 0.0
        self.result_message = "Press 'SPACE' to record your passphrase"
        
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
        self.result_message = "RECORDING: Say 'Open security gate now'..."

    def finish_recording(self):
        self.state = "ANALYZING"
        if len(self.audio_frames) > 0:
            audio = np.concatenate(self.audio_frames, axis=0)
            
            # Check Voice Activity / Energy: did the user actually speak?
            peak_val = float(np.max(np.abs(audio)))
            if peak_val < MIN_SPEECH_PEAK:
                self.state = "DENIED"
                self.last_score = 0.0
                self.result_message = "ACCESS DENIED! No speech detected (Dead Silence)"
                return
            
            # 1. Compute verification score
            self.last_score = voice_score(audio, user_id=self.claimed_user)
            
            if self.last_score >= SECURITY_THRESHOLD:
                self.state = "VERIFIED"
                self.result_message = f"ACCESS GRANTED! Identity Verified: {self.claimed_user}"
            else:
                self.state = "DENIED"
                self.result_message = f"ACCESS DENIED! Score ({self.last_score*100:.1f}%) below threshold {SECURITY_THRESHOLD*100:.0f}%"
        else:
            self.state = "IDLE"
            self.result_message = "No audio captured. Try again."

    def draw_ui(self):
        img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        img[:] = (24, 24, 28)  # Modern dark background
        
        # 1. Header Box
        header_color = (0, 220, 0) if self.state == "VERIFIED" else ((0, 0, 240) if self.state == "DENIED" else (255, 170, 0))
        cv2.rectangle(img, (20, 15), (WIN_W - 20, 85), (35, 35, 42), -1)
        cv2.rectangle(img, (20, 15), (WIN_W - 20, 85), header_color, 2)
        
        cv2.putText(img, "SMART SECURITY GATE: LIVE VOICE VERIFICATION", (35, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
        cv2.putText(img, f"Claimed Identity: {self.claimed_user.upper()} | Threshold: {SECURITY_THRESHOLD*100:.0f}%", 
                    (35, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # 2. Main Status Banner
        status_bg = (15, 60, 15) if self.state == "VERIFIED" else ((15, 15, 70) if self.state == "DENIED" else (40, 35, 20))
        cv2.rectangle(img, (20, 95), (WIN_W - 20, 160), status_bg, -1)
        cv2.rectangle(img, (20, 95), (WIN_W - 20, 160), header_color, 1)
        
        cv2.putText(img, f"STATUS: {self.state}", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_color, 2)
        cv2.putText(img, self.result_message, (35, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

        # 3. Live Waveform Visualizer Area
        cv2.rectangle(img, (20, 175), (WIN_W - 20, 310), (18, 18, 22), -1)
        cv2.rectangle(img, (20, 175), (WIN_W - 20, 310), (60, 60, 70), 1)
        cv2.putText(img, "LIVE MICROPHONE WAVEFORM", (35, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 140), 1)

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
        cv2.rectangle(img, (20, 325), (380, 445), (32, 32, 38), -1)
        cv2.rectangle(img, (20, 325), (380, 445), (60, 60, 70), 1)
        
        cv2.putText(img, "BIOMETRIC MATCH SCORE", (35, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        score_pct = self.last_score * 100
        score_col = (0, 220, 0) if self.last_score >= SECURITY_THRESHOLD else (0, 0, 240)
        cv2.putText(img, f"{score_pct:.1f}%", (35, 395), cv2.FONT_HERSHEY_SIMPLEX, 1.4, score_col, 3)
        cv2.putText(img, f"Raw Score: {self.last_score:.4f} / 1.0", (35, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        # Score bar
        bar_fill = int(120 * min(1.0, self.last_score))
        cv2.rectangle(img, (210, 365), (210 + 120, 385), (50, 50, 60), -1)
        cv2.rectangle(img, (210, 365), (210 + bar_fill, 385), score_col, -1)
        cv2.rectangle(img, (210, 365), (210 + 120, 385), (100, 100, 110), 1)
        cv2.putText(img, "Threshold", (235, 403), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

        # 5. Gate Access & Biometric Decision Panel
        cv2.rectangle(img, (400, 325), (WIN_W - 20, 445), (32, 32, 38), -1)
        cv2.rectangle(img, (400, 325), (WIN_W - 20, 445), (60, 60, 70), 1)
        
        cv2.putText(img, "GATE ACCESS DECISION", (415, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        gate_state_str = "UNLOCKED (OPEN)" if self.state == "VERIFIED" else "LOCKED (RESTRICTED)"
        gate_color = (0, 220, 0) if self.state == "VERIFIED" else (0, 0, 240)
        cv2.putText(img, f"Gate Status: {gate_state_str}", (415, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.50, gate_color, 2)
        cv2.putText(img, f"Target User: {self.claimed_user.upper()}", (415, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 180), 1)
        cv2.putText(img, "Passphrase: 'Open security gate now'", (415, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        # 6. Footer Controls
        cv2.putText(img, "[SPACE]: Record Passphrase  |  [1/2/3]: Switch User  |  [Q]: Quit",
                    (35, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return img

    def run(self):
        print("=" * 60)
        print(" STARTING LIVE VOICE GATE GUI ")
        print(" - Press 'SPACE' to record your 2.5s passphrase")
        print(" - Press '1' for Member 1 | '2' for Member 2 | '3' for Member 3")
        print(" - Press 'q' to exit")
        print("=" * 60)

        # Start live mic stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024
        )
        
        with self.stream:
            while True:
                # Handle states
                if self.state == "COUNTDOWN":
                    elapsed = time.time() - self.countdown_start
                    self.countdown_val = int(3.0 - elapsed) + 1
                    self.result_message = f"GET READY: Recording in {self.countdown_val}s..."
                    if elapsed >= 3.0:
                        self.start_recording()

                elif self.state == "RECORDING":
                    if time.time() - self.record_start >= DURATION:
                        self.finish_recording()

                # Render frame
                frame = self.draw_ui()
                cv2.imshow("Smart Security Gate - Live Voice Verification", frame)

                # Key handling
                key = cv2.waitKey(20) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if self.state not in ("COUNTDOWN", "RECORDING"):
                        self.state = "COUNTDOWN"
                        self.countdown_start = time.time()
                elif key == ord('1'):
                    self.claimed_user = "member1"
                    self.state = "IDLE"
                    self.result_message = f"Switched user to {self.claimed_user.upper()}"
                elif key == ord('2'):
                    self.claimed_user = "member2"
                    self.state = "IDLE"
                    self.result_message = f"Switched user to {self.claimed_user.upper()}"
                elif key == ord('3'):
                    self.claimed_user = "member3"
                    self.state = "IDLE"
                    self.result_message = f"Switched user to {self.claimed_user.upper()}"

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveVoiceGate()
    app.run()

