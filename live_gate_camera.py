import cv2
import time
from collections import deque
from src.face_models import face_score
from src.face_pipeline import crop_face

# ==========================================
# SECURITY CONFIGURATION
# ==========================================
SECURITY_THRESHOLD = 0.90      # Refuse anyone with score < 90%
VOTING_WINDOW = 10             # Number of consecutive frames to evaluate
REQUIRED_PASSES = 7            # Must pass at least 7 out of 10 frames
CLAIMED_USER = "member1"       # The user claiming identity at the gate (press '1' or '2' to switch)
# ==========================================

def run_live_gate():
    global CLAIMED_USER
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    # Buffer to hold scores of the last N frames for voting
    scores_history = deque(maxlen=VOTING_WINDOW)
    
    print("=" * 60)
    print(f" SMART SECURITY GATE: LIVE FACE VERIFICATION ")
    print(f" Claimed Identity:   {CLAIMED_USER}")
    print(f" Security Threshold: {SECURITY_THRESHOLD * 100:.0f}%")
    print(f" Press '1' for member1 | '2' for member2 | 'q' to exit.")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face box and check if a face is present
        face_gray, box = crop_face(frame, return_box=True)

        if face_gray is not None:
            # 1. Compute verification score for the claimed user
            # Check both raw frame and horizontally mirrored frame and take the better match
            # (handles webcams that mirror vs non-mirrored saved photos)
            score_raw = face_score(frame, user_id=CLAIMED_USER, model_type="svm")
            score_flip = face_score(cv2.flip(frame, 1), user_id=CLAIMED_USER, model_type="svm")
            score = max(score_raw, score_flip)
            scores_history.append(score)

            # Draw face bounding box on live camera feed
            fx, fy, fw, fh = box
            box_color = (0, 220, 0) if score >= SECURITY_THRESHOLD else (0, 0, 240)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
            cv2.putText(frame, f"{CLAIMED_USER}: {score * 100:.1f}%", (fx, max(125, fy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
        else:
            score = 0.0

        # 2. Frame Voting Logic (Bonus 2)
        passed_frames = sum(1 for s in scores_history if s >= SECURITY_THRESHOLD)
        
        # 3. Decision Logic
        if face_gray is None:
            gate_decision = "NO FACE DETECTED"
            color = (0, 165, 255)        # Orange
            subtext = "Please look directly at the camera..."
        elif len(scores_history) >= VOTING_WINDOW:
            if passed_frames >= REQUIRED_PASSES:
                gate_decision = "ACCESS GRANTED"
                color = (0, 220, 0)      # Green
                subtext = f"Verified: {CLAIMED_USER} (Score >= 90%)"
            else:
                gate_decision = "ACCESS DENIED"
                color = (0, 0, 240)      # Red
                subtext = f"Claimed: {CLAIMED_USER} - Score below 90% security threshold!"
        else:
            gate_decision = "SCANNING..."
            color = (255, 180, 0)        # Yellow/Cyan
            subtext = f"Collecting face frames: {len(scores_history)}/{VOTING_WINDOW}"

        # 4. Display UI Overlay on Frame
        h, w, _ = frame.shape
        cv2.rectangle(frame, (20, 20), (w - 20, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, 20), (w - 20, 110), color, 2)

        # Status text
        cv2.putText(frame, f"STATUS: {gate_decision}", (35, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Current frame score and voting count
        cv2.putText(frame, f"User: {CLAIMED_USER} | Score: {score * 100:.1f}% | Votes: {passed_frames}/{VOTING_WINDOW} | Thresh: {SECURITY_THRESHOLD * 100:.0f}%", 
                    (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
        cv2.putText(frame, subtext, (35, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Smart Security Gate - Live Camera", frame)

        # Key controls: 'q' to quit, '1' for member1, '2' for member2
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            CLAIMED_USER = "member1"
            scores_history.clear()
        elif key == ord('2'):
            CLAIMED_USER = "member2"
            scores_history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_gate()