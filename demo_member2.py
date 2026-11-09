"""
demo.py - Interactive CLI Demo for Member 2 (Voice & Speech Commands)
Part of: Hierarchical Multimodal Biometric Verification System
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from audio_recorder import record_audio
from command_system import (
    COMMAND_CLASSES,
    ROLE_PERMISSIONS,
    _AUDIT_LOGGER,
    allowed_commands,
    execute_voice_command,
    recognize_command,
)
from voice_verification import voice_score


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def demo_checklist():
    print_header("1. Automated Checklist Verification (10/10)")
    import verify_all_checklist
    verify_all_checklist.run_verification()


def demo_voice_verification():
    print_header("2. Voice Verification Test (Mode B)")
    print("Select audio source for verification:")
    print("  [1] Pre-recorded Genuine sample (Member 2)")
    print("  [2] Pre-recorded Impostor sample (Impostor 1 attempting spoof)")
    print("  [3] Record Live Voice via Microphone (Live Mic)")
    sub_choice = input("\nEnter choice (1-3): ").strip()

    if sub_choice == "1":
        audio_path = "data/voice/member2/session3/passphrase_01.wav"
        print(f"\n[+] Loaded Member 2 Genuine Audio: {audio_path}")
    elif sub_choice == "2":
        audio_path = "data/voice/impostor1/session3/passphrase_01.wav"
        print(f"\n[+] Loaded Impostor 1 Audio: {audio_path}")
    elif sub_choice == "3":
        audio_path = "temp_live_verify.wav"
        print("\nSpeak the passphrase now...")
        record_audio(duration=2.5, output_path=audio_path, verbose=True)
    else:
        print("Invalid choice!")
        return

    claimed_user = input("Claimed Identity (member1 / member2 / member3) [Default: member2]: ").strip() or "member2"

    print(f"\n[Processing] Computing voice_score for claimed user: '{claimed_user}'...")
    try:
        score = voice_score(audio_path, claimed_user)
        threshold = 0.50
        is_verified = (score >= threshold)

        print("\n" + "-" * 55)
        print(f"  Claimed User:     {claimed_user.upper()}")
        print(f"  Match Score:      {score:.4f} (scale 0.0 to 1.0)")
        print(f"  Decision Threshold: {threshold:.2f}")

        if is_verified:
            print("  Final Decision:   [✓ ACCESS GRANTED - Genuine Speaker]")
        else:
            print("  Final Decision:   [✗ ACCESS DENIED - Impostor Rejected]")
        print("-" * 55)
    except Exception as e:
        print(f"Error during verification: {e}")


def demo_command_system():
    print_header("3. Speech Command Recognition & Role-Based Security (Bonus 1)")
    print("Select User / Verification State:")
    print("  [1] Member 1 (Super Admin) - Allowed: Open, Lock, Logs, Status")
    print("  [2] Member 2 (Operator)    - Allowed: Open, Lock, Status (Logs forbidden)")
    print("  [3] Member 3 (Auditor)     - Allowed: Logs, Status (Open & Lock forbidden)")
    print("  [4] Unverified Intruder    - Blocked from all actions (Access Denied)")
    u_choice = input("\nEnter choice (1-4): ").strip()

    users_map = {
        "1": ("member1", True, 2),
        "2": ("member2", True, 2),
        "3": ("member3", True, 2),
        "4": (None, False, 0),
    }

    if u_choice not in users_map:
        print("Invalid choice!")
        return

    user_id, is_verified, level = users_map[u_choice]
    role_name = ROLE_PERMISSIONS[user_id]["role_name"] if user_id else "Unverified Intruder"
    print(f"\nActive User: {user_id or 'None'} ({role_name}) | Verified: {is_verified} | Level: {level}")
    print(f"Authorized Commands: {allowed_commands(user_id, level)}")

    print("\nSelect Command to Execute:")
    print("  [1] Open   (Unlock Gate / Vault)")
    print("  [2] Lock   (Lock Gate / Secure System)")
    print("  [3] Logs   (Access Audit Logs)")
    print("  [4] Status (Check System & Sensor Status)")
    print("  [5] Unknown / Background Noise")
    print("  [6] Speak Command via Live Microphone")
    c_choice = input("\nEnter choice (1-6): ").strip()

    cmd_map = {
        "1": "open",
        "2": "lock",
        "3": "logs",
        "4": "status",
        "5": "unknown",
    }

    if c_choice in cmd_map:
        cmd_target = cmd_map[c_choice]
        if user_id and cmd_target in COMMAND_CLASSES and cmd_target != "unknown":
            audio_source = f"data/commands/{user_id}/session1/{cmd_target}/{cmd_target}_01.wav"
        else:
            audio_source = f"data/commands/unknown/session1/unknown_01.wav" if cmd_target == "unknown" else cmd_target

        result = execute_voice_command(
            audio_or_cmd=audio_source,
            user_id=user_id,
            is_verified=is_verified,
            level_reached=level,
        )
    elif c_choice == "6":
        audio_path = "temp_live_cmd.wav"
        print("\nSay command (Open, Lock, Logs, Status) now...")
        record_audio(duration=1.8, output_path=audio_path, verbose=True)
        rec = recognize_command(audio_path)
        print(f"\n[Voice Recognizer]: Detected Command -> '{rec.upper()}'")
        result = execute_voice_command(
            audio_or_cmd=audio_path,
            user_id=user_id,
            is_verified=is_verified,
            level_reached=level,
        )
    else:
        print("Invalid choice!")
        return

    print("\n" + "-" * 55)
    print(f"Execution Result: [{result['status']}]")
    print(f"Message:          {result['message']}")
    print("-" * 55)


def demo_live_recording():
    print_header("4. Live Microphone Audio Recording Test (Bonus 2)")
    out_file = "live_sample.wav"
    duration = float(input("Recording duration in seconds [Default 2.5]: ").strip() or "2.5")
    print("\nGet ready to speak...")
    audio = record_audio(duration=duration, output_path=out_file, verbose=True)
    print(f"\n[✓] Recording successful! Saved to: {Path(out_file).resolve()}")
    print(f"Captured {len(audio)} samples at 16000 Hz.")


def demo_view_audit_logs():
    print_header("5. View Security Audit Trail (Audit Log)")
    logs = _AUDIT_LOGGER.get_recent_logs(limit=10)
    if not logs:
        print("No audit log events found.")
        return

    print(f"{'Timestamp':<25} | {'User':<10} | {'Role':<14} | {'Command':<8} | {'Outcome'}")
    print("-" * 75)
    for row in logs:
        ts = row.get("timestamp", "")[:19].replace("T", " ")
        u = row.get("user_id", "")
        r = row.get("role", "")
        c = row.get("command", "")
        o = row.get("outcome", "")
        print(f"{ts:<25} | {u:<10} | {r:<14} | {c:<8} | {o}")


def demo_run_experiments():
    print_header("6. Re-run All Experiments & Re-train Models")
    confirm = input("Re-train models and update ROC plots? (y/n) [n]: ").strip().lower()
    if confirm == "y":
        import run_experiments
        run_experiments.main()


def main():
    while True:
        try:
            print_header("Multimodal Biometrics - Member 2 (Voice & Speech Commands)")
            print("  [1] Verify All Checklist Requirements (10/10)")
            print("  [2] Voice Verification Test (Mode B)")
            print("  [3] Speech Commands Recognition & Roles (Bonus 1)")
            print("  [4] Record Live Audio via Microphone (Bonus 2)")
            print("  [5] View Security Audit Trail (Audit Log)")
            print("  [6] Re-run All Experiments (E3 & E8)")
            print("  [0] Exit")

            choice = input("\nEnter operation number (0-6): ").strip()

            if choice == "1":
                demo_checklist()
            elif choice == "2":
                demo_voice_verification()
            elif choice == "3":
                demo_command_system()
            elif choice == "4":
                demo_live_recording()
            elif choice == "5":
                demo_view_audit_logs()
            elif choice == "6":
                demo_run_experiments()
            elif choice == "0":
                print("\nExiting program. Goodbye!")
                break
            else:
                print("\nInvalid selection, please try again.")

            input("\nPress [Enter] to return to the main menu...")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program. Goodbye!")
            break


if __name__ == "__main__":
    main()
