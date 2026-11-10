"""
demo_member2.py - Interactive CLI Demo for Member 2 (Voice Verification)
Part of: Hierarchical Multimodal Biometric Verification System
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from src.audio_recorder import record_audio
from src.voice_verification import voice_score


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def demo_automated_tests():
    print_header("1. Automated Voice Verification Test (test_voice.py)")
    import test_voice
    test_voice.test_voice()


def demo_voice_verification():
    print_header("2. Voice Verification Test (Mode B)")
    print("Select audio source for verification:")
    print("  [1] Pre-recorded Genuine sample (Member 1)")
    print("  [2] Pre-recorded Genuine sample (Member 2)")
    print("  [3] Pre-recorded Impostor sample (Impostor 1 attempting spoof)")
    print("  [4] Record Live Voice via Microphone (Live Mic)")
    sub_choice = input("\nEnter choice (1-4): ").strip()

    if sub_choice == "1":
        audio_path = "data/voice/member1/session3/passphrase_01.wav"
        print(f"\n[+] Loaded Member 1 Genuine Audio: {audio_path}")
    elif sub_choice == "2":
        audio_path = "data/voice/member2/session3/passphrase_01.wav"
        print(f"\n[+] Loaded Member 2 Genuine Audio: {audio_path}")
    elif sub_choice == "3":
        audio_path = "data/voice/impostor1/session3/passphrase_01.wav"
        print(f"\n[+] Loaded Impostor 1 Audio: {audio_path}")
    elif sub_choice == "4":
        audio_path = "temp_live_verify.wav"
        print("\nSpeak the passphrase now ('Open security gate now')...")
        record_audio(duration=2.5, output_path=audio_path, verbose=True)
    else:
        print("Invalid choice!")
        return

    claimed_user = input("Claimed Identity (member1 / member2 / member3) [Default: member1]: ").strip() or "member1"

    print(f"\n[Processing] Computing voice_score for claimed user: '{claimed_user}'...")
    try:
        score = voice_score(audio_path, claimed_user)
        threshold = 0.50
        is_verified = (score >= threshold)

        print("\n" + "-" * 55)
        print(f"  Claimed User:       {claimed_user.upper()}")
        print(f"  Match Score:        {score:.4f} (scale 0.0 to 1.0)")
        print(f"  Decision Threshold: {threshold:.2f}")

        if is_verified:
            print("  Final Decision:     [✓ ACCESS GRANTED - Genuine Speaker]")
        else:
            print("  Final Decision:     [✗ ACCESS DENIED - Impostor Rejected]")
        print("-" * 55)
    except Exception as e:
        print(f"Error during verification: {e}")


def demo_live_recording():
    print_header("3. Live Microphone Audio Recording Test")
    out_file = "live_sample.wav"
    duration = float(input("Recording duration in seconds [Default 2.5]: ").strip() or "2.5")
    print("\nGet ready to speak...")
    audio = record_audio(duration=duration, output_path=out_file, verbose=True)
    print(f"\n[✓] Recording successful! Saved to: {Path(out_file).resolve()}")
    print(f"Captured {len(audio)} samples at 16000 Hz.")


def demo_run_experiments():
    print_header("4. Re-run Voice Verification Experiments (E3)")
    confirm = input("Run experiments and update ROC plots? (y/n) [n]: ").strip().lower()
    if confirm == "y":
        import run_member2_experiments
        run_member2_experiments.main()


def main():
    while True:
        try:
            print_header("Multimodal Biometrics - Member 2 (Voice Verification)")
            print("  [1] Run Automated Voice Verification Tests")
            print("  [2] Interactive Voice Verification (Mode B)")
            print("  [3] Record Live Audio via Microphone")
            print("  [4] Run Voice Experiments & Generate ROCs (E3)")
            print("  [0] Exit")

            choice = input("\nEnter operation number (0-4): ").strip()

            if choice == "1":
                demo_automated_tests()
            elif choice == "2":
                demo_voice_verification()
            elif choice == "3":
                demo_live_recording()
            elif choice == "4":
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
