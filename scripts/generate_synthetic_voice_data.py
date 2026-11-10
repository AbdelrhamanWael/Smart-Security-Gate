"""
generate_synthetic_voice_data.py - Acoustic Audio Dataset Synthesizer (Member 2)
Part of: Hierarchical Multimodal Biometric Verification System

Creates a complete, realistic biometric voice dataset conforming to:
data/voice/{person_id}/{session_id}/passphrase_{i}.wav
data/commands/{person_id}/{session_id}/{command}_{i}.wav
data/commands/unknown/{session_id}/unknown_{i}.wav

Each speaker has unique vocal tract characteristics (fundamental frequency F0,
formant resonances F1-F4, harmonic glottal excitation).
Each command has distinct phonetic formant modulation patterns.
Sessions include realistic variations in mic gain, pitch drift, and ambient noise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

TARGET_SR = 16000
BASE_DIR = Path("data")


# Speaker acoustic profiles: base F0 (Hz) and formants [F1, F2, F3, F4] (Hz)
SPEAKER_PROFILES: Dict[str, Dict] = {
    "member1": {
        "role": "Super Admin",
        "f0": 130.0,
        "formants": [720.0, 1220.0, 2550.0, 3600.0],
        "bandwidths": [80.0, 100.0, 140.0, 180.0],
        "vocal_speed": 1.0,
    },
    "member2": {
        "role": "Operator",
        "f0": 165.0,
        "formants": [600.0, 1450.0, 2700.0, 3750.0],
        "bandwidths": [75.0, 95.0, 130.0, 175.0],
        "vocal_speed": 1.05,
    },
    "member3": {
        "role": "Auditor",
        "f0": 210.0,
        "formants": [820.0, 1600.0, 2850.0, 3900.0],
        "bandwidths": [90.0, 110.0, 150.0, 200.0],
        "vocal_speed": 0.95,
    },
    "impostor1": {
        "role": "Impostor",
        "f0": 115.0,
        "formants": [680.0, 1150.0, 2400.0, 3450.0],
        "bandwidths": [85.0, 105.0, 145.0, 190.0],
        "vocal_speed": 1.0,
    },
    "impostor2": {
        "role": "Impostor",
        "f0": 185.0,
        "formants": [750.0, 1500.0, 2800.0, 3800.0],
        "bandwidths": [80.0, 100.0, 135.0, 180.0],
        "vocal_speed": 1.1,
    },
    "impostor3": {
        "role": "Impostor",
        "f0": 240.0,
        "formants": [850.0, 1700.0, 3000.0, 4100.0],
        "bandwidths": [95.0, 120.0, 160.0, 210.0],
        "vocal_speed": 0.9,
    },
    "impostor4": {
        "role": "Impostor",
        "f0": 145.0,
        "formants": [640.0, 1300.0, 2600.0, 3650.0],
        "bandwidths": [80.0, 95.0, 130.0, 170.0],
        "vocal_speed": 1.02,
    },
    "impostor5": {
        "role": "Impostor",
        "f0": 195.0,
        "formants": [780.0, 1550.0, 2900.0, 3950.0],
        "bandwidths": [85.0, 105.0, 140.0, 185.0],
        "vocal_speed": 0.98,
    },
}

COMMANDS = ["open", "lock", "logs", "status"]
SESSIONS = ["session1", "session2", "session3"]


def _apply_formant_filter(
    signal_in: np.ndarray,
    freq: float,
    bw: float,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Apply a second-order resonator filter at center frequency `freq` with bandwidth `bw`.
    """
    r = np.exp(-np.pi * (bw / sr))
    theta = 2.0 * np.pi * (freq / sr)
    b = np.array([1.0 - r, 0.0, 0.0], dtype=np.float64)
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r], dtype=np.float64)
    # Normalize filter gain
    return signal.lfilter(b, a, signal_in)


def synthesize_utterance(
    speaker_profile: Dict,
    utterance_type: str,
    duration_sec: float = 1.4,
    session_noise_level: float = 0.015,
    pitch_jitter_pct: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthesize an acoustic speech waveform via glottal harmonic excitation
    filtered through speaker-specific and phonetic formant resonators.
    """
    rng = np.random.RandomState(seed)
    sr = TARGET_SR
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    base_f0 = speaker_profile["f0"]
    # Add slight pitch jitter and natural intonation arc
    intonation = 1.0 + 0.08 * np.sin(np.pi * t / duration_sec)
    pitch_noise = rng.normal(0, pitch_jitter_pct * base_f0, total_samples)
    instantaneous_f0 = np.clip(base_f0 * intonation + pitch_noise, 60.0, 450.0)

    # Glottal source (sum of harmonics with 1/n decay)
    phase = 2.0 * np.pi * np.cumsum(instantaneous_f0) / sr
    glottal_source = np.zeros(total_samples, dtype=np.float64)
    num_harmonics = min(35, int((sr / 2.0) / base_f0))
    for h in range(1, num_harmonics + 1):
        amp = (1.0 / (h ** 0.85)) * (0.95 + 0.1 * rng.rand())
        glottal_source += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))

    # Aspiration noise (breathiness)
    aspiration = rng.normal(0, 0.06, total_samples)
    source = glottal_source + aspiration

    # Phonetic modulation of formants and amplitude envelope
    formants = list(speaker_profile["formants"])
    bws = list(speaker_profile["bandwidths"])

    # Temporal amplitude envelope
    envelope = np.zeros(total_samples, dtype=np.float64)
    lead_silence = int(0.08 * sr)
    tail_silence = int(0.08 * sr)
    active_len = total_samples - lead_silence - tail_silence

    if active_len > 0:
        if utterance_type == "passphrase":
            # 3-syllable rhythm for passphrase
            t_act = np.linspace(0, 3 * np.pi, active_len)
            env_active = np.sin(t_act) ** 2 * (0.8 + 0.2 * np.sin(2 * t_act))
        elif utterance_type == "open":
            # 2 distinct syllables ("o-pen")
            t_act = np.linspace(0, 2 * np.pi, active_len)
            env_active = np.sin(t_act) ** 2
            formants[0] *= 0.85
            formants[1] *= 1.15
        elif utterance_type == "lock":
            # 1 sharp syllable with plosive burst at end
            t_act = np.linspace(0, np.pi, active_len)
            env_active = np.sin(t_act) ** 1.5
            formants[0] *= 1.10
            formants[1] *= 0.85
        elif utterance_type == "logs":
            # 1 prolonged syllable with trailing fricative
            t_act = np.linspace(0, np.pi, active_len)
            env_active = np.sin(t_act) ** 1.2
            formants[0] *= 1.05
            formants[1] *= 0.90
        elif utterance_type == "status":
            # 2 syllables ("sta-tus")
            t_act = np.linspace(0, 2 * np.pi, active_len)
            env_active = np.sin(t_act) ** 2
            formants[0] *= 0.95
            formants[1] *= 1.20
        else:  # unknown / noise / random words
            t_act = np.linspace(0, rng.uniform(1, 3) * np.pi, active_len)
            env_active = np.abs(np.sin(t_act))
            formants[0] *= rng.uniform(0.8, 1.2)
            formants[1] *= rng.uniform(0.8, 1.2)

        envelope[lead_silence:lead_silence + active_len] = np.clip(env_active, 0.0, 1.0)

    # Filter source through the 4 vocal tract formants (in parallel + cascade)
    vocal_tract_output = np.zeros(total_samples, dtype=np.float64)
    gains = [1.0, 0.5, 0.25, 0.12]
    for (f, bw, g) in zip(formants, bws, gains):
        filtered = _apply_formant_filter(source, f, bw, sr=sr)
        vocal_tract_output += g * filtered

    # Modulate by envelope
    speech_signal = vocal_tract_output * envelope

    # Add ambient room noise according to session noise level
    room_noise = rng.normal(0, session_noise_level, total_samples)
    # Slight low-pass filtering on noise
    b_noise, a_noise = signal.butter(2, 4000.0 / (sr / 2.0))
    room_noise = signal.lfilter(b_noise, a_noise, room_noise)

    combined = speech_signal + room_noise

    # Peak normalization to ~0.90
    max_peak = np.max(np.abs(combined))
    if max_peak > 1e-5:
        combined = (combined / max_peak) * 0.90

    return combined.astype(np.float32)


def generate_full_dataset(
    passphrase_per_session: int = 25,
    commands_per_session: int = 25,
    unknown_per_session: int = 25,
    base_dir: Path = BASE_DIR,
) -> None:
    """
    Generate the complete project dataset for Member 2:
    - Passphrase: 25 recordings x 3 sessions = 75 per speaker (for 3 members + 5 impostors)
    - Commands: 25 recordings x 4 commands x 3 sessions = 300 per member
    - Unknown class: 25 recordings x 3 sessions = 75 unknown samples
    """
    print("=" * 65)
    print("Generating Authentic Biometric Voice & Speech Command Dataset...")
    print("=" * 65)

    base_dir = Path(base_dir)
    voice_dir = base_dir / "voice"
    cmd_dir = base_dir / "commands"

    seed = 1000

    # 1. Generate Voice Verification Data (Passphrase recordings)
    for speaker_id, profile in SPEAKER_PROFILES.items():
        for s_idx, session in enumerate(SESSIONS, start=1):
            sess_dir = voice_dir / speaker_id / session
            sess_dir.mkdir(parents=True, exist_ok=True)

            # Session-specific acoustics (noise, mic distance)
            sess_noise = 0.008 + s_idx * 0.006
            pitch_jitter = 0.015 + s_idx * 0.005

            for i in range(1, passphrase_per_session + 1):
                seed += 1
                audio = synthesize_utterance(
                    speaker_profile=profile,
                    utterance_type="passphrase",
                    duration_sec=1.5,
                    session_noise_level=sess_noise,
                    pitch_jitter_pct=pitch_jitter,
                    seed=seed,
                )
                wav_path = sess_dir / f"passphrase_{i:02d}.wav"
                wavfile.write(str(wav_path), TARGET_SR, (audio * 32767).astype(np.int16))

        print(f"  [+] Voice passphrase generated for {speaker_id} ({profile['role']})")

    # 2. Generate Speech Command Data (open, lock, logs, status)
    for speaker_id in ["member1", "member2", "member3"]:
        profile = SPEAKER_PROFILES[speaker_id]
        for s_idx, session in enumerate(SESSIONS, start=1):
            for cmd in COMMANDS:
                sess_dir = cmd_dir / speaker_id / session / cmd
                sess_dir.mkdir(parents=True, exist_ok=True)

                sess_noise = 0.008 + s_idx * 0.005

                for i in range(1, commands_per_session + 1):
                    seed += 1
                    audio = synthesize_utterance(
                        speaker_profile=profile,
                        utterance_type=cmd,
                        duration_sec=1.1,
                        session_noise_level=sess_noise,
                        pitch_jitter_pct=0.02,
                        seed=seed,
                    )
                    wav_path = sess_dir / f"{cmd}_{i:02d}.wav"
                    wavfile.write(str(wav_path), TARGET_SR, (audio * 32767).astype(np.int16))

        print(f"  [+] Commands (open, lock, logs, status) generated for {speaker_id}")

    # 3. Generate Unknown class samples (out-of-vocabulary words & noise)
    for s_idx, session in enumerate(SESSIONS, start=1):
        unk_dir = cmd_dir / "unknown" / session
        unk_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, unknown_per_session + 1):
            seed += 1
            # Random speaker profile from impostors
            imp_profile = SPEAKER_PROFILES[f"impostor{(i % 5) + 1}"]
            audio = synthesize_utterance(
                speaker_profile=imp_profile,
                utterance_type="unknown",
                duration_sec=1.2,
                session_noise_level=0.03,
                pitch_jitter_pct=0.03,
                seed=seed,
            )
            wav_path = unk_dir / f"unknown_{i:02d}.wav"
            wavfile.write(str(wav_path), TARGET_SR, (audio * 32767).astype(np.int16))

    print("  [+] Unknown class samples generated.")
    print("=" * 65)
    print("Dataset generation completed successfully!")
    print(f"Voice data location:    {voice_dir.resolve()}")
    print(f"Commands data location: {cmd_dir.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    generate_full_dataset()

