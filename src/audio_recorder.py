"""
audio_recorder.py - Real-Time Audio Recorder (Member 2 / Bonus 2)
Part of: Hierarchical Multimodal Biometric Verification System

Responsibilities:
- Live microphone recording (2-3 seconds) feeding into voice pipeline.
- Dual-engine architecture:
    1. Built-in Windows MCI API (winmm.dll) via ctypes (100% native, zero dependencies).
    2. sounddevice / soundfile engine if available.
- Outputs 16 kHz mono float32 numpy array and/or saves directly to WAV.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import tempfile
import time
from typing import Optional, Union

import numpy as np
import scipy.io.wavfile as wavfile


def _record_with_winmm(
    duration: float = 2.5,
    sr: int = 16000,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Record audio using Windows native Multimedia API (winmm.dll / MCI).
    Zero external packages required.
    """
    winmm = ctypes.windll.winmm
    buffer_out = ctypes.create_unicode_buffer(256)

    def mci_cmd(cmd_str: str) -> None:
        err = winmm.mciSendStringW(cmd_str, buffer_out, 256, 0)
        if err != 0:
            err_buf = ctypes.create_unicode_buffer(256)
            winmm.mciGetErrorStringW(err, err_buf, 256)
            raise RuntimeError(f"MCI Error ({err}): {err_buf.value} from '{cmd_str}'")

    temp_wav = output_path or tempfile.mktemp(suffix=".wav")
    temp_wav_path = Path(temp_wav).resolve()

    try:
        # Close any previous device
        winmm.mciSendStringW("close recsound", None, 0, 0)

        # Open wave audio recorder
        mci_cmd("open new type waveaudio alias recsound")
        # Configure format: 16-bit mono 16kHz
        mci_cmd(f"set recsound time format ms bitspersample 16 channels 1 samplespersec {sr}")
        # Start recording
        mci_cmd("record recsound")

        # Wait for duration
        time.sleep(duration)

        # Stop and save
        mci_cmd("stop recsound")
        mci_cmd(f'save recsound "{temp_wav_path}"')
        mci_cmd("close recsound")

        # Read back recorded WAV
        file_sr, data = wavfile.read(str(temp_wav_path))
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            audio = data.astype(np.float32) / max_val
        else:
            audio = data.astype(np.float32)

        # Clean up temp file if not requested by user
        if output_path is None and temp_wav_path.exists():
            try:
                temp_wav_path.unlink()
            except Exception:
                pass

        return audio

    except Exception as e:
        winmm.mciSendStringW("close recsound", None, 0, 0)
        raise RuntimeError(f"WinMM audio recording failed: {e}")


def record_audio(
    duration: float = 2.5,
    sr: int = 16000,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Record live audio from the default system microphone for `duration` seconds.

    Returns:
      audio: np.ndarray (float32, 1D mono, normalized)
    """
    if verbose:
        print(f"[Recording] Speak now for {duration:.1f} seconds...")

    # Attempt sounddevice first if installed
    try:
        import sounddevice as sd  # type: ignore
        samples = int(duration * sr)
        recording = sd.rec(samples, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        audio = recording.flatten()
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(out_p), sr, (audio * 32767).astype(np.int16))
    except (ImportError, Exception):
        # Native Windows fallback via WinMM
        audio = _record_with_winmm(duration=duration, sr=sr, output_path=output_path)

    if verbose:
        print(f"[Recording] Done. Captured {len(audio)} samples ({len(audio)/sr:.2f}s).")

    return audio


if __name__ == "__main__":
    print("Testing audio_recorder.py (requires working microphone)...")
    try:
        test_clip = record_audio(duration=1.0, verbose=True)
        print(f"Captured audio shape: {test_clip.shape}, dtype: {test_clip.dtype}")
        print("Peak level:", np.max(np.abs(test_clip)))
        print("Recorder working successfully!")
    except Exception as exc:
        print(f"Recorder note (headless/no mic): {exc}")

