"""
voice_features.py - Audio Preprocessing & Feature Extraction Engine (Member 2)
Part of: Hierarchical Multimodal Biometric Verification System

Responsibilities:
1. Audio loading (WAV file or numpy array) & mono/resampling to 16 kHz.
2. Amplitude normalization and DC offset removal.
3. Voice Activity Detection (VAD) / Silence trimming using sliding-window RMS energy.
4. Traditional DSP Mel-Frequency Cepstral Coefficients (MFCC, 13 coefficients + 13 deltas).
5. Zero-Crossing Rate (ZCR) with mean and standard deviation.
6. Root-Mean-Square (RMS) Energy with mean and standard deviation.
7. Fixed-length feature vector generation (56 dimensions).

Constraint: Strictly traditional DSP and machine learning (No Deep Learning).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import scipy.fft as fft
import scipy.io.wavfile as wavfile
import scipy.signal as signal

try:
    import soundfile as sf
except ImportError:
    sf = None


TARGET_SAMPLE_RATE = 16000
FRAME_LENGTH_SEC = 0.025   # 25 ms
HOP_LENGTH_SEC = 0.010     # 10 ms
N_FFT = 512
N_MELS = 26
N_MFCC = 13
PREEMPH_COEFF = 0.97
CEP_LIFTER = 22


def load_audio(
    wav_path_or_array: Union[str, Path, np.ndarray],
    sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio source, convert stereo to mono, normalize to [-1.0, 1.0],
    and resample to the target sample rate (default 16 kHz).
    """
    if isinstance(wav_path_or_array, (str, Path)):
        path = Path(wav_path_or_array)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() == ".wav":
            file_sr, data = wavfile.read(str(path))
        else:
            if sf is not None:
                data, file_sr = sf.read(str(path), dtype="float32")
            else:
                file_sr, data = wavfile.read(str(path))
    elif isinstance(wav_path_or_array, np.ndarray):
        file_sr = sr
        data = wav_path_or_array
    else:
        raise TypeError(
            f"Expected filepath or numpy array, got {type(wav_path_or_array)}"
        )

    # Convert to float32
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float32) / max_val
    else:
        audio = data.astype(np.float32)

    # Convert stereo to mono by averaging channels
    if audio.ndim > 1:
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)
        elif audio.shape[0] == 2:
            audio = np.mean(audio, axis=0)
        else:
            audio = audio.flatten()

    # Resample if needed
    if file_sr != sr and len(audio) > 0:
        gcd = math.gcd(file_sr, sr)
        up = sr // gcd
        down = file_sr // gcd
        audio = signal.resample_poly(audio, up, down).astype(np.float32)

    return audio, sr


def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Remove DC offset and normalize peak amplitude to target_peak.
    """
    if len(audio) == 0:
        return audio

    # Remove DC bias
    audio = audio - np.mean(audio)

    # Peak normalization
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = (audio / max_val) * target_peak

    return audio.astype(np.float32)


def trim_silence_rms(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    frame_sec: float = FRAME_LENGTH_SEC,
    hop_sec: float = HOP_LENGTH_SEC,
    threshold_factor: float = 0.08,
    margin_sec: float = 0.05,
) -> np.ndarray:
    """
    Voice Activity Detection (VAD) via sliding-window RMS energy.
    Trims leading and trailing silence while preserving speech and a safety margin.
    """
    if len(audio) == 0:
        return audio

    frame_len = max(1, int(sr * frame_sec))
    hop_len = max(1, int(sr * hop_sec))

    if len(audio) < frame_len:
        return audio

    # Compute frame-wise RMS
    num_frames = 1 + (len(audio) - frame_len) // hop_len
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    max_rms = np.max(rms)
    noise_floor = np.percentile(rms, 10)
    # Adaptive threshold between noise floor and maximum peak
    vad_threshold = max(noise_floor * 1.5, max_rms * threshold_factor)

    active_frames = np.where(rms >= vad_threshold)[0]
    if len(active_frames) == 0:
        # If quiet signal, return the whole audio
        return audio

    first_frame = active_frames[0]
    last_frame = active_frames[-1]

    margin_samples = int(sr * margin_sec)
    start_sample = max(0, first_frame * hop_len - margin_samples)
    end_sample = min(len(audio), (last_frame + 1) * hop_len + frame_len + margin_samples)

    trimmed = audio[start_sample:end_sample]
    if len(trimmed) < frame_len:
        return audio
    return trimmed


def _hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Mel scale to Frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sr: int = TARGET_SAMPLE_RATE,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Construct triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    num_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(min(f_max, sr / 2.0))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, num_bins - 1)

    fbank = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        if f_center > f_left:
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        if f_right > f_center:
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    return fbank


def compute_mfcc(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    frame_len: int = int(TARGET_SAMPLE_RATE * FRAME_LENGTH_SEC),
    hop_len: int = int(TARGET_SAMPLE_RATE * HOP_LENGTH_SEC),
    preemph: float = PREEMPH_COEFF,
    cep_lifter: int = CEP_LIFTER,
) -> np.ndarray:
    """
    Compute MFCC coefficients for framed audio.
    Returns: (num_frames, n_mfcc)
    """
    if len(audio) < frame_len:
        # Pad with zeros if shorter than one frame
        audio = np.pad(audio, (0, frame_len - len(audio)), mode="constant")

    # 1. Pre-emphasis filter: y[t] = x[t] - preemph * x[t-1]
    emphasized = np.append(audio[0], audio[1:] - preemph * audio[:-1])

    # 2. Framing
    num_frames = 1 + (len(emphasized) - frame_len) // hop_len
    shape = (num_frames, frame_len)
    strides = (emphasized.strides[0] * hop_len, emphasized.strides[0])
    frames = np.lib.stride_tricks.as_strided(emphasized, shape=shape, strides=strides)

    # 3. Windowing (Hamming window)
    window = np.hamming(frame_len).astype(np.float32)
    windowed_frames = frames * window

    # 4. FFT and Power Spectrum
    # N_FFT must be at least frame_len
    fft_size = max(n_fft, 1 << (frame_len - 1).bit_length())
    mag_spectrum = np.abs(fft.rfft(windowed_frames, n=fft_size, axis=1))
    power_spectrum = (mag_spectrum ** 2) / float(fft_size)

    # 5. Mel Filterbank application
    fbank = build_mel_filterbank(n_mels=n_mels, n_fft=fft_size, sr=sr)
    mel_energies = np.dot(power_spectrum, fbank.T)
    mel_energies = np.where(mel_energies == 0, 1e-10, mel_energies)
    log_mel = np.log(mel_energies)

    # 6. Discrete Cosine Transform (DCT Type II, orthonormal)
    mfccs = fft.dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

    # 7. Sinusoidal Cepstral Liftering
    if cep_lifter > 0:
        n = np.arange(n_mfcc)
        lift = 1.0 + (cep_lifter / 2.0) * np.sin(np.pi * n / cep_lifter)
        mfccs = mfccs * lift

    return mfccs.astype(np.float32)


def compute_deltas(features: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute delta (velocity) features along time axis (axis 0).
    Uses the standard regression formula:
    d_t = sum_{n=1}^N n * (c_{t+n} - c_{t-n}) / (2 * sum_{n=1}^N n^2)
    """
    num_frames, num_feats = features.shape
    deltas = np.zeros_like(features)
    denom = 2 * sum(i ** 2 for i in range(1, order + 1))

    # Pad edges symmetrically
    padded = np.pad(features, ((order, order), (0, 0)), mode="edge")

    for t in range(num_frames):
        num = np.zeros(num_feats, dtype=features.dtype)
        for n in range(1, order + 1):
            num += n * (padded[t + order + n] - padded[t + order - n])
        deltas[t] = num / denom

    return deltas.astype(np.float32)


def compute_zcr(
    audio: np.ndarray,
    frame_len: int = int(TARGET_SAMPLE_RATE * FRAME_LENGTH_SEC),
    hop_len: int = int(TARGET_SAMPLE_RATE * HOP_LENGTH_SEC),
) -> np.ndarray:
    """
    Compute frame-wise Zero-Crossing Rate (ZCR).
    Returns: (num_frames,)
    """
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)), mode="constant")

    num_frames = 1 + (len(audio) - frame_len) // hop_len
    shape = (num_frames, frame_len)
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    # Sign change detection
    signs = np.sign(frames)
    # Treat exact 0 as positive to avoid spurious crossings
    signs[signs == 0] = 1
    zc = np.sum(np.abs(signs[:, 1:] - signs[:, :-1]) > 0, axis=1)
    zcr = zc / float(2 * (frame_len - 1))
    return zcr.astype(np.float32)


def compute_rms(
    audio: np.ndarray,
    frame_len: int = int(TARGET_SAMPLE_RATE * FRAME_LENGTH_SEC),
    hop_len: int = int(TARGET_SAMPLE_RATE * HOP_LENGTH_SEC),
) -> np.ndarray:
    """
    Compute frame-wise Root-Mean-Square (RMS) Energy.
    Returns: (num_frames,)
    """
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)), mode="constant")

    num_frames = 1 + (len(audio) - frame_len) // hop_len
    shape = (num_frames, frame_len)
    strides = (audio.strides[0] * hop_len, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return rms.astype(np.float32)


def get_voice_feature_names() -> list[str]:
    """
    Return descriptive names for all elements in the 56-D feature vector.
    """
    names = []
    # 13 MFCC means
    for i in range(N_MFCC):
        names.append(f"mfcc_{i+1}_mean")
    # 13 MFCC stds
    for i in range(N_MFCC):
        names.append(f"mfcc_{i+1}_std")
    # 13 Delta MFCC means
    for i in range(N_MFCC):
        names.append(f"delta_mfcc_{i+1}_mean")
    # 13 Delta MFCC stds
    for i in range(N_MFCC):
        names.append(f"delta_mfcc_{i+1}_std")
    # ZCR & RMS
    names.append("zcr_mean")
    names.append("zcr_std")
    names.append("rms_mean")
    names.append("rms_std")
    return names


def extract_voice_features(
    wav_path_or_array: Union[str, Path, np.ndarray],
    sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Interface Contract for Member 2:
    extract_voice_features(wav_path_or_array) -> np.ndarray (shape (56,))

    Pipeline:
      1. Load audio & resample to 16 kHz mono.
      2. Normalize amplitude & remove DC offset.
      3. Silence trimming / VAD using RMS energy.
      4. Compute 13 MFCCs across frames.
      5. Compute 13 Delta MFCCs across frames.
      6. Compute frame-wise Zero-Crossing Rate (ZCR).
      7. Compute frame-wise RMS Energy.
      8. Calculate mean & std for all features across speech frames.
      9. Combine into a fixed-length 56-dimensional feature vector.
    """
    # 1. Load and resample
    audio, current_sr = load_audio(wav_path_or_array, sr=sr)

    # 2. Normalize
    audio = normalize_amplitude(audio)

    # 3. Silence trimming (VAD)
    audio = trim_silence_rms(audio, sr=current_sr)

    # Ensure audio has sufficient samples for at least one frame
    frame_len = int(current_sr * FRAME_LENGTH_SEC)
    hop_len = int(current_sr * HOP_LENGTH_SEC)
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)), mode="constant")

    # 4. MFCC (num_frames, 13)
    mfccs = compute_mfcc(
        audio,
        sr=current_sr,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        frame_len=frame_len,
        hop_len=hop_len,
    )

    # 5. Delta MFCC (num_frames, 13)
    deltas = compute_deltas(mfccs)

    # 6. ZCR (num_frames,)
    zcr = compute_zcr(audio, frame_len=frame_len, hop_len=hop_len)

    # 7. RMS (num_frames,)
    rms = compute_rms(audio, frame_len=frame_len, hop_len=hop_len)

    # 8. Compute statistical moments (mean, std)
    mfcc_mean = np.mean(mfccs, axis=0)
    mfcc_std = np.std(mfccs, axis=0)

    delta_mean = np.mean(deltas, axis=0)
    delta_std = np.std(deltas, axis=0)

    zcr_mean = np.array([np.mean(zcr)], dtype=np.float32)
    zcr_std = np.array([np.std(zcr)], dtype=np.float32)

    rms_mean = np.array([np.mean(rms)], dtype=np.float32)
    rms_std = np.array([np.std(rms)], dtype=np.float32)

    # 9. Concatenate into fixed-length 56-D vector
    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        zcr_mean,
        zcr_std,
        rms_mean,
        rms_std,
    ]).astype(np.float64)

    # Replace any potential NaNs or Infs with 0.0 for numerical safety
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector


if __name__ == "__main__":
    print("Testing voice_features.py...")
    # Generate test sine chirp audio signal
    t = np.linspace(0, 1.5, int(TARGET_SAMPLE_RATE * 1.5), endpoint=False)
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    # Add brief silence at beginning and end
    test_audio = np.concatenate([
        np.zeros(1600, dtype=np.float32),
        synthetic_signal.astype(np.float32),
        np.zeros(1600, dtype=np.float32),
    ])

    feats = extract_voice_features(test_audio, sr=TARGET_SAMPLE_RATE)
    names = get_voice_feature_names()
    print(f"Feature vector shape: {feats.shape}")
    print(f"Number of feature names: {len(names)}")
    print(f"First 5 features ({names[:5]}): {feats[:5]}")
    print(f"ZCR mean: {feats[52]:.4f}, RMS mean: {feats[54]:.4f}")
    assert feats.shape == (56,), "Feature vector must have length 56!"
    print("voice_features.py self-test passed successfully!")

