"""
voice_dataset.py - Dataset Loader & Session Splitter (Member 2)
Part of: Hierarchical Multimodal Biometric Verification System

Responsibilities:
- Scans and loads audio files from the standardized directory structure:
    data/voice/{person_id}/{session_id}/*.wav
- Extracts and caches 56-D voice feature vectors to ensure fast iteration.
- Enforces session-based grouping (GroupKFold) to prevent data leakage.
- Supports open-set evaluation by holding out unseen impostors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from src.voice_features import extract_voice_features
except ImportError:
    from voice_features import extract_voice_features


@dataclass
class VoiceSample:
    filepath: Path
    person_id: str
    session_id: str
    utterance_type: str = "passphrase"
    features: Optional[np.ndarray] = None


class VoiceDataset:
    def __init__(self, data_root: Union[str, Path] = "data"):
        self.data_root = Path(data_root)
        self.voice_root = self.data_root / "voice"
        self.voice_samples: List[VoiceSample] = []

    def scan_voice_data(self) -> List[VoiceSample]:
        """Scan data/voice directory for passphrase recordings."""
        samples: List[VoiceSample] = []
        if not self.voice_root.exists():
            return samples

        for wav_path in self.voice_root.glob("*/*/*.wav"):
            session_id = wav_path.parent.name
            person_id = wav_path.parent.parent.name
            sample = VoiceSample(
                filepath=wav_path,
                person_id=person_id,
                session_id=session_id,
                utterance_type="passphrase",
            )
            samples.append(sample)

        self.voice_samples = samples
        return samples

    def load_features_for_samples(
        self,
        samples: List[VoiceSample],
        cache_file: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """
        Extract or load cached features for the given samples.

        Returns:
          X: np.ndarray of shape (N, 56)
          person_ids: list of person_id strings
          session_ids: list of session_id strings
          utterance_types: list of utterance strings
        """
        if cache_file and Path(cache_file).exists():
            if verbose:
                print(f"Loading cached features from {cache_file}...")
            cached = np.load(str(cache_file), allow_pickle=True)
            return (
                cached["X"],
                list(cached["person_ids"]),
                list(cached["session_ids"]),
                list(cached["utterance_types"]),
            )

        if verbose:
            print(f"Extracting voice features for {len(samples)} samples...")

        feat_list = []
        person_ids = []
        session_ids = []
        utterance_types = []

        for idx, sample in enumerate(samples):
            try:
                feats = extract_voice_features(sample.filepath)
                sample.features = feats
                feat_list.append(feats)
                person_ids.append(sample.person_id)
                session_ids.append(sample.session_id)
                utterance_types.append(sample.utterance_type)
            except Exception as e:
                print(f"Warning: Failed to extract features from {sample.filepath}: {e}")

            if verbose and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(samples)} samples...")

        X = np.array(feat_list, dtype=np.float64)

        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(cache_file),
                X=X,
                person_ids=person_ids,
                session_ids=session_ids,
                utterance_types=utterance_types,
            )
            if verbose:
                print(f"Features cached to {cache_file}")

        return X, person_ids, session_ids, utterance_types


def get_voice_verification_split(
    target_user: str,
    X: np.ndarray,
    person_ids: List[str],
    session_ids: List[str],
    unseen_impostors: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds biometric verification split for `target_user`:
    - Label 1 = Genuine (target_user)
    - Label 0 = Impostor (other speakers)

    Splits data into:
      - Train/Val set (sessions 1 and 2, with seen speakers)
      - Final Test set (session 3 for genuine + unseen impostors for open-set evaluation)

    Returns:
      (X_train, y_train, groups_train, X_test, y_test, groups_test)
    """
    if unseen_impostors is None:
        unseen_impostors = ["impostor4", "impostor5"]

    train_indices = []
    test_indices = []

    for i, (pid, sid) in enumerate(zip(person_ids, session_ids)):
        if pid in unseen_impostors:
            # Unseen impostors go strictly to the final open-set test
            test_indices.append(i)
        elif sid in ("session1", "session2"):
            # Development / Cross-validation sessions
            train_indices.append(i)
        else:
            # session3 (held-out evaluation session)
            test_indices.append(i)

    train_idx = np.array(train_indices)
    test_idx = np.array(test_indices)

    y_all = np.array([1 if pid == target_user else 0 for pid in person_ids], dtype=int)
    groups_all = np.array(session_ids)

    return (
        X[train_idx],
        y_all[train_idx],
        groups_all[train_idx],
        X[test_idx],
        y_all[test_idx],
        groups_all[test_idx],
    )


