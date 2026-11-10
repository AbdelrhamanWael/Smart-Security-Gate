"""
fusion.py - Early Fusion of Face + Voice (Mode C - Member 3)
Part of: Hierarchical Multimodal Biometric Verification System

Early (feature-level) fusion:
    face vector (HOG+LBP, 2404-D)   voice vector (MFCC+dMFCC+ZCR+RMS stats, 56-D)
            |                                   |
      StandardScaler (face)             StandardScaler (voice)      <- fitted per modality,
            |                                   |                      on TRAINING data only
      PCA -> n_pca dims                         |
            |                                   |
      block-normalise to unit                   |
      average variance                          |
            |                                   |
         x w_face                           x w_voice               <- block weights (tuned)
            +---------------- concatenate -----+
                               |
                      SVM (RBF)  or  Random Forest

Why block weights?  After the PCA the face block has far fewer dimensions than
the raw 2404-D vector, but the two blocks can still contribute unequally to the
distance used by the SVM. w_face / w_voice let us tune which modality is trusted
more, using validation data only (experiment E6).

Interface contract (team_tasks_and_checklists.md, Section 4):
    fuse(face_vec, voice_vec, weights=None)      -> np.ndarray
    fusion_score(face_img, audio, user_id)       -> float in [0, 1]
"""

from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

VOICE_DIM = 56    # MFCC(13)+dMFCC(13) mean/std + ZCR + RMS stats; face dim is inferred (HOG+LBP = 8740 here)
FUSION_DIR = Path("models/fusion")


class EarlyFusionTransformer(BaseEstimator, TransformerMixin):
    """
    Input  : X = [face_features | voice_features]  (the LAST n_voice columns are voice)
    Output : [w_face * PCA(scaled face) | w_voice * scaled voice]

    Everything (scalers, PCA, block normalisation) is fitted in `fit`, so when
    this transformer sits inside a Pipeline that is cross-validated, nothing
    leaks from the validation fold.
    """

    def __init__(self, n_voice: int = VOICE_DIM, n_pca: int = 30, w_face: float = 1.0, w_voice: float = 1.0):
        self.n_voice = n_voice
        self.n_pca = n_pca
        self.w_face = w_face
        self.w_voice = w_voice

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        F, V = X[:, : -self.n_voice], X[:, -self.n_voice:]
        self.face_scaler_ = StandardScaler().fit(F)
        Fs = self.face_scaler_.transform(F)
        k = int(max(2, min(self.n_pca, Fs.shape[0] - 1, Fs.shape[1])))
        self.pca_ = PCA(n_components=k, random_state=42).fit(Fs)
        # Make the face block have the same average per-dimension variance (=1)
        # as the standardised voice block, so the weights are directly comparable.
        self.face_block_scale_ = 1.0 / np.sqrt(np.mean(self.pca_.explained_variance_) + 1e-12)
        self.voice_scaler_ = StandardScaler().fit(V)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        F, V = X[:, : -self.n_voice], X[:, -self.n_voice:]
        f = self.pca_.transform(self.face_scaler_.transform(F)) * self.face_block_scale_
        v = self.voice_scaler_.transform(V)
        return np.hstack([self.w_face * f, self.w_voice * v])


def build_fusion_pipeline(
    model_type: str = "svm",
    n_pca: int = 30,
    w_face: float = 1.0,
    w_voice: float = 1.0,
    C: float = 5.0,
    max_depth: Optional[int] = None,
    n_estimators: int = 300,
) -> Pipeline:
    """Mode C classifier: early-fusion transformer + SVM(RBF) or Random Forest."""
    fuse_step = EarlyFusionTransformer(n_pca=n_pca, w_face=w_face, w_voice=w_voice)
    if model_type == "svm":
        clf = SVC(kernel="rbf", C=C, gamma="scale", probability=True,
                  class_weight="balanced", random_state=42)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([("fuse", fuse_step), ("clf", clf)])


FUSION_PARAM_GRID: Dict[str, list] = {
    "svm": [dict(n_pca=n, w_voice=w, C=c) for n in (20, 40) for w in (0.5, 1.0, 2.0) for c in (1.0, 5.0)],
    "rf": [dict(n_pca=n, max_depth=d) for n in (20, 40) for d in (None, 10)],
}


# ---------------------------------------------------------------------------
# Interface contract: fuse()
# ---------------------------------------------------------------------------
def fuse(
    face_vec: np.ndarray,
    voice_vec: np.ndarray,
    weights: Optional[Tuple[float, float]] = None,
    transformer: Optional[EarlyFusionTransformer] = None,
) -> np.ndarray:
    """
    Build one fused vector from a face vector and a voice vector.

    * With a fitted `transformer` (the one inside a trained fusion pipeline) this is
      exactly what the classifier sees: scaled (training statistics) + PCA + weights.
    * Without one, a stateless fallback is used: each block is L2-normalised and
      the blocks are concatenated with the given weights (default 1:1).
    """
    face_vec = np.asarray(face_vec, dtype=np.float64).ravel()
    voice_vec = np.asarray(voice_vec, dtype=np.float64).ravel()
    if transformer is not None:
        if weights is not None:  # override the block weights on a shallow copy (fitted state is shared)
            transformer = copy.copy(transformer)
            transformer.w_face, transformer.w_voice = weights
        return transformer.transform(np.concatenate([face_vec, voice_vec])[None, :])[0]
    w_f, w_v = weights if weights is not None else (1.0, 1.0)
    f = face_vec / (np.linalg.norm(face_vec) + 1e-12)
    v = voice_vec / (np.linalg.norm(voice_vec) + 1e-12)
    return np.concatenate([w_f * f, w_v * v])


# ---------------------------------------------------------------------------
# Raw-input -> feature helpers (same preprocessing as M1 / M2 live code)
# ---------------------------------------------------------------------------
def face_features_from_image(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """crop_face (Haar + CLAHE) + HOG/LBP. None if no face is detected."""
    from src.face_pipeline import crop_face, extract_face_features
    face = crop_face(image_bgr)
    return None if face is None else extract_face_features(face, mode="both")


def voice_features_from_audio(audio: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    """Same gain handling as VoiceVerifier.predict_score. None for dead silence."""
    from src.voice_features import extract_voice_features, load_audio, normalize_amplitude
    if isinstance(audio, (str, Path)):
        data, _ = load_audio(audio)
    else:
        data = np.asarray(audio, dtype=np.float32)
    if float(np.max(np.abs(data))) < 0.005:  # no microphone signal
        return None
    return extract_voice_features(normalize_amplitude(data, target_peak=0.95))


# ---------------------------------------------------------------------------
# Interface contract: fusion_score()
# ---------------------------------------------------------------------------
_FUSION_CACHE: Dict[str, Pipeline] = {}


def save_fusion_model(pipeline: Pipeline, user_id: str, model_type: str, directory=FUSION_DIR) -> Path:
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"fusion_{user_id}_{model_type}.joblib"
    joblib.dump(pipeline, path)
    _FUSION_CACHE[f"{user_id}_{model_type}"] = pipeline
    return path


def fusion_score(
    face_img: np.ndarray,
    audio: Union[str, Path, np.ndarray],
    user_id: str,
    model_type: str = "svm",
    models_dir=FUSION_DIR,
) -> float:
    """
    Standard Interface Contract for Member 3:
        fusion_score(face_img, audio, user_id) -> float   # match probability [0, 1]

    Returns 0.0 if no face is detected or the audio is silent (fail closed).
    """
    f = face_features_from_image(face_img)
    v = voice_features_from_audio(audio)
    if f is None or v is None:
        return 0.0
    key = f"{user_id}_{model_type}"
    if key not in _FUSION_CACHE:
        path = Path(models_dir) / f"fusion_{user_id}_{model_type}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Fusion model not found: {path}. Run run_member3_experiments.py first.")
        _FUSION_CACHE[key] = joblib.load(path)
    return float(_FUSION_CACHE[key].predict_proba(np.concatenate([f, v])[None, :])[0, 1])
