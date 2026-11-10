"""
hierarchy.py - Hierarchical Verification Pipeline (Mode D - Member 3)
Part of: Hierarchical Multimodal Biometric Verification System

Three sequential levels, each one only runs if the previous one passed:

    Level 1  FACE            s_face   <  T_low     -> ACCESS DENIED (level_reached = 0)
    Level 2  FACE+VOICE      s_fusion <  T_fusion  -> ACCESS DENIED (level_reached = 1)
             (early fusion)
    Level 3  FINGERPRINT     s_fp     <  T_fp      -> stay at Level 2 permissions
                                                      (or reject if require_fingerprint=True)

    level_reached: 0 = failed Level 1, 1 = passed L1 only, 2 = passed L1+L2, 3 = passed all.

Why a hierarchy?
  * Security: an impostor must fool every level (a logical AND).
  * Time: impostors are usually stopped at Level 1, so the passphrase never has to be
    recorded/processed and no fingerprint has to be scanned.
  * Usability: the cheap, lenient first factor keeps genuine users moving.

Thresholds (T_low, T_fusion, T_fp) are chosen on VALIDATION data only and then frozen.

Interface contract (team_tasks_and_checklists.md, Section 4):
    hierarchical_verify(user_id, face_img, audio, fingerprint_img=None)
        -> {"decision": bool, "level_reached": int, "scores": {...}}
`audio` and `fingerprint_img` may also be zero-argument callables (e.g. a microphone
recorder). They are then called ONLY when the previous level passed, which is exactly
what makes the hierarchy cheaper than asking every user for every modality.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BUNDLE_DIR = Path("models/hierarchy")
LEVEL_NAMES = {0: "Denied at Level 1 (face)", 1: "Denied at Level 2 (voice+fusion)",
               2: "Level 2 passed (face+voice)", 3: "Level 3 passed (face+voice+fingerprint)"}


@dataclass
class Thresholds:
    """Frozen decision thresholds of one user's hierarchy."""

    t_low: float = 0.5
    t_fusion: float = 0.5
    t_fp: float = 0.5
    policy: str = "eer"
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vectorised cascade logic (used for evaluation on pre-computed scores)
# ---------------------------------------------------------------------------
def cascade_levels(
    s_face: np.ndarray,
    s_fusion: np.ndarray,
    s_fp: Optional[np.ndarray],
    th: Thresholds,
) -> np.ndarray:
    """Level reached (0..3) for every attempt, following the sequential logic."""
    s_face = np.asarray(s_face, dtype=float)
    s_fusion = np.asarray(s_fusion, dtype=float)
    lvl = np.zeros(len(s_face), dtype=int)
    p1 = s_face >= th.t_low
    lvl[p1] = 1
    p2 = p1 & (s_fusion >= th.t_fusion)
    lvl[p2] = 2
    if s_fp is not None:
        p3 = p2 & (np.asarray(s_fp, dtype=float) >= th.t_fp)
        lvl[p3] = 3
    return lvl


def cascade_score(
    s_face: np.ndarray,
    s_fusion: np.ndarray,
    s_fp: Optional[np.ndarray],
    th: Thresholds,
) -> np.ndarray:
    """
    One scalar per attempt so ROC / AUC / EER can be drawn for a cascade.

    Each stage score is shifted by its own frozen threshold, so 0.5 is the decision
    boundary of that stage; the cascade score is the WORST stage (min = logical AND).
    Therefore `score >= 0.5`  <=>  the attempt passes every level, i.e.
    evaluate(y, cascade_score, threshold=0.5) reproduces the real cascade decisions.
    Sweeping a different threshold on this score slides all stage thresholds together.
    """
    z = [np.asarray(s_face, dtype=float) - th.t_low,
         np.asarray(s_fusion, dtype=float) - th.t_fusion]
    if s_fp is not None:
        z.append(np.asarray(s_fp, dtype=float) - th.t_fp)
    return np.min(np.vstack(z), axis=0) + 0.5


def expected_time(
    levels: np.ndarray,
    t_face: float,
    t_voice: float,
    t_fp: float,
    use_fp: bool = True,
    t_voice_capture: float = 0.0,
    t_fp_capture: float = 0.0,
) -> float:
    """
    Average time per attempt of the hierarchy for the given outcome levels
    (0: only the face was processed, 1: face + voice, >=2: all stages that exist).
    The "flat" system always pays t_face + t_voice (+ t_fp when use_fp).
    """
    levels = np.asarray(levels)
    t = np.full(len(levels), t_face, dtype=float)
    ran_voice = levels >= 1
    t[ran_voice] += t_voice + t_voice_capture
    if use_fp:
        ran_fp = levels >= 2
        t[ran_fp] += t_fp + t_fp_capture
    return float(np.mean(t))


# ---------------------------------------------------------------------------
# Trained bundle (models + frozen thresholds) for the live / demo path
# ---------------------------------------------------------------------------
@dataclass
class GateBundle:
    user_id: str
    face_model: Any
    fusion_model: Any
    fp_model: Any
    thresholds: Thresholds
    meta: Dict[str, Any] = field(default_factory=dict)


def save_bundle(bundle: GateBundle, directory=BUNDLE_DIR) -> Path:
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / f"hier_{bundle.user_id}.joblib"
    joblib.dump(bundle, path)
    _BUNDLES[bundle.user_id] = bundle
    with open(Path(directory) / f"thresholds_{bundle.user_id}.json", "w") as fh:
        json.dump({"user_id": bundle.user_id, **asdict(bundle.thresholds), "models": bundle.meta}, fh, indent=2, default=str)
    return path


_BUNDLES: Dict[str, GateBundle] = {}


def load_bundle(user_id: str, directory=BUNDLE_DIR) -> GateBundle:
    if user_id not in _BUNDLES:
        path = Path(directory) / f"hier_{user_id}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No hierarchy bundle for '{user_id}' ({path}). Run run_member3_experiments.py first "
                f"(the user needs face + voice + fingerprint data)."
            )
        _BUNDLES[user_id] = joblib.load(path)
    return _BUNDLES[user_id]


def _resolve(x):
    return x() if callable(x) else x


def hierarchical_verify(
    user_id: str,
    face_img: Union[np.ndarray, Callable[[], np.ndarray]],
    audio: Union[str, Path, np.ndarray, Callable[[], Any], None],
    fingerprint_img: Union[str, Path, np.ndarray, Callable[[], Any], None] = None,
    require_fingerprint: bool = False,
    bundle_dir=BUNDLE_DIR,
) -> Dict[str, Any]:
    """
    Standard Interface Contract for Member 3.

    Returns
      {
        "decision":      bool   # access granted (Admin Mode)
        "level_reached": int    # 0..3, see module docstring
        "scores":        {"face": float|None, "fusion": float|None, "fingerprint": float|None}
        "stopped_at":    str    # "level1" | "level2" | "level3" | "completed"
        "timing_ms":     {...}  # per-stage processing time (shows the early-exit saving)
        "thresholds":    {...}
      }

    decision is True when Level 2 is passed (Level 3 optional: it only upgrades the
    permissions). With require_fingerprint=True the fingerprint is mandatory and
    decision requires Level 3.
    """
    from src.fingerprint import extract_fingerprint_features, preprocess_fingerprint
    from src.fusion import face_features_from_image, voice_features_from_audio

    b = load_bundle(user_id, bundle_dir)
    th = b.thresholds
    scores: Dict[str, Optional[float]] = {"face": None, "fusion": None, "fingerprint": None}
    timing: Dict[str, float] = {}
    level = 0

    def finish(stopped: str) -> Dict[str, Any]:
        decision = level >= 3 if require_fingerprint else level >= 2
        return {"decision": bool(decision), "level_reached": level, "scores": scores,
                "stopped_at": stopped, "timing_ms": timing,
                "thresholds": {"T_low": th.t_low, "T_fusion": th.t_fusion, "T_fp": th.t_fp}}

    # ---- Level 1: face ----------------------------------------------------
    t0 = time.perf_counter()
    f_feat = face_features_from_image(_resolve(face_img))
    scores["face"] = 0.0 if f_feat is None else float(b.face_model.predict_proba(f_feat[None, :])[0, 1])
    timing["level1_face"] = (time.perf_counter() - t0) * 1000
    if scores["face"] < th.t_low:
        return finish("level1")
    level = 1

    # ---- Level 2: voice + early fusion ------------------------------------
    t0 = time.perf_counter()
    v_feat = voice_features_from_audio(_resolve(audio)) if audio is not None else None
    if f_feat is None or v_feat is None:
        scores["fusion"] = 0.0  # fail closed: missing/silent passphrase
    else:
        scores["fusion"] = float(b.fusion_model.predict_proba(np.concatenate([f_feat, v_feat])[None, :])[0, 1])
    timing["level2_voice_fusion"] = (time.perf_counter() - t0) * 1000
    if scores["fusion"] < th.t_fusion:
        return finish("level2")
    level = 2

    # ---- Level 3: fingerprint ---------------------------------------------
    fp_raw = _resolve(fingerprint_img) if fingerprint_img is not None else None
    if fp_raw is None:
        return finish("completed" if not require_fingerprint else "level3")
    t0 = time.perf_counter()
    fp_gray = preprocess_fingerprint(str(fp_raw) if isinstance(fp_raw, Path) else fp_raw)
    scores["fingerprint"] = 0.0 if fp_gray is None else float(
        b.fp_model.predict_proba(extract_fingerprint_features(fp_gray, mode="both")[None, :])[0, 1])
    timing["level3_fingerprint"] = (time.perf_counter() - t0) * 1000
    if scores["fingerprint"] >= th.t_fp:
        level = 3
        return finish("completed")
    return finish("level3")
