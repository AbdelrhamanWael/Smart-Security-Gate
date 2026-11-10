"""
multimodal_dataset.py - Paired Face/Voice/Fingerprint Dataset Builder (Member 3)
Part of: Hierarchical Multimodal Biometric Verification System

Problem
-------
Early fusion and the hierarchy need *paired* samples (face + voice + fingerprint
of the same attempt). Our impostor data comes from different public sources
(LFW faces, other speakers, SOCOFing/FVC fingers), so impostor pairs have to be
constructed. This is the standard "virtual / chimeric user" protocol.

Protocol (for one target user)
------------------------------
Genuine attempts   : face + voice + fingerprint of the SAME session of the target.
Impostor attempts  : "virtual impostor identities". Each virtual identity owns a
                     fixed set of voice ids, face ids and finger ids, and every
                     identity (person) belongs to exactly ONE virtual identity.
                     Natural pairs are kept when we really have them (e.g. the
                     face+voice+finger of another team member).

Leakage prevention
------------------
* DEV  (training / validation / threshold freezing):
      genuine  = sessions 1 and 2 (group = session id)
      impostor = SEEN virtual identities (voice: sessions 1-2 only)
* TEST (final open-set evaluation, never touched during tuning):
      genuine  = session 3 (raw, non-augmented images only)
      impostor = UNSEEN virtual identities (persons never seen in DEV)
* The group label of a row is the session (genuine) or the virtual identity
  (impostor), so GroupKFold can never split one session / one person across
  train and validation.
* Augmented copies of the target are only used in DEV genuine rows. Other
  people are always used with raw images only.
* Scalers / PCA live inside sklearn Pipelines and are fitted on training folds.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.face_pipeline import crop_face, extract_face_features
from src.fingerprint import extract_fingerprint_features, preprocess_fingerprint
from src.voice_features import extract_voice_features

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEV_SESSIONS = ("session1", "session2")
TEST_SESSION = "session3"
UNSEEN_VOICE = ("impostor3", "impostor4", "impostor5")
UNSEEN_FINGERS = ("finger_108", "finger_109", "finger_110")
N_UNSEEN_FACES = 3


# ---------------------------------------------------------------------------
# Feature pools
# ---------------------------------------------------------------------------
@dataclass
class Pool:
    """Every sample of ONE modality with its metadata."""

    X: np.ndarray
    person: np.ndarray
    session: np.ndarray
    is_aug: np.ndarray
    paths: np.ndarray

    def mask(self, persons=None, sessions=None, raw_only=False) -> np.ndarray:
        m = np.ones(len(self.person), dtype=bool)
        if persons is not None:
            m &= np.isin(self.person, list(persons))
        if sessions is not None:
            m &= np.isin(self.session, list(sessions))
        if raw_only:
            m &= ~self.is_aug
        return m

    def persons(self) -> List[str]:
        return sorted(set(self.person.tolist()))


def _save_pool(path: Path, pool: Pool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=pool.X, person=pool.person, session=pool.session,
                        is_aug=pool.is_aug, paths=pool.paths)


def _load_pool(path: Path) -> Pool:
    d = np.load(path, allow_pickle=False)
    return Pool(d["X"], d["person"], d["session"], d["is_aug"], d["paths"])


def _list_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def _identify(rel_parts: Tuple[str, ...]) -> Tuple[str, str]:
    """(person_id, session_id) from a path relative to data/<modality>/."""
    if rel_parts[0] == "impostors":
        person = "imp_" + rel_parts[1] if not rel_parts[1].startswith("finger_") else rel_parts[1]
        session = rel_parts[2] if len(rel_parts) > 3 else "na"
        return person, session
    return rel_parts[0], rel_parts[1]


def load_face_pool(data_dir="data", cache_dir="cache", rebuild=False, verbose=True) -> Pool:
    cache = Path(cache_dir) / "m3_face_pool.npz"
    if cache.exists() and not rebuild:
        return _load_pool(cache)
    root = Path(data_dir) / "face"
    X, person, session, aug, paths, skipped = [], [], [], [], [], 0
    for p in _list_images(root):
        if "_aug_comp" in p.name:  # extreme distortions: same rule as Member 1
            continue
        img = cv2.imread(str(p))
        face = crop_face(img) if img is not None else None
        if face is None:
            skipped += 1
            continue
        pid, sid = _identify(p.relative_to(root).parts)
        X.append(extract_face_features(face, mode="both"))
        person.append(pid); session.append(sid); aug.append("_aug_" in p.name); paths.append(str(p))
    if verbose:
        print(f"[face pool] {len(X)} samples ({skipped} skipped, no face found)")
    pool = Pool(np.array(X, dtype=np.float32), np.array(person), np.array(session),
                np.array(aug), np.array(paths))
    _save_pool(cache, pool)
    return pool


def load_fingerprint_pool(data_dir="data", cache_dir="cache", rebuild=False, verbose=True) -> Pool:
    cache = Path(cache_dir) / "m3_fp_pool.npz"
    if cache.exists() and not rebuild:
        return _load_pool(cache)
    root = Path(data_dir) / "fingerprint"
    X, person, session, aug, paths = [], [], [], [], []
    for p in _list_images(root):
        fp = preprocess_fingerprint(str(p))
        if fp is None:
            continue
        pid, sid = _identify(p.relative_to(root).parts)
        X.append(extract_fingerprint_features(fp, mode="both"))
        person.append(pid); session.append(sid); aug.append("_aug_" in p.name); paths.append(str(p))
    if verbose:
        print(f"[fingerprint pool] {len(X)} samples")
    pool = Pool(np.array(X, dtype=np.float32), np.array(person), np.array(session),
                np.array(aug), np.array(paths))
    _save_pool(cache, pool)
    return pool


def load_voice_pool(data_dir="data", cache_dir="cache", rebuild=False, verbose=True) -> Pool:
    cache = Path(cache_dir) / "m3_voice_pool.npz"
    if cache.exists() and not rebuild:
        return _load_pool(cache)
    root = Path(data_dir) / "voice"
    X, person, session, paths, skipped = [], [], [], [], 0
    for p in sorted(root.glob("*/*/*.wav")):
        try:
            X.append(extract_voice_features(p))
        except Exception as exc:  # corrupt / too short file
            skipped += 1
            print(f"  [warn] {p}: {exc}")
            continue
        person.append(p.parent.parent.name); session.append(p.parent.name); paths.append(str(p))
    if verbose:
        print(f"[voice pool] {len(X)} samples ({skipped} skipped)")
    pool = Pool(np.array(X, dtype=np.float64), np.array(person), np.array(session),
                np.zeros(len(X), dtype=bool), np.array(paths))
    _save_pool(cache, pool)
    return pool


@dataclass
class Pools:
    face: Pool
    voice: Pool
    fp: Pool


def load_pools(data_dir="data", cache_dir="cache", rebuild=False) -> Pools:
    return Pools(
        load_face_pool(data_dir, cache_dir, rebuild),
        load_voice_pool(data_dir, cache_dir, rebuild),
        load_fingerprint_pool(data_dir, cache_dir, rebuild),
    )


def users_with_all_modalities(pools: Pools, members: Sequence[str] = ("member1", "member2", "member3")) -> Tuple[List[str], List[str]]:
    """(usable_users, skipped_users) - a user needs face + voice + fingerprint data."""
    ok, bad = [], []
    for m in members:
        has = [(pools.face.person == m).any(), (pools.voice.person == m).any(), (pools.fp.person == m).any()]
        (ok if all(has) else bad).append(m)
    return ok, bad


# ---------------------------------------------------------------------------
# Paired rows
# ---------------------------------------------------------------------------
@dataclass
class Rows:
    """Rows of the paired dataset, stored as indices into the three pools."""

    face_idx: np.ndarray
    voice_idx: np.ndarray
    fp_idx: np.ndarray
    y: np.ndarray
    groups: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def identity_plan(target: str, pools: Pools) -> Dict[str, List]:
    """Which persons are SEEN (dev) / UNSEEN (test) impostors, per modality."""
    voice_ids = [p for p in pools.voice.persons() if p != target]
    face_ids = [p for p in pools.face.persons() if p != target]
    fp_ids = [p for p in pools.fp.persons() if p != target]

    unseen_v = [p for p in voice_ids if p in UNSEEN_VOICE]
    lfw = [p for p in face_ids if p.startswith("imp_")]
    unseen_f = lfw[-N_UNSEEN_FACES:]
    unseen_p = [p for p in fp_ids if p in UNSEEN_FINGERS]
    return {
        "seen_voice": [p for p in voice_ids if p not in unseen_v],
        "unseen_voice": unseen_v,
        "seen_face": [p for p in face_ids if p not in unseen_f],
        "unseen_face": unseen_f,
        "seen_fp": [p for p in fp_ids if p not in unseen_p],
        "unseen_fp": unseen_p,
    }


def _virtual_identities(voice: List[str], face: List[str], fp: List[str], prefix: str) -> List[Dict]:
    """Each person ends up in exactly one virtual identity (no person shared)."""
    voice, face, fp = list(voice), list(face), list(fp)
    vids = []
    # natural pairs: a real team member for whom we have all three modalities
    for p in sorted(set(voice) & set(face) & set(fp)):
        if p.startswith("member"):
            vids.append({"name": f"{prefix}_{p}", "voice": [p], "face": [p], "fp": [p]})
            voice.remove(p); face.remove(p); fp.remove(p)
    k = len(voice)
    for i in range(k):
        vids.append({"name": f"{prefix}_chim{i}", "voice": [voice[i]],
                     "face": face[i::k], "fp": fp[i::k]})
    return vids


def build_rows(
    target: str,
    pools: Pools,
    seed: int = 42,
    pairs_per_voice: int = 3,
    n_imp_dev: int = 40,
    n_imp_test: int = 25,
) -> Tuple[Rows, Rows, Dict]:
    """Build leakage-safe DEV and TEST rows for `target`. See module docstring."""
    rng = np.random.default_rng(seed)
    plan = identity_plan(target, pools)

    def draw(pool: Pool, persons, sessions=None, raw_only=True) -> int:
        idx = np.flatnonzero(pool.mask(persons, sessions, raw_only))
        if len(idx) == 0:
            idx = np.flatnonzero(pool.mask(persons, None, raw_only))
        if len(idx) == 0:
            idx = np.flatnonzero(pool.mask(persons, None, False))
        if len(idx) == 0:
            raise ValueError(f"No samples for persons={persons}")
        return int(rng.choice(idx))

    def genuine_rows(sessions, raw_only: bool):
        rows = []
        for s in sessions:
            v_idx = np.flatnonzero(pools.voice.mask([target], [s]))
            for v in v_idx:
                for _ in range(pairs_per_voice):
                    rows.append((draw(pools.face, [target], [s], raw_only),
                                 int(v),
                                 draw(pools.fp, [target], [s], raw_only), 1, s))
        return rows

    def impostor_rows(vids, n_each: int, voice_sessions):
        rows = []
        for vid in vids:
            for _ in range(n_each):
                rows.append((draw(pools.face, vid["face"], None, True),
                             draw(pools.voice, vid["voice"], voice_sessions, True),
                             draw(pools.fp, vid["fp"], None, True), 0, vid["name"]))
        return rows

    seen_vids = _virtual_identities(plan["seen_voice"], plan["seen_face"], plan["seen_fp"], "seen")
    unseen_vids = _virtual_identities(plan["unseen_voice"], plan["unseen_face"], plan["unseen_fp"], "unseen")

    dev = genuine_rows(DEV_SESSIONS, raw_only=False) + impostor_rows(seen_vids, n_imp_dev, DEV_SESSIONS)
    test = genuine_rows([TEST_SESSION], raw_only=True) + impostor_rows(unseen_vids, n_imp_test, None)

    def pack(rows) -> Rows:
        a = np.array([r[:4] for r in rows], dtype=int)
        return Rows(a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.array([r[4] for r in rows]))

    dev_r, test_r = pack(dev), pack(test)
    info = {"plan": plan, "seen_vids": seen_vids, "unseen_vids": unseen_vids,
            "n_dev": len(dev_r), "n_test": len(test_r),
            "dev_genuine": int(dev_r.y.sum()), "test_genuine": int(test_r.y.sum())}
    assert_no_leakage(dev_r, test_r, pools)
    return dev_r, test_r, info


def assert_no_leakage(dev: Rows, test: Rows, pools: Pools) -> None:
    """Hard guarantees: no impostor person and no genuine session crosses DEV/TEST."""
    for name, pool, di, ti in (("face", pools.face, dev.face_idx, test.face_idx),
                               ("voice", pools.voice, dev.voice_idx, test.voice_idx),
                               ("fingerprint", pools.fp, dev.fp_idx, test.fp_idx)):
        dev_imp = set(pool.person[di[dev.y == 0]])
        test_imp = set(pool.person[ti[test.y == 0]])
        shared = dev_imp & test_imp
        assert not shared, f"LEAK: impostor persons in both dev and test ({name}): {shared}"
    dev_sessions = set(pools.voice.session[dev.voice_idx[dev.y == 1]])
    test_sessions = set(pools.voice.session[test.voice_idx[test.y == 1]])
    assert not (dev_sessions & test_sessions), "LEAK: genuine session in both dev and test"
    assert not (set(dev.groups[dev.y == 1]) & set(test.groups[test.y == 1])), "LEAK: group overlap"


def gather(rows: Rows, pools: Pools, face_X=None, voice_X=None, fp_X=None):
    """Feature matrices for rows (optionally from perturbed copies of the pools)."""
    fx = pools.face.X if face_X is None else face_X
    vx = pools.voice.X if voice_X is None else voice_X
    px = pools.fp.X if fp_X is None else fp_X
    return fx[rows.face_idx], vx[rows.voice_idx], px[rows.fp_idx]
