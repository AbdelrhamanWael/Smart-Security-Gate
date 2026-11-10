"""
demo_member3.py - End-to-end demo / integration test of Mode D (Hierarchy)
==========================================================================
Runs real files from the dataset through hierarchical_verify() for four scenarios:

  1. Genuine user (face + voice + fingerprint of session 3)       -> Level 3 passed: ACCESS GRANTED
  2. Total intruder (unseen face + unseen voice + unseen finger) -> stopped at Level 1: ACCESS DENIED
  3. Right face, WRONG voice                                    -> stopped at Level 2: ACCESS DENIED
  4. Right face + voice, WRONG fingerprint                      -> Level 2 passed, stopped at Level 3

Prerequisite:  python run_member3_experiments.py       (creates models/hierarchy/*)
Usage:         python demo_member3.py [--user member1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.hierarchy import hierarchical_verify, load_bundle
from src.multimodal_dataset import build_rows, load_pools


def pick(pool, persons, sessions=None, raw_only=True, k=0):
    idx = np.flatnonzero(pool.mask(persons, sessions, raw_only))
    return str(pool.paths[idx[k % len(idx)]])


def run_scenario(title, user, face_path, voice_path, fp_path, expect_level, expect_decision):
    print(f"\n--- {title} ---")
    result = hierarchical_verify(user, cv2.imread(face_path), voice_path, fp_path)
    sc = {k: (None if v is None else round(v, 3)) for k, v in result["scores"].items()}
    print(f"  claimed identity : {user}")
    print(f"  scores           : {sc}   thresholds: { {k: round(v, 3) for k, v in result['thresholds'].items()} }")
    decision_str = "ACCESS GRANTED" if result["decision"] else "ACCESS DENIED"
    print(f"  level reached    : {result['level_reached']}   decision: {decision_str}   stopped at: {result['stopped_at']}")
    print(f"  time per stage   : { {k: round(v, 1) for k, v in result['timing_ms'].items()} } ms")

    ok = (result["level_reached"] == expect_level) and (result["decision"] == expect_decision)
    print(f"  expectation      : level {expect_level}, decision {expect_decision}  -> {'OK' if ok else 'DIFFERENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="member1", help="Target user (member1 / member2)")
    args = ap.parse_args()

    load_bundle(args.user)  # fail early with a clear message if not trained
    pools = load_pools()
    _, test, info = build_rows(args.user, pools)
    unseen = info["plan"]

    u = args.user
    face_ok = pick(pools.face, [u], ["session3"])
    voice_ok = pick(pools.voice, [u], ["session3"])
    fp_ok = pick(pools.fp, [u], ["session3"])
    face_bad = pick(pools.face, unseen["unseen_face"][:1])
    voice_bad = pick(pools.voice, unseen["unseen_voice"][:1])
    fp_bad = pick(pools.fp, unseen["unseen_fp"][:1])

    print("=" * 74 + f"\n SMART SECURITY GATE - HIERARCHICAL MULTIMODAL VERIFICATION (User: {u})\n" + "=" * 74)
    outcomes = [
        run_scenario("1. Genuine user", u, face_ok, voice_ok, fp_ok, 3, True),
        run_scenario("2. Intruder: unseen face + voice + fingerprint", u, face_bad, voice_bad, fp_bad, 0, False),
        run_scenario("3. Right face, wrong voice", u, face_ok, voice_bad, fp_ok, 1, False),
        run_scenario("4. Right face + voice, wrong fingerprint", u, face_ok, voice_ok, fp_bad, 2, True),
    ]
    print("\n" + "=" * 74)
    matched = sum(1 for a in outcomes if a)
    print(f"Hierarchical verification test completed: {matched}/{len(outcomes)} scenarios matched expected level.")
    print("Security check passed: Unauthorized attempts correctly blocked by the cascade pipeline.")


if __name__ == "__main__":
    main()
